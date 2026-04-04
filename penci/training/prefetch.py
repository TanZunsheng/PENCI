# -*- coding: utf-8 -*-
"""
共享 HDF5 预热与节点级 page-cache 续热工具。
"""

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
prefetch_detail_logger = logging.getLogger("penci.prefetch_detail")
prefetch_detail_logger.addHandler(logging.NullHandler())
prefetch_detail_logger.propagate = False
PREFETCH_PROGRESS_LOG_STEP_PCT = 20.0


def _gib_to_bytes(gib: float) -> int:
    return int(max(0.0, gib) * (1024 ** 3))


def _normalize_progress_log_step_pct(step_pct: float) -> float:
    return min(100.0, max(1.0, float(step_pct)))


def _resolve_prefetch_paths(hdf5_paths: List[str], data_root: str) -> List[str]:
    resolved: List[str] = []
    seen = set()
    for path in hdf5_paths:
        if not path:
            continue
        abs_path = path if os.path.isabs(path) else os.path.join(data_root, path)
        if abs_path in seen:
            continue
        if os.path.isfile(abs_path):
            seen.add(abs_path)
            resolved.append(abs_path)
    return resolved


def get_prefetch_file_plan(dataloader, data_root: str, max_files: int = 0) -> List[str]:
    file_paths: List[str] = []
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "get_prefetch_file_plan"):
        try:
            file_paths = batch_sampler.get_prefetch_file_plan(max_files=max_files)
        except Exception as exc:
            logger.warning("读取 sampler 预热计划失败，将回退到 metadata 顺序: %s", exc)

    if not file_paths:
        dataset = getattr(dataloader, "dataset", None)
        metadata = getattr(dataset, "metadata", None)
        if isinstance(metadata, list):
            for item in metadata:
                h5_path = item.get("hdf5_path")
                if h5_path:
                    file_paths.append(h5_path)

    resolved = _resolve_prefetch_paths(file_paths, data_root)
    return resolved[:max_files] if max_files > 0 else resolved


def get_prefetch_rank_schedule(dataloader) -> List[Dict[str, Any]]:
    schedule: List[Dict[str, Any]] = []
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "get_prefetch_rank_schedule"):
        try:
            schedule = batch_sampler.get_prefetch_rank_schedule()
        except Exception as exc:
            logger.warning("读取 sampler rank 级文件计划失败: %s", exc)
    return schedule


def _summarize_prefetch_plan_files(file_plan: List[Dict[str, Any]]) -> Tuple[int, int]:
    unique_files: Dict[str, int] = {}
    for item in file_plan:
        if not isinstance(item, dict):
            continue
        hdf5_path = str(item.get("hdf5_path", "")).strip()
        if not hdf5_path or hdf5_path in unique_files:
            continue
        unique_files[hdf5_path] = int(item.get("size_bytes", 0))
    return len(unique_files), sum(unique_files.values())


def build_node_union_prefetch_plan(
    rank_schedules: List[List[Dict[str, Any]]],
    data_root: str,
    max_files: int = 0,
) -> List[Dict[str, Any]]:
    merged_windows_by_path: Dict[str, List[Dict[str, Any]]] = {}
    for rank_idx, schedule in enumerate(rank_schedules):
        if not isinstance(schedule, list):
            continue
        for item in schedule:
            if not isinstance(item, dict):
                continue
            hdf5_rel = str(item.get("hdf5_path", "")).strip()
            if not hdf5_rel:
                continue
            hdf5_abs = hdf5_rel if os.path.isabs(hdf5_rel) else os.path.join(data_root, hdf5_rel)
            if not os.path.isfile(hdf5_abs):
                logger.warning("节点级预热计划跳过缺失文件: %s", hdf5_abs)
                continue

            first_use_step = int(item.get("first_batch_idx", 0))
            last_use_step = max(first_use_step, int(item.get("last_batch_idx", first_use_step)))
            merged_windows_by_path.setdefault(hdf5_abs, []).append(
                {
                    "hdf5_path": hdf5_abs,
                    "hdf5_rel": hdf5_rel,
                    "size_bytes": os.path.getsize(hdf5_abs),
                    "first_use_step": first_use_step,
                    "last_use_step": last_use_step,
                    "ranks": {rank_idx},
                }
            )

    plan: List[Dict[str, Any]] = []
    for windows in merged_windows_by_path.values():
        windows.sort(key=lambda item: (item["first_use_step"], item["last_use_step"]))
        merged_windows: List[Dict[str, Any]] = []
        for item in windows:
            if merged_windows and item["first_use_step"] <= merged_windows[-1]["last_use_step"] + 1:
                merged_windows[-1]["last_use_step"] = max(
                    merged_windows[-1]["last_use_step"],
                    item["last_use_step"],
                )
                merged_windows[-1]["ranks"].update(item["ranks"])
            else:
                merged_windows.append(
                    {
                        "hdf5_path": item["hdf5_path"],
                        "hdf5_rel": item["hdf5_rel"],
                        "size_bytes": item["size_bytes"],
                        "first_use_step": item["first_use_step"],
                        "last_use_step": item["last_use_step"],
                        "ranks": set(item["ranks"]),
                    }
                )
        for item in merged_windows:
            item["n_batches"] = item["last_use_step"] - item["first_use_step"] + 1
            item["ranks"] = sorted(item["ranks"])
            plan.append(item)

    plan.sort(key=lambda item: (item["first_use_step"], item["hdf5_path"], item["last_use_step"]))
    return plan[:max_files] if max_files > 0 else plan


class NodePageCachePrefetcher:
    """单节点 page cache 预取器：按窗口预热 + 训练期后台续热。"""

    def __init__(
        self,
        file_plan: List[Dict[str, Any]],
        high_watermark_gb: float,
        low_watermark_gb: float,
        read_chunk_mb: int = 8,
        max_threads: int = 1,
        progress_log_step_pct: float = PREFETCH_PROGRESS_LOG_STEP_PCT,
    ):
        self.file_plan = file_plan
        self.high_watermark_bytes = _gib_to_bytes(high_watermark_gb)
        self.low_watermark_bytes = min(
            self.high_watermark_bytes,
            _gib_to_bytes(max(0.0, low_watermark_gb)),
        )
        self.read_chunk_bytes = max(1, int(read_chunk_mb)) * 1024 * 1024
        self.plan_window_count = len(file_plan)
        self.plan_file_count, self.total_plan_bytes = _summarize_prefetch_plan_files(file_plan)
        self.total_window_bytes = sum(int(item.get("size_bytes", 0)) for item in file_plan)
        self.prefetched_cursor = 0
        self.current_batch_idx = 0
        self.active_prefetched_bytes = 0
        self.active_prefetched_files = 0
        self.total_prefetched_bytes = 0
        self._prefetched = [False] * len(file_plan)
        self._lock = threading.Lock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._refill_requested = False
        self.progress_log_step_pct = _normalize_progress_log_step_pct(progress_log_step_pct)
        self._next_warmup_log_pct = self.progress_log_step_pct

        if max_threads != 1:
            logger.info(
                "节点级后台续热线程数请求为 %d，为避免 NFS 抖动，当前按 1 线程顺序读取执行",
                max_threads,
            )

    def _log_progress_snapshot(self, prefix: str, milestone_pct: float, started_at: float) -> None:
        status = self.get_status()
        elapsed = max(1e-6, time.time() - started_at)
        speed_gib = (status["total_prefetched_bytes"] / (1024 ** 3)) / elapsed
        active_gib = status["active_prefetched_bytes"] / (1024 ** 3)
        high_gib = self.high_watermark_bytes / (1024 ** 3)
        pct = 100.0 if high_gib <= 0 else min(100.0, (active_gib / high_gib) * 100.0)
        remain_gib = max(0.0, high_gib - active_gib)
        eta_min = 0.0 if speed_gib <= 1e-6 else remain_gib / speed_gib / 60.0
        logger.info(
            "[%s] 已达到 %.0f%% | 当前 %.2f/%.2f GiB (%.1f%%) | %.2f GiB/s | ETA %.1f min",
            prefix,
            milestone_pct,
            active_gib,
            high_gib,
            pct,
            speed_gib,
            eta_min,
        )

    def _maybe_log_warmup_progress(self, started_at: float, force: bool = False) -> None:
        if self.high_watermark_bytes <= 0:
            return
        status = self.get_status()
        active_gib = status["active_prefetched_bytes"] / (1024 ** 3)
        high_gib = self.high_watermark_bytes / (1024 ** 3)
        pct = 100.0 if high_gib <= 0 else min(100.0, (active_gib / high_gib) * 100.0)
        while pct + 1e-6 >= self._next_warmup_log_pct:
            milestone_pct = self._next_warmup_log_pct
            self._log_progress_snapshot("节点联合预热进度", milestone_pct, started_at)
            self._next_warmup_log_pct += self.progress_log_step_pct
        if force and pct > 0.0 and self._next_warmup_log_pct <= 100.0:
            self._log_progress_snapshot("节点联合预热进度", pct, started_at)
            self._next_warmup_log_pct = 100.0 + self.progress_log_step_pct

    def _recompute_active_prefetched_bytes_unlocked(self) -> int:
        active_paths: Dict[str, int] = {}
        for prefetched, item in zip(self._prefetched, self.file_plan):
            if prefetched and int(item.get("last_use_step", -1)) >= self.current_batch_idx:
                hdf5_path = str(item.get("hdf5_path", "")).strip()
                if hdf5_path and hdf5_path not in active_paths:
                    active_paths[hdf5_path] = int(item.get("size_bytes", 0))
        active_bytes = sum(active_paths.values())
        self.active_prefetched_bytes = active_bytes
        self.active_prefetched_files = len(active_paths)
        return active_bytes

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "plan_files": self.plan_file_count,
                "plan_windows": self.plan_window_count,
                "prefetched_cursor": self.prefetched_cursor,
                "current_batch_idx": self.current_batch_idx,
                "active_prefetched_bytes": self.active_prefetched_bytes,
                "active_prefetched_files": self.active_prefetched_files,
                "total_prefetched_bytes": self.total_prefetched_bytes,
                "total_plan_bytes": self.total_plan_bytes,
                "total_window_bytes": self.total_window_bytes,
            }

    def _prefetch_next_file(self, reason: str) -> bool:
        with self._lock:
            if self.prefetched_cursor >= len(self.file_plan):
                self._refill_requested = False
                return False
            window_idx = self.prefetched_cursor
            item = self.file_plan[window_idx]
            self.prefetched_cursor += 1

        hdf5_path = item["hdf5_path"]
        file_size = int(item.get("size_bytes", 0))
        try:
            with open(hdf5_path, "rb", buffering=0) as handle:
                while True:
                    buf = handle.read(self.read_chunk_bytes)
                    if not buf:
                        break
        except OSError as exc:
            logger.warning("节点级预热读取失败，跳过文件 %s: %s", hdf5_path, exc)
            with self._lock:
                self._recompute_active_prefetched_bytes_unlocked()
                self._refill_requested = (
                    self.active_prefetched_bytes < self.high_watermark_bytes
                    and self.prefetched_cursor < len(self.file_plan)
                )
            return False

        with self._lock:
            self._prefetched[window_idx] = True
            self.total_prefetched_bytes += file_size
            active_bytes = self._recompute_active_prefetched_bytes_unlocked()
            active_files = self.active_prefetched_files
            current_batch_idx = self.current_batch_idx
            prefetched_cursor = self.prefetched_cursor
            self._refill_requested = (
                active_bytes < self.high_watermark_bytes
                and prefetched_cursor < len(self.file_plan)
            )

        prefix = "启动预热" if reason == "warmup" else "后台续热"
        prefetch_detail_logger.info(
            "[%s] batch_idx=%d | 窗口 %d/%d | active %.2f/%.2f GiB | active_files=%d | 已预热窗口 %d/%d | %s",
            prefix,
            current_batch_idx,
            window_idx + 1,
            self.plan_window_count,
            active_bytes / (1024 ** 3),
            self.high_watermark_bytes / (1024 ** 3),
            active_files,
            prefetched_cursor,
            self.plan_window_count,
            item.get("hdf5_rel", hdf5_path),
        )
        return True

    def warmup_to_high_watermark(self) -> float:
        if self.high_watermark_bytes <= 0:
            logger.info("节点级联合预热目标为 0 GiB，跳过启动前预热")
            return 0.0
        if not self.file_plan:
            logger.warning("节点级联合预热计划为空，跳过启动前预热")
            return 0.0

        logger.info(
            "节点级联合预热计划: 窗口 %d 段 | 唯一文件 %d 个 | 唯一总大小 %.2f GiB | 高水位 %.2f GiB | 低水位 %.2f GiB",
            self.plan_window_count,
            self.plan_file_count,
            self.total_plan_bytes / (1024 ** 3),
            self.high_watermark_bytes / (1024 ** 3),
            self.low_watermark_bytes / (1024 ** 3),
        )

        started_at = time.time()
        while True:
            with self._lock:
                active_bytes = self._recompute_active_prefetched_bytes_unlocked()
                prefetched_cursor = self.prefetched_cursor
            if active_bytes >= self.high_watermark_bytes or prefetched_cursor >= len(self.file_plan):
                break
            progressed = self._prefetch_next_file(reason="warmup")
            if not progressed and prefetched_cursor >= len(self.file_plan):
                break
            self._maybe_log_warmup_progress(started_at)

        self._maybe_log_warmup_progress(started_at, force=True)

        warmed_gib = self.get_status()["active_prefetched_bytes"] / (1024 ** 3)
        logger.info("节点级联合预热完成: 当前逻辑工作集 %.2f GiB", warmed_gib)
        return warmed_gib

    def _run_refill_loop(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.wait(timeout=1.0)
            self._wake_event.clear()
            if self._stop_event.is_set():
                break

            while not self._stop_event.is_set():
                with self._lock:
                    active_bytes = self._recompute_active_prefetched_bytes_unlocked()
                    need_refill = (
                        active_bytes < self.high_watermark_bytes
                        and self.prefetched_cursor < len(self.file_plan)
                    )
                    if not need_refill:
                        self._refill_requested = False
                        break
                progressed = self._prefetch_next_file(reason="refill")
                if not progressed:
                    with self._lock:
                        self._refill_requested = False
                    break

    def start_async_refill(self) -> None:
        if self._thread is not None or not self.file_plan:
            return
        self._thread = threading.Thread(
            target=self._run_refill_loop,
            name="node-page-cache-prefetcher",
            daemon=True,
        )
        self._thread.start()
        prefetch_detail_logger.info(
            "后台续热器已启动：单线程顺序读取，按低/高水位维持 page cache 工作集"
        )

    def update_progress(self, current_batch_idx: int) -> None:
        with self._lock:
            if current_batch_idx < self.current_batch_idx:
                return
            self.current_batch_idx = current_batch_idx
            active_bytes = self._recompute_active_prefetched_bytes_unlocked()
            prefetched_cursor = self.prefetched_cursor
            need_refill = (
                active_bytes < self.low_watermark_bytes
                and prefetched_cursor < len(self.file_plan)
            )
            should_log = need_refill and not self._refill_requested
            if need_refill:
                self._refill_requested = True
            elif active_bytes >= self.high_watermark_bytes or prefetched_cursor >= len(self.file_plan):
                self._refill_requested = False

        if should_log:
            prefetch_detail_logger.info(
                "[后台续热触发] batch_idx=%d | active %.2f/%.2f GiB | 已预热窗口 %d/%d | 将补到 %.2f GiB",
                current_batch_idx,
                active_bytes / (1024 ** 3),
                self.low_watermark_bytes / (1024 ** 3),
                prefetched_cursor,
                self.plan_window_count,
                self.high_watermark_bytes / (1024 ** 3),
            )
        if need_refill:
            self._wake_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        status = self.get_status()
        prefetch_detail_logger.info(
            "后台续热器已停止: batch_idx=%d | active %.2f GiB | 已预热窗口 %d/%d",
            status["current_batch_idx"],
            status["active_prefetched_bytes"] / (1024 ** 3),
            status["prefetched_cursor"],
            status["plan_windows"],
        )


def warmup_hdf5_page_cache(
    file_paths: List[str],
    warmup_gb: float,
    read_chunk_mb: int = 8,
    max_threads: int = 1,
    progress_log_step_pct: float = PREFETCH_PROGRESS_LOG_STEP_PCT,
) -> float:
    target_bytes = _gib_to_bytes(warmup_gb)
    if target_bytes <= 0:
        return 0.0
    if not file_paths:
        logger.warning("预热计划为空，跳过 HDF5 预热")
        return 0.0

    if max_threads != 1:
        logger.info(
            "HDF5 预热线程数请求为 %d，为避免 NFS 抖动，当前按 1 线程顺序读取执行",
            max_threads,
        )

    chunk_bytes = max(1, int(read_chunk_mb)) * 1024 * 1024
    warmed_bytes = 0
    started_at = time.time()
    progress_step_pct = _normalize_progress_log_step_pct(progress_log_step_pct)
    next_progress_log_pct = progress_step_pct

    logger.info(
        "开始 HDF5 预热: 目标 %.1f GiB, 候选文件 %d 个, 读取块 %d MiB",
        warmup_gb,
        len(file_paths),
        read_chunk_mb,
    )

    for index, path in enumerate(file_paths, start=1):
        if warmed_bytes >= target_bytes:
            break
        file_size = os.path.getsize(path)
        try:
            with open(path, "rb", buffering=0) as handle:
                while True:
                    buf = handle.read(chunk_bytes)
                    if not buf:
                        break
        except OSError as exc:
            logger.warning("预热读取失败，跳过文件 %s: %s", path, exc)
            continue

        warmed_bytes += file_size
        elapsed = max(1e-6, time.time() - started_at)
        speed_gib = (warmed_bytes / (1024 ** 3)) / elapsed
        warmed_gib = warmed_bytes / (1024 ** 3)
        target_gib = target_bytes / (1024 ** 3)
        pct = min(100.0, (warmed_bytes / target_bytes) * 100.0)
        eta_sec = max(0.0, (target_gib - warmed_gib) / max(1e-6, speed_gib))
        while pct + 1e-6 >= next_progress_log_pct:
            logger.info(
                "[预热进度] 已达到 %.0f%% | 文件 %d/%d | %.2f/%.2f GiB (%.1f%%) | %.2f GiB/s | ETA %.1f min",
                next_progress_log_pct,
                index,
                len(file_paths),
                warmed_gib,
                target_gib,
                pct,
                speed_gib,
                eta_sec / 60.0,
            )
            next_progress_log_pct += progress_step_pct

    warmed_gib = warmed_bytes / (1024 ** 3)
    if warmed_bytes > 0 and next_progress_log_pct <= 100.0:
        elapsed = max(1e-6, time.time() - started_at)
        speed_gib = (warmed_bytes / (1024 ** 3)) / elapsed
        eta_sec = 0.0
        logger.info(
            "[预热进度] 已达到 %.0f%% | 文件 %d/%d | %.2f/%.2f GiB (%.1f%%) | %.2f GiB/s | ETA %.1f min",
            min(100.0, warmed_bytes / target_bytes * 100.0),
            min(len(file_paths), index),
            len(file_paths),
            warmed_gib,
            target_bytes / (1024 ** 3),
            min(100.0, warmed_bytes / target_bytes * 100.0),
            speed_gib,
            eta_sec / 60.0,
        )
    logger.info("HDF5 预热完成: 已加载 %.2f GiB 到 OS page cache", warmed_gib)
    return warmed_gib
