#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PENCI 训练脚本

用法:
    # 单卡训练
    python scripts/train.py --config configs/default.yaml
    # 多卡 DDP 训练 (4 GPU)
    torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --output_dir outputs/experiment_1
"""

import os
import sys
import math
import time
import json
import threading
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from penci.models import build_penci_from_config
from penci.data import get_train_val_loaders
from penci.physics import SourceSpace, ElectrodeConfigRegistry
from penci.physics.leadfield_manager import LeadfieldManager
from penci.utils.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
prefetch_detail_logger = logging.getLogger("penci.prefetch_detail")
prefetch_detail_logger.addHandler(logging.NullHandler())
prefetch_detail_logger.propagate = False


def configure_prefetch_detail_logger(log_file: Path) -> None:
    """配置节点级预热/续热细节日志，避免刷屏主训练日志。"""
    prefetch_detail_logger.setLevel(logging.INFO)
    prefetch_detail_logger.propagate = False
    prefetch_detail_logger.handlers = []
    detail_handler = logging.FileHandler(log_file, encoding="utf-8")
    detail_handler.setLevel(logging.INFO)
    detail_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    prefetch_detail_logger.addHandler(detail_handler)


def configure_ddp_mode(ddp_mode: str) -> None:
    """
    配置 DDP/NCCL 运行模式

    - prod: 清理常见诊断变量，回到安静/性能优先配置
    - debug: 启用详细分布式诊断日志
    """
    debug_vars = [
        "NCCL_DEBUG",
        "TORCH_DISTRIBUTED_DEBUG",
        "TORCH_FR_BUFFER_SIZE",
        "TORCH_NCCL_TRACE_BUFFER_SIZE",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
    ]

    if ddp_mode == "prod":
        for key in debug_vars:
            os.environ.pop(key, None)
        return

    # debug 模式默认值；用户若已手动设置则保留其值
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    os.environ.setdefault("TORCH_FR_BUFFER_SIZE", "1048576")


def setup_distributed() -> Tuple[int, int, int]:
    """
    检测 torchrun 环境，初始化分布式训练

    返回:
        (rank, local_rank, world_size) — 非分布式时返回 (0, 0, 1)
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # 必须先 set_device，再 init_process_group
        # 否则 NCCL 不知道绑定哪张 GPU，会导致多卡训练卡死
        # 注意：不要传 device_id 参数！该参数会触发 eager NCCL ALLREDUCE，
        # 在本机 PCIe 拓扑下会永久卡死（10 分钟超时后崩溃）
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=60),
        )
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    解包 DDP 包装，返回底层模型

    DDP 包装后 model 是 DistributedDataParallel 对象，不会自动代理
    compute_loss / _prepare_target 等自定义方法，需通过 .module 访问。
    """
    return model.module if hasattr(model, "module") else model


def sync_ranks(world_size: int, local_rank: int) -> None:
    """
    分布式同步屏障（显式指定当前 rank 绑定 GPU）

    传入 device_ids 避免 NCCL 在 barrier 阶段出现 "device unknown" 警告，
    降低多卡初始化阶段卡死风险。
    """
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    min_lr_ratio=0.0,
):
    """带线性预热的余弦退火调度器"""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def reduce_metric(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """在所有 rank 上 all_reduce 一个标量张量并取平均"""
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor


def reduce_metric_sum(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """在所有 rank 上 all_reduce 一个标量张量并求和"""
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _is_floating_tensor(tensor: Optional[torch.Tensor]) -> bool:
    return bool(torch.is_tensor(tensor) and (torch.is_floating_point(tensor) or torch.is_complex(tensor)))


def _is_finite_tensor(tensor: Optional[torch.Tensor]) -> bool:
    if tensor is None:
        return True
    if not _is_floating_tensor(tensor):
        return True
    return bool(torch.isfinite(tensor).all().item())


def _tensor_stats(tensor: Optional[torch.Tensor]) -> Dict[str, Any]:
    """仅在异常路径调用：提取张量统计信息用于 NaN/Inf 诊断。"""
    if tensor is None:
        return {"exists": False}
    if not torch.is_tensor(tensor):
        return {"exists": False, "type": str(type(tensor))}

    t = tensor.detach()
    t_cpu = t.to("cpu")
    stats: Dict[str, Any] = {
        "exists": True,
        "shape": list(t_cpu.shape),
        "dtype": str(t_cpu.dtype),
        "numel": int(t_cpu.numel()),
    }
    if t_cpu.numel() == 0:
        return stats

    if _is_floating_tensor(t_cpu):
        finite_mask = torch.isfinite(t_cpu)
        finite_count = int(finite_mask.sum().item())
        stats["nan_count"] = int(torch.isnan(t_cpu).sum().item())
        stats["inf_count"] = int(torch.isinf(t_cpu).sum().item())
        stats["finite_count"] = finite_count
        if finite_count > 0:
            vals = t_cpu[finite_mask].float()
            stats["min"] = float(vals.min().item())
            stats["max"] = float(vals.max().item())
            stats["mean"] = float(vals.mean().item())
            stats["std"] = float(vals.std(unbiased=False).item())
        return stats

    # 非浮点数据：主要用于结构确认
    vals = t_cpu.float()
    stats["min"] = float(vals.min().item())
    stats["max"] = float(vals.max().item())
    stats["mean"] = float(vals.mean().item())
    return stats


def _any_rank_true(local_flag: bool, device: torch.device, world_size: int) -> bool:
    if world_size <= 1:
        return local_flag
    flag_t = torch.tensor(1 if local_flag else 0, device=device, dtype=torch.int32)
    dist.all_reduce(flag_t, op=dist.ReduceOp.SUM)
    return bool(flag_t.item() > 0)


def _first_non_finite_grad_name(model: nn.Module) -> Optional[str]:
    base_model = unwrap_model(model)
    for name, param in base_model.named_parameters():
        grad = param.grad
        if grad is None or not _is_floating_tensor(grad):
            continue
        if not torch.isfinite(grad).all():
            return name
    return None


class NaNGuard:
    """训练时 NaN/Inf 诊断器：记录坏 batch 来源与关键张量统计。"""

    def __init__(self, cfg: Dict[str, Any], output_dir: Path, rank: int):
        self.enabled = bool(cfg.get("enabled", True))
        self.fail_fast = bool(cfg.get("fail_fast", True))
        self.skip_bad_batch = bool(cfg.get("skip_bad_batch", False))
        self.max_records = max(1, int(cfg.get("max_records", 50)))
        self.max_metadata_items = max(1, int(cfg.get("max_metadata_items", 128)))
        self.dump_tensors = bool(cfg.get("dump_tensors", False))
        self.dump_max_samples = max(1, int(cfg.get("dump_max_samples", 4)))
        record_stem = str(cfg.get("record_file", "bad_batches")).strip() or "bad_batches"
        dump_dir_name = str(cfg.get("dump_dir", "bad_batch_tensors")).strip() or "bad_batch_tensors"
        diagnostics_dir = output_dir / "diagnostics"
        self.record_path = diagnostics_dir / f"{record_stem}.rank{rank}.jsonl"
        self.dump_dir = diagnostics_dir / dump_dir_name
        self.rank = rank
        self.recorded_count = 0

    def _trim_metadata(self, metadata: Any) -> List[Dict[str, Any]]:
        if not isinstance(metadata, list):
            return []
        trimmed: List[Dict[str, Any]] = []
        keys = (
            "dataset",
            "path",
            "channels",
            "fingerprint",
            "hdf5_path",
            "hdf5_idx",
            "sample_index",
        )
        for item in metadata[: self.max_metadata_items]:
            if not isinstance(item, dict):
                continue
            row = {k: item.get(k) for k in keys if k in item}
            trimmed.append(row)
        return trimmed

    def _dump_tensors(
        self,
        *,
        epoch: int,
        batch_idx: int,
        global_step: int,
        batch: Dict[str, Any],
        leadfield: Optional[torch.Tensor],
        output: Optional[Dict[str, torch.Tensor]],
    ) -> Optional[str]:
        if not self.dump_tensors:
            return None

        self.dump_dir.mkdir(parents=True, exist_ok=True)
        dump_path = self.dump_dir / (
            f"bad_batch_rank{self.rank}_step{global_step}_epoch{epoch}_idx{batch_idx}.pt"
        )

        payload = {
            "x": batch["x"][: self.dump_max_samples].detach().cpu(),
            "pos": batch["pos"][: self.dump_max_samples].detach().cpu(),
            "sensor_type": batch["sensor_type"][: self.dump_max_samples].detach().cpu(),
            "metadata": self._trim_metadata(batch.get("metadata")),
            "fingerprint": batch.get("fingerprint", "unknown"),
            "leadfield": None if leadfield is None else leadfield.detach().cpu(),
        }
        if isinstance(output, dict):
            if torch.is_tensor(output.get("reconstruction")):
                payload["reconstruction"] = (
                    output["reconstruction"][: self.dump_max_samples].detach().cpu()
                )
            if torch.is_tensor(output.get("source_activity")):
                payload["source_activity"] = (
                    output["source_activity"][: self.dump_max_samples].detach().cpu()
                )

        torch.save(payload, dump_path)
        return str(dump_path)

    def record(
        self,
        *,
        reason: str,
        stage: str,
        epoch: int,
        batch_idx: int,
        global_step: int,
        world_size: int,
        batch: Dict[str, Any],
        leadfield: Optional[torch.Tensor],
        output: Optional[Dict[str, torch.Tensor]],
        losses: Optional[Dict[str, torch.Tensor]],
        lr: float,
        scaler: Optional[torch.amp.GradScaler],
        grad_norm: Optional[float],
        bad_grad_param: Optional[str],
    ) -> None:
        if not self.enabled:
            return
        if self.recorded_count >= self.max_records:
            return

        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        sample_meta = self._trim_metadata(batch.get("metadata"))

        record = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "stage": stage,
            "rank": self.rank,
            "world_size": int(world_size),
            "epoch": int(epoch),
            "batch_idx": int(batch_idx),
            "global_step": int(global_step),
            "lr": float(lr),
            "amp_scale": float(scaler.get_scale()) if scaler is not None else None,
            "grad_norm": grad_norm,
            "bad_grad_param": bad_grad_param,
            "fingerprint": batch.get("fingerprint", "unknown"),
            "batch_size": int(batch["x"].shape[0]) if torch.is_tensor(batch.get("x")) else None,
            "samples": sample_meta,
            "stats": {
                "x": _tensor_stats(batch.get("x")),
                "pos": _tensor_stats(batch.get("pos")),
                "sensor_type": _tensor_stats(batch.get("sensor_type")),
                "leadfield": _tensor_stats(leadfield),
                "loss": _tensor_stats(None if not losses else losses.get("loss")),
                "recon_loss": _tensor_stats(None if not losses else losses.get("recon_loss")),
                "dynamics_loss": _tensor_stats(None if not losses else losses.get("dynamics_loss")),
                "reconstruction": _tensor_stats(
                    None if not isinstance(output, dict) else output.get("reconstruction")
                ),
                "source_activity": _tensor_stats(
                    None if not isinstance(output, dict) else output.get("source_activity")
                ),
            },
        }
        dump_path = self._dump_tensors(
            epoch=epoch,
            batch_idx=batch_idx,
            global_step=global_step,
            batch=batch,
            leadfield=leadfield,
            output=output,
        )
        if dump_path is not None:
            record["tensor_dump"] = dump_path

        with self.record_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.recorded_count += 1


def _gib_to_bytes(gib: float) -> int:
    return int(max(0.0, gib) * (1024 ** 3))


def _resolve_prefetch_paths(
    hdf5_paths: List[str],
    data_root: str,
) -> List[str]:
    """把相对 HDF5 路径解析成绝对路径，并去重保序。"""
    resolved: List[str] = []
    seen = set()
    for p in hdf5_paths:
        if not p:
            continue
        abs_path = p if os.path.isabs(p) else os.path.join(data_root, p)
        if abs_path in seen:
            continue
        if os.path.isfile(abs_path):
            seen.add(abs_path)
            resolved.append(abs_path)
    return resolved


def get_prefetch_file_plan(
    dataloader,
    data_root: str,
    max_files: int = 0,
) -> List[str]:
    """
    获取当前 epoch 的 HDF5 预热文件列表（绝对路径）。

    优先使用 sampler 暴露的消费顺序计划；若不可用则回退为 metadata 顺序去重。
    """
    file_paths: List[str] = []
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "get_prefetch_file_plan"):
        try:
            file_paths = batch_sampler.get_prefetch_file_plan(max_files=max_files)
        except Exception as e:
            logger.warning(f"读取 sampler 预热计划失败，将回退到 metadata 顺序: {e}")

    if not file_paths:
        dataset = getattr(dataloader, "dataset", None)
        metadata = getattr(dataset, "metadata", None)
        if isinstance(metadata, list):
            for m in metadata:
                h5p = m.get("hdf5_path")
                if h5p:
                    file_paths.append(h5p)

    resolved = _resolve_prefetch_paths(file_paths, data_root)
    if max_files > 0:
        return resolved[:max_files]
    return resolved


def get_prefetch_rank_schedule(dataloader) -> List[Dict[str, Any]]:
    """获取当前 rank 的文件消费窗口计划。"""
    schedule: List[Dict[str, Any]] = []
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "get_prefetch_rank_schedule"):
        try:
            schedule = batch_sampler.get_prefetch_rank_schedule()
        except Exception as e:
            logger.warning(f"读取 sampler rank 级文件计划失败: {e}")
    return schedule


def _summarize_prefetch_plan_files(file_plan: List[Dict[str, Any]]) -> Tuple[int, int]:
    """统计预热计划中的唯一文件数和唯一文件总字节数。"""
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
    """合并所有 rank 的文件消费窗口，构建节点级窗口预热计划。"""
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
                logger.warning(f"节点级预热计划跳过缺失文件: {hdf5_abs}")
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
            if (
                merged_windows
                and item["first_use_step"] <= merged_windows[-1]["last_use_step"] + 1
            ):
                merged_windows[-1]["last_use_step"] = max(
                    merged_windows[-1]["last_use_step"], item["last_use_step"]
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
    if max_files > 0:
        return plan[:max_files]
    return plan


class NodePageCachePrefetcher:
    """单节点 page cache 预取器：按窗口预热 + 训练期后台续热。"""

    def __init__(
        self,
        file_plan: List[Dict[str, Any]],
        high_watermark_gb: float,
        low_watermark_gb: float,
        read_chunk_mb: int = 8,
        max_threads: int = 1,
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

        if max_threads != 1:
            logger.info(
                f"节点级后台续热线程数请求为 {max_threads}，为避免 NFS 抖动，当前按 1 线程顺序读取执行"
            )

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
            with open(hdf5_path, "rb", buffering=0) as f:
                while True:
                    buf = f.read(self.read_chunk_bytes)
                    if not buf:
                        break
        except OSError as e:
            logger.warning(f"节点级预热读取失败，跳过文件 {hdf5_path}: {e}")
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
            f"[{prefix}] batch_idx={current_batch_idx} | "
            f"窗口 {window_idx + 1}/{self.plan_window_count} | "
            f"active {active_bytes / (1024 ** 3):.2f}/{self.high_watermark_bytes / (1024 ** 3):.2f} GiB | "
            f"active_files={active_files} | "
            f"已预热窗口 {prefetched_cursor}/{self.plan_window_count} | {item.get('hdf5_rel', hdf5_path)}"
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
            f"节点级联合预热计划: 窗口 {self.plan_window_count} 段 | "
            f"唯一文件 {self.plan_file_count} 个 | "
            f"唯一总大小 {self.total_plan_bytes / (1024 ** 3):.2f} GiB | "
            f"高水位 {self.high_watermark_bytes / (1024 ** 3):.2f} GiB | "
            f"低水位 {self.low_watermark_bytes / (1024 ** 3):.2f} GiB"
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

            status = self.get_status()
            elapsed = max(1e-6, time.time() - started_at)
            speed_gib = (status["total_prefetched_bytes"] / (1024 ** 3)) / elapsed
            active_gib = status["active_prefetched_bytes"] / (1024 ** 3)
            high_gib = self.high_watermark_bytes / (1024 ** 3)
            pct = 100.0 if high_gib <= 0 else min(100.0, (active_gib / high_gib) * 100.0)
            remain_gib = max(0.0, high_gib - active_gib)
            eta_min = 0.0 if speed_gib <= 1e-6 else remain_gib / speed_gib / 60.0
            logger.info(
                f"[节点联合预热进度] {active_gib:.2f}/{high_gib:.2f} GiB ({pct:.1f}%) | "
                f"{speed_gib:.2f} GiB/s | ETA {eta_min:.1f} min"
            )

        warmed_gib = self.get_status()["active_prefetched_bytes"] / (1024 ** 3)
        logger.info(f"节点级联合预热完成: 当前逻辑工作集 {warmed_gib:.2f} GiB")
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
                f"[后台续热触发] batch_idx={current_batch_idx} | "
                f"active {active_bytes / (1024 ** 3):.2f}/{self.low_watermark_bytes / (1024 ** 3):.2f} GiB | "
                f"已预热窗口 {prefetched_cursor}/{self.plan_window_count} | "
                f"将补到 {self.high_watermark_bytes / (1024 ** 3):.2f} GiB"
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
            f"后台续热器已停止: batch_idx={status['current_batch_idx']} | "
            f"active {status['active_prefetched_bytes'] / (1024 ** 3):.2f} GiB | "
            f"已预热窗口 {status['prefetched_cursor']}/{status['plan_windows']}"
        )


def warmup_hdf5_page_cache(
    file_paths: List[str],
    warmup_gb: float,
    read_chunk_mb: int = 8,
    max_threads: int = 1,
) -> float:
    """
    训练前顺序预热 HDF5 到 OS page cache。

    说明:
        - 默认单线程顺序读取，避免并发随机 I/O 打爆 NFS
        - 返回值是实际预热的数据量（GiB）
    """
    target_bytes = _gib_to_bytes(warmup_gb)
    if target_bytes <= 0:
        return 0.0
    if not file_paths:
        logger.warning("预热计划为空，跳过 HDF5 预热")
        return 0.0

    # 目前默认保守策略：强制单线程顺序读，优先稳态 I/O。
    if max_threads != 1:
        logger.info(
            f"HDF5 预热线程数请求为 {max_threads}，为避免 NFS 抖动，当前按 1 线程顺序读取执行"
        )

    chunk_bytes = max(1, int(read_chunk_mb)) * 1024 * 1024
    warmed_bytes = 0
    started_at = time.time()

    logger.info(
        f"开始 HDF5 预热: 目标 {warmup_gb:.1f} GiB, "
        f"候选文件 {len(file_paths)} 个, 读取块 {read_chunk_mb} MiB"
    )

    for i, path in enumerate(file_paths, start=1):
        if warmed_bytes >= target_bytes:
            break

        file_size = os.path.getsize(path)
        try:
            with open(path, "rb", buffering=0) as f:
                while True:
                    buf = f.read(chunk_bytes)
                    if not buf:
                        break
        except OSError as e:
            logger.warning(f"预热读取失败，跳过文件 {path}: {e}")
            continue

        warmed_bytes += file_size
        elapsed = max(1e-6, time.time() - started_at)
        speed_gib = (warmed_bytes / (1024 ** 3)) / elapsed
        warmed_gib = warmed_bytes / (1024 ** 3)
        target_gib = target_bytes / (1024 ** 3)
        pct = min(100.0, (warmed_bytes / target_bytes) * 100.0)
        eta_sec = max(0.0, (target_gib - warmed_gib) / max(1e-6, speed_gib))
        logger.info(
            f"[预热进度] 文件 {i}/{len(file_paths)} | "
            f"{warmed_gib:.2f}/{target_gib:.2f} GiB ({pct:.1f}%) | "
            f"{speed_gib:.2f} GiB/s | ETA {eta_sec/60:.1f} min"
        )

    warmed_gib = warmed_bytes / (1024 ** 3)
    logger.info(f"HDF5 预热完成: 已加载 {warmed_gib:.2f} GiB 到 OS page cache")
    return warmed_gib


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler,
    step: int,
    epoch: int,
    loss: float,
    save_path,
):
    """保存检查点"""
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(
        {
            'step': step,
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'loss': loss,
        },
        str(save_path),
    )
    logger.info(f"检查点已保存: {save_path}")


def resolve_leadfield_for_batch(
    fingerprint: str,
    leadfield_manager: LeadfieldManager,
    electrode_registry: ElectrodeConfigRegistry,
    device: torch.device,
) -> torch.Tensor:
    """
    根据电极指纹解析对应的导联场矩阵

    BucketBatchSampler 按 (通道数, 电极指纹) 分桶，保证同一 batch 内
    所有样本的电极配置完全一致，因此只需计算一次导联场。

    参数:
        fingerprint: 当前 batch 的电极位置指纹（pos_fingerprint）
        leadfield_manager: 导联场管理器
        electrode_registry: 电极配置注册表
        device: 计算设备

    返回:
        (n_channels, 72) 导联场张量
    """
    channel_names, channel_positions = electrode_registry.get_config_by_fingerprint(
        fingerprint
    )
    return leadfield_manager.get_leadfield(channel_names, channel_positions, device)


def setup_physics(config: dict) -> Tuple[
    Optional[SourceSpace],
    Optional[LeadfieldManager],
    Optional[ElectrodeConfigRegistry],
    bool,
]:
    """
    初始化物理约束组件（SourceSpace + LeadfieldManager + ElectrodeConfigRegistry）

    从配置中读取路径，创建源空间、导联场管理器和电极配置注册表。
    自动为配置中指定的数据集注册电极配置。

    参数:
        config: 完整配置字典

    返回:
        (source_space, leadfield_manager, electrode_registry, loaded_from_archive) 元组
        如果配置中未启用物理约束，所有返回值为 None / False
    """
    physics_cfg = config.get("model", {}).get("physics", {})
    use_fixed_leadfield = physics_cfg.get("use_fixed_leadfield", True)
    leadfield_path = physics_cfg.get("leadfield_path", None)

    if not use_fixed_leadfield or leadfield_path is not None:
        logger.info("物理约束模式: 静态导联场或注意力模式，跳过动态导联场初始化")
        return None, None, None, False

    global_physics = config.get("physics", {})
    subjects_dir = global_physics.get("subjects_dir")
    cache_dir = global_physics.get("leadfield_cache_dir")
    processed_data_dir = global_physics.get("processed_data_dir")
    registry_path = global_physics.get("fingerprint_registry_path")

    if not subjects_dir or not cache_dir:
        raise RuntimeError(
            "动态导联场模式需要 physics.subjects_dir 和 physics.leadfield_cache_dir 配置"
        )

    logger.info("初始化物理约束组件...")

    source_space = SourceSpace(subjects_dir=subjects_dir)
    logger.info(f"  源空间: {source_space.get_source_info()['n_total']} 个源")

    leadfield_manager = LeadfieldManager(
        source_space=source_space,
        subjects_dir=subjects_dir,
        cache_dir=cache_dir,
    )

    if registry_path is not None:
        logger.info(f"从离线存档加载电极配置注册表: {registry_path}")
        electrode_registry = ElectrodeConfigRegistry.load_from_archive(registry_path)
        logger.info(
            f"  已加载 {len(electrode_registry.get_all_fingerprints())} 个唯一电极指纹"
        )
        return source_space, leadfield_manager, electrode_registry, True
    else:
        if not processed_data_dir:
            raise RuntimeError(
                "运行时扫描模式需要 physics.processed_data_dir 配置，"
                "或设置 physics.fingerprint_registry_path 使用离线存档"
            )
        logger.info("未配置 fingerprint_registry_path，使用运行时扫描模式")
        electrode_registry = ElectrodeConfigRegistry(processed_data_dir)

        datasets = config.get("data", {}).get("datasets", [])
        for ds_name in datasets:
            try:
                electrode_registry.register_dataset(ds_name)
            except FileNotFoundError as e:
                logger.warning(f"注册数据集 '{ds_name}' 电极配置失败: {e}")

        logger.info(
            f"  已注册电极配置: {electrode_registry.registered_configs}"
        )
        return source_space, leadfield_manager, electrode_registry, False


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    writer: SummaryWriter = None,
    global_step: int = 0,
    max_steps: int = 100000,
    leadfield_manager: Optional[LeadfieldManager] = None,
    electrode_registry: Optional[ElectrodeConfigRegistry] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    scheduler=None,
    rank: int = 0,
    world_size: int = 1,
    nan_guard: Optional[NaNGuard] = None,
    node_prefetcher: Optional[NodePageCachePrefetcher] = None,
):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_dynamics_loss = 0.0
    num_batches = 0
    skipped_batches = 0

    loss_weights = config.get("training", {}).get("loss", {})
    log_interval = config.get("training", {}).get("log_interval", 100)
    gradient_clip = config.get("training", {}).get("gradient_clip", 1.0)
    guard_enabled = bool(nan_guard is not None and nan_guard.enabled)
    diagnostics_hint = "outputs/.../diagnostics/*.jsonl"
    if nan_guard is not None:
        base = nan_guard.record_path.name.split(".rank")[0]
        diagnostics_hint = str(nan_guard.record_path.parent / f"{base}.rank*.jsonl")

    for batch_idx, batch in enumerate(dataloader):
        x = batch["x"].to(device)
        pos = batch["pos"].to(device)
        sensor_type = batch["sensor_type"].to(device)
        current_batch = {
            "x": x,
            "pos": pos,
            "sensor_type": sensor_type,
            "metadata": batch.get("metadata", []),
            "fingerprint": batch.get("fingerprint", "unknown"),
        }

        leadfield = None
        if leadfield_manager is not None and electrode_registry is not None:
            fingerprint = batch.get("fingerprint", "unknown")
            leadfield = resolve_leadfield_for_batch(
                fingerprint,
                leadfield_manager, electrode_registry, device,
            )

        optimizer.zero_grad(set_to_none=True)
        current_lr = float(optimizer.param_groups[0]["lr"])

        if guard_enabled:
            input_issues: List[str] = []
            if not _is_finite_tensor(x):
                input_issues.append("x_non_finite")
            if not _is_finite_tensor(pos):
                input_issues.append("pos_non_finite")
            if not _is_finite_tensor(leadfield):
                input_issues.append("leadfield_non_finite")

            local_bad_input = len(input_issues) > 0
            any_bad_input = _any_rank_true(local_bad_input, device, world_size)
            if any_bad_input:
                if local_bad_input and nan_guard is not None:
                    nan_guard.record(
                        reason=",".join(input_issues),
                        stage="input",
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=global_step,
                        world_size=world_size,
                        batch=current_batch,
                        leadfield=leadfield,
                        output=None,
                        losses=None,
                        lr=current_lr,
                        scaler=scaler,
                        grad_norm=None,
                        bad_grad_param=None,
                    )
                if nan_guard is not None and nan_guard.skip_bad_batch and not nan_guard.fail_fast:
                    skipped_batches += 1
                    if is_main_process(rank):
                        logger.warning(
                            f"跳过异常 batch (stage=input, epoch={epoch}, batch_idx={batch_idx})"
                        )
                    continue
                raise RuntimeError(
                    f"检测到非有限输入（epoch={epoch}, batch_idx={batch_idx}）。"
                    f"请检查 {diagnostics_hint}"
                )

        use_amp = scaler is not None
        output: Dict[str, torch.Tensor] = {}
        losses: Dict[str, torch.Tensor] = {}
        with torch.autocast(device_type="cuda", enabled=use_amp):
            # 前向传播必须通过 DDP wrapper，才能触发梯度 all-reduce hook
            output = model(x, pos, sensor_type, leadfield=leadfield, return_source=True)
            # 损失计算通过 unwrap_model，避免 DDP 对纯计算方法的 AttributeError
            losses = unwrap_model(model).compute_loss_from_output(
                output, x, loss_weights=loss_weights
            )
            loss = losses["loss"]

        recon_loss = losses["recon_loss"]
        dynamics_loss = losses["dynamics_loss"]
        optimizer_stepped = False
        grad_norm_value: Optional[float] = None
        bad_grad_param: Optional[str] = None

        if guard_enabled:
            fwd_issues: List[str] = []
            if not _is_finite_tensor(output.get("reconstruction")):
                fwd_issues.append("reconstruction_non_finite")
            if not _is_finite_tensor(output.get("source_activity")):
                fwd_issues.append("source_activity_non_finite")
            if not _is_finite_tensor(loss):
                fwd_issues.append("loss_non_finite")
            if not _is_finite_tensor(recon_loss):
                fwd_issues.append("recon_loss_non_finite")
            if not _is_finite_tensor(dynamics_loss):
                fwd_issues.append("dynamics_loss_non_finite")

            local_bad_forward = len(fwd_issues) > 0
            any_bad_forward = _any_rank_true(local_bad_forward, device, world_size)
            if any_bad_forward:
                if local_bad_forward and nan_guard is not None:
                    nan_guard.record(
                        reason=",".join(fwd_issues),
                        stage="forward",
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=global_step,
                        world_size=world_size,
                        batch=current_batch,
                        leadfield=leadfield,
                        output=output,
                        losses=losses,
                        lr=current_lr,
                        scaler=scaler,
                        grad_norm=None,
                        bad_grad_param=None,
                    )
                if nan_guard is not None and nan_guard.skip_bad_batch and not nan_guard.fail_fast:
                    skipped_batches += 1
                    if is_main_process(rank):
                        logger.warning(
                            f"跳过异常 batch (stage=forward, epoch={epoch}, batch_idx={batch_idx})"
                        )
                    continue
                raise RuntimeError(
                    f"检测到非有限前向输出/损失（epoch={epoch}, batch_idx={batch_idx}）。"
                    f"请检查 {diagnostics_hint}"
                )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if gradient_clip > 0:
                grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                grad_norm_value = float(grad_norm_t.detach().item())
            elif guard_enabled:
                grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                grad_norm_value = float(grad_norm_t.detach().item())

            if guard_enabled:
                bwd_issues: List[str] = []
                if grad_norm_value is not None and not math.isfinite(grad_norm_value):
                    bwd_issues.append("grad_norm_non_finite")
                    bad_grad_param = _first_non_finite_grad_name(model)
                local_bad_backward = len(bwd_issues) > 0
                any_bad_backward = _any_rank_true(local_bad_backward, device, world_size)
                if any_bad_backward:
                    if local_bad_backward and nan_guard is not None:
                        nan_guard.record(
                            reason=",".join(bwd_issues),
                            stage="backward",
                            epoch=epoch,
                            batch_idx=batch_idx,
                            global_step=global_step,
                            world_size=world_size,
                            batch=current_batch,
                            leadfield=leadfield,
                            output=output,
                            losses=losses,
                            lr=current_lr,
                            scaler=scaler,
                            grad_norm=grad_norm_value,
                            bad_grad_param=bad_grad_param,
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if nan_guard is not None and nan_guard.skip_bad_batch and not nan_guard.fail_fast:
                        skipped_batches += 1
                        if is_main_process(rank):
                            logger.warning(
                                f"跳过异常 batch (stage=backward, epoch={epoch}, batch_idx={batch_idx})"
                            )
                        continue
                    raise RuntimeError(
                        f"检测到非有限梯度（epoch={epoch}, batch_idx={batch_idx}）。"
                        f"请检查 {diagnostics_hint}"
                    )

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # AMP 出现溢出时会跳过 optimizer.step，此时不应推进 scheduler。
            scale_after = scaler.get_scale()
            optimizer_stepped = scale_after >= scale_before
        else:
            loss.backward()
            if gradient_clip > 0:
                grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                grad_norm_value = float(grad_norm_t.detach().item())
            elif guard_enabled:
                grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                grad_norm_value = float(grad_norm_t.detach().item())

            if guard_enabled:
                bwd_issues: List[str] = []
                if grad_norm_value is not None and not math.isfinite(grad_norm_value):
                    bwd_issues.append("grad_norm_non_finite")
                    bad_grad_param = _first_non_finite_grad_name(model)
                local_bad_backward = len(bwd_issues) > 0
                any_bad_backward = _any_rank_true(local_bad_backward, device, world_size)
                if any_bad_backward:
                    if local_bad_backward and nan_guard is not None:
                        nan_guard.record(
                            reason=",".join(bwd_issues),
                            stage="backward",
                            epoch=epoch,
                            batch_idx=batch_idx,
                            global_step=global_step,
                            world_size=world_size,
                            batch=current_batch,
                            leadfield=leadfield,
                            output=output,
                            losses=losses,
                            lr=current_lr,
                            scaler=scaler,
                            grad_norm=grad_norm_value,
                            bad_grad_param=bad_grad_param,
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if nan_guard is not None and nan_guard.skip_bad_batch and not nan_guard.fail_fast:
                        skipped_batches += 1
                        if is_main_process(rank):
                            logger.warning(
                                f"跳过异常 batch (stage=backward, epoch={epoch}, batch_idx={batch_idx})"
                            )
                        continue
                    raise RuntimeError(
                        f"检测到非有限梯度（epoch={epoch}, batch_idx={batch_idx}）。"
                        f"请检查 {diagnostics_hint}"
                    )

            optimizer.step()
            optimizer_stepped = True

        if scheduler is not None and optimizer_stepped:
            scheduler.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_dynamics_loss += dynamics_loss.item()
        num_batches += 1
        global_step += 1

        if node_prefetcher is not None and is_main_process(rank):
            node_prefetcher.update_progress(batch_idx + 1)

        if global_step >= max_steps:
            if is_main_process(rank):
                logger.info(f"达到最大训练步数 {max_steps}，提前结束当前 epoch")
            break

        if is_main_process(rank) and batch_idx % log_interval == 0:
            avg_loss = total_loss / num_batches
            dyn_val = dynamics_loss.item()
            # 动力学损失通常极小，用科学计数法显示；较大时回退定点格式
            dyn_fmt = f"{dyn_val:.2e}" if dyn_val < 0.001 else f"{dyn_val:.4f}"
            gn_str = ""
            if grad_norm_value is not None and math.isfinite(grad_norm_value):
                gn_str = f" | GradNorm: {grad_norm_value:.4f}"
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) "
                f"Recon: {recon_loss.item():.4f} "
                f"Dynamics: {dyn_fmt}"
                f"{gn_str}"
            )

            if writer is not None and is_main_process(rank):
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/recon_loss", recon_loss.item(), global_step)
                writer.add_scalar("train/dynamics_loss", dynamics_loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                if grad_norm_value is not None and math.isfinite(grad_norm_value):
                    writer.add_scalar("train/grad_norm", grad_norm_value, global_step)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    if skipped_batches > 0 and is_main_process(rank):
        logger.warning(f"本 epoch 跳过异常 batch 数: {skipped_batches}")
    # 跨 rank 归约训练损失，使 rank 0 记录的是全局平均值
    avg_loss_t = reduce_metric(torch.tensor(avg_loss, device=device), world_size)
    return avg_loss_t.item(), global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
    config: dict,
    leadfield_manager: Optional[LeadfieldManager] = None,
    electrode_registry: Optional[ElectrodeConfigRegistry] = None,
    rank: int = 0,
    world_size: int = 1,
):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_pearson = 0.0
    total_snr_db_sum = 0.0
    total_snr_db_count = 0.0
    total_snr_db_skipped = 0.0
    total_nrmse_sum = 0.0
    total_nrmse_count = 0.0
    total_nrmse_skipped = 0.0
    num_batches = 0

    loss_weights = config.get("training", {}).get("loss", {})

    for batch in dataloader:
        x = batch["x"].to(device)
        pos = batch["pos"].to(device)
        sensor_type = batch["sensor_type"].to(device)

        leadfield = None
        if leadfield_manager is not None and electrode_registry is not None:
            fingerprint = batch.get("fingerprint", "unknown")
            leadfield = resolve_leadfield_for_batch(
                fingerprint,
                leadfield_manager, electrode_registry, device,
            )

        # 单次 forward：loss 和指标均从同一次输出计算，避免冗余推理
        output = model(x, pos, sensor_type, leadfield=leadfield, return_source=True)
        reconstruction = output["reconstruction"]
        target = unwrap_model(model)._prepare_target(x, reconstruction.shape[-1])
        losses = unwrap_model(model).compute_loss_from_output(
            output, x, loss_weights=loss_weights
        )
        metrics_batch = compute_all_metrics(target, reconstruction)

        total_loss += losses["loss"].item()
        total_recon_loss += losses["recon_loss"].item()
        total_pearson += metrics_batch["pearson"].item()
        total_snr_db_sum += metrics_batch["snr_db_sum"].item()
        total_snr_db_count += metrics_batch["snr_db_count"].item()
        total_snr_db_skipped += metrics_batch["snr_db_skipped"].item()
        total_nrmse_sum += metrics_batch["nrmse_sum"].item()
        total_nrmse_count += metrics_batch["nrmse_count"].item()
        total_nrmse_skipped += metrics_batch["nrmse_skipped"].item()
        num_batches += 1

    total_loss_t = reduce_metric_sum(torch.tensor(total_loss, device=device), world_size)
    total_recon_loss_t = reduce_metric_sum(
        torch.tensor(total_recon_loss, device=device), world_size
    )
    total_pearson_t = reduce_metric_sum(torch.tensor(total_pearson, device=device), world_size)
    total_num_batches_t = reduce_metric_sum(torch.tensor(num_batches, device=device), world_size)
    total_snr_db_sum_t = reduce_metric_sum(
        torch.tensor(total_snr_db_sum, device=device), world_size
    )
    total_snr_db_count_t = reduce_metric_sum(
        torch.tensor(total_snr_db_count, device=device), world_size
    )
    total_snr_db_skipped_t = reduce_metric_sum(
        torch.tensor(total_snr_db_skipped, device=device), world_size
    )
    total_nrmse_sum_t = reduce_metric_sum(
        torch.tensor(total_nrmse_sum, device=device), world_size
    )
    total_nrmse_count_t = reduce_metric_sum(
        torch.tensor(total_nrmse_count, device=device), world_size
    )
    total_nrmse_skipped_t = reduce_metric_sum(
        torch.tensor(total_nrmse_skipped, device=device), world_size
    )

    if total_snr_db_count_t.item() > 0:
        avg_snr_db = total_snr_db_sum_t / total_snr_db_count_t
    else:
        avg_snr_db = torch.tensor(float("nan"), device=device)

    if total_nrmse_count_t.item() > 0:
        avg_nrmse = total_nrmse_sum_t / total_nrmse_count_t
    else:
        avg_nrmse = torch.tensor(float("nan"), device=device)

    if total_num_batches_t.item() > 0:
        avg_loss_t = total_loss_t / total_num_batches_t
        avg_recon_loss_t = total_recon_loss_t / total_num_batches_t
        avg_pearson_t = total_pearson_t / total_num_batches_t
    else:
        avg_loss_t = torch.tensor(0.0, device=device)
        avg_recon_loss_t = torch.tensor(0.0, device=device)
        avg_pearson_t = torch.tensor(0.0, device=device)

    return {
        "loss": avg_loss_t.item(),
        "recon_loss": avg_recon_loss_t.item(),
        "pearson": avg_pearson_t.item(),
        "snr_db": avg_snr_db.item(),
        "snr_db_valid": total_snr_db_count_t.item(),
        "snr_db_skipped": total_snr_db_skipped_t.item(),
        "nrmse": avg_nrmse.item(),
        "nrmse_valid": total_nrmse_count_t.item(),
        "nrmse_skipped": total_nrmse_skipped_t.item(),
    }


def main():
    parser = argparse.ArgumentParser(description="PENCI 训练脚本")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument(
        "--io_prefetch_warmup_gb",
        type=float,
        default=None,
        help="覆盖 data.io_prefetch.warmup_gb（GiB）",
    )
    parser.add_argument(
        "--ddp_mode",
        type=str,
        default="prod",
        choices=["prod", "debug"],
        help="分布式运行模式: prod=性能优先, debug=诊断优先",
    )
    args = parser.parse_args()

    configure_ddp_mode(args.ddp_mode)

    rank, local_rank, world_size = setup_distributed()
    if not is_main_process(rank):
        logging.getLogger().setLevel(logging.WARNING)

    writer = None
    try:
        config = load_config(args.config)

        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(config.get("paths", {}).get("output_dir", "outputs")) / timestamp

        checkpoint_dir = output_dir / "checkpoints"
        log_dir = output_dir / "logs"

        if is_main_process(rank):
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(exist_ok=True)
            log_dir.mkdir(exist_ok=True)
            # === 文件日志：同时写入带时间戳的 train_<timestamp>.log ===
            _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = output_dir / f"train_{_ts}.log"
            prefetch_detail_log_file = log_dir / f"prefetch_detail_{_ts}.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logging.getLogger().addHandler(file_handler)
            configure_prefetch_detail_logger(prefetch_detail_log_file)
            logger.info(f"训练日志将同步写入: {log_file}")
            logger.info(f"预热细节日志将单独写入: {prefetch_detail_log_file}")
        if is_main_process(rank):
            logger.info(f"输出目录: {output_dir}")
            logger.info(f"DDP 模式: {args.ddp_mode}")
            if args.ddp_mode == "debug":
                logger.info(
                    "DDP 诊断环境: "
                    f"NCCL_DEBUG={os.environ.get('NCCL_DEBUG')}, "
                    f"TORCH_DISTRIBUTED_DEBUG={os.environ.get('TORCH_DISTRIBUTED_DEBUG')}, "
                    f"TORCH_FR_BUFFER_SIZE={os.environ.get('TORCH_FR_BUFFER_SIZE')}"
                )

        if world_size > 1:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if is_main_process(rank):
            logger.info(f"使用设备: {device}")

        # === 物理约束组件初始化 ===
        source_space, leadfield_manager, electrode_registry, registry_from_archive = setup_physics(config)

        # === 创建模型 ===
        if is_main_process(rank):
            logger.info("创建模型...")
        model = build_penci_from_config(config)
        model = model.to(device)
        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=config.get("distributed", {}).get("find_unused_parameters", False),
            )
            # 在 DDP 包装后做首次同步，确保 rank->GPU 映射已明确
            sync_ranks(world_size, local_rank)

        if is_main_process(rank):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")

        # === 创建数据加载器 ===
        if is_main_process(rank):
            logger.info("创建数据加载器...")
        data_config = config.get("data", {})
        training_config = config.get("training", {})
        file_scheduler_cfg = data_config.get("file_scheduler", {})
        io_prefetch_cfg = data_config.get("io_prefetch", {})
        nan_guard_cfg = training_config.get("nan_guard", {})

        use_bucket_sampler = data_config.get("use_bucket_sampler", False)
        use_fingerprint = False
        if leadfield_manager is not None and not use_bucket_sampler:
            if is_main_process(rank):
                logger.warning(
                    "动态导联场模式下建议使用 BucketBatchSampler "
                    "(data.use_bucket_sampler: true)，自动启用"
                )
            use_bucket_sampler = True
        if leadfield_manager is not None:
            use_fingerprint = True
            if is_main_process(rank):
                logger.info("动态导联场模式: 自动启用电极指纹分桶")

        configured_datasets = data_config.get("datasets", ["HBN_EEG"])
        data_root = data_config.get("root_dir", "/work/2024/tanzunsheng/PENCIData")
        train_sampler_file_scheduler = bool(file_scheduler_cfg.get("enabled", False))
        train_sampler_shuffle_within_file = bool(
            file_scheduler_cfg.get("shuffle_within_file", True)
        )
        val_sampler_file_scheduler = bool(
            file_scheduler_cfg.get("val_enabled", train_sampler_file_scheduler)
        )
        val_sampler_shuffle_within_file = bool(
            file_scheduler_cfg.get("val_shuffle_within_file", False)
        )

        train_loader, val_loader = get_train_val_loaders(
            data_root=data_root,
            datasets=configured_datasets,
            batch_size=training_config.get("batch_size", 32),
            num_workers=training_config.get("num_workers", 4),
            use_bucket_sampler=use_bucket_sampler,
            use_fingerprint=use_fingerprint,
            max_length=data_config.get("time_window", 10) * data_config.get("sample_rate", 256),
            target_channels=data_config.get("n_channels", 128) if not use_bucket_sampler else None,
            sampler_file_scheduler=train_sampler_file_scheduler,
            sampler_shuffle_within_file=train_sampler_shuffle_within_file,
            val_sampler_file_scheduler=val_sampler_file_scheduler,
            val_sampler_shuffle_within_file=val_sampler_shuffle_within_file,
            rank=rank,
            world_size=world_size,
        )

        io_prefetch_enabled = bool(io_prefetch_cfg.get("enabled", False))
        io_prefetch_warmup_gb = float(io_prefetch_cfg.get("warmup_gb", 0.0))
        if args.io_prefetch_warmup_gb is not None:
            io_prefetch_warmup_gb = float(args.io_prefetch_warmup_gb)
        io_prefetch_low_watermark_gb = float(
            io_prefetch_cfg.get("low_watermark_gb", 0.75 * io_prefetch_warmup_gb)
        )
        io_prefetch_low_watermark_gb = min(
            io_prefetch_warmup_gb, max(0.0, io_prefetch_low_watermark_gb)
        )
        io_prefetch_chunk_mb = int(io_prefetch_cfg.get("read_chunk_mb", 8))
        io_prefetch_threads = int(io_prefetch_cfg.get("prefetch_threads", 1))
        io_prefetch_each_epoch = bool(io_prefetch_cfg.get("warmup_each_epoch", True))
        io_prefetch_async_refill = bool(io_prefetch_cfg.get("async_refill", True))
        io_prefetch_scope = str(io_prefetch_cfg.get("scope", "node_union")).strip() or "node_union"
        io_prefetch_startup_policy = str(
            io_prefetch_cfg.get("startup_policy", "high_watermark")
        ).strip() or "high_watermark"
        io_prefetch_max_files = int(io_prefetch_cfg.get("max_files", 0))
        io_prefetch_before_val = bool(io_prefetch_cfg.get("warmup_before_val", True))
        io_prefetch_val_warmup_gb = float(
            io_prefetch_cfg.get("warmup_val_gb", io_prefetch_warmup_gb)
        )
        io_prefetch_val_max_files = int(
            io_prefetch_cfg.get("max_files_val", io_prefetch_max_files)
        )
        train_node_union_prefetch = bool(
            io_prefetch_enabled
            and train_sampler_file_scheduler
            and io_prefetch_scope == "node_union"
            and io_prefetch_startup_policy == "high_watermark"
        )
        nan_guard = NaNGuard(nan_guard_cfg, output_dir=output_dir, rank=rank)

        if is_main_process(rank):
            logger.info(f"训练集大小: {len(train_loader.dataset)}")
            logger.info(f"验证集大小: {len(val_loader.dataset)}")
            logger.info(f"BucketBatchSampler: {'启用' if use_bucket_sampler else '禁用'}")
            if use_fingerprint:
                logger.info("电极指纹分桶: 启用")
            logger.info(
                "文件级调度: "
                f"训练={'启用' if train_sampler_file_scheduler else '禁用'} "
                f"(文件内{'打乱' if train_sampler_shuffle_within_file else '顺序'}), "
                f"验证={'启用' if val_sampler_file_scheduler else '禁用'} "
                f"(文件内{'打乱' if val_sampler_shuffle_within_file else '顺序'})"
            )
            if io_prefetch_enabled:
                logger.info(
                    f"I/O 预热窗口: high={io_prefetch_warmup_gb:.1f} GiB, "
                    f"low={io_prefetch_low_watermark_gb:.1f} GiB, "
                    f"scope={io_prefetch_scope}, "
                    f"startup={io_prefetch_startup_policy}, "
                    f"async_refill={'启用' if io_prefetch_async_refill else '禁用'} "
                    f"(CLI 覆盖: {'是' if args.io_prefetch_warmup_gb is not None else '否'})"
                )
                logger.info(
                    "验证前预热: "
                    f"{'启用' if io_prefetch_before_val else '禁用'} | "
                    f"目标 {io_prefetch_val_warmup_gb:.1f} GiB | "
                    f"max_files={io_prefetch_val_max_files if io_prefetch_val_max_files > 0 else '不限'}"
                )
            logger.info(
                "NaN 诊断: "
                f"{'启用' if nan_guard.enabled else '禁用'} | "
                f"fail_fast={nan_guard.fail_fast} | "
                f"skip_bad_batch={nan_guard.skip_bad_batch}"
            )

        # 已从存档加载时，指纹已完备，无需再扫描 ProcessedData
        if electrode_registry is not None and not registry_from_archive:
            loaded_datasets = set()
            for m in train_loader.dataset.metadata:
                loaded_datasets.add(m.get("dataset", "unknown"))
            for ds_name in loaded_datasets:
                if not electrode_registry.has_config_for_dataset(ds_name):
                    try:
                        electrode_registry.register_dataset(ds_name)
                        logger.info(f"  补充注册数据集电极配置: {ds_name}")
                    except FileNotFoundError as e:
                        logger.warning(f"  补充注册数据集 '{ds_name}' 失败: {e}")
            logger.info(
                f"  最终电极配置: {electrode_registry.registered_configs}"
            )
        if electrode_registry is not None and is_main_process(rank):
            logger.info(
                f"  已注册指纹数: {len(electrode_registry.get_all_fingerprints())}"
            )

        # === 导联场预热 ===
        # 预计算所有已注册指纹对应的导联场，避免训练循环中首次命中时卡顿
        # 仅 rank 0 执行预热（填充磁盘缓存），其他 rank 等待后从缓存读取
        if leadfield_manager is not None and electrode_registry is not None:
            all_fps = electrode_registry.get_all_fingerprints()
            if all_fps:
                if is_main_process(rank):
                    logger.info(f"预热导联场: {len(all_fps)} 个唯一电极配置 (仅 rank 0)...")
                    for fp in all_fps:
                        try:
                            names, positions = electrode_registry.get_config_by_fingerprint(fp)
                            L = leadfield_manager.get_leadfield(names, positions, device)
                            logger.info(f"  指纹 {fp}: 导联场 {L.shape}")
                        except Exception as e:
                            logger.warning(f"  指纹 {fp} 导联场预热失败: {e}")
                    logger.info("导联场预热完成")
                # 所有 rank 同步：确保 rank 0 预热完毕，其他 rank 可从磁盘缓存加载
                sync_ranks(world_size, local_rank)

        # === 优化器与调度器 ===
        base_lr = training_config.get("learning_rate", 1e-4)
        effective_lr = base_lr * world_size  # linear scaling rule
        if is_main_process(rank):
            logger.info(
                f"学习率缩放: base_lr={base_lr} × world_size={world_size} = {effective_lr}"
            )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=effective_lr,
            weight_decay=training_config.get("weight_decay", 0.01),
        )

        warmup_steps = training_config.get("warmup_steps", 1000)
        max_steps = training_config.get("max_steps", 100000)
        min_lr_ratio = 1e-6 / effective_lr  # eta_min=1e-6 equivalent
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps,
            max_steps,
            min_lr_ratio,
        )

        writer = SummaryWriter(log_dir) if is_main_process(rank) else None

        # === 混合精度训练 ===
        use_amp = config.get("hardware", {}).get("mixed_precision", True)
        scaler = torch.amp.GradScaler() if use_amp and device.type == "cuda" else None

        # === 恢复训练 ===
        start_epoch = 0
        global_step = 0
        best_val_loss = float("inf")
        val_metrics = {"loss": float("inf")}

        if args.resume:
            map_location = f"cuda:{local_rank}" if world_size > 1 else device
            checkpoint = torch.load(args.resume, map_location=map_location)
            target_model = model.module if hasattr(model, 'module') else model
            target_model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if scaler and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            global_step = checkpoint.get("step", 0)
            start_epoch = checkpoint.get("epoch", 0)
            if is_main_process(rank):
                logger.info(f"已恢复检查点: {args.resume}")

        # === 训练循环 ===
        if is_main_process(rank):
            logger.info("开始训练...")
        max_epochs = max_steps // max(1, len(train_loader)) + 1

        final_epoch = start_epoch
        train_prefetcher: Optional[NodePageCachePrefetcher] = None
        for epoch in range(start_epoch, max_epochs):
            final_epoch = epoch
            if is_main_process(rank):
                logger.info(f"=== Epoch {epoch + 1}/{max_epochs} ===")

            # BucketBatchSampler / DistributedBucketBatchSampler 路径
            if hasattr(train_loader, "batch_sampler") and hasattr(
                train_loader.batch_sampler, "set_epoch"
            ):
                train_loader.batch_sampler.set_epoch(epoch)
            # 标准 DistributedSampler 路径（非 bucket 模式）
            elif hasattr(train_loader, "sampler") and hasattr(
                train_loader.sampler, "set_epoch"
            ):
                train_loader.sampler.set_epoch(epoch)

            if train_prefetcher is not None:
                train_prefetcher.stop()
                train_prefetcher = None

            if io_prefetch_enabled and (
                io_prefetch_each_epoch or epoch == start_epoch
            ):
                if train_node_union_prefetch:
                    local_rank_schedule = get_prefetch_rank_schedule(train_loader)
                    if world_size > 1:
                        gathered_rank_schedules: List[List[Dict[str, Any]]] = [
                            [] for _ in range(world_size)
                        ]
                        dist.all_gather_object(gathered_rank_schedules, local_rank_schedule)
                    else:
                        gathered_rank_schedules = [local_rank_schedule]

                    if is_main_process(rank):
                        plan_counts = [len(items) for items in gathered_rank_schedules]
                        logger.info(
                            "文件窗口来源: "
                            + ", ".join(
                                f"rank{rank_idx}={count}"
                                for rank_idx, count in enumerate(plan_counts)
                            )
                        )
                        node_union_plan = build_node_union_prefetch_plan(
                            gathered_rank_schedules,
                            data_root=data_root,
                            max_files=io_prefetch_max_files,
                        )
                        unique_files, unique_bytes = _summarize_prefetch_plan_files(
                            node_union_plan
                        )
                        logger.info(
                            f"节点级联合预热窗口: {len(node_union_plan)} 段 | "
                            f"唯一文件 {unique_files} 个 | "
                            f"唯一总大小 {unique_bytes / (1024 ** 3):.2f} GiB | "
                            f"rank 窗口总数 {sum(plan_counts)}"
                        )
                        if node_union_plan:
                            train_prefetcher = NodePageCachePrefetcher(
                                node_union_plan,
                                high_watermark_gb=io_prefetch_warmup_gb,
                                low_watermark_gb=io_prefetch_low_watermark_gb,
                                read_chunk_mb=io_prefetch_chunk_mb,
                                max_threads=io_prefetch_threads,
                            )
                            train_prefetcher.warmup_to_high_watermark()
                            if io_prefetch_async_refill:
                                train_prefetcher.start_async_refill()
                        else:
                            logger.warning("节点级联合预热窗口为空，本 epoch 跳过训练前预热")
                    sync_ranks(world_size, local_rank)
                else:
                    if is_main_process(rank):
                        prefetch_plan = get_prefetch_file_plan(
                            train_loader,
                            data_root=data_root,
                            max_files=io_prefetch_max_files,
                        )
                        warmup_hdf5_page_cache(
                            prefetch_plan,
                            warmup_gb=io_prefetch_warmup_gb,
                            read_chunk_mb=io_prefetch_chunk_mb,
                            max_threads=io_prefetch_threads,
                        )
                    sync_ranks(world_size, local_rank)

            train_loss, global_step = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch,
                config,
                writer,
                global_step,
                max_steps=max_steps,
                leadfield_manager=leadfield_manager,
                electrode_registry=electrode_registry,
                scaler=scaler,
                scheduler=scheduler,
                rank=rank,
                world_size=world_size,
                nan_guard=nan_guard,
                node_prefetcher=train_prefetcher if is_main_process(rank) else None,
            )

            if train_prefetcher is not None:
                train_prefetcher.stop()
                train_prefetcher = None

            if io_prefetch_enabled and io_prefetch_before_val:
                if is_main_process(rank):
                    val_prefetch_plan = get_prefetch_file_plan(
                        val_loader,
                        data_root=data_root,
                        max_files=io_prefetch_val_max_files,
                    )
                    warmup_hdf5_page_cache(
                        val_prefetch_plan,
                        warmup_gb=io_prefetch_val_warmup_gb,
                        read_chunk_mb=io_prefetch_chunk_mb,
                        max_threads=io_prefetch_threads,
                    )
                sync_ranks(world_size, local_rank)

            val_metrics = evaluate(
                model,
                val_loader,
                device,
                config,
                leadfield_manager=leadfield_manager,
                electrode_registry=electrode_registry,
                rank=rank,
                world_size=world_size,
            )

            if is_main_process(rank):
                logger.info(
                    f"验证 - Loss: {val_metrics['loss']:.4f}, "
                    f"Recon: {val_metrics['recon_loss']:.4f}, "
                    f"Pearson: {val_metrics['pearson']:.4f}, "
                    f"SNR(dB): {val_metrics['snr_db']:.4f}, "
                    f"NRMSE: {val_metrics['nrmse']:.4f}"
                )
                skipped_low_energy = max(
                    int(val_metrics["snr_db_skipped"]),
                    int(val_metrics["nrmse_skipped"]),
                )
                if skipped_low_energy > 0:
                    logger.info(
                        "验证指标过滤: 跳过低能量通道 "
                        f"{skipped_low_energy} 个 | "
                        f"SNR有效={int(val_metrics['snr_db_valid'])} | "
                        f"NRMSE有效={int(val_metrics['nrmse_valid'])}"
                    )
                if math.isnan(val_metrics["snr_db"]) or math.isnan(val_metrics["nrmse"]):
                    logger.warning("验证集中缺少足够的有效信号，SNR/NRMSE 记为 NaN")
                if writer is not None:
                    writer.add_scalar("val/loss", val_metrics["loss"], global_step)
                    writer.add_scalar("val/recon_loss", val_metrics["recon_loss"], global_step)
                    writer.add_scalar("val/pearson", val_metrics["pearson"], global_step)
                    writer.add_scalar("val/snr_db", val_metrics["snr_db"], global_step)
                    writer.add_scalar("val/nrmse", val_metrics["nrmse"], global_step)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                if is_main_process(rank):
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        global_step,
                        epoch,
                        val_metrics["loss"],
                        checkpoint_dir / "best_model.pt",
                    )
                sync_ranks(world_size, local_rank)

            if (epoch + 1) % 5 == 0:
                if is_main_process(rank):
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        global_step,
                        epoch,
                        val_metrics["loss"],
                        checkpoint_dir / f"epoch_{epoch + 1}.pt",
                    )
                sync_ranks(world_size, local_rank)

            if global_step >= max_steps:
                if is_main_process(rank):
                    logger.info("达到最大训练步数，停止训练")
                break

            if is_main_process(rank):
                logger.info(f"Epoch {epoch + 1} 训练平均损失: {train_loss:.4f}")

        if is_main_process(rank):
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                global_step,
                final_epoch,
                val_metrics["loss"],
                checkpoint_dir / "final_model.pt",
            )
        sync_ranks(world_size, local_rank)

        if is_main_process(rank):
            logger.info("训练完成！")
    finally:
        if 'train_prefetcher' in locals() and train_prefetcher is not None:
            train_prefetcher.stop()
        if writer is not None:
            writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
