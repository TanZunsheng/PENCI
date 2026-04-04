#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线预计算全局电极指纹注册表 + 导联场矩阵

两阶段架构的「离线阶段」：
1. 扫描 PENCIData 下所有 metadata JSON，按目录组流式计算样本指纹
2. 从 .pt 文件读取 normalized pos 计算 pos_fingerprint（与运行时完全一致）
3. 去重：只有真正新的 pos_fingerprint 才需要计算导联场
4. 从 ProcessedData 的 electrodes.tsv 读取米制坐标 + 通道名（用于 MNE 前向计算）
5. 调用 MNE 计算导联场并缓存到磁盘
6. 保存 archive .pt 文件（供 ElectrodeConfigRegistry.load_from_archive() 使用）
7. 可选：回写 metadata JSON 中每条记录的 "fingerprint" 字段（零 I/O 启动）

Scheme B 设计要点：
- 指纹从 .pt 文件的 normalized bfloat16 pos 计算（保证与 dataset.py 运行时一致）
- 导联场从 electrodes.tsv 的米制坐标计算（MNE 需要物理单位）
- 不从 BrainOmni 导入任何代码

用法:
    # 完整运行（扫描 + 计算导联场 + 保存 archive）
    python scripts/precompute_all_leadfields.py \\
        --penci_data /work/2024/tanzunsheng/PENCIData \\
        --processed_data /work/2024/tanzunsheng/ProcessedData \\
        --subjects_dir /work/2024/tanzunsheng/mne_data/MNE-fsaverage-data \\
        --cache_dir /work/2024/tanzunsheng/leadfield_cache \\
        --archive_out /work/2024/tanzunsheng/leadfield_cache/fingerprint_registry.pt

    # 仅扫描（不计算导联场，不调用 MNE）
    python scripts/precompute_all_leadfields.py \\
        --penci_data /work/2024/tanzunsheng/PENCIData \\
        --processed_data /work/2024/tanzunsheng/ProcessedData \\
        --dry_run

    # 回写 fingerprint 到 metadata JSON
    python scripts/precompute_all_leadfields.py \\
        --penci_data /work/2024/tanzunsheng/PENCIData \\
        --processed_data /work/2024/tanzunsheng/ProcessedData \\
        --subjects_dir /work/2024/tanzunsheng/mne_data/MNE-fsaverage-data \\
        --cache_dir /work/2024/tanzunsheng/leadfield_cache \\
        --archive_out /work/2024/tanzunsheng/leadfield_cache/fingerprint_registry.pt \\
        --update_metadata
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from penci.physics.leadfield_manager import (
    _compute_channel_hash,
    compute_fingerprint_from_pos,
)
from penci.physics.electrode_utils import (
    ElectrodeConfigRegistry,
    find_electrodes_tsv,
    read_electrodes_tsv,
    filter_channels_like_postprocess,
    NON_EEG_CHANNEL_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_path_list_arg(raw: str) -> Tuple[str, ...]:
    """解析逗号分隔路径参数。"""
    if not raw.strip():
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


# ============================================================================
# 第一步：扫描 metadata + 计算 pos_fingerprint（从 .pt 文件）
# ============================================================================


def discover_metadata_files(penci_data_dir: str) -> List[Path]:
    """
    发现 PENCIData 下所有 metadata JSON 文件

    搜索两层深度的 *-metadata/ 目录，覆盖两种布局：
      - 扁平布局: PENCIData/{dataset}-metadata/train.json（大部分数据集）
      - 嵌套布局: PENCIData/{parent}/{subdataset}-metadata/train.json（如 Broderick2018）

    参数:
        penci_data_dir: PENCIData 根目录

    返回:
        JSON 文件路径列表（按路径排序，去重）
    """
    root = Path(penci_data_dir)
    json_files: List[Path] = []
    seen: set = set()

    def _collect_from_dir(parent: Path) -> None:
        """从 parent 下查找 *-metadata/ 目录并收集 JSON 文件"""
        if not parent.is_dir():
            return
        for d in sorted(parent.iterdir()):
            if d.is_dir() and d.name.endswith("-metadata"):
                for json_name in ["train.json", "val.json"]:
                    json_path = d / json_name
                    if json_path.exists() and json_path not in seen:
                        seen.add(json_path)
                        json_files.append(json_path)

    # 第一层：PENCIData/{dataset}-metadata/
    _collect_from_dir(root)

    # 第二层：PENCIData/{parent}/{subdataset}-metadata/（嵌套子数据集）
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.endswith("-metadata"):
            _collect_from_dir(child)

    return sorted(json_files)


def compute_fingerprint_from_pt(pt_path: str) -> Optional[str]:
    """
    从 .pt 文件的 pos 张量计算 pos_fingerprint

    与 dataset.py 中 _compute_sample_fingerprint() 逻辑完全一致。

    参数:
        pt_path: .pt 文件路径

    返回:
        16 字符十六进制指纹，失败返回 None
    """
    try:
        data = torch.load(pt_path, weights_only=True)
        pos = data["pos"].float().numpy()  # (C, 6) bfloat16 → float32 → numpy
        xyz = pos[:, :3]  # 仅取空间坐标
        return compute_fingerprint_from_pos(xyz)
    except Exception as e:
        logger.warning(f"无法读取 .pt 文件: {pt_path}: {e}")
        return None


def build_fingerprint_group_key(meta: Dict[str, Any]) -> str:
    """
    为目录级指纹复用构造稳定分组键。

    默认按 (dataset, channels, parent_dir) 分组，避免不同数据集或不同通道数下
    恰好同名目录被错误合并。
    """
    pt_path = str(meta["path"])
    parent_dir = str(Path(pt_path).parent)
    dataset_name = str(meta.get("dataset", "unknown"))
    n_channels = int(meta.get("channels", 0))
    return f"{dataset_name}|{n_channels}|{parent_dir}"


def select_group_validation_indices(
    group_key: str,
    sample_paths: List[str],
    random_checks: int = 2,
) -> List[int]:
    """
    选择目录级轻量校验样本下标。

    规则：
    - 始终包含第 1 个和最后 1 个样本
    - 在中间样本中再确定性随机抽取 2 个
    - 若中间样本不足 2 个，则全部纳入
    """
    if not sample_paths:
        return []

    sample_count = len(sample_paths)
    if sample_count == 1:
        return [0]

    selected = {0, sample_count - 1}
    middle_indices = list(range(1, sample_count - 1))
    if not middle_indices:
        return sorted(selected)

    if len(middle_indices) <= random_checks:
        selected.update(middle_indices)
        return sorted(selected)

    seed = int.from_bytes(hashlib.sha256(group_key.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    random_middle = rng.choice(middle_indices, size=random_checks, replace=False)
    selected.update(int(index) for index in random_middle.tolist())
    return sorted(selected)


def compute_fingerprint_group_task(
    group_key: str,
    sample_paths: List[str],
    small_group_threshold: int = 4,
    random_checks: int = 2,
) -> Dict[str, Any]:
    """
    对单个目录组执行指纹计算与轻量校验。

    返回:
        {
            "mode": "broadcast" | "per_sample",
            "broadcast_fp": Optional[str],
            "sample_fingerprints": List[(path, fp)],
            "validated_count": int,
            "group_size": int,
            "fallback_reason": str,
        }
    """
    ordered_paths = sorted(str(path) for path in sample_paths)
    group_size = len(ordered_paths)

    if group_size <= small_group_threshold:
        return {
            "mode": "per_sample",
            "broadcast_fp": None,
            "sample_fingerprints": [
                (path, compute_fingerprint_from_pt(path))
                for path in ordered_paths
            ],
            "validated_count": group_size,
            "group_size": group_size,
            "fallback_reason": "small_group",
        }

    validation_indices = select_group_validation_indices(
        group_key=group_key,
        sample_paths=ordered_paths,
        random_checks=random_checks,
    )
    checked_fingerprints: Dict[str, Optional[str]] = {}
    baseline_fp: Optional[str] = None
    fallback_reason: Optional[str] = None

    for index in validation_indices:
        path = ordered_paths[index]
        fp = compute_fingerprint_from_pt(path)
        checked_fingerprints[path] = fp

        if fp is None:
            fallback_reason = "validation_failed"
            break
        if baseline_fp is None:
            baseline_fp = fp
        elif fp != baseline_fp:
            fallback_reason = "validation_mismatch"
            break

    if fallback_reason is None and baseline_fp is not None:
        return {
            "mode": "broadcast",
            "broadcast_fp": baseline_fp,
            "sample_fingerprints": [],
            "validated_count": len(validation_indices),
            "group_size": group_size,
            "fallback_reason": "",
        }

    sample_fingerprints: List[Tuple[str, Optional[str]]] = []
    for path in ordered_paths:
        fp = checked_fingerprints.get(path)
        if fp is None:
            fp = compute_fingerprint_from_pt(path)
        sample_fingerprints.append((path, fp))

    return {
        "mode": "per_sample",
        "broadcast_fp": None,
        "sample_fingerprints": sample_fingerprints,
        "validated_count": len(validation_indices),
        "group_size": group_size,
        "fallback_reason": fallback_reason or "unknown",
    }


def register_fingerprint_record(
    path_to_fingerprint: Dict[str, str],
    unique_fingerprints: Dict[str, Dict[str, Any]],
    pt_path: str,
    fingerprint: Optional[str],
    dataset_name: str,
    n_channels: int,
    max_example_candidates: int,
) -> bool:
    """
    记录单个样本的指纹映射，并更新唯一指纹摘要。

    返回:
        是否成功记录（False 表示 fingerprint 无效）。
    """
    if fingerprint in (None, "", "unknown"):
        return False

    fp = str(fingerprint)
    path_to_fingerprint[pt_path] = fp
    if fp not in unique_fingerprints:
        unique_fingerprints[fp] = {
            "datasets": {dataset_name},
            "channels": n_channels,
            "example_candidates": [(pt_path, dataset_name)],
        }
        return True

    unique_fingerprints[fp]["datasets"].add(dataset_name)
    candidates = unique_fingerprints[fp]["example_candidates"]
    existing_datasets = {ds for _, ds in candidates}
    if dataset_name not in existing_datasets and len(candidates) < max_example_candidates:
        candidates.append((pt_path, dataset_name))
    return True


def scan_and_fingerprint(
    penci_data_dir: str,
    max_workers: int = 16,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, str],
    List[Dict[str, Any]],
]:
    """
    扫描所有 metadata，为每个样本计算精确的 pos_fingerprint（支持增量计算）

    增量策略：
    - 已有 fingerprint 字段的样本直接零 I/O 复用
    - 对缺失 fingerprint 的样本按 (dataset, channels, parent_dir) 分组
    - 默认每组仅抽查 4 个样本（首/尾 + 中间随机 2 个）做轻量校验
    - 小目录（<= 4 个样本）直接逐样本计算
    - 若抽查发现不一致或读取失败，则该目录回退为逐样本计算

    参数:
        penci_data_dir: PENCIData 根目录
        max_workers: 并行计算进程数（默认 16）

    返回:
        (unique_fingerprints, path_to_fingerprint, all_metadata) 元组:
        - unique_fingerprints: {pos_fp: {"datasets": set, "channels": int,
          "example_candidates": [(pt_path, dataset_name), ...]}}
          每个 fingerprint 最多保存 MAX_EXAMPLE_CANDIDATES 个候选，优先跨 dataset 多样化
        - path_to_fingerprint: {pt_path: pos_fp}
        - all_metadata: 所有 metadata 记录（含 _json_source 和 _json_index）
    """
    json_files = discover_metadata_files(penci_data_dir)
    logger.info(f"发现 {len(json_files)} 个 metadata JSON 文件")

    # 汇总所有 metadata 记录
    all_metadata: List[Dict[str, Any]] = []
    for json_path in json_files:
        with open(json_path, "r") as f:
            records = json.load(f)
        for idx, record in enumerate(records):
            record["_json_source"] = str(json_path)
            record["_json_index"] = idx
            all_metadata.append(record)

    logger.info(f"共 {len(all_metadata)} 条样本记录")

    # 增量策略：已有 fingerprint 字段的样本直接复用，仅对缺失的样本并行计算
    path_to_fingerprint: Dict[str, str] = {}
    unique_fingerprints: Dict[str, Dict[str, Any]] = {}
    need_compute_groups: Dict[str, List[Tuple[int, str, str, int]]] = defaultdict(list)
    MAX_EXAMPLE_CANDIDATES = 5  # 每个 fingerprint 最多保存的候选数
    SMALL_GROUP_THRESHOLD = 4
    RANDOM_CHECKS = 2

    for idx, meta in enumerate(all_metadata):
        fp = meta.get("fingerprint")
        pt_path = meta["path"]
        dataset_name = meta.get("dataset", "unknown")
        n_channels = meta.get("channels", 0)

        if fp and fp not in (None, "", "unknown"):
            # 已有有效指纹 → 零 I/O 复用
            register_fingerprint_record(
                path_to_fingerprint=path_to_fingerprint,
                unique_fingerprints=unique_fingerprints,
                pt_path=pt_path,
                fingerprint=fp,
                dataset_name=dataset_name,
                n_channels=n_channels,
                max_example_candidates=MAX_EXAMPLE_CANDIDATES,
            )
        else:
            # 缺失指纹 → 需要计算
            group_key = build_fingerprint_group_key(meta)
            need_compute_groups[group_key].append(
                (idx, pt_path, str(dataset_name), int(n_channels))
            )

    total_need_compute = sum(len(items) for items in need_compute_groups.values())
    cached_count = len(all_metadata) - total_need_compute
    logger.info(
        f"增量检测: {cached_count} 个样本已有指纹（零 I/O 复用），"
        f"{total_need_compute} 个样本需要计算"
    )

    if not need_compute_groups:
        logger.info(
            f"\n扫描完成: {len(unique_fingerprints)} 个唯一 pos_fingerprint，"
            f"{len(path_to_fingerprint)} 个样本已标记（全部来自缓存）"
        )
        return unique_fingerprints, path_to_fingerprint, all_metadata

    total_groups = len(need_compute_groups)
    small_group_count = sum(
        1 for items in need_compute_groups.values()
        if len(items) <= SMALL_GROUP_THRESHOLD
    )
    logger.info(
        "目录复用模式: %d 个待计算样本归并为 %d 个目录组 "
        "(小目录 %d, 抽查规则=首尾+中间随机%d个)",
        total_need_compute,
        total_groups,
        small_group_count,
        RANDOM_CHECKS,
    )
    logger.info("并行计算目录组指纹（%d 个进程）...", max_workers)

    failed_count = 0
    fallback_groups = 0
    fallback_reason_counts: Dict[str, int] = defaultdict(int)
    broadcast_groups = 0
    covered_samples = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        grouped_items = [
            (group_key, sorted(items, key=lambda item: item[1]))
            for group_key, items in sorted(need_compute_groups.items(), key=lambda item: item[0])
        ]
        future_to_task: Dict[Any, Tuple[str, List[Tuple[int, str, str, int]]]] = {}
        next_task_idx = 0
        max_inflight = max(max_workers * 4, 1)
        completed_groups = 0
        log_interval = max(total_groups // 20, 1)  # 每 5% 打印一次

        def submit_until_full() -> None:
            nonlocal next_task_idx
            while next_task_idx < len(grouped_items) and len(future_to_task) < max_inflight:
                group_key, group_items = grouped_items[next_task_idx]
                sample_paths = [pt_path for _, pt_path, _, _ in group_items]
                future = executor.submit(
                    compute_fingerprint_group_task,
                    group_key,
                    sample_paths,
                    SMALL_GROUP_THRESHOLD,
                    RANDOM_CHECKS,
                )
                future_to_task[future] = (group_key, group_items)
                next_task_idx += 1

        submit_until_full()

        while future_to_task:
            done_futures, _ = wait(
                list(future_to_task.keys()),
                return_when=FIRST_COMPLETED,
            )

            for future in done_futures:
                group_key, group_items = future_to_task.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    logger.warning("目录组 %s 指纹计算异常: %s", group_key, exc)
                    result = {
                        "mode": "per_sample",
                        "broadcast_fp": None,
                        "sample_fingerprints": [
                            (pt_path, None) for _, pt_path, _, _ in group_items
                        ],
                        "validated_count": 0,
                        "group_size": len(group_items),
                        "fallback_reason": "group_exception",
                    }

                if result["mode"] == "broadcast":
                    broadcast_groups += 1
                    fp = result["broadcast_fp"]
                    for _, pt_path, dataset_name, n_channels in group_items:
                        ok = register_fingerprint_record(
                            path_to_fingerprint=path_to_fingerprint,
                            unique_fingerprints=unique_fingerprints,
                            pt_path=pt_path,
                            fingerprint=fp,
                            dataset_name=dataset_name,
                            n_channels=n_channels,
                            max_example_candidates=MAX_EXAMPLE_CANDIDATES,
                        )
                        if not ok:
                            failed_count += 1
                else:
                    fallback_groups += 1
                    reason = str(result.get("fallback_reason") or "unknown")
                    fallback_reason_counts[reason] += 1
                    fingerprint_by_path = {
                        path: fp
                        for path, fp in result.get("sample_fingerprints", [])
                    }
                    for _, pt_path, dataset_name, n_channels in group_items:
                        ok = register_fingerprint_record(
                            path_to_fingerprint=path_to_fingerprint,
                            unique_fingerprints=unique_fingerprints,
                            pt_path=pt_path,
                            fingerprint=fingerprint_by_path.get(pt_path),
                            dataset_name=dataset_name,
                            n_channels=n_channels,
                            max_example_candidates=MAX_EXAMPLE_CANDIDATES,
                        )
                        if not ok:
                            failed_count += 1

                covered_samples += len(group_items)
                completed_groups += 1
                if completed_groups % log_interval == 0 or completed_groups == total_groups:
                    logger.info(
                        "  进度: 目录组 %d/%d (%d%%), 覆盖样本 %d/%d, "
                        "唯一指纹数: %d, 广播组: %d, 回退组: %d, 失败样本: %d",
                        completed_groups,
                        total_groups,
                        completed_groups * 100 // total_groups,
                        covered_samples,
                        total_need_compute,
                        len(unique_fingerprints),
                        broadcast_groups,
                        fallback_groups,
                        failed_count,
                    )

            submit_until_full()

    if failed_count > 0:
        logger.warning(f"共 {failed_count} 个样本指纹计算失败")
    if fallback_groups > 0:
        logger.info(
            "目录回退统计: %s",
            dict(sorted(fallback_reason_counts.items())),
        )

    logger.info(
        f"\n扫描完成: {len(unique_fingerprints)} 个唯一 pos_fingerprint，"
        f"{len(path_to_fingerprint)} 个样本已标记"
    )
    return unique_fingerprints, path_to_fingerprint, all_metadata


# ============================================================================
# 第二步：为每个唯一 pos_fingerprint 查找对应的 electrodes.tsv 米制坐标
# ============================================================================


def extract_subject_info_from_pt_path(
    pt_path: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    从 .pt 文件路径中提取 subject_id、session、site 信息

    路径模式举例:
        .../HBN_EEG/derivatives/preprocessing/sub-NDARKX701BJ4/eeg/sub-NDARKX701BJ4_task-..._eeg/22_data.pt
        .../ThingsEEG/derivatives/preprocessing/sub-01/ses-01/eeg/.../0_data.pt
        .../THINGS-EEG/derivatives/preprocessing/sub-01/eeg/.../0_data.pt

    返回:
        (subject_id, session, site) — subject_id 不含 "sub-" 前缀
    """
    # 提取 subject_id
    sub_match = re.search(r"sub-([^/\\]+)", pt_path)
    subject_id = sub_match.group(1) if sub_match else None

    # 提取 session
    ses_match = re.search(r"ses-([^/\\]+)", pt_path)
    session = ses_match.group(1) if ses_match else None

    # 提取 site（HBN_EEG 特有: HBN_cmi_bids_{site}）
    site_match = re.search(r"HBN_cmi_bids_([^/\\]+)", pt_path)
    site = site_match.group(1) if site_match else None

    return subject_id, session, site


def find_tsv_for_fingerprint(
    fp_info: Dict[str, Any],
    processed_data_dir: str,
) -> Optional[Tuple[List[str], np.ndarray]]:
    """
    为一个 pos_fingerprint 查找对应的 electrodes.tsv 并返回有效通道配置

    遍历 fp_info["example_candidates"] 中的多个候选 (pt_path, dataset_name)，
    依次尝试查找 electrodes.tsv。第一个成功即返回，提供 fallback 容错。

    参数:
        fp_info: 指纹信息字典，含 datasets/channels/example_candidates
        processed_data_dir: ProcessedData 根目录

    返回:
        (channel_names, channel_positions_m) 或 None
        - channel_names: 有效通道名列表
        - channel_positions_m: (N, 3) 米制坐标数组
    """
    candidates = fp_info["example_candidates"]
    n_candidates = len(candidates)

    for attempt_idx, (pt_path, dataset_name) in enumerate(candidates):
        subject_id, session, site = extract_subject_info_from_pt_path(pt_path)

        if subject_id is None:
            logger.debug(
                f"候选 {attempt_idx + 1}/{n_candidates}: "
                f"无法从路径提取 subject_id: {pt_path}"
            )
            continue

        try:
            tsv_path = find_electrodes_tsv(
                processed_data_dir, dataset_name, subject_id,
                session=session, site=site,
            )
        except FileNotFoundError:
            logger.debug(
                f"候选 {attempt_idx + 1}/{n_candidates}: "
                f"找不到 electrodes.tsv (dataset={dataset_name}, "
                f"subject={subject_id})"
            )
            continue

        # 读取 + 过滤（1:1 复刻 BrainOmniPostProcess 规则）
        try:
            all_channels = read_electrodes_tsv(str(tsv_path))

            # 排除已知非 EEG 通道
            for name in list(all_channels.keys()):
                if name in NON_EEG_CHANNEL_NAMES:
                    del all_channels[name]

            valid_channels = filter_channels_like_postprocess(all_channels)

            channel_names = list(valid_channels.keys())
            channel_positions = np.array(
                [valid_channels[name] for name in channel_names],
                dtype=np.float64,
            )

            if attempt_idx > 0:
                logger.info(
                    f"第 {attempt_idx + 1} 个候选成功 "
                    f"(dataset={dataset_name}, subject={subject_id})"
                )

            return channel_names, channel_positions

        except Exception as e:
            logger.debug(
                f"候选 {attempt_idx + 1}/{n_candidates}: "
                f"处理 electrodes.tsv 失败: {tsv_path}: {e}"
            )
            continue

    # 所有候选均失败
    logger.warning(
        f"所有 {n_candidates} 个候选均未找到有效 electrodes.tsv, "
        f"datasets={fp_info['datasets']}"
    )
    return None


# ============================================================================
# 第三步：计算导联场（调用 MNE）
# ============================================================================


def compute_leadfield_for_config(
    channel_names: List[str],
    channel_positions: np.ndarray,
    subjects_dir: str,
    cache_dir: str,
) -> Optional[str]:
    """
    为一组电极配置计算导联场并缓存

    使用 LeadfieldManager 的完整流程：
    - 如果磁盘缓存已存在，跳过
    - 否则调用 MNE 计算并保存

    参数:
        channel_names: 通道名列表
        channel_positions: (N, 3) 米制坐标
        subjects_dir: FreeSurfer subjects 目录
        cache_dir: 导联场缓存目录

    返回:
        导联场缓存文件路径（成功），或 None（失败）
    """
    from penci.physics.source_space import SourceSpace
    from penci.physics.leadfield_manager import LeadfieldManager

    try:
        ss = SourceSpace(subjects_dir=subjects_dir)
        lm = LeadfieldManager(
            source_space=ss,
            subjects_dir=subjects_dir,
            cache_dir=cache_dir,
        )

        # get_leadfield 内部处理缓存逻辑
        L = lm.get_leadfield(channel_names, channel_positions)

        # 计算 full_fingerprint（用于定位缓存文件）
        full_fp = _compute_channel_hash(channel_names, channel_positions)
        cache_file = os.path.join(cache_dir, f"{full_fp}.pt")

        logger.info(
            f"  导联场: {L.shape}, full_fp={full_fp}, "
            f"缓存={'已存在' if os.path.exists(cache_file) else '新建'}"
        )
        return cache_file

    except Exception as e:
        logger.error(f"  导联场计算失败: {e}")
        return None


# ============================================================================
# 第四步：保存 archive + 可选回写 metadata
# ============================================================================


def save_archive(
    registry: ElectrodeConfigRegistry,
    archive_path: str,
    cache_dir: Optional[str] = None,
) -> None:
    """保存指纹注册表到 archive 文件"""
    registry.save_to_archive(archive_path, leadfield_cache_dir=cache_dir)
    logger.info(f"Archive 已保存: {archive_path}")


def update_metadata_fingerprints(
    all_metadata: List[Dict[str, Any]],
    path_to_fingerprint: Dict[str, str],
) -> int:
    """
    回写 fingerprint 字段到 metadata JSON 文件

    按来源 JSON 文件分组，读取 → 更新 fingerprint 字段 → 写回。

    参数:
        all_metadata: 所有 metadata 记录（含 _json_source 和 _json_index）
        path_to_fingerprint: {pt_path: pos_fingerprint}

    返回:
        更新的 JSON 文件数量
    """
    # 按来源 JSON 文件分组
    json_updates: Dict[str, Dict[int, str]] = defaultdict(dict)
    for meta in all_metadata:
        pt_path = meta["path"]
        json_source = meta["_json_source"]
        json_index = meta["_json_index"]
        if pt_path in path_to_fingerprint:
            json_updates[json_source][json_index] = path_to_fingerprint[pt_path]

    updated_count = 0
    for json_path, index_to_fp in sorted(json_updates.items()):
        with open(json_path, "r") as f:
            records = json.load(f)

        changed = False
        for idx, fp in index_to_fp.items():
            if idx < len(records):
                old_fp = records[idx].get("fingerprint")
                if old_fp != fp:
                    records[idx]["fingerprint"] = fp
                    changed = True

        if changed:
            with open(json_path, "w") as f:
                json.dump(records, f, ensure_ascii=False, indent=4)
            updated_count += 1
            logger.info(f"  已更新: {json_path} ({len(index_to_fp)} 条记录)")

    return updated_count


def discover_template_meta_files(template_roots: Tuple[str, ...]) -> List[Path]:
    """递归发现所有 template_meta.json 文件。"""
    meta_paths: List[Path] = []
    seen: Set[Path] = set()
    for root_str in template_roots:
        root = Path(root_str)
        if not root.exists():
            logger.warning("模板根目录不存在，跳过: %s", root)
            continue
        for meta_path in sorted(root.rglob("template_meta.json")):
            if meta_path not in seen:
                seen.add(meta_path)
                meta_paths.append(meta_path)
    return meta_paths


def load_template_configs(
    template_roots: Tuple[str, ...],
) -> Dict[str, Dict[str, Any]]:
    """
    从外部模板库加载唯一物理模板配置（按 full_fingerprint 去重）。
    """
    template_configs: Dict[str, Dict[str, Any]] = {}
    meta_paths = discover_template_meta_files(template_roots)
    logger.info("发现 %d 个 template_meta.json", len(meta_paths))

    for meta_path in meta_paths:
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except Exception as exc:
            logger.warning("读取模板元数据失败，跳过 %s: %s", meta_path, exc)
            continue

        channel_names = meta.get("channel_names")
        channel_positions = meta.get("channel_positions_m")
        if not channel_names or channel_positions is None:
            logger.warning("模板缺少 channel_names / channel_positions_m，跳过: %s", meta_path)
            continue

        positions_np = np.asarray(channel_positions, dtype=np.float64)
        n_channels = int(meta.get("n_channels", -1))
        if positions_np.ndim != 2 or positions_np.shape != (n_channels, 3):
            logger.warning(
                "模板坐标形状无效，跳过: %s (shape=%s, n_channels=%s)",
                meta_path,
                positions_np.shape,
                n_channels,
            )
            continue

        full_fp = _compute_channel_hash(channel_names, positions_np)
        if full_fp in template_configs:
            template_configs[full_fp].setdefault("duplicate_meta_paths", []).append(str(meta_path))
            continue

        template_configs[full_fp] = {
            "dataset_name": str(meta.get("dataset_name", meta_path.parent.name)),
            "channel_names": list(channel_names),
            "channel_positions": positions_np,
            "meta_path": str(meta_path),
            "n_channels": n_channels,
        }

    logger.info("模板库唯一物理模板数: %d", len(template_configs))
    return template_configs


def copy_or_compute_leadfield(
    full_fp: str,
    existing_cache_path: Optional[str],
    channel_names: List[str],
    channel_positions: np.ndarray,
    subjects_dir: str,
    cache_dir: str,
) -> Optional[str]:
    """
    优先复制已有缓存；若不存在则重新计算。
    """
    dst_path = Path(cache_dir) / f"{full_fp}.pt"
    if dst_path.exists():
        logger.info("  目标缓存已存在: %s", dst_path)
        return str(dst_path)

    if existing_cache_path:
        src_path = Path(existing_cache_path)
        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            logger.info("  已复制已有缓存: %s -> %s", src_path, dst_path)
            return str(dst_path)
        logger.warning("  已有缓存路径不存在，改为重新计算: %s", src_path)

    return compute_leadfield_for_config(
        channel_names=channel_names,
        channel_positions=channel_positions,
        subjects_dir=subjects_dir,
        cache_dir=cache_dir,
    )


def build_registry_from_seed_archive(
    seed_archive_path: str,
    cache_dir: str,
    subjects_dir: str,
) -> Tuple[ElectrodeConfigRegistry, Set[str], int, int]:
    """
    从已有 archive 构建一个按 full_fingerprint 去重的新 registry。
    """
    logger.info("从 seed archive 构建仿真 registry: %s", seed_archive_path)
    archive = torch.load(seed_archive_path, map_location="cpu", weights_only=False)
    configs = archive.get("configs", {})

    registry = ElectrodeConfigRegistry(processed_data_dir="")
    seen_full_fps: Set[str] = set()
    imported_count = 0
    copied_count = 0

    for _, entry in sorted(configs.items()):
        full_fp = str(entry.get("full_fingerprint", ""))
        if not full_fp or full_fp in seen_full_fps:
            continue

        channel_names = entry.get("channel_names")
        channel_positions = entry.get("channel_positions_m")
        if not channel_names or channel_positions is None:
            logger.warning("seed archive 条目缺少通道配置，跳过 full_fp=%s", full_fp)
            continue

        positions_np = np.asarray(channel_positions, dtype=np.float64)
        registry.register_config(
            channel_names=list(channel_names),
            channel_positions=positions_np,
            dataset_name="seed_archive",
        )
        seen_full_fps.add(full_fp)
        imported_count += 1

        cache_path = copy_or_compute_leadfield(
            full_fp=full_fp,
            existing_cache_path=entry.get("leadfield_cache_path"),
            channel_names=list(channel_names),
            channel_positions=positions_np,
            subjects_dir=subjects_dir,
            cache_dir=cache_dir,
        )
        if cache_path is not None:
            copied_count += 1

    logger.info(
        "seed archive 导入完成: %d 个唯一物理模板, %d 个缓存已准备",
        imported_count,
        copied_count,
    )
    return registry, seen_full_fps, imported_count, copied_count


def merge_templates_into_registry(
    registry: ElectrodeConfigRegistry,
    existing_full_fps: Set[str],
    template_configs: Dict[str, Dict[str, Any]],
    subjects_dir: str,
    cache_dir: str,
) -> Tuple[int, int, int]:
    """
    将外部模板库合并进 registry 并离线计算缺失导联场。
    """
    success_count = 0
    skip_count = 0
    fail_count = 0

    for full_fp, entry in sorted(template_configs.items()):
        dataset_name = entry["dataset_name"]
        channel_names = entry["channel_names"]
        channel_positions = entry["channel_positions"]
        meta_path = entry["meta_path"]

        logger.info(
            "\n处理外部模板 full_fp=%s (%sch, dataset=%s)",
            full_fp,
            len(channel_names),
            dataset_name,
        )

        if full_fp in existing_full_fps:
            logger.info("  跳过：已存在于 registry 中")
            skip_count += 1
            continue

        registry.register_config(
            channel_names=list(channel_names),
            channel_positions=channel_positions,
            dataset_name=dataset_name,
        )
        existing_full_fps.add(full_fp)
        logger.info("  已注册模板: %s", meta_path)

        cache_path = copy_or_compute_leadfield(
            full_fp=full_fp,
            existing_cache_path=None,
            channel_names=list(channel_names),
            channel_positions=channel_positions,
            subjects_dir=subjects_dir,
            cache_dir=cache_dir,
        )
        if cache_path is not None:
            success_count += 1
        else:
            fail_count += 1

    logger.info(
        "外部模板处理完成: %d 成功, %d 跳过, %d 失败",
        success_count,
        skip_count,
        fail_count,
    )
    return success_count, skip_count, fail_count


# ============================================================================
# 主流程
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="离线预计算全局电极指纹注册表 + 导联场矩阵",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--penci_data",
        type=str,
        default="/work/2024/tanzunsheng/PENCIData",
        help="PENCIData 根目录（含 {dataset}-metadata/ 和 .pt 文件）",
    )
    parser.add_argument(
        "--processed_data",
        type=str,
        default="/work/2024/tanzunsheng/ProcessedData",
        help="ProcessedData 根目录（含 electrodes.tsv）",
    )
    parser.add_argument(
        "--subjects_dir",
        type=str,
        default="/work/2024/tanzunsheng/mne_data/MNE-fsaverage-data",
        help="FreeSurfer subjects 目录路径",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/work/2024/tanzunsheng/leadfield_cache",
        help="导联场缓存目录",
    )
    parser.add_argument(
        "--archive_out",
        type=str,
        default=None,
        help="输出 archive .pt 文件路径（默认: {cache_dir}/fingerprint_registry.pt）",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="并行计算指纹的进程数（默认 16）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅扫描，不计算导联场，不保存 archive",
    )
    parser.add_argument(
        "--update_metadata",
        action="store_true",
        help="回写 fingerprint 到 metadata JSON 文件",
    )
    parser.add_argument(
        "--seed_archive",
        type=str,
        default="",
        help="可选：已有真实数据 archive，作为底座导入到新的 cache/archive 中",
    )
    parser.add_argument(
        "--template_roots",
        type=str,
        default="",
        help="可选：逗号分隔的外部模板根目录；会递归搜索 template_meta.json 并补算导联场",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细日志",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.archive_out is None:
        args.archive_out = os.path.join(args.cache_dir, "fingerprint_registry.pt")

    template_roots = parse_path_list_arg(args.template_roots)

    # === 合并模式：已有真实 archive + 外部模板库 ===
    if args.seed_archive or template_roots:
        if not args.subjects_dir:
            raise RuntimeError("合并模式需要 --subjects_dir")

        logger.info("=" * 60)
        logger.info("合并模式：seed archive + 外部模板库")
        logger.info("=" * 60)

        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        registry = ElectrodeConfigRegistry(processed_data_dir="")
        existing_full_fps: Set[str] = set()
        imported_count = 0
        copied_count = 0

        if args.seed_archive:
            registry, existing_full_fps, imported_count, copied_count = (
                build_registry_from_seed_archive(
                    seed_archive_path=args.seed_archive,
                    cache_dir=args.cache_dir,
                    subjects_dir=args.subjects_dir,
                )
            )

        if template_roots:
            template_configs = load_template_configs(template_roots)
            merge_templates_into_registry(
                registry=registry,
                existing_full_fps=existing_full_fps,
                template_configs=template_configs,
                subjects_dir=args.subjects_dir,
                cache_dir=args.cache_dir,
            )
        else:
            logger.info("未提供 template_roots，仅导入 seed archive")

        save_archive(registry, args.archive_out, cache_dir=args.cache_dir)
        logger.info(
            "合并模式完成: %d 个唯一物理模板, cache_dir=%s, archive=%s",
            len(existing_full_fps),
            args.cache_dir,
            args.archive_out,
        )
        return

    t_start = time.time()

    # === 第一步：扫描 + 计算 pos_fingerprint ===
    logger.info("=" * 60)
    logger.info("第一步：扫描 metadata + 计算 pos_fingerprint")
    logger.info("=" * 60)

    unique_fps, path_to_fp, all_metadata = scan_and_fingerprint(
        args.penci_data,
        max_workers=args.max_workers,
    )

    # 打印摘要
    logger.info("\n唯一指纹摘要:")
    for fp, info in sorted(unique_fps.items()):
        ds_str = ", ".join(sorted(info["datasets"]))
        logger.info(f"  {fp}: {info['channels']}ch, 数据集=[{ds_str}]")

    if args.dry_run:
        logger.info("\n[dry_run] 仅扫描模式，跳过导联场计算和 archive 保存")
        elapsed = time.time() - t_start
        logger.info(f"\n完成（耗时 {elapsed:.1f}s）")
        return

    # === 第二步：查找 electrodes.tsv + 计算导联场 ===
    logger.info("\n" + "=" * 60)
    logger.info("第二步：查找 electrodes.tsv + 计算导联场")
    logger.info("=" * 60)

    registry = ElectrodeConfigRegistry(args.processed_data)
    success_count = 0
    skip_count = 0
    fail_count = 0

    for fp, info in sorted(unique_fps.items()):
        logger.info(f"\n处理指纹 {fp} ({info['channels']}ch)...")

        # 查找米制坐标
        tsv_result = find_tsv_for_fingerprint(info, args.processed_data)
        if tsv_result is None:
            logger.warning(f"  跳过: 找不到 electrodes.tsv")
            fail_count += 1
            continue

        channel_names, channel_positions = tsv_result
        logger.info(f"  TSV: {len(channel_names)} 个有效通道")

        # 注册到 registry
        registered_fp = registry.register_config(
            channel_names, channel_positions,
        )
        logger.info(f"  注册: pos_fp={registered_fp}")

        # 检查注册后的 pos_fp 是否与 .pt 文件的 fp 匹配
        # 注意：由于坐标空间不同（米制 vs normalized），这里 registered_fp != fp 是预期行为
        # registry 中同时存储了两种映射，在线阶段需要用 .pt 的 fp 来查询
        if registered_fp != fp:
            logger.info(
                f"  注意: TSV 指纹 ({registered_fp}) != .pt 指纹 ({fp})"
                f"（坐标空间不同，这是预期行为）"
            )
            # 额外注册 .pt 的 pos_fingerprint 指向同一组通道配置
            # 这是 Scheme B 的核心：运行时用 .pt 指纹查询，所以必须注册 .pt 指纹
            registry._fingerprint_configs[fp] = (channel_names, channel_positions)
            full_fp = _compute_channel_hash(channel_names, channel_positions)
            registry._pos_to_full_fingerprint[fp] = full_fp
            logger.info(f"  补充注册: .pt 指纹 {fp} → 相同通道配置")

        # 计算导联场
        cache_path = compute_leadfield_for_config(
            channel_names, channel_positions,
            args.subjects_dir, args.cache_dir,
        )

        if cache_path is not None:
            success_count += 1
        else:
            fail_count += 1

    logger.info(
        f"\n导联场计算完成: {success_count} 成功, "
        f"{skip_count} 跳过, {fail_count} 失败"
    )

    # === 第三步：保存 archive ===
    logger.info("\n" + "=" * 60)
    logger.info("第三步：保存 archive")
    logger.info("=" * 60)

    save_archive(registry, args.archive_out, cache_dir=args.cache_dir)

    # === 第四步（可选）：回写 metadata ===
    if args.update_metadata:
        logger.info("\n" + "=" * 60)
        logger.info("第四步：回写 fingerprint 到 metadata JSON")
        logger.info("=" * 60)

        updated = update_metadata_fingerprints(all_metadata, path_to_fp)
        logger.info(f"已更新 {updated} 个 JSON 文件")

    elapsed = time.time() - t_start
    logger.info(f"\n全部完成（耗时 {elapsed:.1f}s）")
    logger.info(f"Archive: {args.archive_out}")
    logger.info(
        f"注册表: {len(registry.get_all_fingerprints())} 个指纹, "
        f"{len(registry.registered_configs)} 个配置"
    )


if __name__ == "__main__":
    main()
