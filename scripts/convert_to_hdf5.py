# -*- coding: utf-8 -*-
"""
通用 .pt → HDF5 格式转换脚本（支持所有 PENCI 数据集）

背景：
    PENCI 共使用 10 个数据集（含 Broderick2018 的 4 个子集），合计约 170 万个
    独立 .pt 文件。随机访问 NFS 时每次都触发 open/stat/read/close 元数据操作，
    造成 iowait≈14%、SSH 卡顿、训练速度仅 ~170 samples/s。
    HBN_EEG（已完成）转换后其余 9 个数据集（~148K 样本）仍以 .pt 形式存在。

方案：
    按受试者将所有 .pt 文件合并为单个 HDF5 文件；DataLoader worker 首次访问时
    打开 HDF5（持久句柄），后续读取仅需一次 chunk 读，无 open/stat/close 开销。

支持的数据集：
    HBN_EEG（默认跳过，已完成）、THINGS-EEG、ThingsEEG、SEED-DV、
    Brennan_Hale2019、Grootswagers2019、Broderick2018_NaturalSpeech、
    Broderick2018_NaturalSpeechReverse、Broderick2018_SpeechInNoise、
    Broderick2018_CocktailParty

HDF5 内部结构（每个受试者文件）：
    /x            (N, C, T) int16   ← bfloat16 原始 bits（view 转换）
    /pos          (C, 6)    int16   ← bfloat16 原始 bits
    /sensor_type  (C,)      int32

metadata 更新（写入各数据集的 train.json / val.json）：
    每条记录新增：
        "hdf5_path": "DATASET-hdf5/sub-XXX.h5"  （相对于 data_root）
        "hdf5_idx":  22                           （在 h5 文件内的行号）

用法：
    # 转换除 HBN_EEG 以外的所有数据集（默认）
    python scripts/convert_to_hdf5.py

    # 转换指定数据集
    python scripts/convert_to_hdf5.py --datasets THINGS-EEG,ThingsEEG,SEED-DV

    # 转换所有（含 HBN_EEG，使用 skip_existing 跳过已完成部分）
    python scripts/convert_to_hdf5.py --datasets all

    # 指定数据根目录和并行数
    python scripts/convert_to_hdf5.py \\
        --data_root /work/2024/tanzunsheng/PENCIData \\
        --workers 8
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── 数据集配置表 ─────────────────────────────────────────────────────────────
# metadata_dir / hdf5_subdir 均相对于 data_root
DEFAULT_DATA_ROOT = "/work/2024/tanzunsheng/PENCIData"

DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "HBN_EEG": {
        "metadata_dir": "HBN_EEG-metadata",
        "hdf5_subdir":  "HBN_EEG-hdf5",
    },
    "THINGS-EEG": {
        "metadata_dir": "THINGS-EEG-metadata",
        "hdf5_subdir":  "THINGS-EEG-hdf5",
    },
    "ThingsEEG": {
        "metadata_dir": "ThingsEEG-metadata",
        "hdf5_subdir":  "ThingsEEG-hdf5",
    },
    "SEED-DV": {
        "metadata_dir": "SEED-DV-metadata",
        "hdf5_subdir":  "SEED-DV-hdf5",
    },
    "Brennan_Hale2019": {
        "metadata_dir": "Brennan_Hale2019-metadata",
        "hdf5_subdir":  "Brennan_Hale2019-hdf5",
    },
    "Grootswagers2019": {
        "metadata_dir": "Grootswagers2019-metadata",
        "hdf5_subdir":  "Grootswagers2019-hdf5",
    },
    # Broderick2018 的 4 个子数据集，metadata 和 hdf5 都放在 Broderick2018/ 子目录下
    "Broderick2018_NaturalSpeech": {
        "metadata_dir": "Broderick2018/Broderick2018_NaturalSpeech-metadata",
        "hdf5_subdir":  "Broderick2018/Broderick2018_NaturalSpeech-hdf5",
    },
    "Broderick2018_NaturalSpeechReverse": {
        "metadata_dir": "Broderick2018/Broderick2018_NaturalSpeechReverse-metadata",
        "hdf5_subdir":  "Broderick2018/Broderick2018_NaturalSpeechReverse-hdf5",
    },
    "Broderick2018_SpeechInNoise": {
        "metadata_dir": "Broderick2018/Broderick2018_SpeechInNoise-metadata",
        "hdf5_subdir":  "Broderick2018/Broderick2018_SpeechInNoise-hdf5",
    },
    "Broderick2018_CocktailParty": {
        "metadata_dir": "Broderick2018/Broderick2018_CocktailParty-metadata",
        "hdf5_subdir":  "Broderick2018/Broderick2018_CocktailParty-hdf5",
    },
}

# 默认转换列表（不含已完成的 HBN_EEG）
DEFAULT_DATASETS = [k for k in DATASET_CONFIGS if k != "HBN_EEG"]


# ─── 受试者级别转换（在子进程中执行）────────────────────────────────────────────

def _convert_one_subject(args: Tuple) -> Dict:
    """
    将一个受试者的所有 .pt 文件转为单个 HDF5 文件。

    在子进程中运行，不依赖任何父进程状态。

    参数:
        args: (subject_id, sample_list, hdf5_dir, hdf5_subdir, data_root, skip_existing)
              hdf5_dir:    HDF5 绝对输出目录
              hdf5_subdir: 相对于 data_root 的子目录名，用于生成 hdf5_path_rel

    返回:
        {
            "subject_id": str,
            "ok": bool,
            "n_ok": int,
            "n_fail": int,
            "hdf5_path": str,          # 相对于 data_root 的路径
            "idx_map": {orig_path: hdf5_idx, ...}  # 原始路径 → HDF5 行号
        }
    """
    try:
        import h5py
        import torch
    except ImportError as e:
        return {"subject_id": args[0], "ok": False, "n_ok": 0, "n_fail": 0,
                "hdf5_path": "", "idx_map": {}, "error": str(e)}

    subject_id, sample_list, hdf5_dir, hdf5_subdir, data_root, skip_existing = args
    hdf5_path_abs = Path(hdf5_dir) / f"{subject_id}.h5"
    hdf5_path_rel = str(Path(hdf5_subdir) / f"{subject_id}.h5")

    # 断点续转：若 HDF5 已存在且样本数匹配则跳过
    if skip_existing and hdf5_path_abs.exists():
        try:
            with h5py.File(str(hdf5_path_abs), "r") as h5:
                existing_n = h5["x"].shape[0]
            if existing_n == len(sample_list):
                idx_map = {meta["path"]: i for i, meta in enumerate(
                    sorted(sample_list, key=lambda m: m["path"])
                )}
                return {
                    "subject_id": subject_id, "ok": True,
                    "n_ok": existing_n, "n_fail": 0,
                    "hdf5_path": hdf5_path_rel, "idx_map": idx_map,
                }
        except Exception:
            pass  # HDF5 损坏，重新转换

    # ── 按路径排序，保证 idx 可复现 ──────────────────────────────────────────
    sample_list_sorted = sorted(sample_list, key=lambda m: m["path"])

    x_list: List[np.ndarray] = []
    idx_map: Dict[str, int] = {}
    pos_ref: Optional[np.ndarray] = None
    sensor_type_ref: Optional[np.ndarray] = None
    sample_shape: Optional[Tuple[int, int]] = None  # (C, T) 从第一个成功样本获取
    n_fail = 0

    for local_idx, meta in enumerate(sample_list_sorted):
        orig_path = meta["path"]
        # 支持 data_root 路径覆盖
        if data_root is not None:
            try:
                rel = os.path.relpath(orig_path, DEFAULT_DATA_ROOT)
                full_path = os.path.join(data_root, rel)
            except ValueError:
                full_path = orig_path
        else:
            full_path = orig_path

        try:
            data = torch.load(full_path, weights_only=True)
            # bfloat16 → int16 原始 bits（HDF5 不原生支持 bfloat16）
            x_i = data["x"].view(torch.int16).numpy()   # (C, T) int16
            x_list.append(x_i)
            idx_map[orig_path] = local_idx

            if sample_shape is None:
                sample_shape = x_i.shape  # (C, T)
            if pos_ref is None:
                pos_ref = data["pos"].view(torch.int16).numpy()    # (C, 6) int16
                sensor_type_ref = data["sensor_type"].numpy()       # (C,) int32
        except Exception:
            n_fail += 1
            # 插入零占位，保持 idx 连续性
            if sample_shape is not None:
                C, T = sample_shape
            else:
                # 还没有成功样本，用 metadata 的 channels 字段推断
                C = int(meta.get("channels", 128))
                T = 2560
            dummy_x = np.zeros((C, T), dtype=np.int16)
            x_list.append(dummy_x)
            if pos_ref is None:
                pos_ref = np.zeros((C, 6), dtype=np.int16)
                sensor_type_ref = np.zeros(C, dtype=np.int32)

    if not x_list:
        return {"subject_id": subject_id, "ok": False, "n_ok": 0, "n_fail": n_fail,
                "hdf5_path": "", "idx_map": {}, "error": "所有样本加载失败"}

    # 若所有样本 shape 一致则直接 stack；否则 fallback 到同 shape 过滤
    try:
        x_array = np.stack(x_list, axis=0)  # (N, C, T) int16
    except ValueError:
        # 极端情况：同一受试者内不同样本 shape 不一致，取众数 shape
        from collections import Counter
        shape_counter = Counter(a.shape for a in x_list)
        dominant_shape = shape_counter.most_common(1)[0][0]
        x_array = np.stack(
            [a if a.shape == dominant_shape else np.zeros(dominant_shape, dtype=np.int16)
             for a in x_list],
            axis=0,
        )

    # ── 写 HDF5 ────────────────────────────────────────────────────────────
    hdf5_path_abs.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(hdf5_path_abs) + ".tmp"

    try:
        with h5py.File(tmp_path, "w") as h5:
            # x: chunk=(1, C, T) 保证单样本随机读取恰好 1 个 chunk
            h5.create_dataset(
                "x",
                data=x_array,
                dtype="int16",
                chunks=(1, x_array.shape[1], x_array.shape[2]),
                # 不压缩：最大化读取速度（压缩会增加 CPU 开销）
            )
            h5.create_dataset("pos", data=pos_ref, dtype="int16")
            h5.create_dataset("sensor_type", data=sensor_type_ref, dtype="int32")
            h5.attrs["subject_id"] = subject_id
            h5.attrs["n_samples"] = len(x_list)
            h5.attrs["n_failed"] = n_fail

        # 原子性改名（避免写入中途崩溃留下损坏文件）
        os.rename(tmp_path, str(hdf5_path_abs))
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return {"subject_id": subject_id, "ok": False, "n_ok": 0, "n_fail": n_fail,
                "hdf5_path": "", "idx_map": {}, "error": str(e)}

    return {
        "subject_id": subject_id,
        "ok": True,
        "n_ok": len(x_list) - n_fail,
        "n_fail": n_fail,
        "hdf5_path": hdf5_path_rel,
        "idx_map": idx_map,
    }


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def group_by_subject(metadata: List[Dict]) -> Dict[str, List[Dict]]:
    """
    按受试者 ID 分组 metadata 记录。

    所有数据集路径均含 /preprocessing/ 段，格式：
        .../preprocessing/sub-XXXXX/...
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for meta in metadata:
        path = meta["path"]
        try:
            subject_id = path.split("/preprocessing/")[1].split("/")[0]
        except IndexError:
            subject_id = "unknown"
        groups[subject_id].append(meta)
    return dict(groups)


def update_metadata_with_hdf5(
    metadata: List[Dict],
    dataset_name: str,
    idx_map_by_subject: Dict[str, Dict[str, int]],
    hdf5_path_by_subject: Dict[str, str],
) -> Tuple[List[Dict], int]:
    """
    为 metadata 中属于 dataset_name 的记录写入 hdf5_path 和 hdf5_idx。

    返回:
        (updated_metadata, n_updated)
    """
    n_updated = 0
    for meta in metadata:
        if meta.get("dataset") != dataset_name:
            continue
        path = meta["path"]
        try:
            subject_id = path.split("/preprocessing/")[1].split("/")[0]
        except IndexError:
            continue

        if subject_id not in idx_map_by_subject:
            continue
        idx_map = idx_map_by_subject[subject_id]
        if path not in idx_map:
            continue

        meta["hdf5_path"] = hdf5_path_by_subject[subject_id]
        meta["hdf5_idx"] = idx_map[path]
        n_updated += 1

    return metadata, n_updated


# ─── 单数据集转换 ─────────────────────────────────────────────────────────────

def convert_dataset(
    dataset_name: str,
    data_root: str,
    n_workers: int,
    skip_existing: bool,
    dry_run: bool,
) -> bool:
    """
    转换单个数据集的所有 .pt 文件到 HDF5。

    返回:
        True 表示成功，False 表示跳过或失败。
    """
    cfg = DATASET_CONFIGS[dataset_name]
    metadata_dir = os.path.join(data_root, cfg["metadata_dir"])
    hdf5_subdir = cfg["hdf5_subdir"]
    hdf5_dir = os.path.join(data_root, hdf5_subdir)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"数据集: {dataset_name}")
    logger.info(f"  metadata 目录: {metadata_dir}")
    logger.info(f"  HDF5 输出目录: {hdf5_dir}")
    logger.info("=" * 60)

    # ── 加载 metadata ────────────────────────────────────────────────────────
    splits = {}
    for split in ("train", "val"):
        json_path = os.path.join(metadata_dir, f"{split}.json")
        if not os.path.exists(json_path):
            logger.warning(f"  找不到 {json_path}，跳过 {split}")
            continue
        with open(json_path, "r") as f:
            splits[split] = json.load(f)
        logger.info(f"  {split}.json: {len(splits[split]):,} 条记录")

    if not splits:
        logger.error(f"  {dataset_name}: 找不到任何 metadata JSON，跳过")
        return False

    # ── 按受试者分组 ─────────────────────────────────────────────────────────
    all_metadata = []
    for records in splits.values():
        all_metadata.extend(records)

    subject_groups = group_by_subject(all_metadata)
    n_subjects = len(subject_groups)
    logger.info(f"  受试者数量: {n_subjects}")
    logger.info(f"  总样本数:   {len(all_metadata):,}")

    if dry_run:
        logger.info("  [dry_run] 不执行实际转换")
        return True

    # ── 并行转换 ─────────────────────────────────────────────────────────────
    task_args = [
        (sid, samples, hdf5_dir, hdf5_subdir, data_root, skip_existing)
        for sid, samples in subject_groups.items()
    ]

    idx_map_by_subject: Dict[str, Dict[str, int]] = {}
    hdf5_path_by_subject: Dict[str, str] = {}
    n_ok_total = 0
    n_fail_total = 0
    n_subjects_ok = 0

    t_start = time.time()
    logger.info(f"  开始转换，使用 {n_workers} 个并行进程 ...")

    with mp.Pool(processes=n_workers) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_convert_one_subject, task_args), start=1
        ):
            sid = result["subject_id"]
            if result["ok"]:
                idx_map_by_subject[sid] = result["idx_map"]
                hdf5_path_by_subject[sid] = result["hdf5_path"]
                n_ok_total += result["n_ok"]
                n_fail_total += result["n_fail"]
                n_subjects_ok += 1
            else:
                err = result.get("error", "未知错误")
                logger.warning(f"    受试者 {sid} 转换失败: {err}")
                n_fail_total += result.get("n_fail", 0)

            # 每 20 个受试者打印一次进度（小数据集受试者少，缩短间隔）
            log_interval = max(1, min(20, n_subjects // 5))
            if i % log_interval == 0 or i == n_subjects:
                elapsed = time.time() - t_start
                speed = n_ok_total / elapsed if elapsed > 0 else 0
                eta = (n_subjects - i) / (i / elapsed) if elapsed > 0 and i > 0 else 0
                logger.info(
                    f"    进度: {i}/{n_subjects} 受试者 | "
                    f"样本 {n_ok_total:,} ok / {n_fail_total} 失败 | "
                    f"{speed:.0f} 样本/s | "
                    f"ETA {eta/60:.1f} min"
                )

    elapsed = time.time() - t_start
    logger.info(
        f"  转换完成: {n_subjects_ok}/{n_subjects} 受试者成功，"
        f"{n_ok_total:,} 样本，{n_fail_total} 失败，耗时 {elapsed/60:.1f} min"
    )

    # ── 更新 metadata JSON ───────────────────────────────────────────────────
    logger.info("  更新 metadata JSON ...")
    for split, records in splits.items():
        updated, n_upd = update_metadata_with_hdf5(
            records, dataset_name, idx_map_by_subject, hdf5_path_by_subject
        )
        json_path = os.path.join(metadata_dir, f"{split}.json")
        tmp_path = json_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(updated, f, ensure_ascii=False, separators=(",", ":"))
        os.rename(tmp_path, json_path)
        logger.info(f"    {split}.json: 已更新 {n_upd:,}/{len(records):,} 条记录")

    return True


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="将 PENCI 各数据集 .pt 文件转换为 HDF5 格式（按受试者分组）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help=(
            "要转换的数据集，逗号分隔；"
            "用 'all' 表示全部（含已完成的 HBN_EEG）。"
            f"可选: {', '.join(DATASET_CONFIGS.keys())}"
        ),
    )
    parser.add_argument(
        "--data_root",
        default=DEFAULT_DATA_ROOT,
        help="PENCIData 根目录",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行进程数",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="跳过已存在且样本数匹配的 HDF5（断点续转，默认开启）",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="强制重新转换所有受试者",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅打印计划，不实际转换",
    )
    args = parser.parse_args()

    if args.no_skip_existing:
        args.skip_existing = False

    # ── 解析目标数据集列表 ────────────────────────────────────────────────────
    if args.datasets.strip().lower() == "all":
        target_datasets = list(DATASET_CONFIGS.keys())
    else:
        target_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    unknown = [d for d in target_datasets if d not in DATASET_CONFIGS]
    if unknown:
        logger.error(f"未知数据集: {unknown}")
        logger.error(f"可选: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("PENCI .pt → HDF5 通用转换工具")
    logger.info(f"  数据根目录  : {args.data_root}")
    logger.info(f"  目标数据集  : {target_datasets}")
    logger.info(f"  并行进程数  : {args.workers}")
    logger.info(f"  断点续转    : {args.skip_existing}")
    logger.info(f"  dry_run     : {args.dry_run}")
    logger.info("=" * 60)

    t_global = time.time()
    n_success = 0

    for ds in target_datasets:
        ok = convert_dataset(
            dataset_name=ds,
            data_root=args.data_root,
            n_workers=args.workers,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
        if ok:
            n_success += 1

    elapsed = time.time() - t_global
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"全部完成！{n_success}/{len(target_datasets)} 个数据集转换成功")
    logger.info(f"总耗时: {elapsed/60:.1f} min")
    logger.info("下一步: 重启训练即可自动使用 HDF5（dataset.py 已支持条件分支）")
    logger.info("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # 避免 fork+torch 的已知问题
    main()
