# -*- coding: utf-8 -*-
"""
HBN_EEG .pt → HDF5 格式转换脚本

背景：
    HBN_EEG 数据集包含 1,568,306 个独立 .pt 文件，随机访问时每次都触发
    NFS 的 open/stat/read/close 元数据操作，造成 iowait≈14%、SSH卡顿、
    训练速度仅 ~170 samples/s（期望 >1000 samples/s）。

方案：
    按受试者将所有 .pt 文件合并为一个 HDF5 文件（3461 个 sub-XXX.h5）。
    DataLoader worker 在第一次访问时打开 HDF5（持久句柄），后续读取
    仅需一次顺序 chunk 读，无 open/stat/close 开销。

输出目录结构：
    {hdf5_dir}/sub-NDARKX701BJ4.h5
    {hdf5_dir}/sub-NDARZD415ZZ1.h5
    ...（共 3461 个）

HDF5 内部结构（每个文件）：
    /x            (N, 128, 2560) int16   ← bfloat16 原始bits（view转换）
    /pos          (128, 6)       int16   ← bfloat16 原始bits
    /sensor_type  (128,)         int32

metadata 更新（写入 train.json / val.json）：
    每条 HBN_EEG 记录新增：
        "hdf5_path": "HBN_EEG-hdf5/sub-XXX.h5"  （相对于 data_root）
        "hdf5_idx":  22                            （在 h5 文件内的行号）

用法：
    # 默认转换（并行 8 进程），输出到 PENCIData/HBN_EEG-hdf5/
    python scripts/convert_hbn_to_hdf5.py

    # 指定路径和并行数
    python scripts/convert_hbn_to_hdf5.py \\
        --data_root /work/2024/tanzunsheng/PENCIData \\
        --hdf5_dir  /work/2024/tanzunsheng/PENCIData/HBN_EEG-hdf5 \\
        --workers 8

    # 仅更新 metadata，跳过已存在的 HDF5（断点续转）
    python scripts/convert_hbn_to_hdf5.py --skip_existing

预计时间：
    - 读+写 960 GB @ NFS ~200 MB/s ≈ 2.5~4 小时（取决于并行度和 NFS 负载）
    - 建议在训练间隙或夜间运行
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
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── 常量 ─────────────────────────────────────────────────────────────────────
DEFAULT_DATA_ROOT = "/work/2024/tanzunsheng/PENCIData"
HBN_METADATA_SUBDIR = "HBN_EEG-metadata"
HDF5_SUBDIR = "HBN_EEG-hdf5"  # 相对于 data_root


# ─── 受试者级别转换（在子进程中执行）────────────────────────────────────────────

def _convert_one_subject(args: Tuple) -> Dict:
    """
    将一个受试者的所有 .pt 文件转为单个 HDF5 文件。

    在子进程中运行，不依赖任何父进程状态。

    参数:
        args: (subject_id, sample_list, hdf5_dir, data_root, skip_existing)
              sample_list: [{"path":..., "meta_idx_train":..., ...}, ...]

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

    subject_id, sample_list, hdf5_dir, data_root, skip_existing = args
    hdf5_path_abs = Path(hdf5_dir) / f"{subject_id}.h5"
    hdf5_path_rel = str(Path(HDF5_SUBDIR) / f"{subject_id}.h5")

    # 断点续转：若 HDF5 已存在且样本数匹配则跳过
    if skip_existing and hdf5_path_abs.exists():
        try:
            with h5py.File(str(hdf5_path_abs), "r") as h5:
                existing_n = h5["x"].shape[0]
            if existing_n == len(sample_list):
                # 重建 idx_map
                idx_map = {meta["path"]: i for i, meta in enumerate(sample_list)}
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
    pos_ref: np.ndarray = None
    sensor_type_ref: np.ndarray = None
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
            x_i = data["x"].view(torch.int16).numpy()   # (128, 2560) int16
            x_list.append(x_i)
            idx_map[orig_path] = local_idx

            if pos_ref is None:
                pos_ref = data["pos"].view(torch.int16).numpy()       # (128, 6) int16
                sensor_type_ref = data["sensor_type"].numpy()          # (128,) int32
        except Exception as e:
            n_fail += 1
            # 插入零占位，保持 idx 连续性
            dummy_x = np.zeros((128, 2560), dtype=np.int16)
            x_list.append(dummy_x)
            if pos_ref is None:
                pos_ref = np.zeros((128, 6), dtype=np.int16)
                sensor_type_ref = np.zeros(128, dtype=np.int32)

    if not x_list:
        return {"subject_id": subject_id, "ok": False, "n_ok": 0, "n_fail": n_fail,
                "hdf5_path": "", "idx_map": {}, "error": "所有样本加载失败"}

    x_array = np.stack(x_list, axis=0)  # (N, 128, 2560) int16

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


# ─── 主流程 ──────────────────────────────────────────────────────────────────

def group_by_subject(metadata: List[Dict]) -> Dict[str, List[Dict]]:
    """按受试者 ID 分组 metadata 记录。"""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for meta in metadata:
        path = meta["path"]
        # 路径格式: .../preprocessing/sub-XXXXX/eeg/...
        try:
            subject_id = path.split("/preprocessing/")[1].split("/")[0]
        except IndexError:
            subject_id = "unknown"
        groups[subject_id].append(meta)
    return dict(groups)


def update_metadata_with_hdf5(
    metadata: List[Dict],
    idx_map_by_subject: Dict[str, Dict[str, int]],
    hdf5_path_by_subject: Dict[str, str],
) -> Tuple[List[Dict], int]:
    """
    为 metadata 列表中的每条 HBN_EEG 记录写入 hdf5_path 和 hdf5_idx。

    返回:
        (updated_metadata, n_updated)
    """
    n_updated = 0
    for meta in metadata:
        if meta.get("dataset") != "HBN_EEG":
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


def main():
    parser = argparse.ArgumentParser(
        description="将 HBN_EEG .pt 文件转换为 HDF5 格式（按受试者分组）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_root",
        default=DEFAULT_DATA_ROOT,
        help="PENCIData 根目录",
    )
    parser.add_argument(
        "--hdf5_dir",
        default=None,
        help="HDF5 输出目录（默认: {data_root}/HBN_EEG-hdf5）",
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

    data_root = args.data_root
    hdf5_dir = args.hdf5_dir or os.path.join(data_root, HDF5_SUBDIR)
    metadata_dir = os.path.join(data_root, HBN_METADATA_SUBDIR)

    logger.info("=" * 60)
    logger.info("HBN_EEG .pt → HDF5 转换")
    logger.info(f"  数据根目录   : {data_root}")
    logger.info(f"  HDF5 输出目录: {hdf5_dir}")
    logger.info(f"  metadata 目录: {metadata_dir}")
    logger.info(f"  并行进程数   : {args.workers}")
    logger.info(f"  断点续转     : {args.skip_existing}")
    logger.info("=" * 60)

    # ── 加载 metadata ───────────────────────────────────────────────────────
    splits = {}
    for split in ("train", "val"):
        json_path = os.path.join(metadata_dir, f"{split}.json")
        if not os.path.exists(json_path):
            logger.warning(f"找不到 {json_path}，跳过")
            continue
        logger.info(f"加载 {split}.json ...")
        with open(json_path, "r") as f:
            splits[split] = json.load(f)
        logger.info(f"  {split}: {len(splits[split]):,} 条记录")

    if not splits:
        logger.error("找不到任何 metadata JSON，退出")
        sys.exit(1)

    # ── 按受试者分组（合并 train+val，转换时一起处理）──────────────────────
    all_metadata = []
    for records in splits.values():
        all_metadata.extend(records)

    subject_groups = group_by_subject(all_metadata)
    n_subjects = len(subject_groups)
    logger.info(f"共 {n_subjects} 个唯一受试者")
    logger.info(f"预计 HDF5 总大小: {n_subjects * 284 / 1024:.1f} GB（约估）")

    if args.dry_run:
        logger.info("[dry_run] 不执行实际转换，退出")
        return

    # ── 并行转换 ────────────────────────────────────────────────────────────
    task_args = [
        (sid, samples, hdf5_dir, data_root, args.skip_existing)
        for sid, samples in subject_groups.items()
    ]

    idx_map_by_subject: Dict[str, Dict[str, int]] = {}
    hdf5_path_by_subject: Dict[str, str] = {}
    n_ok_total = 0
    n_fail_total = 0
    n_subjects_ok = 0

    t_start = time.time()
    logger.info(f"开始转换，使用 {args.workers} 个并行进程 ...")

    with mp.Pool(processes=args.workers) as pool:
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
                logger.warning(f"  受试者 {sid} 转换失败: {err}")
                n_fail_total += result["n_fail"]

            # 每 50 个受试者打印一次进度
            if i % 50 == 0 or i == n_subjects:
                elapsed = time.time() - t_start
                speed = n_ok_total / elapsed if elapsed > 0 else 0
                eta = (n_subjects - i) / (i / elapsed) if elapsed > 0 else 0
                logger.info(
                    f"  进度: {i}/{n_subjects} 受试者 | "
                    f"样本 {n_ok_total:,} ok / {n_fail_total} 失败 | "
                    f"{speed:.0f} 样本/s | "
                    f"ETA {eta/60:.1f} min"
                )

    elapsed = time.time() - t_start
    logger.info(
        f"转换完成: {n_subjects_ok}/{n_subjects} 受试者成功，"
        f"{n_ok_total:,} 样本，{n_fail_total} 失败，耗时 {elapsed/60:.1f} min"
    )

    # ── 更新 metadata JSON ──────────────────────────────────────────────────
    logger.info("更新 metadata JSON ...")
    for split, records in splits.items():
        updated, n_upd = update_metadata_with_hdf5(
            records, idx_map_by_subject, hdf5_path_by_subject
        )
        json_path = os.path.join(metadata_dir, f"{split}.json")
        tmp_path = json_path + ".tmp"

        with open(tmp_path, "w") as f:
            json.dump(updated, f, ensure_ascii=False, separators=(",", ":"))
        os.rename(tmp_path, json_path)

        logger.info(f"  {split}.json: 已更新 {n_upd:,}/{len(records):,} 条 HBN_EEG 记录")

    logger.info("=" * 60)
    logger.info("全部完成！")
    logger.info(f"  HDF5 目录     : {hdf5_dir}")
    logger.info(f"  metadata 已更新: {list(splits.keys())}")
    logger.info(
        "  下一步: 重启训练即可自动使用 HDF5（无需其他配置）"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # 避免 fork+torch 的已知问题
    main()
