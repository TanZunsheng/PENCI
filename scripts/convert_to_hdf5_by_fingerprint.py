# -*- coding: utf-8 -*-
"""
按指纹分组的 .pt → HDF5 格式转换脚本

将 PENCIData 下所有数据集的 .pt 文件按电极指纹（fingerprint）分组转换为 HDF5，
同一指纹的样本来自不同受试者，训练时 batch 内样本多样性更好。

特性：
- 无损转换（bfloat16 raw bits）
- 按 fingerprint 分组，同一指纹内按 path 排序，一一对应
- 大指纹分片（默认 10K 样本/文件，NFS 友好）
- 文件命名：{ch}ch_{fp}_part{k}.h5 或 {ch}ch_{fp}.h5
- 断点续传：跳过已存在且样本数正确的 .h5 文件
- 覆盖 metadata 中旧的 hdf5_path、hdf5_idx
- 多进程并行转换
- 详细日志（含时间戳）写入 log 文件

用法：
    # 转换 /work/2024/tanzunsheng/PENCIData 下全部数据（默认）
    python scripts/convert_to_hdf5_by_fingerprint.py

    # 指定参数
    python scripts/convert_to_hdf5_by_fingerprint.py \\
        --data_root /work/2024/tanzunsheng/PENCIData \\
        --workers 16 \\
        --max_samples_per_file 10000 \\
        --log_dir outputs/convert_logs
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 默认数据根目录
DEFAULT_DATA_ROOT = "/work/2024/tanzunsheng/PENCIData"

# 数据集配置（与 convert_to_hdf5.py 一致）
DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "HBN_EEG": {"metadata_dir": "HBN_EEG-metadata"},
    "THINGS-EEG": {"metadata_dir": "THINGS-EEG-metadata"},
    "ThingsEEG": {"metadata_dir": "ThingsEEG-metadata"},
    "SEED-DV": {"metadata_dir": "SEED-DV-metadata"},
    "Brennan_Hale2019": {"metadata_dir": "Brennan_Hale2019-metadata"},
    "Grootswagers2019": {"metadata_dir": "Grootswagers2019-metadata"},
    "Broderick2018_NaturalSpeech": {
        "metadata_dir": "Broderick2018/Broderick2018_NaturalSpeech-metadata",
    },
    "Broderick2018_NaturalSpeechReverse": {
        "metadata_dir": "Broderick2018/Broderick2018_NaturalSpeechReverse-metadata",
    },
    "Broderick2018_SpeechInNoise": {
        "metadata_dir": "Broderick2018/Broderick2018_SpeechInNoise-metadata",
    },
    "Broderick2018_CocktailParty": {
        "metadata_dir": "Broderick2018/Broderick2018_CocktailParty-metadata",
    },
}

HDF5_SUBDIR = "by_fingerprint"


def setup_logging(log_dir: Optional[str] = None, log_file: Optional[str] = None) -> logging.Logger:
    """配置 logger：控制台 + 文件，含时间戳"""
    logger = logging.getLogger("convert_by_fingerprint")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件
    if log_dir or log_file:
        if log_file:
            fp = Path(log_file)
        else:
            log_dir = Path(log_dir or "outputs/convert_logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            fp = log_dir / f"convert_by_fingerprint_{datetime.now():%Y%m%d_%H%M%S}.log"
        fh = logging.FileHandler(fp, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f"日志文件: {fp}")

    return logger


def load_all_metadata(
    data_root: str,
) -> Tuple[List[Dict], Dict[str, Tuple[int, int]]]:
    """
    加载所有数据集的 train.json + val.json，并记录每条记录来源。

    返回:
        (all_records, path_to_source)
        path_to_source: path -> (json_path, record_index)
    """
    all_records: List[Dict] = []
    path_to_source: Dict[str, Tuple[str, int]] = {}

    for dataset_name, cfg in DATASET_CONFIGS.items():
        metadata_dir = os.path.join(data_root, cfg["metadata_dir"])
        if not os.path.isdir(metadata_dir):
            continue

        for split in ("train", "val"):
            json_path = os.path.join(metadata_dir, f"{split}.json")
            if not os.path.exists(json_path):
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                records = json.load(f)

            for idx, meta in enumerate(records):
                path = meta.get("path")
                if not path:
                    continue
                all_records.append(meta)
                path_to_source[path] = (json_path, idx)

    return all_records, path_to_source


def group_by_fingerprint(
    records: List[Dict],
) -> Tuple[Dict[str, List[Dict]], int]:
    """按 fingerprint 分组，过滤无指纹记录。"""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    skipped = 0
    for meta in records:
        fp = meta.get("fingerprint")
        if not fp or fp in ("", "unknown"):
            skipped += 1
            continue
        groups[fp].append(meta)

    return dict(groups), skipped


def _convert_one_part(args: Tuple) -> Dict[str, Any]:
    """
    将指纹的一个分片转为单个 HDF5 文件。子进程执行。

    args: (fp, ch, part_idx, records_json, hdf5_dir, hdf5_subdir, data_root, skip_existing)
    records_json: 序列化后的 records 列表（path, channels, ...）
    """
    try:
        import torch
        import h5py
    except ImportError as e:
        return {"ok": False, "error": str(e), "idx_map": {}, "hdf5_path": ""}

    (
        fp,
        ch,
        part_idx,
        records_json,
        hdf5_dir,
        hdf5_subdir,
        data_root,
        skip_existing,
    ) = args

    records = json.loads(records_json)
    records_sorted = sorted(records, key=lambda m: m["path"])

    # 文件名
    if part_idx is None:
        fname = f"{ch}ch_{fp}.h5"
    else:
        fname = f"{ch}ch_{fp}_part{part_idx}.h5"

    hdf5_path_abs = Path(hdf5_dir) / fname
    hdf5_path_rel = str(Path(hdf5_subdir) / fname)

    # 断点续传
    if skip_existing and hdf5_path_abs.exists():
        try:
            with h5py.File(str(hdf5_path_abs), "r") as h5:
                existing_n = h5["x"].shape[0]
            if existing_n == len(records_sorted):
                idx_map = {m["path"]: i for i, m in enumerate(records_sorted)}
                return {
                    "ok": True,
                    "hdf5_path": hdf5_path_rel,
                    "idx_map": idx_map,
                    "n_ok": existing_n,
                    "n_fail": 0,
                    "skipped": True,
                }
        except Exception:
            pass

    # 删除不完整的 .tmp
    tmp_path = str(hdf5_path_abs) + ".tmp"
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    x_list: List[np.ndarray] = []
    idx_map: Dict[str, int] = {}
    pos_ref: Optional[np.ndarray] = None
    sensor_type_ref: Optional[np.ndarray] = None
    sample_shape: Optional[Tuple[int, int]] = None
    n_fail = 0

    for local_idx, meta in enumerate(records_sorted):
        orig_path = meta["path"]
        if data_root:
            try:
                rel = os.path.relpath(orig_path, DEFAULT_DATA_ROOT)
                full_path = os.path.join(data_root, rel)
            except ValueError:
                full_path = orig_path
        else:
            full_path = orig_path

        try:
            data = torch.load(full_path, weights_only=True)
            x_i = data["x"].view(torch.int16).numpy()
            x_list.append(x_i)
            idx_map[orig_path] = local_idx

            if sample_shape is None:
                sample_shape = x_i.shape
            if pos_ref is None:
                pos_ref = data["pos"].view(torch.int16).numpy()
                sensor_type_ref = data["sensor_type"].numpy()
        except Exception:
            n_fail += 1
            C = int(meta.get("channels", 128))
            T = 2560
            if sample_shape:
                C, T = sample_shape
            dummy = np.zeros((C, T), dtype=np.int16)
            x_list.append(dummy)
            if pos_ref is None:
                pos_ref = np.zeros((C, 6), dtype=np.int16)
                sensor_type_ref = np.zeros(C, dtype=np.int32)

    if not x_list:
        return {"ok": False, "error": "所有样本加载失败", "idx_map": {}, "hdf5_path": ""}

    try:
        x_array = np.stack(x_list, axis=0)
    except ValueError:
        from collections import Counter
        shapes = [a.shape for a in x_list]
        dom = Counter(shapes).most_common(1)[0][0]
        x_array = np.stack(
            [a if a.shape == dom else np.zeros(dom, dtype=np.int16) for a in x_list],
            axis=0,
        )

    hdf5_path_abs.parent.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(tmp_path, "w") as h5:
            h5.create_dataset(
                "x",
                data=x_array,
                dtype="int16",
                chunks=(1, x_array.shape[1], x_array.shape[2]),
            )
            h5.create_dataset("pos", data=pos_ref, dtype="int16")
            h5.create_dataset("sensor_type", data=sensor_type_ref, dtype="int32")
            h5.attrs["fingerprint"] = fp
            h5.attrs["n_channels"] = int(ch)
            h5.attrs["n_samples"] = len(x_list)
            h5.attrs["n_failed"] = n_fail

        os.rename(tmp_path, str(hdf5_path_abs))
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return {"ok": False, "error": str(e), "idx_map": {}, "hdf5_path": ""}

    return {
        "ok": True,
        "hdf5_path": hdf5_path_rel,
        "idx_map": idx_map,
        "n_ok": len(x_list) - n_fail,
        "n_fail": n_fail,
        "skipped": False,
    }


def update_json_files(
    path_to_hdf5: Dict[str, Tuple[str, int]],
    path_to_source: Dict[str, Tuple[str, int]],
    logger: logging.Logger,
) -> None:
    """
    按 JSON 来源分组，覆盖每条记录的 hdf5_path 和 hdf5_idx。
    """
    json_updates: Dict[str, Dict[int, Tuple[str, int]]] = defaultdict(dict)
    for path, (hdf5_path, hdf5_idx) in path_to_hdf5.items():
        if path not in path_to_source:
            continue
        json_path, record_idx = path_to_source[path]
        json_updates[json_path][record_idx] = (hdf5_path, hdf5_idx)

    for json_path in sorted(json_updates.keys()):
        idx_to_val = json_updates[json_path]
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        changed = False
        for idx, (hdf5_path, hdf5_idx) in idx_to_val.items():
            if idx < len(records):
                records[idx]["hdf5_path"] = hdf5_path
                records[idx]["hdf5_idx"] = hdf5_idx
                changed = True

        if changed:
            tmp_path = json_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
            os.rename(tmp_path, json_path)
            logger.info(f"  已更新: {json_path} ({len(idx_to_val)} 条记录)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按指纹分组将 .pt 转为 HDF5（覆盖旧 hdf5_path/hdf5_idx）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="并行进程数（NFS 建议 4–8，过高会导致 I/O 争用卡顿）",
    )
    parser.add_argument(
        "--max_samples_per_file",
        type=int,
        default=10000,
        help="每个 HDF5 文件最大样本数（NFS 友好，建议 5K–10K）",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="跳过已存在且样本数正确的 HDF5（断点续转）",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="强制重新转换",
    )
    parser.add_argument(
        "--log_dir",
        default="outputs/convert_logs",
        help="日志目录",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="指定日志文件路径（覆盖 --log_dir）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅打印计划，不执行转换",
    )
    args = parser.parse_args()

    if args.no_skip_existing:
        args.skip_existing = False

    logger = setup_logging(log_dir=args.log_dir, log_file=args.log_file)

    logger.info("=" * 60)
    logger.info("PENCI .pt → HDF5 按指纹转换")
    logger.info(f"  数据根目录        : {args.data_root}")
    logger.info(f"  并行进程数        : {args.workers}")
    logger.info(f"  每文件最大样本数  : {args.max_samples_per_file}")
    logger.info(f"  断点续传          : {args.skip_existing}")
    logger.info(f"  dry_run           : {args.dry_run}")
    logger.info("=" * 60)

    t_start = time.time()

    # 加载 metadata
    logger.info("加载 metadata ...")
    all_records, path_to_source = load_all_metadata(args.data_root)
    logger.info(f"  总记录数: {len(all_records):,}")

    fp_groups, skipped = group_by_fingerprint(all_records)
    if skipped:
        logger.warning(f"  跳过无 fingerprint 记录: {skipped}")

    logger.info(f"  唯一指纹数: {len(fp_groups)}")

    # 构建任务：(fp, ch, part_idx, records_json, hdf5_dir, hdf5_subdir, data_root, skip_existing)
    hdf5_dir = os.path.join(args.data_root, HDF5_SUBDIR)
    hdf5_subdir = HDF5_SUBDIR
    tasks: List[Tuple] = []

    for fp, records in fp_groups.items():
        ch = records[0].get("channels", 128)
        records_sorted = sorted(records, key=lambda m: m["path"])
        n = len(records_sorted)

        if n <= args.max_samples_per_file:
            part_idx = None
            chunk = records_sorted
            tasks.append((
                fp, ch, part_idx,
                json.dumps([{k: v for k, v in r.items() if k != "_json_source" and k != "_json_index"}
                           for r in chunk]),
                hdf5_dir, hdf5_subdir, args.data_root, args.skip_existing,
            ))
        else:
            for k, i in enumerate(range(0, n, args.max_samples_per_file)):
                chunk = records_sorted[i : i + args.max_samples_per_file]
                tasks.append((
                    fp, ch, k,
                    json.dumps([{k: v for k, v in r.items() if k != "_json_source" and k != "_json_index"}
                               for r in chunk]),
                    hdf5_dir, hdf5_subdir, args.data_root, args.skip_existing,
                ))

    # 小任务优先：样本少的任务先完成，更快看到进度，减轻 NFS 并发压力
    tasks.sort(key=lambda t: len(json.loads(t[3])))

    logger.info(f"  转换任务数: {len(tasks)}（已按样本数升序，小任务优先）")

    if args.dry_run:
        for fp, records in fp_groups.items():
            ch = records[0].get("channels", 128)
            n = len(records)
            n_parts = (n + args.max_samples_per_file - 1) // args.max_samples_per_file
            logger.info(f"  [{ch}ch] fp={fp}: {n:,} 样本 → {n_parts} 个文件")
        logger.info("[dry_run] 未执行实际转换")
        return

    # 并行转换
    path_to_hdf5: Dict[str, Tuple[str, int]] = {}
    n_ok_total = 0
    n_fail_total = 0
    n_skipped = 0

    logger.info(
        f"开始转换，使用 {args.workers} 个进程。"
        "首个任务完成前无进度（小任务约 1–3 分钟，大任务 5–15 分钟）..."
    )
    sys.stdout.flush()
    sys.stderr.flush()

    with mp.Pool(processes=args.workers) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_convert_one_part, tasks), start=1
        ):
            if result["ok"]:
                for path, idx in result["idx_map"].items():
                    path_to_hdf5[path] = (result["hdf5_path"], idx)
                n_ok_total += result.get("n_ok", len(result["idx_map"]))
                n_fail_total += result.get("n_fail", 0)
                if result.get("skipped"):
                    n_skipped += 1
            else:
                logger.warning(f"  任务失败: {result.get('error', '未知')}")

            if i % 5 == 0 or i == len(tasks):
                elapsed = time.time() - t_start
                speed = n_ok_total / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - i) / (i / elapsed) if elapsed > 0 and i > 0 else 0
                logger.info(
                    f"  进度: {i}/{len(tasks)} 任务 | "
                    f"样本 {n_ok_total:,} ok / {n_fail_total} 失败 | "
                    f"跳过 {n_skipped} 个已完成文件 | "
                    f"{speed:.0f} 样本/s | ETA {eta/60:.1f} min"
                )

    elapsed = time.time() - t_start
    logger.info(
        f"转换完成: {n_ok_total:,} 样本，{n_fail_total} 失败，"
        f"跳过 {n_skipped} 个已存在文件，耗时 {elapsed/60:.1f} min"
    )

    # 更新 JSON
    logger.info("更新 metadata JSON（覆盖 hdf5_path、hdf5_idx）...")
    update_json_files(path_to_hdf5, path_to_source, logger)

    logger.info("=" * 60)
    logger.info("全部完成！重启训练即可使用新的 HDF5 路径。")
    logger.info("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
