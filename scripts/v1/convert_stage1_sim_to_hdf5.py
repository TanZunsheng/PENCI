#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage1 仿真数据 .pt -> HDF5 转换脚本。

转换策略：
1. 扫描 stage1_all_templates 下各通道目录中的 train / val metadata
2. 按 (split, n_channels, layout_fingerprint) 分桶
3. 每个桶写入一个或多个 HDF5 shard
4. 为原 metadata 生成带 hdf5_path / hdf5_idx 的新 JSON，供 sim_pretrain 直接使用

与真实数据按受试者分 HDF5 的策略不同，这里按训练桶切分，目的是让：
- 同一 batch 更容易连续消费同一个 HDF5 文件
- file scheduler / io_prefetch 能顺着真实训练顺序工作
- 避免重复存储同布局共享的 pos / sensor_type / leadfield
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from penci.v1.data.simulation_dataset import _normalize_sim_metadata_item

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def discover_metadata_sources(input_root: Path) -> List[Tuple[str, Path]]:
    """发现 stage1_all_templates 下所有 train / val metadata。"""
    sources: List[Tuple[str, Path]] = []
    for channel_dir in sorted(
        (path for path in input_root.iterdir() if path.is_dir() and path.name.endswith("ch")),
        key=lambda path: int(path.name[:-2]),
    ):
        for split in ("train", "val"):
            metadata_path = channel_dir / f"stage1_{split}_metadata.json"
            if metadata_path.is_file():
                sources.append((split, metadata_path))
    return sources


def load_metadata_source(
    split: str,
    metadata_path: Path,
) -> List[Dict[str, Any]]:
    """加载单个 metadata 文件，并记录来源与 split。"""
    with open(metadata_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"metadata 不是列表: {metadata_path}")

    normalized: List[Dict[str, Any]] = []
    for record in records:
        item = _normalize_sim_metadata_item(record, source_path=metadata_path)
        item["_source_metadata_path"] = str(metadata_path)
        item["_split"] = split
        normalized.append(item)
    return normalized


def load_all_metadata(input_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """加载全部 metadata，并同时保留按来源文件分组的记录列表。"""
    all_records: List[Dict[str, Any]] = []
    records_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for split, metadata_path in discover_metadata_sources(input_root):
        records = load_metadata_source(split, metadata_path)
        all_records.extend(records)
        records_by_source[str(metadata_path)] = records
        logger.info("加载 %s: %d 条记录", metadata_path, len(records))
    return all_records, records_by_source


def group_records_by_bucket(
    records: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, int, str], List[Dict[str, Any]]]:
    """按 (split, channels, fingerprint) 分桶。"""
    groups: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        split = str(record["_split"])
        channels = int(record["channels"])
        fingerprint = str(record.get("fingerprint") or record.get("layout_fingerprint") or "unknown")
        groups[(split, channels, fingerprint)].append(record)
    return groups


def _build_hdf5_file_name(channels: int, fingerprint: str, part_idx: Optional[int]) -> str:
    if part_idx is None:
        return f"{channels}ch_{fingerprint}.h5"
    return f"{channels}ch_{fingerprint}_part{part_idx:03d}.h5"


def _load_stage1_sample(sample_path: str) -> Dict[str, torch.Tensor]:
    data = torch.load(sample_path, map_location="cpu", weights_only=True)
    required = ("x", "pos", "sensor_type", "leadfield", "s_true")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"仿真样本缺少字段 {missing}: {sample_path}")
    return data


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _repeat_static_tensor(tensor: torch.Tensor, repeats: int) -> List[torch.Tensor]:
    return [tensor.clone() for _ in range(repeats)]


def _write_hdf5_shard(
    records: Sequence[Dict[str, Any]],
    output_path: Path,
    skip_existing: bool = True,
) -> Dict[str, Tuple[str, int]]:
    """
    写单个 HDF5 shard，并返回 sample_path -> (hdf5_path, hdf5_idx) 映射。
    """
    sorted_records = sorted(records, key=lambda item: str(item["path"]))
    if skip_existing and output_path.is_file():
        try:
            with h5py.File(output_path, "r") as h5:
                existing_n = int(h5["x"].shape[0])
            if existing_n == len(sorted_records):
                return {
                    str(item["path"]): (str(output_path), idx)
                    for idx, item in enumerate(sorted_records)
                }
        except Exception:
            logger.warning("检测到损坏或不完整 HDF5，将重写: %s", output_path)

    x_list: List[np.ndarray] = []
    s_true_list: List[np.ndarray] = []

    pos_ref: Optional[torch.Tensor] = None
    sensor_type_ref: Optional[torch.Tensor] = None
    leadfield_ref: Optional[torch.Tensor] = None
    pos_samples: Optional[List[torch.Tensor]] = None
    sensor_type_samples: Optional[List[torch.Tensor]] = None
    leadfield_samples: Optional[List[torch.Tensor]] = None

    for sample_idx, record in enumerate(sorted_records):
        sample = _load_stage1_sample(str(record["path"]))
        x = sample["x"].detach().cpu().float()
        pos = sample["pos"].detach().cpu().float()
        sensor_type = sample["sensor_type"].detach().cpu().long()
        leadfield = sample["leadfield"].detach().cpu().float()
        s_true = sample["s_true"].detach().cpu().float()

        x_list.append(x.numpy())
        s_true_list.append(s_true.numpy())

        if pos_ref is None:
            pos_ref = pos
            sensor_type_ref = sensor_type
            leadfield_ref = leadfield
            continue

        if pos_samples is not None:
            pos_samples.append(pos)
        elif not torch.equal(pos, pos_ref):
            pos_samples = _repeat_static_tensor(pos_ref, sample_idx)
            pos_samples.append(pos)

        if sensor_type_samples is not None:
            sensor_type_samples.append(sensor_type)
        elif not torch.equal(sensor_type, sensor_type_ref):
            sensor_type_samples = _repeat_static_tensor(sensor_type_ref, sample_idx)
            sensor_type_samples.append(sensor_type)

        if leadfield_samples is not None:
            leadfield_samples.append(leadfield)
        elif not torch.allclose(leadfield, leadfield_ref):
            leadfield_samples = _repeat_static_tensor(leadfield_ref, sample_idx)
            leadfield_samples.append(leadfield)

    if pos_ref is None or sensor_type_ref is None or leadfield_ref is None:
        raise RuntimeError(f"HDF5 shard 没有任何可写样本: {output_path}")

    x_array = np.stack(x_list, axis=0).astype(np.float32, copy=False)
    s_true_array = np.stack(s_true_list, axis=0).astype(np.float32, copy=False)

    _ensure_parent(output_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with h5py.File(tmp_path, "w") as h5:
        h5.create_dataset(
            "x",
            data=x_array,
            dtype="float32",
            chunks=(1, x_array.shape[1], x_array.shape[2]),
        )
        h5.create_dataset(
            "s_true",
            data=s_true_array,
            dtype="float32",
            chunks=(1, s_true_array.shape[1], s_true_array.shape[2]),
        )

        if pos_samples is None:
            h5.create_dataset("pos", data=pos_ref.numpy(), dtype="float32")
        else:
            pos_array = np.stack([tensor.numpy() for tensor in pos_samples], axis=0).astype(np.float32)
            h5.create_dataset(
                "pos",
                data=pos_array,
                dtype="float32",
                chunks=(1, pos_array.shape[1], pos_array.shape[2]),
            )

        if sensor_type_samples is None:
            h5.create_dataset("sensor_type", data=sensor_type_ref.numpy().astype(np.int32), dtype="int32")
        else:
            sensor_type_array = np.stack(
                [tensor.numpy().astype(np.int32) for tensor in sensor_type_samples],
                axis=0,
            )
            h5.create_dataset(
                "sensor_type",
                data=sensor_type_array,
                dtype="int32",
                chunks=(1, sensor_type_array.shape[1]),
            )

        if leadfield_samples is None:
            h5.create_dataset("leadfield", data=leadfield_ref.numpy(), dtype="float32")
        else:
            leadfield_array = np.stack(
                [tensor.numpy() for tensor in leadfield_samples],
                axis=0,
            ).astype(np.float32)
            h5.create_dataset(
                "leadfield",
                data=leadfield_array,
                dtype="float32",
                chunks=(1, leadfield_array.shape[1], leadfield_array.shape[2]),
            )

        h5.attrs["n_samples"] = len(sorted_records)
        h5.attrs["n_channels"] = int(pos_ref.shape[0])
        h5.attrs["shared_pos"] = pos_samples is None
        h5.attrs["shared_sensor_type"] = sensor_type_samples is None
        h5.attrs["shared_leadfield"] = leadfield_samples is None

    os.replace(tmp_path, output_path)

    return {
        str(item["path"]): (str(output_path), idx)
        for idx, item in enumerate(sorted_records)
    }


def convert_groups_to_hdf5(
    grouped_records: Dict[Tuple[str, int, str], List[Dict[str, Any]]],
    output_root: Path,
    max_samples_per_file: int,
    skip_existing: bool = True,
) -> Dict[str, Tuple[str, int]]:
    """把全部桶写成 HDF5，并返回全局 sample_path -> (path, idx) 映射。"""
    sample_to_hdf5: Dict[str, Tuple[str, int]] = {}
    total_groups = len(grouped_records)
    started_at = time.time()

    for group_idx, ((split, channels, fingerprint), records) in enumerate(sorted(grouped_records.items()), start=1):
        records = sorted(records, key=lambda item: str(item["path"]))
        n_parts = max(1, math.ceil(len(records) / max_samples_per_file))
        logger.info(
            "转换桶 %d/%d | %s | %dch | fp=%s | 样本 %d | 分片 %d",
            group_idx,
            total_groups,
            split,
            channels,
            fingerprint,
            len(records),
            n_parts,
        )

        for part_idx in range(n_parts):
            start = part_idx * max_samples_per_file
            end = min(len(records), (part_idx + 1) * max_samples_per_file)
            part_records = records[start:end]
            file_name = _build_hdf5_file_name(
                channels=channels,
                fingerprint=fingerprint,
                part_idx=part_idx if n_parts > 1 else None,
            )
            output_path = output_root / split / file_name
            mapping = _write_hdf5_shard(
                part_records,
                output_path=output_path,
                skip_existing=skip_existing,
            )
            sample_to_hdf5.update(mapping)

        elapsed_min = (time.time() - started_at) / 60.0
        logger.info("桶完成 | %s | 已耗时 %.1f min", fingerprint, elapsed_min)

    return sample_to_hdf5


def write_updated_metadata_files(
    records_by_source: Dict[str, List[Dict[str, Any]]],
    sample_to_hdf5: Dict[str, Tuple[str, int]],
) -> Dict[str, str]:
    """为每个原 metadata 文件生成一个带 HDF5 索引的新 JSON。"""
    output_paths: Dict[str, str] = {}
    for source_path, records in records_by_source.items():
        source = Path(source_path)
        output_path = source.with_name(source.stem + "_hdf5.json")
        updated_records: List[Dict[str, Any]] = []
        for record in records:
            sample_path = str(record["path"])
            if sample_path not in sample_to_hdf5:
                raise KeyError(f"样本未找到 HDF5 映射: {sample_path}")
            hdf5_path, hdf5_idx = sample_to_hdf5[sample_path]
            cleaned = {
                key: value
                for key, value in record.items()
                if not key.startswith("_")
            }
            cleaned["hdf5_path"] = hdf5_path
            cleaned["hdf5_idx"] = int(hdf5_idx)
            updated_records.append(cleaned)

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(updated_records, handle, ensure_ascii=False, indent=2)
        output_paths[source_path] = str(output_path)
        logger.info("写入新 metadata: %s", output_path)
    return output_paths


def write_aggregate_metadata(
    output_root: Path,
    records_by_source: Dict[str, List[Dict[str, Any]]],
    metadata_outputs: Dict[str, str],
) -> Dict[str, str]:
    """生成 train / val 聚合 metadata，便于直接在配置里引用。"""
    aggregate: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    for source_path, records in records_by_source.items():
        output_path = metadata_outputs[source_path]
        with open(output_path, "r", encoding="utf-8") as handle:
            updated_records = json.load(handle)
        split = str(records[0]["_split"]) if records else "train"
        aggregate[split].extend(updated_records)

    outputs: Dict[str, str] = {}
    for split, records in aggregate.items():
        out_path = output_root / f"stage1_{split}_metadata_hdf5_all.json"
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
        outputs[split] = str(out_path)
        logger.info("写入聚合 metadata: %s (%d 条)", out_path, len(records))
    return outputs


def write_conversion_summary(
    input_root: Path,
    output_root: Path,
    grouped_records: Dict[Tuple[str, int, str], List[Dict[str, Any]]],
    aggregate_outputs: Dict[str, str],
    metadata_outputs: Dict[str, str],
) -> Path:
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "n_buckets": len(grouped_records),
        "n_hdf5_files": sum(1 for _ in output_root.glob("*/*.h5")),
        "aggregate_metadata": aggregate_outputs,
        "per_source_metadata_outputs": metadata_outputs,
    }
    summary_path = output_root / "stage1_sim_hdf5_conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Stage1 simulation .pt files to HDF5")
    parser.add_argument(
        "--input_root",
        type=str,
        default="/work/2024/tanzunsheng/PENCI_sim_data/stage1_all_templates",
        help="Stage1 仿真数据根目录",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="",
        help="HDF5 输出目录，默认写到 <input_root>/hdf5",
    )
    parser.add_argument(
        "--max_samples_per_file",
        type=int,
        default=5000,
        help="单个 HDF5 shard 最多样本数",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="若 HDF5 shard 已存在且样本数匹配，则跳过重写",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_root}")

    output_root = Path(args.output_root).resolve() if args.output_root else input_root / "hdf5"
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("Stage1 仿真 HDF5 转换")
    logger.info("  输入目录: %s", input_root)
    logger.info("  输出目录: %s", output_root)
    logger.info("  每文件最大样本数: %d", args.max_samples_per_file)

    all_records, records_by_source = load_all_metadata(input_root)
    grouped_records = group_records_by_bucket(all_records)
    logger.info("发现 %d 条 metadata, %d 个训练桶", len(all_records), len(grouped_records))

    sample_to_hdf5 = convert_groups_to_hdf5(
        grouped_records=grouped_records,
        output_root=output_root,
        max_samples_per_file=max(1, int(args.max_samples_per_file)),
        skip_existing=bool(args.skip_existing),
    )
    metadata_outputs = write_updated_metadata_files(records_by_source, sample_to_hdf5)
    aggregate_outputs = write_aggregate_metadata(output_root, records_by_source, metadata_outputs)
    summary_path = write_conversion_summary(
        input_root=input_root,
        output_root=output_root,
        grouped_records=grouped_records,
        aggregate_outputs=aggregate_outputs,
        metadata_outputs=metadata_outputs,
    )

    logger.info("转换完成")
    logger.info("  train 聚合 metadata: %s", aggregate_outputs["train"])
    logger.info("  val 聚合 metadata  : %s", aggregate_outputs["val"])
    logger.info("  summary           : %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
