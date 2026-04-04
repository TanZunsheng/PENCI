# -*- coding: utf-8 -*-
"""
V1 仿真数据集

将第一层与第二层仿真数据接口与真实数据加载路径解耦。
支持 .pt 与 HDF5 两种存储格式。
"""

import json
import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

logger = logging.getLogger(__name__)

_CHANNEL_DIR_PATTERN = re.compile(r"^(?P<channels>\d+)ch$")
_SIM_METADATA_FP_KEYS = ("layout_fingerprint", "fingerprint", "layout_full_fingerprint")


def _infer_channels_from_path(path_like: Optional[Union[str, Path]]) -> Optional[int]:
    """从路径片段中推断通道数，例如 */128ch/train/sample_*.pt。"""
    if not path_like:
        return None

    for part in reversed(Path(path_like).parts):
        match = _CHANNEL_DIR_PATTERN.fullmatch(part)
        if match is not None:
            return int(match.group("channels"))
    return None


def _normalize_sim_metadata_item(
    item: Dict[str, Any],
    source_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """补齐仿真 metadata 中的 channels / fingerprint 等分桶关键信息。"""
    normalized = dict(item)

    if "channels" not in normalized:
        channel_hint = None
        for key in ("channels", "n_channels", "n_sensors"):
            value = normalized.get(key)
            if value is not None:
                channel_hint = int(value)
                break
        if channel_hint is None:
            channel_hint = _infer_channels_from_path(normalized.get("path"))
        if channel_hint is None:
            channel_hint = _infer_channels_from_path(source_path)
        if channel_hint is not None:
            normalized["channels"] = int(channel_hint)

    if "fingerprint" not in normalized:
        for key in _SIM_METADATA_FP_KEYS:
            fp = normalized.get(key)
            if fp not in (None, "", "unknown"):
                normalized["fingerprint"] = str(fp)
                break

    return normalized


def load_simulation_metadata_sources(
    metadata_path: Union[str, Path, Sequence[Union[str, Path]]],
) -> List[Dict[str, Any]]:
    """
    读取一个或多个仿真 metadata JSON，并合并为单个 metadata 列表。

    支持：
      - 单个 JSON 路径
      - 多个 JSON 路径组成的列表/元组
    """
    if isinstance(metadata_path, (str, Path)):
        metadata_paths = [metadata_path]
    elif isinstance(metadata_path, Sequence):
        metadata_paths = list(metadata_path)
    else:
        raise TypeError(f"metadata_path 类型不支持: {type(metadata_path)!r}")

    merged: List[Dict[str, Any]] = []
    for source_path in metadata_paths:
        with open(source_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if not isinstance(metadata, list):
            raise ValueError(f"仿真 metadata 文件必须是列表: {source_path}")
        merged.extend(
            _normalize_sim_metadata_item(item, source_path=source_path)
            for item in metadata
        )
    return merged


class _BaseSimulationDataset(Dataset):
    """
    仿真数据基类。

    metadata 支持两种形式：
      1. {"path": "..."}：按路径读取 .pt
      2. 直接内嵌样本字典
    """

    def __init__(
        self,
        metadata_path: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        dtype: torch.dtype = torch.float32,
        data_root: Optional[str] = None,
    ):
        super().__init__()
        if metadata is None and metadata_path is None:
            raise ValueError("必须提供 metadata_path 或 metadata")

        if metadata is None:
            metadata = load_simulation_metadata_sources(metadata_path)
        self.metadata = [_normalize_sim_metadata_item(item) for item in metadata]
        self.dtype = dtype
        self.data_root = data_root
        self._channel_count_cache: Dict[int, int] = {}
        self._fingerprint_cache: Dict[int, str] = {}
        self._h5_handles: Dict[str, "h5py.File"] = {}
        self._h5_pos_cache: Dict[str, torch.Tensor] = {}
        self._h5_sensor_type_cache: Dict[str, torch.Tensor] = {}
        self._h5_leadfield_cache: Dict[str, torch.Tensor] = {}
        self._use_hdf5 = HAS_H5PY and any(
            item.get("hdf5_path") and item.get("hdf5_idx") is not None
            for item in self.metadata
        )
        if self._use_hdf5:
            n_hdf5 = sum(1 for item in self.metadata if item.get("hdf5_path"))
            logger.info("仿真 HDF5 加载模式: %d/%d 个样本可用 HDF5", n_hdf5, len(self.metadata))
        elif any(item.get("hdf5_path") for item in self.metadata) and not HAS_H5PY:
            logger.warning("metadata 含 hdf5_path，但 h5py 未安装，仿真数据将回退到 .pt 加载")

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_item(self, index: int) -> Dict[str, Any]:
        item = self.metadata[index]
        if "path" in item:
            path = Path(item["path"])
            if not path.is_file():
                raise FileNotFoundError(f"仿真样本文件不存在: {path}")
            return torch.load(path, map_location="cpu", weights_only=True)
        return item

    def _to_tensor(self, value: Any, *, is_long: bool = False) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value.clone()
        else:
            tensor = torch.tensor(value)
        return tensor.long() if is_long else tensor.to(self.dtype)

    def _resolve_hdf5_path(self, hdf5_path: str) -> str:
        if os.path.isabs(hdf5_path):
            return hdf5_path
        if self.data_root:
            return os.path.join(self.data_root, hdf5_path)
        return hdf5_path

    def _get_h5_handle(self, h5_abs_path: str) -> "h5py.File":
        if h5_abs_path not in self._h5_handles:
            self._h5_handles[h5_abs_path] = h5py.File(h5_abs_path, "r")
        return self._h5_handles[h5_abs_path]

    def _read_hdf5_dataset(
        self,
        h5: "h5py.File",
        name: str,
        index: int,
    ) -> Optional[torch.Tensor]:
        if name not in h5:
            return None
        dataset = h5[name]
        value = dataset[:] if dataset.ndim <= 2 else dataset[index]
        tensor = torch.from_numpy(value.copy())
        if name == "sensor_type":
            return tensor.long()
        return tensor.to(self.dtype)

    def _load_from_hdf5_common(
        self,
        meta: Dict[str, Any],
        required_keys: Tuple[str, ...],
    ) -> Optional[Dict[str, torch.Tensor]]:
        hdf5_path = meta.get("hdf5_path")
        hdf5_idx = meta.get("hdf5_idx")
        if not hdf5_path or hdf5_idx is None:
            return None

        h5_abs_path = self._resolve_hdf5_path(str(hdf5_path))
        try:
            h5 = self._get_h5_handle(h5_abs_path)

            item: Dict[str, torch.Tensor] = {}
            for key in required_keys:
                if key == "pos":
                    if h5_abs_path not in self._h5_pos_cache:
                        tensor = self._read_hdf5_dataset(h5, "pos", int(hdf5_idx))
                        if tensor is None:
                            return None
                        if h5["pos"].ndim == 2:
                            self._h5_pos_cache[h5_abs_path] = tensor
                    if h5["pos"].ndim == 2:
                        item["pos"] = self._h5_pos_cache[h5_abs_path].clone()
                    else:
                        tensor = self._read_hdf5_dataset(h5, "pos", int(hdf5_idx))
                        if tensor is None:
                            return None
                        item["pos"] = tensor
                elif key == "sensor_type":
                    if h5_abs_path not in self._h5_sensor_type_cache:
                        tensor = self._read_hdf5_dataset(h5, "sensor_type", int(hdf5_idx))
                        if tensor is None:
                            return None
                        if h5["sensor_type"].ndim == 1:
                            self._h5_sensor_type_cache[h5_abs_path] = tensor
                    if h5["sensor_type"].ndim == 1:
                        item["sensor_type"] = self._h5_sensor_type_cache[h5_abs_path].clone()
                    else:
                        tensor = self._read_hdf5_dataset(h5, "sensor_type", int(hdf5_idx))
                        if tensor is None:
                            return None
                        item["sensor_type"] = tensor.long()
                elif key == "leadfield":
                    if h5_abs_path not in self._h5_leadfield_cache:
                        tensor = self._read_hdf5_dataset(h5, "leadfield", int(hdf5_idx))
                        if tensor is None:
                            return None
                        if h5["leadfield"].ndim == 2:
                            self._h5_leadfield_cache[h5_abs_path] = tensor
                    if h5["leadfield"].ndim == 2:
                        item["leadfield"] = self._h5_leadfield_cache[h5_abs_path].clone()
                    else:
                        tensor = self._read_hdf5_dataset(h5, "leadfield", int(hdf5_idx))
                        if tensor is None:
                            return None
                        item["leadfield"] = tensor
                else:
                    tensor = self._read_hdf5_dataset(h5, key, int(hdf5_idx))
                    if tensor is None:
                        return None
                    item[key] = tensor.long() if key == "sensor_type" else tensor

            return item
        except Exception as exc:
            handle = self._h5_handles.pop(h5_abs_path, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
            self._h5_pos_cache.pop(h5_abs_path, None)
            self._h5_sensor_type_cache.pop(h5_abs_path, None)
            self._h5_leadfield_cache.pop(h5_abs_path, None)
            logger.warning("仿真 HDF5 加载失败 %s[%s]: %s，回退到 .pt", h5_abs_path, hdf5_idx, exc)
            return None

    def close_h5_handles(self) -> None:
        for handle in self._h5_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._h5_handles.clear()
        self._h5_pos_cache.clear()
        self._h5_sensor_type_cache.clear()
        self._h5_leadfield_cache.clear()

    def get_channel_count(self, index: int) -> int:
        """获取样本通道数，优先读 metadata，必要时回退到样本本体。"""
        if index in self._channel_count_cache:
            return self._channel_count_cache[index]

        meta = self.metadata[index]
        value = meta.get("channels")
        if value is None:
            for key in ("n_channels", "n_sensors"):
                if meta.get(key) is not None:
                    value = meta[key]
                    break
        if value is None:
            value = _infer_channels_from_path(meta.get("path"))

        if value is None:
            item = self._load_item(index)
            if "x" in item:
                value = int(item["x"].shape[0])
            elif "pos" in item:
                value = int(item["pos"].shape[0])
            elif "leadfield" in item:
                value = int(item["leadfield"].shape[0])
            else:
                raise RuntimeError(f"无法推断仿真样本通道数: index={index}")

        value = int(value)
        meta["channels"] = value
        self._channel_count_cache[index] = value
        return value

    def get_fingerprint(self, index: int) -> str:
        """获取样本布局指纹，优先使用 metadata 中的 layout_fingerprint。"""
        if index in self._fingerprint_cache:
            return self._fingerprint_cache[index]

        meta = self.metadata[index]
        fingerprint: Optional[str] = None
        for key in _SIM_METADATA_FP_KEYS:
            value = meta.get(key)
            if value not in (None, "", "unknown"):
                fingerprint = str(value)
                break

        if fingerprint is None:
            item = self._load_item(index)
            pos = item.get("pos")
            if pos is not None:
                from penci.physics.leadfield_manager import compute_fingerprint_from_pos

                xyz = pos[:, :3]
                if torch.is_tensor(xyz):
                    xyz = xyz.detach().cpu().float().numpy()
                fingerprint = compute_fingerprint_from_pos(xyz)
            else:
                fingerprint = "unknown"

        meta["fingerprint"] = fingerprint
        self._fingerprint_cache[index] = fingerprint
        return fingerprint


class Stage1SimulationDataset(_BaseSimulationDataset):
    """
    第一层仿真数据集。

    样本字段:
        - x: (C, T)
        - pos: (C, 6)
        - sensor_type: (C,)
        - leadfield: (C, 72) 或 (B, C, 72)
        - s_true: (72, T_state)
    """

    REQUIRED_KEYS = ("x", "pos", "sensor_type", "leadfield", "s_true")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        meta = self.metadata[index]
        item = None
        if self._use_hdf5 and meta.get("hdf5_path"):
            item = self._load_from_hdf5_common(meta, self.REQUIRED_KEYS)
        if item is None:
            item = self._load_item(index)
        missing = [k for k in self.REQUIRED_KEYS if k not in item]
        if missing:
            raise KeyError(f"Stage1 仿真样本缺少字段: {missing}")

        return {
            "x": self._to_tensor(item["x"]),
            "pos": self._to_tensor(item["pos"]),
            "sensor_type": self._to_tensor(item["sensor_type"], is_long=True),
            "leadfield": self._to_tensor(item["leadfield"]),
            "s_true": self._to_tensor(item["s_true"]),
        }


class Stage2ConnectivitySimulationDataset(_BaseSimulationDataset):
    """
    第二层连接仿真数据集。

    样本字段:
        - x: (C, T)
        - pos: (C, 6)
        - sensor_type: (C,)
        - leadfield: (C, 72)
        - s_true: (72, T_state)
        - a_true: (72, 72)
    """

    REQUIRED_KEYS = ("x", "pos", "sensor_type", "leadfield", "s_true", "a_true")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        meta = self.metadata[index]
        item = None
        if self._use_hdf5 and meta.get("hdf5_path"):
            item = self._load_from_hdf5_common(meta, self.REQUIRED_KEYS)
        if item is None:
            item = self._load_item(index)
        missing = [k for k in self.REQUIRED_KEYS if k not in item]
        if missing:
            raise KeyError(f"Stage2 仿真样本缺少字段: {missing}")

        return {
            "x": self._to_tensor(item["x"]),
            "pos": self._to_tensor(item["pos"]),
            "sensor_type": self._to_tensor(item["sensor_type"], is_long=True),
            "leadfield": self._to_tensor(item["leadfield"]),
            "s_true": self._to_tensor(item["s_true"]),
            "a_true": self._to_tensor(item["a_true"]),
        }


def create_simulation_dataloader(
    dataset: _BaseSimulationDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    rank: int = 0,
    world_size: int = 1,
    use_bucket_sampler: bool = False,
    use_fingerprint: bool = False,
    file_scheduler: bool = False,
    shuffle_within_file: bool = True,
    drop_last: Optional[bool] = None,
) -> DataLoader:
    """
    为仿真数据创建 DataLoader。

    当 metadata 混合了多个通道数且 batch_size > 1 时，必须启用分桶采样，
    否则默认 collate 会因张量形状不一致而报错。
    """
    if drop_last is None:
        drop_last = shuffle

    if batch_size > 1 and not use_bucket_sampler:
        unique_channels = {dataset.get_channel_count(idx) for idx in range(len(dataset))}
        if len(unique_channels) > 1:
            raise RuntimeError(
                "仿真 metadata 包含多个通道数，必须启用 data.use_bucket_sampler=true "
                "才能混合训练。"
            )

    if use_bucket_sampler:
        from penci.data import DistributedBucketBatchSampler

        has_hdf5 = any(item.get("hdf5_path") for item in dataset.metadata)
        sampler = DistributedBucketBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
            use_fingerprint=use_fingerprint,
            cluster_by_file=has_hdf5,
            file_scheduler=file_scheduler and has_hdf5,
            shuffle_within_file=shuffle_within_file,
        )
        loader_kwargs: Dict[str, Any] = {
            "num_workers": num_workers,
            "pin_memory": True,
            "worker_init_fn": _simulation_worker_init_fn,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
        return DataLoader(dataset, batch_sampler=sampler, **loader_kwargs)

    sampler: Optional[DistributedSampler] = None
    dataloader_shuffle = shuffle
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        dataloader_shuffle = False

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": dataloader_shuffle,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "pin_memory": True,
        "worker_init_fn": _simulation_worker_init_fn,
    }
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)


def _simulation_worker_init_fn(worker_id: int) -> None:
    """清理 fork 继承的 HDF5 句柄，确保每个 worker 自己打开文件。"""
    del worker_id
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    if hasattr(dataset, "close_h5_handles"):
        dataset.close_h5_handles()
