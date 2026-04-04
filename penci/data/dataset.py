# -*- coding: utf-8 -*-
"""
PENCI 数据模块

适配 /work/2024/tanzunsheng/PENCIData 数据格式的数据加载器。

数据格式：
- 每个样本是一个 .pt 文件，包含字典：
  - x: (C, T) - EEG/MEG 信号，dtype=bfloat16
  - pos: (C, 6) - 电极位置和方向，dtype=bfloat16
  - sensor_type: (C,) - 传感器类型，dtype=int32

元数据格式（train.json / val.json）：
- dataset: 数据集名称
- path: 数据文件路径
- channels: 通道数
- is_eeg / is_meg: 数据类型标志

动态通道 + 电极指纹支持：
- BucketBatchSampler 按 (通道数, 电极指纹) 分桶，确保同一 batch 内
  不仅通道数相同，而且电极配置也完全一致
- 同桶内样本共享同一个导联场矩阵，避免冗余计算
- 不同桶之间可以有不同的通道数和电极配置
"""

import os
import json
import logging
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Iterator, List, Optional, Tuple, Any
from pathlib import Path

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

logger = logging.getLogger(__name__)


def _compute_sample_fingerprint(data_path: str, data_root: Optional[str] = None) -> str:
    """
    从 .pt 文件的 pos 张量计算电极位置指纹

    参数:
        data_path: 样本 .pt 文件的原始路径（元数据中的路径）
        data_root: 可选的数据根目录（覆盖绝对路径）

    返回:
        16 字符的十六进制指纹字符串，加载失败返回 "unknown"
    """
    from penci.physics.leadfield_manager import compute_fingerprint_from_pos

    actual_path = data_path
    if data_root is not None:
        rel_path = os.path.relpath(data_path, "/work/2024/tanzunsheng/PENCIData")
        actual_path = os.path.join(data_root, rel_path)

    try:
        data = torch.load(actual_path, weights_only=True)
        pos = data["pos"].float().numpy()  # (C, 6)
        xyz = pos[:, :3]  # 仅取空间坐标
        return compute_fingerprint_from_pos(xyz)
    except Exception as e:
        logger.warning(f"计算指纹失败 {actual_path}: {e}")
        return "unknown"


def _resolve_metadata_dir(data_root: str, dataset_name: str) -> str:
    """
    解析数据集的 metadata 目录路径，兼容两种目录结构:

    - 标准:  {data_root}/{dataset_name}-metadata/       (如 HBN_EEG)
    - 嵌套:  {data_root}/{parent}/{dataset_name}-metadata/  (如 Broderick2018 子数据集)

    嵌套规则: 以第一个下划线分割取前缀作为父目录候选。
    例: Broderick2018_NaturalSpeech → Broderick2018/Broderick2018_NaturalSpeech-metadata/

    参数:
        data_root: 数据根目录 (如 /work/2024/tanzunsheng/PENCIData)
        dataset_name: 数据集名称 (如 "HBN_EEG", "Broderick2018_NaturalSpeech")

    返回:
        metadata 目录的绝对路径

    异常:
        FileNotFoundError: 找不到 metadata 目录
    """
    # 优先尝试标准路径
    standard = os.path.join(data_root, f"{dataset_name}-metadata")
    if os.path.isdir(standard):
        return standard

    # 嵌套搜索: dataset_name 含下划线时，尝试 {prefix}/{dataset_name}-metadata
    # 例: Broderick2018_NaturalSpeech → Broderick2018/Broderick2018_NaturalSpeech-metadata
    parts = dataset_name.split("_", 1)
    nested = None
    if len(parts) > 1:
        parent = parts[0]
        nested = os.path.join(data_root, parent, f"{dataset_name}-metadata")
        if os.path.isdir(nested):
            return nested

    tried = [standard]
    if nested is not None:
        tried.append(nested)
    raise FileNotFoundError(f"找不到数据集 '{dataset_name}' 的 metadata 目录。" f"已尝试: {tried}")


class PENCIDataset(Dataset):
    """
    PENCI 数据集（支持电极指纹预计算）

    参数:
        metadata_path: 元数据 JSON 文件路径（与 metadata 二选一）
        metadata: 已加载的元数据列表（与 metadata_path 二选一）
        data_root: 数据根目录（可选，覆盖元数据中的绝对路径）
        transform: 数据变换函数
        max_length: 最大序列长度（超出将被截断）
        target_channels: 目标通道数（不足将被填充）。
                         设为 None 时保留原始通道数（配合 BucketBatchSampler 使用）。
        datasets: 仅加载指定数据集（列表）
        dtype: 输出数据类型
        precompute_fingerprints: 是否预计算所有样本的电极指纹
        random_crop: 序列超长时是否随机裁窗；验证/评估建议关闭
    """

    def __init__(
        self,
        metadata_path: Optional[str] = None,
        metadata: Optional[List[Dict]] = None,
        data_root: str = None,
        transform: callable = None,
        max_length: int = 2560,
        target_channels: Optional[int] = 128,
        datasets: List[str] = None,
        dtype: torch.dtype = torch.float32,
        precompute_fingerprints: bool = False,
        random_crop: bool = True,
        rank: int = 0,
    ):
        super().__init__()

        self.data_root = data_root
        self.transform = transform
        self.max_length = max_length
        self.target_channels = target_channels
        self.dtype = dtype
        self.random_crop = bool(random_crop)
        self.rank = rank

        if metadata is not None:
            self.metadata = metadata
        elif metadata_path is not None:
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            raise ValueError("必须提供 metadata_path 或 metadata 之一")

        if datasets is not None:
            self.metadata = [m for m in self.metadata if m.get("dataset") in datasets]

        if self.rank == 0:
            logger.info(f"加载了 {len(self.metadata)} 个样本")

        # HDF5 加载支持：per-worker 文件句柄缓存 + pos/sensor_type 缓存
        # DataLoader worker 进程间不共享，每个 worker 独立维护自己的缓存
        self._h5_handles: Dict[str, "h5py.File"] = {}
        self._h5_pos_cache: Dict[str, torch.Tensor] = {}
        self._h5_sensor_type_cache: Dict[str, torch.Tensor] = {}

        # 检测是否有 HDF5 可用
        self._use_hdf5 = HAS_H5PY and any(
            "hdf5_path" in m and "hdf5_idx" in m for m in self.metadata
        )
        if self.rank == 0:
            if self._use_hdf5:
                n_hdf5 = sum(1 for m in self.metadata if "hdf5_path" in m)
                logger.info(
                    f"HDF5 加载模式: {n_hdf5}/{len(self.metadata)} 个样本可用 HDF5 "
                    f"(h5py {'已' if HAS_H5PY else '未'}安装)"
                )
            else:
                if not HAS_H5PY:
                    logger.warning("h5py 未安装，将使用 .pt 文件加载（NFS 性能可能较差）")
                else:
                    logger.info("metadata 中无 hdf5_path 字段，使用 .pt 文件加载")

        # 预计算电极指纹（按唯一样本路径去重后批量计算）
        self._fingerprint_cache: Dict[str, str] = {}
        if precompute_fingerprints:
            self._precompute_fingerprints()

    def _precompute_fingerprints(self) -> None:
        """
        预计算所有样本的电极指纹（支持增量计算）

        增量策略：
        - 已有 fingerprint 字段的样本直接零 I/O 复用
        - 仅对缺失 fingerprint 的样本逐一计算
        - 新增数据集时，170 万老样本零开销，只计算新增的样本

        重要:
            强烈建议先运行离线预计算脚本 (scripts/precompute_all_leadfields.py --update_metadata)
            回写 fingerprint 字段到 metadata JSON，以最大化零 I/O 复用率。
        """
        # 增量策略：逐条检查，已有指纹的直接复用，缺失的收集起来
        need_compute: list = []

        for idx, meta in enumerate(self.metadata):
            fp = meta.get("fingerprint")
            if fp and fp not in (None, "", "unknown"):
                # 已有有效指纹 → 零 I/O 复用
                self._fingerprint_cache[meta["path"]] = fp
            else:
                need_compute.append(idx)

        cached_count = len(self.metadata) - len(need_compute)

        if not need_compute:
            unique_fps = set(self._fingerprint_cache.values())
            if self.rank == 0:
                logger.info(
                    f"电极指纹预计算完成: {len(unique_fps)} 个唯一指纹（全部来自 metadata 缓存）"
                )
            return

        # 仅对缺失指纹的样本逐一计算
        if cached_count > 0:
            if self.rank == 0:
                logger.info(
                    f"增量指纹计算: {cached_count} 个样本已有指纹（零 I/O 复用），"
                    f"{len(need_compute)} 个样本需要计算"
                )
        else:
            if self.rank == 0:
                logger.warning(
                    "metadata 中没有 fingerprint 字段，将逐一加载 .pt 文件计算指纹。"
                    "这在大数据集下会很慢。强烈建议先运行: "
                    "python scripts/precompute_all_leadfields.py --update_metadata"
                )

        for i, idx in enumerate(need_compute):
            meta = self.metadata[idx]
            fp = _compute_sample_fingerprint(meta["path"], self.data_root)
            self._fingerprint_cache[meta["path"]] = fp
            if self.rank == 0 and (i + 1) % 1000 == 0:
                logger.info(f"  指纹计算进度: {i + 1}/{len(need_compute)}")

        unique_fps = set(self._fingerprint_cache.values())
        if self.rank == 0:
            logger.info(
                f"电极指纹预计算完成: {len(unique_fps)} 个唯一指纹"
                f"（缓存复用 {cached_count}, 新计算 {len(need_compute)}）"
            )

    def get_fingerprint(self, idx: int) -> str:
        """获取指定样本的电极指纹"""
        meta = self.metadata[idx]
        path = meta["path"]
        if path in self._fingerprint_cache:
            return self._fingerprint_cache[path]
        fp = _compute_sample_fingerprint(path, self.data_root)
        self._fingerprint_cache[path] = fp
        return fp

    def __len__(self) -> int:
        return len(self.metadata)

    def get_channel_count(self, idx: int) -> int:
        """获取指定样本的通道数（从元数据读取，不加载 .pt 文件）"""
        return self.metadata[idx]["channels"]

    def _get_h5_handle(self, h5_abs_path: str) -> "h5py.File":
        """
        获取 HDF5 文件句柄（per-worker 缓存，避免重复 open/close）

        DataLoader 的每个 worker 进程有独立的 dataset 副本，
        所以这个缓存天然是 worker-local 的，无需加锁。
        """
        if h5_abs_path not in self._h5_handles:
            self._h5_handles[h5_abs_path] = h5py.File(h5_abs_path, "r")
        return self._h5_handles[h5_abs_path]

    def _load_from_hdf5(
        self, meta: Dict
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        从 HDF5 文件加载单个样本

        HDF5 结构（按受试者分组）：
            /x            (N, C, T) int16  ← bfloat16 原始 bits
            /pos          (C, 6)    int16  ← bfloat16 原始 bits（同受试者所有样本共享）
            /sensor_type  (C,)      int32  （同受试者所有样本共享）

        优化策略：
            - 文件句柄 per-worker 缓存（_h5_handles）：首次打开后持久复用
            - pos 和 sensor_type per-file 缓存（_h5_pos_cache / _h5_sensor_type_cache）：
              同一 HDF5 文件内的所有样本共享，只读一次
            - 每次 __getitem__ 仅触发 1 次 HDF5 chunk 读取（读 x[idx]）

        返回:
            (x, pos, sensor_type) 或 None（加载失败时）
        """
        hdf5_rel = meta.get("hdf5_path")
        hdf5_idx = meta.get("hdf5_idx")
        if hdf5_rel is None or hdf5_idx is None:
            return None

        data_root = self.data_root or "/work/2024/tanzunsheng/PENCIData"
        h5_abs_path = os.path.join(data_root, hdf5_rel)

        try:
            h5 = self._get_h5_handle(h5_abs_path)

            # x: 读取单个 chunk (1, C, T)，这是每次 __getitem__ 唯一的 NFS I/O
            # numpy 数组直接 view 为 bfloat16 再转目标 dtype
            x_np = h5["x"][hdf5_idx]  # (C, T) int16 numpy
            x = torch.from_numpy(x_np.copy()).view(torch.bfloat16).to(self.dtype)

            # pos 和 sensor_type: per-file 缓存（同一受试者所有样本共享）
            if h5_abs_path not in self._h5_pos_cache:
                pos_np = h5["pos"][:]
                self._h5_pos_cache[h5_abs_path] = (
                    torch.from_numpy(pos_np.copy()).view(torch.bfloat16).float()
                )
                self._h5_sensor_type_cache[h5_abs_path] = torch.from_numpy(
                    h5["sensor_type"][:].copy()
                ).long()

            pos = self._h5_pos_cache[h5_abs_path].to(self.dtype)
            sensor_type = self._h5_sensor_type_cache[h5_abs_path].clone()

            return x, pos, sensor_type
        except Exception as e:
            # HDF5 读取失败时清除可能损坏的句柄
            handle = self._h5_handles.pop(h5_abs_path, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
            self._h5_pos_cache.pop(h5_abs_path, None)
            self._h5_sensor_type_cache.pop(h5_abs_path, None)
            if self.rank == 0:
                logger.warning(f"HDF5 加载失败 {h5_abs_path}[{hdf5_idx}]: {e}，回退到 .pt")
            return None

    def close_h5_handles(self) -> None:
        """关闭所有缓存的 HDF5 文件句柄（DataLoader worker 退出时调用）"""
        for h in self._h5_handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._h5_handles.clear()
        self._h5_pos_cache.clear()
        self._h5_sensor_type_cache.clear()

    def _load_from_pt(
        self, meta: Dict
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """从 .pt 文件加载单个样本（回退路径）"""
        data_path = meta["path"]
        if self.data_root is not None:
            rel_path = os.path.relpath(data_path, "/work/2024/tanzunsheng/PENCIData")
            data_path = os.path.join(self.data_root, rel_path)

        try:
            data = torch.load(data_path, weights_only=True)
            x = data["x"].to(self.dtype)
            pos = data["pos"].to(self.dtype)
            sensor_type = data["sensor_type"].long()
            return x, pos, sensor_type
        except Exception as e:
            if self.rank == 0:
                logger.warning(f"无法加载 {data_path}: {e}")
            return None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回:
            字典包含:
            - x: EEG/MEG 信号 (C, T)
            - pos: 电极位置 (C, 6)
            - sensor_type: 传感器类型 (C,)
            - n_channels: 原始通道数 (标量 tensor)
            - metadata: 元数据字典（含 fingerprint）
        """
        meta = self.metadata[idx]

        # 优先从 HDF5 加载（I/O 效率高），失败则回退到 .pt
        loaded = None
        if self._use_hdf5 and "hdf5_path" in meta:
            loaded = self._load_from_hdf5(meta)
        if loaded is None:
            loaded = self._load_from_pt(meta)
        if loaded is None:
            return self._get_dummy_sample(meta.get("channels", self.target_channels or 128))

        x, pos, sensor_type = loaded
        C, T = x.shape
        n_channels = C

        if T > self.max_length:
            if self.random_crop:
                start = torch.randint(0, T - self.max_length + 1, (1,)).item()
            else:
                start = 0
            x = x[:, start : start + self.max_length]
        elif T < self.max_length:
            x = F.pad(x, (0, self.max_length - T), mode="constant", value=0)

        if self.target_channels is not None:
            if C < self.target_channels:
                pad_c = self.target_channels - C
                x = F.pad(x, (0, 0, 0, pad_c), mode="constant", value=0)
                pos = F.pad(pos, (0, 0, 0, pad_c), mode="constant", value=0)
                sensor_type = F.pad(sensor_type, (0, pad_c), mode="constant", value=0)
            elif C > self.target_channels:
                x = x[: self.target_channels]
                pos = pos[: self.target_channels]
                sensor_type = sensor_type[: self.target_channels]

        if self.transform is not None:
            x = self.transform(x)

        fingerprint = self.get_fingerprint(idx)

        return {
            "x": x,
            "pos": pos,
            "sensor_type": sensor_type,
            "n_channels": torch.tensor(n_channels, dtype=torch.long),
            "metadata": {
                "dataset": meta.get("dataset", "unknown"),
                "path": meta["path"],
                "channels": n_channels,
                "is_eeg": meta.get("is_eeg", True),
                "is_meg": meta.get("is_meg", False),
                "fingerprint": fingerprint,
                "hdf5_path": meta.get("hdf5_path", ""),
                "hdf5_idx": int(meta.get("hdf5_idx", -1)) if "hdf5_idx" in meta else -1,
                "sample_index": idx,
            },
        }

    def _get_dummy_sample(self, n_channels: int = 128) -> Dict[str, torch.Tensor]:
        """返回一个虚拟样本（用于错误处理）"""
        c = self.target_channels if self.target_channels is not None else n_channels
        return {
            "x": torch.zeros(c, self.max_length, dtype=self.dtype),
            "pos": torch.zeros(c, 6, dtype=self.dtype),
            "sensor_type": torch.zeros(c, dtype=torch.long),
            "n_channels": torch.tensor(c, dtype=torch.long),
            "metadata": {
                "dataset": "dummy",
                "path": "",
                "channels": c,
                "is_eeg": True,
                "is_meg": False,
                "fingerprint": "unknown",
                "hdf5_path": "",
                "hdf5_idx": -1,
                "sample_index": -1,
            },
        }


class BucketBatchSampler(Sampler):
    """
    按 (通道数, 电极指纹) 分桶的批采样器

    确保同一 batch 内所有样本不仅通道数相同，而且电极配置完全一致，从而：
    1. 避免不必要的通道 padding（节省显存）
    2. 同一 batch 内共享同一个导联场矩阵（正确且高效）

    分桶逻辑：
    - 如果 dataset 支持指纹（precompute_fingerprints=True），按 (channels, fingerprint) 分桶
    - 否则退化为按 channels 分桶（向后兼容）
    - 每个桶内随机打乱（训练时）或保持顺序（验证时）
    - 桶间顺序：训练时随机，验证时按桶键排序

    参数:
        dataset: PENCIDataset 实例
        batch_size: 每个 batch 的样本数
        shuffle: 是否打乱（训练时为 True）
        drop_last: 是否丢弃每个桶最后不满一个 batch 的样本
        seed: 随机种子（用于可复现性）
        use_fingerprint: 是否使用电极指纹分桶（需 dataset 已预计算指纹）
    """

    def __init__(
        self,
        dataset: PENCIDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
        use_fingerprint: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.use_fingerprint = use_fingerprint

        # 按 (n_channels, fingerprint) 或 n_channels 分桶
        self.buckets: Dict[Any, List[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            ch = dataset.get_channel_count(idx)
            if use_fingerprint:
                fp = dataset.get_fingerprint(idx)
                bucket_key = (ch, fp)
            else:
                bucket_key = ch
            self.buckets[bucket_key].append(idx)

        self._total_batches = 0
        for key, indices in sorted(self.buckets.items(), key=lambda x: str(x[0])):
            n_batches = len(indices) // batch_size
            if not drop_last and len(indices) % batch_size > 0:
                n_batches += 1
            self._total_batches += n_batches
            if use_fingerprint:
                ch, fp = key
                logger.info(f"  桶 [{ch}ch, fp={fp}]: {len(indices)} 样本, {n_batches} batches")
            else:
                logger.info(f"  桶 [{key} ch]: {len(indices)} 样本, {n_batches} batches")

    def __iter__(self) -> Iterator[List[int]]:
        """生成 batch 索引列表"""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        all_batches = []

        for key in sorted(self.buckets.keys(), key=lambda x: str(x)):
            indices = self.buckets[key].copy()

            # 桶内打乱
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in perm]

            # 划分 batch
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        # 桶间打乱（训练时）
        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in perm]

        yield from all_batches

    def __len__(self) -> int:
        return self._total_batches

    def set_epoch(self, epoch: int) -> None:
        """设置 epoch（用于分布式训练时同步随机种子）"""
        self.epoch = epoch


class DistributedBucketBatchSampler(Sampler):
    """
    DDP 兼容的分桶批采样器（I/O 聚簇优化）

    保证每个 batch 内所有样本共享同一 leadfield（同一 (n_channels, fingerprint) 桶）。
    分布式分配在桶级别内部执行：每个桶 padding 到 (num_replicas * batch_size) 整数倍，
    然后 stride-subsample 给各 rank，确保各 rank batch 数量完全相等。

    I/O 聚簇优化（cluster_by_file=True）：
        HDF5 文件按受试者组织，但指纹桶内包含多个受试者的所有样本（例如 HBN_EEG
        156 万样本分布在 3,461 个 HDF5 文件中，却共享同一个指纹）。
        完全随机打乱导致每个 batch 的 32 个样本来自 ~32 个不同 HDF5 文件，
        32 个 DataLoader worker 同时随机读 → NFS 被打满。

        聚簇策略：桶内先按 hdf5_path 分组，打乱组顺序 + 组内打乱样本，
        再串联成线性序列后切 batch。连续 batch 倾向于来自同一/相邻受试者，
        大幅减少跨文件随机 I/O，同时保持 epoch 级别的随机性。

    当 num_replicas=1, rank=0 时行为与 BucketBatchSampler 完全一致（单卡兼容）。
    """

    def __init__(
        self,
        dataset: PENCIDataset,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
        use_fingerprint: bool = False,
        cluster_by_file: bool = True,
        file_scheduler: bool = False,
        shuffle_within_file: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.use_fingerprint = use_fingerprint
        self.cluster_by_file = cluster_by_file
        self.file_scheduler = file_scheduler
        self.shuffle_within_file = shuffle_within_file

        self.buckets: Dict[Any, List[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            ch = dataset.get_channel_count(idx)
            if use_fingerprint:
                fp = dataset.get_fingerprint(idx)
                bucket_key = (ch, fp)
            else:
                bucket_key = ch
            self.buckets[bucket_key].append(idx)

        # 预构建 idx → hdf5_path 映射（用于聚簇排序）
        self._idx_to_h5: Dict[int, str] = {}
        self._idx_to_h5_idx: Dict[int, int] = {}
        if cluster_by_file:
            for idx in range(len(dataset)):
                meta = dataset.metadata[idx]
                self._idx_to_h5[idx] = meta.get("hdf5_path", "")
                self._idx_to_h5_idx[idx] = int(meta.get("hdf5_idx", -1))

        # epoch 级缓存：避免在同一个 epoch 内重复构建 batch 计划
        self._cached_epoch = -1
        self._cached_batches: List[List[int]] = []
        self._cached_file_plan: List[str] = []
        self._cached_rank_schedule: List[Dict[str, Any]] = []

        self._total_batches = 0
        self._refresh_cached_plan(epoch=0)
        self._total_batches = len(self._cached_batches)

        if self.rank == 0:
            total_batch_size = self.num_replicas * self.batch_size
            for key, indices in sorted(self.buckets.items(), key=lambda x: str(x[0])):
                if self.drop_last:
                    rem = (total_batch_size - (len(indices) % total_batch_size)) % total_batch_size
                    padded_len = len(indices) + rem
                    n_batches = padded_len // total_batch_size
                else:
                    rem = (
                        self.num_replicas - (len(indices) % self.num_replicas)
                    ) % self.num_replicas
                    rank_len = (len(indices) + rem) // self.num_replicas
                    n_batches = rank_len // self.batch_size
                    if rank_len % self.batch_size > 0:
                        n_batches += 1

                if use_fingerprint:
                    ch, fp = key
                    logger.info(
                        f"  桶 [{ch}ch, fp={fp}]: {len(indices)} 样本, 估计 {n_batches} batches"
                    )
                else:
                    logger.info(f"  桶 [{key} ch]: {len(indices)} 样本, 估计 {n_batches} batches")

        if self.rank == 0 and cluster_by_file:
            n_h5_files = len(set(self._idx_to_h5.values()) - {""})
            if n_h5_files > 0:
                logger.info(f"  I/O 聚簇优化: 按 {n_h5_files} 个 HDF5 文件聚簇排列")
        if self.rank == 0 and self.file_scheduler:
            logger.info(
                "  文件级调度: 启用（同一 batch 只来自一个 HDF5 文件，"
                f"整文件 block 连续消费，文件内{'打乱' if self.shuffle_within_file else '顺序'}采样）"
            )

    def _cluster_shuffle(self, indices: List[int], g: torch.Generator) -> List[int]:
        """
        按 HDF5 文件聚簇打乱：受试者级随机 + 受试者内随机

        步骤：
        1. 按 hdf5_path 分组（同一受试者的样本归为一组）
        2. 组内打乱（受试者内样本随机排列）
        3. 打乱组的顺序（受试者间随机排列）
        4. 串联所有组为线性序列

        结果：连续样本大概率来自同一/相邻受试者（同一 HDF5 文件），
        切 batch 后每个 batch 通常只涉及 1~2 个 HDF5 文件而非 32 个。
        """
        groups: Dict[str, List[int]] = defaultdict(list)
        for idx in indices:
            h5_key = self._idx_to_h5.get(idx, "")
            groups[h5_key].append(idx)

        group_keys = sorted(groups.keys())

        # 打乱组顺序
        gperm = torch.randperm(len(group_keys), generator=g).tolist()
        group_keys = [group_keys[i] for i in gperm]

        result = []
        for gk in group_keys:
            # 先按 hdf5_idx 排序，优先顺序读取；再根据配置决定是否组内打乱
            group_indices = sorted(groups[gk], key=lambda i: self._idx_to_h5_idx.get(i, -1))
            if self.shuffle and self.shuffle_within_file:
                iperm = torch.randperm(len(group_indices), generator=g).tolist()
                result.extend(group_indices[i] for i in iperm)
            else:
                result.extend(group_indices)

        return result

    def _build_default_bucket_batches(
        self,
        ids_bucket: List[int],
        g: torch.Generator,
        total_batch_size: int,
    ) -> List[List[int]]:
        """沿用原始分布式分桶逻辑（样本级分配）。"""
        if self.shuffle:
            if self.cluster_by_file and self._idx_to_h5:
                ids_bucket = self._cluster_shuffle(ids_bucket, g)
            else:
                perm = torch.randperm(len(ids_bucket), generator=g).tolist()
                ids_bucket = [ids_bucket[i] for i in perm]

        rank_batches: List[List[int]] = []
        if self.drop_last:
            rem = (total_batch_size - (len(ids_bucket) % total_batch_size)) % total_batch_size
            if rem > 0:
                ids_bucket.extend(ids_bucket[:rem])

            ids_rank = ids_bucket[self.rank :: self.num_replicas]
            for start in range(0, len(ids_rank), self.batch_size):
                batch = ids_rank[start : start + self.batch_size]
                if len(batch) < self.batch_size:
                    continue
                rank_batches.append(batch)
        else:
            rem = (self.num_replicas - (len(ids_bucket) % self.num_replicas)) % self.num_replicas
            if rem > 0:
                ids_bucket.extend(ids_bucket[:rem])

            ids_rank = ids_bucket[self.rank :: self.num_replicas]
            for start in range(0, len(ids_rank), self.batch_size):
                batch = ids_rank[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                rank_batches.append(batch)

        return rank_batches

    @staticmethod
    def _truncate_file_blocks(
        file_blocks: List[List[List[int]]],
        target_batches: int,
    ) -> List[List[List[int]]]:
        """按 batch 数裁剪文件块列表，尽量保留整块连续消费。"""
        if target_batches <= 0:
            return []

        kept_blocks: List[List[List[int]]] = []
        remaining = target_batches
        for block in file_blocks:
            if remaining <= 0:
                break
            if len(block) <= remaining:
                kept_blocks.append(block)
                remaining -= len(block)
                continue
            kept_blocks.append(block[:remaining])
            remaining = 0
            break
        return kept_blocks

    def _build_file_scheduled_bucket_blocks(
        self,
        ids_bucket: List[int],
        g: torch.Generator,
        total_batch_size: int,
    ) -> List[List[List[int]]]:
        """
        文件级调度：先按 HDF5 文件分组，再把整文件 batch block 分配给 rank。

        目的：
        1. 同一 batch 只来自一个 HDF5 文件
        2. 同一文件在当前 rank 上尽量连续消费，符合块级缓存语义
        3. 降低同一时刻跨 rank 争抢同一文件的概率
        """
        groups: Dict[str, List[int]] = defaultdict(list)
        for idx in ids_bucket:
            groups[self._idx_to_h5.get(idx, "")].append(idx)

        # 缺少文件信息时退回默认策略
        if not groups:
            return [
                [batch]
                for batch in self._build_default_bucket_batches(ids_bucket, g, total_batch_size)
            ]

        group_keys = sorted(groups.keys())
        if self.shuffle:
            gperm = torch.randperm(len(group_keys), generator=g).tolist()
            group_keys = [group_keys[i] for i in gperm]

        # 文件数过少时，文件级分配会让部分 rank 没有 batch，退回默认策略
        if len(group_keys) < self.num_replicas:
            return [
                [batch]
                for batch in self._build_default_bucket_batches(ids_bucket, g, total_batch_size)
            ]

        file_batches: List[Tuple[str, List[List[int]]]] = []
        for gk in group_keys:
            file_indices = sorted(groups[gk], key=lambda i: self._idx_to_h5_idx.get(i, -1))
            if self.shuffle and self.shuffle_within_file:
                iperm = torch.randperm(len(file_indices), generator=g).tolist()
                file_indices = [file_indices[i] for i in iperm]

            if self.drop_last:
                usable = (len(file_indices) // self.batch_size) * self.batch_size
                file_indices = file_indices[:usable]

            batches: List[List[int]] = []
            for start in range(0, len(file_indices), self.batch_size):
                batch = file_indices[start : start + self.batch_size]
                if len(batch) < self.batch_size:
                    continue
                batches.append(batch)

            if batches:
                file_batches.append((gk, batches))

        if not file_batches:
            return [
                [batch]
                for batch in self._build_default_bucket_batches(ids_bucket, g, total_batch_size)
            ]

        # 按文件可贡献的 batch 数做负载均衡，减少后续 min_batches 截断造成的样本浪费。
        # 先随机化原始顺序（训练时）再按 batch 数降序稳定排序，可保留同规模文件间的随机性。
        file_batches.sort(key=lambda item: len(item[1]), reverse=True)

        rank_file_blocks: List[List[List[List[int]]]] = [[] for _ in range(self.num_replicas)]
        rank_batch_counts = [0] * self.num_replicas
        for _, batches in file_batches:
            owner_rank = min(
                range(self.num_replicas),
                key=lambda rank_idx: (rank_batch_counts[rank_idx], rank_idx),
            )
            rank_file_blocks[owner_rank].append(batches)
            rank_batch_counts[owner_rank] += len(batches)

        min_batches = min(rank_batch_counts)
        if min_batches <= 0:
            # 某些小桶可能出现极端不均衡，退回默认策略保证训练不挂
            return [
                [batch]
                for batch in self._build_default_bucket_batches(ids_bucket, g, total_batch_size)
            ]

        truncated_rank_blocks = [
            self._truncate_file_blocks(blocks, min_batches) for blocks in rank_file_blocks
        ]
        return truncated_rank_blocks[self.rank]

    def _get_batch_hdf5_path(self, batch: List[int]) -> str:
        """返回 batch 共享的 HDF5 路径；若 batch 混入多个文件则返回空串。"""
        if not batch:
            return ""

        h5_path = self._idx_to_h5.get(batch[0], "")
        if not h5_path:
            return ""

        for idx in batch[1:]:
            if self._idx_to_h5.get(idx, "") != h5_path:
                return ""
        return h5_path

    def _build_epoch_batches(
        self,
        epoch: int,
    ) -> Tuple[List[List[int]], List[str], List[Dict[str, Any]]]:
        """构建指定 epoch 的 batch 列表、去重文件计划和 rank 级窗口计划。"""
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)

        all_batches: List[List[int]] = []
        all_file_blocks: List[List[List[int]]] = []
        total_batch_size = self.num_replicas * self.batch_size

        for key in sorted(self.buckets.keys(), key=lambda x: str(x)):
            ids_bucket = self.buckets[key].copy()

            if self.file_scheduler and self.cluster_by_file and self._idx_to_h5:
                bucket_file_blocks = self._build_file_scheduled_bucket_blocks(
                    ids_bucket, g, total_batch_size
                )
                all_file_blocks.extend(bucket_file_blocks)
            else:
                bucket_batches = self._build_default_bucket_batches(ids_bucket, g, total_batch_size)
                all_batches.extend(bucket_batches)

        if self.file_scheduler and self.cluster_by_file and self._idx_to_h5:
            if self.shuffle and all_file_blocks:
                perm = torch.randperm(len(all_file_blocks), generator=g).tolist()
                all_file_blocks = [all_file_blocks[i] for i in perm]
            all_batches = []
            for file_block in all_file_blocks:
                all_batches.extend(file_block)
        elif self.shuffle:
            if self.cluster_by_file:
                # 按块打乱：保持 I/O 局部性的同时提供训练随机性
                chunk_size = max(1, self.batch_size // 2)
                chunks = []
                for i in range(0, len(all_batches), chunk_size):
                    chunks.append(all_batches[i : i + chunk_size])
                cperm = torch.randperm(len(chunks), generator=g).tolist()
                all_batches = []
                for ci in cperm:
                    all_batches.extend(chunks[ci])
            else:
                perm = torch.randperm(len(all_batches), generator=g).tolist()
                all_batches = [all_batches[i] for i in perm]

        # 预取计划：按 batch 消费顺序提取去重后的 HDF5 文件序列
        file_plan: List[str] = []
        seen: set = set()
        rank_schedule: List[Dict[str, Any]] = []
        current_h5_path = ""
        current_start_idx = -1
        current_batches = 0

        def flush_rank_schedule_segment() -> None:
            nonlocal current_h5_path, current_start_idx, current_batches
            if not current_h5_path or current_start_idx < 0 or current_batches <= 0:
                current_h5_path = ""
                current_start_idx = -1
                current_batches = 0
                return
            rank_schedule.append(
                {
                    "hdf5_path": current_h5_path,
                    "n_batches": current_batches,
                    "first_batch_idx": current_start_idx,
                    "last_batch_idx": current_start_idx + current_batches - 1,
                }
            )
            current_h5_path = ""
            current_start_idx = -1
            current_batches = 0

        for batch in all_batches:
            if not batch:
                continue
            h5_path = self._get_batch_hdf5_path(batch)
            if h5_path and h5_path not in seen:
                seen.add(h5_path)
                file_plan.append(h5_path)
        for batch_idx, batch in enumerate(all_batches):
            if not batch:
                flush_rank_schedule_segment()
                continue
            h5_path = self._get_batch_hdf5_path(batch)
            if not self.file_scheduler or not h5_path:
                flush_rank_schedule_segment()
                continue
            if h5_path != current_h5_path:
                flush_rank_schedule_segment()
                current_h5_path = h5_path
                current_start_idx = batch_idx
                current_batches = 1
            else:
                current_batches += 1
        flush_rank_schedule_segment()

        return all_batches, file_plan, rank_schedule

    def _refresh_cached_plan(self, epoch: int) -> None:
        if self._cached_epoch == epoch:
            return
        batches, file_plan, rank_schedule = self._build_epoch_batches(epoch)
        self._cached_epoch = epoch
        self._cached_batches = batches
        self._cached_file_plan = file_plan
        self._cached_rank_schedule = rank_schedule

    def __iter__(self) -> Iterator[List[int]]:
        """生成当前 rank 的 batch 索引列表"""
        self._refresh_cached_plan(self.epoch)
        yield from self._cached_batches

    def __len__(self) -> int:
        return self._total_batches

    def set_epoch(self, epoch: int) -> None:
        """设置 epoch（用于分布式训练时同步随机种子）"""
        self.epoch = epoch
        self._refresh_cached_plan(epoch)
        self._total_batches = len(self._cached_batches)

    def get_prefetch_file_plan(self, max_files: int = 0) -> List[str]:
        """
        返回当前 epoch 的 HDF5 预取计划（按消费顺序去重）。

        参数:
            max_files: >0 时仅返回前 max_files 个文件；<=0 返回全量
        """
        self._refresh_cached_plan(self.epoch)
        if max_files > 0:
            return self._cached_file_plan[:max_files]
        return list(self._cached_file_plan)

    def get_prefetch_rank_schedule(self) -> List[Dict[str, Any]]:
        """
        返回当前 rank 的 HDF5 文件消费窗口计划。

        仅在 file_scheduler=true 时返回窗口列表；否则返回空列表。
        """
        if not self.file_scheduler:
            return []
        self._refresh_cached_plan(self.epoch)
        return [dict(item) for item in self._cached_rank_schedule]


class PENCICollator:
    """
    PENCI 数据集的批处理器

    将一批样本整理成模型输入格式。
    支持两种模式：
    1. 固定通道数模式（target_channels 不为 None）：直接 stack
    2. 动态通道数模式（配合 BucketBatchSampler）：
       同一 batch 内通道数相同，直接 stack

    batch 输出额外包含 fingerprint 字段（来自第一个样本的 metadata）。
    """

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """
        返回:
            批处理后的字典:
            - x: (B, C, T)
            - pos: (B, C, 6)
            - sensor_type: (B, C)
            - n_channels: (B,) 每个样本的原始通道数
            - metadata: 元数据列表
            - fingerprint: 当前 batch 的电极指纹（来自第一个样本）
        """
        x = torch.stack([item["x"] for item in batch])
        pos = torch.stack([item["pos"] for item in batch])
        sensor_type = torch.stack([item["sensor_type"] for item in batch])
        n_channels = torch.stack([item["n_channels"] for item in batch])

        fingerprint = batch[0]["metadata"].get("fingerprint", "unknown")

        return {
            "x": x,
            "pos": pos,
            "sensor_type": sensor_type,
            "n_channels": n_channels,
            "metadata": [item["metadata"] for item in batch],
            "fingerprint": fingerprint,
        }


def _worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker 初始化函数

    fork 出的子进程会继承父进程的 dataset 对象，但 HDF5 文件句柄
    在 fork 后不可用（POSIX 文件锁语义）。需要清除父进程的缓存，
    让每个 worker 重新打开自己的文件句柄。
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if isinstance(dataset, PENCIDataset):
            dataset.close_h5_handles()


def create_dataloader(
    metadata_path: Optional[str] = None,
    metadata: Optional[List[Dict]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    use_bucket_sampler: bool = False,
    use_fingerprint: bool = False,
    rank: int = 0,
    world_size: int = 1,
    **dataset_kwargs,
) -> DataLoader:
    """
    创建数据加载器

    参数:
        metadata_path: 元数据文件路径（与 metadata 二选一）
        metadata: 已加载的元数据列表（与 metadata_path 二选一）
        batch_size: 批大小
        num_workers: 工作进程数
        shuffle: 是否打乱
        use_bucket_sampler: 是否使用 BucketBatchSampler（动态通道数模式）
        use_fingerprint: 是否使用电极指纹分桶（需配合 use_bucket_sampler=True）
        rank: 当前进程 rank（分布式训练）
        world_size: 总进程数（分布式训练）
        **dataset_kwargs: 传递给 PENCIDataset 的其他参数

    返回:
        DataLoader 实例
    """
    if use_bucket_sampler:
        dataset_kwargs.setdefault("target_channels", None)
    if use_fingerprint:
        dataset_kwargs.setdefault("precompute_fingerprints", True)
    dataset_kwargs.setdefault("rank", rank)
    sampler_file_scheduler = bool(dataset_kwargs.pop("sampler_file_scheduler", False))
    sampler_shuffle_within_file = bool(dataset_kwargs.pop("sampler_shuffle_within_file", True))

    dataset = PENCIDataset(
        metadata_path=metadata_path,
        metadata=metadata,
        **dataset_kwargs,
    )
    collator = PENCICollator()

    if use_bucket_sampler:
        sampler = DistributedBucketBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
            use_fingerprint=use_fingerprint,
            file_scheduler=sampler_file_scheduler,
            shuffle_within_file=sampler_shuffle_within_file,
        )
        loader_kwargs = {
            "num_workers": num_workers,
            "collate_fn": collator,
            "pin_memory": True,
            "worker_init_fn": _worker_init_fn,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            **loader_kwargs,
        )
    else:
        sampler: Optional[DistributedSampler] = None
        dataloader_shuffle = shuffle
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=True,
            )
            dataloader_shuffle = False

        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": dataloader_shuffle,
            "num_workers": num_workers,
            "collate_fn": collator,
            "pin_memory": True,
            "drop_last": True,
            "worker_init_fn": _worker_init_fn,
        }
        if sampler is not None:
            loader_kwargs["sampler"] = sampler
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True

        return DataLoader(
            dataset,
            **loader_kwargs,
        )


def get_train_val_loaders(
    data_root: str = "/work/2024/tanzunsheng/PENCIData",
    datasets: Optional[List[str]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    use_bucket_sampler: bool = False,
    use_fingerprint: bool = False,
    rank: int = 0,
    world_size: int = 1,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    获取训练和验证数据加载器（支持多数据集合并加载）

    遍历 datasets 列表中的每个数据集，通过 _resolve_metadata_dir() 定位其
    metadata 目录，读取 train.json / val.json 并合并为统一的 metadata 列表，
    再传递给 create_dataloader() 构建 DataLoader。

    参数:
        data_root: 数据根目录
        datasets: 数据集名称列表（如 ["HBN_EEG", "Broderick2018_NaturalSpeech", ...]）。
                  默认为 ["HBN_EEG"]。
        batch_size: 批大小
        num_workers: 工作进程数
        use_bucket_sampler: 是否使用 BucketBatchSampler
        use_fingerprint: 是否使用电极指纹分桶（需配合 use_bucket_sampler=True）
        rank: 当前进程 rank（分布式训练）
        world_size: 总进程数（分布式训练）
        **dataset_kwargs: 传递给 PENCIDataset 的其他参数
                          注意: datasets 参数不再通过 dataset_kwargs 传递给
                          PENCIDataset 做二次过滤，因为合并加载已精确控制来源。

    返回:
        (train_loader, val_loader) 元组
    """
    if datasets is None:
        datasets = ["HBN_EEG"]

    # 从 dataset_kwargs 中移除 datasets，避免传递给 PENCIDataset 做冗余过滤
    dataset_kwargs.pop("datasets", None)

    all_train_meta: List[Dict] = []
    all_val_meta: List[Dict] = []

    for ds_name in datasets:
        metadata_dir = _resolve_metadata_dir(data_root, ds_name)
        train_path = os.path.join(metadata_dir, "train.json")
        val_path = os.path.join(metadata_dir, "val.json")

        if not os.path.isfile(train_path):
            logger.warning(f"数据集 '{ds_name}' 缺少 train.json: {train_path}，跳过")
            continue
        if not os.path.isfile(val_path):
            logger.warning(f"数据集 '{ds_name}' 缺少 val.json: {val_path}，跳过")
            continue

        with open(train_path, "r") as f:
            train_meta = json.load(f)
        with open(val_path, "r") as f:
            val_meta = json.load(f)

        all_train_meta.extend(train_meta)
        all_val_meta.extend(val_meta)
        logger.info(
            f"数据集 '{ds_name}': 训练 {len(train_meta)} 样本, "
            f"验证 {len(val_meta)} 样本 (来自 {metadata_dir})"
        )

    if not all_train_meta:
        raise RuntimeError(f"没有加载到任何训练样本。datasets={datasets}, data_root={data_root}")

    logger.info(
        f"合并完成: 训练 {len(all_train_meta)} 样本, "
        f"验证 {len(all_val_meta)} 样本 (共 {len(datasets)} 个数据集)"
    )

    # 验证集支持独立的采样调度参数；这些键不应透传给训练集 dataset 构造。
    val_sampler_file_scheduler = dataset_kwargs.pop("val_sampler_file_scheduler", None)
    val_sampler_shuffle_within_file = dataset_kwargs.pop("val_sampler_shuffle_within_file", None)
    val_random_crop = dataset_kwargs.pop("val_random_crop", False)

    train_loader_kwargs = dict(dataset_kwargs)
    val_loader_kwargs = dict(dataset_kwargs)
    if val_sampler_file_scheduler is None:
        val_sampler_file_scheduler = False
    if val_sampler_shuffle_within_file is None:
        val_sampler_shuffle_within_file = False
    val_loader_kwargs["sampler_file_scheduler"] = bool(val_sampler_file_scheduler)
    val_loader_kwargs["sampler_shuffle_within_file"] = bool(val_sampler_shuffle_within_file)
    val_loader_kwargs["random_crop"] = bool(val_random_crop)

    train_loader = create_dataloader(
        metadata=all_train_meta,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        use_bucket_sampler=use_bucket_sampler,
        use_fingerprint=use_fingerprint,
        rank=rank,
        world_size=world_size,
        **train_loader_kwargs,
    )

    val_loader = create_dataloader(
        metadata=all_val_meta,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        use_bucket_sampler=use_bucket_sampler,
        use_fingerprint=use_fingerprint,
        rank=rank,
        world_size=world_size,
        **val_loader_kwargs,
    )

    return train_loader, val_loader


# 数据增强变换
class RandomScaling:
    """随机缩放增强"""

    def __init__(self, min_scale: float = 0.8, max_scale: float = 1.2):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.empty(1).uniform_(self.min_scale, self.max_scale).item()
        return x * scale


class RandomNoise:
    """随机噪声增强"""

    def __init__(self, noise_std: float = 0.01):
        self.noise_std = noise_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.noise_std
        return x + noise


class Compose:
    """组合多个变换"""

    def __init__(self, transforms: List[callable]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x
