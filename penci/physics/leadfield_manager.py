# -*- coding: utf-8 -*-
"""
导联场管理器模块

核心功能：
1. 使用 MNE 前向模型计算导联场矩阵 (n_channels, 72)
2. 基于通道配置 hash 的两级缓存（内存 LRU + 磁盘文件）
3. 文件锁防止多进程同时计算相同配置
4. 为 PhysicsDecoder 提供导联场张量

技术决策：
- 导联场使用 fixed orientation（法线方向），输出 (n_channels, n_sources)
- 缓存 key 基于排序的通道名 + 坐标的 hash
- 无 MNE 环境直接报错，不允许 fallback 到随机矩阵
- fsaverage 不存在直接报错，不自动下载

数据流：
    electrodes.tsv (米制坐标)
         │
         │ electrode_utils.read_electrodes_tsv + filter
         ▼
    channel_names + channel_positions
         │
         │ LeadfieldManager.get_leadfield()
         ▼
    (n_channels, 72) 导联场张量 (torch.Tensor)
"""

import hashlib
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _compute_channel_hash(
    channel_names: List[str],
    channel_positions: np.ndarray,
    precision_mm: float = 0.1,
) -> str:
    """
    计算通道配置的唯一 hash（电极指纹）

    基于排序后的通道名和量化坐标，确保相同配置始终产生相同 hash。
    坐标量化到 precision_mm 精度（默认 0.1mm），避免 float32/float64 转换
    带来的浮点噪声导致缓存失效。

    参数:
        channel_names: 通道名列表
        channel_positions: (n_channels, 3) 坐标数组（米制）
        precision_mm: 坐标量化精度（毫米），默认 0.1mm

    返回:
        16 字符的十六进制 hash 字符串
    """
    # 按通道名排序以确保一致性（使用 zip + sorted 更明确）
    sorted_pairs = sorted(
        zip(channel_names, channel_positions), key=lambda x: x[0]
    )
    sorted_names = [p[0] for p in sorted_pairs]
    sorted_positions = np.array([p[1] for p in sorted_pairs])

    # 构建 hash 输入
    hasher = hashlib.sha256()

    # 通道名
    for name in sorted_names:
        hasher.update(name.encode("utf-8"))

    # 坐标量化：米制 -> 整数（单位 = precision_mm）
    # 例如 precision_mm=0.1 时，乘以 10000（1m = 1000mm / 0.1mm = 10000 单位）
    scale = 1000.0 / precision_mm  # 米 -> 量化单位
    pos_int = np.round(sorted_positions * scale).astype(np.int32)

    # 包含 shape 和 dtype 信息防止碰撞（如 (64,3) vs (192,1) 同 tobytes）
    hasher.update(str(pos_int.shape).encode("utf-8"))
    hasher.update(str(pos_int.dtype).encode("utf-8"))
    hasher.update(pos_int.tobytes())

    return hasher.hexdigest()[:16]


def compute_fingerprint_from_pos(
    positions: np.ndarray,
    precision_mm: float = 0.1,
) -> str:
    """
    仅从坐标数组计算电极指纹（不含通道名）

    用于从 .pt 样本的 pos 张量计算指纹，以便在数据集层面做分桶。
    .pt 文件中不包含通道名，仅有 pos: (C, 6) 张量。

    计算方式与 _compute_channel_hash 的坐标部分一致：
    按行排序 -> 量化 -> SHA-256 前 16 位。

    参数:
        positions: (n_channels, 3) xyz 坐标数组（米制）
        precision_mm: 坐标量化精度（毫米），默认 0.1mm

    返回:
        16 字符的十六进制 hash 字符串
    """
    # 按行排序（先 x，再 y，再 z）确保顺序无关
    sorted_idx = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
    sorted_positions = positions[sorted_idx]

    # 量化
    scale = 1000.0 / precision_mm
    pos_int = np.round(sorted_positions * scale).astype(np.int32)

    hasher = hashlib.sha256()
    hasher.update(str(pos_int.shape).encode("utf-8"))
    hasher.update(str(pos_int.dtype).encode("utf-8"))
    hasher.update(pos_int.tobytes())

    return hasher.hexdigest()[:16]


class LeadfieldManager:
    """
    导联场管理器

    负责导联场矩阵的计算、缓存和检索。

    使用流程:
        1. 初始化时传入 SourceSpace、subjects_dir 和 cache_dir
        2. 调用 get_leadfield(channel_names, channel_positions) 获取导联场
        3. 首次调用会使用 MNE 计算并缓存，后续调用直接从缓存返回

    缓存策略:
        - 内存级: OrderedDict LRU 缓存（最大条目数可配置）
        - 磁盘级: {cache_dir}/{hash}.pt 文件
        - 文件锁: {cache_dir}/{hash}.lock 防止多进程竞争

    参数:
        source_space: SourceSpace 实例，定义 72 个源
        subjects_dir: FreeSurfer subjects 目录路径
        cache_dir: 导联场缓存目录路径
        max_memory_cache: 内存缓存最大条目数
        subject: FreeSurfer 被试名称，默认 'fsaverage'
    """

    def __init__(
        self,
        source_space,  # SourceSpace 实例，延迟类型检查以避免循环导入
        subjects_dir: str,
        cache_dir: str,
        max_memory_cache: int = 16,
        subject: str = "fsaverage",
    ):
        self._source_space = source_space
        self._subjects_dir = Path(subjects_dir)
        self._cache_dir = Path(cache_dir)
        self._max_memory_cache = max_memory_cache
        self._subject = subject

        # 创建缓存目录
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # 内存 LRU 缓存: hash -> torch.Tensor
        self._memory_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        logger.info(
            f"LeadfieldManager 初始化:\n"
            f"  subjects_dir: {self._subjects_dir}\n"
            f"  cache_dir: {self._cache_dir}\n"
            f"  max_memory_cache: {max_memory_cache}"
        )

    def get_leadfield(
        self,
        channel_names: List[str],
        channel_positions: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        获取导联场矩阵

        参数:
            channel_names: 通道名列表，长度 n_channels
            channel_positions: (n_channels, 3) 米制坐标数组
            device: 输出张量的设备（默认 CPU）

        返回:
            torch.Tensor: (n_channels, 72) 导联场矩阵，dtype=float32

        异常:
            ImportError: 未安装 MNE
            FileNotFoundError: fsaverage 不存在
            RuntimeError: MNE 前向计算失败
        """
        n_channels = len(channel_names)
        assert channel_positions.shape == (n_channels, 3), (
            f"坐标形状错误: {channel_positions.shape}，预期 ({n_channels}, 3)"
        )

        # 1. 计算 hash
        config_hash = _compute_channel_hash(channel_names, channel_positions)

        # 2. 检查内存缓存
        if config_hash in self._memory_cache:
            logger.debug(f"内存缓存命中: {config_hash}")
            # 移动到最近使用位置
            self._memory_cache.move_to_end(config_hash)
            tensor = self._memory_cache[config_hash]
            if device is not None:
                tensor = tensor.to(device)
            return tensor

        # 3. 检查磁盘缓存
        cache_file = self._cache_dir / f"{config_hash}.pt"
        if cache_file.exists():
            logger.info(f"磁盘缓存命中: {cache_file}")
            tensor = torch.load(cache_file, weights_only=True)
            self._put_memory_cache(config_hash, tensor)
            if device is not None:
                tensor = tensor.to(device)
            return tensor

        # 4. 缓存未命中 — 用 MNE 计算
        logger.info(
            f"缓存未命中，开始计算导联场 "
            f"(hash={config_hash}, n_channels={n_channels})..."
        )
        tensor = self._compute_leadfield(channel_names, channel_positions)

        # 5. 写入磁盘缓存（带文件锁）
        self._save_to_disk(config_hash, tensor)

        # 6. 写入内存缓存
        self._put_memory_cache(config_hash, tensor)

        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def get_leadfield_for_batch(
        self,
        channel_names: List[str],
        channel_positions: np.ndarray,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        获取批量导联场矩阵

        同一 batch 内通道数和通道配置相同（由 BucketBatchSampler 保证），
        因此只需计算一次导联场并扩展到 batch 维度。

        参数:
            channel_names: 通道名列表
            channel_positions: (n_channels, 3) 坐标
            batch_size: batch 大小
            device: 输出张量的设备

        返回:
            torch.Tensor: (batch_size, n_channels, 72) 导联场矩阵
        """
        # 获取单个导联场 (n_channels, 72)
        leadfield = self.get_leadfield(channel_names, channel_positions, device)
        # 扩展到 batch 维度 (B, n_channels, 72)
        return leadfield.unsqueeze(0).expand(batch_size, -1, -1)

    def _compute_leadfield(
        self,
        channel_names: List[str],
        channel_positions: np.ndarray,
    ) -> torch.Tensor:
        """
        使用 MNE 计算导联场矩阵

        步骤:
        1. 创建 MNE Info（通道信息 + 坐标 montage）
        2. 获取混合源空间（来自 SourceSpace）
        3. 读取 BEM 模型
        4. 计算前向解 (make_forward_solution)
        5. 转为面向表面法线的自由方向（surf_ori=True, force_fixed=False）
           注意：混合源空间不能使用 force_fixed=True（体积源无法线）
        6. 从完整源空间中提取 72 个脑区对应的导联场列
           - 皮层源: 取法线方向分量的平均
           - 体积源: 取 L2 范数（3 方向向量幅度）的平均

        参数:
            channel_names: 通道名列表
            channel_positions: (n_channels, 3) 坐标（米制）

        返回:
            torch.Tensor: (n_channels, 72) 导联场矩阵
        """
        try:
            import mne
        except ImportError:
            raise ImportError(
                "需要安装 MNE-Python 来计算导联场: pip install mne\n"
                "不允许 fallback 到随机矩阵。"
            )

        subjects_dir = str(self._subjects_dir)
        subject = self._subject
        t0 = time.time()

        # === 1. 创建 MNE Info ===
        info = mne.create_info(
            ch_names=list(channel_names),
            sfreq=256.0,  # 采样率对导联场无影响，使用默认值
            ch_types="eeg",
        )

        # 创建 montage
        ch_pos = {
            name: channel_positions[i]
            for i, name in enumerate(channel_names)
        }
        montage = mne.channels.make_dig_montage(
            ch_pos=ch_pos, coord_frame="head"
        )
        info.set_montage(montage)

        # === 2. 获取源空间 ===
        # 使用 SourceSpace 提供的混合源空间
        src = self._source_space.src
        logger.info(f"  源空间: {len(src)} 个空间")

        # === 3. 读取 BEM ===
        bem_path = (
            self._subjects_dir / subject / "bem"
            / f"{subject}-5120-5120-5120-bem-sol.fif"
        )
        if not bem_path.exists():
            bem_path = (
                self._subjects_dir / subject / "bem"
                / f"{subject}-5120-bem-sol.fif"
            )
        if not bem_path.exists():
            bem_dir = self._subjects_dir / subject / "bem"
            available = list(bem_dir.glob("*-bem-sol.fif")) if bem_dir.exists() else []
            raise FileNotFoundError(
                f"BEM 解决方案文件不存在: {bem_path}\n"
                f"可用文件: {[f.name for f in available]}"
            )
        bem = mne.read_bem_solution(str(bem_path), verbose=False)

        # === 4. 计算前向解 ===
        logger.info("  计算前向解...")
        fwd = mne.make_forward_solution(
            info,
            trans="fsaverage",  # fsaverage 的 trans 是恒等变换
            src=src,
            bem=bem,
            eeg=True,
            meg=False,
            verbose=False,
        )

        # === 5. 转为面向表面法线方向（自由方向） ===
        # 注意：混合源空间（surface + volume）不能使用 force_fixed=True，
        # 因为体积源没有有意义的法线方向。
        # 使用 surf_ori=True 将皮层源旋转到表面法线坐标系（第一个分量=法线方向），
        # 体积源保持 xyz 三个自由方向。
        fwd_surf_ori = mne.convert_forward_solution(
            fwd, surf_ori=True, force_fixed=False, verbose=False
        )

        # 自由方向导联场: (n_channels, n_total_sources * 3)
        L_full = fwd_surf_ori["sol"]["data"]
        logger.info(f"  完整导联场形状（自由方向）: {L_full.shape}")

        # === 6. 提取 72 个脑区质心对应的导联场列 ===
        L_72 = self._extract_region_leadfield(fwd_surf_ori, src)

        elapsed = time.time() - t0
        logger.info(
            f"  导联场计算完成: ({L_72.shape[0]}, {L_72.shape[1]})，"
            f"耗时 {elapsed:.1f}s"
        )

        return torch.from_numpy(L_72).float()

    def _extract_region_leadfield(self, fwd_surf_ori, src) -> np.ndarray:
        """
        从完整前向解（自由方向）中提取 72 个脑区的导联场列

        导联场矩阵为自由方向 (n_channels, n_total_sources * 3)，
        使用 surf_ori=True 后每个源点占 3 列：
        - 皮层源: col[3i+0]=切向1, col[3i+1]=切向2, col[3i+2]=法线方向
        - 体积源: col[3i+0]=X, col[3i+1]=Y, col[3i+2]=Z

        提取策略:
        - 68 个皮层标签: 取 col[3i+2]（法线方向），对标签内源点取平均
        - 4 个皮层下结构: 取 sqrt(Lx²+Ly²+Lz²)（L2 范数），对源点取平均

        参数:
            fwd_surf_ori: MNE 自由方向前向解（surf_ori=True, force_fixed=False）
            src: MNE 混合源空间

        返回:
            np.ndarray: (n_channels, 72) 导联场矩阵
        """
        import mne

        L_full = fwd_surf_ori["sol"]["data"]  # (n_channels, n_total_sources * 3)
        n_channels = L_full.shape[0]

        # 读取 DK68 标签
        labels = mne.read_labels_from_annot(
            subject=self._subject,
            parc="aparc",
            subjects_dir=str(self._subjects_dir),
        )
        cortical_labels = [
            label for label in labels if "unknown" not in label.name.lower()
        ]

        # 获取皮层源空间的顶点编号
        vertno_lh = src[0]["vertno"]
        vertno_rh = src[1]["vertno"]

        n_lh = len(vertno_lh)
        n_rh = len(vertno_rh)

        # 自由方向下每个源点占 3 列
        # 构建顶点到源索引的映射（导联场列 = src_idx * 3 + orientation_offset）
        vert_to_src_idx_lh = {v: i for i, v in enumerate(vertno_lh)}
        vert_to_src_idx_rh = {v: i + n_lh for i, v in enumerate(vertno_rh)}

        L_72 = np.zeros((n_channels, 72), dtype=np.float64)

        # === 提取 68 个皮层标签的导联场 ===
        # surf_ori=True 后法线方向在第 3 分量（偏移量 +2）
        # 参考: MNE convert_forward_solution 中 force_fixed 分支取 source_nn[2::3]
        for idx, label in enumerate(cortical_labels):
            if label.hemi == "lh":
                vert_to_src_idx = vert_to_src_idx_lh
            else:
                vert_to_src_idx = vert_to_src_idx_rh

            # 收集该标签中属于源空间的顶点的法线方向导联场列
            normal_col_indices = []
            for v in label.vertices:
                if v in vert_to_src_idx:
                    src_idx = vert_to_src_idx[v]
                    # 法线方向 = 每个源点 3 列中的第 2 列（偏移 +2）
                    normal_col_indices.append(src_idx * 3 + 2)

            if len(normal_col_indices) == 0:
                logger.warning(
                    f"标签 '{label.name}' 没有源空间中的顶点，使用零向量"
                )
                continue

            # 取所有源点法线方向导联场的平均
            L_72[:, idx] = L_full[:, normal_col_indices].mean(axis=1)

        # === 提取 4 个皮层下结构的导联场 ===
        # 体积源空间从 src[2] 开始，源索引偏移量为皮层源总数
        src_offset = n_lh + n_rh
        for sub_idx, sub_src in enumerate(src[2:]):
            n_active = sub_src["nuse"]

            if n_active == 0:
                logger.warning(
                    f"皮层下源空间 {sub_idx} 没有活跃源点，使用零向量"
                )
                continue

            # 体积源无法线方向，取 3 方向导联场的 L2 范数（向量幅度）
            # 向量化提取: 构建所有源点的 3 方向列索引
            base_cols = np.arange(n_active) + src_offset
            col_x = base_cols * 3        # X 方向列
            col_y = base_cols * 3 + 1    # Y 方向列
            col_z = base_cols * 3 + 2    # Z 方向列

            # (n_channels, n_active) 每个方向
            Lx = L_full[:, col_x]
            Ly = L_full[:, col_y]
            Lz = L_full[:, col_z]

            # L2 范数: sqrt(Lx² + Ly² + Lz²)，形状 (n_channels, n_active)
            magnitudes = np.sqrt(Lx ** 2 + Ly ** 2 + Lz ** 2)

            # 取所有源点幅度的平均
            L_72[:, 68 + sub_idx] = magnitudes.mean(axis=1)

            src_offset += n_active

        return L_72

    def _put_memory_cache(self, key: str, tensor: torch.Tensor) -> None:
        """写入内存 LRU 缓存"""
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
        else:
            self._memory_cache[key] = tensor.cpu()
            # 淘汰最旧条目
            while len(self._memory_cache) > self._max_memory_cache:
                evicted_key, _ = self._memory_cache.popitem(last=False)
                logger.debug(f"内存缓存淘汰: {evicted_key}")

    def _save_to_disk(self, config_hash: str, tensor: torch.Tensor) -> None:
        """
        写入磁盘缓存（带简单文件锁）

        使用 .lock 文件防止多进程同时写入同一缓存文件。
        """
        cache_file = self._cache_dir / f"{config_hash}.pt"
        lock_file = self._cache_dir / f"{config_hash}.lock"

        # 简单文件锁：尝试创建 .lock 文件
        max_retries = 30
        for attempt in range(max_retries):
            try:
                # O_CREAT | O_EXCL 保证原子性创建
                import os
                fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                # 其他进程正在写入，等待
                if attempt < max_retries - 1:
                    logger.debug(
                        f"等待文件锁: {lock_file} (尝试 {attempt + 1}/{max_retries})"
                    )
                    time.sleep(1.0)
                else:
                    # 超时，强制删除锁（可能是残留锁）
                    logger.warning(f"文件锁超时，强制删除: {lock_file}")
                    try:
                        lock_file.unlink()
                    except OSError:
                        pass
                    return

        try:
            # 再次检查缓存文件（可能其他进程已经写入）
            if not cache_file.exists():
                # 先写入临时文件，再原子重命名
                tmp_file = self._cache_dir / f"{config_hash}.tmp"
                torch.save(tensor.cpu(), str(tmp_file))
                tmp_file.rename(cache_file)
                logger.info(f"导联场缓存已保存: {cache_file}")
            else:
                logger.debug(f"缓存文件已存在（其他进程已写入）: {cache_file}")
        finally:
            # 释放文件锁
            try:
                lock_file.unlink()
            except OSError:
                pass

    def clear_memory_cache(self) -> None:
        """清空内存缓存"""
        self._memory_cache.clear()
        logger.info("内存缓存已清空")

    @property
    def cache_dir(self) -> Path:
        """缓存目录路径"""
        return self._cache_dir

    @property
    def memory_cache_size(self) -> int:
        """当前内存缓存条目数"""
        return len(self._memory_cache)

    def get_cached_hashes(self) -> List[str]:
        """列出磁盘缓存中已有的 hash"""
        return [
            f.stem for f in self._cache_dir.glob("*.pt")
            if not f.name.endswith(".tmp")
        ]
