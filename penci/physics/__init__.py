# -*- coding: utf-8 -*-
"""
物理约束模块

提供基于物理模型的导联场计算和管理功能：
- SourceSpace: 72 个源空间定义 (DK68 皮层脑区 + 4 个皮层下结构)
- ElectrodeUtils: 从 ProcessedData 读取电极坐标并复制 BrainOmniPostProcess 的通道过滤逻辑
- LeadfieldManager: 使用 MNE 计算导联场矩阵，支持 hash 缓存和文件锁
- compute_fingerprint_from_pos: 从坐标数组计算电极位置指纹
"""

from penci.physics.source_space import SourceSpace
from penci.physics.electrode_utils import (
    read_electrodes_tsv,
    filter_channels_like_postprocess,
    find_electrodes_tsv,
    get_valid_channels_for_dataset,
    ElectrodeConfigRegistry,
)
from penci.physics.leadfield_manager import compute_fingerprint_from_pos

__all__ = [
    "SourceSpace",
    "read_electrodes_tsv",
    "filter_channels_like_postprocess",
    "find_electrodes_tsv",
    "get_valid_channels_for_dataset",
    "ElectrodeConfigRegistry",
    "compute_fingerprint_from_pos",
    "load_registry_from_archive",
]


def load_registry_from_archive(archive_path: str) -> "ElectrodeConfigRegistry":
    """
    便捷函数：从离线预计算存档加载 ElectrodeConfigRegistry

    等价于 ElectrodeConfigRegistry.load_from_archive(archive_path)

    参数:
        archive_path: 存档文件路径 (.pt)

    返回:
        已填充的 ElectrodeConfigRegistry 实例
    """
    return ElectrodeConfigRegistry.load_from_archive(archive_path)

# 延迟导入 LeadfieldManager（依赖 MNE，且文件可能尚未创建）
def __getattr__(name):
    if name == "LeadfieldManager":
        from penci.physics.leadfield_manager import LeadfieldManager
        return LeadfieldManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
