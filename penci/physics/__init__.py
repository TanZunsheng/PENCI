# -*- coding: utf-8 -*-
"""
物理约束模块

提供基于物理模型的导联场计算和管理功能：
- SourceSpace: 72 个源空间定义 (DK68 皮层脑区 + 4 个皮层下结构)
- ElectrodeUtils: 从 ProcessedData 读取电极坐标并复制 BrainOmniPostProcess 的通道过滤逻辑
- LeadfieldManager: 使用 MNE 计算导联场矩阵，支持 hash 缓存和文件锁
"""

from penci.physics.source_space import SourceSpace
from penci.physics.electrode_utils import (
    read_electrodes_tsv,
    filter_channels_like_postprocess,
    find_electrodes_tsv,
    get_valid_channels_for_dataset,
    ElectrodeConfigRegistry,
)

__all__ = [
    "SourceSpace",
    "read_electrodes_tsv",
    "filter_channels_like_postprocess",
    "find_electrodes_tsv",
    "get_valid_channels_for_dataset",
    "ElectrodeConfigRegistry",
]

# 延迟导入 LeadfieldManager（依赖 MNE，且文件可能尚未创建）
def __getattr__(name):
    if name == "LeadfieldManager":
        from penci.physics.leadfield_manager import LeadfieldManager
        return LeadfieldManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
