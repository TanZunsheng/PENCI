# -*- coding: utf-8 -*-
"""
共享动态导联场辅助函数。
"""

import logging
from typing import Optional, Tuple

import torch

from penci.physics import ElectrodeConfigRegistry, SourceSpace
from penci.physics.leadfield_manager import LeadfieldManager

logger = logging.getLogger(__name__)


def resolve_leadfield_for_batch(
    fingerprint: str,
    leadfield_manager: LeadfieldManager,
    electrode_registry: ElectrodeConfigRegistry,
    device: torch.device,
) -> torch.Tensor:
    channel_names, channel_positions = electrode_registry.get_config_by_fingerprint(
        fingerprint
    )
    return leadfield_manager.get_leadfield(channel_names, channel_positions, device)


def setup_physics(config: dict) -> Tuple[
    Optional[SourceSpace],
    Optional[LeadfieldManager],
    Optional[ElectrodeConfigRegistry],
    bool,
]:
    physics_cfg = config.get("model", {}).get("physics", {})
    use_fixed_leadfield = physics_cfg.get("use_fixed_leadfield", True)
    leadfield_path = physics_cfg.get("leadfield_path", None)

    if not use_fixed_leadfield or leadfield_path is not None:
        logger.info("物理约束模式: 静态导联场或注意力模式，跳过动态导联场初始化")
        return None, None, None, False

    global_physics = config.get("physics", {})
    subjects_dir = global_physics.get("subjects_dir")
    cache_dir = global_physics.get("leadfield_cache_dir")
    processed_data_dir = global_physics.get("processed_data_dir")
    registry_path = global_physics.get("fingerprint_registry_path")

    if not subjects_dir or not cache_dir:
        raise RuntimeError(
            "动态导联场模式需要 physics.subjects_dir 和 physics.leadfield_cache_dir 配置"
        )

    logger.info("初始化物理约束组件...")
    source_space = SourceSpace(subjects_dir=subjects_dir)
    logger.info("  源空间: %s 个源", source_space.get_source_info()["n_total"])

    leadfield_manager = LeadfieldManager(
        source_space=source_space,
        subjects_dir=subjects_dir,
        cache_dir=cache_dir,
    )

    if registry_path is not None:
        logger.info("从离线存档加载电极配置注册表: %s", registry_path)
        electrode_registry = ElectrodeConfigRegistry.load_from_archive(registry_path)
        logger.info(
            "  已加载 %d 个唯一电极指纹",
            len(electrode_registry.get_all_fingerprints()),
        )
        return source_space, leadfield_manager, electrode_registry, True

    if not processed_data_dir:
        raise RuntimeError(
            "运行时扫描模式需要 physics.processed_data_dir 配置，"
            "或设置 physics.fingerprint_registry_path 使用离线存档"
        )

    logger.info("未配置 fingerprint_registry_path，使用运行时扫描模式")
    electrode_registry = ElectrodeConfigRegistry(processed_data_dir)
    datasets = config.get("data", {}).get("datasets", [])
    for ds_name in datasets:
        try:
            electrode_registry.register_dataset(ds_name)
        except FileNotFoundError as exc:
            logger.warning("注册数据集 '%s' 电极配置失败: %s", ds_name, exc)

    logger.info("  已注册电极配置: %s", electrode_registry.registered_configs)
    return source_space, leadfield_manager, electrode_registry, False
