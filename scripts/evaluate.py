#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PENCI 独立评估脚本

三层级评估体系:
  层级 1 — 传感器空间重建指标 (Pearson / SNR / NRMSE)
  层级 2 — 源空间对比 sLORETA (区域级 Pearson 相关)
  层级 3 — 仿真数据 DLE (Dipole Localization Error)

用法:
    # 全部评估
    python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt

    # 仅传感器空间指标
    python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt --eval_mode sensor

    # 仅仿真 DLE
    python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt --eval_mode simulation

依赖:
    - MNE-Python >= 1.0 (层级 2 & 3)
    - fsaverage 数据 (层级 2 & 3)
"""

import os
import sys
import json
import argparse
import logging
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from penci.models import PENCI, build_penci_from_config
from penci.data import get_train_val_loaders
from penci.physics import SourceSpace, ElectrodeConfigRegistry
from penci.physics.leadfield_manager import LeadfieldManager
from penci.utils.metrics import (
    pearson_correlation,
    signal_to_noise_ratio,
    normalized_rmse,
    compute_all_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_from_checkpoint(
    config: dict,
    checkpoint_path: str,
    device: torch.device,
) -> PENCI:
    """
    从检查点加载模型

    参数:
        config: 完整配置字典
        checkpoint_path: 检查点文件路径
        device: 计算设备

    返回:
        加载了权重的模型 (eval 模式)
    """
    model = build_penci_from_config(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # 处理 DDP 权重 (去除 "module." 前缀)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        cleaned_state_dict[new_key] = value

    model.load_state_dict(cleaned_state_dict)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型已加载: {total_params:,} 参数, 设备: {device}")
    return model


def setup_physics(config: dict) -> Tuple[
    Optional[SourceSpace],
    Optional[LeadfieldManager],
    Optional[ElectrodeConfigRegistry],
]:
    """
    初始化物理约束组件 (与 train.py 中相同的逻辑)

    参数:
        config: 完整配置字典

    返回:
        (source_space, leadfield_manager, electrode_registry) 元组
    """
    physics_cfg = config.get("model", {}).get("physics", {})
    use_fixed_leadfield = physics_cfg.get("use_fixed_leadfield", True)
    leadfield_path = physics_cfg.get("leadfield_path", None)

    if not use_fixed_leadfield or leadfield_path is not None:
        logger.info("物理约束模式: 静态导联场或注意力模式，跳过动态导联场初始化")
        return None, None, None

    global_physics = config.get("physics", {})
    subjects_dir = global_physics.get("subjects_dir")
    cache_dir = global_physics.get("leadfield_cache_dir")
    registry_path = global_physics.get("fingerprint_registry_path")

    if not subjects_dir or not cache_dir:
        raise RuntimeError(
            "评估需要 physics.subjects_dir 和 physics.leadfield_cache_dir 配置"
        )

    source_space = SourceSpace(subjects_dir=subjects_dir)
    leadfield_manager = LeadfieldManager(
        source_space=source_space,
        subjects_dir=subjects_dir,
        cache_dir=cache_dir,
    )

    if registry_path is not None:
        electrode_registry = ElectrodeConfigRegistry.load_from_archive(registry_path)
        logger.info(
            f"电极配置注册表: {len(electrode_registry.get_all_fingerprints())} 个指纹"
        )
    else:
        processed_data_dir = global_physics.get("processed_data_dir")
        if not processed_data_dir:
            raise RuntimeError(
                "评估需要 physics.fingerprint_registry_path 或 physics.processed_data_dir"
            )
        electrode_registry = ElectrodeConfigRegistry(processed_data_dir)
        datasets = config.get("data", {}).get("datasets", [])
        for ds_name in datasets:
            try:
                electrode_registry.register_dataset(ds_name)
            except FileNotFoundError as e:
                logger.warning(f"注册数据集 '{ds_name}' 电极配置失败: {e}")

    return source_space, leadfield_manager, electrode_registry


def resolve_leadfield_for_batch(
    fingerprint: str,
    leadfield_manager: LeadfieldManager,
    electrode_registry: ElectrodeConfigRegistry,
    device: torch.device,
) -> torch.Tensor:
    """根据电极指纹解析导联场矩阵"""
    channel_names, channel_positions = electrode_registry.get_config_by_fingerprint(
        fingerprint
    )
    return leadfield_manager.get_leadfield(channel_names, channel_positions, device)


# ═══════════════════════════════════════════════════════════════════
# 层级 1: 传感器空间重建指标
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_sensor_space(
    model: PENCI,
    dataloader,
    device: torch.device,
    config: dict,
    leadfield_manager: Optional[LeadfieldManager] = None,
    electrode_registry: Optional[ElectrodeConfigRegistry] = None,
) -> Dict[str, float]:
    """
    层级 1: 传感器空间重建质量评估

    对测试集逐 batch 运行模型推理，计算:
    - Pearson 相关系数 (波形相似度)
    - SNR (dB) (信噪比)
    - NRMSE (归一化均方根误差)
    - MSE 重建损失

    参数:
        model: PENCI 模型 (eval 模式)
        dataloader: 测试数据加载器
        device: 计算设备
        config: 配置字典
        leadfield_manager: 导联场管理器
        electrode_registry: 电极配置注册表

    返回:
        指标字典 {"pearson": float, "snr_db": float, "nrmse": float, "recon_loss": float}
    """
    logger.info("=" * 60)
    logger.info("层级 1: 传感器空间重建指标")
    logger.info("=" * 60)

    model.eval()
    loss_weights = config.get("training", {}).get("loss", {})

    total_pearson = 0.0
    total_snr = 0.0
    total_nrmse = 0.0
    total_recon_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        x = batch["x"].to(device)
        pos = batch["pos"].to(device)
        sensor_type = batch["sensor_type"].to(device)

        leadfield = None
        if leadfield_manager is not None and electrode_registry is not None:
            fingerprint = batch.get("fingerprint", "unknown")
            leadfield = resolve_leadfield_for_batch(
                fingerprint, leadfield_manager, electrode_registry, device,
            )

        # 前向传播
        output = model(x, pos, sensor_type, leadfield=leadfield, return_source=True)
        reconstruction = output["reconstruction"]

        # 准备目标 (下采样输入以匹配输出长度)
        target = model._prepare_target(x, reconstruction.shape[-1])

        # 计算指标
        metrics = compute_all_metrics(target, reconstruction)
        recon_loss = F.mse_loss(reconstruction, target)

        total_pearson += metrics["pearson"].item()
        total_snr += metrics["snr_db"].item()
        total_nrmse += metrics["nrmse"].item()
        total_recon_loss += recon_loss.item()
        num_batches += 1

        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"  进度: [{batch_idx + 1}/{len(dataloader)}] "
                f"Pearson={metrics['pearson'].item():.4f} "
                f"SNR={metrics['snr_db'].item():.1f}dB "
                f"NRMSE={metrics['nrmse'].item():.4f}"
            )

    if num_batches == 0:
        logger.warning("没有可用的评估数据")
        return {"pearson": 0.0, "snr_db": 0.0, "nrmse": 0.0, "recon_loss": 0.0}

    results = {
        "pearson": total_pearson / num_batches,
        "snr_db": total_snr / num_batches,
        "nrmse": total_nrmse / num_batches,
        "recon_loss": total_recon_loss / num_batches,
    }

    logger.info(f"传感器空间指标:")
    logger.info(f"  Pearson 相关系数: {results['pearson']:.4f}")
    logger.info(f"  SNR:              {results['snr_db']:.2f} dB")
    logger.info(f"  NRMSE:            {results['nrmse']:.4f}")
    logger.info(f"  重建损失 (MSE):   {results['recon_loss']:.6f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# 层级 2: 源空间对比 sLORETA
# ═══════════════════════════════════════════════════════════════════

def _build_mne_forward_and_inverse(
    config: dict,
    channel_names: List[str],
    channel_positions: np.ndarray,
    eval_config: dict,
) -> Tuple:
    """
    构建 MNE 正向模型和 sLORETA 逆算子

    参数:
        config: 完整配置字典
        channel_names: 电极名称列表
        channel_positions: 电极位置 (N, 3) 米制
        eval_config: 评估配置段

    返回:
        (fwd, inv_op, info, src) 元组
    """
    import mne
    from mne.minimum_norm import make_inverse_operator

    subjects_dir = config.get("physics", {}).get("subjects_dir")
    subject = "fsaverage"

    # 创建 Info 对象
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=config.get("data", {}).get("sample_rate", 256),
        ch_types=["eeg"] * len(channel_names),
    )

    # 设置电极位置
    montage_positions = {}
    for i, name in enumerate(channel_names):
        montage_positions[name] = channel_positions[i]

    montage = mne.channels.make_dig_montage(
        ch_pos=montage_positions,
        coord_frame="head",
    )
    info.set_montage(montage)

    # 设置 EEG 参考投影
    info = mne.io.RawArray(
        np.zeros((len(channel_names), 1)),
        info,
    ).set_eeg_reference(projection=True).info

    # 正向模型
    src_path = str(
        Path(subjects_dir) / subject / "bem" / f"{subject}-ico-5-src.fif"
    )
    bem_path = str(
        Path(subjects_dir) / subject / "bem"
        / f"{subject}-5120-5120-5120-bem-sol.fif"
    )

    fwd = mne.make_forward_solution(
        info,
        trans=subject,
        src=src_path,
        bem=bem_path,
        eeg=True,
        meg=False,
        mindist=5.0,
        n_jobs=1,
        verbose=False,
    )

    # 噪声协方差 (ad-hoc)
    noise_cov = mne.make_ad_hoc_cov(info)

    # 逆算子
    source_config = eval_config.get("source_comparison", {})
    inv_op = make_inverse_operator(
        info,
        fwd,
        noise_cov,
        loose=source_config.get("loose", 0.2),
        depth=source_config.get("depth", 0.8),
        verbose=False,
    )

    src = fwd["src"]

    return fwd, inv_op, info, src


def _aggregate_stc_to_regions(
    stc_data: np.ndarray,
    stc_vertices: list,
    source_space: SourceSpace,
    subjects_dir: str,
    subject: str = "fsaverage",
) -> np.ndarray:
    """
    将顶点级 SourceEstimate 数据聚合到 72 个脑区

    sLORETA 输出 ~20484 顶点 → 按 DK68 parcellation 聚合到 68 个皮层区域
    + 4 个皮层下区域设为 0 (sLORETA 仅估计皮层源)

    参数:
        stc_data: (n_vertices, n_times) 源活动数据
        stc_vertices: [lh_vertices, rh_vertices] 顶点索引
        source_space: PENCI 源空间定义
        subjects_dir: FreeSurfer subjects 目录
        subject: 被试名称

    返回:
        (72, n_times) 区域级源活动
    """
    import mne

    # 读取 DK68 标签
    labels = mne.read_labels_from_annot(
        subject=subject,
        parc="aparc",
        subjects_dir=subjects_dir,
    )
    cortical_labels = [
        label for label in labels if "unknown" not in label.name.lower()
    ]

    # 构建顶点到索引的映射
    n_lh = len(stc_vertices[0])
    lh_vert_set = set(stc_vertices[0])
    rh_vert_set = set(stc_vertices[1])

    # 顶点号 -> stc_data 行索引
    lh_vert_to_idx = {v: i for i, v in enumerate(stc_vertices[0])}
    rh_vert_to_idx = {v: i + n_lh for i, v in enumerate(stc_vertices[1])}

    n_times = stc_data.shape[1]
    region_activity = np.zeros((72, n_times))

    # 按 source_space 中的顺序对齐 DK68 标签
    source_names = source_space.names
    cortical_label_dict = {label.name: label for label in cortical_labels}

    for i, name in enumerate(source_names[:68]):
        # 尝试精确匹配
        label = cortical_label_dict.get(name)
        if label is None:
            # 尝试模糊匹配 (去除半球后缀后匹配)
            for lname, lobj in cortical_label_dict.items():
                if lname.rstrip("-lh").rstrip("-rh") == name.rstrip("-lh").rstrip("-rh"):
                    label = lobj
                    break

        if label is None:
            continue

        # 获取该标签中属于 stc 的顶点
        if label.hemi == "lh":
            vert_to_idx = lh_vert_to_idx
        else:
            vert_to_idx = rh_vert_to_idx

        label_indices = []
        for v in label.vertices:
            idx = vert_to_idx.get(v)
            if idx is not None:
                label_indices.append(idx)

        if label_indices:
            region_activity[i] = np.mean(
                np.abs(stc_data[label_indices, :]), axis=0
            )

    # 皮层下 4 个区域: sLORETA 无法估计，保持为 0
    # (PENCI 的 source_activity 对应的 indices 68-71)

    return region_activity


@torch.no_grad()
def evaluate_source_space(
    model: PENCI,
    dataloader,
    device: torch.device,
    config: dict,
    source_space: SourceSpace,
    leadfield_manager: Optional[LeadfieldManager] = None,
    electrode_registry: Optional[ElectrodeConfigRegistry] = None,
    max_batches: int = 20,
) -> Dict[str, float]:
    """
    层级 2: 源空间对比 sLORETA

    在真实测试数据上比较 PENCI 源估计与 sLORETA 逆解:
    - 将 PENCI 72 区域输出与 sLORETA 聚合到 72 区域的结果做 Pearson 相关
    - 由于 sLORETA 无法估计皮层下源, 仅比较 68 个皮层区域

    参数:
        model: PENCI 模型 (eval 模式)
        dataloader: 测试数据加载器
        device: 计算设备
        config: 配置字典
        source_space: PENCI 源空间
        leadfield_manager: 导联场管理器
        electrode_registry: 电极配置注册表
        max_batches: 最大评估 batch 数 (sLORETA 较慢)

    返回:
        指标字典 {"source_pearson_cortical": float, "n_evaluated": int}
    """
    import mne
    from mne.minimum_norm import apply_inverse

    logger.info("=" * 60)
    logger.info("层级 2: 源空间对比 sLORETA")
    logger.info("=" * 60)

    model.eval()
    eval_config = config.get("evaluation", {})
    source_config = eval_config.get("source_comparison", {})
    lambda2 = source_config.get("lambda2", 1.0 / 9.0)
    subjects_dir = config.get("physics", {}).get("subjects_dir")
    sample_rate = config.get("data", {}).get("sample_rate", 256)

    all_correlations = []
    fwd_cache = {}  # 缓存不同电极配置的正向模型

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        x = batch["x"].to(device)
        pos = batch["pos"].to(device)
        sensor_type = batch["sensor_type"].to(device)
        fingerprint = batch.get("fingerprint", "unknown")

        leadfield = None
        if leadfield_manager is not None and electrode_registry is not None:
            leadfield = resolve_leadfield_for_batch(
                fingerprint, leadfield_manager, electrode_registry, device,
            )

        # --- PENCI 源估计 ---
        output = model(x, pos, sensor_type, leadfield=leadfield, return_source=True)
        penci_source = output["source_activity"]  # (B, 72, T', D)
        # 取各脑区时间序列的模 (沿特征维度取范数) -> (B, 72, T')
        penci_region = torch.norm(penci_source, dim=-1).cpu().numpy()

        # --- sLORETA 逆解 (逐样本) ---
        x_np = x.cpu().float().numpy()

        # 获取或构建正向模型
        if fingerprint not in fwd_cache:
            try:
                if electrode_registry is not None:
                    ch_names, ch_positions = electrode_registry.get_config_by_fingerprint(fingerprint)
                    fwd, inv_op, info, src = _build_mne_forward_and_inverse(
                        config, ch_names, ch_positions.numpy(), eval_config,
                    )
                    fwd_cache[fingerprint] = (fwd, inv_op, info, src)
                else:
                    logger.warning(f"无电极配置注册表，跳过 sLORETA 对比")
                    continue
            except Exception as e:
                logger.warning(f"构建正向模型失败 (fingerprint={fingerprint}): {e}")
                continue

        fwd, inv_op, info, src = fwd_cache[fingerprint]

        for sample_idx in range(x_np.shape[0]):
            try:
                # 创建 Evoked 对象
                evoked = mne.EvokedArray(
                    x_np[sample_idx],  # (C, T)
                    info,
                    tmin=0.0,
                    verbose=False,
                )

                # 应用 sLORETA
                stc = apply_inverse(
                    evoked, inv_op, lambda2=lambda2,
                    method="sLORETA", pick_ori=None, verbose=False,
                )

                # 聚合到 72 区域
                sloreta_region = _aggregate_stc_to_regions(
                    stc.data, stc.vertices, source_space, subjects_dir,
                )

                # PENCI 对应样本的 72 区域
                penci_sample = penci_region[sample_idx]  # (72, T')

                # 对齐时间维度 (取较短的)
                min_t = min(sloreta_region.shape[1], penci_sample.shape[1])
                sloreta_region = sloreta_region[:68, :min_t]  # 仅皮层
                penci_cortical = penci_sample[:68, :min_t]

                # 逐区域 Pearson 相关系数
                correlations = []
                for r in range(68):
                    s = sloreta_region[r]
                    p = penci_cortical[r]
                    if np.std(s) > 1e-10 and np.std(p) > 1e-10:
                        corr = np.corrcoef(s, p)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)

                if correlations:
                    all_correlations.append(np.mean(correlations))

            except Exception as e:
                logger.debug(f"sLORETA 处理样本 {sample_idx} 失败: {e}")
                continue

        logger.info(
            f"  进度: [{batch_idx + 1}/{min(len(dataloader), max_batches)}] "
            f"已评估 {len(all_correlations)} 个样本"
        )

    if not all_correlations:
        logger.warning("没有成功完成的 sLORETA 对比")
        return {"source_pearson_cortical": 0.0, "n_evaluated": 0}

    mean_corr = float(np.mean(all_correlations))
    std_corr = float(np.std(all_correlations))

    results = {
        "source_pearson_cortical": mean_corr,
        "source_pearson_std": std_corr,
        "n_evaluated": len(all_correlations),
    }

    logger.info(f"源空间对比指标:")
    logger.info(f"  皮层区域 Pearson: {mean_corr:.4f} +/- {std_corr:.4f}")
    logger.info(f"  评估样本数:       {len(all_correlations)}")

    return results


# ═══════════════════════════════════════════════════════════════════
# 层级 3: 仿真数据 DLE (Dipole Localization Error)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_simulation(
    model: PENCI,
    device: torch.device,
    config: dict,
    source_space: SourceSpace,
    electrode_registry: Optional[ElectrodeConfigRegistry] = None,
) -> Dict[str, float]:
    """
    层级 3: 仿真数据偶极子定位误差 (DLE)

    流程:
    1. 在已知位置放置偶极子，用 MNE 正向模型生成仿真 EEG
    2. 分别用 PENCI 和 sLORETA 估计源
    3. 计算 DLE (估计峰值到真实位置的欧氏距离, mm)

    PENCI 输出 72 区域 → 取 peak region 质心坐标
    sLORETA 输出 ~20484 顶点 → 取 peak 顶点坐标

    参数:
        model: PENCI 模型 (eval 模式)
        device: 计算设备
        config: 配置字典
        source_space: PENCI 源空间
        electrode_registry: 电极配置注册表 (用于获取电极配置)

    返回:
        指标字典 {"dle_penci_mm": float, "dle_sloreta_mm": float, ...}
    """
    import mne
    from mne.minimum_norm import make_inverse_operator, apply_inverse
    from mne.simulation import simulate_sparse_stc, simulate_evoked

    logger.info("=" * 60)
    logger.info("层级 3: 仿真数据 DLE (Dipole Localization Error)")
    logger.info("=" * 60)

    model.eval()
    eval_config = config.get("evaluation", {})
    sim_config = eval_config.get("simulation", {})
    n_trials = sim_config.get("n_trials", 50)
    n_dipoles = sim_config.get("n_dipoles", 1)
    nave = sim_config.get("nave", 20)
    source_config = eval_config.get("source_comparison", {})
    lambda2 = source_config.get("lambda2", 1.0 / 9.0)

    subjects_dir = config.get("physics", {}).get("subjects_dir")
    sample_rate = config.get("data", {}).get("sample_rate", 256)

    # 获取一组电极配置 (用第一个已注册的指纹)
    if electrode_registry is None:
        logger.error("仿真评估需要电极配置注册表")
        return {"dle_penci_mm": float("nan"), "dle_sloreta_mm": float("nan")}

    all_fps = electrode_registry.get_all_fingerprints()
    if not all_fps:
        logger.error("没有可用的电极配置")
        return {"dle_penci_mm": float("nan"), "dle_sloreta_mm": float("nan")}

    # 使用第一个电极配置
    fp = all_fps[0]
    ch_names, ch_positions = electrode_registry.get_config_by_fingerprint(fp)
    ch_names = list(ch_names)
    ch_pos_np = ch_positions.numpy() if hasattr(ch_positions, 'numpy') else np.array(ch_positions)
    n_channels = len(ch_names)

    logger.info(f"仿真使用电极配置: {n_channels} 通道, 指纹={fp}")

    # 构建 MNE 正向模型
    fwd, inv_op, info, src = _build_mne_forward_and_inverse(
        config, ch_names, ch_pos_np, eval_config,
    )

    # 仿真时间设置
    duration = 0.6  # 600ms
    times = np.arange(int(sample_rate * duration)) / sample_rate - 0.1

    def data_fun(times: np.ndarray) -> np.ndarray:
        """仿真偶极子时间函数: 30Hz 正弦波 × 高斯包络"""
        return 50e-9 * np.sin(30.0 * times) * np.exp(
            -((times - 0.15) ** 2) / 0.01
        )

    noise_cov = mne.make_ad_hoc_cov(info)

    # PENCI 源区域质心坐标 (米)
    region_positions = source_space.positions  # (72, 3) 米

    dles_penci = []
    dles_sloreta = []

    for trial in range(n_trials):
        try:
            # 1. 生成仿真数据
            stc_true = simulate_sparse_stc(
                src,
                n_dipoles=n_dipoles,
                times=times,
                random_state=trial,
                data_fun=data_fun,
            )
            evoked_sim = simulate_evoked(
                fwd, stc_true, info, noise_cov,
                nave=nave,
                verbose=False,
            )

            # 真实源位置 (取第一个偶极子)
            true_vertex = None
            true_pos = None
            for hemi_idx in range(2):
                if len(stc_true.vertices[hemi_idx]) > 0:
                    true_vertex = stc_true.vertices[hemi_idx][0]
                    true_pos = src[hemi_idx]["rr"][true_vertex]  # 米
                    break

            if true_pos is None:
                continue

            # 2. sLORETA 估计
            stc_sloreta = apply_inverse(
                evoked_sim, inv_op, lambda2=lambda2,
                method="sLORETA", pick_ori=None, verbose=False,
            )

            # sLORETA peak 位置
            n_lh_sloreta = len(stc_sloreta.vertices[0])
            peak_power = np.mean(stc_sloreta.data ** 2, axis=1)
            peak_idx = np.argmax(peak_power)
            if peak_idx < n_lh_sloreta:
                peak_vert = stc_sloreta.vertices[0][peak_idx]
                peak_pos_sloreta = src[0]["rr"][peak_vert]
            else:
                peak_vert = stc_sloreta.vertices[1][peak_idx - n_lh_sloreta]
                peak_pos_sloreta = src[1]["rr"][peak_vert]

            dle_sloreta = np.linalg.norm(true_pos - peak_pos_sloreta) * 1000.0  # mm
            dles_sloreta.append(dle_sloreta)

            # 3. PENCI 估计
            # 将仿真 EEG 转为模型输入
            eeg_data = evoked_sim.data  # (C, T)
            eeg_tensor = torch.from_numpy(eeg_data).float().unsqueeze(0).to(device)  # (1, C, T)

            # 位置和传感器类型
            pos_6d = np.zeros((n_channels, 6))
            pos_6d[:, :3] = ch_pos_np
            pos_6d[:, 3:] = [0, 0, 1]  # 默认法向量
            pos_tensor = torch.from_numpy(pos_6d).float().unsqueeze(0).to(device)
            sensor_type_tensor = torch.zeros(1, n_channels, dtype=torch.long, device=device)

            # 获取导联场
            from penci.physics.leadfield_manager import LeadfieldManager as LM
            leadfield = resolve_leadfield_for_batch(
                fp,
                LeadfieldManager(
                    source_space=source_space,
                    subjects_dir=subjects_dir,
                    cache_dir=config.get("physics", {}).get("leadfield_cache_dir"),
                ),
                electrode_registry,
                device,
            )

            output = model(
                eeg_tensor, pos_tensor, sensor_type_tensor,
                leadfield=leadfield, return_source=True,
            )
            penci_source = output["source_activity"]  # (1, 72, T', D)

            # 取各区域能量: (72,)
            region_energy = torch.norm(penci_source[0], dim=-1).mean(dim=-1).cpu().numpy()

            # 取 peak region
            peak_region_idx = np.argmax(region_energy)
            peak_pos_penci = region_positions[peak_region_idx]  # 米

            dle_penci = np.linalg.norm(true_pos - peak_pos_penci) * 1000.0  # mm
            dles_penci.append(dle_penci)

            if (trial + 1) % 10 == 0:
                logger.info(
                    f"  试验 {trial + 1}/{n_trials}: "
                    f"DLE(PENCI)={dle_penci:.1f}mm, "
                    f"DLE(sLORETA)={dle_sloreta:.1f}mm"
                )

        except Exception as e:
            logger.debug(f"仿真试验 {trial} 失败: {e}")
            continue

    if not dles_penci:
        logger.warning("没有成功完成的仿真试验")
        return {
            "dle_penci_mm": float("nan"),
            "dle_sloreta_mm": float("nan"),
            "n_trials_completed": 0,
        }

    results = {
        "dle_penci_mm": float(np.mean(dles_penci)),
        "dle_penci_std_mm": float(np.std(dles_penci)),
        "dle_penci_median_mm": float(np.median(dles_penci)),
        "dle_sloreta_mm": float(np.mean(dles_sloreta)),
        "dle_sloreta_std_mm": float(np.std(dles_sloreta)),
        "dle_sloreta_median_mm": float(np.median(dles_sloreta)),
        "n_trials_completed": len(dles_penci),
    }

    logger.info(f"仿真评估结果 ({len(dles_penci)}/{n_trials} 试验成功):")
    logger.info(
        f"  PENCI  DLE:   {results['dle_penci_mm']:.1f} +/- "
        f"{results['dle_penci_std_mm']:.1f} mm "
        f"(中位数: {results['dle_penci_median_mm']:.1f} mm)"
    )
    logger.info(
        f"  sLORETA DLE:  {results['dle_sloreta_mm']:.1f} +/- "
        f"{results['dle_sloreta_std_mm']:.1f} mm "
        f"(中位数: {results['dle_sloreta_median_mm']:.1f} mm)"
    )

    return results


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PENCI 独立评估脚本")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="模型检查点路径",
    )
    parser.add_argument(
        "--eval_mode", type=str, default="all",
        choices=["all", "sensor", "source", "simulation"],
        help="评估模式: all=全部, sensor=传感器空间, source=源空间对比, simulation=仿真DLE",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="评估结果输出目录",
    )
    parser.add_argument(
        "--max_source_batches", type=int, default=20,
        help="源空间对比最大 batch 数 (sLORETA 较慢)",
    )
    parser.add_argument(
        "--n_simulation_trials", type=int, default=None,
        help="仿真试验次数 (覆盖配置)",
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 覆盖仿真试验次数
    if args.n_simulation_trials is not None:
        config.setdefault("evaluation", {}).setdefault("simulation", {})
        config["evaluation"]["simulation"]["n_trials"] = args.n_simulation_trials

    # 输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"评估设备: {device}")

    # 加载模型
    model = load_model_from_checkpoint(config, args.checkpoint, device)

    # 物理组件
    source_space, leadfield_manager, electrode_registry = setup_physics(config)

    # 数据加载器 (使用验证集作为测试集)
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    configured_datasets = data_config.get("datasets", ["HBN_EEG"])

    use_bucket_sampler = data_config.get("use_bucket_sampler", False)
    use_fingerprint = leadfield_manager is not None

    _, val_loader = get_train_val_loaders(
        data_root=data_config.get("root_dir", "/work/2024/tanzunsheng/PENCIData"),
        datasets=configured_datasets,
        batch_size=training_config.get("batch_size", 32),
        num_workers=training_config.get("num_workers", 4),
        use_bucket_sampler=use_bucket_sampler,
        use_fingerprint=use_fingerprint,
        max_length=data_config.get("time_window", 10) * data_config.get("sample_rate", 256),
        target_channels=data_config.get("n_channels", 128) if not use_bucket_sampler else None,
    )
    logger.info(f"评估数据集大小: {len(val_loader.dataset)}")

    # 收集所有结果
    all_results = {}

    # === 层级 1: 传感器空间 ===
    if args.eval_mode in ("all", "sensor"):
        sensor_results = evaluate_sensor_space(
            model, val_loader, device, config,
            leadfield_manager=leadfield_manager,
            electrode_registry=electrode_registry,
        )
        all_results["sensor_space"] = sensor_results

    # === 层级 2: 源空间对比 ===
    if args.eval_mode in ("all", "source"):
        if source_space is None:
            logger.warning("源空间未初始化，跳过层级 2 评估")
        else:
            try:
                source_results = evaluate_source_space(
                    model, val_loader, device, config, source_space,
                    leadfield_manager=leadfield_manager,
                    electrode_registry=electrode_registry,
                    max_batches=args.max_source_batches,
                )
                all_results["source_space"] = source_results
            except ImportError as e:
                logger.warning(f"MNE-Python 未安装，跳过层级 2: {e}")
            except Exception as e:
                logger.error(f"层级 2 评估失败: {e}")

    # === 层级 3: 仿真 DLE ===
    if args.eval_mode in ("all", "simulation"):
        if source_space is None:
            logger.warning("源空间未初始化，跳过层级 3 评估")
        else:
            try:
                sim_results = evaluate_simulation(
                    model, device, config, source_space,
                    electrode_registry=electrode_registry,
                )
                all_results["simulation"] = sim_results
            except ImportError as e:
                logger.warning(f"MNE-Python 未安装，跳过层级 3: {e}")
            except Exception as e:
                logger.error(f"层级 3 评估失败: {e}")

    # === 保存结果 ===
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"评估结果已保存: {results_path}")

    # 打印最终摘要
    logger.info("=" * 60)
    logger.info("评估摘要")
    logger.info("=" * 60)

    if "sensor_space" in all_results:
        s = all_results["sensor_space"]
        logger.info(
            f"  [传感器空间] Pearson={s['pearson']:.4f} "
            f"SNR={s['snr_db']:.2f}dB NRMSE={s['nrmse']:.4f}"
        )

    if "source_space" in all_results:
        s = all_results["source_space"]
        logger.info(
            f"  [源空间对比] 皮层 Pearson={s['source_pearson_cortical']:.4f} "
            f"(n={s['n_evaluated']})"
        )

    if "simulation" in all_results:
        s = all_results["simulation"]
        logger.info(
            f"  [仿真 DLE] PENCI={s['dle_penci_mm']:.1f}mm "
            f"sLORETA={s['dle_sloreta_mm']:.1f}mm "
            f"(n={s['n_trials_completed']})"
        )

    logger.info(f"详细结果: {results_path}")


if __name__ == "__main__":
    main()
