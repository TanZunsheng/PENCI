# -*- coding: utf-8 -*-
"""
PENCI 训练监控和评估指标

提供传感器空间重建质量的核心指标:
- Pearson 相关系数: 衡量波形相似度
- SNR (dB): 信噪比
- NRMSE: 归一化均方根误差

所有函数接受 (B, C, T) 张量，返回标量。
"""

import torch
from typing import Dict


def pearson_correlation(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    逐通道 Pearson 相关系数，返回全 batch 均值

    参数:
        x: 原始信号 (B, C, T)
        x_hat: 重建信号 (B, C, T)

    返回:
        标量张量，所有通道和样本的平均 Pearson 相关系数
    """
    # 沿时间维度去均值
    x_mean = x.mean(dim=-1, keepdim=True)
    x_hat_mean = x_hat.mean(dim=-1, keepdim=True)
    x_centered = x - x_mean
    x_hat_centered = x_hat - x_hat_mean

    # 逐通道计算相关系数: (B, C)
    numerator = (x_centered * x_hat_centered).sum(dim=-1)
    denominator = (
        torch.sqrt((x_centered ** 2).sum(dim=-1))
        * torch.sqrt((x_hat_centered ** 2).sum(dim=-1))
    )

    # 避免除零
    corr = numerator / (denominator + 1e-8)

    return corr.mean()


def signal_to_noise_ratio(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    信噪比 (dB)

    SNR = 10 * log10(||x||^2 / ||x - x_hat||^2)

    参数:
        x: 原始信号 (B, C, T)
        x_hat: 重建信号 (B, C, T)

    返回:
        标量张量，所有通道和样本的平均 SNR (dB)
    """
    # 逐通道信号功率和误差功率: (B, C)
    signal_power = (x ** 2).sum(dim=-1)
    noise_power = ((x - x_hat) ** 2).sum(dim=-1)

    # 避免 log(0)
    snr = 10.0 * torch.log10(signal_power / (noise_power + 1e-8))

    return snr.mean()


def normalized_rmse(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    归一化均方根误差

    NRMSE = ||x - x_hat|| / ||x||

    参数:
        x: 原始信号 (B, C, T)
        x_hat: 重建信号 (B, C, T)

    返回:
        标量张量，所有通道和样本的平均 NRMSE (越低越好)
    """
    # 逐通道范数: (B, C)
    error_norm = torch.sqrt(((x - x_hat) ** 2).sum(dim=-1))
    signal_norm = torch.sqrt((x ** 2).sum(dim=-1))

    nrmse = error_norm / (signal_norm + 1e-8)

    return nrmse.mean()


def compute_all_metrics(x: torch.Tensor, x_hat: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    一次计算所有传感器空间指标

    参数:
        x: 原始信号 (B, C, T)
        x_hat: 重建信号 (B, C, T)

    返回:
        字典: {"pearson": ..., "snr_db": ..., "nrmse": ...}
    """
    return {
        "pearson": pearson_correlation(x, x_hat),
        "snr_db": signal_to_noise_ratio(x, x_hat),
        "nrmse": normalized_rmse(x, x_hat),
    }
