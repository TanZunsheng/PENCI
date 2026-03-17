# -*- coding: utf-8 -*-
"""
PENCI 训练监控和评估指标

提供传感器空间重建质量的核心指标:
- Pearson 相关系数: 衡量波形相似度
- SNR (dB): 信噪比
- NRMSE: 归一化均方根误差

所有函数接受 (B, C, T) 张量，返回标量。
"""

from typing import Dict

import torch


DEFAULT_SIGNAL_NORM_EPS = 1e-12


def _safe_masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    对 (B, C) 指标张量按有效 mask 求和/均值。

    返回:
        {
            "mean": 标量，若无有效项则为 nan,
            "sum": 标量，总和（无有效项时为 0）,
            "count": 标量，有效项数量,
            "skipped": 标量，被跳过项数量,
        }
    """
    valid_values = values.masked_select(valid_mask)
    count = values.new_tensor(float(valid_values.numel()))
    skipped = values.new_tensor(float(values.numel() - valid_values.numel()))

    if valid_values.numel() == 0:
        return {
            "mean": values.new_tensor(float("nan")),
            "sum": values.new_zeros(()),
            "count": count,
            "skipped": skipped,
        }

    total = valid_values.sum()
    return {
        "mean": total / count,
        "sum": total,
        "count": count,
        "skipped": skipped,
    }


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


def signal_to_noise_ratio_stats(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    signal_norm_eps: float = DEFAULT_SIGNAL_NORM_EPS,
) -> Dict[str, torch.Tensor]:
    """
    信噪比 (dB) 统计量。

    对真值能量过低的通道，SNR 在数学上不稳定/无定义，因此直接跳过。
    """
    signal_power = (x ** 2).sum(dim=-1)
    noise_power = ((x - x_hat) ** 2).sum(dim=-1)
    valid_mask = torch.sqrt(signal_power) > signal_norm_eps
    snr = 10.0 * torch.log10(signal_power / (noise_power + 1e-8))
    return _safe_masked_mean(snr, valid_mask)


def signal_to_noise_ratio(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    signal_norm_eps: float = DEFAULT_SIGNAL_NORM_EPS,
) -> torch.Tensor:
    """
    信噪比 (dB)

    SNR = 10 * log10(||x||^2 / ||x - x_hat||^2)

    参数:
        x: 原始信号 (B, C, T)
        x_hat: 重建信号 (B, C, T)

    返回:
        标量张量，所有通道和样本的平均 SNR (dB)
    """
    return signal_to_noise_ratio_stats(
        x, x_hat, signal_norm_eps=signal_norm_eps
    )["mean"]


def normalized_rmse_stats(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    signal_norm_eps: float = DEFAULT_SIGNAL_NORM_EPS,
) -> Dict[str, torch.Tensor]:
    """
    NRMSE 统计量。

    对真值范数过低的通道，NRMSE 的分母接近 0，会人为放大误差，因此直接跳过。
    """
    error_norm = torch.sqrt(((x - x_hat) ** 2).sum(dim=-1))
    signal_norm = torch.sqrt((x ** 2).sum(dim=-1))
    valid_mask = signal_norm > signal_norm_eps
    nrmse = error_norm / (signal_norm + 1e-8)
    return _safe_masked_mean(nrmse, valid_mask)


def normalized_rmse(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    signal_norm_eps: float = DEFAULT_SIGNAL_NORM_EPS,
) -> torch.Tensor:
    """
    归一化均方根误差

    NRMSE = ||x - x_hat|| / ||x||

    参数:
        x: 原始信号 (B, C, T)
        x_hat: 重建信号 (B, C, T)

    返回:
        标量张量，所有通道和样本的平均 NRMSE (越低越好)
    """
    return normalized_rmse_stats(
        x, x_hat, signal_norm_eps=signal_norm_eps
    )["mean"]


def compute_all_metrics(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    signal_norm_eps: float = DEFAULT_SIGNAL_NORM_EPS,
) -> Dict[str, torch.Tensor]:
    """
    一次计算所有传感器空间指标

    参数:
        x: 原始信号 (B, C, T)
        x_hat: 重建信号 (B, C, T)

    返回:
        字典:
        {
            "pearson": ...,
            "snr_db": ...,
            "snr_db_sum": ...,
            "snr_db_count": ...,
            "snr_db_skipped": ...,
            "nrmse": ...,
            "nrmse_sum": ...,
            "nrmse_count": ...,
            "nrmse_skipped": ...,
        }
    """
    snr_stats = signal_to_noise_ratio_stats(
        x, x_hat, signal_norm_eps=signal_norm_eps
    )
    nrmse_stats = normalized_rmse_stats(
        x, x_hat, signal_norm_eps=signal_norm_eps
    )
    return {
        "pearson": pearson_correlation(x, x_hat),
        "snr_db": snr_stats["mean"],
        "snr_db_sum": snr_stats["sum"],
        "snr_db_count": snr_stats["count"],
        "snr_db_skipped": snr_stats["skipped"],
        "nrmse": nrmse_stats["mean"],
        "nrmse_sum": nrmse_stats["sum"],
        "nrmse_count": nrmse_stats["count"],
        "nrmse_skipped": nrmse_stats["skipped"],
    }
