# -*- coding: utf-8 -*-
"""
V1 状态与连接指标

用于第一层状态恢复与第二层连接恢复评估。
"""

from typing import Dict

import torch


def state_mse(s_true: torch.Tensor, s_pred: torch.Tensor) -> torch.Tensor:
    """
    状态均方误差。

    输入:
        s_true: (B, N, T) 或 (N, T)
        s_pred: (B, N, T) 或 (N, T)
    """
    if s_true.dim() == 2:
        s_true = s_true.unsqueeze(0)
    if s_pred.dim() == 2:
        s_pred = s_pred.unsqueeze(0)
    return torch.mean((s_true - s_pred) ** 2)


def state_pearson(s_true: torch.Tensor, s_pred: torch.Tensor) -> torch.Tensor:
    """
    状态 Pearson 相关。
    """
    if s_true.dim() == 2:
        s_true = s_true.unsqueeze(0)
    if s_pred.dim() == 2:
        s_pred = s_pred.unsqueeze(0)

    true_center = s_true - s_true.mean(dim=-1, keepdim=True)
    pred_center = s_pred - s_pred.mean(dim=-1, keepdim=True)
    numerator = (true_center * pred_center).sum(dim=-1)
    denominator = torch.sqrt((true_center ** 2).sum(dim=-1)) * torch.sqrt((pred_center ** 2).sum(dim=-1))
    return (numerator / (denominator + 1e-8)).mean()


def state_temporal_smoothness(s: torch.Tensor) -> torch.Tensor:
    """
    状态时间平滑性（越小越平滑）。
    """
    if s.dim() == 2:
        s = s.unsqueeze(0)
    if s.shape[-1] <= 1:
        return s.new_zeros(())
    return torch.mean((s[:, :, 1:] - s[:, :, :-1]) ** 2)


def state_distribution_drift(s_ref: torch.Tensor, s_pred: torch.Tensor) -> torch.Tensor:
    """
    状态分布漂移（均值差 + 标准差差）。
    """
    if s_ref.dim() == 2:
        s_ref = s_ref.unsqueeze(0)
    if s_pred.dim() == 2:
        s_pred = s_pred.unsqueeze(0)

    ref_mean = s_ref.mean()
    pred_mean = s_pred.mean()
    ref_std = s_ref.std(unbiased=False)
    pred_std = s_pred.std(unbiased=False)
    return torch.abs(ref_mean - pred_mean) + torch.abs(ref_std - pred_std)


def connectivity_relative_error(a_pred: torch.Tensor, a_true: torch.Tensor) -> torch.Tensor:
    """
    连接相对误差: ||A_pred - A_true||_F / ||A_true||_F
    """
    num = torch.linalg.norm(a_pred - a_true)
    den = torch.linalg.norm(a_true) + 1e-8
    return num / den


def connectivity_edge_f1(
    a_pred: torch.Tensor,
    a_true: torch.Tensor,
    threshold: float = 1e-4,
) -> torch.Tensor:
    """
    边恢复 F1（基于 |A| > threshold 的有向边集合）。
    """
    pred_edge = (torch.abs(a_pred) > threshold).to(torch.float32)
    true_edge = (torch.abs(a_true) > threshold).to(torch.float32)

    tp = torch.sum(pred_edge * true_edge)
    fp = torch.sum(pred_edge * (1.0 - true_edge))
    fn = torch.sum((1.0 - pred_edge) * true_edge)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2.0 * precision * recall / (precision + recall + 1e-8)


def connectivity_direction_accuracy(
    a_pred: torch.Tensor,
    a_true: torch.Tensor,
    threshold: float = 1e-4,
) -> torch.Tensor:
    """
    方向恢复准确率（仅在存在真实边的节点对上统计）。
    """
    n = a_true.shape[0]
    correct = a_true.new_zeros(())
    total = a_true.new_zeros(())

    for i in range(n):
        for j in range(i + 1, n):
            true_ij = torch.abs(a_true[i, j]) > threshold
            true_ji = torch.abs(a_true[j, i]) > threshold
            if not (true_ij or true_ji):
                continue

            pred_ij = torch.abs(a_pred[i, j]) > threshold
            pred_ji = torch.abs(a_pred[j, i]) > threshold
            total = total + 1.0
            if (pred_ij == true_ij) and (pred_ji == true_ji):
                correct = correct + 1.0

    if float(total.item()) == 0.0:
        return a_true.new_tensor(float("nan"))
    return correct / total


def compute_state_metrics(s_true: torch.Tensor, s_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    汇总状态指标。
    """
    return {
        "state_mse": state_mse(s_true, s_pred),
        "state_pearson": state_pearson(s_true, s_pred),
        "state_temporal_smoothness": state_temporal_smoothness(s_pred),
        "state_distribution_drift": state_distribution_drift(s_true, s_pred),
    }


def compute_connectivity_metrics(
    a_true: torch.Tensor,
    a_pred: torch.Tensor,
    threshold: float = 1e-4,
) -> Dict[str, torch.Tensor]:
    """
    汇总连接指标。
    """
    return {
        "connectivity_relative_error": connectivity_relative_error(a_pred, a_true),
        "connectivity_edge_f1": connectivity_edge_f1(a_pred, a_true, threshold=threshold),
        "connectivity_direction_accuracy": connectivity_direction_accuracy(
            a_pred,
            a_true,
            threshold=threshold,
        ),
    }
