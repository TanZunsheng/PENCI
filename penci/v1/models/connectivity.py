# -*- coding: utf-8 -*-
"""
PENCI V1 第二层静态连接模型

固定阶数 p=2:
    S_hat_{t+1} = A_1 S_t + A_2 S_{t-1}
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticConnectivityModel(nn.Module):
    """
    V1 最小静态有效连接模型。

    输入:
        source_state: (B, N, T)
    输出:
        pred_next: (B, N, T-2), 对应预测的 [S_2, ..., S_{T-1}]
    """

    def __init__(
        self,
        n_sources: int = 72,
        lag_order: int = 2,
    ):
        super().__init__()
        if lag_order != 2:
            raise ValueError(f"V1 固定 lag_order=2，收到: {lag_order}")

        self.n_sources = n_sources
        self.lag_order = lag_order

        self.a1_raw = nn.Parameter(torch.zeros(n_sources, n_sources))
        self.a2_raw = nn.Parameter(torch.zeros(n_sources, n_sources))

        offdiag_mask = torch.ones(n_sources, n_sources)
        offdiag_mask.fill_diagonal_(0.0)
        self.register_buffer("offdiag_mask", offdiag_mask)

    def constrained_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回零对角约束后的 A1, A2。
        """
        a1 = self.a1_raw * self.offdiag_mask
        a2 = self.a2_raw * self.offdiag_mask
        return a1, a2

    def export_a_base(self) -> torch.Tensor:
        """
        导出组合静态连接矩阵 A_base = A1 + A2（零对角约束后）。
        """
        a1, a2 = self.constrained_matrices()
        return a1 + a2

    def spectral_radius(self) -> torch.Tensor:
        """
        返回 A_base 的谱半径。
        """
        a_base = self.export_a_base().detach()
        eigvals = torch.linalg.eigvals(a_base)
        return torch.max(torch.abs(eigvals)).real

    @torch.no_grad()
    def apply_spectral_radius_clip(self, max_radius: float = 0.98) -> Dict[str, float]:
        """
        显式谱半径裁剪（epoch 级调用）。
        """
        current_radius = float(self.spectral_radius().item())
        if current_radius <= max_radius or current_radius == 0.0:
            return {"before": current_radius, "after": current_radius, "scaled": 0.0}

        scale = max_radius / (current_radius + 1e-12)
        self.a1_raw.mul_(scale)
        self.a2_raw.mul_(scale)
        after = float(self.spectral_radius().item())
        return {"before": current_radius, "after": after, "scaled": 1.0}

    def predict_next(
        self,
        source_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测下一时刻状态，并返回监督目标。

        参数:
            source_state: (B, N, T)
        返回:
            pred_next: (B, N, T-2)
            target_next: (B, N, T-2)
        """
        if source_state.dim() != 3:
            raise ValueError(f"source_state 需要形状 (B, N, T)，收到: {tuple(source_state.shape)}")
        if source_state.shape[1] != self.n_sources:
            raise ValueError(
                f"source_state 第二维应为 n_sources={self.n_sources}，收到: {source_state.shape[1]}"
            )
        if source_state.shape[-1] < 3:
            raise ValueError("source_state 时间长度至少为 3，才能构造 p=2 预测")

        a1, a2 = self.constrained_matrices()
        s_tm1 = source_state[:, :, :-2]
        s_t = source_state[:, :, 1:-1]
        target_next = source_state[:, :, 2:]

        pred_next = torch.einsum("ij,bjt->bit", a1, s_t) + torch.einsum("ij,bjt->bit", a2, s_tm1)
        return pred_next, target_next

    def forward(self, source_state: torch.Tensor) -> torch.Tensor:
        pred_next, _ = self.predict_next(source_state)
        return pred_next

    def l1_sparsity_loss(self) -> torch.Tensor:
        """
        L1 稀疏正则（作用于 A_base）。
        """
        return torch.mean(torch.abs(self.export_a_base()))

    def stability_penalty(self, max_radius: float = 0.98) -> torch.Tensor:
        """
        谱半径稳定性惩罚：max(0, rho(A_base)-max_radius)。
        """
        radius = self.spectral_radius()
        penalty = torch.clamp(radius - max_radius, min=0.0)
        return penalty

    def compute_loss(
        self,
        source_state: torch.Tensor,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算第二层最小损失。
        """
        if loss_weights is None:
            loss_weights = {
                "state_prediction": 1.0,
                "l1_sparsity": 1e-3,
                "stability": 1e-2,
            }

        pred_next, target_next = self.predict_next(source_state)
        state_prediction_loss = F.mse_loss(pred_next, target_next)
        l1_loss = self.l1_sparsity_loss()
        stability_loss = self.stability_penalty()

        total_loss = (
            loss_weights.get("state_prediction", 1.0) * state_prediction_loss
            + loss_weights.get("l1_sparsity", 1e-3) * l1_loss
            + loss_weights.get("stability", 1e-2) * stability_loss
        )

        return {
            "loss": total_loss,
            "state_prediction_loss": state_prediction_loss,
            "l1_sparsity_loss": l1_loss,
            "stability_penalty": stability_loss,
        }



def build_stage2_model_from_config(config: Any) -> StaticConnectivityModel:
    """
    从配置构建 V1 第二层静态连接模型。
    """
    if hasattr(config, "model"):
        model_cfg = config.model
    else:
        model_cfg = config.get("model", config)

    stage2_cfg = model_cfg.get("stage2", {})
    return StaticConnectivityModel(
        n_sources=model_cfg.get("n_neuro", 72),
        lag_order=stage2_cfg.get("lag_order", 2),
    )
