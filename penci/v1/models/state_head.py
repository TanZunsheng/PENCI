# -*- coding: utf-8 -*-
"""
V1 第一层状态头

将多尺度源空间特征映射为显式脑区状态:
    S_t: (B, 72, T_state)
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class StateHead(nn.Module):
    """
    多尺度状态头（V1）

    输入:
        high_tokens: (B, N_source, T_high, D)
        mid_tokens:  (B, N_source, T_mid, D)
        low_tokens:  (B, N_source, T_low, D)

    输出:
        source_state: (B, N_source, T_high)
    """

    def __init__(
        self,
        n_dim: int = 256,
        hidden_dim: int = 128,
        activation: str = "identity",
    ):
        super().__init__()

        self.high_proj = nn.Linear(n_dim, hidden_dim)
        self.mid_proj = nn.Linear(n_dim, hidden_dim)
        self.low_proj = nn.Linear(n_dim, hidden_dim)

        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

        if activation == "tanh":
            self.output_activation: nn.Module = nn.Tanh()
        elif activation == "identity":
            self.output_activation = nn.Identity()
        else:
            raise ValueError(f"不支持的状态头输出激活: {activation}")

    @staticmethod
    def _resample_tokens(x: torch.Tensor, target_steps: int) -> torch.Tensor:
        """
        对 token 序列做线性重采样并保持 (B, N, T, D) 约定。
        """
        if x.shape[2] == target_steps:
            return x
        batch_size, n_source, _, n_dim = x.shape
        x = rearrange(x, "B N T D -> (B N) D T")
        x = F.interpolate(x, size=target_steps, mode="linear", align_corners=False)
        return rearrange(x, "(B N) D T -> B N T D", B=batch_size, N=n_source, D=n_dim)

    def forward(
        self,
        high_tokens: torch.Tensor,
        mid_tokens: torch.Tensor,
        low_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            high_tokens: (B, N, T_high, D)
            mid_tokens: (B, N, T_mid, D)
            low_tokens: (B, N, T_low, D)
        返回:
            显式脑区状态 (B, N, T_high)
        """
        target_steps = high_tokens.shape[2]
        mid_tokens = self._resample_tokens(mid_tokens, target_steps)
        low_tokens = self._resample_tokens(low_tokens, target_steps)

        high_hidden = self.high_proj(high_tokens)
        mid_hidden = self.mid_proj(mid_tokens)
        low_hidden = self.low_proj(low_tokens)

        fused = torch.cat([high_hidden, mid_hidden, low_hidden], dim=-1)
        fused = rearrange(fused, "B N T D -> (B N) D T")
        state = self.fuse(fused).squeeze(1)
        state = self.output_activation(state)
        return rearrange(state, "(B N) T -> B N T", B=high_tokens.shape[0], N=high_tokens.shape[1])

    def forward_from_dict(self, source_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        从 encode_multiscale 输出字典构造显式状态。
        """
        return self.forward(
            source_features["source_tokens_high"],
            source_features["source_tokens_mid"],
            source_features["source_tokens_low"],
        )
