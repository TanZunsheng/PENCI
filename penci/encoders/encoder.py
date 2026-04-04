# -*- coding: utf-8 -*-
"""
PENCI 编码器模块

将原始 EEG/MEG 信号编码为源空间神经活动表示，并为 V1 提供多尺度编码接口。
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from penci.encoders.backward_solution import BackWardSolution
from penci.encoders.sensor_embed import BrainSensorModule
from penci.modules.seanet import SEANetEncoder


class BrainTokenizerEncoder(nn.Module):
    """
    脑信号编码器（无量化版本）

    来源：移植自 BrainOmni 项目
    原始文件：/work/2024/tanzunsheng/Code/BrainOmni/model_utils/module.py
    原始符号：BrainTokenizerEncoder
    """

    def __init__(
        self,
        n_filters: int,
        ratios: List[int],
        kernel_size: int,
        last_kernel_size: int,
        n_dim: int,
        n_head: int,
        dropout: float,
        n_neuro: int,
    ):
        super().__init__()

        self.n_dim = n_dim
        self.n_neuro = n_neuro

        self.seanet_encoder = SEANetEncoder(
            channels=1,
            dimension=n_dim,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
        )

        self.sensor_module = BrainSensorModule(n_dim)
        self.neuros = nn.Parameter(torch.randn(n_neuro, n_dim))
        self.backwardsolution = BackWardSolution(
            n_dim=n_dim,
            n_head=n_head,
            dropout=dropout,
        )
        self.k_proj = nn.Linear(n_dim, n_dim)

        self.scale_projectors = nn.ModuleDict()
        for factor, in_channels in self.seanet_encoder.stage_output_channels.items():
            key = f"x_{factor}x"
            if in_channels == n_dim:
                self.scale_projectors[key] = nn.Identity()
            else:
                self.scale_projectors[key] = nn.Linear(in_channels, n_dim)

    def _project_stage_to_dim(self, stage_tensor: torch.Tensor, scale_key: str) -> torch.Tensor:
        """
        将 SEANet 某个尺度输出投影到统一特征维度。

        参数:
            stage_tensor: (B, C_stage, T_scale)
            scale_key: 尺度键，如 x_8x
        返回:
            (B, D, T_scale)
        """
        projector = self.scale_projectors[scale_key]
        stage_tensor = rearrange(stage_tensor, "B C T -> B T C")
        stage_tensor = projector(stage_tensor)
        return rearrange(stage_tensor, "B T D -> B D T")

    def _infer_source_tokens(
        self,
        sensor_tokens: torch.Tensor,
        sensor_embedding: torch.Tensor,
        num_windows: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将传感器时序特征映射到源空间。

        参数:
            sensor_tokens: (B, C, W, D)，其中 W = N * T_scale
            sensor_embedding: (B, C, D)
            num_windows: N
        返回:
            source_tokens_flat: (B, N_neuro, W, D)
            source_tokens_window: (B, N_neuro, N, T_scale, D)
        """
        batch_size, _, total_steps, _ = sensor_tokens.shape

        sensor_embedding_expanded = rearrange(
            sensor_embedding.unsqueeze(2).expand(-1, -1, total_steps, -1),
            "B C W D -> (B W) C D",
        )
        sensor_tokens = rearrange(sensor_tokens, "B C W D -> (B W) C D")

        neuros = self.neuros.type_as(sensor_tokens).unsqueeze(0).expand(
            sensor_tokens.shape[0], -1, -1
        )
        source_tokens = self.backwardsolution(
            neuros,
            self.k_proj(sensor_tokens + sensor_embedding_expanded),
            sensor_tokens,
        )
        source_tokens = rearrange(
            source_tokens,
            "(B W) S D -> B S W D",
            B=batch_size,
            W=total_steps,
        )
        source_tokens_window = rearrange(
            source_tokens,
            "B S (N T) D -> B S N T D",
            N=num_windows,
        )
        return source_tokens, source_tokens_window

    @staticmethod
    def _select_scale_triplet(scale_factors: List[int]) -> Tuple[int, int, int]:
        """
        选择 V1 默认使用的高/中/低三种时间尺度。

        约定：下采样倍数越小，时间分辨率越高。
        """
        if len(scale_factors) == 1:
            only = scale_factors[0]
            return only, only, only
        if len(scale_factors) == 2:
            return scale_factors[0], scale_factors[0], scale_factors[1]
        high = scale_factors[0]
        low = scale_factors[-1]
        mid = scale_factors[len(scale_factors) // 2]
        return high, mid, low

    def forward_multiscale(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        多尺度编码接口（V1）。

        参数:
            x: 原始 EEG 信号 (B, C, N, L)
            pos: 电极位置 (B, C, 6)
            sensor_type: 传感器类型 (B, C)
        返回:
            包含多尺度源特征的字典。
        """
        batch_size, n_channels, num_windows, _ = x.shape

        x = rearrange(x, "B C N L -> (B C N) 1 L")
        seanet_output = self.seanet_encoder(x, return_intermediates=True)
        sensor_embedding = self.sensor_module(pos, sensor_type)  # (B, C, D)

        stage_factors = list(seanet_output["stage_factors"])
        low_factor = max(stage_factors)

        projected_stages: Dict[int, torch.Tensor] = {}
        for factor in stage_factors:
            stage = seanet_output["stage_by_factor"][factor]
            key = f"x_{factor}x"
            projected_stages[factor] = self._project_stage_to_dim(stage, key)

        # 低分辨率分支使用 SEANet 最终输出（包含 LSTM 与最后投影）
        projected_stages[low_factor] = seanet_output["final"]

        source_tokens_by_scale: Dict[str, torch.Tensor] = {}
        source_windows_by_scale: Dict[str, torch.Tensor] = {}
        scale_time_steps: Dict[str, int] = {}

        for factor in sorted(projected_stages.keys()):
            tokens = projected_stages[factor]
            tokens = rearrange(
                tokens,
                "(B C N) D T -> B C (N T) D",
                B=batch_size,
                C=n_channels,
                N=num_windows,
            )
            source_flat, source_window = self._infer_source_tokens(
                tokens,
                sensor_embedding=sensor_embedding,
                num_windows=num_windows,
            )
            key = f"x_{factor}x"
            source_tokens_by_scale[key] = source_flat
            source_windows_by_scale[key] = source_window
            scale_time_steps[key] = source_window.shape[3]

        high_factor, mid_factor, low_factor = self._select_scale_triplet(sorted(projected_stages.keys()))
        high_key = f"x_{high_factor}x"
        mid_key = f"x_{mid_factor}x"
        low_key = f"x_{low_factor}x"

        return {
            "source_tokens_high": source_tokens_by_scale[high_key],
            "source_tokens_mid": source_tokens_by_scale[mid_key],
            "source_tokens_low": source_tokens_by_scale[low_key],
            "source_windows_high": source_windows_by_scale[high_key],
            "source_windows_mid": source_windows_by_scale[mid_key],
            "source_windows_low": source_windows_by_scale[low_key],
            "source_tokens_by_scale": source_tokens_by_scale,
            "source_windows_by_scale": source_windows_by_scale,
            "sensor_embedding": sensor_embedding,
            "seanet_info": {
                "hop_length": int(seanet_output["hop_length"]),
                "stage_factors": sorted(projected_stages.keys()),
                "selected_scales": {
                    "high": high_key,
                    "mid": mid_key,
                    "low": low_key,
                },
                "scale_time_steps": scale_time_steps,
            },
        }

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        兼容旧接口：仅返回最低分辨率分支。

        返回:
            源空间特征 (B, N_neuro, N, T, D)
        """
        multiscale = self.forward_multiscale(x, pos, sensor_type)
        return multiscale["source_windows_low"]


class PENCIEncoder(nn.Module):
    """
    PENCI 编码器（输入适配 + BrainTokenizer 编码）。

    输入:
        - x: (C, T) 或 (B, C, T)
        - pos: (C, 6) 或 (B, C, 6)
        - sensor_type: (C,) 或 (B, C)
    """

    def __init__(
        self,
        n_dim: int = 256,
        n_neuro: int = 72,
        n_head: int = 4,
        dropout: float = 0.0,
        n_filters: int = 32,
        ratios: List[int] = None,
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        window_size: int = 512,
    ):
        super().__init__()

        if ratios is None:
            ratios = [8, 4, 2]

        self.window_size = window_size
        self.n_neuro = n_neuro
        self.n_dim = n_dim

        self.encoder = BrainTokenizerEncoder(
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            n_dim=n_dim,
            n_head=n_head,
            dropout=dropout,
            n_neuro=n_neuro,
        )

    def _normalize_inputs(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        统一输入为 batch 形式。
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            pos = pos.unsqueeze(0)
            sensor_type = sensor_type.unsqueeze(0)
        return x, pos, sensor_type

    def _windowize(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        按固定窗口切分输入时间序列。
        """
        batch_size, n_channels, total_length = x.shape
        num_windows = total_length // self.window_size
        if num_windows == 0:
            num_windows = 1
            pad_len = self.window_size - total_length
            x = F.pad(x, (0, pad_len))
            total_length = x.shape[-1]

        truncated_length = num_windows * self.window_size
        x = x[:, :, :truncated_length]
        x = rearrange(
            x,
            "B C (N L) -> B C N L",
            N=num_windows,
            L=self.window_size,
        )

        return x, {
            "batch_size": batch_size,
            "n_channels": n_channels,
            "input_length": total_length,
            "truncated_length": truncated_length,
            "num_windows": num_windows,
            "window_size": self.window_size,
        }

    def encode_multiscale(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        V1 多尺度接口。

        返回字段:
            - source_tokens_high / mid / low: (B, 72, T_scale, D)
            - source_windows_high / mid / low: (B, 72, N, T_scale_per_window, D)
            - sensor_embedding: (B, C, D)
            - window_info: 窗口切分信息
        """
        x, pos, sensor_type = self._normalize_inputs(x, pos, sensor_type)
        x_window, window_info = self._windowize(x)
        multiscale = self.encoder.forward_multiscale(x_window, pos, sensor_type)
        multiscale["window_info"] = window_info
        return multiscale

    def encode_source_features(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        `encode_multiscale` 的语义别名。
        """
        return self.encode_multiscale(x, pos, sensor_type)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        兼容旧接口：返回低分辨率源特征 (B, N_neuro, N, T', D)。
        """
        x, pos, sensor_type = self._normalize_inputs(x, pos, sensor_type)
        x_window, _ = self._windowize(x)
        return self.encoder(x_window, pos, sensor_type)
