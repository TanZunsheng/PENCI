# -*- coding: utf-8 -*-
"""
PENCI V1 第一层模型

主路径:
    EEG -> 多尺度源特征 -> 显式状态 S_t -> 物理投影 EEG_hat
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from penci.encoders.encoder import PENCIEncoder
from penci.shared.models.physics_decoder import PhysicsDecoder
from penci.v1.models.state_head import StateHead


class Stage1Model(nn.Module):
    """
    V1 第一层模型（不包含 DynamicsCore / A_base 学习）。
    """

    def __init__(
        self,
        n_dim: int = 256,
        n_neuro: int = 72,
        n_head: int = 4,
        dropout: float = 0.0,
        n_filters: int = 32,
        ratios: list = None,
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        window_size: int = 512,
        n_sensors: int = 128,
        state_hidden_dim: int = 128,
        state_activation: str = "identity",
        use_fixed_leadfield: bool = True,
        leadfield_path: Optional[str] = None,
        sensor_average_reference: bool = False,
    ):
        super().__init__()

        if ratios is None:
            ratios = [8, 4, 2]

        self.encoder = PENCIEncoder(
            n_dim=n_dim,
            n_neuro=n_neuro,
            n_head=n_head,
            dropout=dropout,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            window_size=window_size,
        )
        self.state_head = StateHead(
            n_dim=n_dim,
            hidden_dim=state_hidden_dim,
            activation=state_activation,
        )
        self.decoder = PhysicsDecoder(
            n_dim=n_dim,
            n_sensors=n_sensors,
            n_sources=n_neuro,
            use_fixed_leadfield=use_fixed_leadfield,
            leadfield_path=leadfield_path,
            n_head=n_head,
            dropout=dropout,
            projection_only=True,
        )
        self.sensor_average_reference = bool(sensor_average_reference)

    @staticmethod
    def _prepare_target(x: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        将输入信号重采样到目标长度。
        """
        if x.shape[-1] == target_length:
            return x
        return F.interpolate(x, size=target_length, mode="linear", align_corners=False)

    @staticmethod
    def _prepare_state_target(s_true: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        将监督状态重采样到模型状态长度。
        """
        if s_true.dim() == 2:
            s_true = s_true.unsqueeze(0)
        if s_true.shape[-1] == target_length:
            return s_true
        return F.interpolate(s_true, size=target_length, mode="linear", align_corners=False)

    @staticmethod
    def _build_valid_channel_mask(
        x: torch.Tensor,
        n_channels: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        构造有效通道 mask。

        参数:
            x: (B, C, T)
            n_channels: (B,) 或标量；为 None 时默认所有通道都有效
        """
        if n_channels is None or x.dim() != 3:
            return None
        if not torch.is_tensor(n_channels):
            n_channels = torch.as_tensor(n_channels, device=x.device)
        if n_channels.dim() == 0:
            n_channels = n_channels.unsqueeze(0)
        if n_channels.numel() == 1 and x.shape[0] > 1:
            n_channels = n_channels.expand(x.shape[0])
        channel_index = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        return channel_index < n_channels.to(device=x.device).view(-1, 1)

    @classmethod
    def _apply_average_reference(
        cls,
        x: torch.Tensor,
        n_channels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        对传感器信号应用平均参考，并忽略 padding 通道。
        """
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True
        valid_mask = cls._build_valid_channel_mask(x, n_channels)
        if valid_mask is None:
            centered = x - x.mean(dim=1, keepdim=True)
        else:
            valid = valid_mask.unsqueeze(-1).to(dtype=x.dtype)
            denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=x.dtype)
            reference = (x * valid).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
            centered = (x - reference) * valid
        return centered.squeeze(0) if squeeze_batch else centered

    @classmethod
    def _masked_sensor_mse(
        cls,
        pred: torch.Tensor,
        target: torch.Tensor,
        n_channels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        对有效传感器元素计算 MSE。
        """
        error = (pred - target) ** 2
        valid_mask = cls._build_valid_channel_mask(target, n_channels)
        if valid_mask is None:
            return error.mean()
        valid = valid_mask.unsqueeze(-1).to(dtype=error.dtype)
        denom = (valid_mask.sum().to(dtype=error.dtype) * error.shape[-1]).clamp_min(1.0)
        return (error * valid).sum() / denom

    def align_sensor_space(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        n_channels: Optional[torch.Tensor] = None,
        average_reference: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 target / reconstruction 对齐到相同的传感器表示空间。
        """
        apply_average_reference = (
            self.sensor_average_reference if average_reference is None else bool(average_reference)
        )
        if not apply_average_reference:
            return target, reconstruction
        return (
            self._apply_average_reference(target, n_channels=n_channels),
            self._apply_average_reference(reconstruction, n_channels=n_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        leadfield: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
        return_features: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        参数:
            x: (B, C, T) 或 (C, T)
            pos: (B, C, 6) 或 (C, 6)
            sensor_type: (B, C) 或 (C,)
            leadfield: (C, 72) 或 (B, C, 72)
            target_length: 重建目标长度，默认与输入 T 一致
        返回:
            {
              "source_state": (B, 72, T_state),
              "reconstruction": (B, C, T_target),
              "source_features": ... (可选)
            }
        """
        source_features = self.encoder.encode_multiscale(x, pos, sensor_type)
        source_state = self.state_head.forward_from_dict(source_features)

        if target_length is None:
            target_length = x.shape[-1] if x.dim() == 3 else x.shape[-1]

        reconstruction = self.decoder.project_source_state_to_sensor(
            source_state,
            leadfield=leadfield,
            target_length=target_length,
        )

        output = {
            "source_state": source_state,
            "reconstruction": reconstruction,
        }
        if return_features:
            output["source_features"] = source_features
        return output

    def compute_stage1_loss_from_output(
        self,
        output: Dict[str, torch.Tensor],
        x: torch.Tensor,
        s_true: Optional[torch.Tensor] = None,
        n_channels: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        average_reference: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        从已有 forward 输出计算损失。
        """
        if loss_weights is None:
            loss_weights = {
                "reconstruction": 1.0,
                "state_supervision": 1.0,
                "state_smoothness": 1e-2,
                "state_energy": 1e-3,
            }

        reconstruction = output["reconstruction"]
        source_state = output["source_state"]
        target = self._prepare_target(x, reconstruction.shape[-1])
        target, reconstruction = self.align_sensor_space(
            target,
            reconstruction,
            n_channels=n_channels,
            average_reference=average_reference,
        )
        reconstruction_loss = self._masked_sensor_mse(
            reconstruction,
            target,
            n_channels=n_channels,
        )
        if source_state.shape[-1] > 1:
            state_smoothness_loss = torch.mean(
                (source_state[:, :, 1:] - source_state[:, :, :-1]) ** 2
            )
        else:
            state_smoothness_loss = source_state.new_zeros(())
        state_energy_loss = torch.mean(source_state**2)

        if s_true is not None:
            s_target = self._prepare_state_target(s_true, source_state.shape[-1])
            state_supervision_loss = F.mse_loss(source_state, s_target)
        else:
            state_supervision_loss = source_state.new_zeros(())

        total_loss = (
            loss_weights.get("reconstruction", 1.0) * reconstruction_loss
            + loss_weights.get("state_smoothness", 1e-2) * state_smoothness_loss
            + loss_weights.get("state_energy", 1e-3) * state_energy_loss
            + loss_weights.get("state_supervision", 1.0) * state_supervision_loss
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "state_supervision_loss": state_supervision_loss,
            "state_smoothness_loss": state_smoothness_loss,
            "state_energy_loss": state_energy_loss,
        }

    def compute_stage1_loss_real(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        leadfield: Optional[torch.Tensor] = None,
        n_channels: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        average_reference: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        第一层真实数据损失。
        """
        output = self.forward(
            x,
            pos,
            sensor_type,
            leadfield=leadfield,
            target_length=x.shape[-1],
            return_features=False,
        )
        return self.compute_stage1_loss_from_output(
            output=output,
            x=x,
            s_true=None,
            n_channels=n_channels,
            loss_weights=loss_weights,
            average_reference=average_reference,
        )

    def compute_stage1_loss_sim(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        s_true: torch.Tensor,
        leadfield: Optional[torch.Tensor] = None,
        n_channels: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        average_reference: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        第一层仿真数据损失。
        """
        output = self.forward(
            x,
            pos,
            sensor_type,
            leadfield=leadfield,
            target_length=x.shape[-1],
            return_features=False,
        )
        return self.compute_stage1_loss_from_output(
            output=output,
            x=x,
            s_true=s_true,
            n_channels=n_channels,
            loss_weights=loss_weights,
            average_reference=average_reference,
        )


def build_stage1_model_from_config(config: Any) -> Stage1Model:
    """
    从配置构建 V1 第一层模型。
    """
    if hasattr(config, "model"):
        model_cfg = config.model
    else:
        model_cfg = config.get("model", config)

    seanet_cfg = model_cfg.get("seanet", {})
    stage1_cfg = model_cfg.get("stage1", {})
    physics_cfg = model_cfg.get("physics", {})
    data_cfg = config.get("data", {}) if hasattr(config, "get") else {}

    return Stage1Model(
        n_dim=model_cfg.get("n_dim", 256),
        n_neuro=model_cfg.get("n_neuro", 72),
        n_head=model_cfg.get("n_head", 4),
        dropout=model_cfg.get("dropout", 0.0),
        n_filters=seanet_cfg.get("n_filters", 32),
        ratios=seanet_cfg.get("ratios", [8, 4, 2]),
        kernel_size=seanet_cfg.get("kernel_size", 7),
        last_kernel_size=seanet_cfg.get("last_kernel_size", 7),
        window_size=model_cfg.get("window_size", data_cfg.get("window_length", 512)),
        n_sensors=data_cfg.get("n_channels", 128),
        state_hidden_dim=stage1_cfg.get("hidden_dim", 128),
        state_activation=stage1_cfg.get("state_activation", "identity"),
        use_fixed_leadfield=physics_cfg.get("use_fixed_leadfield", True),
        leadfield_path=physics_cfg.get("leadfield_path", None),
        sensor_average_reference=physics_cfg.get("sensor_average_reference", False),
    )
