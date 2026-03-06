# -*- coding: utf-8 -*-
"""
PENCI 完整模型

Physics-constrained End-to-end Neural Connectivity Inference

"三明治"架构：
1. 编码器（Encoder）：传感器空间 -> 源空间
2. 动力学核心（Dynamics Core）：源空间时间演化
3. 物理解码器（Physics Decoder）：源空间 -> 传感器空间

训练目标：
- 重建损失：预测的传感器信号 vs 真实信号
- 动力学正则化：源空间活动的平滑性/稀疏性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Any, Dict, Optional, Tuple

from penci.encoders.encoder import PENCIEncoder, BrainTokenizerEncoder
from penci.encoders.sensor_embed import BrainSensorModule
from penci.models.dynamics import DynamicsCore, DynamicsRNN
from penci.models.physics_decoder import PhysicsDecoder


class PENCI(nn.Module):
    """
    PENCI: Physics-constrained End-to-end Neural Connectivity Inference
    
    完整的端到端神经连接推断模型。
    
    架构流程：
    1. 输入：原始 EEG/MEG 信号 (B, C, T) + 电极位置 (B, C, 6) + 传感器类型 (B, C)
    2. 编码器：将传感器空间信号编码到隐式源空间 -> (B, N_neuro, T', D)
    3. 动力学核心：建模源空间的时间演化 -> (B, N_neuro, T', D)
    4. 物理解码器：将源空间活动映射回传感器空间 -> (B, C, T)
    
    参数:
        # 通用参数
        n_dim: 特征维度
        n_neuro: 隐式源数量
        n_head: 注意力头数
        dropout: Dropout 比率
        
        # 编码器参数
        n_filters: SEANet 基础滤波器数
        ratios: 下采样比例列表
        kernel_size: 卷积核大小
        last_kernel_size: 最后一层卷积核大小
        window_size: 信号分段窗口大小
        
        # 动力学参数
        dynamics_type: 动力学模型类型，"transformer" 或 "rnn"
        dynamics_layers: 动力学模型层数
        dynamics_heads: Transformer 注意力头数
        dynamics_ff_dim: FFN 隐藏层维度
        
        # 解码器参数
        use_fixed_leadfield: 是否使用固定导联场
        leadfield_path: 导联场矩阵文件路径
    """
    
    def __init__(
        self,
        # 通用参数
        n_dim: int = 256,
        n_neuro: int = 72,
        n_head: int = 4,
        dropout: float = 0.1,
        # 编码器参数
        n_filters: int = 32,
        ratios: list = None,
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        window_size: int = 320,
        # 动力学参数
        dynamics_type: str = "transformer",
        dynamics_layers: int = 4,
        dynamics_heads: int = 8,
        dynamics_ff_dim: int = 1024,
        # 解码器参数
        use_fixed_leadfield: bool = True,
        leadfield_path: str = None,
        n_sensors: int = 128,
    ):
        super().__init__()
        
        if ratios is None:
            ratios = [8, 4, 2]
            
        self.n_dim = n_dim
        self.n_neuro = n_neuro
        self.n_sensors = n_sensors
        self.window_size = window_size
        
        # 1. 编码器
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
        
        # 2. 动力学核心
        if dynamics_type.lower() == "transformer":
            self.dynamics = DynamicsCore(
                n_dim=n_dim,
                n_layers=dynamics_layers,
                n_heads=dynamics_heads,
                dim_feedforward=dynamics_ff_dim,
                dropout=dropout,
            )
        else:
            self.dynamics = DynamicsRNN(
                n_dim=n_dim,
                hidden_dim=dynamics_ff_dim,
                n_layers=dynamics_layers,
                dropout=dropout,
            )
        
        # 3. 传感器嵌入（解码时需要）
        self.sensor_module = BrainSensorModule(n_dim)
        
        # 4. 物理解码器
        self.decoder = PhysicsDecoder(
            n_dim=n_dim,
            n_sensors=n_sensors,
            n_sources=n_neuro,
            use_fixed_leadfield=use_fixed_leadfield,
            leadfield_path=leadfield_path,
            n_head=n_head,
            dropout=dropout,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        leadfield: Optional[torch.Tensor] = None,
        return_source: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        参数:
            x: 原始 EEG/MEG 信号 (B, C, T) 或 (C, T)
            pos: 电极位置 (B, C, 6) 或 (C, 6)
            sensor_type: 传感器类型 (B, C) 或 (C,)
            leadfield: 导联场矩阵，支持两种形状:
                       - (N_sensors, N_sources): 全 batch 共享
                       - (B, N_sensors, N_sources): per-sample 导联场
                       仅导联场模式需要（如果解码器未加载静态导联场）
            return_source: 是否返回源空间活动
            
        返回:
            字典包含:
            - 'reconstruction': 重建的传感器信号 (B, C, T')
            - 'source_activity': 源空间活动 (B, N_neuro, T', D)（如果 return_source=True）
        """
        # 处理无 batch 维度的输入
        if x.dim() == 2:
            x = x.unsqueeze(0)
            pos = pos.unsqueeze(0)
            sensor_type = sensor_type.unsqueeze(0)
        
        B, C, T = x.shape
        
        # 1. 编码：传感器空间 -> 源空间
        # 输出: (B, N_neuro, N, T_enc, D)
        source_encoded = self.encoder(x, pos, sensor_type)
        
        # 重排列为 (B, N_neuro, N*T_enc, D)
        B, N_neuro, N, T_enc, D = source_encoded.shape
        source_flat = rearrange(source_encoded, "B N Ns T D -> B N (Ns T) D")
        
        # 2. 动力学演化
        # 输入输出: (B, N_neuro, T_total, D)
        source_evolved = self.dynamics(source_flat)
        
        # 3. 计算传感器嵌入
        sensor_embedding = self.sensor_module(pos, sensor_type)  # (B, C, D)
        
        # 4. 物理解码：源空间 -> 传感器空间
        # 传递导联场到解码器（动态导联场模式）
        reconstruction = self.decoder(
            source_evolved, sensor_embedding, leadfield=leadfield
        )
        
        output = {"reconstruction": reconstruction}
        
        if return_source:
            output["source_activity"] = source_evolved
            output["source_encoded"] = source_encoded
        
        return output
    
    def compute_loss(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        leadfield: Optional[torch.Tensor] = None,
        target: torch.Tensor = None,
        loss_weights: Dict[str, float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算训练损失
        
        参数:
            x: 输入信号 (B, C, T)
            pos: 电极位置 (B, C, 6)
            sensor_type: 传感器类型 (B, C)
            leadfield: 导联场矩阵，支持两种形状:
                       - (N_sensors, N_sources): 全 batch 共享
                       - (B, N_sensors, N_sources): per-sample 导联场
                       仅导联场模式需要（如果解码器未加载静态导联场）
            target: 目标信号 (B, C, T')，如果为 None 则使用输入作为目标
            loss_weights: 损失权重字典
            
        返回:
            字典包含各项损失和总损失
        """
        if loss_weights is None:
            loss_weights = {"reconstruction": 1.0, "dynamics": 0.1}
        
        # 前向传播（传递导联场到 forward）
        output = self.forward(x, pos, sensor_type, leadfield=leadfield, return_source=True)
        reconstruction = output["reconstruction"]
        source_activity = output["source_activity"]
        
        # 准备目标
        if target is None:
            # 自重建任务：目标是输入信号的下采样版本
            target = self._prepare_target(x, reconstruction.shape[-1])
        
        # 1. 重建损失
        recon_loss = F.mse_loss(reconstruction, target)
        
        # 2. 动力学正则化损失（源活动的时间平滑性）
        # 计算时间维度上的一阶差分
        dynamics_loss = torch.mean(
            (source_activity[:, :, 1:, :] - source_activity[:, :, :-1, :]) ** 2
        )
        
        # 总损失
        total_loss = (
            loss_weights.get("reconstruction", 1.0) * recon_loss +
            loss_weights.get("dynamics", 0.1) * dynamics_loss
        )
        
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "dynamics_loss": dynamics_loss,
        }
    
    def _prepare_target(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """准备目标信号（下采样到匹配输出长度）"""
        B, C, T = x.shape
        if T == target_len:
            return x
        # 简单平均池化下采样
        x_reshaped = x.view(B, C, target_len, T // target_len)
        return x_reshaped.mean(dim=-1)


class PENCILite(nn.Module):
    """
    PENCI 轻量版
    
    简化版本，用于快速实验和资源受限场景。
    
    主要简化：
    - 使用更浅的编码器
    - 使用 RNN 替代 Transformer 动力学
    - 使用固定导联场解码器
    """
    
    def __init__(
        self,
        n_dim: int = 128,
        n_neuro: int = 32,
        n_head: int = 4,
        dropout: float = 0.1,
        n_filters: int = 16,
        ratios: list = None,
        dynamics_layers: int = 2,
        n_sensors: int = 128,
    ):
        super().__init__()
        
        if ratios is None:
            ratios = [8, 4]  # 总下采样 32x
            
        self.n_dim = n_dim
        self.n_neuro = n_neuro
        
        # 简化编码器
        self.encoder = PENCIEncoder(
            n_dim=n_dim,
            n_neuro=n_neuro,
            n_head=n_head,
            dropout=dropout,
            n_filters=n_filters,
            ratios=ratios,
            window_size=256,
        )
        
        # RNN 动力学
        self.dynamics = DynamicsRNN(
            n_dim=n_dim,
            hidden_dim=n_dim * 2,
            n_layers=dynamics_layers,
            dropout=dropout,
        )
        
        # 传感器嵌入
        self.sensor_module = BrainSensorModule(n_dim)
        
        # 固定导联场解码器
        self.decoder = PhysicsDecoder(
            n_dim=n_dim,
            n_sensors=n_sensors,
            n_sources=n_neuro,
            use_fixed_leadfield=True,
            n_head=n_head,
            dropout=dropout,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        leadfield: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        简化的前向传播
        
        参数:
            x: 原始 EEG/MEG 信号 (B, C, T) 或 (C, T)
            pos: 电极位置 (B, C, 6) 或 (C, 6)
            sensor_type: 传感器类型 (B, C) 或 (C,)
            leadfield: 导联场矩阵，支持两种形状:
                       - (N_sensors, N_sources): 全 batch 共享
                       - (B, N_sensors, N_sources): per-sample 导联场
                       仅导联场模式需要（如果解码器未加载静态导联场）
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            pos = pos.unsqueeze(0)
            sensor_type = sensor_type.unsqueeze(0)
        
        # 编码
        source_encoded = self.encoder(x, pos, sensor_type)
        B, N_neuro, N, T_enc, D = source_encoded.shape
        source_flat = rearrange(source_encoded, "B N Ns T D -> B N (Ns T) D")
        
        # 动力学
        source_evolved = self.dynamics(source_flat)
        
        # 传感器嵌入
        sensor_embedding = self.sensor_module(pos, sensor_type)  # (B, C, D)
        
        # 解码（传递导联场到解码器）
        reconstruction = self.decoder(
            source_evolved, sensor_embedding, leadfield=leadfield
        )
        
        return {"reconstruction": reconstruction}


def build_penci_from_config(config: Any) -> PENCI:
    """
    从配置构建 PENCI 模型
    
    参数:
        config: OmegaConf 配置对象或字典
        
    返回:
        PENCI 模型实例
    """
    if hasattr(config, "model"):
        model_cfg = config.model
    else:
        model_cfg = config.get("model", config)
    
    # 提取参数
    n_dim = model_cfg.get("n_dim", 256)
    n_neuro = model_cfg.get("n_neuro", 72)
    n_head = model_cfg.get("n_head", 4)
    dropout = model_cfg.get("dropout", 0.1)
    
    # SEANet 参数
    seanet_cfg = model_cfg.get("seanet", {})
    n_filters = seanet_cfg.get("n_filters", 32)
    ratios = seanet_cfg.get("ratios", [8, 4, 2])
    kernel_size = seanet_cfg.get("kernel_size", 7)
    last_kernel_size = seanet_cfg.get("last_kernel_size", 7)
    
    # 动力学参数
    dynamics_cfg = model_cfg.get("dynamics", {})
    dynamics_type = dynamics_cfg.get("type", "transformer")
    dynamics_layers = dynamics_cfg.get("n_layers", 4)
    dynamics_heads = dynamics_cfg.get("n_heads", 8)
    dynamics_ff_dim = dynamics_cfg.get("dim_feedforward", 1024)
    
    # 物理解码器参数
    physics_cfg = model_cfg.get("physics", {})
    use_fixed_leadfield = physics_cfg.get("use_fixed_leadfield", True)
    leadfield_path = physics_cfg.get("leadfield_path", None)
    
    # 数据参数
    data_cfg = config.get("data", {}) if hasattr(config, "get") else {}
    n_sensors = data_cfg.get("n_channels", 128)
    
    return PENCI(
        n_dim=n_dim,
        n_neuro=n_neuro,
        n_head=n_head,
        dropout=dropout,
        n_filters=n_filters,
        ratios=ratios,
        kernel_size=kernel_size,
        last_kernel_size=last_kernel_size,
        dynamics_type=dynamics_type,
        dynamics_layers=dynamics_layers,
        dynamics_heads=dynamics_heads,
        dynamics_ff_dim=dynamics_ff_dim,
        use_fixed_leadfield=use_fixed_leadfield,
        leadfield_path=leadfield_path,
        n_sensors=n_sensors,
    )
