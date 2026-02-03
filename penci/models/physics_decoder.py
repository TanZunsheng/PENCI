# -*- coding: utf-8 -*-
"""
物理解码器模块

基于物理约束的解码器，使用导联场矩阵（Lead Field Matrix）将源空间
神经活动映射回传感器空间。

物理背景：
- 导联场矩阵 L 描述了源空间到传感器空间的线性映射
- 传感器测量 = L @ 源活动
- 这是 EEG/MEG 前向模型的核心

设计选择：
1. 固定导联场：使用预计算的物理导联场矩阵，不参与训练
2. 可学习导联场：将导联场作为可学习参数，端到端优化
3. 注意力机制：使用 ForwardSolution 模拟导联场的作用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from penci.encoders.backward_solution import ForwardSolution
from penci.modules.seanet import SEANetEncoder


class PhysicsDecoder(nn.Module):
    """
    物理约束解码器
    
    将源空间的神经活动映射回传感器空间的 EEG/MEG 信号。
    
    支持两种模式：
    1. 固定导联场模式：使用预计算的导联场矩阵
    2. 注意力模式：使用可学习的交叉注意力（ForwardSolution）
    
    参数:
        n_dim: 特征维度
        n_sensors: 传感器数量
        n_sources: 源数量
        use_fixed_leadfield: 是否使用固定导联场
        leadfield_path: 导联场矩阵文件路径
        n_head: 注意力头数（仅注意力模式）
        dropout: Dropout 比率
    """
    
    def __init__(
        self,
        n_dim: int = 256,
        n_sensors: int = 128,
        n_sources: int = 64,
        use_fixed_leadfield: bool = False,
        leadfield_path: str = None,
        n_head: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.n_dim = n_dim
        self.n_sensors = n_sensors
        self.n_sources = n_sources
        self.use_fixed_leadfield = use_fixed_leadfield
        
        if use_fixed_leadfield:
            # 固定导联场模式
            if leadfield_path is not None:
                # 从文件加载
                leadfield = torch.load(leadfield_path)
                if isinstance(leadfield, dict):
                    leadfield = leadfield.get("leadfield", leadfield.get("L"))
            else:
                # 使用单位矩阵作为默认（如果源数等于传感器数）
                # 或者使用随机初始化的矩阵
                leadfield = torch.randn(n_sensors, n_sources) / (n_sources ** 0.5)
            
            # 注册为 buffer（不参与训练）
            self.register_buffer("leadfield", leadfield)
            
            # 时间维度的特征解码
            self.temporal_decoder = nn.Sequential(
                nn.Linear(n_dim, n_dim),
                nn.GELU(),
                nn.Linear(n_dim, 1),
            )
        else:
            # 注意力模式：使用 ForwardSolution
            self.forward_solution = ForwardSolution(
                n_dim=n_dim, n_head=n_head, dropout=dropout
            )
            
            # 时间维度的特征解码
            self.temporal_decoder = nn.Sequential(
                nn.Linear(n_dim, n_dim),
                nn.GELU(),
                nn.Linear(n_dim, 1),
            )
    
    def forward(
        self, 
        source_activity: torch.Tensor,
        sensor_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        参数:
            source_activity: 源空间神经活动 (B, N_sources, T, D) 或 (B, N_sources, D)
            sensor_embedding: 传感器嵌入 (B, N_sensors, D)，仅注意力模式需要
            
        返回:
            传感器空间重建 (B, N_sensors, T) 或 (B, N_sensors, 1)
        """
        if self.use_fixed_leadfield:
            return self._forward_leadfield(source_activity)
        else:
            return self._forward_attention(source_activity, sensor_embedding)
    
    def _forward_leadfield(self, source_activity: torch.Tensor) -> torch.Tensor:
        """使用固定导联场矩阵的前向传播"""
        # source_activity: (B, N_sources, T, D) 或 (B, N_sources, D)
        has_time = source_activity.dim() == 4
        
        if has_time:
            B, N_sources, T, D = source_activity.shape
            # 先解码时间维度的特征 -> 标量
            # (B, N_sources, T, D) -> (B, N_sources, T, 1) -> (B, N_sources, T)
            source_signal = self.temporal_decoder(source_activity).squeeze(-1)
        else:
            B, N_sources, D = source_activity.shape
            T = 1
            source_signal = self.temporal_decoder(source_activity).squeeze(-1)  # (B, N_sources)
            source_signal = source_signal.unsqueeze(-1)  # (B, N_sources, 1)
        
        # 应用导联场矩阵：(N_sensors, N_sources) @ (B, N_sources, T) -> (B, N_sensors, T)
        # 使用 einsum 进行批量矩阵乘法
        sensor_signal = torch.einsum("cs,bst->bct", self.leadfield, source_signal)
        
        return sensor_signal
    
    def _forward_attention(
        self, 
        source_activity: torch.Tensor,
        sensor_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """使用注意力机制的前向传播"""
        # source_activity: (B, N_sources, T, D)
        has_time = source_activity.dim() == 4
        
        if has_time:
            B, N_sources, T, D = source_activity.shape
            # 对每个时间步应用 ForwardSolution
            # 重排列: (B, N_sources, T, D) -> (B*T, N_sources, D)
            source_flat = rearrange(source_activity, "B N T D -> (B T) N D")
            sensor_flat = sensor_embedding.unsqueeze(1).repeat(1, T, 1, 1)
            sensor_flat = rearrange(sensor_flat, "B T C D -> (B T) C D")
            
            # ForwardSolution: (B*T, N_sensors, D)
            sensor_features = self.forward_solution(sensor_flat, source_flat)
            
            # 解码为信号: (B*T, N_sensors, D) -> (B*T, N_sensors, 1)
            sensor_signal = self.temporal_decoder(sensor_features).squeeze(-1)
            
            # 重排列回: (B*T, N_sensors) -> (B, N_sensors, T)
            sensor_signal = rearrange(sensor_signal, "(B T) C -> B C T", B=B, T=T)
        else:
            # source_activity: (B, N_sources, D)
            B, N_sources, D = source_activity.shape
            
            # ForwardSolution
            sensor_features = self.forward_solution(sensor_embedding, source_activity)
            
            # 解码
            sensor_signal = self.temporal_decoder(sensor_features).squeeze(-1)  # (B, N_sensors)
        
        return sensor_signal


class SEANetPhysicsDecoder(nn.Module):
    """
    基于 SEANet 的物理解码器
    
    来源：移植自 BrainOmni 项目的 BrainTokenizerDecoder
    原始文件：/work/2024/tanzunsheng/Code/BrainOmni/model_utils/module.py
    
    架构：
    1. ForwardSolution: 源空间 -> 传感器空间（特征级别）
    2. SEANetDecoder: 特征 -> 原始信号（上采样）
    
    注意：由于 PENCI 不需要 SEANetDecoder（只需要物理约束），
    这个类主要用于兼容性和对比实验。
    
    参数:
        n_dim: 特征维度
        n_head: 注意力头数
        n_filters: SEANet 基础滤波器数
        ratios: 上采样比例列表
        kernel_size: 卷积核大小
        last_kernel_size: 最后一层卷积核大小
        dropout: Dropout 比率
    """
    
    def __init__(
        self,
        n_dim: int,
        n_head: int,
        n_filters: int,
        ratios: list,
        kernel_size: int,
        last_kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        
        # 前向求解模块
        self.forwardsolution = ForwardSolution(n_dim, n_head, dropout)
        
        # 注意：这里不引入 SEANetDecoder，因为 PENCI 的设计不需要它
        # 如果需要完整的信号重建，可以添加简单的上采样模块
        self.upsample = nn.Sequential(
            nn.Linear(n_dim, n_dim * 4),
            nn.GELU(),
            nn.Linear(n_dim * 4, 1),
        )
        
        # 计算总上采样倍数
        import numpy as np
        self.upsample_factor = int(np.prod(ratios))
        
    def forward(
        self, 
        source_activity: torch.Tensor, 
        sensor_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            source_activity: 源空间神经活动 (B, N_sources, N, T, D)
            sensor_embedding: 传感器嵌入 (B, N_sensors, D)
            
        返回:
            重建的传感器信号 (B, N_sensors, N, L)
            其中 L = T * upsample_factor
        """
        B, N_sources, N, T, D = source_activity.shape
        N_sensors = sensor_embedding.shape[1]
        
        # 重排列
        source_flat = rearrange(source_activity, "B S N T D -> (B N T) S D")
        sensor_flat = sensor_embedding.view(B, -1, 1, 1, D).repeat(1, 1, N, T, 1)
        sensor_flat = rearrange(sensor_flat, "B C N T D -> (B N T) C D")
        
        # ForwardSolution
        sensor_features = self.forwardsolution(sensor_flat, source_flat)
        
        # 简单上采样（而非完整的 SEANetDecoder）
        sensor_signal = self.upsample(sensor_features).squeeze(-1)
        
        # 重排列
        sensor_signal = rearrange(
            sensor_signal, "(B N T) C -> B C N T", B=B, N=N, T=T
        )
        
        # 时间维度上采样
        sensor_signal = F.interpolate(
            rearrange(sensor_signal, "B C N T -> (B C) 1 (N T)"),
            scale_factor=self.upsample_factor,
            mode="linear",
            align_corners=False,
        )
        sensor_signal = rearrange(
            sensor_signal, "(B C) 1 L -> B C L", B=B, C=N_sensors
        )
        
        return sensor_signal
