# -*- coding: utf-8 -*-
"""
传感器嵌入模块

来源：移植自 BrainOmni 项目
原始文件：/work/2024/tanzunsheng/Code/BrainOmni/model_utils/module.py
原始符号：BrainSensorModule
改动说明：
  - 保持核心实现不变
  - 添加中文注释
  - 模块独立拆分
"""

import torch
import torch.nn as nn
from penci.modules.attention import RMSNorm, FeedForward


class BrainSensorModule(nn.Module):
    """
    传感器嵌入模块
    
    将电极位置坐标和传感器类型融合为统一的传感器嵌入表示
    
    参数:
        n_dim: 嵌入维度
        
    输入:
        pos: 电极位置和方向 (B, C, 6)，包含 (x, y, z, nx, ny, nz)
        sensor_type: 传感器类型 (B, C)，0=EEG, 1=MEG_grad, 2=MEG_mag
        
    输出:
        sensor_embedding: 融合后的传感器嵌入 (B, C, D)
    """
    
    def __init__(self, n_dim):
        super().__init__()
        # 传感器类型嵌入：0=EEG, 1=MEG_grad, 2=MEG_mag
        self.sensor_embedding_layer = nn.Embedding(3, n_dim)
        # 位置嵌入：将 6 维位置映射到 n_dim
        self.pos_embedding_layer = nn.Sequential(
            nn.Linear(6, n_dim // 2),
            nn.SELU(),
            nn.Linear(n_dim // 2, n_dim),
        )
        # 融合 MLP
        self.aggregate_mlp = FeedForward(n_dim, 0.0)
        self.norm = RMSNorm(n_dim)

    def forward(self, pos: torch.Tensor, sensor_type: torch.Tensor):
        """
        参数:
            pos: 电极位置和方向 (B, C, 6)
            sensor_type: 传感器类型 (B, C)
            
        返回:
            融合后的传感器嵌入 (B, C, D)
        """
        # 位置嵌入
        x = self.pos_embedding_layer(pos)
        # 加上传感器类型嵌入
        x = x + self.sensor_embedding_layer(sensor_type).type_as(x)
        # 通过 MLP 融合
        x = x + self.aggregate_mlp(x)
        return self.norm(x)
