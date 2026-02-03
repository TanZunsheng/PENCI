# -*- coding: utf-8 -*-
"""
反向求解模块

来源：移植自 BrainOmni 项目
原始文件：/work/2024/tanzunsheng/Code/BrainOmni/model_utils/module.py
原始符号：BackWardSolution, ForwardSolution
改动说明：
  - 保持核心实现不变
  - 添加中文注释
  - 模块独立拆分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class BackWardSolution(nn.Module):
    """
    反向求解模块（传感器空间 -> 源空间）
    
    通过交叉注意力机制，将传感器空间的观测映射到隐式源空间。
    这是 EEG 逆问题的神经网络近似解法。
    
    参数:
        n_dim: 特征维度
        n_head: 注意力头数
        dropout: Dropout 比率
        
    输入:
        neuros: 源空间查询向量 (B, N_source, D)
        k: 传感器空间的键 (B, N_sensor, D)
        x: 传感器空间的值 (B, N_sensor, D)
        
    输出:
        源空间特征 (B, N_source, D)
    """
    
    def __init__(self, n_dim: int, n_head: int, dropout: float):
        super().__init__()
        assert n_dim % n_head == 0, f"n_dim ({n_dim}) 必须能被 n_head ({n_head}) 整除"
        self.n_dim = n_dim
        self.n_head = n_head
        self.dropout = dropout
        # 值投影层
        self.v = nn.Linear(n_dim, n_dim)
        # 输出投影层
        self.proj = nn.Linear(n_dim, n_dim)

    def forward(self, neuros: torch.Tensor, k: torch.Tensor, x: torch.Tensor):
        """
        参数:
            neuros: 源空间查询向量 (B, N_source, D)，作为可学习参数
            k: 传感器空间的键 (B, N_sensor, D)，通常是 sensor_embedding + encoded_signal
            x: 传感器空间的值 (B, N_sensor, D)，编码后的信号特征
            
        返回:
            源空间特征 (B, N_source, D)
        """
        B, N_q, _ = neuros.shape
        # 重排列为多头注意力格式
        q = rearrange(neuros, "B T (H D) -> B H T D", H=self.n_head)
        k = rearrange(k, "B T (H D) -> B H T D", H=self.n_head)
        v = rearrange(self.v(x), "B T (H D) -> B H T D", H=self.n_head)
        # 计算缩放点积注意力
        output = (
            F.scaled_dot_product_attention(
                query=q, key=k, value=v, 
                dropout_p=self.dropout if self.training else 0.0, 
                is_causal=False
            )
            .transpose(1, 2)
            .contiguous()
        )
        output = output.view(B, N_q, -1)
        return self.proj(output)


class ForwardSolution(nn.Module):
    """
    前向求解模块（源空间 -> 传感器空间）
    
    通过交叉注意力机制，将源空间的神经活动映射回传感器空间。
    这模拟了 EEG 前向模型（导联场矩阵）的作用。
    
    参数:
        n_dim: 特征维度
        n_head: 注意力头数
        dropout: Dropout 比率
        
    输入:
        sensor_embedding: 传感器嵌入 (B, N_sensor, D)
        neurons: 源空间神经活动 (B, N_source, D)
        
    输出:
        传感器空间重建 (B, N_sensor, D)
    """
    
    def __init__(self, n_dim: int, n_head: int, dropout: float):
        super().__init__()
        assert n_dim % n_head == 0, f"n_dim ({n_dim}) 必须能被 n_head ({n_head}) 整除"
        self.n_dim = n_dim
        self.n_head = n_head
        self.dropout = dropout
        # 键值投影层
        self.kv = nn.Linear(n_dim, 2 * n_dim)
        # 输出投影层
        self.proj = nn.Linear(n_dim, n_dim)

    def forward(self, sensor_embedding: torch.Tensor, neurons: torch.Tensor):
        """
        参数:
            sensor_embedding: 传感器嵌入作为查询 (B, N_sensor, D)
            neurons: 源空间神经活动作为键值 (B, N_source, D)
            
        返回:
            传感器空间重建 (B, N_sensor, D)
        """
        B, C, _ = sensor_embedding.shape
        # 从源空间计算键值
        kv = self.kv(neurons)
        k, v = torch.split(kv, split_size_or_sections=self.n_dim, dim=-1)
        # 重排列为多头注意力格式
        q = rearrange(sensor_embedding, "B T (H D) -> B H T D", H=self.n_head)
        k = rearrange(k, "B T (H D) -> B H T D", H=self.n_head)
        v = rearrange(v, "B T (H D) -> B H T D", H=self.n_head)
        # 计算缩放点积注意力
        output = (
            F.scaled_dot_product_attention(
                query=q, key=k, value=v, 
                dropout_p=self.dropout if self.training else 0.0, 
                is_causal=False
            )
            .transpose(1, 2)
            .contiguous()
        )
        output = output.view(B, C, -1)
        return self.proj(output)
