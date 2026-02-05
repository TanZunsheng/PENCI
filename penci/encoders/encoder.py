# -*- coding: utf-8 -*-
"""
PENCI 编码器模块

将原始 EEG/MEG 信号编码为源空间的神经活动表示
"""

import torch
import torch.nn as nn
from einops import rearrange

from penci.modules.seanet import SEANetEncoder
from penci.encoders.sensor_embed import BrainSensorModule
from penci.encoders.backward_solution import BackWardSolution


class BrainTokenizerEncoder(nn.Module):
    """
    脑信号编码器（无量化版本）
    
    来源：移植自 BrainOmni 项目
    原始文件：/work/2024/tanzunsheng/Code/BrainOmni/model_utils/module.py
    原始符号：BrainTokenizerEncoder
    改动说明：
      - 移除量化层（BrainQuantizer）
      - 直接输出源空间特征
      - 添加中文注释
    
    架构流程：
    1. SEANet 编码：(B, C, N, L) -> (B, C, N*T, D)
       - 对每个通道独立进行时域卷积编码
       - 输出压缩的时间特征
       
    2. 传感器嵌入：(pos, sensor_type) -> (B, C, D)
       - 融合电极位置和传感器类型信息
       
    3. 反向求解：(B, C, W, D) -> (B, N_neuro, W, D)
       - 通过交叉注意力将传感器空间映射到源空间
       - 隐式学习 EEG 逆问题
    
    参数:
        n_filters: SEANet 基础滤波器数
        ratios: 下采样比例列表
        kernel_size: 卷积核大小
        last_kernel_size: 最后一层卷积核大小
        n_dim: 特征维度
        n_head: 注意力头数
        dropout: Dropout 比率
        n_neuro: 隐式源空间维度（虚拟源数量）
    """
    
    def __init__(
        self,
        n_filters: int,
        ratios: list,
        kernel_size: int,
        last_kernel_size: int,
        n_dim: int,
        n_head: int,
        dropout: float,
        n_neuro: int,
    ):
        super().__init__()
        
        # SEANet 编码器：将原始信号编码为高维特征
        self.seanet_encoder = SEANetEncoder(
            channels=1,
            dimension=n_dim,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
        )
        
        # 传感器嵌入模块
        self.sensor_module = BrainSensorModule(n_dim)
        
        # 可学习的源空间基向量
        # 这些向量代表隐式的神经源位置
        self.neuros = nn.Parameter(torch.randn(n_neuro, n_dim))
        
        # 反向求解模块
        self.backwardsolution = BackWardSolution(
            n_dim=n_dim, n_head=n_head, dropout=dropout
        )
        
        # 键投影层
        self.k_proj = nn.Linear(n_dim, n_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        pos: torch.Tensor, 
        sensor_type: torch.Tensor
    ) -> torch.Tensor:
        """
        参数:
            x: 原始 EEG 信号 (B, C, N, L)
               - B: batch size
               - C: 通道数（电极数）
               - N: 分段数
               - L: 每段长度
            pos: 电极位置 (B, C, 6)，包含 (x, y, z, nx, ny, nz)
            sensor_type: 传感器类型 (B, C)，0=EEG, 1=MEG_grad, 2=MEG_mag
            
        返回:
            源空间特征 (B, N_neuro, N, T, D)
            - N_neuro: 源数量
            - N: 分段数
            - T: 时间步数（编码后）
            - D: 特征维度
        """
        B, C, N, L = x.shape
        
        # 1. SEANet 编码
        # 将每个通道的每个分段独立编码
        x = rearrange(x, "B C N L -> (B C N) 1 L")
        x = self.seanet_encoder(x)  # (B*C*N, D, T)
        x = rearrange(x, "(B C N) D T -> B C (N T) D", B=B, C=C, N=N)
        
        B, C, W, _ = x.shape  # W = N * T
        
        # 2. 计算传感器嵌入
        sensor_embedding = self.sensor_module(pos, sensor_type)  # (B, C, D)
        
        # 3. 扩展传感器嵌入到所有时间步
        sensor_embedding_expanded = rearrange(
            sensor_embedding.unsqueeze(2).repeat(1, 1, W, 1), 
            "B C W D -> (B W) C D"
        )
        x = rearrange(x, "B C W D -> (B W) C D")
        
        # 4. 反向求解：传感器空间 -> 源空间
        neuros = self.neuros.type_as(x).unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.backwardsolution(
            neuros, 
            self.k_proj(x + sensor_embedding_expanded), 
            x
        )
        
        # 5. 重排列输出格式
        x = rearrange(x, "(B N T) C D -> B C (N T) D", B=B, N=N)
        return rearrange(x, "B C (N T) D -> B C N T D", N=N)


class PENCIEncoder(nn.Module):
    """
    PENCI 编码器（简化版本）
    
    针对 PENCI 项目优化的编码器，直接接受标准输入格式。
    
    输入格式适配 /work/2024/tanzunsheng/PENCIData 数据：
    - x: (C, T) 或 (B, C, T)
    - pos: (C, 6) 或 (B, C, 6)
    - sensor_type: (C,) 或 (B, C)
    
    参数:
        config: 配置字典或 OmegaConf 对象
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
        window_size: int = 320,  # SEANet 编码窗口大小
    ):
        super().__init__()
        
        if ratios is None:
            ratios = [8, 4, 2]  # 总下采样 64x
            
        self.window_size = window_size
        self.n_neuro = n_neuro
        self.n_dim = n_dim
        
        # 核心编码器
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
    
    def forward(
        self, 
        x: torch.Tensor, 
        pos: torch.Tensor, 
        sensor_type: torch.Tensor
    ) -> torch.Tensor:
        """
        参数:
            x: EEG 信号 (B, C, T) 或 (C, T)
            pos: 电极位置 (B, C, 6) 或 (C, 6)
            sensor_type: 传感器类型 (B, C) 或 (C,)
            
        返回:
            源空间特征 (B, N_neuro, N, T', D)
        """
        # 处理无 batch 维度的输入
        if x.dim() == 2:
            x = x.unsqueeze(0)
            pos = pos.unsqueeze(0)
            sensor_type = sensor_type.unsqueeze(0)
        
        B, C, T = x.shape
        
        # 将时间序列分段
        N = T // self.window_size
        if N == 0:
            N = 1
            # 填充到 window_size
            pad_len = self.window_size - T
            x = torch.nn.functional.pad(x, (0, pad_len))
        
        # 截断到整数倍
        T_truncated = N * self.window_size
        x = x[:, :, :T_truncated]
        
        # 重塑为 (B, C, N, L) 格式
        x = x.view(B, C, N, self.window_size)
        
        return self.encoder(x, pos, sensor_type)
