# -*- coding: utf-8 -*-
"""
LSTM 模块

来源：移植自 BrainOmni 项目
原始文件：/work/2024/tanzunsheng/Code/BrainOmni/model_utils/lstm.py
原始符号：SLSTM
改动说明：
  - 保持核心实现不变
  - 添加中文注释
"""

from torch import nn


class SLSTM(nn.Module):
    """
    流式 LSTM：不需要担心隐藏状态或数据布局
    
    期望卷积布局的输入：(B, C, T)
    
    参数:
        dimension: 特征维度
        num_layers: LSTM 层数
        skip: 是否使用跳跃连接
        bidirectional: 是否使用双向 LSTM
    """

    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        skip: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.skip = skip
        self.lstm = nn.LSTM(
            dimension, dimension, num_layers, bidirectional=bidirectional
        )

    def forward(self, x):
        # x: (B, C, T) -> (T, B, C)
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.bidirectional:
            x = x.repeat(1, 1, 2)
        if self.skip:
            y = y + x
        # y: (T, B, C) -> (B, C, T)
        y = y.permute(1, 2, 0)
        return y
