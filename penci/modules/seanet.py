# -*- coding: utf-8 -*-
"""
SEANet 编码器模块

来源：移植自 BrainOmni 项目 (基于 Meta Encodec)
原始文件：/work/2024/tanzunsheng/Code/BrainOmni/model_utils/seanet.py
原始符号：SEANetEncoder, SEANetResnetBlock, Snake1d
改动说明：
  - 保持核心实现不变
  - 添加中文注释
  - 移除 SEANetDecoder（PENCI 不使用其解码器）
"""

import typing as tp
import numpy as np
import torch.nn as nn
import torch
from penci.modules.conv import SConv1d
from penci.modules.lstm import SLSTM


@torch.jit.script
def snake(x, alpha):
    """Snake 激活函数"""
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    """
    1D Snake 激活函数
    
    Snake 激活是一种可学习的周期性激活函数，
    对音频/信号处理任务特别有效
    """
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class SEANetResnetBlock(nn.Module):
    """
    SEANet 残差块
    
    参数:
        dim: 输入/输出维度
        kernel_sizes: 卷积核大小列表
        dilations: 膨胀率列表
        activation: 激活函数类型
        activation_params: 激活函数参数
        norm: 归一化方法
        norm_params: 归一化参数
        causal: 是否使用因果卷积
        pad_mode: 填充模式
        compress: 残差分支的压缩比
        true_skip: 是否使用真跳跃连接
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: tp.List[int] = [3, 1],
        dilations: tp.List[int] = [1, 1],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "weight_norm",
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(
            dilations
        ), "卷积核大小数量应与膨胀率数量匹配"
        act = getattr(nn, activation) if activation != "Snake" else Snake1d
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params) if activation != "Snake" else act(in_chs),
                SConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(nn.Module):
    """
    SEANet 编码器
    
    用于将原始 EEG 信号编码为高维特征表示
    
    参数:
        channels: 输入通道数（对于 EEG 单通道为 1）
        dimension: 输出特征维度
        n_filters: 基础滤波器数量
        n_residual_layers: 残差层数量
        ratios: 下采样比例列表（总下采样 = 所有比例的乘积）
        activation: 激活函数
        activation_params: 激活函数参数
        norm: 归一化方法
        norm_params: 归一化参数
        kernel_size: 初始卷积核大小
        last_kernel_size: 最后卷积层核大小
        residual_kernel_size: 残差层核大小
        dilation_base: 膨胀基数
        causal: 是否使用因果卷积
        pad_mode: 填充模式
        true_skip: 是否使用真跳跃连接
        compress: 残差分支压缩比
        lstm: LSTM 层数
        bidirectional: 是否使用双向 LSTM
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "weight_norm",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = False,
        compress: int = 2,
        lstm: int = 2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        # 编码器使用反向的比例（上采样比例的逆序）
        self.ratios = list(reversed(ratios))
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)  # 总下采样倍数

        act = getattr(nn, activation) if activation != "Snake" else Snake1d
        mult = 1
        model: tp.List[nn.Module] = [
            SConv1d(
                channels,
                mult * n_filters,
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]

        # 记录每个下采样阶段的输出索引与通道数，供多尺度特征导出使用
        self.stage_output_indices: tp.Dict[int, int] = {}
        self.stage_output_channels: tp.Dict[int, int] = {}
        stage_factor = 1
        
        # 下采样到原始信号尺度
        for ratio in self.ratios:
            # 添加残差层
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm=norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]
            # 添加下采样层
            model += [
                (
                    act(**activation_params)
                    if activation != "Snake"
                    else act(mult * n_filters)
                ),
                SConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
            stage_factor *= ratio
            self.stage_output_indices[stage_factor] = len(model) - 1
            self.stage_output_channels[stage_factor] = mult * n_filters * 2
            mult *= 2

        if lstm:
            model += [
                SLSTM(mult * n_filters, num_layers=lstm, bidirectional=bidirectional)
            ]

        mult = mult * 2 if bidirectional else mult
        model += [
            (
                act(**activation_params)
                if activation != "Snake"
                else act(mult * n_filters)
            ),
            SConv1d(
                mult * n_filters,
                dimension,
                last_kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]

        self.model = nn.Sequential(*model)
        self.stage_factors = sorted(self.stage_output_indices.keys())

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> tp.Union[torch.Tensor, tp.Dict[str, tp.Any]]:
        """
        参数:
            x: 输入信号 (B, 1, T)
            return_intermediates: 是否返回多尺度中间特征
        返回:
            编码特征 (B, D, T')，其中 T' = T / hop_length
        """
        if not return_intermediates:
            return self.model(x)

        stage_outputs: tp.Dict[str, torch.Tensor] = {}
        stage_by_factor: tp.Dict[int, torch.Tensor] = {}
        current = x
        for idx, layer in enumerate(self.model):
            current = layer(current)
            for factor, stage_idx in self.stage_output_indices.items():
                if idx == stage_idx:
                    key = f"x_{factor}x"
                    stage_outputs[key] = current
                    stage_by_factor[factor] = current

        return {
            "final": current,
            "stage_outputs": stage_outputs,
            "stage_by_factor": stage_by_factor,
            "stage_factors": list(self.stage_factors),
            "hop_length": int(self.hop_length),
        }
