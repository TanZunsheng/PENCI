# -*- coding: utf-8 -*-
"""
注意力机制相关模块

来源：移植自 BrainOmni 项目
原始文件：/work/2024/tanzunsheng/Code/BrainOmni/model_utils/attn.py
原始符号：RMSNorm, FeedForward, SelfAttention, RotaryEmbedding
改动说明：
  - 保持核心实现不变
  - 添加中文注释
  - 移除 SpatialTemporalAttentionBlock（PENCI 不使用）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Identity(nn.Module):
    """恒等映射模块"""
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        return args


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization
    比 LayerNorm 更简单高效的归一化方法
    """
    def __init__(self, n_dim, elementwise_affine=True, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim)) if elementwise_affine else 1.0
        self.eps = eps

    def forward(self, x: torch.Tensor):
        weight = self.weight
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (weight * x).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE)
    用于在自注意力中编码相对位置信息
    """
    def __init__(self, n_dim, init_seq_len, base=10000):
        super().__init__()
        self.register_buffer(
            "freqs",
            1.0 / (base ** (torch.arange(0, n_dim, 2)[: (n_dim // 2)].float() / n_dim)),
        )
        self._set_rotate_cache(init_seq_len)

    def _set_rotate_cache(self, seq_len):
        self.max_seq_len_cache = seq_len
        t = torch.arange(seq_len, device=self.freqs.device).type_as(self.freqs)
        rotate = torch.outer(t, self.freqs).float()
        self.register_buffer("rotate", torch.polar(torch.ones_like(rotate), rotate))

    def reshape_for_broadcast(self, x: torch.Tensor):
        """
        x: (Batch, seq, n_head, d_head)
        rotate: (seq, dim)
        """
        B, T, H, D = x.shape
        if T > self.max_seq_len_cache:
            self._set_rotate_cache(T)
        rotate = self.rotate[:T, :]
        assert H * D == rotate.shape[1]
        return rearrange(rotate, "T (H D)-> T H D", H=H).unsqueeze(0)

    def forward(self, q, k):
        assert len(q.shape) == len(k.shape) == 4
        q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        rotate = self.reshape_for_broadcast(q_)
        q_out = torch.view_as_real(q_ * rotate).flatten(3)
        k_out = torch.view_as_real(k_ * rotate).flatten(3)
        return q_out.type_as(q), k_out.type_as(k)


class SelfAttnBlock(nn.Module):
    """
    自注意力块：包含注意力层和前馈网络
    """
    def __init__(self, n_dim, n_head, dropout, causal, rope):
        super().__init__()
        self.pre_attn_norm = RMSNorm(n_dim)
        self.attn = SelfAttention(n_dim, n_head, dropout, causal=causal, rope=rope)
        self.pre_ff_norm = RMSNorm(n_dim)
        self.ff = FeedForward(n_dim, dropout)

    def forward(self, x, mask=None):
        """
        参数:
            x: 输入张量
            mask: 注意力掩码，True 表示参与注意力计算
        """
        x = x + self.attn(self.pre_attn_norm(x), mask)
        x = x + self.ff(self.pre_ff_norm(x))
        return x


class SelfAttention(nn.Module):
    """
    自注意力机制
    
    参数:
        n_dim: 特征维度
        n_head: 注意力头数
        dropout: dropout 比率
        causal: 是否使用因果掩码
        rope: 是否使用旋转位置编码
    """
    def __init__(
        self, n_dim, n_head, dropout, causal: bool = False, rope: bool = False
    ):
        super().__init__()
        assert n_dim % n_head == 0
        self.dropout = dropout
        self.n_dim = n_dim
        self.n_head = n_head
        self.causal = causal
        self.qkv = nn.Linear(n_dim, 3 * n_dim)
        self.proj = nn.Linear(n_dim, n_dim)
        self.rope = rope
        self.rope_embedding_layer = (
            RotaryEmbedding(n_dim=n_dim, init_seq_len=240) if self.rope else Identity()
        )

    def forward(self, x: torch.Tensor, mask=None):
        """
        参数:
            x: 输入张量 (B, T, C)
            mask: 注意力掩码，True 表示参与注意力计算
        """
        B, T, C = x.shape
        x = self.qkv(x)
        q, k, v = torch.split(x, split_size_or_sections=self.n_dim, dim=-1)

        # 有无 rope 对形状变换有影响
        if self.rope:
            q = q.view(B, T, self.n_head, -1)
            k = k.view(B, T, self.n_head, -1)
            q, k = self.rope_embedding_layer(q, k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
        else:
            q = rearrange(q, "B T (H D) -> B H T D", H=self.n_head)
            k = rearrange(k, "B T (H D) -> B H T D", H=self.n_head)

        v = rearrange(v, "B T (H D) -> B H T D", H=self.n_head)

        # 添加 head_dim
        if mask is not None:
            mask = mask.unsqueeze(1)

        output = (
            F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )
            .transpose(1, 2)
            .contiguous()
        )
        output = output.view(B, T, -1)
        return self.proj(output)


class FeedForward(nn.Module):
    """
    前馈网络：两层全连接 + SELU 激活
    """
    def __init__(self, n_dim, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_dim, int(4 * n_dim)),
            nn.SELU(),
            nn.Linear(int(4 * n_dim), n_dim),
            nn.Dropout(dropout) if dropout != 0.0 else nn.Identity(),
        )

    def forward(self, x):
        return self.layer(x)
