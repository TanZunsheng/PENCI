# -*- coding: utf-8 -*-
"""
动力学核心模块（原创）

PENCI 的核心创新：使用 Transformer 建模神经动力学演化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from penci.modules.attention import RMSNorm, FeedForward, SelfAttnBlock


class DynamicsCore(nn.Module):
    """动力学核心：使用 Transformer 建模源空间时间演化"""
    
    def __init__(
        self,
        n_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        causal: bool = True,
    ):
        super().__init__()
        
        self.n_dim = n_dim
        self.n_layers = n_layers
        self.causal = causal
        self.max_seq_len = max_seq_len
        
        self.layers = nn.ModuleList([
            SelfAttnBlock(
                n_dim=n_dim,
                n_head=n_heads,
                dropout=dropout,
                causal=causal,
                rope=True,
            )
            for _ in range(n_layers)
        ])
        
        self.final_norm = RMSNorm(n_dim)
        
    def forward(
        self, 
        x: torch.Tensor,
        return_all_layers: bool = False,
    ) -> torch.Tensor:
        """
        x: (B, T, D) 或 (B, N_neuro, T, D)
        返回: 相同形状
        """
        reshape_4d = False
        B, N_neuro = 0, 0
        if x.dim() == 4:
            reshape_4d = True
            B, N_neuro, T, D = x.shape
            x = rearrange(x, "B N T D -> (B N) T D")
        
        all_outputs = []
        for layer in self.layers:
            x = layer(x)
            if return_all_layers:
                all_outputs.append(x)
        
        x = self.final_norm(x)
        
        if reshape_4d:
            x = rearrange(x, "(B N) T D -> B N T D", B=B, N=N_neuro)
            if return_all_layers:
                all_outputs = [rearrange(o, "(B N) T D -> B N T D", B=B, N=N_neuro) for o in all_outputs]
        
        if return_all_layers:
            return x, all_outputs
        return x


class DynamicsRNN(nn.Module):
    """动力学 RNN：轻量级替代方案"""
    
    def __init__(
        self,
        n_dim: int = 256,
        hidden_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        rnn_type: str = "lstm",
    ):
        super().__init__()
        
        self.n_dim = n_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        self.input_proj = nn.Linear(n_dim, hidden_dim)
        
        rnn_class = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_class(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Linear(output_dim, n_dim)
        
        self.norm = RMSNorm(n_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D) 或 (B, N_neuro, T, D)
        返回: 相同形状
        """
        reshape_4d = False
        B, N_neuro = 0, 0
        if x.dim() == 4:
            reshape_4d = True
            B, N_neuro, T, D = x.shape
            x = rearrange(x, "B N T D -> (B N) T D")
        
        x = self.input_proj(x)
        x, _ = self.rnn(x)
        x = self.output_proj(x)
        x = self.norm(x)
        
        if reshape_4d:
            x = rearrange(x, "(B N) T D -> B N T D", B=B, N=N_neuro)
        
        return x
