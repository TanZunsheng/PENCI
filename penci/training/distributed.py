# -*- coding: utf-8 -*-
"""
共享分布式训练辅助函数。
"""

import os
from datetime import timedelta
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


def setup_distributed() -> Tuple[int, int, int]:
    """
    检测 torchrun 环境并初始化分布式训练。
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=60),
        )
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def sync_ranks(world_size: int, local_rank: int) -> None:
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])


def reduce_metric(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor


def reduce_metric_sum(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
