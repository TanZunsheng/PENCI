# -*- coding: utf-8 -*-
"""
共享训练基础设施。
"""

from penci.training.distributed import (
    cleanup_distributed,
    is_main_process,
    reduce_metric,
    reduce_metric_sum,
    setup_distributed,
    sync_ranks,
    unwrap_model,
)
from penci.training.physics import resolve_leadfield_for_batch, setup_physics
from penci.training.prefetch import (
    NodePageCachePrefetcher,
    build_node_union_prefetch_plan,
    get_prefetch_file_plan,
    get_prefetch_rank_schedule,
    warmup_hdf5_page_cache,
)

__all__ = [
    "cleanup_distributed",
    "is_main_process",
    "reduce_metric",
    "reduce_metric_sum",
    "setup_distributed",
    "sync_ranks",
    "unwrap_model",
    "resolve_leadfield_for_batch",
    "setup_physics",
    "NodePageCachePrefetcher",
    "build_node_union_prefetch_plan",
    "get_prefetch_file_plan",
    "get_prefetch_rank_schedule",
    "warmup_hdf5_page_cache",
]
