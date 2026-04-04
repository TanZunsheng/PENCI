#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
兼容入口：旧版单阶段训练脚本。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from penci.training.prefetch import (  # noqa: F401
    NodePageCachePrefetcher,
    build_node_union_prefetch_plan,
    get_prefetch_file_plan,
    get_prefetch_rank_schedule,
    warmup_hdf5_page_cache,
)
from scripts.legacy.train import main

__all__ = [
    "NodePageCachePrefetcher",
    "build_node_union_prefetch_plan",
    "get_prefetch_file_plan",
    "get_prefetch_rank_schedule",
    "warmup_hdf5_page_cache",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
