# -*- coding: utf-8 -*-
"""
兼容入口：PENCI V1 仿真数据集。
"""

from penci.v1.data.simulation_dataset import (
    Stage1SimulationDataset,
    Stage2ConnectivitySimulationDataset,
)

__all__ = [
    "Stage1SimulationDataset",
    "Stage2ConnectivitySimulationDataset",
]
