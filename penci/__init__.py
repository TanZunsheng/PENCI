# -*- coding: utf-8 -*-
"""
PENCI: Physics-constrained End-to-end Neural Connectivity Inference
物理约束端到端神经连接推断框架

当前主线为 V1 两阶段建模：
1. Stage1: 从 EEG 恢复显式脑区状态 S_t，并通过物理投影闭环
2. Stage2: 在冻结的 S_t 上学习静态有效连接 A_base
3. 共享层: 编码器、动力学模块、物理解码器与训练基础设施
"""

__version__ = "0.2.0"
__author__ = "PENCI Team"

from penci.shared.models import DynamicsCore, DynamicsRNN, PhysicsDecoder, SEANetPhysicsDecoder
from penci.v1 import (
    Stage1Model,
    Stage1SimulationDataset,
    Stage2ConnectivitySimulationDataset,
    StateHead,
    StaticConnectivityModel,
    build_stage1_model_from_config,
    build_stage2_model_from_config,
)

__all__ = [
    "DynamicsCore",
    "DynamicsRNN",
    "PhysicsDecoder",
    "SEANetPhysicsDecoder",
    "Stage1Model",
    "StateHead",
    "StaticConnectivityModel",
    "Stage1SimulationDataset",
    "Stage2ConnectivitySimulationDataset",
    "build_stage1_model_from_config",
    "build_stage2_model_from_config",
]
