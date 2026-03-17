# -*- coding: utf-8 -*-
"""
传感器空间指标的鲁棒性测试。
"""

import math

import torch

from penci.utils.metrics import compute_all_metrics


def test_metrics_skip_low_energy_channels():
    """全 0 通道应被 SNR/NRMSE 跳过，不污染正常通道统计。"""
    x = torch.tensor(
        [
            [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        ]
    )
    x_hat = torch.zeros_like(x)

    metrics = compute_all_metrics(x, x_hat)

    assert metrics["snr_db_count"].item() == 3.0
    assert metrics["snr_db_skipped"].item() == 1.0
    assert math.isfinite(metrics["snr_db"].item())
    assert abs(metrics["snr_db"].item()) < 1e-6

    assert metrics["nrmse_count"].item() == 3.0
    assert metrics["nrmse_skipped"].item() == 1.0
    assert math.isfinite(metrics["nrmse"].item())
    assert abs(metrics["nrmse"].item() - 1.0) < 1e-6


def test_metrics_all_low_energy_channels_return_nan():
    """当所有通道都无有效信号时，SNR/NRMSE 应返回 NaN 而不是 -inf。"""
    x = torch.zeros(2, 3, 8)
    x_hat = torch.ones_like(x)

    metrics = compute_all_metrics(x, x_hat)

    assert metrics["snr_db_count"].item() == 0.0
    assert metrics["snr_db_skipped"].item() == 6.0
    assert math.isnan(metrics["snr_db"].item())

    assert metrics["nrmse_count"].item() == 0.0
    assert metrics["nrmse_skipped"].item() == 6.0
    assert math.isnan(metrics["nrmse"].item())
