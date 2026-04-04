#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PENCI V1 第二层评估脚本

评估内容:
  - A_true 恢复误差
  - 拓扑恢复指标 (edge F1)
  - 方向恢复指标
  - VAR(p=2) 基线对比
  - 时间反转 sanity check
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from penci.v1.data import Stage2ConnectivitySimulationDataset
from penci.v1.models import build_stage1_model_from_config, build_stage2_model_from_config
from penci.utils.state_metrics import compute_connectivity_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(config: Dict) -> torch.device:
    requested = config.get("hardware", {}).get("device", "cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loader(config: Dict) -> DataLoader:
    data_cfg = config.get("data", {})
    meta = data_cfg.get("stage2_sim_eval_metadata") or data_cfg.get("stage2_sim_val_metadata")
    if not meta:
        raise RuntimeError("需要 data.stage2_sim_eval_metadata 或 stage2_sim_val_metadata")
    dataset = Stage2ConnectivitySimulationDataset(metadata_path=meta)
    batch_size = config.get("evaluation", {}).get("batch_size", config.get("training", {}).get("batch_size", 8))
    num_workers = config.get("evaluation", {}).get("num_workers", 0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def load_stage1_frozen(config: Dict, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    stage1 = build_stage1_model_from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    stage1.load_state_dict(state_dict, strict=False)
    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad = False
    return stage1


def load_stage2(config: Dict, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    stage2 = build_stage2_model_from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    stage2.load_state_dict(state_dict, strict=False)
    stage2.eval()
    return stage2


def fit_var_p2(source_state: torch.Tensor, ridge: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    用最小二乘拟合 VAR(p=2) 基线。

    参数:
        source_state: (B, N, T)
    返回:
        (A1, A2): 均为 (N, N)
    """
    batch_size, n, t = source_state.shape
    if t < 3:
        raise ValueError("source_state 时间长度至少为 3")

    y = source_state[:, :, 2:]  # (B, N, T-2)
    x1 = source_state[:, :, 1:-1]
    x2 = source_state[:, :, :-2]
    x = torch.cat([x1, x2], dim=1)  # (B, 2N, T-2)

    y = y.permute(0, 2, 1).reshape(-1, n)  # (B*(T-2), N)
    x = x.permute(0, 2, 1).reshape(-1, 2 * n)  # (B*(T-2), 2N)

    xt = x.transpose(0, 1)
    gram = xt @ x
    gram = gram + ridge * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    w = torch.linalg.solve(gram, xt @ y).transpose(0, 1)  # (N, 2N)
    a1 = w[:, :n]
    a2 = w[:, n:]
    return a1, a2


def pdc_proxy_from_a(a_base: torch.Tensor) -> torch.Tensor:
    """
    PDC 近似代理：按列归一化的系数强度。
    """
    denom = torch.sqrt(torch.sum(a_base ** 2, dim=0, keepdim=True)) + 1e-8
    return a_base / denom


@torch.no_grad()
def evaluate(
    stage1: torch.nn.Module,
    stage2: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_ground_truth_state: bool,
    threshold: float,
) -> Dict[str, float]:
    sums = {
        "state_prediction_loss": 0.0,
        "next_sensor_loss": 0.0,
        "time_reversal_ratio": 0.0,
        "connectivity_relative_error": 0.0,
        "connectivity_edge_f1": 0.0,
        "connectivity_direction_accuracy": 0.0,
        "var_connectivity_relative_error": 0.0,
        "var_connectivity_edge_f1": 0.0,
        "var_connectivity_direction_accuracy": 0.0,
        "granger_connectivity_relative_error": 0.0,
        "granger_connectivity_edge_f1": 0.0,
        "granger_connectivity_direction_accuracy": 0.0,
        "pdc_connectivity_relative_error": 0.0,
        "pdc_connectivity_edge_f1": 0.0,
        "pdc_connectivity_direction_accuracy": 0.0,
    }
    count = 0
    count_a_true = 0

    a_model = stage2.export_a_base().detach()
    pdc_model = pdc_proxy_from_a(a_model)

    for batch in loader:
        x = batch["x"].to(device)
        pos = batch["pos"].to(device)
        sensor_type = batch["sensor_type"].to(device)
        leadfield = batch["leadfield"].to(device)
        s_true = batch["s_true"].to(device)
        a_true = batch.get("a_true")
        if a_true is not None:
            a_true = a_true.to(device)

        if use_ground_truth_state:
            source_state = s_true
        else:
            source_state = stage1(
                x,
                pos,
                sensor_type,
                leadfield=leadfield,
                target_length=x.shape[-1],
                return_features=False,
            )["source_state"]

        pred_next, target_next = stage2.predict_next(source_state)
        state_prediction_loss = F.mse_loss(pred_next, target_next)
        pred_sensor = stage1.decoder.project_source_state(pred_next, leadfield=leadfield)
        target_sensor = stage1.decoder.project_source_state(target_next, leadfield=leadfield)
        next_sensor_loss = F.mse_loss(pred_sensor, target_sensor)

        reverse_state = torch.flip(source_state, dims=[-1])
        reverse_pred, reverse_target = stage2.predict_next(reverse_state)
        reverse_loss = F.mse_loss(reverse_pred, reverse_target)
        time_reversal_ratio = float((reverse_loss / (state_prediction_loss + 1e-8)).item())

        sums["state_prediction_loss"] += float(state_prediction_loss.item())
        sums["next_sensor_loss"] += float(next_sensor_loss.item())
        sums["time_reversal_ratio"] += time_reversal_ratio
        count += 1

        if a_true is not None:
            for i in range(a_true.shape[0]):
                metrics = compute_connectivity_metrics(
                    a_true=a_true[i],
                    a_pred=a_model,
                    threshold=threshold,
                )
                sums["connectivity_relative_error"] += float(metrics["connectivity_relative_error"].item())
                sums["connectivity_edge_f1"] += float(metrics["connectivity_edge_f1"].item())
                sums["connectivity_direction_accuracy"] += float(
                    metrics["connectivity_direction_accuracy"].item()
                )

                var_a1, var_a2 = fit_var_p2(source_state[i : i + 1])
                var_a_base = var_a1 + var_a2
                granger_proxy = var_a_base
                pdc_proxy = pdc_proxy_from_a(var_a_base)
                var_metrics = compute_connectivity_metrics(
                    a_true=a_true[i],
                    a_pred=var_a_base,
                    threshold=threshold,
                )
                sums["var_connectivity_relative_error"] += float(
                    var_metrics["connectivity_relative_error"].item()
                )
                sums["var_connectivity_edge_f1"] += float(var_metrics["connectivity_edge_f1"].item())
                sums["var_connectivity_direction_accuracy"] += float(
                    var_metrics["connectivity_direction_accuracy"].item()
                )

                granger_metrics = compute_connectivity_metrics(
                    a_true=a_true[i],
                    a_pred=granger_proxy,
                    threshold=threshold,
                )
                sums["granger_connectivity_relative_error"] += float(
                    granger_metrics["connectivity_relative_error"].item()
                )
                sums["granger_connectivity_edge_f1"] += float(
                    granger_metrics["connectivity_edge_f1"].item()
                )
                sums["granger_connectivity_direction_accuracy"] += float(
                    granger_metrics["connectivity_direction_accuracy"].item()
                )

                pdc_metrics = compute_connectivity_metrics(
                    a_true=a_true[i],
                    a_pred=pdc_proxy,
                    threshold=threshold,
                )
                sums["pdc_connectivity_relative_error"] += float(
                    pdc_metrics["connectivity_relative_error"].item()
                )
                sums["pdc_connectivity_edge_f1"] += float(pdc_metrics["connectivity_edge_f1"].item())
                sums["pdc_connectivity_direction_accuracy"] += float(
                    pdc_metrics["connectivity_direction_accuracy"].item()
                )
                count_a_true += 1

    if count == 0:
        return {k: float("nan") for k in sums}

    results = {k: v / count for k, v in sums.items()}
    if count_a_true == 0:
        results["connectivity_relative_error"] = float("nan")
        results["connectivity_edge_f1"] = float("nan")
        results["connectivity_direction_accuracy"] = float("nan")
        results["var_connectivity_relative_error"] = float("nan")
        results["var_connectivity_edge_f1"] = float("nan")
        results["var_connectivity_direction_accuracy"] = float("nan")
        results["granger_connectivity_relative_error"] = float("nan")
        results["granger_connectivity_edge_f1"] = float("nan")
        results["granger_connectivity_direction_accuracy"] = float("nan")
        results["pdc_connectivity_relative_error"] = float("nan")
        results["pdc_connectivity_edge_f1"] = float("nan")
        results["pdc_connectivity_direction_accuracy"] = float("nan")
    else:
        results["connectivity_relative_error"] = sums["connectivity_relative_error"] / count_a_true
        results["connectivity_edge_f1"] = sums["connectivity_edge_f1"] / count_a_true
        results["connectivity_direction_accuracy"] = (
            sums["connectivity_direction_accuracy"] / count_a_true
        )
        results["var_connectivity_relative_error"] = (
            sums["var_connectivity_relative_error"] / count_a_true
        )
        results["var_connectivity_edge_f1"] = sums["var_connectivity_edge_f1"] / count_a_true
        results["var_connectivity_direction_accuracy"] = (
            sums["var_connectivity_direction_accuracy"] / count_a_true
        )
        results["granger_connectivity_relative_error"] = (
            sums["granger_connectivity_relative_error"] / count_a_true
        )
        results["granger_connectivity_edge_f1"] = (
            sums["granger_connectivity_edge_f1"] / count_a_true
        )
        results["granger_connectivity_direction_accuracy"] = (
            sums["granger_connectivity_direction_accuracy"] / count_a_true
        )
        results["pdc_connectivity_relative_error"] = (
            sums["pdc_connectivity_relative_error"] / count_a_true
        )
        results["pdc_connectivity_edge_f1"] = sums["pdc_connectivity_edge_f1"] / count_a_true
        results["pdc_connectivity_direction_accuracy"] = (
            sums["pdc_connectivity_direction_accuracy"] / count_a_true
        )

    results["model_pdc_l1"] = float(torch.mean(torch.abs(pdc_model)).item())
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Stage2 connectivity model")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--stage1_checkpoint", type=str, required=True, help="冻结第一层 checkpoint")
    parser.add_argument("--stage2_checkpoint", type=str, required=True, help="第二层 checkpoint")
    parser.add_argument("--output", type=str, default=None, help="评估结果输出路径")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config)
    loader = build_loader(config)

    eval_cfg = config.get("evaluation", {})
    use_ground_truth_state = bool(eval_cfg.get("use_ground_truth_state", False))
    threshold = float(eval_cfg.get("connectivity_threshold", 1e-4))

    stage1 = load_stage1_frozen(config, args.stage1_checkpoint, device)
    stage2 = load_stage2(config, args.stage2_checkpoint, device)
    results = evaluate(
        stage1=stage1,
        stage2=stage2,
        loader=loader,
        device=device,
        use_ground_truth_state=use_ground_truth_state,
        threshold=threshold,
    )
    logger.info("Stage2 评估结果: %s", results)

    output_path = Path(
        args.output or config.get("paths", {}).get("stage2_eval_output", "outputs/stage2_eval.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("评估结果已保存到: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
