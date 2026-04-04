#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PENCI V1 第二层训练脚本

流程:
  1. 加载并冻结第一层 Stage1 模型
  2. 导出 S_t（默认 stop-gradient）
  3. 训练 StaticConnectivityModel (p=2)
  4. 训练后导出 A_base
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


def build_loaders(config: Dict) -> Tuple[DataLoader, Optional[DataLoader]]:
    data_cfg = config.get("data", {})
    train_meta = data_cfg.get("stage2_sim_train_metadata")
    val_meta = data_cfg.get("stage2_sim_val_metadata")
    if not train_meta:
        raise RuntimeError("需要 data.stage2_sim_train_metadata")

    batch_size = config.get("training", {}).get("batch_size", 8)
    num_workers = config.get("training", {}).get("num_workers", 0)

    train_ds = Stage2ConnectivitySimulationDataset(metadata_path=train_meta)
    val_ds = Stage2ConnectivitySimulationDataset(metadata_path=val_meta) if val_meta else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if val_ds is not None
        else None
    )
    return train_loader, val_loader


def load_stage1_frozen(config: Dict, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_stage1_model_from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def compute_stage2_batch_loss(
    stage1_model: torch.nn.Module,
    stage2_model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    loss_weights: Dict[str, float],
    use_ground_truth_state: bool,
    max_radius: float,
) -> Dict[str, torch.Tensor]:
    x = batch["x"].to(device)
    pos = batch["pos"].to(device)
    sensor_type = batch["sensor_type"].to(device)
    leadfield = batch["leadfield"].to(device)

    if use_ground_truth_state:
        source_state = batch["s_true"].to(device).detach()
    else:
        with torch.no_grad():
            source_state = stage1_model(
                x,
                pos,
                sensor_type,
                leadfield=leadfield,
                target_length=x.shape[-1],
                return_features=False,
            )["source_state"].detach()

    pred_next, target_next = stage2_model.predict_next(source_state)
    state_prediction_loss = F.mse_loss(pred_next, target_next)

    pred_sensor = stage1_model.decoder.project_source_state(pred_next, leadfield=leadfield)
    target_sensor = stage1_model.decoder.project_source_state(target_next, leadfield=leadfield)
    next_sensor_loss = F.mse_loss(pred_sensor, target_sensor)

    l1_loss = stage2_model.l1_sparsity_loss()
    stability_loss = stage2_model.stability_penalty(max_radius=max_radius)
    total_loss = (
        loss_weights.get("state_prediction", 1.0) * state_prediction_loss
        + loss_weights.get("next_sensor", 1.0) * next_sensor_loss
        + loss_weights.get("l1_sparsity", 1e-3) * l1_loss
        + loss_weights.get("stability", 1e-2) * stability_loss
    )
    return {
        "loss": total_loss,
        "state_prediction_loss": state_prediction_loss,
        "next_sensor_loss": next_sensor_loss,
        "l1_sparsity_loss": l1_loss,
        "stability_penalty": stability_loss,
    }


def train_one_epoch(
    stage1_model: torch.nn.Module,
    stage2_model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: Dict[str, float],
    use_ground_truth_state: bool,
    max_radius: float,
) -> Dict[str, float]:
    stage2_model.train()
    sums = {
        "loss": 0.0,
        "state_prediction_loss": 0.0,
        "next_sensor_loss": 0.0,
        "l1_sparsity_loss": 0.0,
        "stability_penalty": 0.0,
    }
    count = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss_dict = compute_stage2_batch_loss(
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            batch=batch,
            device=device,
            loss_weights=loss_weights,
            use_ground_truth_state=use_ground_truth_state,
            max_radius=max_radius,
        )
        loss_dict["loss"].backward()
        optimizer.step()

        for key in sums:
            sums[key] += float(loss_dict[key].item())
        count += 1

    return {k: (v / max(1, count)) for k, v in sums.items()}


@torch.no_grad()
def eval_one_epoch(
    stage1_model: torch.nn.Module,
    stage2_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_weights: Dict[str, float],
    use_ground_truth_state: bool,
    max_radius: float,
) -> Dict[str, float]:
    stage2_model.eval()
    sums = {
        "loss": 0.0,
        "state_prediction_loss": 0.0,
        "next_sensor_loss": 0.0,
        "l1_sparsity_loss": 0.0,
        "stability_penalty": 0.0,
    }
    count = 0

    for batch in loader:
        loss_dict = compute_stage2_batch_loss(
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            batch=batch,
            device=device,
            loss_weights=loss_weights,
            use_ground_truth_state=use_ground_truth_state,
            max_radius=max_radius,
        )
        for key in sums:
            sums[key] += float(loss_dict[key].item())
        count += 1

    return {k: (v / max(1, count)) for k, v in sums.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Stage2 connectivity model")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--stage1_checkpoint", type=str, required=True, help="冻结第一层 checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="第二层继续训练 checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config)
    train_loader, val_loader = build_loaders(config)

    stage1_model = load_stage1_frozen(config, args.stage1_checkpoint, device)
    stage2_model = build_stage2_model_from_config(config).to(device)

    training_cfg = config.get("training", {})
    loss_weights = training_cfg.get(
        "loss",
        {
            "state_prediction": 1.0,
            "next_sensor": 1.0,
            "l1_sparsity": 1e-3,
            "stability": 1e-2,
        },
    )
    epochs = training_cfg.get("epochs", 20)
    lr = training_cfg.get("learning_rate", 5e-4)
    weight_decay = training_cfg.get("weight_decay", 0.0)
    max_radius = training_cfg.get("max_radius", 0.98)
    use_ground_truth_state = bool(training_cfg.get("use_ground_truth_state", False))

    optimizer = torch.optim.AdamW(stage2_model.parameters(), lr=lr, weight_decay=weight_decay)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        stage2_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    output_dir = Path(args.output_dir or config.get("paths", {}).get("output_dir", "outputs/stage2"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "stage2_best.pt"
    last_ckpt = output_dir / "stage2_last.pt"
    a_base_path = output_dir / "a_base.pt"
    metrics_path = output_dir / "stage2_metrics.json"

    best_score = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_weights=loss_weights,
            use_ground_truth_state=use_ground_truth_state,
            max_radius=max_radius,
        )
        val_metrics = (
            eval_one_epoch(
                stage1_model=stage1_model,
                stage2_model=stage2_model,
                loader=val_loader,
                device=device,
                loss_weights=loss_weights,
                use_ground_truth_state=use_ground_truth_state,
                max_radius=max_radius,
            )
            if val_loader is not None
            else {"loss": train_metrics["loss"]}
        )

        clip_info = stage2_model.apply_spectral_radius_clip(max_radius=max_radius)
        score = val_metrics["loss"] if val_loader is not None else train_metrics["loss"]
        logger.info(
            "Epoch %d/%d | train=%.6f | val=%.6f | rho_before=%.4f | rho_after=%.4f",
            epoch,
            epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            clip_info["before"],
            clip_info["after"],
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": stage2_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "clip_info": clip_info,
            "a_base": stage2_model.export_a_base().detach().cpu(),
            "config": config,
        }
        torch.save(ckpt, last_ckpt)
        if score < best_score:
            best_score = score
            torch.save(ckpt, best_ckpt)
            torch.save({"a_base": ckpt["a_base"]}, a_base_path)

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "clip_info": clip_info,
            }
        )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"best_score": best_score, "history": history}, f, ensure_ascii=False, indent=2)

    logger.info("Stage2 训练完成 | best=%.6f | A_base=%s", best_score, a_base_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
