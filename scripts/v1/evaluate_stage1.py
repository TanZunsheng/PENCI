#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PENCI V1 第一层评估脚本

评估内容:
  - 传感器重建指标
  - 状态恢复指标（若有 s_true）
  - 状态平滑性
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from penci.data import PENCICollator, PENCIDataset
from penci.training.physics import resolve_leadfield_for_batch, setup_physics
from penci.v1.data import Stage1SimulationDataset, create_simulation_dataloader
from penci.v1.models import build_stage1_model_from_config
from penci.utils.metrics import compute_all_metrics
from penci.utils.state_metrics import state_mse, state_pearson, state_temporal_smoothness

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _normalize_progress_log_pct(step_pct: float) -> float:
    return min(100.0, max(1.0, float(step_pct)))


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(config: Dict) -> torch.device:
    device_name = config.get("hardware", {}).get("device", "cuda")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_eval_loader(config: Dict, dataset_mode: str) -> DataLoader:
    data_cfg = config.get("data", {})
    batch_size = config.get("evaluation", {}).get(
        "batch_size", config.get("training", {}).get("batch_size", 8)
    )
    num_workers = config.get("evaluation", {}).get("num_workers", 0)

    if dataset_mode == "sim":
        meta = data_cfg.get("stage1_sim_eval_metadata") or data_cfg.get("stage1_sim_val_metadata")
        if not meta:
            raise RuntimeError(
                "sim 模式需要 data.stage1_sim_eval_metadata 或 stage1_sim_val_metadata"
            )
        dataset = Stage1SimulationDataset(metadata_path=meta)
        use_bucket_sampler = bool(data_cfg.get("use_bucket_sampler", False))
        use_fingerprint = bool(data_cfg.get("use_fingerprint", use_bucket_sampler))
        file_scheduler_cfg = data_cfg.get("file_scheduler", {})
        return create_simulation_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            use_bucket_sampler=use_bucket_sampler,
            use_fingerprint=use_fingerprint,
            file_scheduler=bool(file_scheduler_cfg.get("val_enabled", False)),
            shuffle_within_file=bool(file_scheduler_cfg.get("val_shuffle_within_file", False)),
            drop_last=False,
        )

    meta = data_cfg.get("real_eval_metadata") or data_cfg.get("real_val_metadata")
    if not meta:
        raise RuntimeError("real 模式需要 data.real_eval_metadata 或 real_val_metadata")
    dataset = PENCIDataset(
        metadata_path=meta,
        max_length=data_cfg.get("window_length", 2560) * max(1, data_cfg.get("time_window", 10)),
        target_channels=data_cfg.get("n_channels", 128),
        random_crop=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=PENCICollator(),
    )


def load_default_leadfield(config: Dict, device: torch.device) -> Optional[torch.Tensor]:
    physics_cfg = config.get("model", {}).get("physics", {})
    leadfield_path = physics_cfg.get("leadfield_path")
    if not leadfield_path:
        return None
    leadfield = torch.load(leadfield_path, map_location=device)
    if isinstance(leadfield, dict):
        leadfield = leadfield.get("leadfield", leadfield.get("L"))
    return leadfield


def resolve_leadfield(
    batch: Dict[str, torch.Tensor],
    default_leadfield: Optional[torch.Tensor],
    device: torch.device,
    leadfield_manager: Optional[object] = None,
    electrode_registry: Optional[object] = None,
) -> Optional[torch.Tensor]:
    if "leadfield" in batch:
        return batch["leadfield"].to(device)
    if leadfield_manager is not None and electrode_registry is not None:
        return resolve_leadfield_for_batch(
            batch.get("fingerprint", "unknown"),
            leadfield_manager,
            electrode_registry,
            device,
        )
    if default_leadfield is not None:
        return default_leadfield.to(device)
    return None


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    default_leadfield: Optional[torch.Tensor],
    leadfield_manager: Optional[object] = None,
    electrode_registry: Optional[object] = None,
    use_amp: bool = False,
    progress_log_pct: float = 10.0,
) -> Dict[str, float]:
    model.eval()
    progress_log_pct = _normalize_progress_log_pct(progress_log_pct)

    metric_sums = {
        "recon_loss": 0.0,
        "pearson": 0.0,
        "snr_db": 0.0,
        "nrmse": 0.0,
        "state_smoothness": 0.0,
        "state_mse": 0.0,
        "state_pearson": 0.0,
    }
    num_batches = 0
    num_state_batches = 0
    started_at = time.time()
    total_batches: Optional[int] = None
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    next_log_pct = progress_log_pct

    for batch_idx, batch in enumerate(loader, start=1):
        x = batch["x"].to(device)
        pos = batch["pos"].to(device)
        sensor_type = batch["sensor_type"].to(device)
        n_channels = batch.get("n_channels")
        if n_channels is not None:
            n_channels = n_channels.to(device)
        s_true = batch.get("s_true")
        if s_true is not None:
            s_true = s_true.to(device)

        leadfield = resolve_leadfield(
            batch,
            default_leadfield,
            device,
            leadfield_manager=leadfield_manager,
            electrode_registry=electrode_registry,
        )
        with torch.autocast(device_type="cuda", enabled=use_amp):
            output = model(
                x,
                pos,
                sensor_type,
                leadfield=leadfield,
                target_length=x.shape[-1],
                return_features=False,
            )
        recon = output["reconstruction"]
        state = output["source_state"]
        target = model._prepare_target(x, recon.shape[-1])
        target, recon = model.align_sensor_space(
            target,
            recon,
            n_channels=n_channels,
        )
        recon_metrics = compute_all_metrics(target, recon, n_channels=n_channels)

        metric_sums["recon_loss"] += float(
            model._masked_sensor_mse(recon, target, n_channels=n_channels).item()
        )
        metric_sums["pearson"] += float(recon_metrics["pearson"].item())
        metric_sums["snr_db"] += float(recon_metrics["snr_db"].item())
        metric_sums["nrmse"] += float(recon_metrics["nrmse"].item())
        metric_sums["state_smoothness"] += float(state_temporal_smoothness(state).item())
        num_batches += 1

        if s_true is not None:
            s_target = model._prepare_state_target(s_true, state.shape[-1])
            metric_sums["state_mse"] += float(state_mse(s_target, state).item())
            metric_sums["state_pearson"] += float(state_pearson(s_target, state).item())
            num_state_batches += 1

        if total_batches and total_batches > 0:
            progress_pct = min(100.0, (batch_idx / total_batches) * 100.0)
            while progress_pct + 1e-6 >= next_log_pct:
                elapsed = max(1e-6, time.time() - started_at)
                batches_per_sec = batch_idx / elapsed
                remain_batches = max(0, total_batches - batch_idx)
                eta_min = (
                    0.0 if batches_per_sec <= 1e-6 else remain_batches / batches_per_sec / 60.0
                )
                logger.info(
                    "[评估进度] 已达到 %.0f%% | batch %d/%d | %.2f batch/s | ETA %.1f min",
                    next_log_pct,
                    batch_idx,
                    total_batches,
                    batches_per_sec,
                    eta_min,
                )
                next_log_pct += progress_log_pct

    if num_batches == 0:
        return {key: float("nan") for key in metric_sums}

    results = {key: value / num_batches for key, value in metric_sums.items()}
    if num_state_batches == 0:
        results["state_mse"] = float("nan")
        results["state_pearson"] = float("nan")
    else:
        results["state_mse"] = metric_sums["state_mse"] / num_state_batches
        results["state_pearson"] = metric_sums["state_pearson"] / num_state_batches
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Stage1 model")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="sim",
        choices=["sim", "real"],
        help="评估数据类型",
    )
    parser.add_argument("--output", type=str, default=None, help="评估结果输出 JSON 路径")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config)
    loader = build_eval_loader(config, args.dataset_mode)
    model = build_stage1_model_from_config(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    default_leadfield = load_default_leadfield(config, device)
    leadfield_manager = None
    electrode_registry = None
    if args.dataset_mode == "real" and default_leadfield is None:
        _, leadfield_manager, electrode_registry, _ = setup_physics(config)
    use_amp = bool(
        config.get("evaluation", {}).get("use_amp", config.get("training", {}).get("use_amp", True))
    )
    progress_log_pct = float(config.get("evaluation", {}).get("progress_log_pct", 10.0))

    results = evaluate(
        model,
        loader,
        device,
        default_leadfield,
        leadfield_manager=leadfield_manager,
        electrode_registry=electrode_registry,
        use_amp=use_amp and device.type == "cuda",
        progress_log_pct=progress_log_pct,
    )
    logger.info("Stage1 评估结果: %s", results)

    output_path = Path(
        args.output or config.get("paths", {}).get("stage1_eval_output", "outputs/stage1_eval.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("评估结果已保存到: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
