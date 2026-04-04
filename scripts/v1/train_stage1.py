#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PENCI V1 第一层训练脚本。

支持两种模式:
  1. sim_pretrain: 仿真数据预训练
  2. real_finetune: 真实 EEG 微调（支持 DDP / HDF5 / file scheduler /
     node-union page-cache prefetch）
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from penci.data import create_dataloader, get_train_val_loaders
from penci.v1.data import Stage1SimulationDataset, create_simulation_dataloader
from penci.v1.models import build_stage1_model_from_config
from penci.training import (
    NodePageCachePrefetcher,
    build_node_union_prefetch_plan,
    cleanup_distributed,
    get_prefetch_file_plan,
    get_prefetch_rank_schedule,
    is_main_process,
    reduce_metric_sum,
    resolve_leadfield_for_batch,
    setup_distributed,
    setup_physics,
    sync_ranks,
    unwrap_model,
    warmup_hdf5_page_cache,
)
from penci.utils.metrics import compute_all_metrics
from penci.utils.state_metrics import state_mse, state_pearson, state_temporal_smoothness

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def configure_process_logging(rank: int) -> None:
    """多卡训练时仅保留主进程的 INFO 日志，其他 rank 降为 WARNING。"""
    if is_main_process(rank):
        logging.getLogger().setLevel(logging.INFO)
        return
    logging.getLogger().setLevel(logging.WARNING)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(config: Dict[str, Any], local_rank: int, world_size: int) -> torch.device:
    requested = config.get("hardware", {}).get("device", "cuda")
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP 模式需要可用的 CUDA 设备")
        return torch.device("cuda", local_rank)
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_sim_loader(
    dataset: Stage1SimulationDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    rank: int,
    world_size: int,
    use_bucket_sampler: bool = False,
    use_fingerprint: bool = False,
    file_scheduler: bool = False,
    shuffle_within_file: bool = True,
) -> DataLoader:
    return create_simulation_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        rank=rank,
        world_size=world_size,
        use_bucket_sampler=use_bucket_sampler,
        use_fingerprint=use_fingerprint,
        file_scheduler=file_scheduler,
        shuffle_within_file=shuffle_within_file,
        drop_last=shuffle,
    )


def build_stage1_loaders(
    config: Dict[str, Any],
    mode: str,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, Optional[DataLoader], str]:
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 8)
    num_workers = train_cfg.get("num_workers", 0)
    data_root = data_cfg.get("data_root", "/work/2024/tanzunsheng/PENCIData")
    use_bucket_sampler = bool(data_cfg.get("use_bucket_sampler", False))
    use_fingerprint = bool(data_cfg.get("use_fingerprint", use_bucket_sampler))
    file_scheduler_cfg = data_cfg.get("file_scheduler", {})

    if mode == "sim_pretrain":
        train_meta = data_cfg.get("stage1_sim_train_metadata")
        val_meta = data_cfg.get("stage1_sim_val_metadata")
        if not train_meta:
            raise RuntimeError("sim_pretrain 模式需要 data.stage1_sim_train_metadata")

        train_ds = Stage1SimulationDataset(metadata_path=train_meta)
        val_ds = Stage1SimulationDataset(metadata_path=val_meta) if val_meta else None
        train_loader = _build_sim_loader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            rank=rank,
            world_size=world_size,
            use_bucket_sampler=use_bucket_sampler,
            use_fingerprint=use_fingerprint,
            file_scheduler=bool(file_scheduler_cfg.get("enabled", False)),
            shuffle_within_file=bool(file_scheduler_cfg.get("shuffle_within_file", True)),
        )
        val_loader = (
            _build_sim_loader(
                val_ds,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                rank=rank,
                world_size=world_size,
                use_bucket_sampler=use_bucket_sampler,
                use_fingerprint=use_fingerprint,
                file_scheduler=bool(file_scheduler_cfg.get("val_enabled", False)),
                shuffle_within_file=bool(file_scheduler_cfg.get("val_shuffle_within_file", False)),
            )
            if val_ds is not None
            else None
        )
        return train_loader, val_loader, data_root

    max_length = data_cfg.get("window_length", 256) * max(1, data_cfg.get("time_window", 10))
    common_loader_kwargs = {
        "max_length": max_length,
        "sampler_file_scheduler": bool(file_scheduler_cfg.get("enabled", False)),
        "sampler_shuffle_within_file": bool(file_scheduler_cfg.get("shuffle_within_file", True)),
    }
    if not use_bucket_sampler:
        common_loader_kwargs["target_channels"] = data_cfg.get("n_channels", 128)
    val_loader_kwargs = dict(common_loader_kwargs)
    val_loader_kwargs["sampler_file_scheduler"] = bool(file_scheduler_cfg.get("val_enabled", False))
    val_loader_kwargs["sampler_shuffle_within_file"] = bool(
        file_scheduler_cfg.get("val_shuffle_within_file", False)
    )
    val_loader_kwargs["random_crop"] = False

    datasets = data_cfg.get("datasets")
    if datasets:
        return (
            *get_train_val_loaders(
                data_root=data_root,
                datasets=datasets,
                batch_size=batch_size,
                num_workers=num_workers,
                use_bucket_sampler=use_bucket_sampler,
                use_fingerprint=use_fingerprint,
                rank=rank,
                world_size=world_size,
                **common_loader_kwargs,
                val_sampler_file_scheduler=val_loader_kwargs["sampler_file_scheduler"],
                val_sampler_shuffle_within_file=val_loader_kwargs["sampler_shuffle_within_file"],
                val_random_crop=val_loader_kwargs["random_crop"],
            ),
            data_root,
        )

    train_meta = data_cfg.get("real_train_metadata")
    val_meta = data_cfg.get("real_val_metadata")
    if not train_meta:
        raise RuntimeError("real_finetune 模式需要 data.real_train_metadata 或 data.datasets")

    train_loader = create_dataloader(
        metadata_path=train_meta,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        use_bucket_sampler=use_bucket_sampler,
        use_fingerprint=use_fingerprint,
        rank=rank,
        world_size=world_size,
        **common_loader_kwargs,
    )
    val_loader = (
        create_dataloader(
            metadata_path=val_meta,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            use_bucket_sampler=use_bucket_sampler,
            use_fingerprint=use_fingerprint,
            rank=rank,
            world_size=world_size,
            **val_loader_kwargs,
        )
        if val_meta
        else None
    )
    return train_loader, val_loader, data_root


def maybe_set_epoch(dataloader: Optional[DataLoader], epoch: int) -> None:
    if dataloader is None:
        return
    if hasattr(dataloader, "batch_sampler") and hasattr(dataloader.batch_sampler, "set_epoch"):
        dataloader.batch_sampler.set_epoch(epoch)
    elif hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)


def load_default_leadfield(config: Dict[str, Any], device: torch.device) -> Optional[torch.Tensor]:
    physics_cfg = config.get("model", {}).get("physics", {})
    leadfield_path = physics_cfg.get("leadfield_path")
    if not leadfield_path:
        return None
    leadfield = torch.load(leadfield_path, map_location=device)
    if isinstance(leadfield, dict):
        leadfield = leadfield.get("leadfield", leadfield.get("L"))
    return leadfield


def resolve_leadfield(
    batch: Dict[str, Any],
    device: torch.device,
    default_leadfield: Optional[torch.Tensor],
    leadfield_manager: Any,
    electrode_registry: Any,
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


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any],
    device: torch.device,
    loss_weights: Dict[str, float],
    default_leadfield: Optional[torch.Tensor],
    leadfield_manager: Any,
    electrode_registry: Any,
    rank: int,
    world_size: int,
    gradient_clip: float,
    epoch: int,
    total_epochs: int,
    node_prefetcher: Optional[NodePageCachePrefetcher] = None,
) -> Dict[str, float]:
    model.train()
    use_amp = scaler is not None and scaler.is_enabled()
    progress_log_batches = 100
    started_at = time.time()
    total_batches: Optional[int] = None
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None

    loss_sum = torch.zeros(1, device=device)
    recon_sum = torch.zeros(1, device=device)
    state_sup_sum = torch.zeros(1, device=device)
    state_smooth_sum = torch.zeros(1, device=device)
    state_energy_sum = torch.zeros(1, device=device)
    count = torch.zeros(1, device=device)
    loader_iter = iter(loader)
    batch_idx = 0
    while True:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        batch_idx += 1

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
            device=device,
            default_leadfield=default_leadfield,
            leadfield_manager=leadfield_manager,
            electrode_registry=electrode_registry,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            output = model(
                x,
                pos,
                sensor_type,
                leadfield=leadfield,
                target_length=x.shape[-1],
                return_features=False,
            )
            losses = unwrap_model(model).compute_stage1_loss_from_output(
                output=output,
                x=x,
                s_true=s_true,
                n_channels=n_channels,
                loss_weights=loss_weights,
            )
            loss = losses["loss"]

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        loss_sum += loss.detach()
        recon_sum += losses["reconstruction_loss"].detach()
        state_sup_sum += losses["state_supervision_loss"].detach()
        state_smooth_sum += losses["state_smoothness_loss"].detach()
        state_energy_sum += losses["state_energy_loss"].detach()
        count += 1

        if node_prefetcher is not None and is_main_process(rank):
            node_prefetcher.update_progress(batch_idx)

        if (
            is_main_process(rank)
            and progress_log_batches > 0
            and batch_idx % progress_log_batches == 0
        ):
            elapsed = max(1e-6, time.time() - started_at)
            batches_per_sec = batch_idx / elapsed
            eta_min = float("nan")
            batch_total_display = "?"
            if total_batches is not None and total_batches > 0:
                batch_total_display = str(total_batches)
                remain_batches = max(0, total_batches - batch_idx)
                if batches_per_sec > 1e-6:
                    eta_min = remain_batches / batches_per_sec / 60.0

            denom_so_far = max(1.0, float(count.item()))
            logger.info(
                "[训练进度] epoch %d/%d | batch %d/%s | %.2f batch/s | ETA %.1f min | "
                "avg_loss=%.6f | avg_recon=%.6f | avg_smooth=%.6f",
                epoch + 1,
                total_epochs,
                batch_idx,
                batch_total_display,
                batches_per_sec,
                eta_min,
                float(loss_sum.item() / denom_so_far),
                float(recon_sum.item() / denom_so_far),
                float(state_smooth_sum.item() / denom_so_far),
            )

    reduce_metric_sum(loss_sum, world_size)
    reduce_metric_sum(recon_sum, world_size)
    reduce_metric_sum(state_sup_sum, world_size)
    reduce_metric_sum(state_smooth_sum, world_size)
    reduce_metric_sum(state_energy_sum, world_size)
    reduce_metric_sum(count, world_size)

    denom = max(1.0, float(count.item()))
    return {
        "loss": float(loss_sum.item() / denom),
        "reconstruction_loss": float(recon_sum.item() / denom),
        "state_supervision_loss": float(state_sup_sum.item() / denom),
        "state_smoothness_loss": float(state_smooth_sum.item() / denom),
        "state_energy_loss": float(state_energy_sum.item() / denom),
    }


@torch.no_grad()
def evaluate_one_epoch(
    model: torch.nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    default_leadfield: Optional[torch.Tensor],
    leadfield_manager: Any,
    electrode_registry: Any,
    world_size: int,
) -> Dict[str, float]:
    if loader is None:
        return {
            "recon_loss": float("nan"),
            "pearson": float("nan"),
            "snr_db": float("nan"),
            "nrmse": float("nan"),
            "state_mse": float("nan"),
            "state_pearson": float("nan"),
            "state_smoothness": float("nan"),
        }

    model.eval()
    recon_loss_sum = torch.zeros(1, device=device)
    pearson_sum = torch.zeros(1, device=device)
    snr_sum = torch.zeros(1, device=device)
    nrmse_sum = torch.zeros(1, device=device)
    smoothness_sum = torch.zeros(1, device=device)
    count = torch.zeros(1, device=device)
    state_mse_sum = torch.zeros(1, device=device)
    state_pearson_sum = torch.zeros(1, device=device)
    state_count = torch.zeros(1, device=device)

    for batch in loader:
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
            device=device,
            default_leadfield=default_leadfield,
            leadfield_manager=leadfield_manager,
            electrode_registry=electrode_registry,
        )

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
        target = unwrap_model(model)._prepare_target(x, recon.shape[-1])
        target, recon = unwrap_model(model).align_sensor_space(
            target,
            recon,
            n_channels=n_channels,
        )
        recon_metrics = compute_all_metrics(target, recon, n_channels=n_channels)

        recon_loss_sum += unwrap_model(model)._masked_sensor_mse(
            recon,
            target,
            n_channels=n_channels,
        )
        pearson_sum += recon_metrics["pearson"].detach()
        snr_sum += recon_metrics["snr_db"].detach()
        nrmse_sum += recon_metrics["nrmse"].detach()
        smoothness_sum += state_temporal_smoothness(state).detach()
        count += 1

        if s_true is not None:
            s_target = unwrap_model(model)._prepare_state_target(s_true, state.shape[-1])
            state_mse_sum += state_mse(s_target, state).detach()
            state_pearson_sum += state_pearson(s_target, state).detach()
            state_count += 1

    for tensor in [
        recon_loss_sum,
        pearson_sum,
        snr_sum,
        nrmse_sum,
        smoothness_sum,
        count,
        state_mse_sum,
        state_pearson_sum,
        state_count,
    ]:
        reduce_metric_sum(tensor, world_size)

    denom = max(1.0, float(count.item()))
    results = {
        "recon_loss": float(recon_loss_sum.item() / denom),
        "pearson": float(pearson_sum.item() / denom),
        "snr_db": float(snr_sum.item() / denom),
        "nrmse": float(nrmse_sum.item() / denom),
        "state_smoothness": float(smoothness_sum.item() / denom),
        "state_mse": float("nan"),
        "state_pearson": float("nan"),
    }
    if float(state_count.item()) > 0:
        state_denom = float(state_count.item())
        results["state_mse"] = float(state_mse_sum.item() / state_denom)
        results["state_pearson"] = float(state_pearson_sum.item() / state_denom)
    return results


def maybe_prepare_train_prefetch(
    train_loader: DataLoader,
    config: Dict[str, Any],
    data_root: str,
    mode: str,
    rank: int,
    local_rank: int,
    world_size: int,
    epoch: int,
    start_epoch: int,
) -> Optional[NodePageCachePrefetcher]:
    io_cfg = config.get("data", {}).get("io_prefetch", {})
    if not io_cfg.get("enabled", False):
        return None
    if not io_cfg.get("warmup_each_epoch", True) and epoch != start_epoch:
        return None

    warmup_gb = float(io_cfg.get("warmup_gb", 0.0))
    low_watermark_gb = float(io_cfg.get("low_watermark_gb", 0.75 * warmup_gb))
    read_chunk_mb = int(io_cfg.get("read_chunk_mb", 8))
    prefetch_threads = int(io_cfg.get("prefetch_threads", 1))
    max_files = int(io_cfg.get("max_files", 0))
    async_refill = bool(io_cfg.get("async_refill", True))
    scope = str(io_cfg.get("scope", "node_union")).strip() or "node_union"
    startup_policy = str(io_cfg.get("startup_policy", "high_watermark")).strip() or "high_watermark"
    train_node_union_prefetch = scope == "node_union" and startup_policy == "high_watermark"

    prefetch_plan = get_prefetch_file_plan(
        train_loader,
        data_root=data_root,
        max_files=max_files,
    )
    if not prefetch_plan:
        return None

    train_prefetcher: Optional[NodePageCachePrefetcher] = None
    if train_node_union_prefetch:
        local_rank_schedule = get_prefetch_rank_schedule(train_loader)
        if world_size > 1:
            gathered_rank_schedules: List[List[Dict[str, Any]]] = [[] for _ in range(world_size)]
            dist.all_gather_object(gathered_rank_schedules, local_rank_schedule)
        else:
            gathered_rank_schedules = [local_rank_schedule]

        if is_main_process(rank):
            node_union_plan = build_node_union_prefetch_plan(
                gathered_rank_schedules,
                data_root=data_root,
                max_files=max_files,
            )
            if node_union_plan:
                train_prefetcher = NodePageCachePrefetcher(
                    node_union_plan,
                    high_watermark_gb=warmup_gb,
                    low_watermark_gb=low_watermark_gb,
                    read_chunk_mb=read_chunk_mb,
                    max_threads=prefetch_threads,
                )
                train_prefetcher.warmup_to_high_watermark()
                if async_refill:
                    train_prefetcher.start_async_refill()
            else:
                logger.warning("节点级联合预热窗口为空，本 epoch 跳过训练前预热")
        sync_ranks(world_size, local_rank)
        return train_prefetcher

    if is_main_process(rank):
        warmup_hdf5_page_cache(
            prefetch_plan,
            warmup_gb=warmup_gb,
            read_chunk_mb=read_chunk_mb,
            max_threads=prefetch_threads,
        )
    sync_ranks(world_size, local_rank)
    return None


def maybe_prepare_val_prefetch(
    val_loader: Optional[DataLoader],
    config: Dict[str, Any],
    data_root: str,
    mode: str,
    rank: int,
    local_rank: int,
    world_size: int,
) -> None:
    if val_loader is None:
        return
    io_cfg = config.get("data", {}).get("io_prefetch", {})
    if not io_cfg.get("enabled", False) or not io_cfg.get("warmup_before_val", False):
        return

    prefetch_plan = get_prefetch_file_plan(
        val_loader,
        data_root=data_root,
        max_files=int(io_cfg.get("max_files_val", io_cfg.get("max_files", 0))),
    )
    if not prefetch_plan:
        return

    if is_main_process(rank):
        warmup_hdf5_page_cache(
            prefetch_plan,
            warmup_gb=float(io_cfg.get("warmup_val_gb", io_cfg.get("warmup_gb", 0.0))),
            read_chunk_mb=int(io_cfg.get("read_chunk_mb", 8)),
            max_threads=int(io_cfg.get("prefetch_threads", 1)),
        )
    sync_ranks(world_size, local_rank)


def load_checkpoint_state_dict(
    checkpoint_path: str,
    device: torch.device,
    world_size: int,
    local_rank: int,
) -> Dict[str, torch.Tensor]:
    """加载 checkpoint，并兼容完整训练 checkpoint 或纯 state_dict。"""
    map_location = device if world_size == 1 else {"cuda:0": f"cuda:{local_rank}"}
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Stage1 model")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument(
        "--mode",
        type=str,
        default="sim_pretrain",
        choices=["sim_pretrain", "real_finetune"],
        help="训练模式",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="可选：仅加载模型权重作为初始化，不恢复 optimizer/epoch",
    )
    parser.add_argument("--resume", type=str, default=None, help="可选：继续训练的 checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()

    if args.init_checkpoint and args.resume:
        raise RuntimeError("--init_checkpoint 与 --resume 不能同时使用")

    rank, local_rank, world_size = setup_distributed()
    configure_process_logging(rank)
    try:
        config = load_config(args.config)
        device = resolve_device(config, local_rank, world_size)
        train_cfg = config.get("training", {})
        ddp_cfg = train_cfg.get("ddp", {})
        find_unused_parameters = bool(ddp_cfg.get("find_unused_parameters", False))
        train_loader, val_loader, data_root = build_stage1_loaders(
            config=config,
            mode=args.mode,
            rank=rank,
            world_size=world_size,
        )

        base_model = build_stage1_model_from_config(config).to(device)
        if world_size > 1:
            model: torch.nn.Module = DDP(
                base_model,
                device_ids=[local_rank],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            model = base_model

        default_leadfield = load_default_leadfield(config, device)
        leadfield_manager = None
        electrode_registry = None
        if args.mode == "real_finetune" and default_leadfield is None:
            _, leadfield_manager, electrode_registry, _ = setup_physics(config)

        epochs = int(train_cfg.get("epochs", 10))
        lr = float(train_cfg.get("learning_rate", 1e-4))
        weight_decay = float(train_cfg.get("weight_decay", 0.0))
        gradient_clip = float(train_cfg.get("gradient_clip", 0.0))
        loss_weights = train_cfg.get(
            "loss",
            {
                "reconstruction": 1.0,
                "state_supervision": 1.0,
                "state_smoothness": 1e-2,
                "state_energy": 1e-3,
            },
        )
        use_amp = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        start_epoch = 0
        if args.init_checkpoint:
            state_dict = load_checkpoint_state_dict(
                args.init_checkpoint,
                device=device,
                world_size=world_size,
                local_rank=local_rank,
            )
            unwrap_model(model).load_state_dict(state_dict, strict=False)
            if is_main_process(rank):
                logger.info("使用预训练权重初始化: %s", args.init_checkpoint)
        if args.resume:
            map_location = device if world_size == 1 else {"cuda:0": f"cuda:{local_rank}"}
            checkpoint = torch.load(args.resume, map_location=map_location)
            unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=False)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = int(checkpoint.get("epoch", 0))
            if is_main_process(rank):
                logger.info("恢复训练: %s", args.resume)

        output_dir = Path(
            args.output_dir or config.get("paths", {}).get("output_dir", f"outputs/{args.mode}")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = output_dir / f"stage1_{args.mode}_best.pt"
        last_ckpt_path = output_dir / f"stage1_{args.mode}_last.pt"
        metrics_path = output_dir / f"stage1_{args.mode}_metrics.json"

        best_score = float("inf")
        history: List[Dict[str, Any]] = []
        if is_main_process(rank):
            logger.info(
                "开始 Stage1 训练 | mode=%s | device=%s | world_size=%d | epochs=%d",
                args.mode,
                device,
                world_size,
                epochs,
            )

        for epoch in range(start_epoch, epochs):
            if is_main_process(rank):
                logger.info("开始 Epoch %d/%d", epoch + 1, epochs)
            maybe_set_epoch(train_loader, epoch)
            maybe_set_epoch(val_loader, epoch)

            train_prefetcher = maybe_prepare_train_prefetch(
                train_loader=train_loader,
                config=config,
                data_root=data_root,
                mode=args.mode,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                epoch=epoch,
                start_epoch=start_epoch,
            )
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                loss_weights=loss_weights,
                default_leadfield=default_leadfield,
                leadfield_manager=leadfield_manager,
                electrode_registry=electrode_registry,
                rank=rank,
                world_size=world_size,
                gradient_clip=gradient_clip,
                epoch=epoch,
                total_epochs=epochs,
                node_prefetcher=train_prefetcher if is_main_process(rank) else None,
            )
            if train_prefetcher is not None:
                train_prefetcher.stop()

            maybe_prepare_val_prefetch(
                val_loader=val_loader,
                config=config,
                data_root=data_root,
                mode=args.mode,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
            )
            val_metrics = evaluate_one_epoch(
                model=model,
                loader=val_loader,
                device=device,
                default_leadfield=default_leadfield,
                leadfield_manager=leadfield_manager,
                electrode_registry=electrode_registry,
                world_size=world_size,
            )
            score = val_metrics["recon_loss"] if val_loader is not None else train_metrics["loss"]

            if is_main_process(rank):
                logger.info(
                    "Epoch %d/%d | train_loss=%.6f | val_recon=%.6f | val_pearson=%.4f",
                    epoch + 1,
                    epochs,
                    train_metrics["loss"],
                    val_metrics["recon_loss"],
                    val_metrics["pearson"],
                )
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "config": config,
                    "mode": args.mode,
                }
                torch.save(checkpoint, last_ckpt_path)
                if score < best_score:
                    best_score = score
                    torch.save(checkpoint, best_ckpt_path)

                history.append(
                    {
                        "epoch": epoch + 1,
                        "train": train_metrics,
                        "val": val_metrics,
                    }
                )
                with open(metrics_path, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "best_score": best_score,
                            "history": history,
                        },
                        handle,
                        ensure_ascii=False,
                        indent=2,
                    )

        if is_main_process(rank):
            logger.info("Stage1 训练完成 | best=%.6f | output=%s", best_score, output_dir)
        return 0
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
