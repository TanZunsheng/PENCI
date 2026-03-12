#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PENCI 训练脚本

用法:
    # 单卡训练
    python scripts/train.py --config configs/default.yaml
    # 多卡 DDP 训练 (4 GPU)
    torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --output_dir outputs/experiment_1
"""

import os
import sys
import math
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from penci.models import build_penci_from_config
from penci.data import get_train_val_loaders
from penci.physics import SourceSpace, ElectrodeConfigRegistry
from penci.physics.leadfield_manager import LeadfieldManager
from penci.utils.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_distributed() -> Tuple[int, int, int]:
    """
    检测 torchrun 环境，初始化分布式训练

    返回:
        (rank, local_rank, world_size) — 非分布式时返回 (0, 0, 1)
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    min_lr_ratio=0.0,
):
    """带线性预热的余弦退火调度器"""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def reduce_metric(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """在所有 rank 上 all_reduce 一个标量张量并取平均"""
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler,
    step: int,
    epoch: int,
    loss: float,
    save_path,
):
    """保存检查点"""
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(
        {
            'step': step,
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'loss': loss,
        },
        str(save_path),
    )
    logger.info(f"检查点已保存: {save_path}")


def resolve_leadfield_for_batch(
    fingerprint: str,
    leadfield_manager: LeadfieldManager,
    electrode_registry: ElectrodeConfigRegistry,
    device: torch.device,
) -> torch.Tensor:
    """
    根据电极指纹解析对应的导联场矩阵

    BucketBatchSampler 按 (通道数, 电极指纹) 分桶，保证同一 batch 内
    所有样本的电极配置完全一致，因此只需计算一次导联场。

    参数:
        fingerprint: 当前 batch 的电极位置指纹（pos_fingerprint）
        leadfield_manager: 导联场管理器
        electrode_registry: 电极配置注册表
        device: 计算设备

    返回:
        (n_channels, 72) 导联场张量
    """
    channel_names, channel_positions = electrode_registry.get_config_by_fingerprint(
        fingerprint
    )
    return leadfield_manager.get_leadfield(channel_names, channel_positions, device)


def setup_physics(config: dict) -> Tuple[
    Optional[SourceSpace],
    Optional[LeadfieldManager],
    Optional[ElectrodeConfigRegistry],
]:
    """
    初始化物理约束组件（SourceSpace + LeadfieldManager + ElectrodeConfigRegistry）

    从配置中读取路径，创建源空间、导联场管理器和电极配置注册表。
    自动为配置中指定的数据集注册电极配置。

    参数:
        config: 完整配置字典

    返回:
        (source_space, leadfield_manager, electrode_registry) 元组
        如果配置中未启用物理约束，所有返回值为 None
    """
    physics_cfg = config.get("model", {}).get("physics", {})
    use_fixed_leadfield = physics_cfg.get("use_fixed_leadfield", True)
    leadfield_path = physics_cfg.get("leadfield_path", None)

    if not use_fixed_leadfield or leadfield_path is not None:
        logger.info("物理约束模式: 静态导联场或注意力模式，跳过动态导联场初始化")
        return None, None, None

    global_physics = config.get("physics", {})
    subjects_dir = global_physics.get("subjects_dir")
    cache_dir = global_physics.get("leadfield_cache_dir")
    processed_data_dir = global_physics.get("processed_data_dir")
    registry_path = global_physics.get("fingerprint_registry_path")

    if not subjects_dir or not cache_dir:
        raise RuntimeError(
            "动态导联场模式需要 physics.subjects_dir 和 physics.leadfield_cache_dir 配置"
        )

    logger.info("初始化物理约束组件...")

    source_space = SourceSpace(subjects_dir=subjects_dir)
    logger.info(f"  源空间: {source_space.get_source_info()['n_total']} 个源")

    leadfield_manager = LeadfieldManager(
        source_space=source_space,
        subjects_dir=subjects_dir,
        cache_dir=cache_dir,
    )

    if registry_path is not None:
        logger.info(f"从离线存档加载电极配置注册表: {registry_path}")
        electrode_registry = ElectrodeConfigRegistry.load_from_archive(registry_path)
        logger.info(
            f"  已加载 {len(electrode_registry.get_all_fingerprints())} 个唯一电极指纹"
        )
    else:
        if not processed_data_dir:
            raise RuntimeError(
                "运行时扫描模式需要 physics.processed_data_dir 配置，"
                "或设置 physics.fingerprint_registry_path 使用离线存档"
            )
        logger.info("未配置 fingerprint_registry_path，使用运行时扫描模式")
        electrode_registry = ElectrodeConfigRegistry(processed_data_dir)

        datasets = config.get("data", {}).get("datasets", [])
        for ds_name in datasets:
            try:
                electrode_registry.register_dataset(ds_name)
            except FileNotFoundError as e:
                logger.warning(f"注册数据集 '{ds_name}' 电极配置失败: {e}")

        logger.info(
            f"  已注册电极配置: {electrode_registry.registered_configs}"
        )

    return source_space, leadfield_manager, electrode_registry


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    writer: SummaryWriter = None,
    global_step: int = 0,
    leadfield_manager: Optional[LeadfieldManager] = None,
    electrode_registry: Optional[ElectrodeConfigRegistry] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    scheduler=None,
    rank: int = 0,
    world_size: int = 1,
):
    """训练一个 epoch"""
    del world_size  # 保留参数以对齐 DDP 接口
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_dynamics_loss = 0.0
    num_batches = 0

    loss_weights = config.get("training", {}).get("loss", {})
    log_interval = config.get("training", {}).get("log_interval", 100)
    gradient_clip = config.get("training", {}).get("gradient_clip", 1.0)

    for batch_idx, batch in enumerate(dataloader):
        x = batch["x"].to(device)
        pos = batch["pos"].to(device)
        sensor_type = batch["sensor_type"].to(device)

        leadfield = None
        if leadfield_manager is not None and electrode_registry is not None:
            fingerprint = batch.get("fingerprint", "unknown")
            leadfield = resolve_leadfield_for_batch(
                fingerprint,
                leadfield_manager, electrode_registry, device,
            )

        optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None
        with torch.autocast(device_type="cuda", enabled=use_amp):
            losses = model.compute_loss(
                x, pos, sensor_type,
                leadfield=leadfield,
                loss_weights=loss_weights,
            )
            loss = losses["loss"]

        recon_loss = losses["recon_loss"]
        dynamics_loss = losses["dynamics_loss"]

        if scaler is not None:
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

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_dynamics_loss += dynamics_loss.item()
        num_batches += 1
        global_step += 1

        if is_main_process(rank) and batch_idx % log_interval == 0:
            avg_loss = total_loss / num_batches
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) "
                f"Recon: {recon_loss.item():.4f} "
                f"Dynamics: {dynamics_loss.item():.4f}"
            )

            if writer is not None and is_main_process(rank):
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/recon_loss", recon_loss.item(), global_step)
                writer.add_scalar("train/dynamics_loss", dynamics_loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
    config: dict,
    leadfield_manager: Optional[LeadfieldManager] = None,
    electrode_registry: Optional[ElectrodeConfigRegistry] = None,
    rank: int = 0,
    world_size: int = 1,
):
    """评估模型"""
    del rank
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_pearson = 0.0
    total_snr_db = 0.0
    total_nrmse = 0.0
    num_batches = 0

    loss_weights = config.get("training", {}).get("loss", {})

    for batch in dataloader:
        x = batch["x"].to(device)
        pos = batch["pos"].to(device)
        sensor_type = batch["sensor_type"].to(device)

        leadfield = None
        if leadfield_manager is not None and electrode_registry is not None:
            fingerprint = batch.get("fingerprint", "unknown")
            leadfield = resolve_leadfield_for_batch(
                fingerprint,
                leadfield_manager, electrode_registry, device,
            )

        losses = model.compute_loss(
            x, pos, sensor_type,
            leadfield=leadfield,
            loss_weights=loss_weights,
        )

        output = model(x, pos, sensor_type, leadfield=leadfield, return_source=True)
        reconstruction = output["reconstruction"]
        target = (
            model.module._prepare_target(x, reconstruction.shape[-1])
            if hasattr(model, 'module')
            else model._prepare_target(x, reconstruction.shape[-1])
        )
        metrics_batch = compute_all_metrics(target, reconstruction)

        total_loss += losses["loss"].item()
        total_recon_loss += losses["recon_loss"].item()
        total_pearson += metrics_batch["pearson"].item()
        total_snr_db += metrics_batch["snr_db"].item()
        total_nrmse += metrics_batch["nrmse"].item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0.0
    avg_pearson = total_pearson / num_batches if num_batches > 0 else 0.0
    avg_snr_db = total_snr_db / num_batches if num_batches > 0 else 0.0
    avg_nrmse = total_nrmse / num_batches if num_batches > 0 else 0.0

    avg_loss_t = reduce_metric(torch.tensor(avg_loss, device=device), world_size)
    avg_recon_loss_t = reduce_metric(torch.tensor(avg_recon_loss, device=device), world_size)
    avg_pearson_t = reduce_metric(torch.tensor(avg_pearson, device=device), world_size)
    avg_snr_db_t = reduce_metric(torch.tensor(avg_snr_db, device=device), world_size)
    avg_nrmse_t = reduce_metric(torch.tensor(avg_nrmse, device=device), world_size)

    return {
        "loss": avg_loss_t.item(),
        "recon_loss": avg_recon_loss_t.item(),
        "pearson": avg_pearson_t.item(),
        "snr_db": avg_snr_db_t.item(),
        "nrmse": avg_nrmse_t.item(),
    }


def main():
    rank, local_rank, world_size = setup_distributed()
    if not is_main_process(rank):
        logging.getLogger().setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="PENCI 训练脚本")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    writer = None
    try:
        config = load_config(args.config)

        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(config.get("paths", {}).get("output_dir", "outputs")) / timestamp

        checkpoint_dir = output_dir / "checkpoints"
        log_dir = output_dir / "logs"

        if is_main_process(rank):
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(exist_ok=True)
            log_dir.mkdir(exist_ok=True)
        if world_size > 1:
            dist.barrier()

        if is_main_process(rank):
            logger.info(f"输出目录: {output_dir}")

        if world_size > 1:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if is_main_process(rank):
            logger.info(f"使用设备: {device}")

        # === 物理约束组件初始化 ===
        source_space, leadfield_manager, electrode_registry = setup_physics(config)

        # === 创建模型 ===
        if is_main_process(rank):
            logger.info("创建模型...")
        model = build_penci_from_config(config)
        model = model.to(device)
        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=config.get("distributed", {}).get("find_unused_parameters", False),
            )

        if is_main_process(rank):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")

        # === 创建数据加载器 ===
        if is_main_process(rank):
            logger.info("创建数据加载器...")
        data_config = config.get("data", {})
        training_config = config.get("training", {})

        use_bucket_sampler = data_config.get("use_bucket_sampler", False)
        use_fingerprint = False
        if leadfield_manager is not None and not use_bucket_sampler:
            if is_main_process(rank):
                logger.warning(
                    "动态导联场模式下建议使用 BucketBatchSampler "
                    "(data.use_bucket_sampler: true)，自动启用"
                )
            use_bucket_sampler = True
        if leadfield_manager is not None:
            use_fingerprint = True
            if is_main_process(rank):
                logger.info("动态导联场模式: 自动启用电极指纹分桶")

        configured_datasets = data_config.get("datasets", ["HBN_EEG"])

        train_loader, val_loader = get_train_val_loaders(
            data_root=data_config.get("root_dir", "/work/2024/tanzunsheng/PENCIData"),
            datasets=configured_datasets,
            batch_size=training_config.get("batch_size", 32),
            num_workers=training_config.get("num_workers", 4),
            use_bucket_sampler=use_bucket_sampler,
            use_fingerprint=use_fingerprint,
            max_length=data_config.get("time_window", 10) * data_config.get("sample_rate", 256),
            target_channels=data_config.get("n_channels", 128) if not use_bucket_sampler else None,
            rank=rank,
            world_size=world_size,
        )

        if is_main_process(rank):
            logger.info(f"训练集大小: {len(train_loader.dataset)}")
            logger.info(f"验证集大小: {len(val_loader.dataset)}")
            logger.info(f"BucketBatchSampler: {'启用' if use_bucket_sampler else '禁用'}")
            if use_fingerprint:
                logger.info("电极指纹分桶: 启用")

        # 自动注册数据加载器中实际出现的所有数据集的电极配置
        if electrode_registry is not None:
            loaded_datasets = set()
            for m in train_loader.dataset.metadata:
                loaded_datasets.add(m.get("dataset", "unknown"))
            for ds_name in loaded_datasets:
                if not electrode_registry.has_config_for_dataset(ds_name):
                    try:
                        electrode_registry.register_dataset(ds_name)
                        logger.info(f"  补充注册数据集电极配置: {ds_name}")
                    except FileNotFoundError as e:
                        logger.warning(f"  补充注册数据集 '{ds_name}' 失败: {e}")
            logger.info(
                f"  最终电极配置: {electrode_registry.registered_configs}"
            )
            logger.info(
                f"  已注册指纹: {electrode_registry.get_all_fingerprints()}"
            )

        # === 导联场预热 ===
        # 预计算所有已注册指纹对应的导联场，避免训练循环中首次命中时卡顿
        if leadfield_manager is not None and electrode_registry is not None:
            all_fps = electrode_registry.get_all_fingerprints()
            if all_fps:
                logger.info(f"预热导联场: {len(all_fps)} 个唯一电极配置...")
                for fp in all_fps:
                    try:
                        names, positions = electrode_registry.get_config_by_fingerprint(fp)
                        L = leadfield_manager.get_leadfield(names, positions, device)
                        logger.info(f"  指纹 {fp}: 导联场 {L.shape}")
                    except Exception as e:
                        logger.warning(f"  指纹 {fp} 导联场预热失败: {e}")
                logger.info("导联场预热完成")

        # === 优化器与调度器 ===
        base_lr = training_config.get("learning_rate", 1e-4)
        effective_lr = base_lr * world_size  # linear scaling rule
        if is_main_process(rank):
            logger.info(
                f"学习率缩放: base_lr={base_lr} × world_size={world_size} = {effective_lr}"
            )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=effective_lr,
            weight_decay=training_config.get("weight_decay", 0.01),
        )

        warmup_steps = training_config.get("warmup_steps", 1000)
        max_steps = training_config.get("max_steps", 100000)
        min_lr_ratio = 1e-6 / effective_lr  # eta_min=1e-6 equivalent
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps,
            max_steps,
            min_lr_ratio,
        )

        writer = SummaryWriter(log_dir) if is_main_process(rank) else None

        # === 混合精度训练 ===
        use_amp = config.get("hardware", {}).get("mixed_precision", True)
        scaler = torch.amp.GradScaler() if use_amp and device.type == "cuda" else None

        # === 恢复训练 ===
        start_epoch = 0
        global_step = 0
        best_val_loss = float("inf")
        val_metrics = {"loss": float("inf")}

        if args.resume:
            map_location = f"cuda:{local_rank}" if world_size > 1 else device
            checkpoint = torch.load(args.resume, map_location=map_location)
            target_model = model.module if hasattr(model, 'module') else model
            target_model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if scaler and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            global_step = checkpoint.get("step", 0)
            start_epoch = checkpoint.get("epoch", 0)
            if is_main_process(rank):
                logger.info(f"已恢复检查点: {args.resume}")

        # === 训练循环 ===
        if is_main_process(rank):
            logger.info("开始训练...")
        max_epochs = max_steps // max(1, len(train_loader)) + 1

        final_epoch = start_epoch
        for epoch in range(start_epoch, max_epochs):
            final_epoch = epoch
            if is_main_process(rank):
                logger.info(f"=== Epoch {epoch + 1}/{max_epochs} ===")

            if hasattr(train_loader, "batch_sampler") and hasattr(
                train_loader.batch_sampler, "set_epoch"
            ):
                train_loader.batch_sampler.set_epoch(epoch)

            train_loss, global_step = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch,
                config,
                writer,
                global_step,
                leadfield_manager=leadfield_manager,
                electrode_registry=electrode_registry,
                scaler=scaler,
                scheduler=scheduler,
                rank=rank,
                world_size=world_size,
            )

            val_metrics = evaluate(
                model,
                val_loader,
                device,
                config,
                leadfield_manager=leadfield_manager,
                electrode_registry=electrode_registry,
                rank=rank,
                world_size=world_size,
            )

            if is_main_process(rank):
                logger.info(
                    f"验证 - Loss: {val_metrics['loss']:.4f}, "
                    f"Recon: {val_metrics['recon_loss']:.4f}, "
                    f"Pearson: {val_metrics['pearson']:.4f}, "
                    f"SNR(dB): {val_metrics['snr_db']:.4f}, "
                    f"NRMSE: {val_metrics['nrmse']:.4f}"
                )
                if writer is not None:
                    writer.add_scalar("val/loss", val_metrics["loss"], global_step)
                    writer.add_scalar("val/recon_loss", val_metrics["recon_loss"], global_step)
                    writer.add_scalar("val/pearson", val_metrics["pearson"], global_step)
                    writer.add_scalar("val/snr_db", val_metrics["snr_db"], global_step)
                    writer.add_scalar("val/nrmse", val_metrics["nrmse"], global_step)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                if is_main_process(rank):
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        global_step,
                        epoch,
                        val_metrics["loss"],
                        checkpoint_dir / "best_model.pt",
                    )
                if world_size > 1:
                    dist.barrier()

            if (epoch + 1) % 5 == 0:
                if is_main_process(rank):
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        global_step,
                        epoch,
                        val_metrics["loss"],
                        checkpoint_dir / f"epoch_{epoch + 1}.pt",
                    )
                if world_size > 1:
                    dist.barrier()

            if global_step >= max_steps:
                if is_main_process(rank):
                    logger.info("达到最大训练步数，停止训练")
                break

            if is_main_process(rank):
                logger.info(f"Epoch {epoch + 1} 训练平均损失: {train_loss:.4f}")

        if is_main_process(rank):
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                global_step,
                final_epoch,
                val_metrics["loss"],
                checkpoint_dir / "final_model.pt",
            )
        if world_size > 1:
            dist.barrier()

        if is_main_process(rank):
            logger.info("训练完成！")
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
