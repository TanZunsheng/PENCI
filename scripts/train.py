#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PENCI 训练脚本

用法:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --output_dir outputs/experiment_1
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from penci.models import PENCI, build_penci_from_config
from penci.data import get_train_val_loaders
from penci.physics import SourceSpace, ElectrodeConfigRegistry
from penci.physics.leadfield_manager import LeadfieldManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    step: int,
    loss: float,
    save_path: str,
):
    """保存检查点"""
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    logger.info(f"检查点已保存: {save_path}")


def resolve_leadfield_for_batch(
    batch_metadata: List[Dict],
    n_channels: int,
    leadfield_manager: LeadfieldManager,
    electrode_registry: ElectrodeConfigRegistry,
    device: torch.device,
) -> torch.Tensor:
    """
    根据 batch 元数据解析对应的导联场矩阵

    BucketBatchSampler 保证同一 batch 内所有样本通道数相同，
    因此只需计算一次导联场并扩展到 batch 维度。

    参数:
        batch_metadata: batch 中每个样本的元数据列表
        n_channels: 当前 batch 的通道数
        leadfield_manager: 导联场管理器
        electrode_registry: 电极配置注册表
        device: 计算设备

    返回:
        (n_channels, 72) 导联场张量
    """
    dataset_name = batch_metadata[0]["dataset"]
    channel_names, channel_positions = electrode_registry.get_config(
        dataset_name, n_channels
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

    if not all([subjects_dir, cache_dir, processed_data_dir]):
        raise RuntimeError(
            "动态导联场模式需要 physics.subjects_dir、"
            "physics.leadfield_cache_dir 和 physics.processed_data_dir 配置"
        )

    logger.info("初始化物理约束组件...")

    source_space = SourceSpace(subjects_dir=subjects_dir)
    logger.info(f"  源空间: {source_space.get_source_info()['n_total']} 个源")

    leadfield_manager = LeadfieldManager(
        source_space=source_space,
        subjects_dir=subjects_dir,
        cache_dir=cache_dir,
    )

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
):
    """训练一个 epoch"""
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
            n_channels = batch["n_channels"][0].item()
            leadfield = resolve_leadfield_for_batch(
                batch["metadata"], n_channels,
                leadfield_manager, electrode_registry, device,
            )

        optimizer.zero_grad()
        losses = model.compute_loss(
            x, pos, sensor_type,
            leadfield=leadfield,
            loss_weights=loss_weights,
        )

        loss = losses["loss"]
        recon_loss = losses["recon_loss"]
        dynamics_loss = losses["dynamics_loss"]

        loss.backward()

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_dynamics_loss += dynamics_loss.item()
        num_batches += 1
        global_step += 1

        if batch_idx % log_interval == 0:
            avg_loss = total_loss / num_batches
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) "
                f"Recon: {recon_loss.item():.4f} "
                f"Dynamics: {dynamics_loss.item():.4f}"
            )

            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/recon_loss", recon_loss.item(), global_step)
                writer.add_scalar("train/dynamics_loss", dynamics_loss.item(), global_step)

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
):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    num_batches = 0

    loss_weights = config.get("training", {}).get("loss", {})

    for batch in dataloader:
        x = batch["x"].to(device)
        pos = batch["pos"].to(device)
        sensor_type = batch["sensor_type"].to(device)

        leadfield = None
        if leadfield_manager is not None and electrode_registry is not None:
            n_channels = batch["n_channels"][0].item()
            leadfield = resolve_leadfield_for_batch(
                batch["metadata"], n_channels,
                leadfield_manager, electrode_registry, device,
            )

        losses = model.compute_loss(
            x, pos, sensor_type,
            leadfield=leadfield,
            loss_weights=loss_weights,
        )

        total_loss += losses["loss"].item()
        total_recon_loss += losses["recon_loss"].item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0.0

    return {"loss": avg_loss, "recon_loss": avg_recon_loss}


def main():
    parser = argparse.ArgumentParser(description="PENCI 训练脚本")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.get("paths", {}).get("output_dir", "outputs")) / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    logger.info(f"输出目录: {output_dir}")

    device = torch.device(
        config.get("hardware", {}).get("device", "cuda")
        if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"使用设备: {device}")

    # === 物理约束组件初始化 ===
    source_space, leadfield_manager, electrode_registry = setup_physics(config)

    # === 创建模型 ===
    logger.info("创建模型...")
    model = build_penci_from_config(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    # === 创建数据加载器 ===
    logger.info("创建数据加载器...")
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    use_bucket_sampler = data_config.get("use_bucket_sampler", False)
    if leadfield_manager is not None and not use_bucket_sampler:
        logger.warning(
            "动态导联场模式下建议使用 BucketBatchSampler "
            "(data.use_bucket_sampler: true)，自动启用"
        )
        use_bucket_sampler = True

    configured_datasets = data_config.get("datasets", ["HBN_EEG"])

    train_loader, val_loader = get_train_val_loaders(
        data_root=data_config.get("root_dir", "/work/2024/tanzunsheng/PENCIData"),
        dataset_name=configured_datasets[0],
        batch_size=training_config.get("batch_size", 32),
        num_workers=training_config.get("num_workers", 4),
        use_bucket_sampler=use_bucket_sampler,
        max_length=data_config.get("time_window", 10) * data_config.get("sample_rate", 256),
        target_channels=data_config.get("n_channels", 128) if not use_bucket_sampler else None,
        datasets=configured_datasets,
    )

    logger.info(f"训练集大小: {len(train_loader.dataset)}")
    logger.info(f"验证集大小: {len(val_loader.dataset)}")
    logger.info(f"BucketBatchSampler: {'启用' if use_bucket_sampler else '禁用'}")

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

    # === 优化器 ===
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config.get("learning_rate", 1e-4),
        weight_decay=training_config.get("weight_decay", 0.01),
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.get("max_steps", 100000),
        eta_min=1e-6,
    )

    writer = SummaryWriter(log_dir)

    # === 恢复训练 ===
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint.get("step", 0)
        logger.info(f"已恢复检查点: {args.resume}")

    # === 混合精度训练 ===
    use_amp = config.get("hardware", {}).get("mixed_precision", True)
    scaler = torch.amp.GradScaler() if use_amp and device.type == "cuda" else None

    # === 训练循环 ===
    logger.info("开始训练...")
    max_epochs = training_config.get("max_steps", 100000) // len(train_loader) + 1
    save_interval = training_config.get("save_interval", 5000)
    eval_interval = training_config.get("eval_interval", 1000)

    for epoch in range(start_epoch, max_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{max_epochs} ===")

        if hasattr(train_loader, "batch_sampler") and hasattr(
            train_loader.batch_sampler, "set_epoch"
        ):
            train_loader.batch_sampler.set_epoch(epoch)

        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            config, writer, global_step,
            leadfield_manager=leadfield_manager,
            electrode_registry=electrode_registry,
        )

        scheduler.step()

        val_metrics = evaluate(
            model, val_loader, device, config,
            leadfield_manager=leadfield_manager,
            electrode_registry=electrode_registry,
        )
        logger.info(
            f"验证 - Loss: {val_metrics['loss']:.4f}, "
            f"Recon: {val_metrics['recon_loss']:.4f}"
        )

        writer.add_scalar("val/loss", val_metrics["loss"], global_step)
        writer.add_scalar("val/recon_loss", val_metrics["recon_loss"], global_step)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model, optimizer, global_step, val_metrics["loss"],
                checkpoint_dir / "best_model.pt"
            )

        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, global_step, val_metrics["loss"],
                checkpoint_dir / f"epoch_{epoch + 1}.pt"
            )

        if global_step >= training_config.get("max_steps", 100000):
            logger.info("达到最大训练步数，停止训练")
            break

    save_checkpoint(
        model, optimizer, global_step, val_metrics["loss"],
        checkpoint_dir / "final_model.pt"
    )

    writer.close()
    logger.info("训练完成！")


if __name__ == "__main__":
    main()
