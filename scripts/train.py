#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PENCI 训练脚本

用法:
    python scripts/train.py --config configs/default.yaml
    
    # 覆盖配置参数
    python scripts/train.py --config configs/default.yaml training.batch_size=16
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from penci.models import PENCI, build_penci_from_config
from penci.data import get_train_val_loaders

# 设置日志
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


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    writer: SummaryWriter = None,
    global_step: int = 0,
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
        # 移动数据到设备
        x = batch["x"].to(device)
        pos = batch["pos"].to(device)
        sensor_type = batch["sensor_type"].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        losses = model.compute_loss(x, pos, sensor_type, loss_weights=loss_weights)
        
        loss = losses["loss"]
        recon_loss = losses["recon_loss"]
        dynamics_loss = losses["dynamics_loss"]
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        # 累积统计
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_dynamics_loss += dynamics_loss.item()
        num_batches += 1
        global_step += 1
        
        # 日志记录
        if batch_idx % log_interval == 0:
            avg_loss = total_loss / num_batches
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) "
                f"Recon: {recon_loss.item():.4f} "
                f"Dynamics: {dynamics_loss.item():.4f}"
            )
            
            # TensorBoard
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
        
        losses = model.compute_loss(x, pos, sensor_type, loss_weights=loss_weights)
        
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
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置输出目录
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
    
    # 设置设备
    device = torch.device(
        config.get("hardware", {}).get("device", "cuda") 
        if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"使用设备: {device}")
    
    # 创建模型
    logger.info("创建模型...")
    model = build_penci_from_config(config)
    model = model.to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    
    train_loader, val_loader = get_train_val_loaders(
        data_root=data_config.get("root_dir", "/work/2024/tanzunsheng/PENCIData"),
        dataset_name=data_config.get("datasets", ["HBN_EEG"])[0],
        batch_size=training_config.get("batch_size", 32),
        num_workers=training_config.get("num_workers", 4),
        max_length=data_config.get("time_window", 10) * data_config.get("sample_rate", 256),
        target_channels=data_config.get("n_channels", 128),
    )
    
    logger.info(f"训练集大小: {len(train_loader.dataset)}")
    logger.info(f"验证集大小: {len(val_loader.dataset)}")
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config.get("learning_rate", 1e-4),
        weight_decay=training_config.get("weight_decay", 0.01),
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.get("max_steps", 100000),
        eta_min=1e-6,
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 恢复训练
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint.get("step", 0)
        logger.info(f"已恢复检查点: {args.resume}")
    
    # 混合精度训练
    use_amp = config.get("hardware", {}).get("mixed_precision", True)
    scaler = torch.amp.GradScaler() if use_amp and device.type == "cuda" else None
    
    # 训练循环
    logger.info("开始训练...")
    max_epochs = training_config.get("max_steps", 100000) // len(train_loader) + 1
    save_interval = training_config.get("save_interval", 5000)
    eval_interval = training_config.get("eval_interval", 1000)
    
    for epoch in range(start_epoch, max_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{max_epochs} ===")
        
        # 训练
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            config, writer, global_step
        )
        
        scheduler.step()
        
        # 评估
        val_metrics = evaluate(model, val_loader, device, config)
        logger.info(
            f"验证 - Loss: {val_metrics['loss']:.4f}, "
            f"Recon: {val_metrics['recon_loss']:.4f}"
        )
        
        writer.add_scalar("val/loss", val_metrics["loss"], global_step)
        writer.add_scalar("val/recon_loss", val_metrics["recon_loss"], global_step)
        
        # 保存最佳模型
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model, optimizer, global_step, val_metrics["loss"],
                checkpoint_dir / "best_model.pt"
            )
        
        # 定期保存
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, global_step, val_metrics["loss"],
                checkpoint_dir / f"epoch_{epoch + 1}.pt"
            )
        
        # 检查是否达到最大步数
        if global_step >= training_config.get("max_steps", 100000):
            logger.info("达到最大训练步数，停止训练")
            break
    
    # 保存最终模型
    save_checkpoint(
        model, optimizer, global_step, val_metrics["loss"],
        checkpoint_dir / "final_model.pt"
    )
    
    writer.close()
    logger.info("训练完成！")


if __name__ == "__main__":
    main()
