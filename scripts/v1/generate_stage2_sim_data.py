#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 V1 第二层连接仿真数据

样本字段:
  x, pos, sensor_type, leadfield, s_true, a_true
"""

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Dict, List, Tuple

import torch


def sample_sparse_matrix(n: int, sparsity: float, scale: float) -> torch.Tensor:
    mask = (torch.rand(n, n) < sparsity).float()
    mat = torch.randn(n, n) * scale * mask
    mat.fill_diagonal_(0.0)
    return mat


def make_stable_var_matrices(n_sources: int, sparsity: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    a1 = sample_sparse_matrix(n_sources, sparsity=sparsity, scale=0.08)
    a2 = sample_sparse_matrix(n_sources, sparsity=sparsity, scale=0.05)
    a_base = a1 + a2
    eigvals = torch.linalg.eigvals(a_base)
    rho = torch.max(torch.abs(eigvals)).real
    if float(rho.item()) > 0.95:
        scale = 0.95 / (float(rho.item()) + 1e-8)
        a1 = a1 * scale
        a2 = a2 * scale
    return a1, a2


def rollout_var2(
    a1: torch.Tensor,
    a2: torch.Tensor,
    time_steps: int,
    process_noise: float = 0.05,
) -> torch.Tensor:
    n_sources = a1.shape[0]
    s = torch.zeros(n_sources, time_steps)
    s[:, 0] = torch.randn(n_sources)
    s[:, 1] = torch.randn(n_sources)
    for t in range(2, time_steps):
        s[:, t] = (
            torch.einsum("ij,j->i", a1, s[:, t - 1])
            + torch.einsum("ij,j->i", a2, s[:, t - 2])
            + process_noise * torch.randn(n_sources)
        )
    return s


def generate_sample(
    n_sensors: int,
    n_sources: int,
    time_steps: int,
    obs_noise: float,
) -> Dict[str, torch.Tensor]:
    a1, a2 = make_stable_var_matrices(n_sources=n_sources)
    a_true = a1 + a2
    s_true = rollout_var2(a1, a2, time_steps=time_steps)
    leadfield = torch.randn(n_sensors, n_sources) / (n_sources ** 0.5)
    x = torch.einsum("cs,st->ct", leadfield, s_true) + obs_noise * torch.randn(n_sensors, time_steps)
    pos = torch.randn(n_sensors, 6)
    sensor_type = torch.zeros(n_sensors, dtype=torch.long)
    return {
        "x": x.float(),
        "pos": pos.float(),
        "sensor_type": sensor_type,
        "leadfield": leadfield.float(),
        "s_true": s_true.float(),
        "a_true": a_true.float(),
    }


def dump_split(
    output_dir: Path,
    split: str,
    n_samples: int,
    n_sensors: int,
    n_sources: int,
    time_steps: int,
    obs_noise: float,
) -> str:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    metadata: List[Dict[str, str]] = []

    for i in range(n_samples):
        sample = generate_sample(
            n_sensors=n_sensors,
            n_sources=n_sources,
            time_steps=time_steps,
            obs_noise=obs_noise,
        )
        sample_path = split_dir / f"sample_{i:06d}.pt"
        torch.save(sample, sample_path)
        metadata.append({"path": str(sample_path)})

    meta_path = output_dir / f"stage2_{split}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return str(meta_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Stage2 simulation data")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--n_train", type=int, default=512, help="训练样本数")
    parser.add_argument("--n_val", type=int, default=128, help="验证样本数")
    parser.add_argument("--n_sensors", type=int, default=128, help="传感器数量")
    parser.add_argument("--n_sources", type=int, default=72, help="源数量")
    parser.add_argument("--time_steps", type=int, default=512, help="时间长度")
    parser.add_argument("--obs_noise", type=float, default=0.02, help="观测噪声标准差")
    parser.add_argument("--seed", type=int, default=123, help="随机种子")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_meta = dump_split(
        output_dir=output_dir,
        split="train",
        n_samples=args.n_train,
        n_sensors=args.n_sensors,
        n_sources=args.n_sources,
        time_steps=args.time_steps,
        obs_noise=args.obs_noise,
    )
    val_meta = dump_split(
        output_dir=output_dir,
        split="val",
        n_samples=args.n_val,
        n_sensors=args.n_sensors,
        n_sources=args.n_sources,
        time_steps=args.time_steps,
        obs_noise=args.obs_noise,
    )

    print(json.dumps({"stage2_train_metadata": train_meta, "stage2_val_metadata": val_meta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
