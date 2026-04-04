#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 V1 第一层正式仿真预训练数据。

输出字段保持与 Stage1SimulationDataset 兼容：
  x, pos, sensor_type, leadfield, s_true

在 metadata 中额外记录：
  sim_type, noise_type, layout_type, snr_db, layout_fingerprint

数据丰富度来源：
  1. 多种源动力学族（平滑网络 / 振荡网络 / 突发瞬态 / 多尺度混合）
  2. 多种观测噪声与伪迹（白噪声 / 漂移 / 肌电 / dropout / 混合）
  3. 混合真实缓存导联场与合成几何导联场
"""

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from penci.physics.leadfield_manager import _compute_channel_hash, compute_fingerprint_from_pos

PRESETS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "n_train": 512,
        "n_val": 128,
        "real_layout_prob": 0.25,
        "snr_db_min": 6.0,
        "snr_db_max": 18.0,
    },
    "formal": {
        "n_train": 20000,
        "n_val": 4000,
        "real_layout_prob": 0.70,
        "snr_db_min": 2.0,
        "snr_db_max": 18.0,
    },
    "large": {
        "n_train": 50000,
        "n_val": 10000,
        "real_layout_prob": 0.75,
        "snr_db_min": 1.0,
        "snr_db_max": 20.0,
    },
}

DEFAULT_SIM_TYPES: Tuple[str, ...] = (
    "smooth_network",
    "band_oscillation",
    "burst_transient",
    "multiscale_hybrid",
)
DEFAULT_NOISE_TYPES: Tuple[str, ...] = (
    "white",
    "colored",
    "muscle",
    "dropout",
    "mixed",
)
DEFAULT_SYNTHETIC_LAYOUTS: Tuple[str, ...] = (
    "synthetic_uniform",
    "synthetic_frontal_dense",
    "synthetic_temporal_bias",
)


@dataclass
class LayoutTemplate:
    """单个可复用的布局模板。"""

    pos: torch.Tensor
    leadfield: torch.Tensor
    layout_type: str
    layout_source: str
    layout_dataset: str = ""
    layout_fingerprint: str = ""
    layout_full_fingerprint: str = ""
    layout_leadfield_cache_path: str = ""


@dataclass
class GenerationContext:
    """离线仿真生成上下文。"""

    n_sensors: int
    n_sources: int
    time_steps: int
    sampling_rate: float
    layout_mode: str
    real_layout_prob: float
    snr_db_min: float
    snr_db_max: float
    sim_types: Sequence[str]
    noise_types: Sequence[str]
    synthetic_layouts: Sequence[str]
    source_positions: torch.Tensor
    real_layouts: Sequence[LayoutTemplate]
    registry_layout_count: int
    external_layout_count: int
    generator: torch.Generator


def sample_uniform(
    shape: Tuple[int, ...],
    low: float,
    high: float,
    generator: torch.Generator,
) -> torch.Tensor:
    return low + (high - low) * torch.rand(shape, generator=generator)


def sample_choice(options: Sequence[str], generator: torch.Generator) -> str:
    index = int(torch.randint(len(options), (1,), generator=generator).item())
    return options[index]


def sample_bool(probability: float, generator: torch.Generator) -> bool:
    return bool(torch.rand(1, generator=generator).item() < probability)


def parse_path_list_arg(raw: str) -> Tuple[str, ...]:
    if not raw.strip():
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def l2_normalize(x: torch.Tensor, dim: int) -> torch.Tensor:
    denom = torch.linalg.norm(x, dim=dim, keepdim=True).clamp_min(1e-6)
    return x / denom


def standardize_per_channel(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return x / std


def standardize_global(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean()
    return x / x.std().clamp_min(1e-6)


def build_source_positions(n_sources: int) -> torch.Tensor:
    """在脑壳内构造一组固定 72 源位置。"""
    index = torch.arange(n_sources, dtype=torch.float32)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * (index + 0.5) / max(1, n_sources)
    radius = torch.sqrt((1.0 - y * y).clamp_min(1e-6))
    theta = index * golden_angle
    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius
    base = torch.stack([x, y, z], dim=-1)
    radial = 0.045 + 0.012 * torch.sin(0.5 * theta).abs()
    return base * radial.unsqueeze(-1)


def sample_sensor_xyz(
    n_sensors: int,
    layout_family: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """按不同布局偏好采样传感器位置。"""
    points: List[torch.Tensor] = []
    chunk = max(256, n_sensors * 4)
    while sum(point.shape[0] for point in points) < n_sensors:
        candidates = torch.randn(chunk, 3, generator=generator)
        candidates = l2_normalize(candidates, dim=-1)
        x = candidates[:, 0]
        y = candidates[:, 1]
        z = candidates[:, 2]
        if layout_family == "synthetic_frontal_dense":
            weight = 0.20 + 0.80 * torch.sigmoid(4.0 * y + 1.5 * z)
        elif layout_family == "synthetic_temporal_bias":
            weight = 0.20 + 0.80 * torch.sigmoid(4.5 * x.abs() - 1.0)
        else:
            weight = torch.ones(chunk)
        keep = torch.rand(chunk, generator=generator) < weight.clamp(0.05, 1.0)
        accepted = candidates[keep]
        if accepted.numel() > 0:
            points.append(accepted)
    xyz = torch.cat(points, dim=0)[:n_sensors]
    xyz = xyz + 0.035 * torch.randn(n_sensors, 3, generator=generator)
    xyz = l2_normalize(xyz, dim=-1)
    radius = sample_uniform((n_sensors, 1), 0.088, 0.102, generator)
    return xyz * radius


def normalize_eeg_pos_like_brainomni(pos: torch.Tensor) -> torch.Tensor:
    """
    复刻 BrainOmniPostProcess 对 EEG 坐标的归一化规则。

    真实 EEG `.pt` 中：
    - `pos[:, :3]` 会做去中心 + 全局尺度归一化
    - `pos[:, 3:]` 对 EEG 为零方向向量
    """
    pos = pos.clone().float()
    xyz = pos[:, :3]
    xyz = xyz - xyz.mean(dim=0, keepdim=True)
    scale = torch.sqrt(3.0 * torch.mean(torch.sum(xyz**2, dim=1))).clamp_min(1e-10)
    pos[:, :3] = xyz / scale
    return pos


def build_pos_from_xyz(xyz: torch.Tensor) -> torch.Tensor:
    # 与真实 EEG `.pt` 对齐：方向向量为零，位置使用 BrainOmni 风格归一化。
    pos = torch.cat([xyz, torch.zeros_like(xyz)], dim=-1)
    return normalize_eeg_pos_like_brainomni(pos)


def fingerprint_to_seed(base_seed: int, fingerprint: str) -> int:
    if not fingerprint:
        return int(base_seed)
    offset = int(fingerprint[:16], 16)
    modulus = (1 << 63) - 1
    return int((base_seed * 1000003 + offset) % modulus)


def normalize_leadfield(leadfield: torch.Tensor) -> torch.Tensor:
    """
    统一导联场 dtype，不再做逐源列标准化。

    Stage1 连接建模版本需要保留 sim / real 一致的前向尺度约定，这里仅做
    float32 规范化，避免再次改写 source-wise 量纲。
    """
    return leadfield.float()


def persist_layout_leadfield(
    cache_dir: Optional[str],
    subdir: str,
    full_fingerprint: str,
    leadfield: torch.Tensor,
) -> str:
    if not cache_dir or not full_fingerprint:
        return ""

    cache_path = Path(cache_dir) / subdir / f"{full_fingerprint}.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.is_file():
        torch.save({"leadfield": leadfield.cpu()}, cache_path)
    return str(cache_path)


def build_geometry_leadfield(
    sensor_xyz: torch.Tensor,
    source_positions: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """构造几何一致的合成导联场。"""
    sensor_normals = l2_normalize(sensor_xyz, dim=-1)
    diff = sensor_xyz[:, None, :] - source_positions[None, :, :]
    dist = torch.linalg.norm(diff, dim=-1).clamp_min(0.015)
    radial = (diff * sensor_normals[:, None, :]).sum(dim=-1) / dist
    source_normals = l2_normalize(source_positions, dim=-1)
    tangential = (sensor_xyz[:, None, :] * source_normals[None, :, :]).sum(dim=-1)
    hemispheric = sensor_xyz[:, None, 0] * source_positions[None, :, 0]
    leadfield = (0.85 * radial + 0.10 * tangential + 0.05 * hemispheric) / dist.pow(1.75)
    column_gain = sample_uniform((1, source_positions.shape[0]), 0.70, 1.35, generator)
    low_rank_u = torch.randn(sensor_xyz.shape[0], 4, generator=generator)
    low_rank_v = torch.randn(4, source_positions.shape[0], generator=generator)
    leadfield = leadfield * column_gain + 0.04 * (low_rank_u @ low_rank_v) / math.sqrt(4.0)
    return normalize_leadfield(leadfield)


def load_real_layout_pool(
    registry_path: Optional[str],
    n_sensors: int,
    n_sources: int,
    max_real_layouts: int,
    sim_leadfield_cache_dir: Optional[str] = None,
    seen_full_fingerprints: Optional[set] = None,
) -> List[LayoutTemplate]:
    if not registry_path:
        return []
    path = Path(registry_path)
    if not path.is_file():
        return []

    archive = torch.load(path, map_location="cpu", weights_only=False)
    configs = archive.get("configs", {})
    layouts: List[LayoutTemplate] = []
    seen_full_fingerprints = seen_full_fingerprints if seen_full_fingerprints is not None else set()

    for _, entry in configs.items():
        channel_positions = entry.get("channel_positions_m")
        leadfield_path = entry.get("leadfield_cache_path")
        if channel_positions is None or not leadfield_path:
            continue
        full_fingerprint = str(entry.get("full_fingerprint", ""))
        if not full_fingerprint or full_fingerprint in seen_full_fingerprints:
            continue

        xyz = torch.as_tensor(channel_positions, dtype=torch.float32)
        if xyz.ndim != 2 or xyz.shape[0] != n_sensors or xyz.shape[1] != 3:
            continue

        leadfield_file = Path(leadfield_path)
        if not leadfield_file.is_file():
            continue
        leadfield = torch.load(leadfield_file, map_location="cpu", weights_only=True)
        if isinstance(leadfield, dict):
            leadfield = leadfield.get("leadfield", leadfield.get("L"))
        if leadfield is None:
            continue
        leadfield = torch.as_tensor(leadfield, dtype=torch.float32)
        if leadfield.shape != (n_sensors, n_sources):
            continue
        leadfield = normalize_leadfield(leadfield)
        layout_fingerprint = str(compute_fingerprint_from_pos(np.asarray(channel_positions)))
        cached_path = persist_layout_leadfield(
            sim_leadfield_cache_dir,
            "registry",
            full_fingerprint,
            leadfield,
        )

        layouts.append(
            LayoutTemplate(
                pos=build_pos_from_xyz(xyz),
                leadfield=leadfield,
                layout_type="real_cached",
                layout_source="archive",
                layout_dataset="archive",
                layout_fingerprint=layout_fingerprint,
                layout_full_fingerprint=full_fingerprint,
                layout_leadfield_cache_path=cached_path,
            )
        )
        seen_full_fingerprints.add(full_fingerprint)
        if max_real_layouts > 0 and len(layouts) >= max_real_layouts:
            break

    return layouts


def load_external_template_pool(
    template_roots: Sequence[str],
    n_sensors: int,
    source_positions: torch.Tensor,
    max_external_layouts: int,
    base_seed: int,
    sim_leadfield_cache_dir: Optional[str] = None,
    seen_full_fingerprints: Optional[set] = None,
) -> List[LayoutTemplate]:
    seen_full_fingerprints = seen_full_fingerprints if seen_full_fingerprints is not None else set()

    meta_paths: List[Path] = []
    for root_str in template_roots:
        root = Path(root_str)
        if not root.exists():
            continue
        meta_paths.extend(sorted(root.rglob("template_meta.json")))

    layouts: List[LayoutTemplate] = []
    for meta_path in meta_paths:
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)

        if int(meta.get("n_channels", -1)) != n_sensors:
            continue

        channel_names = meta.get("channel_names")
        channel_positions = meta.get("channel_positions_m")
        if not channel_names or channel_positions is None:
            continue

        positions_np = np.asarray(channel_positions, dtype=np.float64)
        if positions_np.ndim != 2 or positions_np.shape != (n_sensors, 3):
            continue

        pos_fingerprint = str(compute_fingerprint_from_pos(positions_np))
        full_fingerprint = str(_compute_channel_hash(channel_names, positions_np))
        if full_fingerprint in seen_full_fingerprints:
            continue

        xyz = torch.as_tensor(positions_np, dtype=torch.float32)
        cached_path = ""
        cache_file = None
        if sim_leadfield_cache_dir and full_fingerprint:
            cache_file = (
                Path(sim_leadfield_cache_dir) / "external_template" / f"{full_fingerprint}.pt"
            )

        if cache_file is not None and cache_file.is_file():
            leadfield = torch.load(cache_file, map_location="cpu", weights_only=True)
            if isinstance(leadfield, dict):
                leadfield = leadfield.get("leadfield", leadfield.get("L"))
            if leadfield is None:
                continue
            leadfield = torch.as_tensor(leadfield, dtype=torch.float32)
        else:
            leadfield_generator = torch.Generator().manual_seed(
                fingerprint_to_seed(base_seed, pos_fingerprint)
            )
            leadfield = build_geometry_leadfield(xyz, source_positions, leadfield_generator)

        if leadfield.shape != (n_sensors, source_positions.shape[0]):
            continue
        leadfield = normalize_leadfield(leadfield)
        cached_path = persist_layout_leadfield(
            sim_leadfield_cache_dir,
            "external_template",
            full_fingerprint,
            leadfield,
        )
        layouts.append(
            LayoutTemplate(
                pos=build_pos_from_xyz(xyz),
                leadfield=leadfield,
                layout_type="external_template",
                layout_source="external_template_geometry",
                layout_dataset=str(meta.get("dataset_name", "external_template")),
                layout_fingerprint=pos_fingerprint,
                layout_full_fingerprint=full_fingerprint,
                layout_leadfield_cache_path=cached_path,
            )
        )
        seen_full_fingerprints.add(full_fingerprint)
        if max_external_layouts > 0 and len(layouts) >= max_external_layouts:
            break

    return layouts


def generate_latent_ar(
    n_latent: int,
    time_steps: int,
    generator: torch.Generator,
    phi_range: Tuple[float, float],
    noise_range: Tuple[float, float],
) -> torch.Tensor:
    latent = torch.zeros(n_latent, time_steps)
    latent[:, 0] = torch.randn(n_latent, generator=generator)
    phi = sample_uniform((n_latent, 1), phi_range[0], phi_range[1], generator)
    innovation = sample_uniform((n_latent, 1), noise_range[0], noise_range[1], generator)
    for time_index in range(1, time_steps):
        latent[:, time_index] = phi.squeeze(-1) * latent[:, time_index - 1] + innovation.squeeze(
            -1
        ) * torch.randn(n_latent, generator=generator)
    return standardize_per_channel(latent)


def build_sparse_mixing(
    n_sources: int,
    n_latent: int,
    generator: torch.Generator,
) -> torch.Tensor:
    mixing = torch.randn(n_sources, n_latent, generator=generator)
    mask = torch.rand(n_sources, n_latent, generator=generator) < 0.35
    mixing = mixing * mask
    empty_rows = mask.sum(dim=-1) == 0
    if empty_rows.any():
        fallback_index = torch.randint(
            n_latent, (int(empty_rows.sum().item()),), generator=generator
        )
        mixing[empty_rows] = 0.0
        mixing[empty_rows, fallback_index] = torch.randn(
            int(empty_rows.sum().item()), generator=generator
        )
    mixing = l2_normalize(mixing, dim=-1)
    row_gain = sample_uniform((n_sources, 1), 0.65, 1.35, generator)
    return mixing * row_gain


def finalize_state(state: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    state = standardize_per_channel(state)
    amplitude = sample_uniform((state.shape[0], 1), 0.45, 1.35, generator)
    source_dropout = (torch.rand(state.shape[0], 1, generator=generator) > 0.08).float()
    state = state * amplitude * source_dropout
    return state.float()


def generate_smooth_network_state(context: GenerationContext) -> torch.Tensor:
    n_latent = 8
    latent = generate_latent_ar(
        n_latent=n_latent,
        time_steps=context.time_steps,
        generator=context.generator,
        phi_range=(0.94, 0.995),
        noise_range=(0.03, 0.12),
    )
    residual = generate_latent_ar(
        n_latent=context.n_sources,
        time_steps=context.time_steps,
        generator=context.generator,
        phi_range=(0.88, 0.97),
        noise_range=(0.01, 0.06),
    )
    state = build_sparse_mixing(context.n_sources, n_latent, context.generator) @ latent
    state = state + 0.18 * residual
    return finalize_state(state, context.generator)


def generate_band_oscillation_state(context: GenerationContext) -> torch.Tensor:
    n_latent = 10
    time_axis = torch.arange(context.time_steps, dtype=torch.float32) / context.sampling_rate
    latent = torch.zeros(n_latent, context.time_steps)
    for latent_index in range(n_latent):
        frequency = float(sample_uniform((1,), 2.0, 35.0, context.generator).item())
        phase = float(sample_uniform((1,), 0.0, 2.0 * math.pi, context.generator).item())
        harmonic_scale = float(sample_uniform((1,), 0.12, 0.40, context.generator).item())
        envelope = generate_latent_ar(
            n_latent=1,
            time_steps=context.time_steps,
            generator=context.generator,
            phi_range=(0.96, 0.995),
            noise_range=(0.02, 0.08),
        )[0]
        envelope = 0.25 + 0.75 * torch.sigmoid(1.4 * envelope)
        carrier = torch.sin(2.0 * math.pi * frequency * time_axis + phase)
        harmonic = torch.sin(2.0 * math.pi * 0.5 * frequency * time_axis + 0.5 * phase)
        latent[latent_index] = envelope * (carrier + harmonic_scale * harmonic)
    state = build_sparse_mixing(context.n_sources, n_latent, context.generator) @ latent
    state = state + 0.10 * generate_latent_ar(
        n_latent=context.n_sources,
        time_steps=context.time_steps,
        generator=context.generator,
        phi_range=(0.75, 0.90),
        noise_range=(0.01, 0.04),
    )
    return finalize_state(state, context.generator)


def generate_burst_transient_state(context: GenerationContext) -> torch.Tensor:
    n_latent = 12
    time_axis = torch.arange(context.time_steps, dtype=torch.float32)
    latent = torch.zeros(n_latent, context.time_steps)
    for latent_index in range(n_latent):
        n_bursts = int(torch.randint(2, 6, (1,), generator=context.generator).item())
        series = torch.zeros(context.time_steps)
        for _ in range(n_bursts):
            center = int(
                torch.randint(0, context.time_steps, (1,), generator=context.generator).item()
            )
            width = float(
                sample_uniform(
                    (1,), 8.0, max(24.0, context.time_steps / 7.0), context.generator
                ).item()
            )
            amplitude = float(sample_uniform((1,), 0.6, 2.2, context.generator).item())
            envelope = torch.exp(-0.5 * ((time_axis - center) / width) ** 2)
            if sample_bool(0.65, context.generator):
                frequency = float(sample_uniform((1,), 4.0, 28.0, context.generator).item())
                phase = float(sample_uniform((1,), 0.0, 2.0 * math.pi, context.generator).item())
                carrier = torch.sin(
                    2.0 * math.pi * frequency * time_axis / context.sampling_rate + phase
                )
                series = series + amplitude * envelope * carrier
            else:
                sign = -1.0 if sample_bool(0.5, context.generator) else 1.0
                series = series + sign * amplitude * envelope
        latent[latent_index] = series
    state = build_sparse_mixing(context.n_sources, n_latent, context.generator) @ latent
    baseline = generate_latent_ar(
        n_latent=context.n_sources,
        time_steps=context.time_steps,
        generator=context.generator,
        phi_range=(0.65, 0.88),
        noise_range=(0.02, 0.09),
    )
    state = state + 0.12 * baseline
    return finalize_state(state, context.generator)


def generate_multiscale_hybrid_state(context: GenerationContext) -> torch.Tensor:
    smooth = generate_smooth_network_state(context)
    oscillation = generate_band_oscillation_state(context)
    burst = generate_burst_transient_state(context)
    segment_count = int(torch.randint(2, 5, (1,), generator=context.generator).item())
    weights = torch.zeros(3, context.time_steps)
    boundaries = torch.linspace(0, context.time_steps, steps=segment_count + 1).round().long()
    components = torch.stack([smooth, oscillation, burst], dim=0)
    for segment_index in range(segment_count):
        left = int(boundaries[segment_index].item())
        right = int(boundaries[segment_index + 1].item())
        active = int(torch.randint(0, 3, (1,), generator=context.generator).item())
        segment_weights = torch.full((3, right - left), 0.15)
        segment_weights[active] = 0.70
        segment_weights[(active + 1) % 3] += 0.10
        weights[:, left:right] = segment_weights
    state = (components * weights.unsqueeze(1)).sum(dim=0)
    return finalize_state(state, context.generator)


def generate_state(context: GenerationContext, sim_type: str) -> torch.Tensor:
    if sim_type == "smooth_network":
        return generate_smooth_network_state(context)
    if sim_type == "band_oscillation":
        return generate_band_oscillation_state(context)
    if sim_type == "burst_transient":
        return generate_burst_transient_state(context)
    if sim_type == "multiscale_hybrid":
        return generate_multiscale_hybrid_state(context)
    raise ValueError(f"未知 sim_type: {sim_type}")


def generate_colored_noise(
    n_sensors: int,
    time_steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    steps = 0.04 * torch.randn(n_sensors, time_steps, generator=generator)
    drift = torch.cumsum(steps, dim=-1)
    drift = standardize_per_channel(drift)
    white = torch.randn(n_sensors, time_steps, generator=generator)
    return 0.75 * drift + 0.25 * white


def generate_muscle_noise(
    n_sensors: int,
    time_steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    high_freq = torch.randn(n_sensors, time_steps + 1, generator=generator)
    high_freq = high_freq[:, 1:] - high_freq[:, :-1]
    envelopes = torch.zeros(n_sensors, time_steps)
    time_axis = torch.arange(time_steps, dtype=torch.float32)
    active_channels = torch.rand(n_sensors, generator=generator) < 0.35
    for channel_index in range(n_sensors):
        if not active_channels[channel_index]:
            continue
        burst_count = int(torch.randint(1, 4, (1,), generator=generator).item())
        for _ in range(burst_count):
            center = int(torch.randint(0, time_steps, (1,), generator=generator).item())
            width = float(sample_uniform((1,), 8.0, max(18.0, time_steps / 10.0), generator).item())
            envelopes[channel_index] += torch.exp(-0.5 * ((time_axis - center) / width) ** 2)
    return high_freq * (0.2 + 0.8 * envelopes)


def generate_dropout_artifact(clean: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    n_sensors, time_steps = clean.shape
    mask = torch.ones_like(clean)
    active_channels = torch.rand(n_sensors, generator=generator) < 0.20
    for channel_index in range(n_sensors):
        if not active_channels[channel_index]:
            continue
        segment_count = int(torch.randint(1, 4, (1,), generator=generator).item())
        for _ in range(segment_count):
            start = int(torch.randint(0, max(1, time_steps - 8), (1,), generator=generator).item())
            width = int(
                torch.randint(8, max(16, time_steps // 5), (1,), generator=generator).item()
            )
            end = min(time_steps, start + width)
            attenuation = float(sample_uniform((1,), 0.0, 0.35, generator).item())
            mask[channel_index, start:end] *= attenuation
    artifact = clean * mask - clean
    artifact = artifact + 0.15 * torch.randn(n_sensors, time_steps, generator=generator)
    return artifact


def build_artifact(
    clean: torch.Tensor, noise_type: str, generator: torch.Generator
) -> torch.Tensor:
    n_sensors, time_steps = clean.shape
    if noise_type == "white":
        return torch.randn(n_sensors, time_steps, generator=generator)
    if noise_type == "colored":
        return generate_colored_noise(n_sensors, time_steps, generator)
    if noise_type == "muscle":
        return generate_muscle_noise(n_sensors, time_steps, generator)
    if noise_type == "dropout":
        return generate_dropout_artifact(clean, generator)
    if noise_type == "mixed":
        white = torch.randn(n_sensors, time_steps, generator=generator)
        colored = generate_colored_noise(n_sensors, time_steps, generator)
        muscle = generate_muscle_noise(n_sensors, time_steps, generator)
        dropout = generate_dropout_artifact(clean, generator)
        return 0.30 * white + 0.25 * colored + 0.20 * muscle + 0.25 * dropout
    raise ValueError(f"未知 noise_type: {noise_type}")


def apply_noise(
    clean: torch.Tensor,
    noise_type: str,
    snr_db: float,
    generator: torch.Generator,
) -> torch.Tensor:
    artifact = build_artifact(clean, noise_type, generator)
    artifact = standardize_global(artifact)
    clean_std = clean.std().clamp_min(1e-6)
    target_noise_std = clean_std / (10.0 ** (snr_db / 20.0))
    return clean + artifact * target_noise_std


def choose_layout(context: GenerationContext) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]:
    use_real = False
    if context.layout_mode == "real_only":
        use_real = True
    elif context.layout_mode == "hybrid_real":
        use_real = bool(context.real_layouts) and sample_bool(
            context.real_layout_prob, context.generator
        )

    if use_real:
        if not context.real_layouts:
            raise RuntimeError("layout_mode=real_only，但没有可用的真实缓存布局")
        template = context.real_layouts[
            int(torch.randint(len(context.real_layouts), (1,), generator=context.generator).item())
        ]
        return (
            template.pos.clone(),
            template.leadfield.clone(),
            {
                "layout_type": template.layout_type,
                "layout_source": template.layout_source,
                "layout_dataset": template.layout_dataset,
                "layout_fingerprint": template.layout_fingerprint,
                "layout_full_fingerprint": template.layout_full_fingerprint,
                "layout_leadfield_cache_path": template.layout_leadfield_cache_path,
            },
        )

    layout_type = sample_choice(context.synthetic_layouts, context.generator)
    sensor_xyz = sample_sensor_xyz(context.n_sensors, layout_type, context.generator)
    pos = build_pos_from_xyz(sensor_xyz)
    leadfield = build_geometry_leadfield(sensor_xyz, context.source_positions, context.generator)
    return (
        pos,
        leadfield,
        {
            "layout_type": layout_type,
            "layout_source": "synthetic_geometry",
            "layout_dataset": "synthetic_geometry",
            "layout_fingerprint": "",
            "layout_full_fingerprint": "",
            "layout_leadfield_cache_path": "",
        },
    )


def generate_sample(context: GenerationContext) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    sim_type = sample_choice(context.sim_types, context.generator)
    noise_type = sample_choice(context.noise_types, context.generator)
    snr_db = float(
        sample_uniform((1,), context.snr_db_min, context.snr_db_max, context.generator).item()
    )

    pos, leadfield, layout_meta = choose_layout(context)
    s_true = generate_state(context, sim_type)
    clean = torch.einsum("cs,st->ct", leadfield, s_true)
    x = apply_noise(clean, noise_type, snr_db, context.generator)
    x = x * sample_uniform((context.n_sensors, 1), 0.92, 1.08, context.generator)

    sample = {
        "x": x.float(),
        "pos": pos.float(),
        "sensor_type": torch.zeros(context.n_sensors, dtype=torch.long),
        "leadfield": leadfield.float(),
        "s_true": s_true.float(),
    }
    metadata = {
        "sim_type": sim_type,
        "noise_type": noise_type,
        "layout_type": layout_meta["layout_type"],
        "layout_source": layout_meta["layout_source"],
        "layout_dataset": layout_meta["layout_dataset"],
        "layout_fingerprint": layout_meta["layout_fingerprint"],
        "layout_full_fingerprint": layout_meta["layout_full_fingerprint"],
        "layout_leadfield_cache_path": layout_meta["layout_leadfield_cache_path"],
        "snr_db": round(snr_db, 4),
    }
    return sample, metadata


def dump_split(
    output_dir: Path,
    split: str,
    n_samples: int,
    context: GenerationContext,
    print_every: int,
) -> Tuple[str, Dict[str, Any]]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    metadata: List[Dict[str, Any]] = []
    sim_counts: Counter[str] = Counter()
    noise_counts: Counter[str] = Counter()
    layout_counts: Counter[str] = Counter()
    layout_source_counts: Counter[str] = Counter()
    layout_dataset_counts: Counter[str] = Counter()
    snr_values: List[float] = []

    for sample_index in range(n_samples):
        sample, sample_meta = generate_sample(context)
        sample_path = split_dir / f"sample_{sample_index:06d}.pt"
        torch.save(sample, sample_path)

        record = {"path": str(sample_path), **sample_meta}
        metadata.append(record)
        sim_counts[sample_meta["sim_type"]] += 1
        noise_counts[sample_meta["noise_type"]] += 1
        layout_counts[sample_meta["layout_type"]] += 1
        layout_source_counts[sample_meta["layout_source"]] += 1
        layout_dataset_counts[sample_meta["layout_dataset"]] += 1
        snr_values.append(float(sample_meta["snr_db"]))

        if print_every > 0 and (sample_index + 1) % print_every == 0:
            print(
                f"[{split}] generated {sample_index + 1}/{n_samples} samples | "
                f"sim={sample_meta['sim_type']} noise={sample_meta['noise_type']} "
                f"layout={sample_meta['layout_type']} snr_db={sample_meta['snr_db']:.2f}",
                flush=True,
            )

    meta_path = output_dir / f"stage1_{split}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    summary = {
        "n_samples": n_samples,
        "sim_type_counts": dict(sorted(sim_counts.items())),
        "noise_type_counts": dict(sorted(noise_counts.items())),
        "layout_type_counts": dict(sorted(layout_counts.items())),
        "layout_source_counts": dict(sorted(layout_source_counts.items())),
        "layout_dataset_counts": dict(sorted(layout_dataset_counts.items())),
        "snr_db": {
            "min": min(snr_values) if snr_values else None,
            "max": max(snr_values) if snr_values else None,
            "mean": sum(snr_values) / len(snr_values) if snr_values else None,
        },
        "metadata_path": str(meta_path),
    }
    return str(meta_path), summary


def parse_list_arg(raw: str, default: Sequence[str]) -> Tuple[str, ...]:
    if not raw.strip():
        return tuple(default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def build_context(args: argparse.Namespace) -> GenerationContext:
    source_positions = build_source_positions(args.n_sources)
    seen_full_fingerprints: set = set()
    registry_layouts = load_real_layout_pool(
        registry_path=args.registry_path,
        n_sensors=args.n_sensors,
        n_sources=args.n_sources,
        max_real_layouts=args.max_real_layouts,
        sim_leadfield_cache_dir=args.sim_leadfield_cache_dir,
        seen_full_fingerprints=seen_full_fingerprints,
    )
    external_layouts = load_external_template_pool(
        template_roots=parse_path_list_arg(args.external_template_roots),
        n_sensors=args.n_sensors,
        source_positions=source_positions,
        max_external_layouts=args.max_external_layouts,
        base_seed=args.seed,
        sim_leadfield_cache_dir=args.sim_leadfield_cache_dir,
        seen_full_fingerprints=seen_full_fingerprints,
    )
    real_layouts = list(registry_layouts) + list(external_layouts)
    if args.layout_mode == "real_only" and not real_layouts:
        raise RuntimeError(
            "layout_mode=real_only，但 registry 与外部模板库中都没有匹配当前 n_sensors 的真实布局"
        )
    if args.layout_mode == "hybrid_real" and not real_layouts:
        print("[warning] 未找到可用真实缓存布局，将退化为 synthetic_only", flush=True)
        layout_mode = "synthetic_only"
    else:
        layout_mode = args.layout_mode

    return GenerationContext(
        n_sensors=args.n_sensors,
        n_sources=args.n_sources,
        time_steps=args.time_steps,
        sampling_rate=args.sampling_rate,
        layout_mode=layout_mode,
        real_layout_prob=args.real_layout_prob,
        snr_db_min=args.snr_db_min,
        snr_db_max=args.snr_db_max,
        sim_types=parse_list_arg(args.sim_types, DEFAULT_SIM_TYPES),
        noise_types=parse_list_arg(args.noise_types, DEFAULT_NOISE_TYPES),
        synthetic_layouts=parse_list_arg(args.synthetic_layouts, DEFAULT_SYNTHETIC_LAYOUTS),
        source_positions=source_positions,
        real_layouts=real_layouts,
        registry_layout_count=len(registry_layouts),
        external_layout_count=len(external_layouts),
        generator=torch.Generator().manual_seed(args.seed),
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Generate Stage1 simulation pretraining data")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument(
        "--preset",
        type=str,
        default="formal",
        choices=sorted(PRESETS.keys()),
        help="默认样本规模与 SNR/真实布局比例预设",
    )
    parser.add_argument("--n_train", type=int, default=None, help="训练样本数；默认取 preset")
    parser.add_argument("--n_val", type=int, default=None, help="验证样本数；默认取 preset")
    parser.add_argument("--n_sensors", type=int, default=128, help="传感器数量")
    parser.add_argument("--n_sources", type=int, default=72, help="源数量")
    parser.add_argument("--time_steps", type=int, default=512, help="时间长度")
    parser.add_argument(
        "--sampling_rate", type=float, default=256.0, help="采样率，仅用于振荡频率标定"
    )
    parser.add_argument(
        "--layout_mode",
        type=str,
        default="hybrid_real",
        choices=["hybrid_real", "real_only", "synthetic_only"],
        help="布局/导联场来源模式",
    )
    parser.add_argument(
        "--registry_path",
        type=str,
        default="/work/2024/tanzunsheng/leadfield_cache/fingerprint_registry.pt",
        help="真实缓存布局注册表路径",
    )
    parser.add_argument(
        "--max_real_layouts", type=int, default=0, help="限制真实布局池大小；0 表示全部"
    )
    parser.add_argument(
        "--external_template_roots",
        type=str,
        default="/work/2024/tanzunsheng/EEG_Electrode_Templates",
        help="逗号分隔的外部模板库根目录；会递归搜索 template_meta.json",
    )
    parser.add_argument(
        "--max_external_layouts",
        type=int,
        default=0,
        help="限制外部模板布局池大小；0 表示全部",
    )
    parser.add_argument(
        "--sim_leadfield_cache_dir",
        type=str,
        default="",
        help="可选：将仿真复用布局的导联场单独缓存到该目录，和真实 leadfield_cache 分离",
    )
    parser.add_argument(
        "--real_layout_prob",
        type=float,
        default=None,
        help="hybrid_real 模式下采样真实缓存布局的概率；默认取 preset",
    )
    parser.add_argument(
        "--snr_db_min",
        type=float,
        default=None,
        help="最小 SNR(dB)；默认取 preset",
    )
    parser.add_argument(
        "--snr_db_max",
        type=float,
        default=None,
        help="最大 SNR(dB)；默认取 preset",
    )
    parser.add_argument(
        "--sim_types",
        type=str,
        default=",".join(DEFAULT_SIM_TYPES),
        help="逗号分隔的仿真动力学类型列表",
    )
    parser.add_argument(
        "--noise_types",
        type=str,
        default=",".join(DEFAULT_NOISE_TYPES),
        help="逗号分隔的噪声/伪迹类型列表",
    )
    parser.add_argument(
        "--synthetic_layouts",
        type=str,
        default=",".join(DEFAULT_SYNTHETIC_LAYOUTS),
        help="逗号分隔的合成布局类型列表",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--print_every", type=int, default=500, help="生成进度打印步长；0 表示静默")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    args.n_train = int(preset["n_train"] if args.n_train is None else args.n_train)
    args.n_val = int(preset["n_val"] if args.n_val is None else args.n_val)
    args.real_layout_prob = float(
        preset["real_layout_prob"] if args.real_layout_prob is None else args.real_layout_prob
    )
    args.snr_db_min = float(preset["snr_db_min"] if args.snr_db_min is None else args.snr_db_min)
    args.snr_db_max = float(preset["snr_db_max"] if args.snr_db_max is None else args.snr_db_max)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = build_context(args)

    train_meta, train_summary = dump_split(
        output_dir=output_dir,
        split="train",
        n_samples=args.n_train,
        context=context,
        print_every=args.print_every,
    )
    val_meta, val_summary = dump_split(
        output_dir=output_dir,
        split="val",
        n_samples=args.n_val,
        context=context,
        print_every=args.print_every,
    )

    summary = {
        "preset": args.preset,
        "output_dir": str(output_dir),
        "n_sensors": args.n_sensors,
        "n_sources": args.n_sources,
        "time_steps": args.time_steps,
        "sampling_rate": args.sampling_rate,
        "layout_mode": context.layout_mode,
        "real_layout_prob": args.real_layout_prob,
        "n_real_layouts_loaded": len(context.real_layouts),
        "n_registry_layouts_loaded": context.registry_layout_count,
        "n_external_layouts_loaded": context.external_layout_count,
        "sim_leadfield_cache_dir": args.sim_leadfield_cache_dir or None,
        "external_template_roots": list(parse_path_list_arg(args.external_template_roots)),
        "sim_types": list(context.sim_types),
        "noise_types": list(context.noise_types),
        "synthetic_layouts": list(context.synthetic_layouts),
        "train": train_summary,
        "val": val_summary,
        "stage1_train_metadata": train_meta,
        "stage1_val_metadata": val_meta,
    }
    summary_path = output_dir / "stage1_dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
