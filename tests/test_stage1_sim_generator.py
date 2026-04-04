# -*- coding: utf-8 -*-
"""Stage1 正式仿真生成器回归测试。"""

import json
import subprocess
import sys
from pathlib import Path

import torch

from penci.v1.data import Stage1SimulationDataset


def test_stage1_sim_generator_rich_metadata(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "v1" / "generate_stage1_sim_data.py"
    output_dir = tmp_path / "stage1_sim"
    command = [
        sys.executable,
        str(script),
        "--output_dir",
        str(output_dir),
        "--preset",
        "smoke",
        "--n_train",
        "6",
        "--n_val",
        "2",
        "--n_sensors",
        "16",
        "--time_steps",
        "64",
        "--layout_mode",
        "synthetic_only",
        "--print_every",
        "0",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    summary_path = output_dir / "stage1_dataset_summary.json"
    assert summary_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["train"]["n_samples"] == 6
    assert summary["val"]["n_samples"] == 2
    assert summary["layout_mode"] == "synthetic_only"
    assert set(summary["sim_types"]) >= {
        "smooth_network",
        "band_oscillation",
        "burst_transient",
        "multiscale_hybrid",
    }

    train_meta_path = Path(summary["stage1_train_metadata"])
    metadata = json.loads(train_meta_path.read_text(encoding="utf-8"))
    assert len(metadata) == 6
    first_record = metadata[0]
    for key in ["path", "sim_type", "noise_type", "layout_type", "snr_db"]:
        assert key in first_record

    dataset = Stage1SimulationDataset(metadata_path=str(train_meta_path))
    item = dataset[0]
    assert set(item.keys()) == {"x", "pos", "sensor_type", "leadfield", "s_true"}
    assert item["x"].shape == torch.Size([16, 64])
    assert item["pos"].shape == torch.Size([16, 6])
    assert item["leadfield"].shape == torch.Size([16, 72])
    assert item["s_true"].shape == torch.Size([72, 64])
    assert torch.allclose(item["pos"][:, 3:], torch.zeros_like(item["pos"][:, 3:]))


def test_stage1_sim_generator_supports_external_templates(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "v1" / "generate_stage1_sim_data.py"
    template_dir = tmp_path / "templates" / "DemoEEG" / "16ch" / "demo-fp"
    template_dir.mkdir(parents=True, exist_ok=True)

    positions = []
    for index in range(16):
        angle = 2.0 * torch.pi * index / 16.0
        positions.append(
            [
                float(0.08 * torch.cos(torch.tensor(angle)).item()),
                float(0.08 * torch.sin(torch.tensor(angle)).item()),
                float(0.05 + 0.01 * (index % 4)),
            ]
        )

    template_meta = {
        "dataset_name": "DemoEEG",
        "n_channels": 16,
        "channel_names": [f"Ch{index:02d}" for index in range(16)],
        "channel_positions_m": positions,
    }
    (template_dir / "template_meta.json").write_text(
        json.dumps(template_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_dir = tmp_path / "stage1_external"
    sim_cache_dir = tmp_path / "sim_leadfield_cache"
    command = [
        sys.executable,
        str(script),
        "--output_dir",
        str(output_dir),
        "--preset",
        "smoke",
        "--n_train",
        "4",
        "--n_val",
        "2",
        "--n_sensors",
        "16",
        "--time_steps",
        "64",
        "--layout_mode",
        "real_only",
        "--registry_path",
        str(tmp_path / "missing_registry.pt"),
        "--external_template_roots",
        str(tmp_path / "templates"),
        "--sim_leadfield_cache_dir",
        str(sim_cache_dir),
        "--print_every",
        "0",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    summary = json.loads((output_dir / "stage1_dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["layout_mode"] == "real_only"
    assert summary["n_registry_layouts_loaded"] == 0
    assert summary["n_external_layouts_loaded"] == 1
    assert summary["n_real_layouts_loaded"] == 1
    assert summary["sim_leadfield_cache_dir"] == str(sim_cache_dir)

    metadata = json.loads(Path(summary["stage1_train_metadata"]).read_text(encoding="utf-8"))
    assert len(metadata) == 4
    assert metadata[0]["layout_type"] == "external_template"
    assert metadata[0]["layout_source"] == "external_template_geometry"
    assert metadata[0]["layout_dataset"] == "DemoEEG"
    assert metadata[0]["layout_leadfield_cache_path"].endswith(".pt")
    assert Path(metadata[0]["layout_leadfield_cache_path"]).is_file()
