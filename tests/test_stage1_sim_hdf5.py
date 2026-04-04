# -*- coding: utf-8 -*-
"""Stage1 仿真 HDF5 转换与加载回归测试。"""

import json
import subprocess
import sys
from pathlib import Path

import torch

from penci.v1.data import Stage1SimulationDataset, create_simulation_dataloader


def _write_stage1_sample(path: Path, n_channels: int, fill_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.full((n_channels, 32), fill_value, dtype=torch.float32)
    pos = torch.zeros((n_channels, 6), dtype=torch.float32)
    pos[:, 0] = torch.linspace(0.0, 1.0, n_channels)
    sensor_type = torch.zeros(n_channels, dtype=torch.long)
    leadfield = torch.full((n_channels, 72), fill_value * 0.1, dtype=torch.float32)
    s_true = torch.full((72, 32), fill_value * 0.2, dtype=torch.float32)
    torch.save(
        {
            "x": x,
            "pos": pos,
            "sensor_type": sensor_type,
            "leadfield": leadfield,
            "s_true": s_true,
        },
        path,
    )


def _build_stage1_sim_tree(root: Path) -> None:
    specs = {
        "8ch": {
            "train": [(0, "fp_08", 1.0), (1, "fp_08", 2.0)],
            "val": [(0, "fp_08", 3.0)],
        },
        "16ch": {
            "train": [(0, "fp_16_a", 4.0), (1, "fp_16_b", 5.0)],
            "val": [(0, "fp_16_a", 6.0)],
        },
    }

    for channel_name, split_specs in specs.items():
        n_channels = int(channel_name[:-2])
        channel_dir = root / channel_name
        for split, rows in split_specs.items():
            metadata = []
            for idx, fingerprint, fill_value in rows:
                sample_path = channel_dir / split / f"sample_{idx:06d}.pt"
                _write_stage1_sample(sample_path, n_channels=n_channels, fill_value=fill_value)
                metadata.append(
                    {
                        "path": str(sample_path),
                        "layout_fingerprint": fingerprint,
                        "snr_db": 10.0 + fill_value,
                    }
                )
            metadata_path = channel_dir / f"stage1_{split}_metadata.json"
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def test_convert_stage1_sim_to_hdf5_round_trip(tmp_path):
    input_root = tmp_path / "stage1_all_templates"
    _build_stage1_sim_tree(input_root)
    output_root = tmp_path / "stage1_hdf5"

    script = Path(__file__).resolve().parents[1] / "scripts" / "v1" / "convert_stage1_sim_to_hdf5.py"
    command = [
        sys.executable,
        str(script),
        "--input_root",
        str(input_root),
        "--output_root",
        str(output_root),
        "--max_samples_per_file",
        "2",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    train_meta_hdf5 = input_root / "8ch" / "stage1_train_metadata_hdf5.json"
    assert train_meta_hdf5.is_file()
    aggregate_train = output_root / "stage1_train_metadata_hdf5_all.json"
    aggregate_val = output_root / "stage1_val_metadata_hdf5_all.json"
    assert aggregate_train.is_file()
    assert aggregate_val.is_file()

    aggregate_records = json.loads(aggregate_train.read_text(encoding="utf-8"))
    assert all("hdf5_path" in record and "hdf5_idx" in record for record in aggregate_records)
    assert all(Path(record["hdf5_path"]).is_file() for record in aggregate_records)

    dataset = Stage1SimulationDataset(metadata_path=str(train_meta_hdf5))
    item = dataset[0]
    assert item["x"].shape == torch.Size([8, 32])
    assert item["leadfield"].shape == torch.Size([8, 72])
    assert torch.allclose(item["x"], torch.full((8, 32), 1.0))
    assert torch.allclose(item["leadfield"], torch.full((8, 72), 0.1))

    mixed_dataset = Stage1SimulationDataset(metadata_path=[str(aggregate_train), str(aggregate_val)])
    loader = create_simulation_dataloader(
        dataset=mixed_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        use_bucket_sampler=True,
        use_fingerprint=True,
        file_scheduler=True,
        drop_last=False,
    )
    batch_sampler = loader.batch_sampler
    assert batch_sampler is not None
    prefetch_plan = batch_sampler.get_prefetch_file_plan()
    assert prefetch_plan
    assert all(path.endswith(".h5") for path in prefetch_plan)
