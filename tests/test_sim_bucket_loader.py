# -*- coding: utf-8 -*-
"""仿真混合通道分桶加载回归测试。"""

import json
from pathlib import Path

import pytest
import torch

from penci.v1.data import Stage1SimulationDataset, create_simulation_dataloader
from scripts.v1.train_stage1 import build_stage1_loaders


def _write_stage1_sample(path: Path, n_channels: int, fill_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.full((n_channels, 32), fill_value, dtype=torch.float32)
    pos = torch.zeros((n_channels, 6), dtype=torch.float32)
    pos[:, 0] = torch.linspace(0.0, 1.0, n_channels)
    sensor_type = torch.zeros(n_channels, dtype=torch.long)
    leadfield = torch.full((n_channels, 72), fill_value, dtype=torch.float32)
    s_true = torch.full((72, 32), fill_value, dtype=torch.float32)
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


def _build_mixed_metadata(tmp_path: Path) -> list[Path]:
    root_8 = tmp_path / "stage1_all_templates" / "8ch"
    root_16 = tmp_path / "stage1_all_templates" / "16ch"

    metadata_8 = []
    for idx in range(2):
        sample_path = root_8 / "train" / f"sample_{idx:06d}.pt"
        _write_stage1_sample(sample_path, n_channels=8, fill_value=3.0)
        metadata_8.append(
            {
                "path": str(sample_path),
                "layout_fingerprint": "fp_08_a",
            }
        )

    metadata_16 = []
    for idx in range(2):
        sample_path = root_16 / "train" / f"sample_{idx:06d}.pt"
        _write_stage1_sample(sample_path, n_channels=16, fill_value=1.0)
        metadata_16.append(
            {
                "path": str(sample_path),
                "layout_fingerprint": "fp_16_a",
            }
        )
    for idx in range(2, 4):
        sample_path = root_16 / "train" / f"sample_{idx:06d}.pt"
        _write_stage1_sample(sample_path, n_channels=16, fill_value=2.0)
        metadata_16.append(
            {
                "path": str(sample_path),
                "layout_fingerprint": "fp_16_b",
            }
        )

    meta_8_path = tmp_path / "8ch_train_metadata.json"
    meta_16_path = tmp_path / "16ch_train_metadata.json"
    meta_8_path.write_text(json.dumps(metadata_8, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_16_path.write_text(json.dumps(metadata_16, ensure_ascii=False, indent=2), encoding="utf-8")
    return [meta_8_path, meta_16_path]


def test_stage1_sim_dataset_supports_multiple_metadata_sources(tmp_path):
    metadata_paths = _build_mixed_metadata(tmp_path)
    dataset = Stage1SimulationDataset(metadata_path=metadata_paths)

    assert len(dataset) == 6
    assert {dataset.get_channel_count(idx) for idx in range(len(dataset))} == {8, 16}
    assert {dataset.get_fingerprint(idx) for idx in range(len(dataset))} == {
        "fp_08_a",
        "fp_16_a",
        "fp_16_b",
    }


def test_create_simulation_dataloader_buckets_mixed_channels_and_fingerprints(tmp_path):
    metadata_paths = _build_mixed_metadata(tmp_path)
    dataset = Stage1SimulationDataset(metadata_path=metadata_paths)
    loader = create_simulation_dataloader(
        dataset=dataset,
        batch_size=2,
        num_workers=0,
        shuffle=False,
        use_bucket_sampler=True,
        use_fingerprint=True,
        drop_last=False,
    )

    batches = list(loader)
    assert len(batches) == 3
    assert sorted(batch["x"].shape[1] for batch in batches) == [8, 16, 16]

    batch_constants = set()
    for batch in batches:
        per_sample_means = batch["x"].mean(dim=(1, 2)).tolist()
        rounded = {round(value, 4) for value in per_sample_means}
        assert len(rounded) == 1
        batch_constants.add(next(iter(rounded)))

    assert batch_constants == {1.0, 2.0, 3.0}


def test_create_simulation_dataloader_requires_bucketing_for_mixed_channels(tmp_path):
    metadata_paths = _build_mixed_metadata(tmp_path)
    dataset = Stage1SimulationDataset(metadata_path=metadata_paths)

    with pytest.raises(RuntimeError, match="多个通道数"):
        create_simulation_dataloader(
            dataset=dataset,
            batch_size=2,
            num_workers=0,
            shuffle=True,
            use_bucket_sampler=False,
        )


def test_build_stage1_loaders_accepts_mixed_sim_metadata_list(tmp_path):
    metadata_paths = _build_mixed_metadata(tmp_path)
    config = {
        "data": {
            "stage1_sim_train_metadata": [str(path) for path in metadata_paths],
            "stage1_sim_val_metadata": [str(path) for path in metadata_paths],
            "use_bucket_sampler": True,
            "use_fingerprint": True,
        },
        "training": {
            "batch_size": 2,
            "num_workers": 0,
        },
    }

    train_loader, val_loader, _ = build_stage1_loaders(
        config=config,
        mode="sim_pretrain",
        rank=0,
        world_size=1,
    )

    train_batches = list(train_loader)
    val_batches = list(val_loader)
    assert len(train_batches) == 3
    assert len(val_batches) == 3
    assert {batch["x"].shape[1] for batch in train_batches} == {8, 16}
    assert {batch["x"].shape[1] for batch in val_batches} == {8, 16}
