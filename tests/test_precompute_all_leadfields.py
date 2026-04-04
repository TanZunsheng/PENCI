# -*- coding: utf-8 -*-
"""离线指纹预计算脚本的目录级复用测试。"""

import json
from pathlib import Path

import torch

from scripts.precompute_all_leadfields import (
    build_fingerprint_group_key,
    compute_fingerprint_group_task,
    compute_fingerprint_from_pt,
    scan_and_fingerprint,
    select_group_validation_indices,
)


def _write_pt(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pos = torch.full((4, 6), float(value), dtype=torch.bfloat16)
    torch.save(
        {
            "x": torch.zeros((4, 8), dtype=torch.bfloat16),
            "pos": pos,
            "sensor_type": torch.zeros(4, dtype=torch.int32),
        },
        path,
    )


def test_select_group_validation_indices_is_deterministic():
    sample_paths = [f"/tmp/sample_{index:03d}.pt" for index in range(10)]
    indices_a = select_group_validation_indices("demo|4|/tmp/run_a", sample_paths)
    indices_b = select_group_validation_indices("demo|4|/tmp/run_a", sample_paths)

    assert indices_a == indices_b
    assert indices_a[0] == 0
    assert indices_a[-1] == 9
    assert len(indices_a) == 4


def test_compute_fingerprint_group_task_small_group_falls_back_to_per_sample(tmp_path):
    sample_paths = []
    for index in range(4):
        sample_path = tmp_path / "demo" / "run_small" / f"{index}_data.pt"
        _write_pt(sample_path, value=1.0)
        sample_paths.append(str(sample_path))

    result = compute_fingerprint_group_task("Demo|4|run_small", sample_paths)

    assert result["mode"] == "per_sample"
    assert result["fallback_reason"] == "small_group"
    assert len(result["sample_fingerprints"]) == 4
    assert len({fp for _, fp in result["sample_fingerprints"]}) == 1


def test_compute_fingerprint_group_task_validation_mismatch_falls_back(tmp_path):
    group_key = "Demo|4|run_mismatch"
    sample_paths = []
    for index in range(6):
        sample_path = tmp_path / "demo" / "run_mismatch" / f"{index}_data.pt"
        _write_pt(sample_path, value=1.0)
        sample_paths.append(str(sample_path))

    selected_indices = select_group_validation_indices(group_key, sorted(sample_paths))
    mismatch_index = selected_indices[-1]
    _write_pt(Path(sorted(sample_paths)[mismatch_index]), value=9.0)

    result = compute_fingerprint_group_task(group_key, sample_paths)

    assert result["mode"] == "per_sample"
    assert result["fallback_reason"] == "validation_mismatch"
    fps = {path: fp for path, fp in result["sample_fingerprints"]}
    assert fps[sorted(sample_paths)[0]] != fps[sorted(sample_paths)[mismatch_index]]


def test_scan_and_fingerprint_reuses_directory_and_falls_back_on_mismatch(tmp_path):
    data_root = tmp_path / "PENCIData"
    dataset_root = data_root / "DemoSet"
    metadata_root = data_root / "DemoSet-metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)

    records = []

    stable_dir = dataset_root / "derivatives" / "preprocessing" / "sub-01" / "eeg" / "run_stable"
    for index in range(6):
        sample_path = stable_dir / f"{index}_data.pt"
        _write_pt(sample_path, value=1.0)
        records.append(
            {
                "dataset": "DemoSet",
                "path": str(sample_path),
                "channels": 4,
                "is_eeg": True,
                "is_meg": False,
            }
        )

    mismatch_dir = dataset_root / "derivatives" / "preprocessing" / "sub-01" / "eeg" / "run_mismatch"
    mismatch_paths = []
    for index in range(6):
        sample_path = mismatch_dir / f"{index}_data.pt"
        _write_pt(sample_path, value=1.0)
        mismatch_paths.append(sample_path)

    mismatch_key = build_fingerprint_group_key(
        {
            "dataset": "DemoSet",
            "path": str(mismatch_paths[0]),
            "channels": 4,
        }
    )
    selected_indices = select_group_validation_indices(
        mismatch_key,
        [str(path) for path in sorted(mismatch_paths)],
    )
    _write_pt(sorted(mismatch_paths)[selected_indices[-1]], value=9.0)

    for sample_path in mismatch_paths:
        records.append(
            {
                "dataset": "DemoSet",
                "path": str(sample_path),
                "channels": 4,
                "is_eeg": True,
                "is_meg": False,
            }
        )

    (metadata_root / "train.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    unique_fps, path_to_fp, all_metadata = scan_and_fingerprint(str(data_root), max_workers=1)

    assert len(all_metadata) == len(records)
    assert len(path_to_fp) == len(records)
    assert len(unique_fps) == 2

    stable_fp = compute_fingerprint_from_pt(str(stable_dir / "0_data.pt"))
    mismatch_base_fp = compute_fingerprint_from_pt(str(mismatch_dir / "0_data.pt"))
    mismatch_tail_fp = compute_fingerprint_from_pt(str(sorted(mismatch_paths)[selected_indices[-1]]))

    assert stable_fp == mismatch_base_fp
    assert mismatch_tail_fp != mismatch_base_fp
    assert path_to_fp[str(stable_dir / "0_data.pt")] == stable_fp
    assert path_to_fp[str(mismatch_dir / "0_data.pt")] == mismatch_base_fp
    assert path_to_fp[str(sorted(mismatch_paths)[selected_indices[-1]])] == mismatch_tail_fp
