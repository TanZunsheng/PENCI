# -*- coding: utf-8 -*-
"""
文件级调度的回归测试。
"""

import json
from typing import Dict, List

from penci.data.dataset import DistributedBucketBatchSampler, get_train_val_loaders


class _DummyDataset:
    """仅提供 sampler 所需接口的最小数据集桩。"""

    def __init__(self, file_batch_counts: List[int], batch_size: int):
        self.metadata: List[Dict] = []
        for file_idx, n_batches in enumerate(file_batch_counts):
            hdf5_path = f"file_{file_idx}.h5"
            for hdf5_idx in range(n_batches * batch_size):
                self.metadata.append(
                    {
                        "channels": 128,
                        "fingerprint": "fp_128",
                        "hdf5_path": hdf5_path,
                        "hdf5_idx": hdf5_idx,
                    }
                )

    def __len__(self) -> int:
        return len(self.metadata)

    def get_channel_count(self, idx: int) -> int:
        return int(self.metadata[idx]["channels"])

    def get_fingerprint(self, idx: int) -> str:
        return str(self.metadata[idx]["fingerprint"])


def test_file_scheduler_balances_batches_without_mixing_files():
    """
    文件级调度应优先平衡各 rank 的 batch 数，避免过度截断验证 batch。

    这个用例里 3 个文件分别贡献 6 / 5 / 5 个完整 batch。
    旧的按文件 round-robin 分配会落成 11 vs 5，最终只能保留 5 vs 5。
    修复后应能保留 6 vs 6，同时每个 batch 仍只来自单个文件。
    """
    batch_size = 2
    dataset = _DummyDataset(file_batch_counts=[6, 5, 5], batch_size=batch_size)

    samplers = [
        DistributedBucketBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            num_replicas=2,
            rank=rank,
            shuffle=False,
            drop_last=True,
            use_fingerprint=True,
            cluster_by_file=True,
            file_scheduler=True,
            shuffle_within_file=False,
        )
        for rank in range(2)
    ]

    rank_batches = [list(iter(sampler)) for sampler in samplers]

    assert [len(batches) for batches in rank_batches] == [6, 6]

    covered = set()
    for batches in rank_batches:
        for batch in batches:
            batch_files = {dataset.metadata[idx]["hdf5_path"] for idx in batch}
            assert len(batch_files) == 1
            covered.update(batch)

    assert len(covered) == 24


def test_get_train_val_loaders_strips_val_sampler_kwargs_from_train_dataset(tmp_path):
    """验证集专用的 sampler 参数不应透传到训练集 dataset 构造。"""
    metadata_root = tmp_path / "ToyDataset-metadata"
    metadata_root.mkdir()

    sample_meta = [
        {
            "path": str(tmp_path / "sample_0.pt"),
            "channels": 128,
            "fingerprint": "fp_128",
            "hdf5_path": "toy_0.h5",
            "hdf5_idx": 0,
        },
        {
            "path": str(tmp_path / "sample_1.pt"),
            "channels": 128,
            "fingerprint": "fp_128",
            "hdf5_path": "toy_0.h5",
            "hdf5_idx": 1,
        },
    ]

    for split in ("train", "val"):
        with open(metadata_root / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(sample_meta, f)

    train_loader, val_loader = get_train_val_loaders(
        data_root=str(tmp_path),
        datasets=["ToyDataset"],
        batch_size=1,
        num_workers=0,
        use_bucket_sampler=True,
        use_fingerprint=True,
        rank=0,
        world_size=1,
        max_length=2560,
        target_channels=None,
        precompute_fingerprints=False,
        val_sampler_file_scheduler=True,
        val_sampler_shuffle_within_file=False,
    )

    assert train_loader is not None
    assert val_loader is not None
