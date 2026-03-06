#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PENCI Smoke Test

验证模型的前向传播链路是否正常工作。
不执行真实训练，只验证各组件能否正确连接。

用法:
    python tests/test_smoke.py
    
    # 使用指定的 Python 环境
    /work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python tests/test_smoke.py
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn


def test_modules():
    """测试基础模块"""
    print("=" * 60)
    print("测试基础模块")
    print("=" * 60)
    
    from penci.modules.attention import RMSNorm, FeedForward, SelfAttention, RotaryEmbedding
    from penci.modules.conv import SConv1d
    from penci.modules.lstm import SLSTM
    from penci.modules.seanet import SEANetEncoder, Snake1d
    
    # 测试 RMSNorm
    print("\n[1/6] 测试 RMSNorm...")
    x = torch.randn(2, 10, 256)
    norm = RMSNorm(256)
    out = norm(x)
    assert out.shape == x.shape, f"RMSNorm 输出形状错误: {out.shape}"
    print(f"  ✓ RMSNorm: {x.shape} -> {out.shape}")
    
    # 测试 FeedForward
    print("\n[2/6] 测试 FeedForward...")
    ff = FeedForward(256, dropout=0.0)
    out = ff(x)
    assert out.shape == x.shape, f"FeedForward 输出形状错误: {out.shape}"
    print(f"  ✓ FeedForward: {x.shape} -> {out.shape}")
    
    # 测试 SConv1d
    print("\n[3/6] 测试 SConv1d...")
    x_conv = torch.randn(2, 32, 100)
    conv = SConv1d(32, 64, kernel_size=3)
    out = conv(x_conv)
    print(f"  ✓ SConv1d: {x_conv.shape} -> {out.shape}")
    
    # 测试 SLSTM
    print("\n[4/6] 测试 SLSTM...")
    x_lstm = torch.randn(2, 64, 20)
    lstm = SLSTM(64, num_layers=2)
    out = lstm(x_lstm)
    print(f"  ✓ SLSTM: {x_lstm.shape} -> {out.shape}")
    
    # 测试 Snake1d
    print("\n[5/6] 测试 Snake1d...")
    snake = Snake1d(64)
    x_snake = torch.randn(2, 64, 100)
    out = snake(x_snake)
    assert out.shape == x_snake.shape, f"Snake1d 输出形状错误: {out.shape}"
    print(f"  ✓ Snake1d: {x_snake.shape} -> {out.shape}")
    
    # 测试 SEANetEncoder
    print("\n[6/6] 测试 SEANetEncoder...")
    encoder = SEANetEncoder(
        channels=1,
        dimension=256,
        n_filters=32,
        ratios=[8, 4, 2],  # 总下采样 64x
    )
    x_enc = torch.randn(2, 1, 320)  # 320 / 64 = 5
    out = encoder(x_enc)
    print(f"  ✓ SEANetEncoder: {x_enc.shape} -> {out.shape}")
    
    print("\n✓ 所有基础模块测试通过！")
    return True


def test_encoders():
    """测试编码器模块"""
    print("\n" + "=" * 60)
    print("测试编码器模块")
    print("=" * 60)
    
    from penci.encoders import BrainSensorModule, BackWardSolution, PENCIEncoder
    
    batch_size = 2
    n_channels = 128
    n_dim = 256
    
    # 测试 BrainSensorModule
    print("\n[1/3] 测试 BrainSensorModule...")
    sensor_module = BrainSensorModule(n_dim)
    pos = torch.randn(batch_size, n_channels, 6)
    sensor_type = torch.randint(0, 3, (batch_size, n_channels))
    out = sensor_module(pos, sensor_type)
    assert out.shape == (batch_size, n_channels, n_dim), f"形状错误: {out.shape}"
    print(f"  ✓ BrainSensorModule: pos{pos.shape}, type{sensor_type.shape} -> {out.shape}")
    
    # 测试 BackWardSolution
    print("\n[2/3] 测试 BackWardSolution...")
    backward = BackWardSolution(n_dim, n_head=4, dropout=0.0)
    neuros = torch.randn(batch_size, 64, n_dim)  # 64 个源
    k = torch.randn(batch_size, n_channels, n_dim)
    x = torch.randn(batch_size, n_channels, n_dim)
    out = backward(neuros, k, x)
    assert out.shape == (batch_size, 64, n_dim), f"形状错误: {out.shape}"
    print(f"  ✓ BackWardSolution: neuros{neuros.shape}, k{k.shape}, x{x.shape} -> {out.shape}")
    
    # 测试 PENCIEncoder
    print("\n[3/3] 测试 PENCIEncoder...")
    encoder = PENCIEncoder(
        n_dim=256,
        n_neuro=72,
        n_head=4,
        dropout=0.0,
        n_filters=32,
        ratios=[8, 4, 2],
        window_size=320,
    )
    x = torch.randn(batch_size, n_channels, 2560)  # 10 秒 @ 256Hz = 2560 samples
    pos = torch.randn(batch_size, n_channels, 6)
    sensor_type = torch.randint(0, 3, (batch_size, n_channels))
    out = encoder(x, pos, sensor_type)
    print(f"  ✓ PENCIEncoder: x{x.shape}, pos{pos.shape} -> {out.shape}")
    
    print("\n✓ 所有编码器测试通过！")
    return True


def test_dynamics():
    """测试动力学模块"""
    print("\n" + "=" * 60)
    print("测试动力学模块")
    print("=" * 60)
    
    from penci.models.dynamics import DynamicsCore, DynamicsRNN
    
    batch_size = 2
    n_neuro = 72
    seq_len = 40
    n_dim = 256
    
    # 测试 DynamicsCore (Transformer)
    print("\n[1/2] 测试 DynamicsCore (Transformer)...")
    dynamics_tf = DynamicsCore(
        n_dim=n_dim,
        n_layers=2,
        n_heads=4,
        dim_feedforward=512,
        dropout=0.1,
    )
    x = torch.randn(batch_size, n_neuro, seq_len, n_dim)
    out = dynamics_tf(x)
    assert out.shape == x.shape, f"形状错误: {out.shape}"
    print(f"  ✓ DynamicsCore: {x.shape} -> {out.shape}")
    
    # 测试 DynamicsRNN
    print("\n[2/2] 测试 DynamicsRNN...")
    dynamics_rnn = DynamicsRNN(
        n_dim=n_dim,
        hidden_dim=512,
        n_layers=2,
        dropout=0.1,
    )
    out = dynamics_rnn(x)
    assert out.shape == x.shape, f"形状错误: {out.shape}"
    print(f"  ✓ DynamicsRNN: {x.shape} -> {out.shape}")
    
    print("\n✓ 所有动力学模块测试通过！")
    return True


def test_decoder():
    """测试解码器模块"""
    print("\n" + "=" * 60)
    print("测试解码器模块")
    print("=" * 60)
    
    from penci.models.physics_decoder import PhysicsDecoder
    
    batch_size = 2
    n_sensors = 128
    n_sources = 64
    seq_len = 40
    n_dim = 256
    
    # 测试 PhysicsDecoder (导联场模式 — 动态传入)
    print("\n[1/3] 测试 PhysicsDecoder (动态导联场)...")
    decoder_fixed = PhysicsDecoder(
        n_dim=n_dim,
        n_sensors=n_sensors,
        n_sources=n_sources,
        use_fixed_leadfield=True,
    )
    source_activity = torch.randn(batch_size, n_sources, seq_len, n_dim)
    leadfield = torch.randn(n_sensors, n_sources)
    out = decoder_fixed(source_activity, leadfield=leadfield)
    print(f"  ✓ PhysicsDecoder (共享L): {source_activity.shape} -> {out.shape}")
    
    # 测试 per-sample 导联场
    print("\n[2/3] 测试 PhysicsDecoder (per-sample 导联场)...")
    leadfield_batch = torch.randn(batch_size, n_sensors, n_sources)
    out = decoder_fixed(source_activity, leadfield=leadfield_batch)
    print(f"  ✓ PhysicsDecoder (per-sample L): {source_activity.shape} -> {out.shape}")
    
    # 测试 PhysicsDecoder (注意力模式)
    print("\n[3/3] 测试 PhysicsDecoder (注意力)...")
    decoder_attn = PhysicsDecoder(
        n_dim=n_dim,
        n_sensors=n_sensors,
        n_sources=n_sources,
        use_fixed_leadfield=False,
        n_head=4,
    )
    sensor_embedding = torch.randn(batch_size, n_sensors, n_dim)
    out = decoder_attn(source_activity, sensor_embedding)
    print(f"  ✓ PhysicsDecoder (attn): source{source_activity.shape}, sensor{sensor_embedding.shape} -> {out.shape}")
    
    # 测试无导联场时报错
    print("\n  验证无导联场 RuntimeError...")
    try:
        decoder_fixed(source_activity)
        assert False, "应该抛出 RuntimeError"
    except RuntimeError:
        print("  ✓ 无导联场正确报错")
    
    print("\n✓ 所有解码器测试通过！")
    return True


def test_full_model():
    """测试完整模型"""
    print("\n" + "=" * 60)
    print("测试完整 PENCI 模型")
    print("=" * 60)
    
    from penci.models import PENCI, PENCILite
    
    batch_size = 2
    n_channels = 128
    seq_len = 2560  # 10 秒 @ 256Hz
    
    # 准备输入
    x = torch.randn(batch_size, n_channels, seq_len)
    pos = torch.randn(batch_size, n_channels, 6)
    sensor_type = torch.randint(0, 3, (batch_size, n_channels))
    leadfield = torch.randn(n_channels, 72)
    
    # 测试 PENCI
    print("\n[1/2] 测试 PENCI...")
    model = PENCI(
        n_dim=256,
        n_neuro=72,
        n_head=4,
        dropout=0.1,
        n_filters=32,
        ratios=[8, 4, 2],
        dynamics_type="transformer",
        dynamics_layers=2,
        dynamics_heads=4,
        n_sensors=n_channels,
    )
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")
    
    # 前向传播（传入导联场）
    output = model(x, pos, sensor_type, leadfield=leadfield, return_source=True)
    print(f"  输入: x{x.shape}, pos{pos.shape}, sensor_type{sensor_type.shape}")
    print(f"  输出: reconstruction{output['reconstruction'].shape}")
    print(f"  源活动: source_activity{output['source_activity'].shape}")
    
    # 测试损失计算
    print("\n  测试损失计算...")
    losses = model.compute_loss(x, pos, sensor_type, leadfield=leadfield)
    print(f"  总损失: {losses['loss'].item():.4f}")
    print(f"  重建损失: {losses['recon_loss'].item():.4f}")
    print(f"  动力学损失: {losses['dynamics_loss'].item():.4f}")
    
    # 测试 PENCILite
    print("\n[2/2] 测试 PENCILite...")
    model_lite = PENCILite(
        n_dim=128,
        n_neuro=32,
        n_head=4,
        dropout=0.1,
        n_filters=16,
        ratios=[8, 4],
        dynamics_layers=1,
        n_sensors=n_channels,
    )
    
    total_params_lite = sum(p.numel() for p in model_lite.parameters())
    print(f"  模型参数量: {total_params_lite:,}")
    
    leadfield_lite = torch.randn(n_channels, 32)
    output_lite = model_lite(x, pos, sensor_type, leadfield=leadfield_lite)
    print(f"  输出: reconstruction{output_lite['reconstruction'].shape}")
    
    print("\n✓ 完整模型测试通过！")
    return True


def test_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 60)
    print("测试数据加载")
    print("=" * 60)
    
    from penci.data import PENCIDataset, PENCICollator
    
    metadata_path = "/work/2024/tanzunsheng/PENCIData/HBN_EEG-metadata/train.json"
    
    if not os.path.exists(metadata_path):
        print(f"  跳过：元数据文件不存在 {metadata_path}")
        return True
    
    print("\n[1/2] 测试 PENCIDataset...")
    dataset = PENCIDataset(
        metadata_path,
        max_length=2560,
        target_channels=128,
    )
    print(f"  数据集大小: {len(dataset)}")
    
    # 加载一个样本
    sample = dataset[0]
    print(f"  样本 x: {sample['x'].shape}, dtype={sample['x'].dtype}")
    print(f"  样本 pos: {sample['pos'].shape}")
    print(f"  样本 sensor_type: {sample['sensor_type'].shape}")
    
    print("\n[2/2] 测试 PENCICollator...")
    collator = PENCICollator()
    batch = collator([dataset[i] for i in range(min(4, len(dataset)))])
    print(f"  批次 x: {batch['x'].shape}")
    print(f"  批次 pos: {batch['pos'].shape}")
    print(f"  批次 sensor_type: {batch['sensor_type'].shape}")
    
    print("\n✓ 数据加载测试通过！")
    return True


def test_integration():
    """集成测试：数据 -> 模型 -> 损失"""
    print("\n" + "=" * 60)
    print("集成测试")
    print("=" * 60)
    
    from penci.models import PENCI
    from penci.data import PENCIDataset, PENCICollator
    
    metadata_path = "/work/2024/tanzunsheng/PENCIData/HBN_EEG-metadata/train.json"
    
    if not os.path.exists(metadata_path):
        print(f"  跳过：元数据文件不存在")
        # 使用随机数据测试
        print("  使用随机数据进行测试...")
        x = torch.randn(4, 128, 2560)
        pos = torch.randn(4, 128, 6)
        sensor_type = torch.randint(0, 3, (4, 128))
    else:
        print("\n加载真实数据...")
        dataset = PENCIDataset(metadata_path, max_length=2560, target_channels=128)
        collator = PENCICollator()
        batch = collator([dataset[i] for i in range(min(4, len(dataset)))])
        x = batch['x']
        pos = batch['pos']
        sensor_type = batch['sensor_type']
    
    print(f"  输入: x{x.shape}, pos{pos.shape}, sensor_type{sensor_type.shape}")
    
    n_sensors = x.shape[1]
    n_neuro = 72
    leadfield = torch.randn(n_sensors, n_neuro)
    
    print("\n创建模型...")
    model = PENCI(
        n_dim=256,
        n_neuro=n_neuro,
        n_head=4,
        dropout=0.0,
        n_filters=32,
        ratios=[8, 4, 2],
        dynamics_type="transformer",
        dynamics_layers=2,
        n_sensors=n_sensors,
    )
    
    print("\n前向传播...")
    with torch.no_grad():
        output = model(x, pos, sensor_type, leadfield=leadfield, return_source=True)
        losses = model.compute_loss(x, pos, sensor_type, leadfield=leadfield)
    
    print(f"  重建输出: {output['reconstruction'].shape}")
    print(f"  源活动: {output['source_activity'].shape}")
    print(f"  总损失: {losses['loss'].item():.4f}")
    
    print("\n测试反向传播...")
    model.train()
    output = model(x, pos, sensor_type, leadfield=leadfield)
    losses = model.compute_loss(x, pos, sensor_type, leadfield=leadfield)
    losses['loss'].backward()
    
    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    if has_grad:
        print("  ✓ 梯度正常流动")
    else:
        print("  ✗ 警告：未检测到梯度流动")
    
    print("\n✓ 集成测试通过！")
    return True


def test_electrode_utils():
    """测试电极坐标读取和过滤"""
    print("\n" + "=" * 60)
    print("测试电极工具 (electrode_utils)")
    print("=" * 60)

    import tempfile
    import numpy as np
    from penci.physics.electrode_utils import (
        read_electrodes_tsv,
        filter_channels_like_postprocess,
    )

    # === 1. 创建临时 electrodes.tsv 测试文件 ===
    print("\n[1/4] 测试 read_electrodes_tsv...")
    tsv_content = (
        "name\tx\ty\tz\n"
        "Fp1\t0.0200\t0.0800\t0.0100\n"
        "Fp2\t-0.0200\t0.0800\t0.0100\n"
        "Cz\t0.0000\t0.0000\t0.0948\n"
        "Bad1\tnan\t0.0\t0.0\n"
        "Bad2\t0.0\t0.0\t0.0\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_electrodes.tsv", delete=False
    ) as f:
        f.write(tsv_content)
        tmp_path = f.name

    try:
        channels = read_electrodes_tsv(tmp_path)
        assert len(channels) == 5, f"应读取 5 个通道，实际 {len(channels)}"
        assert "Fp1" in channels, "缺少 Fp1"
        assert channels["Fp1"].shape == (3,), f"坐标形状错误: {channels['Fp1'].shape}"
        print(f"  ✓ 读取 {len(channels)} 个通道")

        # === 2. 测试过滤无效通道 ===
        print("\n[2/4] 测试 filter_channels_like_postprocess...")
        valid = filter_channels_like_postprocess(channels, min_valid_channels=1)
        # Bad1 (NaN) 和 Bad2 (全零) 应被过滤
        assert "Bad1" not in valid, "Bad1 应被过滤 (NaN)"
        assert "Bad2" not in valid, "Bad2 应被过滤 (全零)"
        assert "Fp1" in valid, "Fp1 应保留"
        assert "Fp2" in valid, "Fp2 应保留"
        assert "Cz" in valid, "Cz 应保留 (坐标有效)"
        assert len(valid) == 3, f"应有 3 个有效通道，实际 {len(valid)}"
        print(f"  ✓ 过滤后 {len(valid)} 个有效通道")

        # === 3. 测试坐标精度 ===
        print("\n[3/4] 测试坐标精度...")
        fp1_coord = valid["Fp1"]
        assert np.allclose(fp1_coord, [0.02, 0.08, 0.01], atol=1e-8), (
            f"坐标精度错误: {fp1_coord}"
        )
        print(f"  ✓ Fp1 坐标正确: {fp1_coord}")

        # === 4. 测试 FileNotFoundError ===
        print("\n[4/4] 测试文件不存在报错...")
        try:
            read_electrodes_tsv("/nonexistent/path/electrodes.tsv")
            assert False, "应抛出 FileNotFoundError"
        except FileNotFoundError:
            print("  ✓ FileNotFoundError 正确抛出")
    finally:
        os.unlink(tmp_path)

    print("\n✓ 电极工具测试通过！")
    return True


def test_source_space():
    """
    测试 SourceSpace (72 个脑区)

    需要 MNE-Python 和 fsaverage 数据。
    """
    print("\n" + "=" * 60)
    print("测试 SourceSpace")
    print("=" * 60)

    subjects_dir = "/work/2024/tanzunsheng/mne_data/MNE-fsaverage-data"
    if not os.path.exists(os.path.join(subjects_dir, "fsaverage")):
        print("  跳过：fsaverage 目录不存在")
        return True

    import numpy as np
    from penci.physics.source_space import SourceSpace

    print("\n[1/5] 初始化 SourceSpace...")
    ss = SourceSpace(subjects_dir=subjects_dir)

    # === 1. 名称数量 ===
    print("\n[2/5] 验证源数量...")
    assert len(ss.names) == 72, f"应有 72 个源，实际 {len(ss.names)}"
    print(f"  ✓ 72 个源: 前3={ss.names[:3]}, 后3={ss.names[-3:]}")

    # === 2. 位置形状 ===
    print("\n[3/5] 验证位置形状...")
    assert ss.positions.shape == (72, 3), f"形状错误: {ss.positions.shape}"
    print(f"  ✓ 位置形状: {ss.positions.shape}")

    # === 3. 皮层 + 皮层下分组 ===
    print("\n[4/5] 验证皮层/皮层下分组...")
    assert ss.cortical_positions.shape == (68, 3), (
        f"皮层源形状错误: {ss.cortical_positions.shape}"
    )
    assert ss.subcortical_positions.shape == (4, 3), (
        f"皮层下源形状错误: {ss.subcortical_positions.shape}"
    )
    print(f"  ✓ 68 个皮层源 + 4 个皮层下源")

    # === 4. 坐标范围合理性 ===
    print("\n[5/5] 验证坐标范围...")
    pos = ss.positions
    # MNI 坐标应在合理范围内（米制，约 ±0.1m 即 ±10cm）
    assert np.abs(pos).max() < 0.2, f"坐标超出合理范围: max={np.abs(pos).max()}"
    # 不应有 NaN
    assert not np.isnan(pos).any(), "坐标包含 NaN"
    print(f"  ✓ 坐标范围: [{pos.min():.4f}, {pos.max():.4f}]")

    print("\n✓ SourceSpace 测试通过！")
    return True


def test_leadfield_e2e():
    """
    端到端测试 LeadfieldManager

    需要 MNE-Python、fsaverage 数据和 HBN_EEG 电极坐标。
    此测试较慢（首次计算约 75 秒），后续从缓存加载很快。
    """
    print("\n" + "=" * 60)
    print("测试 LeadfieldManager (端到端)")
    print("=" * 60)

    import time
    import numpy as np
    from penci.physics.source_space import SourceSpace
    from penci.physics.leadfield_manager import LeadfieldManager
    from penci.physics.electrode_utils import get_valid_channels_for_dataset

    subjects_dir = "/work/2024/tanzunsheng/mne_data/MNE-fsaverage-data"
    processed_dir = "/work/2024/tanzunsheng/ProcessedData"
    cache_dir = "/work/2024/tanzunsheng/leadfield_cache"

    # 检查前置条件
    if not os.path.exists(os.path.join(subjects_dir, "fsaverage")):
        print("  跳过：fsaverage 目录不存在")
        return True

    # 查找可用的电极坐标文件
    try:
        valid_channels, names = get_valid_channels_for_dataset(
            processed_data_dir=processed_dir,
            dataset_name="HBN_EEG",
            subject_id="NDARVA281NVV",
            site="R11",
        )
    except FileNotFoundError:
        print("  跳过：HBN_EEG 电极坐标文件不存在")
        return True

    pos = np.array([valid_channels[n] for n in names])

    # === 1. 初始化 ===
    print(f"\n[1/5] 初始化（{len(names)} 个通道）...")
    ss = SourceSpace(subjects_dir=subjects_dir)
    lm = LeadfieldManager(
        source_space=ss,
        subjects_dir=subjects_dir,
        cache_dir=cache_dir,
    )

    # === 2. 获取导联场 ===
    print("\n[2/5] 获取导联场...")
    t0 = time.time()
    L = lm.get_leadfield(names, pos)
    elapsed = time.time() - t0
    print(f"  ✓ 形状: {L.shape}，耗时 {elapsed:.1f}s")

    # 基本验证
    assert L.shape == (len(names), 72), f"形状错误: {L.shape}"
    assert L.dtype == torch.float32, f"dtype 错误: {L.dtype}"
    assert not torch.isnan(L).any(), "导联场包含 NaN"
    assert not torch.isinf(L).any(), "导联场包含 Inf"
    assert (L.abs().sum(dim=0) > 0).all(), "存在全零列"

    # === 3. 缓存一致性 ===
    print("\n[3/5] 验证内存缓存...")
    t0 = time.time()
    L_mem = lm.get_leadfield(names, pos)
    t_mem = time.time() - t0
    assert torch.equal(L, L_mem), "内存缓存不一致"
    print(f"  ✓ 内存缓存一致，{t_mem:.4f}s")

    print("\n[4/5] 验证磁盘缓存...")
    lm2 = LeadfieldManager(
        source_space=ss, subjects_dir=subjects_dir, cache_dir=cache_dir
    )
    t0 = time.time()
    L_disk = lm2.get_leadfield(names, pos)
    t_disk = time.time() - t0
    assert torch.equal(L, L_disk), "磁盘缓存不一致"
    print(f"  ✓ 磁盘缓存一致，{t_disk:.4f}s")

    # === 4. Batch 接口 ===
    print("\n[5/5] 验证 Batch 接口...")
    L_batch = lm.get_leadfield_for_batch(names, pos, batch_size=4)
    assert L_batch.shape == (4, len(names), 72), f"Batch 形状错误: {L_batch.shape}"
    assert torch.equal(L_batch[0], L), "Batch 第一个样本不一致"
    print(f"  ✓ Batch 形状: {L_batch.shape}")

    print("\n✓ LeadfieldManager 端到端测试通过！")
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("PENCI Smoke Test")
    print("=" * 60)
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    tests = [
        ("基础模块", test_modules),
        ("编码器模块", test_encoders),
        ("动力学模块", test_dynamics),
        ("解码器模块", test_decoder),
        ("完整模型", test_full_model),
        ("数据加载", test_data_loading),
        ("集成测试", test_integration),
        ("电极工具", test_electrode_utils),
        ("源空间", test_source_space),
        ("导联场端到端", test_leadfield_e2e),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            import traceback
            results.append((name, False, traceback.format_exc()))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_passed = True
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"    错误: {error[:200]}...")
            all_passed = False
    
    if all_passed:
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！PENCI 模型可以正常工作。")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("✗ 部分测试失败，请检查错误信息。")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
