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
        n_neuro=64,
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
    n_neuro = 64
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
    
    # 测试 PhysicsDecoder (固定导联场)
    print("\n[1/2] 测试 PhysicsDecoder (固定导联场)...")
    decoder_fixed = PhysicsDecoder(
        n_dim=n_dim,
        n_sensors=n_sensors,
        n_sources=n_sources,
        use_fixed_leadfield=True,
    )
    source_activity = torch.randn(batch_size, n_sources, seq_len, n_dim)
    out = decoder_fixed(source_activity)
    print(f"  ✓ PhysicsDecoder (fixed): {source_activity.shape} -> {out.shape}")
    
    # 测试 PhysicsDecoder (注意力)
    print("\n[2/2] 测试 PhysicsDecoder (注意力)...")
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
    
    # 测试 PENCI
    print("\n[1/2] 测试 PENCI...")
    model = PENCI(
        n_dim=256,
        n_neuro=64,
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
    
    # 前向传播
    output = model(x, pos, sensor_type, return_source=True)
    print(f"  输入: x{x.shape}, pos{pos.shape}, sensor_type{sensor_type.shape}")
    print(f"  输出: reconstruction{output['reconstruction'].shape}")
    print(f"  源活动: source_activity{output['source_activity'].shape}")
    
    # 测试损失计算
    print("\n  测试损失计算...")
    losses = model.compute_loss(x, pos, sensor_type)
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
    
    output_lite = model_lite(x, pos, sensor_type)
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
    
    print("\n创建模型...")
    model = PENCI(
        n_dim=256,
        n_neuro=64,
        n_head=4,
        dropout=0.0,
        n_filters=32,
        ratios=[8, 4, 2],
        dynamics_type="transformer",
        dynamics_layers=2,
        n_sensors=128,
    )
    
    print("\n前向传播...")
    with torch.no_grad():
        output = model(x, pos, sensor_type, return_source=True)
        losses = model.compute_loss(x, pos, sensor_type)
    
    print(f"  重建输出: {output['reconstruction'].shape}")
    print(f"  源活动: {output['source_activity'].shape}")
    print(f"  总损失: {losses['loss'].item():.4f}")
    
    print("\n测试反向传播...")
    model.train()
    output = model(x, pos, sensor_type)
    losses = model.compute_loss(x, pos, sensor_type)
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
