#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练日志深度分析脚本

分析 train_*.log 文件，输出：
  1. 吞吐量与 GPU 利用率估算
  2. 每步耗时分布（检测数据加载瓶颈）
  3. Loss 曲线异常检测
  4. 综合诊断结论与优化建议

用法:
    python tests/analyze_training_log.py
    python tests/analyze_training_log.py --log outputs/train_xxx/train_xxx.log
    python tests/analyze_training_log.py --log outputs/train_xxx/train_xxx.log --batch_size 32 --n_gpus 8
"""

import sys
import os
import re
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ─── 日志解析 ───────────────────────────────────────────────────────────────

STEP_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"Epoch (\d+) \[(\d+)/(\d+)\] "
    r"Loss: ([\d.e+\-]+) \(avg: ([\d.e+\-]+)\) "
    r"Recon: ([\d.e+\-]+) Dynamics: ([\d.e+\-]+)"
)


def parse_log(log_path: str):
    """解析训练日志，返回每步的记录列表"""
    records = []
    with open(log_path, "r") as f:
        for line in f:
            m = STEP_PATTERN.search(line)
            if m:
                ts_str, epoch, step, total, loss, avg_loss, recon, dynamics = m.groups()
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
                records.append({
                    "ts": ts,
                    "epoch": int(epoch),
                    "step": int(step),
                    "total": int(total),
                    "loss": float(loss),
                    "avg_loss": float(avg_loss),
                    "recon": float(recon),
                    "dynamics": float(dynamics),
                })
    return records


# ─── 分析函数 ────────────────────────────────────────────────────────────────

def analyze_throughput(records, batch_size_per_gpu: int, n_gpus: int):
    """分析训练吞吐量与 GPU 利用率"""
    if len(records) < 2:
        print("[吞吐量] 数据不足，跳过")
        return {}

    # 计算每步耗时（秒）
    step_times = []
    for i in range(1, len(records)):
        dt = (records[i]["ts"] - records[i - 1]["ts"]).total_seconds()
        step_times.append(dt)

    # 过滤明显异常值（超过平均值5倍的点，多为 epoch 切换或 eval 暂停）
    median_dt = sorted(step_times)[len(step_times) // 2]
    normal_times = [t for t in step_times if t < median_dt * 10]

    if not normal_times:
        print("[吞吐量] 无有效数据")
        return {}

    avg_step_time = sum(normal_times) / len(normal_times)
    min_step_time = min(normal_times)
    max_step_time = max(normal_times)

    # 总时间与总样本
    total_seconds = (records[-1]["ts"] - records[0]["ts"]).total_seconds()
    total_steps = records[-1]["step"] - records[0]["step"]
    samples_per_step = batch_size_per_gpu * n_gpus
    total_samples = total_steps * samples_per_step

    # 吞吐量
    global_throughput = total_samples / max(total_seconds, 1)
    per_gpu_throughput = global_throughput / n_gpus
    step_throughput = samples_per_step / avg_step_time

    print("\n" + "=" * 60)
    print("【1. 训练吞吐量分析】")
    print("=" * 60)
    print(f"  总步数           : {total_steps}")
    print(f"  总训练时长       : {total_seconds:.1f} 秒 ({total_seconds/60:.1f} 分钟)")
    print(f"  每张卡 batch_size: {batch_size_per_gpu}")
    print(f"  GPU 数量         : {n_gpus}")
    print(f"  每步全局样本数   : {samples_per_step}")
    print(f"")
    print(f"  平均每步耗时     : {avg_step_time*1000:.1f} ms")
    print(f"  最快步耗时       : {min_step_time*1000:.1f} ms")
    print(f"  最慢步耗时       : {max_step_time*1000:.1f} ms")
    print(f"")
    print(f"  全局吞吐量       : {global_throughput:.1f} samples/s")
    print(f"  单卡吞吐量       : {per_gpu_throughput:.1f} samples/s")

    # GPU 利用率估算（对比 4090D 理论峰值）
    # RTX 4090D: ~82.6 TFLOPS FP16, 24 GB VRAM
    # 经验值：简单模型在 4090D 上 FP16 约 1000-5000 samples/s（取决于模型大小）
    # 我们用"每步计算时间 / 每步总时间"来估算实际利用率
    # 实际计算时间 ≈ min_step_time（理论下界，数据一定准备好时的纯计算时间）
    compute_ratio = min_step_time / avg_step_time if avg_step_time > 0 else 0

    print(f"")
    print(f"  GPU 计算时间占比 : {compute_ratio*100:.1f}%")
    print(f"  （= 最快步耗时 / 平均步耗时，越接近100%说明 GPU 一直在算，不等数据）")

    if per_gpu_throughput < 50:
        print(f"\n  ⚠️  警告: 单卡吞吐量 {per_gpu_throughput:.1f} samples/s 极低!")
        print(f"      对于 4090D 这类高端 GPU，正常情况下应在 200-1000+ samples/s")
        if batch_size_per_gpu <= 4:
            print(f"      当前 batch_size={batch_size_per_gpu} 过小，GPU 计算完就在等数据")
    elif per_gpu_throughput < 200:
        print(f"\n  ⚠️  警告: 单卡吞吐量 {per_gpu_throughput:.1f} samples/s 偏低")
    else:
        print(f"\n  ✅ 吞吐量正常")

    return {
        "avg_step_time": avg_step_time,
        "step_times": step_times,
        "global_throughput": global_throughput,
        "per_gpu_throughput": per_gpu_throughput,
        "compute_ratio": compute_ratio,
    }


def analyze_step_time_distribution(step_times):
    """分析每步耗时分布，找出数据加载瓶颈"""
    if not step_times:
        return

    print("\n" + "=" * 60)
    print("【2. 每步耗时分布（数据加载瓶颈检测）】")
    print("=" * 60)

    sorted_times = sorted(step_times)
    n = len(sorted_times)
    median = sorted_times[n // 2]
    p95 = sorted_times[int(n * 0.95)]
    p99 = sorted_times[int(n * 0.99)]
    p_max = sorted_times[-1]

    print(f"  中位数 (P50) : {median*1000:.1f} ms")
    print(f"  P95          : {p95*1000:.1f} ms")
    print(f"  P99          : {p99*1000:.1f} ms")
    print(f"  最大值       : {p_max*1000:.1f} ms")

    # 统计慢步（超过中位数 3 倍）
    slow_threshold = median * 3
    slow_steps = [t for t in step_times if t > slow_threshold]
    slow_ratio = len(slow_steps) / len(step_times) * 100

    print(f"")
    print(f"  慢步定义（> 中位数×3 = {slow_threshold*1000:.0f} ms）: {len(slow_steps)} 步 ({slow_ratio:.1f}%)")

    if slow_ratio > 5:
        print(f"  ❌ 严重: {slow_ratio:.1f}% 的步骤出现明显卡顿，数据加载是主要瓶颈")
        print(f"     建议: 减小 num_workers（防止 NFS 过载），或增大 batch_size 摊薄加载代价")
    elif slow_ratio > 1:
        print(f"  ⚠️  轻微: {slow_ratio:.1f}% 的步骤出现卡顿")
    else:
        print(f"  ✅ 耗时分布均匀，无明显数据加载瓶颈")

    # 耗时分布直方图（文字版）
    print(f"\n  耗时直方图 (每格 = 总步数的 1/20):")
    bins = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, float("inf")]
    labels = ["<50ms", "50-100ms", "100-150ms", "150-200ms", "200-300ms",
              "300-500ms", "500ms-1s", "1-2s", "2-5s", ">5s"]
    counts = [0] * len(labels)
    for t in step_times:
        for i, b in enumerate(bins[1:]):
            if t < b:
                counts[i] += 1
                break

    bar_max = max(counts) if counts else 1
    bar_width = 30
    for label, count in zip(labels, counts):
        bar_len = int(count / bar_max * bar_width)
        bar = "█" * bar_len
        print(f"  {label:>12s} | {bar:<{bar_width}} {count}")


def analyze_loss(records):
    """分析 loss 曲线，检测异常"""
    if not records:
        return

    print("\n" + "=" * 60)
    print("【3. Loss 曲线分析】")
    print("=" * 60)

    losses = [r["loss"] for r in records]
    steps = [r["step"] for r in records]

    first_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)
    max_loss = max(losses)

    print(f"  初始 loss       : {first_loss:.4f}")
    print(f"  最终 loss       : {final_loss:.4f}")
    print(f"  最小 loss       : {min_loss:.4f}  (step {steps[losses.index(min_loss)]})")
    print(f"  最大 loss       : {max_loss:.4f}  (step {steps[losses.index(max_loss)]})")

    if final_loss < first_loss:
        drop_ratio = (first_loss - final_loss) / first_loss * 100
        print(f"  Loss 下降幅度   : {drop_ratio:.1f}%")
    else:
        print(f"  ⚠️  Loss 未下降（{first_loss:.4f} → {final_loss:.4f}）")

    # 检测 loss 爆炸事件（突然暴增超过前一步的 10 倍）
    explosions = []
    for i in range(1, len(losses)):
        if losses[i - 1] > 0 and losses[i] > losses[i - 1] * 10 and losses[i] > 100:
            explosions.append((steps[i], losses[i - 1], losses[i]))

    if explosions:
        print(f"\n  ❌ 检测到 {len(explosions)} 次 Loss 爆炸事件:")
        for step, prev, curr in explosions[:5]:
            print(f"     step {step:5d}: {prev:.2f} → {curr:.2f}  (×{curr/prev:.0f})")
        if len(explosions) > 5:
            print(f"     ... 共 {len(explosions)} 次")
        print(f"\n  Loss 爆炸可能原因:")
        print(f"   a) AMP 混合精度下梯度溢出（GradScaler 会自动跳过该步并缩小 scale）")
        print(f"   b) 某些桶（不同通道数）的数据幅值差异很大，缺乏归一化")
        print(f"   c) 学习率过大（当前设置了 LR×world_size 线性缩放，可能偏大）")
    else:
        print(f"\n  ✅ 无 Loss 爆炸事件")

    # 检测 loss 是否收敛
    if len(losses) >= 100:
        last_100_avg = sum(losses[-100:]) / 100
        first_100_avg = sum(losses[:100]) / 100
        if last_100_avg < first_100_avg * 0.1:
            print(f"\n  ✅ Loss 收敛良好 (后100步均值 {last_100_avg:.4f} vs 前100步 {first_100_avg:.4f})")
        elif last_100_avg < first_100_avg * 0.5:
            print(f"\n  ✅ Loss 持续下降中 (后100步均值 {last_100_avg:.4f})")
        else:
            print(f"\n  ⚠️  Loss 下降缓慢 (后100步均值 {last_100_avg:.4f} vs 前100步 {first_100_avg:.4f})")

    # Dynamics loss 分析
    dynamics = [r["dynamics"] for r in records]
    avg_dynamics = sum(dynamics) / len(dynamics)
    print(f"\n  Dynamics loss 均值: {avg_dynamics:.2e}")
    if avg_dynamics < 1e-5:
        print(f"  ⚠️  Dynamics loss 极小（{avg_dynamics:.2e}），对总损失几乎无影响")
        print(f"      建议适当增大 dynamics 权重（当前 0.1），或检查动力学模型输出")


def analyze_bucket_switching(records, step_times):
    """通过耗时突变检测 bucket 切换（数据集切换导致的一次性延迟）"""
    if len(step_times) < 10:
        return

    print("\n" + "=" * 60)
    print("【4. 数据集 Bucket 切换检测】")
    print("=" * 60)

    median = sorted(step_times)[len(step_times) // 2]
    spike_threshold = median * 5
    spikes = [(records[i + 1]["step"], step_times[i])
              for i in range(len(step_times))
              if step_times[i] > spike_threshold]

    if spikes:
        print(f"  检测到 {len(spikes)} 次显著耗时突刺（>{spike_threshold*1000:.0f}ms）:")
        for step, t in spikes[:10]:
            print(f"     step {step:5d}: {t*1000:.0f} ms")
        if len(spikes) > 10:
            print(f"     ... 共 {len(spikes)} 次")
        print(f"\n  这些突刺通常来自 BucketBatchSampler 切换桶时需要重新加载不同电极配置的数据")
        print(f"  建议: 适当增大 num_workers 的 prefetch_factor，或减少桶切换频率")
    else:
        print(f"  ✅ 无明显 Bucket 切换延迟")


def print_summary_recommendations(throughput_info, batch_size_per_gpu, n_gpus):
    """综合诊断与优化建议"""
    print("\n" + "=" * 60)
    print("【5. 综合诊断结论与优化建议】")
    print("=" * 60)

    issues = []
    suggestions = []

    per_gpu = throughput_info.get("per_gpu_throughput", 0)
    compute_ratio = throughput_info.get("compute_ratio", 1.0)

    if batch_size_per_gpu <= 4:
        issues.append(f"batch_size={batch_size_per_gpu} 严重过小")
        suggestions.append(
            f"将 batch_size 从 {batch_size_per_gpu} 增大到 32（default.yaml），"
            f"GPU 才能充分并行计算，预计吞吐量提升 {32//batch_size_per_gpu}x"
        )

    if compute_ratio < 0.5:
        issues.append(f"GPU 计算时间占比仅 {compute_ratio*100:.0f}%，大量时间在等数据")
        suggestions.append(
            "数据加载是主要瓶颈。建议: "
            "(1) 检查 num_workers 是否合适（NFS 下 2-4 个即可）; "
            "(2) 如有本地 SSD 考虑将热数据 rsync 到本地"
        )

    if per_gpu < 50 and batch_size_per_gpu > 4:
        issues.append(f"单卡吞吐量偏低 ({per_gpu:.0f} samples/s)")
        suggestions.append(
            "检查模型是否有不必要的 CPU-GPU 同步（如频繁的 .item() 调用），"
            "考虑开启 torch.compile"
        )

    if not issues:
        print("  ✅ 未发现明显性能问题")
    else:
        print(f"  发现 {len(issues)} 个问题:\n")
        for i, issue in enumerate(issues, 1):
            print(f"  [{i}] ❌ {issue}")
        print(f"\n  优化建议:\n")
        for i, s in enumerate(suggestions, 1):
            print(f"  [{i}] 💡 {s}")

    print(f"\n  当前配置总结:")
    print(f"       GPU 数量: {n_gpus}")
    print(f"       batch_size/GPU: {batch_size_per_gpu}")
    print(f"       全局 batch_size: {batch_size_per_gpu * n_gpus}")
    if batch_size_per_gpu <= 4:
        print(f"\n  ⭐ 最重要: 当前更像是轻量验证配置，建议切到 V1 正式训练入口。")
        print(f"     正式训练请使用 configs/stage1_real.yaml（当前 batch_size=8 / num_workers=1），")
        print(f"     命令: torchrun --nproc_per_node={n_gpus} scripts/v1/train_stage1.py --config configs/stage1_real.yaml")


# ─── 主入口 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PENCI 训练日志分析工具")
    parser.add_argument(
        "--log", type=str, default=None,
        help="训练日志路径（默认自动找最新的 outputs/train_*/train_*.log）"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="每张卡的 batch_size（默认自动推断：smoke_test=4, default=32）"
    )
    parser.add_argument(
        "--n_gpus", type=int, default=8,
        help="GPU 数量（默认 8）"
    )
    args = parser.parse_args()

    # 自动寻找最新日志
    if args.log is None:
        outputs_dir = project_root / "outputs"
        log_files = sorted(outputs_dir.glob("train_*/train_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            print("未找到训练日志，请用 --log 指定路径")
            sys.exit(1)
        log_path = str(log_files[0])
        print(f"自动选择最新日志: {log_path}")
    else:
        log_path = args.log

    # 解析日志
    records = parse_log(log_path)
    if not records:
        print(f"日志中未找到训练步骤记录，请检查路径: {log_path}")
        sys.exit(1)

    print(f"\n共解析到 {len(records)} 条训练步骤记录")
    print(f"训练时间范围: {records[0]['ts'].strftime('%H:%M:%S')} → {records[-1]['ts'].strftime('%H:%M:%S')}")

    # 自动推断 batch_size
    if args.batch_size is None:
        # 从日志路径名猜测
        if "smoke" in log_path.lower():
            batch_size = 4
            print(f"检测到 smoke_test 配置，使用 batch_size=4")
        else:
            batch_size = 32
            print(f"使用默认 batch_size=32")
    else:
        batch_size = args.batch_size

    # 分析
    throughput_info = analyze_throughput(records, batch_size, args.n_gpus)

    step_times = throughput_info.get("step_times", [])
    if step_times:
        analyze_step_time_distribution(step_times)
        analyze_bucket_switching(records, step_times)

    analyze_loss(records)
    print_summary_recommendations(throughput_info, batch_size, args.n_gpus)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
