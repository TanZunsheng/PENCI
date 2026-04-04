#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练期间系统资源诊断脚本

在训练运行时，另开一个终端执行本脚本，输出：
  1. GPU 状态（显存占用、利用率、温度、功耗）
  2. CPU / 内存 / Swap 使用情况
  3. NFS I/O 压力（调用速率、慢操作）
  4. 进程状态（D 状态进程、高 CPU 进程）
  5. 综合诊断结论

用法:
    python tests/diagnose_system.py            # 单次快照
    python tests/diagnose_system.py --watch    # 每 5 秒刷新一次
    python tests/diagnose_system.py --watch --interval 10
"""

import sys
import os
import re
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ─── 工具函数 ────────────────────────────────────────────────────────────────

def run_cmd(cmd: str, timeout: int = 10) -> str:
    """执行 shell 命令，返回 stdout，失败返回空字符串"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except Exception as e:
        return f"[命令执行失败: {e}]"


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"【{title}】")
    print("=" * 60)


# ─── GPU 诊断 ────────────────────────────────────────────────────────────────

def diagnose_gpu():
    """查询所有 GPU 的显存、利用率、温度、功耗"""
    section("1. GPU 状态")

    query = (
        "index,name,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,temperature.gpu,power.draw,power.limit"
    )
    out = run_cmd(f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits")

    if not out or "命令执行失败" in out:
        print("  ❌ nvidia-smi 不可用")
        return

    gpus = []
    for line in out.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        try:
            gpu = {
                "idx": int(parts[0]),
                "name": parts[1],
                "util_gpu": float(parts[2]),
                "util_mem": float(parts[3]),
                "mem_used": float(parts[4]),
                "mem_total": float(parts[5]),
                "temp": float(parts[6]),
                "power": float(parts[7]) if parts[7] != "N/A" else 0,
                "power_limit": float(parts[8]) if parts[8] != "N/A" else 0,
            }
            gpus.append(gpu)
        except ValueError:
            continue

    if not gpus:
        print("  无法解析 GPU 信息")
        return

    print(f"  {'GPU':<5} {'型号':<20} {'利用率':>8} {'显存使用':>12} {'显存占比':>8} {'温度':>6} {'功耗':>10}")
    print(f"  {'-'*5} {'-'*20} {'-'*8} {'-'*12} {'-'*8} {'-'*6} {'-'*10}")

    low_util_gpus = []
    for g in gpus:
        mem_ratio = g["mem_used"] / g["mem_total"] * 100 if g["mem_total"] > 0 else 0
        power_str = f"{g['power']:.0f}W/{g['power_limit']:.0f}W" if g["power_limit"] > 0 else f"{g['power']:.0f}W"
        print(
            f"  GPU{g['idx']:<2} {g['name']:<20} {g['util_gpu']:>7.1f}% "
            f"  {g['mem_used']:>6.0f}/{g['mem_total']:.0f}MB {mem_ratio:>7.1f}% "
            f"  {g['temp']:>5.0f}°C {power_str:>10}"
        )
        if g["util_gpu"] < 30:
            low_util_gpus.append(g)

    # 统计
    avg_util = sum(g["util_gpu"] for g in gpus) / len(gpus)
    avg_mem_ratio = sum(g["mem_used"] / g["mem_total"] for g in gpus) / len(gpus) * 100
    total_mem_used = sum(g["mem_used"] for g in gpus) / 1024

    print(f"\n  平均 GPU 利用率  : {avg_util:.1f}%")
    print(f"  平均显存占用率   : {avg_mem_ratio:.1f}%")
    print(f"  总显存使用       : {total_mem_used:.1f} GB")

    # 诊断
    if low_util_gpus:
        print(f"\n  ⚠️  {len(low_util_gpus)} 张 GPU 利用率低于 30%:")
        for g in low_util_gpus:
            print(f"     GPU{g['idx']}: 利用率 {g['util_gpu']:.1f}%, 显存 {g['mem_used']:.0f}/{g['mem_total']:.0f}MB")
        print(f"\n  低 GPU 利用率的常见原因:")
        print(f"   a) batch_size 太小 → GPU 算完立即空转等下一个 batch")
        print(f"   b) 数据加载跟不上（NFS IO 瓶颈）→ GPU 等数据")
        print(f"   c) 模型太小 → 计算量少，GPU 很快算完")

    if avg_mem_ratio < 30:
        print(f"\n  ⚠️  显存利用率仅 {avg_mem_ratio:.1f}%，说明 batch_size 可以更大")
        est_max_bs = int(32 * (80 / avg_mem_ratio)) if avg_mem_ratio > 0 else 32
        print(f"     当前显存使用量仅 {avg_mem_ratio:.1f}%，建议将 batch_size 增大到 {min(est_max_bs, 128)}")


# ─── CPU / 内存诊断 ──────────────────────────────────────────────────────────

def diagnose_cpu_memory():
    """查询 CPU 使用率、内存、Swap、进程数"""
    section("2. CPU / 内存状态")

    # vmstat 1次 2行（取第二行，跳过header和第一行预热数据）
    vmstat_out = run_cmd("vmstat 1 2 | tail -1")
    if vmstat_out and "命令执行失败" not in vmstat_out:
        parts = vmstat_out.split()
        if len(parts) >= 17:
            r_queue = int(parts[0])
            b_blocked = int(parts[1])
            swpd = int(parts[2])
            free_kb = int(parts[3])
            cache_kb = int(parts[5])
            cs = int(parts[11])
            us = int(parts[12])
            sy = int(parts[13])
            id_ = int(parts[14])
            wa = int(parts[15])

            print(f"  CPU 使用率    : us={us}% sy={sy}% id={id_}% wa={wa}%")
            print(f"  运行队列      : {r_queue} 个进程等待 CPU")
            print(f"  阻塞进程数    : {b_blocked} 个（IO 等待）")
            print(f"  上下文切换    : {cs} 次/秒")
            print(f"  可用内存      : {free_kb/1024:.0f} MB  |  Page Cache: {cache_kb/1024/1024:.0f} GB")
            print(f"  Swap 使用     : {swpd/1024:.0f} MB")

            if wa > 20:
                print(f"\n  ❌ IO 等待高 ({wa}%)，磁盘/NFS IO 是严重瓶颈")
            elif wa > 5:
                print(f"\n  ⚠️  IO 等待偏高 ({wa}%)")
            else:
                print(f"\n  ✅ IO 等待正常 ({wa}%)")

            if cs > 50000:
                print(f"  ❌ 上下文切换极高 ({cs}/s)，进程数量过多")
            elif cs > 20000:
                print(f"  ⚠️  上下文切换偏高 ({cs}/s)，建议减少 num_workers")

    # 内存详情
    free_out = run_cmd("free -h | grep Mem")
    if free_out and "命令执行失败" not in free_out:
        print(f"\n  内存详情: {free_out}")

    # D 状态进程
    d_state = run_cmd("ps aux | awk '$8 ~ /D/ {print $0}' | wc -l")
    if d_state.isdigit():
        n_d = int(d_state)
        print(f"  D 状态进程数  : {n_d} 个（不可中断睡眠，通常是 NFS 等待）")
        if n_d > 20:
            print(f"  ❌ D 状态进程过多，NFS 服务器响应迟缓")
        elif n_d > 5:
            print(f"  ⚠️  存在 D 状态进程，NFS 有一定压力")

    # Python 训练进程数
    py_procs = run_cmd("pgrep -c python || echo 0")
    print(f"  Python 进程数  : {py_procs.strip()} 个")


# ─── NFS 诊断 ────────────────────────────────────────────────────────────────

def diagnose_nfs():
    """分析 NFS 客户端状态"""
    section("3. NFS I/O 状态")

    # 获取 nfsstat 一次性输出
    out = run_cmd("nfsstat -c 2>/dev/null")
    if not out or "命令执行失败" in out or "No RPC" in out:
        print("  nfsstat 不可用，跳过 NFS 诊断")
        return

    # 解析 NFSv4 关键操作
    def extract_nfs_op(text, op_name):
        pattern = rf"{op_name}\s+(\d+)\s+([\d]+)%"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return int(m.group(1)), int(m.group(2))
        return 0, 0

    total_calls_m = re.search(r"calls\s+retrans\s+authrefrsh\s*\n\s*(\d+)", out)
    total_calls = int(total_calls_m.group(1)) if total_calls_m else 0

    read_cnt, read_pct = extract_nfs_op(out, "read")
    write_cnt, write_pct = extract_nfs_op(out, "write")
    open_cnt, open_pct = extract_nfs_op(out, "open_noat")
    close_cnt, close_pct = extract_nfs_op(out, "close")
    getattr_cnt, getattr_pct = extract_nfs_op(out, "getattr")
    delegreturn_cnt, delegreturn_pct = extract_nfs_op(out, "delegreturn")

    print(f"  累计 NFS 调用总数 : {total_calls:,}")
    print(f"")
    print(f"  操作分布 (累计):")
    print(f"    {'read':>12} : {read_cnt:>12,}  ({read_pct:>2}%)")
    print(f"    {'write':>12} : {write_cnt:>12,}  ({write_pct:>2}%)")
    print(f"    {'open_noat':>12} : {open_cnt:>12,}  ({open_pct:>2}%)  ← 文件打开（每个.pt文件一次）")
    print(f"    {'close':>12} : {close_cnt:>12,}  ({close_pct:>2}%)")
    print(f"    {'getattr':>12} : {getattr_cnt:>12,}  ({getattr_pct:>2}%)")
    print(f"    {'delegreturn':>12} : {delegreturn_cnt:>12,}  ({delegreturn_pct:>2}%)  ← NFS委托回收")

    # 诊断
    issues = []
    if delegreturn_pct > 10:
        issues.append(
            f"delegreturn 占 {delegreturn_pct}%（正常应 <5%）\n"
            f"     原因: 多个 worker 竞争访问同批文件，NFS 服务端频繁撤销委托\n"
            f"     建议: shuffle=True 时同一文件被多次读取，考虑预先 shuffle 并缓存"
        )
    if open_pct > 15:
        issues.append(
            f"open_noat 占 {open_pct}%，文件打开操作过于频繁\n"
            f"     原因: {total_calls:,} 次累计调用中 {open_cnt:,} 次是文件打开\n"
            f"     说明: 数据集有 {open_cnt:,} 个独立 .pt 文件被频繁打开，NFS 元数据压力大"
        )

    if issues:
        print(f"\n  发现 {len(issues)} 个 NFS 问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  [{i}] ⚠️  {issue}")
    else:
        print(f"\n  ✅ NFS 操作分布正常")

    # 检查 NFS 挂载点
    mount_out = run_cmd("mount | grep nfs")
    if mount_out:
        print(f"\n  NFS 挂载点:")
        for line in mount_out.split("\n")[:5]:
            print(f"    {line}")


# ─── 进程诊断 ────────────────────────────────────────────────────────────────

def diagnose_processes():
    """查看训练相关进程"""
    section("4. 训练进程状态")

    # 统计 torchrun / python 进程
    ps_out = run_cmd(
        "ps aux --sort=-%cpu | grep -E '(python|torchrun)' | grep -v grep | head -20"
    )
    if ps_out and "命令执行失败" not in ps_out:
        lines = ps_out.strip().split("\n")
        print(f"  Top CPU 进程（共 {len(lines)} 个）:")
        print(f"  {'PID':>8} {'CPU%':>6} {'MEM%':>6} {'状态':>4}  命令")
        print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*4}  {'-'*30}")
        for line in lines[:15]:
            parts = line.split(None, 10)
            if len(parts) >= 11:
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                stat = parts[7]
                cmd = parts[10][:60]
                print(f"  {pid:>8} {cpu:>6} {mem:>6} {stat:>4}  {cmd}")

    # 统计 DataLoader worker 数量
    worker_count = run_cmd(
        "ps aux | grep 'python' | grep -v grep | grep -c 'DataLoader' || "
        "ps aux | grep 'python' | grep -v grep | wc -l"
    )
    total_py = run_cmd("pgrep -c python 2>/dev/null || echo 0")
    print(f"\n  总 Python 进程数: {total_py.strip()}")
    print(f"  (包含 8 个训练主进程 + num_workers × 8 个 DataLoader worker)")


# ─── 本地磁盘检查 ────────────────────────────────────────────────────────────

def check_local_disk():
    """检查是否有本地 SSD 可以用于缓存数据"""
    section("5. 本地磁盘检查（数据迁移可行性）")

    df_out = run_cmd("df -h --output=source,fstype,size,used,avail,target 2>/dev/null | grep -v tmpfs | grep -v nfs")
    if df_out and "命令执行失败" not in df_out:
        print("  非 NFS/tmpfs 磁盘:")
        print(f"  {'来源':<20} {'类型':<8} {'大小':>8} {'已用':>8} {'可用':>8}  挂载点")
        for line in df_out.split("\n"):
            if line and "Filesystem" not in line:
                parts = line.split()
                if len(parts) >= 6:
                    print(f"  {parts[0]:<20} {parts[1]:<8} {parts[2]:>8} {parts[3]:>8} {parts[4]:>8}  {parts[5]}")

    # 检查 /tmp /local /scratch
    for path in ["/tmp", "/local", "/scratch", "/data", "/nvme"]:
        if os.path.exists(path):
            stat_out = run_cmd(f"df -h {path} | tail -1")
            if stat_out:
                print(f"\n  {path}: {stat_out}")

    # 数据集大小估算
    data_root = "/work/2024/tanzunsheng/PENCIData"
    hbn_size = run_cmd(f"du -sh {data_root}/HBN_EEG 2>/dev/null | cut -f1")
    if hbn_size and "命令执行失败" not in hbn_size:
        print(f"\n  HBN_EEG 数据集大小: {hbn_size} （占总数据 91%）")
        print(f"  如本地磁盘有足够空间，建议 rsync 到本地后训练，完全绕开 NFS")


# ─── 综合建议 ────────────────────────────────────────────────────────────────

def print_final_summary():
    section("6. 综合优化建议（按优先级）")

    print("""
  [优先级 1] ⭐ 使用正式配置启动训练（最重要）
     当前问题: smoke_test 的 batch_size=4 远低于 4090D 的最佳工作点
     解决: torchrun --nproc_per_node=8 scripts/v1/train_stage1.py --config configs/stage1_real.yaml
     预期效果: 使用 V1 主线真实数据训练入口，避免误用已归档的旧主线脚本

  [优先级 2] ⭐ 控制 num_workers，继续优先稳住 NFS I/O 压力
     当前: stage1_real.yaml 已将 num_workers 设为 1，避免多进程并发抢占 NFS
     修改: 如需调参，请在 configs/stage1_real.yaml 中将 num_workers 控制在 1-2
     预期效果: context switch 从 ~35000/s 降到 ~20000/s, SSH 卡顿减轻

  [优先级 3] 💡 检查是否有本地 SSD（见上方磁盘检查）
     如有可用本地盘，将 HBN_EEG（最大数据集）rsync 过去
     命令: rsync -av /work/2024/tanzunsheng/PENCIData/HBN_EEG /local/penci_data/
     效果: 彻底消除 NFS delegreturn 问题

  [优先级 4] 💡 Loss 爆炸问题（steps 18-22 出现 ~162000 的异常值）
     原因: 不同 bucket 的数据幅值差异可能很大（如 60ch vs 128ch 数据集）
     建议: 在 PENCICollator 中加入 per-sample 归一化，或检查数据预处理是否一致

  [监控命令]
     # 实时 GPU 监控（每秒刷新）
     watch -n 1 nvidia-smi

     # 系统资源
     vmstat 2

     # NFS 压力
     nfsstat -c 5
    """)


# ─── 主入口 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PENCI 训练系统诊断工具")
    parser.add_argument("--watch", action="store_true", help="持续监控模式")
    parser.add_argument("--interval", type=int, default=5, help="刷新间隔（秒），默认 5")
    args = parser.parse_args()

    def run_once():
        print(f"\n{'#' * 60}")
        print(f"  PENCI 训练系统诊断报告")
        print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  节点: {run_cmd('hostname')}")
        print(f"{'#' * 60}")

        diagnose_gpu()
        diagnose_cpu_memory()
        diagnose_nfs()
        diagnose_processes()
        check_local_disk()
        print_final_summary()

    if args.watch:
        print(f"持续监控模式，每 {args.interval} 秒刷新一次，Ctrl+C 退出")
        try:
            while True:
                os.system("clear")
                run_once()
                print(f"\n下次刷新: {args.interval} 秒后... (Ctrl+C 退出)")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n已退出监控")
    else:
        run_once()


if __name__ == "__main__":
    main()
