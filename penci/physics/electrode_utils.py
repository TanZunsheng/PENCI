# -*- coding: utf-8 -*-
"""
电极工具模块

从 ProcessedData (BIDS 格式) 的 electrodes.tsv 读取电极坐标，
并复制 BrainOmniPostProcess 的通道过滤逻辑，确保训练时使用的通道
与预处理后 .pt 文件中的通道完全一致。

关键约束：
- 坐标单位为米（CapTrak 坐标系）
- 不从 .pt 文件反推坐标，必须从 electrodes.tsv 读取原始米制坐标
- 通道过滤逻辑与 BrainOmniPostProcess (brainomni_postprocess.py) 完全一致

数据集路径模式：
- 通用: {root}/{dataset}/bids/derivatives/preprocessing/sub-{subject}/eeg/sub-{subject}_space-CapTrak_electrodes.tsv
- HBN_EEG: {root}/HBN_EEG/HBN_cmi_bids_{site}/bids/derivatives/preprocessing/sub-{subject}/eeg/...
- ThingsEEG: {root}/ThingsEEG/bids/derivatives/preprocessing/sub-{subject}/ses-{session}/eeg/...
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def find_electrodes_tsv(
    processed_data_dir: str,
    dataset_name: str,
    subject_id: str,
    session: Optional[str] = None,
    site: Optional[str] = None,
) -> Path:
    """
    查找 electrodes.tsv 文件路径

    参数:
        processed_data_dir: ProcessedData 根目录
                           例如 /work/2024/tanzunsheng/ProcessedData
        dataset_name: 数据集名称，如 "HBN_EEG", "THINGS-EEG", "ThingsEEG" 等
        subject_id: 被试 ID（不含 "sub-" 前缀）
        session: 会话 ID（仅 ThingsEEG 需要，不含 "ses-" 前缀）
        site: 站点名称（仅 HBN_EEG 需要，如 "SI", "RU", "CBIC", "CUNY"）

    返回:
        electrodes.tsv 文件的 Path 对象

    异常:
        FileNotFoundError: 找不到 electrodes.tsv 文件
    """
    root = Path(processed_data_dir)

    # 构建被试前缀
    sub_prefix = f"sub-{subject_id}"

    # 根据数据集确定路径
    if dataset_name == "HBN_EEG":
        # HBN_EEG 有站点子目录: HBN_EEG/HBN_cmi_bids_{site}/bids/...
        if site is not None:
            candidates = [
                root / "HBN_EEG" / f"HBN_cmi_bids_{site}" / "bids"
                / "derivatives" / "preprocessing" / sub_prefix / "eeg"
                / f"{sub_prefix}_space-CapTrak_electrodes.tsv"
            ]
        else:
            # 不知道站点，搜索所有站点子目录
            hbn_root = root / "HBN_EEG"
            candidates = []
            if hbn_root.exists():
                for site_dir in sorted(hbn_root.iterdir()):
                    if site_dir.is_dir() and site_dir.name.startswith("HBN_cmi_bids_"):
                        candidate = (
                            site_dir / "bids" / "derivatives" / "preprocessing"
                            / sub_prefix / "eeg"
                            / f"{sub_prefix}_space-CapTrak_electrodes.tsv"
                        )
                        candidates.append(candidate)

    elif dataset_name == "ThingsEEG" and session is not None:
        # ThingsEEG 有 session 层级
        candidates = [
            root / dataset_name / "bids" / "derivatives" / "preprocessing"
            / sub_prefix / f"ses-{session}" / "eeg"
            / f"{sub_prefix}_ses-{session}_space-CapTrak_electrodes.tsv"
        ]
    else:
        # 通用路径模式（Brennan_Hale2019, THINGS-EEG, Grootswagers2019, SEED-DV, ThingsEEG 无 session）
        candidates = [
            root / dataset_name / "bids" / "derivatives" / "preprocessing"
            / sub_prefix / "eeg"
            / f"{sub_prefix}_space-CapTrak_electrodes.tsv"
        ]

    # 查找存在的候选路径
    for candidate in candidates:
        if candidate.exists():
            logger.debug(f"找到 electrodes.tsv: {candidate}")
            return candidate

    # 所有候选路径都不存在
    raise FileNotFoundError(
        f"找不到 electrodes.tsv 文件。\n"
        f"数据集: {dataset_name}, 被试: {subject_id}"
        f"{f', 会话: {session}' if session else ''}"
        f"{f', 站点: {site}' if site else ''}\n"
        f"搜索路径:\n" +
        "\n".join(f"  - {c}" for c in candidates)
    )


def read_electrodes_tsv(tsv_path: str) -> Dict[str, np.ndarray]:
    """
    读取 electrodes.tsv 文件，返回通道名到米制坐标的映射

    文件格式（TSV，可能有 BOM 头）:
        name    x    y    z    [impedance]
        Fp1     0.02 0.08 0.01 ...

    参数:
        tsv_path: electrodes.tsv 文件路径

    返回:
        Dict[str, np.ndarray]: 通道名 -> (3,) xyz 坐标数组（米，CapTrak 坐标系）

    异常:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式错误
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"electrodes.tsv 文件不存在: {tsv_path}")

    channels: Dict[str, np.ndarray] = {}

    with open(tsv_path, "r", encoding="utf-8-sig") as f:
        # utf-8-sig 自动处理 BOM 头 (\ufeff)
        header = f.readline().strip()
        columns = header.split("\t")

        # 验证必要列
        required = {"name", "x", "y", "z"}
        col_set = set(columns)
        if not required.issubset(col_set):
            raise ValueError(
                f"electrodes.tsv 缺少必要列: {required - col_set}\n"
                f"实际列: {columns}\n"
                f"文件: {tsv_path}"
            )

        name_idx = columns.index("name")
        x_idx = columns.index("x")
        y_idx = columns.index("y")
        z_idx = columns.index("z")

        for line_num, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                logger.warning(
                    f"electrodes.tsv 第 {line_num} 行格式错误，跳过: {line}"
                )
                continue

            ch_name = parts[name_idx]
            try:
                x = float(parts[x_idx])
                y = float(parts[y_idx])
                z = float(parts[z_idx])
            except (ValueError, IndexError) as e:
                logger.warning(
                    f"electrodes.tsv 第 {line_num} 行坐标解析失败: {e}"
                )
                continue

            channels[ch_name] = np.array([x, y, z], dtype=np.float64)

    if len(channels) == 0:
        raise ValueError(f"electrodes.tsv 中没有有效的通道数据: {tsv_path}")

    logger.debug(f"从 {tsv_path} 读取了 {len(channels)} 个通道坐标")
    return channels


def filter_channels_like_postprocess(
    all_channels: Dict[str, np.ndarray],
    min_valid_channels: int = 10,
) -> Dict[str, np.ndarray]:
    """
    复制 BrainOmniPostProcess 的通道过滤逻辑

    过滤规则（与 brainomni_postprocess.py 第 514-549 行完全一致）:
    1. 检查每个通道的坐标是否有效
    2. 无效条件: 坐标含 NaN 或坐标全零 (atol=1e-10)
    3. 移除所有无效通道
    4. 要求有效通道数 >= min_valid_channels

    注意: 此函数不做 EEG/MEG 类型过滤（pick_types），因为 electrodes.tsv
    本身只包含 EEG 通道。Cz 等参考/misc 通道如果不在 montage 的 ch_pos
    中（即在 electrodes.tsv 中无有效坐标），会被自动过滤掉。但实际上
    HBN_EEG 的 Cz 在 electrodes.tsv 中有坐标 — Cz 是被 pick_types
    排除的（它的类型是 misc 而非 eeg）。由于 electrodes.tsv 不包含
    通道类型信息，我们需要额外的排除列表来处理这种情况。

    参数:
        all_channels: 通道名 -> (3,) 坐标 字典（来自 read_electrodes_tsv）
        min_valid_channels: 最少有效通道数，低于此值将报错

    返回:
        Dict[str, np.ndarray]: 过滤后的通道名 -> 坐标字典（保持原始顺序）

    异常:
        ValueError: 有效通道数少于 min_valid_channels
    """
    valid_channels: Dict[str, np.ndarray] = {}
    invalid_names: List[str] = []

    for ch_name, coord in all_channels.items():
        # 检查坐标是否有效（与 BrainOmniPostProcess 逻辑一致）
        if np.isnan(coord).any():
            invalid_names.append(ch_name)
            continue
        if np.allclose(coord, 0, atol=1e-10):
            invalid_names.append(ch_name)
            continue

        valid_channels[ch_name] = coord

    if invalid_names:
        logger.info(
            f"过滤掉 {len(invalid_names)} 个无效坐标通道: {invalid_names}"
        )

    if len(valid_channels) < min_valid_channels:
        raise ValueError(
            f"有效通道数太少: {len(valid_channels)} < {min_valid_channels}\n"
            f"总通道: {len(all_channels)}, 无效通道: {invalid_names}"
        )

    logger.debug(
        f"通道过滤: {len(all_channels)} -> {len(valid_channels)} "
        f"(移除 {len(invalid_names)} 个)"
    )
    return valid_channels


# BrainOmniPostProcess 中已知会被 pick_types 排除的非 EEG 通道名
# 这些通道虽然在 electrodes.tsv 中有坐标，但类型不是 EEG
# 用于从 electrodes.tsv 通道列表中额外排除
NON_EEG_CHANNEL_NAMES = frozenset({
    "Cz",  # HBN_EEG 中的参考电极，类型为 misc
})


def get_valid_channels_for_dataset(
    processed_data_dir: str,
    dataset_name: str,
    subject_id: str,
    session: Optional[str] = None,
    site: Optional[str] = None,
    exclude_non_eeg: bool = True,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    获取指定数据集/被试的有效电极通道和坐标

    完整流程:
    1. 查找 electrodes.tsv
    2. 读取所有通道坐标
    3. 排除已知非 EEG 通道（如 Cz 参考电极）
    4. 过滤无效坐标通道
    5. 返回有效通道字典和有序通道名列表

    参数:
        processed_data_dir: ProcessedData 根目录
        dataset_name: 数据集名称
        subject_id: 被试 ID
        session: 会话 ID（可选）
        site: 站点名称（可选，仅 HBN_EEG）
        exclude_non_eeg: 是否排除已知非 EEG 通道

    返回:
        (valid_channels, channel_names) 元组:
        - valid_channels: Dict[str, np.ndarray] 通道名 -> (3,) 米制坐标
        - channel_names: List[str] 有序通道名列表
    """
    # 1. 查找 electrodes.tsv
    tsv_path = find_electrodes_tsv(
        processed_data_dir, dataset_name, subject_id,
        session=session, site=site,
    )
    logger.info(f"读取电极坐标: {tsv_path}")

    # 2. 读取所有通道
    all_channels = read_electrodes_tsv(str(tsv_path))

    # 3. 排除已知非 EEG 通道
    if exclude_non_eeg:
        before_count = len(all_channels)
        excluded = []
        for name in list(all_channels.keys()):
            if name in NON_EEG_CHANNEL_NAMES:
                del all_channels[name]
                excluded.append(name)
        if excluded:
            logger.info(
                f"排除非 EEG 通道: {excluded} "
                f"({before_count} -> {len(all_channels)})"
            )

    # 4. 过滤无效坐标
    valid_channels = filter_channels_like_postprocess(all_channels)

    # 5. 返回有序列表
    channel_names = list(valid_channels.keys())
    logger.info(
        f"数据集 {dataset_name} 被试 {subject_id}: "
        f"{len(channel_names)} 个有效通道"
    )

    return valid_channels, channel_names


def _find_reference_subject(
    processed_data_dir: str,
    dataset_name: str,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    为指定数据集自动查找一个参考被试（用于读取典型电极配置）

    扫描 ProcessedData/{dataset} 目录结构，返回第一个有 electrodes.tsv 的被试信息。

    参数:
        processed_data_dir: ProcessedData 根目录
        dataset_name: 数据集名称

    返回:
        (subject_id, session, site) 元组
        - subject_id: 被试 ID（不含 "sub-" 前缀）
        - session: 会话 ID（不含 "ses-" 前缀，仅 ThingsEEG 等有 session 的数据集）
        - site: 站点名称（仅 HBN_EEG）

    异常:
        FileNotFoundError: 找不到任何有效被试
    """
    import re

    root = Path(processed_data_dir)

    if dataset_name == "HBN_EEG":
        # HBN_EEG: 搜索 HBN_cmi_bids_{site} 子目录
        hbn_root = root / "HBN_EEG"
        if hbn_root.exists():
            for site_dir in sorted(hbn_root.iterdir()):
                if not site_dir.is_dir() or not site_dir.name.startswith("HBN_cmi_bids_"):
                    continue
                site_name = site_dir.name.replace("HBN_cmi_bids_", "")
                deriv_dir = site_dir / "bids" / "derivatives" / "preprocessing"
                if not deriv_dir.exists():
                    continue
                for sub_dir in sorted(deriv_dir.iterdir()):
                    if sub_dir.is_dir() and sub_dir.name.startswith("sub-"):
                        subject_id = sub_dir.name[4:]  # 去掉 "sub-" 前缀
                        try:
                            find_electrodes_tsv(
                                processed_data_dir, dataset_name, subject_id,
                                site=site_name,
                            )
                            return subject_id, None, site_name
                        except FileNotFoundError:
                            continue

    elif dataset_name == "ThingsEEG":
        # ThingsEEG: 可能有 session 层级
        deriv_dir = root / dataset_name / "bids" / "derivatives" / "preprocessing"
        if deriv_dir.exists():
            for sub_dir in sorted(deriv_dir.iterdir()):
                if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
                    continue
                subject_id = sub_dir.name[4:]
                # 先尝试带 session
                for ses_dir in sorted(sub_dir.iterdir()):
                    if ses_dir.is_dir() and ses_dir.name.startswith("ses-"):
                        session = ses_dir.name[4:]
                        try:
                            find_electrodes_tsv(
                                processed_data_dir, dataset_name, subject_id,
                                session=session,
                            )
                            return subject_id, session, None
                        except FileNotFoundError:
                            continue
                # 无 session 模式
                try:
                    find_electrodes_tsv(
                        processed_data_dir, dataset_name, subject_id,
                    )
                    return subject_id, None, None
                except FileNotFoundError:
                    continue

    else:
        # 通用数据集
        deriv_dir = root / dataset_name / "bids" / "derivatives" / "preprocessing"
        if deriv_dir.exists():
            for sub_dir in sorted(deriv_dir.iterdir()):
                if sub_dir.is_dir() and sub_dir.name.startswith("sub-"):
                    subject_id = sub_dir.name[4:]
                    try:
                        find_electrodes_tsv(
                            processed_data_dir, dataset_name, subject_id,
                        )
                        return subject_id, None, None
                    except FileNotFoundError:
                        continue

    raise FileNotFoundError(
        f"在 {processed_data_dir}/{dataset_name} 中找不到有效的参考被试\n"
        f"请确认 ProcessedData 目录中包含至少一个被试的 electrodes.tsv"
    )


class ElectrodeConfigRegistry:
    """
    电极配置注册表

    为每个 (dataset, n_channels) 组合维护一份典型电极配置（通道名 + 米制坐标）。
    利用的关键事实：同一数据集中使用相同 EEG 帽的所有被试共享相同的通道名称集，
    因此只需读取一个参考被试的 electrodes.tsv 即可代表整个数据集。

    使用方式:
        1. 初始化时传入 processed_data_dir
        2. 调用 register_dataset() 注册数据集（自动扫描参考被试）
        3. 训练循环中调用 get_config() 根据 batch 的 dataset_name 获取电极配置

    配合 LeadfieldManager 使用:
        registry = ElectrodeConfigRegistry(processed_data_dir)
        registry.register_dataset("HBN_EEG")
        names, positions = registry.get_config("HBN_EEG", 128)
        leadfield = leadfield_manager.get_leadfield(names, positions, device)
    """

    def __init__(self, processed_data_dir: str):
        """
        参数:
            processed_data_dir: ProcessedData 根目录（含 BIDS 格式数据 + electrodes.tsv）
        """
        self._processed_data_dir = processed_data_dir
        # 缓存: (dataset_name, n_channels) -> (channel_names, channel_positions)
        self._configs: Dict[Tuple[str, int], Tuple[List[str], np.ndarray]] = {}

    def register_dataset(self, dataset_name: str) -> None:
        """
        注册一个数据集的电极配置

        自动查找参考被试，读取 electrodes.tsv 并缓存。

        参数:
            dataset_name: 数据集名称（如 "HBN_EEG", "SEED-DV"）

        异常:
            FileNotFoundError: 找不到参考被试或 electrodes.tsv
        """
        # 查找参考被试
        subject_id, session, site = _find_reference_subject(
            self._processed_data_dir, dataset_name
        )
        logger.info(
            f"数据集 '{dataset_name}' 参考被试: sub-{subject_id}"
            f"{f' ses-{session}' if session else ''}"
            f"{f' site-{site}' if site else ''}"
        )

        # 读取有效电极配置
        valid_channels, channel_names = get_valid_channels_for_dataset(
            self._processed_data_dir, dataset_name, subject_id,
            session=session, site=site,
        )

        # 构建坐标数组
        channel_positions = np.array(
            [valid_channels[name] for name in channel_names],
            dtype=np.float64,
        )
        n_channels = len(channel_names)

        # 注册
        self._configs[(dataset_name, n_channels)] = (channel_names, channel_positions)
        logger.info(
            f"注册电极配置: ({dataset_name}, {n_channels}ch) "
            f"-> {n_channels} 通道"
        )

    def get_config(
        self,
        dataset_name: str,
        n_channels: int,
    ) -> Tuple[List[str], np.ndarray]:
        """
        获取指定数据集+通道数的电极配置

        参数:
            dataset_name: 数据集名称
            n_channels: 通道数

        返回:
            (channel_names, channel_positions) 元组:
            - channel_names: List[str] 有序通道名列表
            - channel_positions: (n_channels, 3) 米制坐标数组

        异常:
            KeyError: 未注册该配置
        """
        key = (dataset_name, n_channels)
        if key not in self._configs:
            raise KeyError(
                f"未注册的电极配置: {key}\n"
                f"已注册配置: {list(self._configs.keys())}\n"
                f"请先调用 register_dataset('{dataset_name}')"
            )
        return self._configs[key]

    def has_config(self, dataset_name: str, n_channels: int) -> bool:
        """检查是否已注册指定 (dataset, n_channels) 配置"""
        return (dataset_name, n_channels) in self._configs

    def has_config_for_dataset(self, dataset_name: str) -> bool:
        """检查是否已注册指定数据集的任意通道配置"""
        return any(ds == dataset_name for ds, _ in self._configs)

    @property
    def registered_configs(self) -> List[Tuple[str, int]]:
        """列出所有已注册的 (dataset, n_channels) 配置"""
        return list(self._configs.keys())
