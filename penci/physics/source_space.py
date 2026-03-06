# -*- coding: utf-8 -*-
"""
源空间定义模块

定义 PENCI 使用的 72 个源空间位置：
- 68 个 Desikan-Killiany (aparc) 皮层脑区质心
- 4 个皮层下结构质心：左右海马体、左右杏仁核

使用 fsaverage 模板头模型。

依赖：
- MNE-Python >= 1.8.0
- fsaverage 数据（需手动下载到 subjects_dir）
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 皮层下结构的 FreeSurfer aseg 标签名
SUBCORTICAL_LABELS = [
    "Left-Hippocampus",
    "Right-Hippocampus",
    "Left-Amygdala",
    "Right-Amygdala",
]

# DK68 aparc 分区应包含 68 个标签（34 每半球，不含 'unknown'）
N_CORTICAL_LABELS = 68
N_SUBCORTICAL_LABELS = len(SUBCORTICAL_LABELS)
N_TOTAL_SOURCES = N_CORTICAL_LABELS + N_SUBCORTICAL_LABELS  # = 72


class SourceSpace:
    """
    72 个源空间定义

    通过 MNE 从 fsaverage 提取：
    - 68 个 DK68 皮层脑区质心位置（米制，MNI 坐标）
    - 4 个皮层下结构质心位置（米制，MNI 坐标）

    属性:
        positions: (72, 3) numpy 数组，每个源的 (x, y, z) 坐标（米）
        names: 长度 72 的列表，每个源的名称
        cortical_positions: (68, 3) 皮层源坐标
        subcortical_positions: (4, 3) 皮层下源坐标
        src: MNE 混合源空间对象（surface + volume）
    """

    def __init__(
        self,
        subjects_dir: str,
        subject: str = "fsaverage",
    ):
        """
        参数:
            subjects_dir: FreeSurfer subjects 目录路径
                          例如 /work/2024/tanzunsheng/mne_data/MNE-fsaverage-data
            subject: FreeSurfer 被试名称，默认 'fsaverage'
        """
        self._subjects_dir = Path(subjects_dir)
        self._subject = subject

        # 验证 fsaverage 存在
        fsaverage_dir = self._subjects_dir / subject
        if not fsaverage_dir.exists():
            raise FileNotFoundError(
                f"fsaverage 目录不存在: {fsaverage_dir}\n"
                f"请手动下载 fsaverage 到 {subjects_dir}：\n"
                f"  root.zip: https://osf.io/3bxqt/download?version=2\n"
                f"  bem.zip:  https://osf.io/7ve8g/download?version=4"
            )

        # 延迟计算：首次访问时才计算源位置
        self._positions: Optional[np.ndarray] = None
        self._names: Optional[List[str]] = None
        self._cortical_positions: Optional[np.ndarray] = None
        self._subcortical_positions: Optional[np.ndarray] = None
        self._cortical_names: Optional[List[str]] = None
        self._subcortical_names: Optional[List[str]] = None
        self._src = None

    @property
    def positions(self) -> np.ndarray:
        """(72, 3) 所有源的坐标（米，MNI 空间）"""
        if self._positions is None:
            self._compute_source_positions()
        return self._positions

    @property
    def names(self) -> List[str]:
        """长度 72 的列表，每个源的名称"""
        if self._names is None:
            self._compute_source_positions()
        return self._names

    @property
    def cortical_positions(self) -> np.ndarray:
        """(68, 3) 皮层源坐标"""
        if self._cortical_positions is None:
            self._compute_source_positions()
        return self._cortical_positions

    @property
    def subcortical_positions(self) -> np.ndarray:
        """(4, 3) 皮层下源坐标"""
        if self._subcortical_positions is None:
            self._compute_source_positions()
        return self._subcortical_positions

    @property
    def src(self):
        """MNE 混合源空间对象"""
        if self._src is None:
            self._build_source_space()
        return self._src

    def _compute_source_positions(self) -> None:
        """计算 72 个源的质心位置"""
        try:
            import mne
        except ImportError:
            raise ImportError(
                "需要安装 MNE-Python: pip install mne\n"
                "不允许 fallback 到随机矩阵。"
            )

        subjects_dir = str(self._subjects_dir)
        subject = self._subject

        # === 1. 读取 DK68 皮层标签并计算质心 ===
        logger.info("读取 DK68 (aparc) 皮层标签...")
        labels = mne.read_labels_from_annot(
            subject=subject,
            parc="aparc",
            subjects_dir=subjects_dir,
        )

        # 过滤掉 'unknown' 标签
        cortical_labels = [
            label for label in labels if "unknown" not in label.name.lower()
        ]

        if len(cortical_labels) != N_CORTICAL_LABELS:
            raise ValueError(
                f"预期 {N_CORTICAL_LABELS} 个皮层标签，"
                f"但得到 {len(cortical_labels)} 个。"
                f"标签名称: {[l.name for l in cortical_labels]}"
            )

        # 读取源空间以获取顶点坐标
        src_surf = mne.read_source_spaces(
            str(
                self._subjects_dir
                / subject
                / "bem"
                / f"{subject}-ico-5-src.fif"
            ),
            verbose=False,
        )

        cortical_positions = []
        cortical_names = []

        for label in cortical_labels:
            # 确定半球索引
            hemi_idx = 0 if label.hemi == "lh" else 1
            src_hemi = src_surf[hemi_idx]

            # 获取该标签在源空间中的顶点位置
            # label.vertices 是 FreeSurfer 表面上的顶点编号
            # src_hemi['vertno'] 是源空间中使用的顶点子集
            # src_hemi['rr'] 是所有表面顶点的坐标（米）
            label_vertex_positions = src_hemi["rr"][label.vertices]

            # 计算质心（简单平均）
            centroid = label_vertex_positions.mean(axis=0)
            cortical_positions.append(centroid)
            cortical_names.append(label.name)

        self._cortical_positions = np.array(cortical_positions, dtype=np.float64)
        self._cortical_names = cortical_names
        logger.info(f"  计算了 {len(cortical_names)} 个皮层脑区质心")

        # === 2. 计算皮层下结构质心 ===
        logger.info("计算皮层下结构质心...")

        # 使用 MNE 的体积源空间提取皮层下位置
        # 需要 BEM 模型和 aseg.mgz
        bem_path = (
            self._subjects_dir
            / subject
            / "bem"
            / f"{subject}-5120-5120-5120-bem-sol.fif"
        )
        if not bem_path.exists():
            # 尝试其他常见 BEM 文件名
            bem_path = (
                self._subjects_dir
                / subject
                / "bem"
                / f"{subject}-5120-bem-sol.fif"
            )
        if not bem_path.exists():
            # 列出 bem 目录中可用的文件
            bem_dir = self._subjects_dir / subject / "bem"
            available = list(bem_dir.glob("*.fif")) if bem_dir.exists() else []
            raise FileNotFoundError(
                f"BEM 解决方案文件不存在: {bem_path}\n"
                f"可用的 BEM 文件: {[f.name for f in available]}"
            )

        # 读取 BEM
        bem = mne.read_bem_solution(str(bem_path), verbose=False)

        # 创建体积源空间
        vol_src = mne.setup_volume_source_space(
            subject=subject,
            mri="aseg.mgz",
            pos=5.0,  # 5mm 网格间距
            bem=bem,
            volume_label=SUBCORTICAL_LABELS,
            subjects_dir=subjects_dir,
            verbose=False,
        )

        subcortical_positions = []
        subcortical_names = []

        for src_space in vol_src:
            seg_name = src_space.get("seg_name", "unknown")
            # 只取被使用的源点（inuse == 1）
            in_use_mask = src_space["inuse"].astype(bool)
            if in_use_mask.sum() == 0:
                logger.warning(f"  皮层下区域 '{seg_name}' 没有活跃源点")
                # 使用所有点
                positions = src_space["rr"]
            else:
                positions = src_space["rr"][in_use_mask]

            centroid = positions.mean(axis=0)
            subcortical_positions.append(centroid)
            subcortical_names.append(seg_name)

        if len(subcortical_positions) != N_SUBCORTICAL_LABELS:
            raise ValueError(
                f"预期 {N_SUBCORTICAL_LABELS} 个皮层下结构，"
                f"但得到 {len(subcortical_positions)} 个。"
                f"名称: {subcortical_names}"
            )

        self._subcortical_positions = np.array(
            subcortical_positions, dtype=np.float64
        )
        self._subcortical_names = subcortical_names
        logger.info(f"  计算了 {len(subcortical_names)} 个皮层下结构质心")

        # === 3. 合并 ===
        self._positions = np.vstack(
            [self._cortical_positions, self._subcortical_positions]
        )
        self._names = self._cortical_names + self._subcortical_names

        assert self._positions.shape == (N_TOTAL_SOURCES, 3), (
            f"源空间形状错误: {self._positions.shape}，预期 ({N_TOTAL_SOURCES}, 3)"
        )
        logger.info(f"源空间总计 {N_TOTAL_SOURCES} 个源")

    def _build_source_space(self) -> None:
        """构建 MNE 混合源空间对象（用于前向计算）"""
        try:
            import mne
        except ImportError:
            raise ImportError("需要安装 MNE-Python")

        subjects_dir = str(self._subjects_dir)
        subject = self._subject

        # 读取皮层源空间
        src_surf = mne.read_source_spaces(
            str(
                self._subjects_dir
                / subject
                / "bem"
                / f"{subject}-ico-5-src.fif"
            ),
            verbose=False,
        )

        # 读取 BEM
        bem_path = (
            self._subjects_dir
            / subject
            / "bem"
            / f"{subject}-5120-5120-5120-bem-sol.fif"
        )
        if not bem_path.exists():
            bem_path = (
                self._subjects_dir
                / subject
                / "bem"
                / f"{subject}-5120-bem-sol.fif"
            )

        bem = mne.read_bem_solution(str(bem_path), verbose=False)

        # 创建体积源空间
        vol_src = mne.setup_volume_source_space(
            subject=subject,
            mri="aseg.mgz",
            pos=5.0,
            bem=bem,
            volume_label=SUBCORTICAL_LABELS,
            subjects_dir=subjects_dir,
            verbose=False,
        )

        # 合并为混合源空间
        self._src = src_surf + vol_src

    def get_source_info(self) -> Dict:
        """返回源空间的详细信息"""
        return {
            "n_total": N_TOTAL_SOURCES,
            "n_cortical": N_CORTICAL_LABELS,
            "n_subcortical": N_SUBCORTICAL_LABELS,
            "cortical_names": self._cortical_names or [],
            "subcortical_names": self._subcortical_names or SUBCORTICAL_LABELS,
            "subjects_dir": str(self._subjects_dir),
            "subject": self._subject,
        }
