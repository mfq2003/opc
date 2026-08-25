"""本模块从公开版图栅格提取确定性的几何特征并检测训练分布外样本。

输入为二维二值图或数值特征矩阵，输出为稳定的 geometry-v1 特征字典和 OOD 判定。
关键依赖为 NumPy；这些特征在 Qwen 标签失效时可作为权威标签来源。
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


FEATURE_NAMES = ("fill_ratio", "horizontal_edge_density", "vertical_edge_density", "corner_density", "bbox_aspect")
SIMPLEOPC_SEGMENT_FEATURE_NAMES = (
    "midpoint_x_nm",
    "midpoint_y_nm",
    "segment_length_nm",
    "is_horizontal",
    "normal_x",
    "normal_y",
    "initial_epe_sign",
)


def feature_names_for_version(feature_version: str):
    """返回版本化决策树字段，拒绝把旧全图特征和新边段特征静默混用。"""
    versions = {
        "geometry-v1": FEATURE_NAMES,
        "simpleopc-segment-v1": SIMPLEOPC_SEGMENT_FEATURE_NAMES,
    }
    try:
        return versions[str(feature_version)]
    except KeyError as exc:
        raise ValueError(f"未知 feature_version：{feature_version}") from exc


def extract_geometry_features(image: np.ndarray) -> Dict[str, float]:
    """从二维二值栅格抽取五个归一化几何特征。"""
    binary = np.asarray(image, dtype=bool)
    if binary.ndim != 2 or min(binary.shape) < 2:
        raise ValueError("image 必须是边长至少为 2 的二维数组")
    height, width = binary.shape
    horizontal = np.count_nonzero(binary[:, 1:] != binary[:, :-1]) / (height * (width - 1))
    vertical = np.count_nonzero(binary[1:, :] != binary[:-1, :]) / ((height - 1) * width)
    corners = (binary[:-1, :-1] != binary[1:, 1:]) & (binary[1:, :-1] != binary[:-1, 1:])
    locations = np.argwhere(binary)
    if len(locations) == 0:
        aspect = 0.0
    else:
        extents = locations.max(axis=0) - locations.min(axis=0) + 1
        aspect = float(extents[1] / extents[0])
    return {
        "fill_ratio": float(binary.mean()),
        "horizontal_edge_density": float(horizontal),
        "vertical_edge_density": float(vertical),
        "corner_density": float(corners.mean()),
        "bbox_aspect": aspect,
    }


def feature_vector(features: Dict[str, float]) -> np.ndarray:
    """按照固定字段顺序将特征字典变为决策树可用的向量。"""
    return np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64)


def is_out_of_distribution(vector: Sequence[float], training_vectors: Iterable[Sequence[float]], zscore_limit: float = 3.0) -> bool:
    """用逐维稳健 Z 分数判断样本是否超出训练分布。"""
    train = np.asarray(list(training_vectors), dtype=np.float64)
    value = np.asarray(vector, dtype=np.float64)
    if train.ndim != 2 or train.shape[0] < 2 or train.shape[1] != value.size:
        raise ValueError("训练特征至少需要两个且维度与输入一致")
    center = np.median(train, axis=0)
    mad = np.median(np.abs(train - center), axis=0)
    scale = np.where(mad < 1e-12, 1.0, 1.4826 * mad)
    return bool(np.any(np.abs((value - center) / scale) > zscore_limit))
