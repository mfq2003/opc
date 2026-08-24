"""本模块计算用于 PPO 与闭环标签的确定性逐点边界放置误差（EPE）。

输入为形状相同的目标二值图和印刷二值图，以及以像素为单位的容差；输出为目标边界样本数、
超过容差的 EPE N 和所有最近边界距离之和 EPE D。关键依赖为 NumPy 与 SciPy 距离变换；
此定义用于项目内细颗粒度标签，和 OpenILT 上游日志中的总 EPE 需分别记录，不能相互替代。
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from scipy.ndimage import distance_transform_edt


class EpeMeasurement(BaseModel):
    """保存一次目标边界到印刷边界的可审计 EPE N/EPE D 测量结果。"""

    schema_version: str = "1.0"
    sample_count: int = Field(ge=0)
    tolerance_pixels: float = Field(ge=0)
    epe_n: int = Field(ge=0)
    epe_d: float = Field(ge=0)
    mean_distance_pixels: float = Field(ge=0)


def foreground_boundary(image: np.ndarray) -> np.ndarray:
    """返回四邻域中至少有一个背景像素的前景边界掩膜。"""
    binary = np.asarray(image, dtype=bool)
    if binary.ndim != 2 or min(binary.shape) < 2:
        raise ValueError("图像必须是边长至少为 2 的二维数组")
    padded = np.pad(binary, 1, constant_values=False)
    center = padded[1:-1, 1:-1]
    interior = center & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return center & ~interior


def measure_boundary_epe(target: np.ndarray, printed: np.ndarray, tolerance_pixels: float = 0.5) -> EpeMeasurement:
    """测量目标前景边界到印刷前景边界的最近欧氏距离并统计 N/D。"""
    target_binary = np.asarray(target, dtype=bool)
    printed_binary = np.asarray(printed, dtype=bool)
    if target_binary.shape != printed_binary.shape:
        raise ValueError("目标图与印刷图形状必须一致")
    if tolerance_pixels < 0:
        raise ValueError("EPE 容差不能为负数")
    target_boundary = foreground_boundary(target_binary)
    printed_boundary = foreground_boundary(printed_binary)
    sample_count = int(np.count_nonzero(target_boundary))
    if sample_count == 0:
        return EpeMeasurement(sample_count=0, tolerance_pixels=tolerance_pixels, epe_n=0, epe_d=0.0, mean_distance_pixels=0.0)
    if not np.any(printed_boundary):
        distances = np.full(sample_count, float(max(target_binary.shape)), dtype=np.float64)
    else:
        nearest_printed_boundary = distance_transform_edt(~printed_boundary)
        distances = nearest_printed_boundary[target_boundary]
    total_distance = float(np.sum(distances))
    return EpeMeasurement(
        sample_count=sample_count,
        tolerance_pixels=float(tolerance_pixels),
        epe_n=int(np.count_nonzero(distances > tolerance_pixels)),
        epe_d=total_distance,
        mean_distance_pixels=float(total_distance / sample_count),
    )

