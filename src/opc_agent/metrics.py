"""本模块提供与 OpenILT 评估语义对齐的轻量指标和位移离散化工具。

输入为二值 NumPy 图像、物理指标或纳米位移，输出为 OpcMetrics 数值字段、论文加权损失和九分类动作编号。
关键依赖仅为 NumPy，便于在 CPU 单元测试中验证确定性逻辑。
"""
from __future__ import annotations

import time
from typing import Mapping, Tuple

import numpy as np

from .models import OpcMetrics


DISPLACEMENT_CLASSES_NM: Tuple[int, ...] = (-40, -30, -20, -10, 0, 10, 20, 30, 40)
SIMPLEOPC_LOSS_VERSION = "paper-weighted-sum-initial-normalized-v1"
RECIPE_OPC_LOSS_VERSION = "paper-weighted-sum-raw-v1"


def weighted_opc_loss(metrics: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """计算论文形式的原始加权和，训练与质量验收必须共同调用。"""
    expected = {"l2", "epe", "pvb"}
    if set(metrics) != expected or set(weights) != expected:
        raise ValueError("OPC 损失必须恰好包含 l2、epe、pvb")
    values = {name: float(metrics[name]) for name in expected}
    coefficients = {name: float(weights[name]) for name in expected}
    if any(value < 0 for value in values.values()):
        raise ValueError("OPC 物理指标不能为负")
    if any(value < 0 for value in coefficients.values()) or sum(coefficients.values()) <= 0:
        raise ValueError("OPC 损失权重必须非负且至少一个大于零")
    return float(sum(coefficients[name] * values[name] for name in expected))


def quantize_displacement(displacement_nm: float) -> int:
    """将 [-40, 40] nm 的连续位移映射到最近的九分类编号。"""
    value = float(np.clip(displacement_nm, -40, 40))
    return int(np.argmin(np.abs(np.asarray(DISPLACEMENT_CLASSES_NM) - value)))


def dequantize_displacement(displacement_class: int) -> int:
    """将合法九分类编号还原为对应的代表位移（nm）。"""
    if not 0 <= displacement_class < len(DISPLACEMENT_CLASSES_NM):
        raise ValueError("位移分类必须在 0 到 8 之间")
    return DISPLACEMENT_CLASSES_NM[displacement_class]


def _boundary(binary: np.ndarray) -> np.ndarray:
    """返回目标图中与四邻域不同的前景边界像素。"""
    padded = np.pad(binary.astype(bool), 1, constant_values=False)
    center = padded[1:-1, 1:-1]
    interior = center & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return center & ~interior


def compute_binary_metrics(
    target: np.ndarray, printed_nominal: np.ndarray, printed_max: np.ndarray, printed_min: np.ndarray
) -> OpcMetrics:
    """计算二值 L2、PVB 和简化 EPE；形状不一致会显式失败。"""
    started = time.perf_counter()
    arrays = [np.asarray(item, dtype=bool) for item in (target, printed_nominal, printed_max, printed_min)]
    if len({item.shape for item in arrays}) != 1 or arrays[0].ndim != 2:
        raise ValueError("四张输入图必须是形状相同的二维数组")
    target_b, nominal_b, maximum_b, minimum_b = arrays
    l2 = float(np.count_nonzero(target_b != nominal_b))
    pvb = float(np.count_nonzero(maximum_b != minimum_b))
    boundary_error = _boundary(target_b) & (target_b != nominal_b)
    epe_n = int(np.count_nonzero(boundary_error))
    epe_d = float(epe_n)
    return OpcMetrics(l2=l2, pvb=pvb, epe_n=epe_n, epe_d=epe_d, runtime_seconds=time.perf_counter() - started)
