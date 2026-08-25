"""本模块验证二值指标计算和连续位移到九分类动作的确定性映射。

输入为小型 NumPy 合成图和位移数值，输出为可人工推导的 L2/PVB/EPE 与类别断言。
关键依赖为 NumPy 和 pytest，不需要 OpenILT 或 CUDA。
"""
import numpy as np
import pytest

from opc_agent.metrics import (
    compute_binary_metrics,
    dequantize_displacement,
    quantize_displacement,
    weighted_opc_loss,
)


def test_binary_metrics_counts_nominal_and_process_window_errors():
    """一个边界缺失像素同时产生 L2、PVB、EPE N 和 EPE D。"""
    target = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    nominal = np.zeros((3, 3))
    maximum = target.copy()
    minimum = nominal.copy()
    metrics = compute_binary_metrics(target, nominal, maximum, minimum)
    assert (metrics.l2, metrics.pvb, metrics.epe_n, metrics.epe_d) == (1.0, 1.0, 1, 1.0)


@pytest.mark.parametrize("value, expected", [(-100, 0), (-14, 3), (4, 4), (39, 8)])
def test_displacement_quantization(value, expected):
    """位移钳制并映射到最近的分类中心。"""
    assert quantize_displacement(value) == expected
    assert -40 <= dequantize_displacement(expected) <= 40


def test_weighted_opc_loss_preserves_paper_coefficients_before_single_scale():
    """共享损失必须直接计算 L2+100*EPE+PVB，不能分别按指标初值改写相对权重。"""
    metrics = {"l2": 116184, "epe": 86, "pvb": 45874}
    weights = {"l2": 1, "epe": 100, "pvb": 1}
    assert weighted_opc_loss(metrics, weights) == 170658.0
