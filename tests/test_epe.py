"""本模块验证确定性逐点边界 EPE N/EPE D 的距离定义与异常输入处理。

输入为小型二值矩形、平移图形和错误形状图形；输出为零误差、正误差和明确异常断言。
关键依赖为 NumPy、SciPy 与 pytest；测试不调用 GPU、OpenILT、网络或 API。
"""
import numpy as np
import pytest

from opc_agent.epe import measure_boundary_epe


def test_identical_shapes_have_zero_epe():
    """完全相同的目标和印刷边界必须没有任何放置误差。"""
    image = np.zeros((8, 8), dtype=np.uint8)
    image[2:6, 2:6] = 1
    measurement = measure_boundary_epe(image, image)
    assert measurement.sample_count == 12
    assert measurement.epe_n == 0
    assert measurement.epe_d == 0


def test_shifted_shape_has_positive_distance_and_violations():
    """平移一个像素的矩形应有正距离和超过半像素容差的样本。"""
    target = np.zeros((10, 10), dtype=np.uint8)
    printed = np.zeros((10, 10), dtype=np.uint8)
    target[3:7, 3:7] = 1
    printed[3:7, 4:8] = 1
    measurement = measure_boundary_epe(target, printed, tolerance_pixels=0.5)
    assert measurement.epe_n > 0
    assert measurement.epe_d > 0
    assert measurement.mean_distance_pixels > 0


def test_epe_rejects_mismatched_shapes():
    """不同尺寸的图像不能产生没有几何意义的边界距离。"""
    with pytest.raises(ValueError, match="形状必须一致"):
        measure_boundary_epe(np.zeros((4, 4)), np.zeros((5, 5)))

