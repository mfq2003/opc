"""本模块验证确定性几何特征与训练分布外检测的基本行为。

输入为合成的二维图像和五维训练特征，输出为稳定特征字段和值域及 OOD 路由依据。
关键依赖为 NumPy 与 pytest，测试在纯 CPU 环境完成。
"""
import numpy as np

from opc_agent.features import FEATURE_NAMES, extract_geometry_features, feature_vector, is_out_of_distribution


def test_geometry_features_are_complete_and_deterministic():
    """简单矩形应生成固定的五项几何特征。"""
    image = np.zeros((4, 4), dtype=np.uint8)
    image[1:3, 1:3] = 1
    features = extract_geometry_features(image)
    assert tuple(features) == FEATURE_NAMES
    assert features["fill_ratio"] == 0.25
    assert feature_vector(features).shape == (5,)


def test_ood_detects_large_feature_shift():
    """远离常量训练簇的样本必须被路由到精算层。"""
    training = [[0, 0, 0, 0, 0], [0.01, 0.01, 0.01, 0.01, 0.01]]
    assert is_out_of_distribution([10, 0, 0, 0, 0], training)

