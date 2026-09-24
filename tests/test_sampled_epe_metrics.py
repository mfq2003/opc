"""本模块验证 Recipe v2 点身份门禁与 1 nm EPE N/EPE D 统计。

输入为小型矩形 target、平移后的 printed、伪造 v2 segment 中点和最小 result 记录；输出为
严格大于阈值的违规计数、只累加违规距离的 EPE D，以及点集漂移时的明确失败。测试不调用
GPU、OpenILT、Solver、网络或 API。
"""
from types import SimpleNamespace

import numpy as np
import pytest

from opc_agent.sampled_epe_metrics import (
    _summarize,
    _v2_protocol,
    _validate_layout_coverage,
    _validate_rebuilt_points,
    measure_v2_points,
)


def _point(point_id, x, y, normal):
    """构造只含评价所需字段的轻量 v2 EPE 点。"""
    return SimpleNamespace(point_id=point_id, base_xy=(x, y), normal_xy=normal)


def test_epe_d_only_sums_v2_point_distances_strictly_over_tolerance():
    """距离等于或低于 1 nm 不违规，EPE D 只累加大于 1 nm 的完整距离。"""
    target = np.zeros((20, 20), dtype=np.uint8)
    printed = np.zeros((20, 20), dtype=np.uint8)
    target[4:16, 4:16] = 1
    printed[4:16, 6:18] = 1
    points = [
        _point("left-v2", 4, 10, (-1, 0)),
        _point("top-v2", 10, 4, (0, -1)),
    ]
    measured = measure_v2_points(
        target, printed, points, scale_nm_per_pixel=1, tolerance_nm=1
    )
    summary = _summarize(measured)
    assert measured[0].distance_nm == 2
    assert measured[0].violation is True
    assert measured[1].distance_nm == 0
    assert measured[1].violation is False
    assert summary.sample_count == 2
    assert summary.epe_n == 1
    assert summary.epe_d_nm == 2


def test_rebuilt_v2_points_must_match_complete_result_identity():
    """重建点必须与完整动作和搜索顺序集合一致，并要求最终独立回放通过。"""
    points = [_point("p0", 1, 1, (-1, 0)), _point("p1", 2, 1, (1, 0))]
    result = {
        "best": {"actions": {"p0": 2, "p1": 3}, "recipe_sha256": "a" * 64},
        "point_order": ["p1", "p0"],
        "final_replay_equal": True,
    }
    assert _validate_rebuilt_points(points, result) == "a" * 64
    result["best"]["actions"].pop("p1")
    with pytest.raises(RuntimeError, match="重建点与 result 动作点不一致"):
        _validate_rebuilt_points(points, result)


def test_v2_protocol_requires_explicit_fragment_and_geometry_adapter():
    """统计协议必须来自冻结 v2 配置，不能使用旧 v1 分段默认值。"""
    config = {"recipe_v2": {
        "nm_per_coordinate": 1.0,
        "fragment_parameters_nm": {"corner": 16, "uniform": 32},
        "geometry_adapter": {"version": "openilt-dissect-parent-edge-adapter-v2"},
    }}
    scale, fragment, geometry = _v2_protocol(config)
    assert scale == 1.0
    assert fragment == {"corner": 16, "uniform": 32}
    assert geometry["version"] == "openilt-dissect-parent-edge-adapter-v2"


def test_layout_coverage_rejects_partial_download():
    """本地缺少配置声明的任一版图时必须失败，不能输出部分统计。"""
    with pytest.raises(RuntimeError, match="missing=.*M1_test2"):
        _validate_layout_coverage(["M1_test1"], ["M1_test1", "M1_test2"])
