"""本模块固定 Recipe PPO v2 与 OpenILT ``dissect`` 之间的纯 CPU 几何适配契约。

测试使用内存正交矩形、二值 target 和 Fake dissect，不导入 OpenILT、CUDA、PyTorch 或
Gymnasium。覆盖内容包括：上游短下降边的逆向返回修复、父 edge 的完整连续覆盖、全局
最短段门槛、数据库坐标与法线探针的严格整数要求，以及会改变物理尺度或法线语义的输入
必须改变 fragmentation 和 EPE point identity。这里的通过只代表项目侧几何 contract，
不能替代配置锁定 OpenILT 提交上的十图集成、真实光刻 sensitivity 或 PPO 验收。
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np
import pytest

from opc_agent.recipe_v2 import dissect_global_fragments, outward_normal_from_target
from opc_agent.recipe_v2_contract import FragmentParameters


Polygon = Tuple[Tuple[int, int], ...]
Segment = Tuple[Tuple[int, int], Tuple[int, int]]


def _edges(polygon: Sequence[Sequence[int]]) -> Tuple[Segment, ...]:
    """按 polygon 遍历顺序返回闭合正交边。"""
    points = tuple(tuple(point) for point in polygon)
    return tuple(
        (points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )


def _rectangle() -> Tuple[Polygon, np.ndarray]:
    """建立边长均不小于默认 16 坐标门槛的矩形及其二值 target。"""
    polygon: Polygon = ((10, 10), (70, 10), (70, 30), (10, 30))
    target = np.zeros((80, 100), dtype=np.uint8)
    target[10:31, 10:71] = 1
    return polygon, target


def _fragment(
    polygon: Sequence[Sequence[int]],
    target: np.ndarray,
    dissect_fn: Callable[..., Sequence[Sequence[Sequence[int]]]],
    nm_per_coordinate: float = 1.0,
):
    """使用默认全局 FRAG 参数调用项目侧 adapter。"""
    return dissect_global_fragments(
        polygons=(polygon,),
        target_image=target,
        fragment_parameters=FragmentParameters(16, 32),
        nm_per_coordinate=nm_per_coordinate,
        dissect_fn=dissect_fn,
    )


def test_short_descending_edge_is_reoriented_to_parent_traversal() -> None:
    """上游短下降边即使反向返回，也必须恢复父 edge 顺序、ID 和外法线。"""
    polygon, target = _rectangle()
    source_edges = _edges(polygon)

    def upstream_short_edge_order(points, lenCorner, lenUniform):
        del points, lenCorner, lenUniform
        return (
            source_edges[0],
            source_edges[1],
            source_edges[2],
            (source_edges[3][1], source_edges[3][0]),
        )

    geometry = _fragment(polygon, target, upstream_short_edge_order)

    assert geometry.segments_by_polygon == (source_edges,)
    assert [point.source_edge_index for point in geometry.epe_points] == [0, 1, 2, 3]
    assert [point.segment_index for point in geometry.epe_points] == [0, 0, 0, 0]
    left = geometry.epe_points[3]
    assert (left.segment_start_xy, left.segment_end_xy) == source_edges[3]
    assert left.normal_xy == (-1, 0)


def test_missing_parent_edge_is_rejected() -> None:
    """dissect 漏掉任一原始 edge 时不得生成看似完整的 EPE 点集。"""
    polygon, target = _rectangle()
    source_edges = _edges(polygon)

    def missing_edge(points, lenCorner, lenUniform):
        del points, lenCorner, lenUniform
        return source_edges[:-1]

    with pytest.raises(ValueError, match="未覆盖"):
        _fragment(polygon, target, missing_edge)


@pytest.mark.parametrize(
    "top_segments",
    (
        (((10, 10), (35, 10)), ((45, 10), (70, 10))),
        (((10, 10), (45, 10)), ((35, 10), (70, 10))),
    ),
    ids=("gap", "overlap"),
)
def test_parent_edge_gap_or_overlap_is_rejected(
    top_segments: Tuple[Segment, Segment],
) -> None:
    """首尾看似覆盖时，中间的间隙或重叠仍必须显式失败。"""
    polygon, target = _rectangle()
    source_edges = _edges(polygon)

    def discontinuous_edge(points, lenCorner, lenUniform):
        del points, lenCorner, lenUniform
        return (*top_segments, *source_edges[1:])

    with pytest.raises(ValueError, match="间隙、重叠或乱序"):
        _fragment(polygon, target, discontinuous_edge)


def test_subminimum_short_edge_is_rejected() -> None:
    """上游短 edge 分支不能绕过 v2 的全局最短 segment 门槛。"""
    polygon: Polygon = ((10, 10), (50, 10), (50, 22), (10, 22))
    target = np.zeros((60, 80), dtype=np.uint8)
    target[10:23, 10:51] = 1

    def unsplit_edges(points, lenCorner, lenUniform):
        del lenCorner, lenUniform
        return _edges(points)

    with pytest.raises(ValueError, match="小于全局最小分段长度"):
        _fragment(polygon, target, unsplit_edges)


def test_fractional_polygon_coordinate_is_rejected_without_truncation() -> None:
    """小数 polygon 坐标不能被 ``int`` 静默截断后冒充上游整数 DBU。"""
    _, target = _rectangle()
    polygon = ((10, 10), (70.25, 10), (70, 30), (10, 30))

    def passthrough(points, lenCorner, lenUniform):
        del lenCorner, lenUniform
        return _edges(points)

    with pytest.raises(ValueError, match="整数"):
        _fragment(polygon, target, passthrough)  # type: ignore[arg-type]


def test_fractional_dissect_segment_coordinate_is_rejected_without_truncation() -> None:
    """小数 dissect 端点不能经截断后错误映射到合法父 edge。"""
    polygon, target = _rectangle()
    source_edges = _edges(polygon)

    def fractional_segment(points, lenCorner, lenUniform):
        del points, lenCorner, lenUniform
        return (
            ((10, 10), (70.25, 10)),
            *source_edges[1:],
        )

    with pytest.raises(ValueError, match="整数"):
        _fragment(polygon, target, fractional_segment)


def test_fractional_normal_probe_coordinate_is_rejected_without_truncation() -> None:
    """法线探针必须是精确正整数，不能把 2.5 静默当成 2。"""
    polygon, target = _rectangle()

    with pytest.raises(ValueError, match="正整数"):
        outward_normal_from_target(
            polygon[0],
            polygon[1],
            target,
            probe_coordinate=2.5,  # type: ignore[arg-type]
        )


def test_fragment_identity_binds_nm_per_coordinate() -> None:
    """相同坐标拓扑在不同物理 DBU 下必须具有不同 fragmentation 和 point identity。"""
    polygon, target = _rectangle()

    def passthrough(points, lenCorner, lenUniform):
        del lenCorner, lenUniform
        return _edges(points)

    one_nm = _fragment(polygon, target, passthrough, nm_per_coordinate=1.0)
    two_nm = _fragment(polygon, target, passthrough, nm_per_coordinate=2.0)

    assert one_nm.topology_sha256 == two_nm.topology_sha256
    assert one_nm.fragmentation_sha256 != two_nm.fragmentation_sha256
    assert {point.point_id for point in one_nm.epe_points}.isdisjoint(
        point.point_id for point in two_nm.epe_points
    )


def test_fragment_identity_binds_derived_normal_semantics() -> None:
    """同一 segment 拓扑若由 target 推导出不同法线，identity 也必须随之改变。"""
    polygon, inside_target = _rectangle()
    outside_target = np.asarray(1 - inside_target, dtype=np.uint8)

    def passthrough(points, lenCorner, lenUniform):
        del lenCorner, lenUniform
        return _edges(points)

    inside = _fragment(polygon, inside_target, passthrough)
    outside = _fragment(polygon, outside_target, passthrough)

    assert inside.topology_sha256 == outside.topology_sha256
    assert tuple(point.normal_xy for point in inside.epe_points) != tuple(
        point.normal_xy for point in outside.epe_points
    )
    assert inside.fragmentation_sha256 != outside.fragmentation_sha256
    assert {point.point_id for point in inside.epe_points}.isdisjoint(
        point.point_id for point in outside.epe_points
    )


def test_nonidentity_raster_mapping_is_rejected_until_transform_is_implemented() -> None:
    """非一一 raster/DBU 映射尚无坐标变换实现时必须提前失败。"""
    polygon, target = _rectangle()

    def passthrough(points, lenCorner, lenUniform):
        del lenCorner, lenUniform
        return _edges(points)

    with pytest.raises(ValueError, match="raster_scale=1"):
        dissect_global_fragments(
            (polygon,), target, FragmentParameters(), 1, passthrough, raster_scale=2
        )
    with pytest.raises(ValueError, match="raster_offset_xy"):
        dissect_global_fragments(
            (polygon,),
            target,
            FragmentParameters(),
            1,
            passthrough,
            raster_offset_xy=(1, 0),
        )


def test_normal_probe_parameter_is_bound_into_fragment_identity() -> None:
    """即使两种采样距离得到同一法线，也不能复用旧 fragmentation/point ID。"""
    polygon, target = _rectangle()

    def passthrough(points, lenCorner, lenUniform):
        del lenCorner, lenUniform
        return _edges(points)

    near = dissect_global_fragments(
        (polygon,), target, FragmentParameters(), 1, passthrough, normal_probe_coordinate=1
    )
    far = dissect_global_fragments(
        (polygon,), target, FragmentParameters(), 1, passthrough, normal_probe_coordinate=2
    )
    assert near.topology_sha256 == far.topology_sha256
    assert tuple(point.normal_xy for point in near.epe_points) == tuple(
        point.normal_xy for point in far.epe_points
    )
    assert near.fragmentation_sha256 != far.fragmentation_sha256
    assert {point.point_id for point in near.epe_points}.isdisjoint(
        point.point_id for point in far.epe_points
    )
