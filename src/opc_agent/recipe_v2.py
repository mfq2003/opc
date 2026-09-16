"""本模块实现 Recipe PPO v2 的 CPU 几何、控制点适配、冻结 observation 与回报协议核心。

输入是全局 FRAG 分段、原始 target 上的 EPE 点、逐点法向动作、可替换 solver 和独立 Golden
evaluator；输出是合法性证据、不可随 prefix 变化的 Actor observation、dense/terminal episode 轨迹
及 final-only Recipe 数据。真实 mask 移动仍由注入的 solver 负责，本模块不导入或修改 OpenILT。
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np

from .recipe_v2_contract import (
    ACTOR_GEOMETRY_FIELDS,
    DEFAULT_CONTROL_PROBE_DISTANCE_NM,
    DEFAULT_GAE_LAMBDA,
    DEFAULT_GAMMA,
    DEFAULT_TRAINING_REWARD_SCALE,
    EPE_ACTION_OFFSETS_NM,
    EPE_DENSE_PROTOCOL,
    EPE_TERMINAL_PROTOCOL,
    EPE_TRAINING_PROTOCOLS,
    ActionCandidate,
    EPEControlPoint,
    FragmentParameters,
    GoldenControlPoint,
    GoldenEvaluation,
    GoldenMetrics,
    GoldenPointSet,
    ProbeWindow,
    RECIPE_V2_ACCEPTED_METRICS_SOURCE,
    RECIPE_V2_ACTION_SEMANTICS,
    RECIPE_V2_ACTOR_STATE,
    RECIPE_V2_ENV_VERSION,
    RECIPE_V2_LABEL_VERSION,
    RECIPE_V2_LOSS_VERSION,
    RECIPE_V2_POINT_VERSION,
    RECIPE_V2_SCHEMA_VERSION,
    V2SolverResult,
    array_sha256,
    observation_version,
    require_sha256,
)


GEOMETRY_ADAPTER_VERSION = "openilt-dissect-parent-edge-adapter-v2"
RASTER_MAPPING_VERSION = "db-coordinate-equals-raster-pixel-v1"
NORMAL_PROBE_SEMANTICS_VERSION = "target-two-sided-axis-probe-v1"
UPSTREAM_MIN_FRAGMENT_RULE_VERSION = "min-corner-uniform-coordinate-v1"
DEFAULT_NORMAL_PROBE_COORDINATE = 2
GOLDEN_POINT_SET_VERSION = "frozen-target-boundary-points-v1"


class LocalEPESolver(Protocol):
    """定义 v2 环境所需的最小 solver 接口；FRAG 在实例生命周期内固定。"""

    epe_points: Sequence[EPEControlPoint]
    fragment_parameters: FragmentParameters
    target_image: np.ndarray
    nm_per_coordinate: float
    layout_sha256: str
    revision: str

    def solve(self, normal_offsets_nm: Mapping[str, float]) -> V2SolverResult:
        """以完整 point_id->法向位移 Recipe 运行内部 mask OPC。"""


class GoldenEvaluator(Protocol):
    """定义与可移动 Recipe 控制点分离的固定验收入口。"""

    evaluator_version: str
    evaluator_source_sha256: str
    frozen_target_sha256: str
    sampling_state_sha256: str
    nm_per_coordinate: float
    coordinate_system_sha256: str
    evaluator_contract_sha256: str

    def evaluate(self, result: V2SolverResult) -> GoldenEvaluation:
        """只从固定 evaluator 状态和 solver 栅格结果计算 Golden 指标。"""


def _sha256_json(payload: object) -> str:
    """对规范 JSON 计算稳定 SHA256。"""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_arrays(*arrays: np.ndarray) -> str:
    """连同 dtype/shape 对多个 NumPy 数组计算稳定 SHA256。"""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _binary_image(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """把二维图像冻结为布尔数组，并显式拒绝非法形状和数值。"""
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("target/printed 图像必须是二维数组")
    if not np.all(np.isfinite(array)):
        raise ValueError("target/printed 图像不能包含 NaN 或无穷值")
    return np.asarray(array >= float(threshold), dtype=bool)


def _integer_coordinate(value: object, name: str) -> int:
    """严格接收整数数据库坐标，禁止 bool 或浮点数经 ``int`` 静默截断。"""
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是整数数据库坐标")
    return int(value)


def _integer_pair(value: Sequence[object], name: str) -> Tuple[int, int]:
    """严格规范化二维数据库坐标。"""
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} 必须是二维整数坐标") from exc
    if len(items) != 2:
        raise ValueError(f"{name} 必须是二维整数坐标")
    return (
        _integer_coordinate(items[0], name),
        _integer_coordinate(items[1], name),
    )


def _discrete_action_index(action: object, action_count: int, name: str = "action") -> int:
    """只接受标量整数离散动作，拒绝 1.9、bool 和非标量数组。"""
    array = np.asarray(action)
    if array.shape != ():
        raise TypeError(f"{name} 必须是单个整数离散编号")
    value = array.item()
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} 必须是单个整数离散编号")
    index = int(value)
    if not 0 <= index < int(action_count):
        raise ValueError(f"{name} 必须是动作表内的单个离散编号")
    return index


def _nm_to_coordinate(value_nm: float, nm_per_coordinate: float, name: str) -> int:
    """把纳米值无歧义地量化为整数数据库坐标。"""
    scale = float(nm_per_coordinate)
    value = float(value_nm)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("nm_per_coordinate 必须是有限正数")
    if not np.isfinite(value):
        raise ValueError(f"{name} 必须是有限数")
    coordinate = int(np.rint(value / scale))
    if coordinate <= 0:
        raise ValueError(f"{name} 量化后必须至少为一个数据库坐标")
    realized = coordinate * scale
    if not np.isclose(realized, value, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"{name}={value}nm 不能被 nm_per_coordinate={scale} 无损表示；"
            "禁止把全局分段/probe 参数静默四舍五入"
        )
    return coordinate


def build_probe_window(
    point: Union[EPEControlPoint, GoldenControlPoint],
    normal_offset_nm: float,
    probe_distance_nm: float,
    nm_per_coordinate: float,
    target_image: np.ndarray,
) -> ProbeWindow:
    """构造 q=p+delta*n 两侧的固定 probe，并在任何数组索引前完成边界检查。"""
    target = _binary_image(target_image)
    moved = point.moved_xy(normal_offset_nm, nm_per_coordinate)
    probe_coordinate = _nm_to_coordinate(
        probe_distance_nm,
        nm_per_coordinate,
        "probe_distance_nm",
    )
    normal_x, normal_y = point.normal_xy
    inner = (
        int(moved[0] - normal_x * probe_coordinate),
        int(moved[1] - normal_y * probe_coordinate),
    )
    outer = (
        int(moved[0] + normal_x * probe_coordinate),
        int(moved[1] + normal_y * probe_coordinate),
    )
    height, width = target.shape
    coordinates = (moved, inner, outer)
    in_bounds = all(0 <= x < width and 0 <= y < height for x, y in coordinates)
    inner_valid = False
    outer_valid = False
    if in_bounds:
        inner_valid = bool(target[inner[1], inner[0]])
        outer_valid = bool(not target[outer[1], outer[0]])
    delta_coordinate = (
        (moved[0] - point.base_xy[0]) * normal_x
        + (moved[1] - point.base_xy[1]) * normal_y
    )
    realized = float(delta_coordinate * float(nm_per_coordinate))
    return ProbeWindow(
        point_id=point.point_id,
        normal_offset_nm=float(normal_offset_nm),
        realized_normal_offset_nm=realized,
        quantization_error_nm=float(realized - float(normal_offset_nm)),
        moved_xy=moved,
        inner_xy=inner,
        outer_xy=outer,
        in_bounds=in_bounds,
        inner_target_valid=inner_valid,
        outer_target_valid=outer_valid,
    )


def build_action_candidates(
    point: EPEControlPoint,
    action_offsets_nm: Sequence[float],
    probe_distance_nm: float,
    nm_per_coordinate: float,
    target_image: np.ndarray,
) -> Tuple[ActionCandidate, ...]:
    """生成动作表，量化重合者标为 alias，target/probe 无效者保留证据但不可执行。"""
    offsets = tuple(float(value) for value in action_offsets_nm)
    if not offsets or len(offsets) != len(set(offsets)):
        raise ValueError("EPE 动作表必须是非空且无重复的有限数序列")
    if any(not np.isfinite(value) for value in offsets):
        raise ValueError("EPE 动作表不能包含 NaN 或无穷值")
    seen: Dict[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]], int] = {}
    candidates: List[ActionCandidate] = []
    for action_class, offset in enumerate(offsets):
        probe = build_probe_window(
            point=point,
            normal_offset_nm=offset,
            probe_distance_nm=probe_distance_nm,
            nm_per_coordinate=nm_per_coordinate,
            target_image=target_image,
        )
        key = (probe.moved_xy, probe.inner_xy, probe.outer_xy)
        alias = seen.get(key)
        if alias is None:
            seen[key] = action_class
        candidates.append(ActionCandidate(
            action_class=action_class,
            probe=probe,
            alias_of_action_class=alias,
        ))
    return tuple(candidates)


def candidate_action_mask(candidates: Sequence[ActionCandidate]) -> np.ndarray:
    """返回可供 preflight/MaskablePPO 使用的布尔动作掩码。"""
    values = tuple(candidates)
    if not values:
        raise ValueError("候选动作不能为空")
    expected = list(range(len(values)))
    if [candidate.action_class for candidate in values] != expected:
        raise ValueError("候选动作必须按连续 action_class 排序")
    return np.asarray([candidate.valid for candidate in values], dtype=bool)


def require_all_actions_valid_for_plain_ppo(
    point_candidates: Mapping[str, Sequence[ActionCandidate]],
) -> None:
    """在普通 PPO 没有真正 action mask 时，拒绝任何含非法候选的训练配置。"""
    failures = {}
    for point_id, candidates in point_candidates.items():
        invalid = [
            candidate.action_class
            for candidate in candidates
            if not candidate.valid
        ]
        if invalid:
            failures[str(point_id)] = invalid
    if failures:
        raise ValueError(
            "普通 stable-baselines3 PPO 不会应用 action mask；当前动作/probe 组合存在非法类，"
            f"必须先缩小动作范围、扩大并验证 probe，或引入真正的 masked policy：{failures}"
        )


def control_move_signs(
    points: Sequence[EPEControlPoint],
    normal_offsets_nm: Mapping[str, float],
    printed_image: np.ndarray,
    target_image: np.ndarray,
    probe_distance_nm: float,
    nm_per_coordinate: float,
) -> Tuple[Tuple[str, float], ...]:
    """以移动 crossing 的固定 probe 产生 solver 法向方向；双侧同时违规时显式失败。"""
    printed = _binary_image(printed_image)
    target = _binary_image(target_image)
    if printed.shape != target.shape:
        raise ValueError("printed 与 target 形状必须一致")
    point_ids = [point.point_id for point in points]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("EPE point_id 不允许重复")
    if set(normal_offsets_nm) != set(point_ids):
        raise ValueError("normal_offsets_nm 必须恰好覆盖全部 EPE point_id")
    signs = []
    for point in points:
        probe = build_probe_window(
            point,
            normal_offsets_nm[point.point_id],
            probe_distance_nm,
            nm_per_coordinate,
            target,
        )
        if not probe.target_valid:
            raise ValueError(
                f"EPE 点 {point.point_id} 的移动 crossing 不能形成一内一外的有效 probe"
            )
        underprint = not bool(printed[probe.inner_xy[1], probe.inner_xy[0]])
        overprint = bool(printed[probe.outer_xy[1], probe.outer_xy[0]])
        if underprint and overprint:
            raise ValueError(
                f"EPE 点 {point.point_id} 的 inner/outer 同时违规，单一移动方向不再唯一"
            )
        sign = 1.0 if underprint else (-1.0 if overprint else 0.0)
        signs.append((point.point_id, sign))
    return tuple(signs)


def outward_normal_from_target(
    segment_start_xy: Tuple[int, int],
    segment_end_xy: Tuple[int, int],
    target_image: np.ndarray,
    probe_coordinate: int = 2,
) -> Tuple[int, int]:
    """通过 target 栅格两侧采样判定外法线，结果不依赖 polygon winding。"""
    start = _integer_pair(segment_start_xy, "segment_start_xy")
    end = _integer_pair(segment_end_xy, "segment_end_xy")
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if (dx == 0) == (dy == 0):
        raise ValueError("外法线只支持非零正交 segment")
    distance = _integer_coordinate(probe_coordinate, "normal probe_coordinate 正整数")
    if distance <= 0:
        raise ValueError("normal probe_coordinate 必须为正整数")
    tangent = (
        0 if dx == 0 else (1 if dx > 0 else -1),
        0 if dy == 0 else (1 if dy > 0 else -1),
    )
    first = (tangent[1], -tangent[0])
    second = (-first[0], -first[1])
    middle = (int(np.rint((start[0] + end[0]) / 2)), int(np.rint((start[1] + end[1]) / 2)))
    target = _binary_image(target_image)
    height, width = target.shape
    samples = (
        (middle[0] + first[0] * distance, middle[1] + first[1] * distance),
        (middle[0] + second[0] * distance, middle[1] + second[1] * distance),
    )
    if any(not (0 <= x < width and 0 <= y < height) for x, y in samples):
        raise ValueError("target 边法向探针越出图像")
    inside = tuple(bool(target[y, x]) for x, y in samples)
    if inside[0] == inside[1]:
        raise ValueError("无法从 target 两侧唯一确定 segment 外法线")
    return second if inside[0] else first


def _corner_types(polygon: Sequence[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    """以 turn 与 signed area 的相对符号计算凹凸角，反转 winding 后类型保持不变。"""
    points = tuple(_integer_pair(point, "polygon 顶点") for point in polygon)
    if len(points) < 4 or len(set(points)) != len(points):
        raise ValueError("polygon 至少四个不重复顶点，且不能重复闭合点")
    area_twice = sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )
    if area_twice == 0:
        raise ValueError("polygon signed area 不能为零")
    result = {}
    for index, current in enumerate(points):
        previous = points[index - 1]
        following = points[(index + 1) % len(points)]
        incoming = (current[0] - previous[0], current[1] - previous[1])
        outgoing = (following[0] - current[0], following[1] - current[1])
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if cross == 0:
            result[current] = 0
        else:
            result[current] = 1 if cross * area_twice > 0 else -1
    return result


def _segment_on_edge(
    segment: Tuple[Tuple[int, int], Tuple[int, int]],
    edge: Tuple[Tuple[int, int], Tuple[int, int]],
) -> bool:
    """判断一个非零正交 segment 是否完整位于原始 target edge 上。"""
    (sx, sy), (ex, ey) = segment
    (ax, ay), (bx, by) = edge
    if ax == bx:
        return sx == ex == ax and min(ay, by) <= min(sy, ey) <= max(sy, ey) <= max(ay, by)
    if ay == by:
        return sy == ey == ay and min(ax, bx) <= min(sx, ex) <= max(sx, ex) <= max(ax, bx)
    raise ValueError("原始 target edge 必须正交")


def _orient_like_edge(
    segment: Tuple[Tuple[int, int], Tuple[int, int]],
    edge: Tuple[Tuple[int, int], Tuple[int, int]],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """统一 segment 方向，使其与所属原始 edge 的遍历方向一致。"""
    start, end = segment
    edge_start, edge_end = edge
    edge_vector = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])
    segment_vector = (end[0] - start[0], end[1] - start[1])
    return segment if edge_vector[0] * segment_vector[0] + edge_vector[1] * segment_vector[1] > 0 else (end, start)


@dataclass(frozen=True)
class FragmentedGeometry:
    """保存全局 FRAG 参数生成的稳定分段、EPE 点和拓扑哈希。"""

    fragment_parameters: FragmentParameters
    segments_by_polygon: Tuple[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...], ...]
    epe_points: Tuple[EPEControlPoint, ...]
    topology_sha256: str
    fragmentation_sha256: str


def dissect_global_fragments(
    polygons: Sequence[Sequence[Tuple[int, int]]],
    target_image: np.ndarray,
    fragment_parameters: FragmentParameters,
    nm_per_coordinate: float,
    dissect_fn: Callable[..., Sequence[Sequence[Sequence[int]]]],
    normal_probe_coordinate: int = DEFAULT_NORMAL_PROBE_COORDINATE,
    raster_scale: float = 1.0,
    raster_offset_xy: Tuple[int, int] = (0, 0),
) -> FragmentedGeometry:
    """把两个全局长度传给每次 dissect，并从返回 segment 构造全新的 EPE point ID。

    当前 adapter 只冻结了数据库坐标与 target raster 像素一一对应、无偏移的映射。若上游使用
    其他缩放或原点，必须先实现并测试显式坐标变换，不能继续沿用当前法线采样。
    """
    if not callable(dissect_fn):
        raise TypeError("dissect_fn 必须可调用")
    normal_probe_coordinate = _integer_coordinate(
        normal_probe_coordinate,
        "normal_probe_coordinate 正整数",
    )
    if normal_probe_coordinate <= 0:
        raise ValueError("normal_probe_coordinate 必须为正整数")
    if not np.isfinite(float(raster_scale)) or float(raster_scale) != 1.0:
        raise ValueError("当前 v2 adapter 只支持 raster_scale=1 的一一坐标映射")
    raster_offset = _integer_pair(raster_offset_xy, "raster_offset_xy")
    if raster_offset != (0, 0):
        raise ValueError("当前 v2 adapter 只支持 raster_offset_xy=(0,0)")
    target = _binary_image(target_image)
    corner_coordinate = _nm_to_coordinate(
        fragment_parameters.corner_length_nm,
        nm_per_coordinate,
        "corner_length_nm",
    )
    uniform_coordinate = _nm_to_coordinate(
        fragment_parameters.uniform_length_nm,
        nm_per_coordinate,
        "uniform_length_nm",
    )
    normalized_polygons = tuple(
        tuple(_integer_pair(point, "polygon 顶点") for point in polygon)
        for polygon in polygons
    )
    if not normalized_polygons:
        raise ValueError("至少需要一个 target polygon")
    grouped: List[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...]] = []
    metadata = []
    for polygon_index, polygon in enumerate(normalized_polygons):
        corner_lookup = _corner_types(polygon)
        edges = tuple(
            (polygon[index], polygon[(index + 1) % len(polygon)])
            for index in range(len(polygon))
        )
        for edge in edges:
            if (edge[0][0] == edge[1][0]) == (edge[0][1] == edge[1][1]):
                raise ValueError("target polygon 只能包含非零正交 edge")
        raw = dissect_fn(
            [list(point) for point in polygon],
            lenCorner=corner_coordinate,
            lenUniform=uniform_coordinate,
        )
        by_edge: Dict[int, List[Tuple[Tuple[int, int], Tuple[int, int]]]] = {
            index: [] for index in range(len(edges))
        }
        for raw_segment in raw:
            if len(raw_segment) != 2:
                raise ValueError("dissect 必须返回两端点 segment")
            segment = tuple(_integer_pair(point, "dissect segment 端点") for point in raw_segment)
            if len(segment[0]) != 2 or len(segment[1]) != 2 or segment[0] == segment[1]:
                raise ValueError("dissect 返回了非法或零长 segment")
            parents = [edge_index for edge_index, edge in enumerate(edges) if _segment_on_edge(segment, edge)]
            if len(parents) != 1:
                raise ValueError("dissect segment 无法唯一映射到原始 target edge")
            edge_index = parents[0]
            by_edge[edge_index].append(_orient_like_edge(segment, edges[edge_index]))
        polygon_segments = []
        for edge_index, edge in enumerate(edges):
            edge_start, edge_end = edge
            tangent = (
                0 if edge_start[0] == edge_end[0] else (1 if edge_end[0] > edge_start[0] else -1),
                0 if edge_start[1] == edge_end[1] else (1 if edge_end[1] > edge_start[1] else -1),
            )
            segments = by_edge[edge_index]
            if not segments:
                raise ValueError(f"dissect 未覆盖 polygon {polygon_index} edge {edge_index}")
            segments.sort(key=lambda segment: (
                (segment[0][0] - edge_start[0]) * tangent[0]
                + (segment[0][1] - edge_start[1]) * tangent[1]
            ))
            if segments[0][0] != edge_start or segments[-1][1] != edge_end:
                raise ValueError("dissect segment 没有完整覆盖所属原始 edge")
            for previous, following in zip(segments[:-1], segments[1:]):
                if previous[1] != following[0]:
                    raise ValueError("dissect segment 在原始 edge 上存在间隙、重叠或乱序")
            minimum_coordinate = min(corner_coordinate, uniform_coordinate)
            for segment_index, segment in enumerate(segments):
                length = abs(segment[1][0] - segment[0][0]) + abs(segment[1][1] - segment[0][1])
                if length < minimum_coordinate:
                    raise ValueError(
                        "dissect 生成了小于全局最小分段长度的 segment；"
                        "短 edge 也不能绕过 v2 合法性门槛"
                    )
                polygon_segments.append(segment)
                metadata.append((
                    polygon_index,
                    edge_index,
                    segment_index,
                    segment,
                    corner_lookup.get(segment[0], 0),
                    corner_lookup.get(segment[1], 0),
                ))
        if not polygon_segments:
            raise ValueError("dissect 未生成任何 segment")
        grouped.append(tuple(polygon_segments))
    topology_payload = {
        "segments": [
            [[list(start), list(end)] for start, end in segments]
            for segments in grouped
        ],
    }
    topology_sha256 = _sha256_json(topology_payload)
    point_geometry = []
    for polygon_index, edge_index, segment_index, segment, start_kind, end_kind in metadata:
        start, end = segment
        base = (int(np.rint((start[0] + end[0]) / 2)), int(np.rint((start[1] + end[1]) / 2)))
        normal = outward_normal_from_target(
            start,
            end,
            target,
            probe_coordinate=normal_probe_coordinate,
        )
        point_geometry.append({
            "polygon_index": polygon_index,
            "source_edge_index": edge_index,
            "segment_index": segment_index,
            "segment_start_xy": list(start),
            "segment_end_xy": list(end),
            "base_xy": list(base),
            "normal_xy": list(normal),
            "start_corner_type": start_kind,
            "end_corner_type": end_kind,
        })
    fragmentation_sha256 = _sha256_json({
        "adapter_version": GEOMETRY_ADAPTER_VERSION,
        "fragment_parameters": fragment_parameters.as_dict(),
        "nm_per_coordinate": float(nm_per_coordinate),
        "topology_sha256": topology_sha256,
        "target_sha256": _sha256_arrays(target),
        "raster_mapping_version": RASTER_MAPPING_VERSION,
        "raster_scale": 1.0,
        "raster_offset_xy": [0, 0],
        "normal_semantics_version": NORMAL_PROBE_SEMANTICS_VERSION,
        "normal_probe_coordinate": normal_probe_coordinate,
        "minimum_fragment_rule_version": UPSTREAM_MIN_FRAGMENT_RULE_VERSION,
        "point_geometry": point_geometry,
    })
    points = []
    for record in point_geometry:
        points.append(EPEControlPoint(
            point_id=(
                f"polygon-{record['polygon_index']}-edge-{record['source_edge_index']}"
                f"-segment-{record['segment_index']}-fragmentation-{fragmentation_sha256}"
            ),
            polygon_index=record["polygon_index"],
            source_edge_index=record["source_edge_index"],
            segment_index=record["segment_index"],
            base_xy=tuple(record["base_xy"]),
            segment_start_xy=tuple(record["segment_start_xy"]),
            segment_end_xy=tuple(record["segment_end_xy"]),
            normal_xy=tuple(record["normal_xy"]),
            start_corner_type=record["start_corner_type"],
            end_corner_type=record["end_corner_type"],
        ))
    point_ids = [point.point_id for point in points]
    if len(point_ids) != len(set(point_ids)):
        raise RuntimeError("FRAG 分段生成了重复 EPE point_id")
    return FragmentedGeometry(
        fragment_parameters=fragment_parameters,
        segments_by_polygon=tuple(grouped),
        epe_points=tuple(points),
        topology_sha256=topology_sha256,
        fragmentation_sha256=fragmentation_sha256,
    )


def build_golden_point_set(
    target_image: np.ndarray,
    points: Sequence[GoldenControlPoint],
    nm_per_coordinate: float,
    source_sha256: str,
    version: str = GOLDEN_POINT_SET_VERSION,
    threshold: float = 0.5,
) -> GoldenPointSet:
    """在 FRAG 之前从冻结原始 target 构造独立 Golden 点集身份。"""
    target = _binary_image(target_image, threshold)
    return GoldenPointSet(
        version=str(version),
        frozen_target_sha256=_sha256_arrays(target),
        nm_per_coordinate=float(nm_per_coordinate),
        source_sha256=require_sha256(source_sha256, "Golden point-set source_sha256"),
        points=tuple(points),
    )


def _identity_raster_coordinate_system_sha256(
    frozen_target_sha256: str,
    nm_per_coordinate: float,
) -> str:
    """返回当前一一 DBU/raster 映射的冻结坐标系统身份。"""
    scale = float(nm_per_coordinate)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("nm_per_coordinate 必须是有限正数")
    return _sha256_json({
        "frozen_target_sha256": require_sha256(
            frozen_target_sha256,
            "coordinate frozen_target_sha256",
        ),
        "nm_per_coordinate": scale,
        "raster_mapping_version": RASTER_MAPPING_VERSION,
        "raster_scale": 1.0,
        "raster_offset_xy": [0, 0],
    })


def identity_raster_coordinate_system_sha256(
    frozen_target_sha256: str,
    nm_per_coordinate: float,
) -> str:
    """公开返回 v2 当前冻结的一一 DBU/raster 坐标系统身份。"""
    return _identity_raster_coordinate_system_sha256(
        frozen_target_sha256,
        nm_per_coordinate,
    )


class FixedProbeGoldenEvaluator:
    """提供 CPU/Fake-solver 用固定 Golden probe evaluator；不冒充 OpenILT 正式 epecheck。"""

    evaluator_version = "fixed-target-probe-golden-diagnostic-v1"

    @property
    def frozen_target_sha256(self) -> str:
        """返回冻结 Golden target 内容哈希。"""
        return self._frozen_target_sha256

    @property
    def nm_per_coordinate(self) -> float:
        """返回 Golden 点集冻结的物理坐标单位。"""
        return self._nm_per_coordinate

    @property
    def coordinate_system_sha256(self) -> str:
        """返回 Golden 点集的 raster/DBU 坐标系统哈希。"""
        return self._coordinate_system_sha256

    @property
    def evaluator_source_sha256(self) -> str:
        """返回 evaluator 实现来源哈希。"""
        return self._evaluator_source_sha256

    @property
    def golden_point_set(self) -> GoldenPointSet:
        """返回类型上独立于 FRAG 的冻结 Golden 点集。"""
        return self._golden_point_set

    @property
    def sampling_state_sha256(self) -> str:
        """返回固定 Golden probe 采样状态哈希。"""
        return self._sampling_state_sha256

    @property
    def evaluator_contract_sha256(self) -> str:
        """返回 evaluator 来源、target、采样和参数的总 contract 哈希。"""
        return self._evaluator_contract_sha256

    def __init__(
        self,
        target_image: np.ndarray,
        golden_point_set: GoldenPointSet,
        reward_weights: Mapping[str, float],
        diagnostic_probe_distance_nm: float,
        evaluator_source_sha256: str,
        threshold: float = 0.5,
    ):
        if not isinstance(golden_point_set, GoldenPointSet):
            raise TypeError("golden_point_set 必须是独立于 FRAG 的 GoldenPointSet")
        self._target = _binary_image(target_image, threshold)
        self._frozen_target_sha256 = _sha256_arrays(self._target)
        if golden_point_set.frozen_target_sha256 != self.frozen_target_sha256:
            raise ValueError("GoldenPointSet 与 evaluator 冻结 target 不一致")
        self._weights = {name: float(value) for name, value in reward_weights.items()}
        GoldenMetrics(0, 0, 0).weighted_loss(self._weights)
        self._threshold = float(threshold)
        if not np.isfinite(self._threshold) or not 0.0 <= self._threshold <= 1.0:
            raise ValueError("Golden threshold 必须是 [0,1] 内有限数")
        self._diagnostic_probe_distance_nm = float(diagnostic_probe_distance_nm)
        self._nm_per_coordinate = float(golden_point_set.nm_per_coordinate)
        self._coordinate_system_sha256 = _identity_raster_coordinate_system_sha256(
            self.frozen_target_sha256,
            self.nm_per_coordinate,
        )
        self._evaluator_source_sha256 = require_sha256(
            evaluator_source_sha256,
            "Golden evaluator source_sha256",
        )
        self._golden_point_set = golden_point_set
        probes = []
        for point in golden_point_set.points:
            probe = build_probe_window(
                point,
                normal_offset_nm=0.0,
                probe_distance_nm=diagnostic_probe_distance_nm,
                nm_per_coordinate=golden_point_set.nm_per_coordinate,
                target_image=self._target,
            )
            if not probe.target_valid:
                raise ValueError(f"Golden 点 {point.point_id} 不能形成固定的一内一外 probe")
            probes.append((point.point_id, probe.inner_xy, probe.outer_xy))
        if not probes:
            raise ValueError("Golden evaluator 至少需要一个固定点")
        self._probes = tuple(sorted(probes))
        self._sampling_state_sha256 = _sha256_json({
            "point_set_sha256": golden_point_set.point_set_sha256,
            "diagnostic_probe_distance_nm": self._diagnostic_probe_distance_nm,
            "probes": [
                {
                    "point_id": point_id,
                    "inner_xy": list(inner),
                    "outer_xy": list(outer),
                }
                for point_id, inner, outer in self._probes
            ],
        })
        self._evaluator_contract_sha256 = _sha256_json({
            "evaluator_version": self.evaluator_version,
            "evaluator_source_sha256": self.evaluator_source_sha256,
            "frozen_target_sha256": self.frozen_target_sha256,
            "sampling_state_sha256": self.sampling_state_sha256,
            "coordinate_system_sha256": self.coordinate_system_sha256,
            "parameters": {
                "diagnostic_probe_distance_nm": self._diagnostic_probe_distance_nm,
                "nm_per_coordinate": self.nm_per_coordinate,
                "raster_mapping_version": RASTER_MAPPING_VERSION,
                "raster_scale": 1.0,
                "raster_offset_xy": [0, 0],
                "threshold": self._threshold,
                "formal_openilt_epecheck": False,
                "reward_weights": dict(sorted(self._weights.items())),
            },
        })

    def evaluate(self, result: V2SolverResult) -> GoldenEvaluation:
        """使用冻结 target/probe 评分；接口无法读取 normal_offsets_nm 或 moved_xy。"""
        nominal = _binary_image(result.printed_nominal, self._threshold)
        maximum = _binary_image(result.printed_max, self._threshold)
        minimum = _binary_image(result.printed_min, self._threshold)
        if nominal.shape != self._target.shape or maximum.shape != self._target.shape or minimum.shape != self._target.shape:
            raise ValueError("solver printed 图必须与冻结 Golden target 同形状")
        l2 = float(np.count_nonzero(nominal != self._target))
        pvb = float(np.count_nonzero(maximum != minimum))
        epe = 0
        for _point_id, inner, outer in self._probes:
            epe += int(not nominal[inner[1], inner[0]])
            epe += int(nominal[outer[1], outer[0]])
        metrics = GoldenMetrics(l2=l2, epe=float(epe), pvb=pvb)
        return GoldenEvaluation(
            metrics=metrics,
            raw_weighted_loss=metrics.weighted_loss(self._weights),
            evaluator_version=self.evaluator_version,
            evaluator_source_sha256=self.evaluator_source_sha256,
            evaluator_contract_sha256=self.evaluator_contract_sha256,
            evaluator_parameters=(
                ("diagnostic_probe_distance_nm", self._diagnostic_probe_distance_nm),
                ("nm_per_coordinate", self.nm_per_coordinate),
                ("raster_mapping_version", RASTER_MAPPING_VERSION),
                ("raster_scale", 1.0),
                ("raster_offset_xy", [0, 0]),
                ("threshold", self._threshold),
                ("formal_openilt_epecheck", False),
                ("reward_weights", dict(sorted(self._weights.items()))),
                ("golden_point_set_version", self.golden_point_set.version),
                ("golden_point_set_source_sha256", self.golden_point_set.source_sha256),
                ("golden_point_set_sha256", self.golden_point_set.point_set_sha256),
            ),
        )


def _center_crop(image: np.ndarray, center_xy: Tuple[int, int], size: int) -> np.ndarray:
    """以零填充方式裁剪固定中心的正方形 patch。"""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("局部图像裁剪只接受二维数组")
    patch_size = int(size)
    if patch_size <= 0 or patch_size % 2:
        raise ValueError("patch_size 必须是正偶数")
    center_x, center_y = (int(center_xy[0]), int(center_xy[1]))
    radius = patch_size // 2
    output = np.zeros((patch_size, patch_size), dtype=np.float32)
    source_x0 = max(0, center_x - radius)
    source_y0 = max(0, center_y - radius)
    source_x1 = min(array.shape[1], center_x + radius)
    source_y1 = min(array.shape[0], center_y + radius)
    target_x0 = source_x0 - (center_x - radius)
    target_y0 = source_y0 - (center_y - radius)
    target_x1 = target_x0 + source_x1 - source_x0
    target_y1 = target_y0 + source_y1 - source_y0
    if source_x1 > source_x0 and source_y1 > source_y0:
        output[target_y0:target_y1, target_x0:target_x1] = array[source_y0:source_y1, source_x0:source_x1]
    return output


def _segment_normal_channel(point: EPEControlPoint, patch_size: int) -> np.ndarray:
    """在局部 patch 中标记所属 segment 与从基准点出发的外法线。"""
    channel = np.zeros((patch_size, patch_size), dtype=np.float32)
    radius = patch_size // 2
    start = (
        point.segment_start_xy[0] - point.base_xy[0] + radius,
        point.segment_start_xy[1] - point.base_xy[1] + radius,
    )
    end = (
        point.segment_end_xy[0] - point.base_xy[0] + radius,
        point.segment_end_xy[1] - point.base_xy[1] + radius,
    )
    if start[0] == end[0]:
        x = start[0]
        y0, y1 = sorted((start[1], end[1]))
        if 0 <= x < patch_size:
            channel[max(0, y0):min(patch_size, y1 + 1), x] = 1.0
    else:
        y = start[1]
        x0, x1 = sorted((start[0], end[0]))
        if 0 <= y < patch_size:
            channel[y, max(0, x0):min(patch_size, x1 + 1)] = 1.0
    normal_length = min(8, radius - 1)
    for distance in range(1, normal_length + 1):
        x = radius + point.normal_xy[0] * distance
        y = radius + point.normal_xy[1] * distance
        if 0 <= x < patch_size and 0 <= y < patch_size:
            channel[y, x] = 0.5
    return channel


@dataclass(frozen=True)
class FrozenPointObservation:
    """保存一个 point_id 的冻结 Actor 图像、几何向量和内容哈希。"""

    point_id: str
    image: np.ndarray
    vector: np.ndarray
    version: str
    sha256: str

    def as_dict(self) -> Dict[str, np.ndarray]:
        """返回可安全交给环境调用方修改的数组副本。"""
        return {"image": self.image.copy(), "vector": self.vector.copy()}


class FrozenObservationCache:
    """从 all-delta=0 基准一次性构建全部 point observation，禁止 prefix 回灌 Actor。"""

    def __init__(
        self,
        points: Sequence[EPEControlPoint],
        fragment_parameters: FragmentParameters,
        target_image: np.ndarray,
        baseline_result: V2SolverResult,
        patch_size: int,
        nm_per_coordinate: float,
    ):
        self._version = observation_version(patch_size)
        self._patch_size = int(patch_size)
        self._fragment_parameters = fragment_parameters
        self._target = _binary_image(target_image).astype(np.float32)
        arrays = (
            _binary_image(baseline_result.mask_image).astype(np.float32),
            _binary_image(baseline_result.printed_nominal).astype(np.float32),
        )
        if any(array.shape != self._target.shape for array in arrays):
            raise ValueError("baseline mask/printed 必须与 target 同形状")
        signs = baseline_result.sign_by_point_id()
        point_ids = [point.point_id for point in points]
        if len(point_ids) != len(set(point_ids)) or set(signs) != set(point_ids):
            raise ValueError("baseline recipe_epe_signs 必须恰好覆盖唯一 EPE point_id")
        self._baseline_state_sha256 = _sha256_json({
            "actor_state": RECIPE_V2_ACTOR_STATE,
            "fragment_parameters": fragment_parameters.as_dict(),
            "point_ids": sorted(point_ids),
            "recipe_epe_signs": [
                [point_id, float(signs[point_id])]
                for point_id in sorted(point_ids)
            ],
            "raster_sha256": _sha256_arrays(
                self._target,
                baseline_result.mask_image,
                baseline_result.printed_nominal,
                baseline_result.printed_max,
                baseline_result.printed_min,
            ),
        })
        view_nm = float(self.patch_size) * float(nm_per_coordinate)
        if not np.isfinite(view_nm) or view_nm <= 0:
            raise ValueError("patch 物理视野必须是有限正数")
        observations = {}
        for point in points:
            height, width = self._target.shape
            coordinates = (
                point.base_xy,
                point.segment_start_xy,
                point.segment_end_xy,
            )
            if any(not (0 <= x < width and 0 <= y < height) for x, y in coordinates):
                raise ValueError(f"EPE 点 {point.point_id} 的基准或 segment 越出 target 图像")
            marker = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            center = self.patch_size // 2
            marker[center - 1:center + 2, center - 1:center + 2] = 1.0
            image = np.stack((
                _center_crop(self._target, point.base_xy, self.patch_size),
                _center_crop(arrays[0], point.base_xy, self.patch_size),
                _center_crop(arrays[1], point.base_xy, self.patch_size),
                marker,
                _segment_normal_channel(point, self.patch_size),
            )).astype(np.float32)
            segment_length_nm = point.segment_length_coordinate * float(nm_per_coordinate)
            start_distance_nm = (
                abs(point.base_xy[0] - point.segment_start_xy[0])
                + abs(point.base_xy[1] - point.segment_start_xy[1])
            ) * float(nm_per_coordinate)
            end_distance_nm = (
                abs(point.segment_end_xy[0] - point.base_xy[0])
                + abs(point.segment_end_xy[1] - point.base_xy[1])
            ) * float(nm_per_coordinate)
            tangent_x, tangent_y = point.tangent_xy
            vector = np.asarray((
                float(tangent_x),
                float(tangent_y),
                float(point.normal_xy[0]),
                float(point.normal_xy[1]),
                float(segment_length_nm / view_nm),
                float(start_distance_nm / segment_length_nm),
                float(end_distance_nm / segment_length_nm),
                float(point.start_corner_type),
                float(point.end_corner_type),
                float(signs[point.point_id]),
                float(fragment_parameters.corner_length_nm / view_nm),
                float(fragment_parameters.uniform_length_nm / view_nm),
            ), dtype=np.float32)
            if vector.shape != (len(ACTOR_GEOMETRY_FIELDS),):
                raise RuntimeError("Actor 几何向量字段数量与 contract 不一致")
            digest = _sha256_json({
                "point_id": point.point_id,
                "version": self.version,
                "baseline_state_sha256": self.baseline_state_sha256,
                "image_sha256": _sha256_arrays(image),
                "vector_sha256": _sha256_arrays(vector),
            })
            image.setflags(write=False)
            vector.setflags(write=False)
            observations[point.point_id] = FrozenPointObservation(
                point_id=point.point_id,
                image=image,
                vector=vector,
                version=self.version,
                sha256=digest,
            )
        self._observations = MappingProxyType(observations)
        self._cache_sha256 = _sha256_json({
            point_id: observation.sha256
            for point_id, observation in sorted(observations.items())
        })

    @property
    def version(self) -> str:
        """返回冻结 observation contract 版本。"""
        return self._version

    @property
    def patch_size(self) -> int:
        """返回冻结 patch 边长。"""
        return self._patch_size

    @property
    def fragment_parameters(self) -> FragmentParameters:
        """返回构建 cache 时冻结的 FRAG 参数。"""
        return self._fragment_parameters

    @property
    def baseline_state_sha256(self) -> str:
        """返回绑定 raster、point IDs 和 baseline EPE signs 的 Actor 状态哈希。"""
        return self._baseline_state_sha256

    @property
    def cache_sha256(self) -> str:
        """返回全部逐点 observation 内容哈希。"""
        return self._cache_sha256

    def get(self, point_id: str) -> FrozenPointObservation:
        """按 point_id 返回防御性副本；调用方不能污染 cache 内部数组。"""
        try:
            observation = self._observations[str(point_id)]
        except KeyError as exc:
            raise KeyError(f"未知 EPE point_id：{point_id}") from exc
        image = observation.image.copy()
        vector = observation.vector.copy()
        image.setflags(write=False)
        vector.setflags(write=False)
        return FrozenPointObservation(
            point_id=observation.point_id,
            image=image,
            vector=vector,
            version=observation.version,
            sha256=observation.sha256,
        )

    def batch(self, point_ids: Sequence[str]) -> Dict[str, np.ndarray]:
        """按给定 point_id 顺序堆叠冻结 observation，供因子化 Actor 批量前向。"""
        ids = tuple(str(point_id) for point_id in point_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("batch point_ids 不允许重复")
        items = [self.get(point_id) for point_id in ids]
        return {
            "image": np.stack([item.image for item in items]).astype(np.float32),
            "vector": np.stack([item.vector for item in items]).astype(np.float32),
        }


class LocalEPEEpisode:
    """实现 dense/terminal 两种逐点协议；不依赖 Gym，便于 CPU Fake solver 验证。"""

    def __init__(
        self,
        solver: LocalEPESolver,
        golden_evaluator: GoldenEvaluator,
        reward_weights: Mapping[str, float],
        training_protocol: str,
        patch_size: int = 64,
        action_offsets_nm: Sequence[float] = EPE_ACTION_OFFSETS_NM,
        control_probe_distance_nm: float = DEFAULT_CONTROL_PROBE_DISTANCE_NM,
        training_reward_scale: float = DEFAULT_TRAINING_REWARD_SCALE,
        shuffle_points: bool = True,
    ):
        if training_protocol not in EPE_TRAINING_PROTOCOLS:
            raise ValueError(f"training_protocol 必须属于 {EPE_TRAINING_PROTOCOLS}")
        self._solver = solver
        self._golden_evaluator = golden_evaluator
        self._reward_weights = {name: float(value) for name, value in reward_weights.items()}
        GoldenMetrics(0, 0, 0).weighted_loss(self._reward_weights)
        self._training_protocol = str(training_protocol)
        self._observation_version = observation_version(patch_size)
        self._patch_size = int(patch_size)
        self._action_offsets_nm = tuple(float(value) for value in action_offsets_nm)
        if not self._action_offsets_nm or len(self._action_offsets_nm) != len(set(self._action_offsets_nm)):
            raise ValueError("action_offsets_nm 必须非空且无重复")
        self._control_probe_distance_nm = float(control_probe_distance_nm)
        _nm_to_coordinate(
            self._control_probe_distance_nm,
            solver.nm_per_coordinate,
            "control_probe_distance_nm",
        )
        self._training_reward_scale = float(training_reward_scale)
        if not np.isfinite(self._training_reward_scale) or self._training_reward_scale <= 0:
            raise ValueError("training_reward_scale 必须是有限正数")
        self._shuffle_points = bool(shuffle_points)
        self._epe_points = tuple(solver.epe_points)
        point_ids = [point.point_id for point in self._epe_points]
        if not point_ids or len(point_ids) != len(set(point_ids)):
            raise ValueError("v2 环境至少需要一个且不允许重复的 EPE point_id")
        self._point_by_id = MappingProxyType({point.point_id: point for point in self.epe_points})
        self._fragment_parameters = solver.fragment_parameters
        if not isinstance(self._fragment_parameters, FragmentParameters):
            raise TypeError("solver.fragment_parameters 必须使用 v2 FragmentParameters")
        self._nm_per_coordinate = float(solver.nm_per_coordinate)
        if not np.isfinite(self._nm_per_coordinate) or self._nm_per_coordinate <= 0:
            raise ValueError("solver.nm_per_coordinate 必须是有限正数")
        self._target_sha256 = _sha256_arrays(_binary_image(solver.target_image))
        self._layout_sha256 = require_sha256(solver.layout_sha256, "solver.layout_sha256")
        self._solver_revision = str(solver.revision)
        if not self._solver_revision:
            raise ValueError("solver.revision 不能为空")
        required_evaluator_fields = (
            "evaluator_version",
            "evaluator_source_sha256",
            "frozen_target_sha256",
            "sampling_state_sha256",
            "nm_per_coordinate",
            "coordinate_system_sha256",
            "evaluator_contract_sha256",
        )
        if any(not hasattr(golden_evaluator, name) for name in required_evaluator_fields):
            raise TypeError("Golden evaluator 缺少冻结 target/sampling/contract 身份字段")
        self._evaluator_version = str(golden_evaluator.evaluator_version)
        self._evaluator_source_sha256 = require_sha256(
            golden_evaluator.evaluator_source_sha256,
            "Golden evaluator source_sha256",
        )
        self._golden_target_sha256 = require_sha256(
            golden_evaluator.frozen_target_sha256,
            "Golden evaluator frozen_target_sha256",
        )
        self._golden_sampling_sha256 = require_sha256(
            golden_evaluator.sampling_state_sha256,
            "Golden evaluator sampling_state_sha256",
        )
        self._golden_nm_per_coordinate = float(golden_evaluator.nm_per_coordinate)
        if not np.isclose(
            self._golden_nm_per_coordinate,
            self._nm_per_coordinate,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("Golden evaluator 与 solver 的 nm_per_coordinate 不一致")
        self._golden_coordinate_system_sha256 = require_sha256(
            golden_evaluator.coordinate_system_sha256,
            "Golden evaluator coordinate_system_sha256",
        )
        expected_coordinate_system_sha256 = _identity_raster_coordinate_system_sha256(
            self._target_sha256,
            self._nm_per_coordinate,
        )
        if self._golden_coordinate_system_sha256 != expected_coordinate_system_sha256:
            raise ValueError("Golden evaluator 与 solver 的 raster/坐标系统身份不一致")
        self._evaluator_contract_sha256 = require_sha256(
            golden_evaluator.evaluator_contract_sha256,
            "Golden evaluator contract_sha256",
        )
        if self._golden_target_sha256 != self._target_sha256:
            raise ValueError("Golden evaluator 必须绑定 solver 使用的同一冻结原始 target")
        self._point_set_sha256 = _sha256_json([
            {
                "point_id": point.point_id,
                "polygon_index": point.polygon_index,
                "source_edge_index": point.source_edge_index,
                "segment_index": point.segment_index,
                "base_xy": list(point.base_xy),
                "segment_start_xy": list(point.segment_start_xy),
                "segment_end_xy": list(point.segment_end_xy),
                "normal_xy": list(point.normal_xy),
                "start_corner_type": point.start_corner_type,
                "end_corner_type": point.end_corner_type,
            }
            for point in sorted(self._epe_points, key=lambda item: item.point_id)
        ])
        self._candidate_tables = MappingProxyType({
            point.point_id: build_action_candidates(
                point,
                self.action_offsets_nm,
                self.control_probe_distance_nm,
                solver.nm_per_coordinate,
                solver.target_image,
            )
            for point in self.epe_points
        })
        self._action_table_sha256 = _sha256_json({
            point_id: [
                {
                    "action_class": candidate.action_class,
                    "offset_nm": candidate.probe.normal_offset_nm,
                    "realized_nm": candidate.probe.realized_normal_offset_nm,
                    "moved_xy": list(candidate.probe.moved_xy),
                    "inner_xy": list(candidate.probe.inner_xy),
                    "outer_xy": list(candidate.probe.outer_xy),
                    "alias": candidate.alias_of_action_class,
                    "in_bounds": candidate.probe.in_bounds,
                    "target_valid": candidate.probe.target_valid,
                }
                for candidate in candidates
            ]
            for point_id, candidates in sorted(self._candidate_tables.items())
        })
        self._geometry_identity_sha256 = _sha256_json({
            "environment": RECIPE_V2_ENV_VERSION,
            "layout_sha256": self._layout_sha256,
            "solver_revision": self._solver_revision,
            "target_sha256": self._target_sha256,
            "nm_per_coordinate": self._nm_per_coordinate,
            "fragment_parameters": self._fragment_parameters.as_dict(),
            "point_set_sha256": self.point_set_sha256,
            "action_table_sha256": self.action_table_sha256,
        })
        self._baseline_result: Optional[V2SolverResult] = None
        self._baseline_golden: Optional[GoldenEvaluation] = None
        self._observation_cache: Optional[FrozenObservationCache] = None
        self._baseline_solver_calls = 0
        self._candidate_solver_calls = 0
        self._final_replay_solver_calls = 0
        self._rng = np.random.default_rng(0)
        self._schedule: Tuple[str, ...] = tuple(point_ids)
        self._cursor = 0
        self._offsets = {point_id: 0.0 for point_id in point_ids}
        self._current_golden: Optional[GoldenEvaluation] = None
        self._final_golden: Optional[GoldenEvaluation] = None
        self._final_result: Optional[V2SolverResult] = None
        self._trajectory: List[dict] = []
        self._diagnostic_best_prefix: Optional[dict] = None
        self._terminated = False

    @property
    def episode_horizon(self) -> int:
        """返回每个 EPE point 恰好一次决策所需的步数。"""
        return len(self.epe_points)

    @property
    def solver(self) -> LocalEPESolver:
        """返回构造时绑定且不可重指向的 solver。"""
        return self._solver

    @property
    def golden_evaluator(self) -> GoldenEvaluator:
        """返回构造时绑定且不可重指向的 Golden evaluator。"""
        return self._golden_evaluator

    @property
    def reward_weights(self) -> Dict[str, float]:
        """返回固定 reward 权重副本，禁止外部原地修改 episode 口径。"""
        return dict(self._reward_weights)

    @property
    def training_protocol(self) -> str:
        """返回构造时冻结的 dense/terminal 协议。"""
        return self._training_protocol

    @property
    def observation_version(self) -> str:
        """返回构造时冻结的 observation 版本。"""
        return self._observation_version

    @property
    def patch_size(self) -> int:
        """返回构造时冻结的单尺度 patch 边长。"""
        return self._patch_size

    @property
    def action_offsets_nm(self) -> Tuple[float, ...]:
        """返回构造时冻结的离散法向动作表。"""
        return self._action_offsets_nm

    @property
    def control_probe_distance_nm(self) -> float:
        """返回构造时冻结的内部控制 probe 距离。"""
        return self._control_probe_distance_nm

    @property
    def training_reward_scale(self) -> float:
        """返回只影响训练 reward、不影响 raw metrics 的冻结比例。"""
        return self._training_reward_scale

    @property
    def shuffle_points(self) -> bool:
        """返回构造时冻结的 schedule 打乱选项。"""
        return self._shuffle_points

    @property
    def epe_points(self) -> Tuple[EPEControlPoint, ...]:
        """返回构造时冻结的 Recipe EPE 点集合。"""
        return self._epe_points

    @property
    def point_set_sha256(self) -> str:
        """返回冻结 Recipe 点几何哈希。"""
        return self._point_set_sha256

    @property
    def action_table_sha256(self) -> str:
        """返回冻结动作/量化/probe 候选表哈希。"""
        return self._action_table_sha256

    @property
    def geometry_identity_sha256(self) -> str:
        """返回绑定版图、尺度、点集和动作表的几何身份。"""
        return self._geometry_identity_sha256

    @property
    def point_ids(self) -> Tuple[str, ...]:
        """返回 solver 当前分段生成的 point ID。"""
        return tuple(point.point_id for point in self.epe_points)

    @property
    def fragment_parameters(self) -> FragmentParameters:
        """返回 episode 创建时冻结的 FRAG 参数。"""
        return self._fragment_parameters

    @property
    def nm_per_coordinate(self) -> float:
        """返回环境创建时冻结的数据库坐标物理单位。"""
        return float(self._nm_per_coordinate)

    @property
    def layout_sha256(self) -> str:
        """返回 episode 创建时冻结的版图内容哈希。"""
        return self._layout_sha256

    @property
    def solver_revision(self) -> str:
        """返回 episode 创建时冻结的 solver/OpenILT revision。"""
        return self._solver_revision

    @property
    def schedule(self) -> Tuple[str, ...]:
        """返回当前 episode 的点访问顺序。"""
        return tuple(self._schedule)

    @property
    def point_order_sha256(self) -> str:
        """返回当前完整 schedule 的稳定哈希。"""
        return _sha256_json(list(self._schedule))

    @property
    def solver_call_counts(self) -> Dict[str, int]:
        """分别报告基准、训练候选和 final replay solver 调用，避免口径混淆。"""
        return {
            "baseline_solver": int(self._baseline_solver_calls),
            "candidate_solver": int(self._candidate_solver_calls),
            "final_replay_solver": int(self._final_replay_solver_calls),
        }

    @property
    def observation_cache(self) -> FrozenObservationCache:
        """返回已建立的冻结 observation；未 reset 时显式失败。"""
        if self._observation_cache is None:
            raise RuntimeError("环境必须先 reset 才能读取 observation cache")
        return self._observation_cache

    def action_mask(self, point_id: Optional[str] = None) -> np.ndarray:
        """返回指定点或当前待决策点的合法动作掩码。"""
        selected = str(point_id) if point_id is not None else self._active_point_id()
        try:
            return candidate_action_mask(self._candidate_tables[selected])
        except KeyError as exc:
            raise KeyError(f"未知 EPE point_id：{selected}") from exc

    def require_plain_ppo_compatible(self) -> None:
        """确认普通 SB3 PPO 不会采到非法 action；默认 16/±40 配置预期会失败。"""
        require_all_actions_valid_for_plain_ppo(self._candidate_tables)

    def _assert_fragment_parameters_unchanged(self) -> None:
        """拒绝在 episode 内改变求解几何、单位、版图版本或 Golden contract。"""
        if self.solver.fragment_parameters != self._fragment_parameters:
            raise RuntimeError("FRAG 参数已改变；必须重建 solver、EPE 点和冻结 observation")
        if tuple(self.solver.epe_points) != self.epe_points:
            raise RuntimeError("EPE 点集合已改变；必须重建 v2 episode 和冻结 observation")
        if not np.isclose(float(self.solver.nm_per_coordinate), self._nm_per_coordinate, rtol=0.0, atol=0.0):
            raise RuntimeError("nm_per_coordinate 已改变；禁止复用既有动作和 observation")
        if _sha256_arrays(_binary_image(self.solver.target_image)) != self._target_sha256:
            raise RuntimeError("冻结 target 已改变；禁止继续当前 v2 episode")
        if require_sha256(self.solver.layout_sha256, "solver.layout_sha256") != self._layout_sha256:
            raise RuntimeError("solver layout_sha256 已改变；禁止继续当前 v2 episode")
        if str(self.solver.revision) != self._solver_revision:
            raise RuntimeError("solver revision 已改变；禁止继续当前 v2 episode")
        if str(self.golden_evaluator.evaluator_version) != self._evaluator_version:
            raise RuntimeError("Golden evaluator version 已改变；禁止继续当前 v2 episode")
        if require_sha256(
            self.golden_evaluator.evaluator_source_sha256,
            "Golden evaluator source_sha256",
        ) != self._evaluator_source_sha256:
            raise RuntimeError("Golden evaluator source 已改变；禁止继续当前 v2 episode")
        if require_sha256(
            self.golden_evaluator.frozen_target_sha256,
            "Golden evaluator frozen_target_sha256",
        ) != self._golden_target_sha256:
            raise RuntimeError("Golden evaluator target 已改变；禁止继续当前 v2 episode")
        if require_sha256(
            self.golden_evaluator.sampling_state_sha256,
            "Golden evaluator sampling_state_sha256",
        ) != self._golden_sampling_sha256:
            raise RuntimeError("Golden evaluator sampling 已改变；禁止继续当前 v2 episode")
        if not np.isclose(
            float(self.golden_evaluator.nm_per_coordinate),
            self._golden_nm_per_coordinate,
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError("Golden evaluator nm_per_coordinate 已改变；禁止继续当前 v2 episode")
        if require_sha256(
            self.golden_evaluator.coordinate_system_sha256,
            "Golden evaluator coordinate_system_sha256",
        ) != self._golden_coordinate_system_sha256:
            raise RuntimeError("Golden evaluator coordinate system 已改变；禁止继续当前 v2 episode")
        if require_sha256(
            self.golden_evaluator.evaluator_contract_sha256,
            "Golden evaluator contract_sha256",
        ) != self._evaluator_contract_sha256:
            raise RuntimeError("Golden evaluator contract 已改变；禁止继续当前 v2 episode")

    def _validate_golden(self, golden: GoldenEvaluation) -> GoldenEvaluation:
        """把 evaluator 返回值绑定到 episode 权重与冻结 evaluator contract。"""
        if not isinstance(golden, GoldenEvaluation):
            raise TypeError("Golden evaluator 必须返回 GoldenEvaluation")
        expected = golden.metrics.weighted_loss(self.reward_weights)
        if not np.isclose(
            expected,
            float(golden.raw_weighted_loss),
            rtol=1e-12,
            atol=1e-9,
        ):
            raise ValueError("Golden raw_weighted_loss 与 episode 固定 reward_weights 不一致")
        if golden.evaluator_version != self._evaluator_version:
            raise ValueError("Golden evaluation version 与冻结 evaluator 不一致")
        if golden.evaluator_source_sha256 != self._evaluator_source_sha256:
            raise ValueError("Golden evaluation source 与冻结 evaluator 不一致")
        if golden.evaluator_contract_sha256 != self._evaluator_contract_sha256:
            raise ValueError("Golden evaluation contract 与冻结 evaluator 不一致")
        return golden

    @staticmethod
    def _snapshot_solver_result(result: object) -> V2SolverResult:
        """强制 solver 返回正式类型，并复制为与回调方后续突变隔离的只读快照。"""
        if not isinstance(result, V2SolverResult):
            raise TypeError("solver.solve 必须返回 V2SolverResult，不能使用未校验 duck object")
        return V2SolverResult(
            mask_image=np.array(result.mask_image, copy=True),
            printed_nominal=np.array(result.printed_nominal, copy=True),
            printed_max=np.array(result.printed_max, copy=True),
            printed_min=np.array(result.printed_min, copy=True),
            recipe_epe_signs=tuple(result.recipe_epe_signs),
            mask_sha256=result.mask_sha256,
            internal_trace=copy.deepcopy(tuple(result.internal_trace)),
        )

    def _ensure_baseline(self) -> None:
        """缓存 all-delta=0 的唯一 Actor 基准，并明确计入一次 solver 调用。"""
        self._assert_fragment_parameters_unchanged()
        if self._baseline_result is not None:
            return
        zero = {point_id: 0.0 for point_id in self.point_ids}
        self._baseline_solver_calls += 1
        result = self._snapshot_solver_result(self.solver.solve(zero))
        self._assert_fragment_parameters_unchanged()
        golden = self._validate_golden(self.golden_evaluator.evaluate(result))
        self._assert_fragment_parameters_unchanged()
        cache = FrozenObservationCache(
            points=self.epe_points,
            fragment_parameters=self._fragment_parameters,
            target_image=self.solver.target_image,
            baseline_result=result,
            patch_size=self.patch_size,
            nm_per_coordinate=self._nm_per_coordinate,
        )
        self._baseline_result = result
        self._baseline_golden = golden
        self._observation_cache = cache

    def _active_point_id(self) -> str:
        """返回当前待决策 point_id；episode 结束后返回最后一个点供终止 observation 使用。"""
        if not self._schedule:
            raise RuntimeError("point schedule 为空")
        index = min(self._cursor, len(self._schedule) - 1)
        return self._schedule[index]

    def _observation(self) -> Dict[str, np.ndarray]:
        """只从冻结 cache 读取 Actor 输入，不接触当前 prefix result 或 loss。"""
        return self.observation_cache.get(self._active_point_id()).as_dict()

    def reset(
        self,
        seed: Optional[int] = None,
        point_order: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, np.ndarray], dict]:
        """恢复全零 Recipe，并为全部 EPE point 建立无遗漏、无重复 schedule。"""
        self._assert_fragment_parameters_unchanged()
        self._ensure_baseline()
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        expected = tuple(self.point_ids)
        if point_order is not None:
            schedule = tuple(str(point_id) for point_id in point_order)
            if len(schedule) != len(set(schedule)) or set(schedule) != set(expected):
                raise ValueError("point_order 必须恰好包含全部 EPE point_id 且无重复")
        else:
            schedule_list = list(expected)
            if self.shuffle_points:
                self._rng.shuffle(schedule_list)
            schedule = tuple(schedule_list)
        self._schedule = schedule
        self._cursor = 0
        self._offsets = {point_id: 0.0 for point_id in expected}
        self._current_golden = self._baseline_golden
        self._final_golden = None
        self._final_result = None
        self._terminated = False
        baseline = self._baseline_golden
        if baseline is None:
            raise RuntimeError("baseline Golden evaluation 未建立")
        self._diagnostic_best_prefix = {
            "step": 0,
            "raw_metrics": baseline.metrics.as_dict(),
            "raw_weighted_loss": float(baseline.raw_weighted_loss),
        }
        self._trajectory = [{
            "step": 0,
            "point_id": None,
            "action_class": None,
            "normal_offset_nm": 0.0,
            "raw_metrics": baseline.metrics.as_dict(),
            "raw_weighted_loss_before": baseline.raw_weighted_loss,
            "raw_weighted_loss_after": baseline.raw_weighted_loss,
            "scaled_training_reward": 0.0,
            "candidate_solver_called": False,
            "final_recipe_complete": False,
        }]
        return self._observation(), {
            "environment": RECIPE_V2_ENV_VERSION,
            "observation_version": self.observation_version,
            "actor_observation_state": RECIPE_V2_ACTOR_STATE,
            "training_protocol": self.training_protocol,
            "point_count": len(expected),
            "epe_point_count": len(expected),
            "frag_point_count": 0,
            "fragment_parameters": self._fragment_parameters.as_dict(),
            "point_order_sha256": self.point_order_sha256,
            "baseline_state_sha256": self.observation_cache.baseline_state_sha256,
            "initial_raw_metrics": baseline.metrics.as_dict(),
            "initial_raw_weighted_loss": baseline.raw_weighted_loss,
            "solver_call_counts": self.solver_call_counts,
        }

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        """只更新当前 point；dense 每步求解，terminal 仅完整 Recipe 终点求解。"""
        if self._current_golden is None:
            raise RuntimeError("环境必须先 reset")
        if self._terminated or self._cursor >= len(self._schedule):
            raise RuntimeError("episode 已结束，请先 reset")
        self._assert_fragment_parameters_unchanged()
        action_class = _discrete_action_index(action, len(self.action_offsets_nm))
        point_id = self._active_point_id()
        candidate = self._candidate_tables[point_id][action_class]
        if not candidate.valid:
            raise ValueError(
                f"point {point_id} 的 action {action_class} 不是独立且 target-valid 的 probe 候选"
            )
        if point_id in {item["point_id"] for item in self._trajectory if item["point_id"] is not None}:
            raise RuntimeError(f"point {point_id} 在同一 episode 被重复决策")
        previous_loss = float(self._current_golden.raw_weighted_loss)
        candidate_offsets = dict(self._offsets)
        candidate_offsets[point_id] = float(candidate.probe.normal_offset_nm)
        candidate_cursor = self._cursor + 1
        complete = candidate_cursor == len(self._schedule)
        result: Optional[V2SolverResult] = None
        golden: Optional[GoldenEvaluation] = None
        candidate_called = False
        if self.training_protocol == EPE_DENSE_PROTOCOL or complete:
            self._candidate_solver_calls += 1
            result = self._snapshot_solver_result(self.solver.solve(dict(candidate_offsets)))
            candidate_called = True
            self._assert_fragment_parameters_unchanged()
            golden = self._validate_golden(self.golden_evaluator.evaluate(result))
            self._assert_fragment_parameters_unchanged()
        if self.training_protocol == EPE_DENSE_PROTOCOL:
            if golden is None:
                raise RuntimeError("dense 协议每一步都必须产生 Golden evaluation")
            reward = self.training_reward_scale * (previous_loss - golden.raw_weighted_loss)
        elif complete:
            if golden is None or self._baseline_golden is None:
                raise RuntimeError("terminal 终点必须产生基准与候选 Golden evaluation")
            reward = self.training_reward_scale * (
                self._baseline_golden.raw_weighted_loss - golden.raw_weighted_loss
            )
        else:
            reward = 0.0
        current = golden if golden is not None else self._current_golden
        if current is None:
            raise RuntimeError("当前 Golden evaluation 不存在")
        item = {
            "step": candidate_cursor,
            "point_id": point_id,
            "action_class": action_class,
            "normal_offset_nm": float(candidate.probe.normal_offset_nm),
            "moved_xy": list(candidate.probe.moved_xy),
            "raw_metrics": current.metrics.as_dict() if candidate_called else None,
            "raw_weighted_loss_before": previous_loss if candidate_called else None,
            "raw_weighted_loss_after": current.raw_weighted_loss if candidate_called else None,
            "scaled_training_reward": float(reward),
            "candidate_solver_called": candidate_called,
            "final_recipe_complete": complete,
            "recipe_offsets_nm": dict(sorted(candidate_offsets.items())),
            "solver_call_counts": self.solver_call_counts,
        }
        # solver/evaluator/reward 全部成功后再一次性提交 episode 状态；失败时 cursor/Recipe/轨迹不变。
        self._offsets = candidate_offsets
        self._cursor = candidate_cursor
        if golden is not None:
            self._current_golden = golden
        self._trajectory.append(item)
        if candidate_called and self.training_protocol == EPE_DENSE_PROTOCOL:
            best = self._diagnostic_best_prefix
            if best is None or current.raw_weighted_loss < best["raw_weighted_loss"]:
                self._diagnostic_best_prefix = {
                    "step": candidate_cursor,
                    "raw_metrics": current.metrics.as_dict(),
                    "raw_weighted_loss": float(current.raw_weighted_loss),
                }
        if complete:
            if result is None or golden is None:
                raise RuntimeError("完整 Recipe 必须有 final solver/Golden 结果")
            self._final_result = result
            self._final_golden = golden
            self._terminated = True
        return self._observation(), float(reward), complete, copy.deepcopy(item)

    @property
    def trajectory(self) -> Tuple[dict, ...]:
        """返回不可由调用方原地修改的完整轨迹。"""
        return tuple(copy.deepcopy(self._trajectory))

    @property
    def diagnostic_best_prefix(self) -> Optional[dict]:
        """返回仅供诊断的 best-prefix；它不提供任何 final Recipe 字段。"""
        return copy.deepcopy(self._diagnostic_best_prefix)

    @property
    def final_recipe_offsets_nm(self) -> Dict[str, float]:
        """只在全部 point 都已决策后返回完整 Recipe。"""
        if not self._terminated:
            raise RuntimeError("final Recipe 尚不完整")
        return {point_id: float(value) for point_id, value in sorted(self._offsets.items())}

    @property
    def final_golden_evaluation(self) -> GoldenEvaluation:
        """只返回完整 final Recipe 的 Golden 指标，绝不回退 best-prefix。"""
        if self._final_golden is None:
            raise RuntimeError("final Golden evaluation 尚不存在")
        return self._final_golden

    @property
    def final_result(self) -> V2SolverResult:
        """返回完整 final Recipe 的 solver 防御性快照。"""
        if self._final_result is None:
            raise RuntimeError("final solver result 尚不存在")
        return self._snapshot_solver_result(self._final_result)

    @property
    def final_recipe_sha256(self) -> str:
        """返回绑定版图、尺度、点几何、动作表与实际移动坐标的完整 Recipe 哈希。"""
        return self._recipe_hash_for_offsets(self.final_recipe_offsets_nm)

    def _recipe_hash_for_offsets(self, offsets_nm: Mapping[str, float]) -> str:
        """为完整 offset 映射计算不受 schedule 顺序影响的稳定身份。"""
        if set(offsets_nm) != set(self.point_ids):
            raise ValueError("Recipe hash 输入必须恰好覆盖全部 EPE point_id")
        realized_points = []
        for point in sorted(self.epe_points, key=lambda item: item.point_id):
            realized_points.append(point.as_recipe_dict(
                normal_offset_nm=float(offsets_nm[point.point_id]),
                nm_per_coordinate=self._nm_per_coordinate,
            ))
        return _sha256_json({
            "environment": RECIPE_V2_ENV_VERSION,
            "geometry_identity_sha256": self.geometry_identity_sha256,
            "action_table_sha256": self.action_table_sha256,
            "realized_epe_points": realized_points,
        })

    def replay_complete_action_map(
        self,
        actions_by_point_id: Mapping[str, int],
    ) -> Tuple[Dict[str, float], V2SolverResult, GoldenEvaluation, str]:
        """按 point_id 组装完整 Recipe 并只调用一次 solver，作为 batch final replay 核心。"""
        self._ensure_baseline()
        self._assert_fragment_parameters_unchanged()
        if set(actions_by_point_id) != set(self.point_ids):
            raise ValueError("batch final actions 必须恰好覆盖全部 EPE point_id")
        offsets = {}
        for point_id in self.point_ids:
            action_class = _discrete_action_index(
                actions_by_point_id[point_id],
                len(self.action_offsets_nm),
                name=f"point {point_id} action",
            )
            candidate = self._candidate_tables[point_id][action_class]
            if not candidate.valid:
                raise ValueError(f"point {point_id} 的 final action 不是合法候选")
            offsets[point_id] = float(candidate.probe.normal_offset_nm)
        self._final_replay_solver_calls += 1
        result = self._snapshot_solver_result(self.solver.solve(dict(offsets)))
        self._assert_fragment_parameters_unchanged()
        golden = self._validate_golden(self.golden_evaluator.evaluate(result))
        self._assert_fragment_parameters_unchanged()
        recipe_hash = self._recipe_hash_for_offsets(offsets)
        return dict(sorted(offsets.items())), result, golden, recipe_hash


def build_v2_recipe_payload(
    episode: LocalEPEEpisode,
    epe_model_sha256: str,
    frag_model_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict:
    """导出 final-only v2 Recipe；best-prefix 只能写入显式 diagnostic 字段。"""
    episode._assert_fragment_parameters_unchanged()
    final_offsets = episode.final_recipe_offsets_nm
    final = episode.final_golden_evaluation
    normalized_sources = {}
    for name, value in sorted(source_hashes.items()):
        source_name = str(name)
        if not source_name:
            raise ValueError("source_hashes 的名称不能为空")
        normalized_sources[source_name] = require_sha256(value, f"source_hashes[{source_name}]")
    if not normalized_sources:
        raise ValueError("source_hashes 至少必须记录一个项目/上游来源文件")

    def optional_model_sha256(value: str, name: str) -> Optional[str]:
        text = str(value)
        return None if not text else require_sha256(text, name)

    points = []
    for point in sorted(episode.epe_points, key=lambda value: value.point_id):
        points.append(point.as_recipe_dict(
            normal_offset_nm=final_offsets[point.point_id],
            nm_per_coordinate=episode.nm_per_coordinate,
        ))
    payload = {
        "schema_version": RECIPE_V2_SCHEMA_VERSION,
        "environment": RECIPE_V2_ENV_VERSION,
        "label_version": RECIPE_V2_LABEL_VERSION,
        "point_version": RECIPE_V2_POINT_VERSION,
        "action_semantics": RECIPE_V2_ACTION_SEMANTICS,
        "training_protocol": episode.training_protocol,
        "accepted_metrics_source": None,
        "required_accepted_metrics_source": RECIPE_V2_ACCEPTED_METRICS_SOURCE,
        "actor_observation_state": RECIPE_V2_ACTOR_STATE,
        "observation_version": episode.observation_version,
        "patch_shape": [5, int(episode.patch_size), int(episode.patch_size)],
        "vector_shape": [len(ACTOR_GEOMETRY_FIELDS)],
        "loss_version": RECIPE_V2_LOSS_VERSION,
        "gamma": DEFAULT_GAMMA,
        "gae_lambda": DEFAULT_GAE_LAMBDA,
        "epe_probe_distance_nm": float(episode.control_probe_distance_nm),
        **episode.fragment_parameters.as_dict(),
        "epe_points": points,
        "raw_metrics": {
            **final.metrics.as_dict(),
            "weighted_loss": float(final.raw_weighted_loss),
        },
        "raw_metrics_source": f"{episode.training_protocol}_final",
        "optional_extended_metrics": {
            "epe_n": None,
            "epe_d": None,
            "mrc_violations": None,
        },
        "golden_evaluator": {
            "version": final.evaluator_version,
            "source_sha256": final.evaluator_source_sha256,
            "frozen_target_sha256": episode._golden_target_sha256,
            "sampling_state_sha256": episode._golden_sampling_sha256,
            "nm_per_coordinate": episode._golden_nm_per_coordinate,
            "coordinate_system_sha256": episode._golden_coordinate_system_sha256,
            "contract_sha256": final.evaluator_contract_sha256,
            "parameters": final.parameters_dict(),
        },
        "training_reward_scale": float(episode.training_reward_scale),
        "point_order_sha256": require_sha256(episode.point_order_sha256, "point_order_sha256"),
        "baseline_state_sha256": require_sha256(
            episode.observation_cache.baseline_state_sha256,
            "baseline_state_sha256",
        ),
        "action_table_sha256": require_sha256(
            episode.action_table_sha256,
            "action_table_sha256",
        ),
        "point_set_sha256": require_sha256(episode.point_set_sha256, "point_set_sha256"),
        "geometry_identity_sha256": require_sha256(
            episode.geometry_identity_sha256,
            "geometry_identity_sha256",
        ),
        "final_recipe_complete": True,
        "final_recipe_sha256": require_sha256(
            episode.final_recipe_sha256,
            "final_recipe_sha256",
        ),
        "full_recipe_replay_sha256": None,
        "sequential_final_replay_gap": None,
        "solver_call_counts": episode.solver_call_counts,
        "diagnostic_best_prefix": episode.diagnostic_best_prefix,
        "epe_model_sha256": optional_model_sha256(epe_model_sha256, "epe_model_sha256"),
        "frag_model_sha256": optional_model_sha256(frag_model_sha256, "frag_model_sha256"),
        "layout_sha256": episode.layout_sha256,
        "openilt_revision": episode.solver_revision,
        "source_hashes": normalized_sources,
        "status": "diagnostic_only",
    }
    forbidden = {"epe_control_distance_nm", "best_recipe_offsets_nm", "golden_epe_distance_nm"}
    pending = [payload]
    found = set()
    while pending:
        value = pending.pop()
        if isinstance(value, Mapping):
            found.update(forbidden.intersection(str(key) for key in value))
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
    if found:
        raise RuntimeError("v2 产物包含禁止的全局 EPE/best-prefix 字段")
    return payload
