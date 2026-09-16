"""本模块定义局部 EPE、全局 FRAG Recipe PPO v2 的无 GPU 数据协议。

输入是冻结的分段参数、原始 target 上的 EPE 基准点、逐点法向动作及 solver 栅格结果；输出是
可由几何测试、Fake solver、Gym 环境和产物导出共同复用的版本常量与不可变数据对象。本模块不
导入 Gymnasium、PyTorch 或 OpenILT，也不修改 v1 协议，目的是让已验收的 v1 产物保持可回放。
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Optional, Tuple

import numpy as np


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def require_sha256(value: str, name: str) -> str:
    """校验并返回小写 SHA256，禁止用描述字符串冒充内容身份。"""
    text = str(value).lower()
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} 必须是 64 位小写十六进制 SHA256")
    return text


def array_sha256(value: np.ndarray) -> str:
    """连同 dtype 和 shape 计算单个数组的稳定 SHA256。"""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _integer_pair(value, name: str) -> Tuple[int, int]:
    """把二维整数序列规范化为不可变 tuple，并拒绝浮点截断。"""
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} 必须是二维整数序列") from exc
    if len(items) != 2:
        raise ValueError(f"{name} 必须是二维值")
    if any(
        not isinstance(item, (int, np.integer)) or isinstance(item, (bool, np.bool_))
        for item in items
    ):
        raise TypeError(f"{name} 必须使用整数数据库坐标，禁止静默截断浮点数")
    return int(items[0]), int(items[1])


def _freeze_parameter(value: object, name: str) -> object:
    """把 evaluator 参数递归规范化为不可变 JSON 值，阻断 accessor 反向污染。"""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"Golden evaluator 参数 {name} 不能包含 NaN 或无穷值")
        return float(value)
    if isinstance(value, ABCMapping):
        normalized = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if not key or key in normalized:
                raise ValueError(f"Golden evaluator 参数 {name} 的映射键不能为空或重复")
            normalized[key] = _freeze_parameter(raw_value, f"{name}.{key}")
        return MappingProxyType(dict(sorted(normalized.items())))
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_parameter(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"Golden evaluator 参数 {name} 只允许 JSON 标量、映射或序列")


def _thaw_parameter(value: object) -> object:
    """把内部不可变参数恢复为调用方可独立修改的 JSON 友好副本。"""
    if isinstance(value, ABCMapping):
        return {str(key): _thaw_parameter(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_parameter(item) for item in value]
    return value


RECIPE_V2_SCHEMA_VERSION = "2.0"
RECIPE_V2_ENV_VERSION = "simpleopc-recipe-local-epe-global-frag-v2"
RECIPE_V2_POINT_VERSION = "target-epe-normal-point-v2"
RECIPE_V2_LABEL_VERSION = "ppo-recipe-local-epe-global-frag-v2"
RECIPE_V2_ACTION_SEMANTICS = (
    "per_epe_point_normal_offset_and_two_global_fragment_lengths"
)
RECIPE_V2_ACCEPTED_METRICS_SOURCE = "batched_final_replay"
RECIPE_V2_ACTOR_STATE = "frozen_zero_offset_baseline"
RECIPE_V2_LOSS_VERSION = "paper-weighted-sum-raw-v1"

EPE_DENSE_PROTOCOL = "ppo_dense_sequential"
EPE_TERMINAL_PROTOCOL = "ppo_terminal_full_recipe"
EPE_TRAINING_PROTOCOLS = (EPE_DENSE_PROTOCOL, EPE_TERMINAL_PROTOCOL)
EPE_ACTION_OFFSETS_NM = (-40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0)
DEFAULT_FRAGMENT_PARAMETERS_NM = (16.0, 32.0)
DEFAULT_CONTROL_PROBE_DISTANCE_NM = 16.0
DEFAULT_TRAINING_REWARD_SCALE = 1e-5
DEFAULT_GAMMA = 1.0
DEFAULT_GAE_LAMBDA = 1.0

# 64 只作为接口 smoke；128 是首个正式候选。多尺度尚未冻结，不能静默映射到任一版本。
SUPPORTED_SINGLE_SCALE_PATCH_SIZES = (64, 128)
ACTOR_GEOMETRY_FIELDS = (
    "tangent_x",
    "tangent_y",
    "normal_x",
    "normal_y",
    "segment_length_over_view",
    "distance_to_start_over_segment",
    "distance_to_end_over_segment",
    "start_corner_type",
    "end_corner_type",
    "baseline_epe_sign",
    "corner_length_over_view",
    "uniform_length_over_view",
)
FORBIDDEN_ACTOR_FIELDS = frozenset({
    "base_x",
    "base_y",
    "absolute_x",
    "absolute_y",
    "episode_progress",
    "current_offset",
    "current_loss",
    "best_prefix",
    "lower_action_bound",
    "upper_action_bound",
})


def observation_version(patch_size: int) -> str:
    """为不同单尺度 patch 返回不同版本；未冻结的尺寸显式失败。"""
    if not isinstance(patch_size, (int, np.integer)) or isinstance(patch_size, (bool, np.bool_)):
        raise TypeError("patch_size 必须是整数")
    size = int(patch_size)
    if size not in SUPPORTED_SINGLE_SCALE_PATCH_SIZES:
        raise ValueError("v2 单尺度 observation 目前只版本化 64×64 与 128×128")
    role = "smoke" if size == 64 else "formal-candidate"
    return f"local-epe-frozen-baseline-5x{size}-geom12-v2-{role}"


@dataclass(frozen=True)
class FragmentParameters:
    """保存一张版图在整个 EPE episode 内不可变的两个全局分段长度。"""

    corner_length_nm: float = DEFAULT_FRAGMENT_PARAMETERS_NM[0]
    uniform_length_nm: float = DEFAULT_FRAGMENT_PARAMETERS_NM[1]

    def __post_init__(self) -> None:
        if not np.isfinite(self.corner_length_nm) or self.corner_length_nm <= 0:
            raise ValueError("corner_length_nm 必须是有限正数")
        if not np.isfinite(self.uniform_length_nm) or self.uniform_length_nm <= 0:
            raise ValueError("uniform_length_nm 必须是有限正数")

    def as_dict(self) -> Dict[str, float]:
        """返回与 v2 Recipe 字段一致的 JSON 映射。"""
        return {
            "corner_length_nm": float(self.corner_length_nm),
            "uniform_length_nm": float(self.uniform_length_nm),
        }


@dataclass(frozen=True)
class EPEControlPoint:
    """记录原始 target 分段上的稳定 EPE 基准点和指向图形外部的法线。"""

    point_id: str
    polygon_index: int
    source_edge_index: int
    segment_index: int
    base_xy: Tuple[int, int]
    segment_start_xy: Tuple[int, int]
    segment_end_xy: Tuple[int, int]
    normal_xy: Tuple[int, int]
    start_corner_type: int = 0
    end_corner_type: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.point_id, str) or not self.point_id:
            raise ValueError("point_id 不能为空")
        indices = (self.polygon_index, self.source_edge_index, self.segment_index)
        if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) for value in indices):
            raise TypeError("polygon/source-edge/segment 索引必须是整数")
        normalized_indices = tuple(int(value) for value in indices)
        if min(normalized_indices) < 0:
            raise ValueError("polygon/source-edge/segment 索引不能为负")

        start = _integer_pair(self.segment_start_xy, "segment_start_xy")
        end = _integer_pair(self.segment_end_xy, "segment_end_xy")
        base = _integer_pair(self.base_xy, "base_xy")
        normal = _integer_pair(self.normal_xy, "normal_xy")
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if (dx == 0) == (dy == 0):
            raise ValueError("EPE point 只能绑定非零正交 segment")
        if normal not in {(-1, 0), (1, 0), (0, -1), (0, 1)}:
            raise ValueError("normal_xy 必须是轴对齐单位法线")
        if dx * normal[0] + dy * normal[1] != 0:
            raise ValueError("normal_xy 必须与所属 segment 垂直")
        if dx == 0:
            on_segment = base[0] == start[0] and min(start[1], end[1]) <= base[1] <= max(start[1], end[1])
        else:
            on_segment = base[1] == start[1] and min(start[0], end[0]) <= base[0] <= max(start[0], end[0])
        if not on_segment:
            raise ValueError("base_xy 必须落在绑定的 segment 上")
        corner_types = (self.start_corner_type, self.end_corner_type)
        if any(
            not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            for value in corner_types
        ):
            raise TypeError("corner_type 必须是整数")
        if int(self.start_corner_type) not in {-1, 0, 1} or int(self.end_corner_type) not in {-1, 0, 1}:
            raise ValueError("corner_type 只能是 -1（凹）、0（非角点）或 1（凸）")
        object.__setattr__(self, "polygon_index", normalized_indices[0])
        object.__setattr__(self, "source_edge_index", normalized_indices[1])
        object.__setattr__(self, "segment_index", normalized_indices[2])
        object.__setattr__(self, "base_xy", base)
        object.__setattr__(self, "segment_start_xy", start)
        object.__setattr__(self, "segment_end_xy", end)
        object.__setattr__(self, "normal_xy", normal)
        object.__setattr__(self, "start_corner_type", int(self.start_corner_type))
        object.__setattr__(self, "end_corner_type", int(self.end_corner_type))

    @property
    def tangent_xy(self) -> Tuple[int, int]:
        """返回 segment 从 start 指向 end 的轴对齐单位切向。"""
        dx = self.segment_end_xy[0] - self.segment_start_xy[0]
        dy = self.segment_end_xy[1] - self.segment_start_xy[1]
        return (
            0 if dx == 0 else (1 if dx > 0 else -1),
            0 if dy == 0 else (1 if dy > 0 else -1),
        )

    @property
    def segment_length_coordinate(self) -> int:
        """返回 segment 的坐标单位曼哈顿长度。"""
        return abs(self.segment_end_xy[0] - self.segment_start_xy[0]) + abs(
            self.segment_end_xy[1] - self.segment_start_xy[1]
        )

    def moved_xy(self, normal_offset_nm: float, nm_per_coordinate: float) -> Tuple[int, int]:
        """按 q=p+delta*n 量化到数据库坐标，绝不沿切向移动。"""
        scale = float(nm_per_coordinate)
        offset = float(normal_offset_nm)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("nm_per_coordinate 必须是有限正数")
        if not np.isfinite(offset):
            raise ValueError("normal_offset_nm 必须是有限数")
        delta_coordinate = int(np.rint(offset / scale))
        return (
            int(self.base_xy[0] + delta_coordinate * self.normal_xy[0]),
            int(self.base_xy[1] + delta_coordinate * self.normal_xy[1]),
        )

    def as_recipe_dict(self, normal_offset_nm: float, nm_per_coordinate: float) -> Dict[str, object]:
        """返回 v2 产物要求的基准、法线、动作和移动后坐标。"""
        moved = self.moved_xy(normal_offset_nm, nm_per_coordinate)
        realized_offset = (
            (moved[0] - self.base_xy[0]) * self.normal_xy[0]
            + (moved[1] - self.base_xy[1]) * self.normal_xy[1]
        ) * float(nm_per_coordinate)
        return {
            "point_id": self.point_id,
            "base_xy": [int(self.base_xy[0]), int(self.base_xy[1])],
            "normal_xy": [int(self.normal_xy[0]), int(self.normal_xy[1])],
            "normal_offset_nm": float(normal_offset_nm),
            "realized_normal_offset_nm": float(realized_offset),
            "quantization_error_nm": float(realized_offset - float(normal_offset_nm)),
            "moved_xy": [int(moved[0]), int(moved[1])],
            "segment_id": (
                f"polygon-{self.polygon_index}-edge-{self.source_edge_index}"
                f"-segment-{self.segment_index}"
            ),
        }


@dataclass(frozen=True)
class GoldenControlPoint:
    """保存独立于 FRAG/Recipe 点集的冻结 Golden target 边界采样点。"""

    point_id: str
    base_xy: Tuple[int, int]
    normal_xy: Tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.point_id, str) or not self.point_id:
            raise ValueError("Golden point_id 不能为空")
        base = _integer_pair(self.base_xy, "Golden base_xy")
        normal = _integer_pair(self.normal_xy, "Golden normal_xy")
        if normal not in {(-1, 0), (1, 0), (0, -1), (0, 1)}:
            raise ValueError("Golden normal_xy 必须是轴对齐单位法线")
        object.__setattr__(self, "base_xy", base)
        object.__setattr__(self, "normal_xy", normal)

    def moved_xy(self, normal_offset_nm: float, nm_per_coordinate: float) -> Tuple[int, int]:
        """仅为复用固定 probe 构造接口；Golden evaluator 只允许 offset=0。"""
        if float(normal_offset_nm) != 0.0:
            raise ValueError("Golden 点固定在原始 target 边界，禁止施加 Recipe 位移")
        scale = float(nm_per_coordinate)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("nm_per_coordinate 必须是有限正数")
        return self.base_xy

    def as_hash_dict(self) -> Dict[str, object]:
        """返回只含冻结 target 几何的稳定哈希字段。"""
        return {
            "point_id": self.point_id,
            "base_xy": [self.base_xy[0], self.base_xy[1]],
            "normal_xy": [self.normal_xy[0], self.normal_xy[1]],
        }


@dataclass(frozen=True)
class GoldenPointSet:
    """冻结原始 target 上、在任何 FRAG 分段之前建立的 Golden 采样点集。"""

    version: str
    frozen_target_sha256: str
    nm_per_coordinate: float
    source_sha256: str
    points: Tuple[GoldenControlPoint, ...]
    point_set_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("Golden point-set version 不能为空")
        target_sha256 = require_sha256(self.frozen_target_sha256, "Golden frozen_target_sha256")
        source_sha256 = require_sha256(self.source_sha256, "Golden point-set source_sha256")
        scale = float(self.nm_per_coordinate)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Golden nm_per_coordinate 必须是有限正数")
        points = tuple(self.points)
        if not points or any(not isinstance(point, GoldenControlPoint) for point in points):
            raise TypeError("GoldenPointSet 只能包含至少一个 GoldenControlPoint")
        ids = [point.point_id for point in points]
        if len(ids) != len(set(ids)):
            raise ValueError("Golden point_id 不允许重复")
        points = tuple(sorted(points, key=lambda point: point.point_id))
        canonical = {
            "version": self.version,
            "frozen_target_sha256": target_sha256,
            "nm_per_coordinate": scale,
            "source_sha256": source_sha256,
            "points": [point.as_hash_dict() for point in points],
        }
        digest = hashlib.sha256(
            json.dumps(
                canonical,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "frozen_target_sha256", target_sha256)
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "nm_per_coordinate", scale)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "point_set_sha256", digest)


@dataclass(frozen=True)
class ProbeWindow:
    """保存一个逐点法向动作量化后的 crossing 及固定内外 probe 坐标。"""

    point_id: str
    normal_offset_nm: float
    realized_normal_offset_nm: float
    quantization_error_nm: float
    moved_xy: Tuple[int, int]
    inner_xy: Tuple[int, int]
    outer_xy: Tuple[int, int]
    in_bounds: bool
    inner_target_valid: bool
    outer_target_valid: bool

    @property
    def target_valid(self) -> bool:
        """只有 inner 位于 target 内且 outer 位于 target 外时控制语义才有效。"""
        return bool(self.in_bounds and self.inner_target_valid and self.outer_target_valid)


@dataclass(frozen=True)
class ActionCandidate:
    """记录一个离散动作的量化、别名、边界和 target-validity 证据。"""

    action_class: int
    probe: ProbeWindow
    alias_of_action_class: Optional[int] = None

    @property
    def valid(self) -> bool:
        """动作必须有独立栅格结果，且 probe 边界和 target 语义均有效。"""
        return self.alias_of_action_class is None and self.probe.target_valid


@dataclass(frozen=True)
class GoldenMetrics:
    """保存固定 Golden evaluator 返回的原始 L2/EPE/PVB 指标。"""

    l2: float
    epe: float
    pvb: float

    def __post_init__(self) -> None:
        values = (float(self.l2), float(self.epe), float(self.pvb))
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("Golden 指标必须是有限非负数")

    def as_dict(self) -> Dict[str, float]:
        """返回原始指标映射。"""
        return {"l2": float(self.l2), "epe": float(self.epe), "pvb": float(self.pvb)}

    def weighted_loss(self, weights: Dict[str, float]) -> float:
        """计算固定的 L2+EPE+PVB 加权和，不做运行时归一化。"""
        if set(weights) != {"l2", "epe", "pvb"}:
            raise ValueError("reward_weights 必须恰好包含 l2、epe、pvb")
        coefficients = {name: float(value) for name, value in weights.items()}
        if any(not np.isfinite(value) or value < 0 for value in coefficients.values()):
            raise ValueError("reward_weights 必须是有限非负数")
        if sum(coefficients.values()) <= 0:
            raise ValueError("reward_weights 至少一个必须大于零")
        metrics = self.as_dict()
        return float(sum(metrics[name] * coefficients[name] for name in metrics))


@dataclass(frozen=True)
class V2SolverResult:
    """保存候选 solver 输出；不携带 Golden target 或可移动 Recipe 点。"""

    mask_image: np.ndarray
    printed_nominal: np.ndarray
    printed_max: np.ndarray
    printed_min: np.ndarray
    recipe_epe_signs: Tuple[Tuple[str, float], ...]
    mask_sha256: str
    internal_trace: Tuple[dict, ...] = ()

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(value) for value in (
            self.mask_image,
            self.printed_nominal,
            self.printed_max,
            self.printed_min,
        ))
        if len({array.shape for array in arrays}) != 1 or arrays[0].ndim != 2:
            raise ValueError("solver 的 mask/nominal/max/min 必须是同形状二维图")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("solver 的 mask/nominal/max/min 不能包含 NaN 或无穷值")
        if any(np.any((array < 0) | (array > 1)) for array in arrays):
            raise ValueError("solver 的 mask/nominal/max/min 必须使用 [0,1] 归一化栅格")
        frozen_arrays = []
        for array in arrays:
            frozen = np.ascontiguousarray(array).copy()
            frozen.setflags(write=False)
            frozen_arrays.append(frozen)
        normalized_signs = []
        for raw_point_id, raw_sign in tuple(self.recipe_epe_signs):
            point_id = str(raw_point_id)
            sign = float(raw_sign)
            if not point_id:
                raise ValueError("recipe_epe_signs 的 point_id 不能为空")
            if not np.isfinite(sign) or sign not in {-1.0, 0.0, 1.0}:
                raise ValueError("recipe_epe_signs 只能包含有限的 -1/0/1")
            normalized_signs.append((point_id, sign))
        point_ids = [point_id for point_id, _ in normalized_signs]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("recipe_epe_signs 不允许重复 point_id")
        mask_sha256 = require_sha256(self.mask_sha256, "mask_sha256")
        if mask_sha256 != array_sha256(frozen_arrays[0]):
            raise ValueError("mask_sha256 与 mask_image 内容不一致")
        object.__setattr__(self, "mask_image", frozen_arrays[0])
        object.__setattr__(self, "printed_nominal", frozen_arrays[1])
        object.__setattr__(self, "printed_max", frozen_arrays[2])
        object.__setattr__(self, "printed_min", frozen_arrays[3])
        object.__setattr__(self, "recipe_epe_signs", tuple(normalized_signs))
        object.__setattr__(self, "mask_sha256", mask_sha256)
        object.__setattr__(self, "internal_trace", tuple(self.internal_trace))

    def sign_by_point_id(self) -> Dict[str, float]:
        """返回逐点内部 mask 移动方向的副本。"""
        return {str(point_id): float(value) for point_id, value in self.recipe_epe_signs}


@dataclass(frozen=True)
class GoldenEvaluation:
    """保存独立 Golden evaluator 的指标、原始损失和来源身份。"""

    metrics: GoldenMetrics
    raw_weighted_loss: float
    evaluator_version: str
    evaluator_source_sha256: str
    evaluator_contract_sha256: str
    evaluator_parameters: Tuple[Tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.raw_weighted_loss) or self.raw_weighted_loss < 0:
            raise ValueError("raw_weighted_loss 必须是有限非负数")
        if not self.evaluator_version:
            raise ValueError("Golden evaluator 必须记录版本")
        source_sha256 = require_sha256(self.evaluator_source_sha256, "Golden evaluator source_sha256")
        contract_sha256 = require_sha256(
            self.evaluator_contract_sha256,
            "Golden evaluator contract_sha256",
        )
        normalized_parameters = tuple(
            (str(name), _freeze_parameter(value, str(name)))
            for name, value in tuple(self.evaluator_parameters)
        )
        names = [name for name, _ in normalized_parameters]
        if any(not name for name in names):
            raise ValueError("Golden evaluator 参数名不能为空")
        if len(names) != len(set(names)):
            raise ValueError("Golden evaluator 参数名不允许重复")
        object.__setattr__(self, "evaluator_source_sha256", source_sha256)
        object.__setattr__(self, "evaluator_contract_sha256", contract_sha256)
        object.__setattr__(self, "evaluator_parameters", normalized_parameters)

    def parameters_dict(self) -> Dict[str, object]:
        """返回冻结 Golden evaluator 参数的 JSON 友好副本。"""
        return {
            str(name): _thaw_parameter(value)
            for name, value in self.evaluator_parameters
        }
