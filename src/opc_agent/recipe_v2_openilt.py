"""本模块实现 Recipe PPO v2 的只读 OpenILT solver、正式 Golden evaluator 与单图预检。

输入为锁定且干净的 OpenILT clone、ICCAD13 GLP、全局 FRAG 参数和完整逐点法向 Recipe；输出为
真实光刻栅格、固定 target Golden 指标、动作/probe 合法率及少量逐点 sensitivity 证据。本模块只
导入上游函数，不修改 OpenILT，也不启动 PPO；所有实验工件由调用方写入项目 ``runs/``。
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .recipe_v2 import (
    GEOMETRY_ADAPTER_VERSION,
    NORMAL_PROBE_SEMANTICS_VERSION,
    RASTER_MAPPING_VERSION,
    UPSTREAM_MIN_FRAGMENT_RULE_VERSION,
    build_action_candidates,
    build_probe_window,
    dissect_global_fragments,
    identity_raster_coordinate_system_sha256,
    LocalEPEEpisode,
)
from .recipe_v2_contract import (
    EPE_DENSE_PROTOCOL,
    EPE_TERMINAL_PROTOCOL,
    EPEControlPoint,
    FragmentParameters,
    GoldenEvaluation,
    GoldenMetrics,
    V2SolverResult,
    array_sha256,
)
from .recipe_v2_visualization import (
    PPO_INPUT_SELECTION_POLICY,
    save_v2_ppo_input_examples,
)


OPENILT_GOLDEN_EVALUATOR_VERSION = "openilt-evaluation-epecheck-v1"
OPENILT_GOLDEN_CONFIG_EVALUATOR = "openilt_evaluation_epecheck"
CONTROL_CONFLICT_POLICY = "both-sides-conflict-stay-v1"
PREFLIGHT_CONFLICT_POLICY = CONTROL_CONFLICT_POLICY
PREFLIGHT_POINT_SELECTION_POLICY = (
    "baseline-any-step-active-nonconflict-edge-stratified-v2"
)


def _sha256_json(payload: object) -> str:
    """计算规范 JSON 的稳定 SHA256。"""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _number(value: object) -> float:
    """把单元素 Torch 张量或普通数值转换为 Python float。"""
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value)


def _result_control_conflicts(result: V2SolverResult) -> Tuple[dict, ...]:
    """从 solver 内部轨迹提取带轮次的双侧控制冲突记录。"""
    conflicts = []
    for record in result.internal_trace:
        inner_step = int(record.get("inner_step", -1))
        for raw in record.get("control_conflicts", ()):
            normalized = dict(raw)
            normalized["inner_step"] = inner_step
            conflicts.append(normalized)
    return tuple(conflicts)


def _baseline_active_point_ids(result: V2SolverResult) -> Tuple[str, ...]:
    """返回基线 solver 任一内部轮次中出现过非零控制方向的点。"""
    active = set()
    trace_has_point_ids = False
    for record in result.internal_trace:
        if "active_point_ids" not in record:
            continue
        trace_has_point_ids = True
        active.update(str(point_id) for point_id in record["active_point_ids"])
    if not trace_has_point_ids:
        active.update(
            point_id
            for point_id, sign in result.recipe_epe_signs
            if float(sign) != 0.0
        )
    return tuple(sorted(active))


def _round_robin_by_geometry(
    points: Sequence[EPEControlPoint],
) -> Tuple[EPEControlPoint, ...]:
    """按法线和角点类型稳定轮询，避免样本被单一几何类型占满。"""
    buckets = {}
    for point in points:
        key = (
            tuple(point.normal_xy),
            int(point.start_corner_type),
            int(point.end_corner_type),
        )
        buckets.setdefault(key, []).append(point)
    ordered = []
    keys = sorted(buckets)
    while keys:
        next_keys = []
        for key in keys:
            bucket = buckets[key]
            ordered.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return tuple(ordered)


def _select_sensitivity_points(
    points: Sequence[EPEControlPoint], limit: int
) -> Tuple[EPEControlPoint, ...]:
    """先覆盖不同原始边，再用剩余 segment 补足，两阶段均按几何特征轮询。"""
    first_per_edge = []
    remaining = []
    seen_edges = set()
    for point in points:
        edge_key = (int(point.polygon_index), int(point.source_edge_index))
        if edge_key in seen_edges:
            remaining.append(point)
        else:
            seen_edges.add(edge_key)
            first_per_edge.append(point)
    ordered = (
        _round_robin_by_geometry(first_per_edge)
        + _round_robin_by_geometry(remaining)
    )
    return ordered[:limit]


def _sensitivity_actions(
    preflight: Mapping[str, object], action_offsets: Sequence[float]
) -> Tuple[float, ...]:
    """读取显式 sensitivity 动作，并拒绝零动作、重复值和动作表外取值。"""
    raw_actions = preflight.get("sensitivity_actions_nm")
    if raw_actions is None:
        nonzero = [value for value in action_offsets if value != 0.0]
        preferred = [value for value in (-10.0, 10.0) if value in nonzero]
        raw_actions = preferred or nonzero[:2]
    if isinstance(raw_actions, (str, bytes)):
        raise ValueError("preflight.sensitivity_actions_nm 必须是数值列表")
    try:
        selected = tuple(float(value) for value in raw_actions)
    except (TypeError, ValueError) as error:
        raise ValueError("preflight.sensitivity_actions_nm 必须是数值列表") from error
    if len(selected) < 2:
        raise ValueError("preflight.sensitivity_actions_nm 至少需要两个非零动作")
    if any(not np.isfinite(value) or value == 0.0 for value in selected):
        raise ValueError("preflight.sensitivity_actions_nm 只能包含有限非零动作")
    if len(set(selected)) != len(selected):
        raise ValueError("preflight.sensitivity_actions_nm 不允许重复动作")
    action_set = {float(value) for value in action_offsets}
    if any(value not in action_set for value in selected):
        raise ValueError("preflight.sensitivity_actions_nm 必须是 EPE 动作表的子集")
    return selected


def _validate_configured_golden_identity(
    golden_config: Mapping[str, object],
    evaluator: object,
    layout_parent: str,
    require_layout_contract: bool = False,
) -> Tuple[str, ...]:
    """核对全局与逐版图 Golden 字段，未冻结的 null 字段保持跳过。"""
    if golden_config.get("evaluator") != OPENILT_GOLDEN_CONFIG_EVALUATOR:
        raise ValueError(
            "golden.evaluator 必须为 " + OPENILT_GOLDEN_CONFIG_EVALUATOR
        )
    actual = {
        "constraint_coordinate": int(evaluator.epe_constraint_coordinate),
        "source_sha256": str(evaluator.evaluator_source_sha256),
        "sampling_state_sha256": str(evaluator.sampling_state_sha256),
        "coordinate_system_sha256": str(evaluator.coordinate_system_sha256),
        "evaluator_contract_sha256": str(evaluator.evaluator_contract_sha256),
    }
    verified = []
    for field, actual_value in actual.items():
        expected = golden_config.get(field)
        if expected is None:
            continue
        if field == "constraint_coordinate":
            matches = isinstance(expected, int) and not isinstance(expected, bool)
            matches = matches and expected == actual_value
        else:
            matches = str(expected).lower() == str(actual_value).lower()
        if not matches:
            raise RuntimeError(
                f"Golden 身份不匹配：golden.{field}={expected!r}，实际为 {actual_value!r}"
            )
        verified.append(field)
    layout_contracts = golden_config.get("layout_contracts")
    if layout_contracts is not None:
        if not isinstance(layout_contracts, Mapping):
            raise TypeError("golden.layout_contracts 必须是逐版图映射")
        expected_layout = layout_contracts.get(str(layout_parent))
        if expected_layout is None and not require_layout_contract:
            return tuple(verified)
        if not isinstance(expected_layout, Mapping):
            raise RuntimeError(
                f"Golden 身份缺少版图 {layout_parent} 的冻结 layout contract"
            )
        for field in (
            "sampling_state_sha256",
            "coordinate_system_sha256",
            "evaluator_contract_sha256",
        ):
            expected = expected_layout.get(field)
            if str(expected).lower() != str(actual[field]).lower():
                raise RuntimeError(
                    "Golden 逐版图身份不匹配："
                    f"golden.layout_contracts.{layout_parent}.{field}="
                    f"{expected!r}，实际为 {actual[field]!r}"
                )
            verified.append(f"layout_contracts.{layout_parent}.{field}")
    return tuple(verified)


def _validate_openilt(openilt_dir: Path, expected_commit: str) -> str:
    """核对 OpenILT 入口、固定提交和 tracked diff。"""
    required = (
        openilt_dir / "pyilt" / "simpleopc.py",
        openilt_dir / "pyilt" / "evaluation.py",
        openilt_dir / "utils" / "polygon.py",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"未找到完整 OpenILT v2 依赖：{openilt_dir}")
    revision = subprocess.run(
        ["git", "-C", str(openilt_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != str(expected_commit):
        raise RuntimeError("OpenILT 提交与 v2 配置不一致")
    diff = subprocess.run(
        ["git", "-C", str(openilt_dir), "diff", "--quiet", "HEAD", "--"],
        check=False,
    )
    if diff.returncode == 1:
        raise RuntimeError("OpenILT tracked 文件存在修改")
    if diff.returncode != 0:
        raise RuntimeError("无法核验 OpenILT tracked diff")
    return revision


class OpenILTV2Solver:
    """按逐点法向 crossing 控制内部 segment 法向移动的只读 OpenILT solver。"""

    def __init__(
        self,
        openilt_dir: Path,
        expected_commit: str,
        layout_path: Path,
        fragment_parameters: FragmentParameters,
        reward_weights: Mapping[str, float],
        control_probe_distance_nm: float = 16.0,
        lithography_config: Path = Path("config/lithosimple.txt"),
        simulator: str = "simple",
        image_size: Tuple[int, int] = (2048, 2048),
        nm_per_coordinate: float = 1.0,
        inner_step_sizes_nm: Sequence[float] = (8, 8, 8, 8, 4, 4, 4, 4),
        mask_displacement_limit_nm: float = 24.0,
        threshold: float = 0.5,
        normal_probe_coordinate: int = 2,
        raster_scale: float = 1.0,
        raster_offset_xy: Tuple[int, int] = (0, 0),
        control_conflict_policy: str = CONTROL_CONFLICT_POLICY,
    ):
        import torch

        self.openilt_dir = Path(openilt_dir).resolve()
        self.layout_path = Path(layout_path).resolve()
        if not self.layout_path.is_file():
            raise FileNotFoundError(f"v2 GLP 不存在：{self.layout_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("v2 OpenILT preflight 要求 CUDA GPU")
        if simulator not in {"simple", "exact"}:
            raise ValueError("simulator 必须是 simple 或 exact")
        if control_conflict_policy != CONTROL_CONFLICT_POLICY:
            raise ValueError(
                f"control_conflict_policy 必须为 {CONTROL_CONFLICT_POLICY}"
            )
        self.control_conflict_policy = str(control_conflict_policy)
        self.revision = _validate_openilt(self.openilt_dir, expected_commit)
        self.fragment_parameters = fragment_parameters
        self.nm_per_coordinate = float(nm_per_coordinate)
        if not np.isfinite(self.nm_per_coordinate) or self.nm_per_coordinate <= 0:
            raise ValueError("nm_per_coordinate 必须是有限正数")
        self.control_probe_distance_nm = float(control_probe_distance_nm)
        if not np.isfinite(self.control_probe_distance_nm) or self.control_probe_distance_nm <= 0:
            raise ValueError("control_probe_distance_nm 必须是有限正数")
        try:
            raw_image_size = tuple(image_size)
        except TypeError as exc:
            raise TypeError("image_size 必须是两个正整数") from exc
        if len(raw_image_size) != 2 or any(
            not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            for value in raw_image_size
        ):
            raise TypeError("image_size 必须是两个正整数")
        self.image_size = (int(raw_image_size[0]), int(raw_image_size[1]))
        if min(self.image_size) <= 0:
            raise ValueError("image_size 必须是两个正整数")
        self.inner_step_sizes_nm = tuple(float(value) for value in inner_step_sizes_nm)
        if not self.inner_step_sizes_nm or any(
            not np.isfinite(value) or value <= 0 for value in self.inner_step_sizes_nm
        ):
            raise ValueError("inner_step_sizes_nm 必须是非空正数序列")
        self.mask_displacement_limit_nm = float(mask_displacement_limit_nm)
        if not np.isfinite(self.mask_displacement_limit_nm) or self.mask_displacement_limit_nm <= 0:
            raise ValueError("mask_displacement_limit_nm 必须大于零")
        self.threshold = float(threshold)
        if not np.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold 必须是 [0,1] 内有限数")
        self.reward_weights = {name: float(value) for name, value in reward_weights.items()}
        GoldenMetrics(0, 0, 0).weighted_loss(self.reward_weights)

        config_path = Path(lithography_config)
        if not config_path.is_absolute():
            config_path = self.openilt_dir / config_path
        config_path = config_path.resolve()
        try:
            config_relative = config_path.relative_to(self.openilt_dir).as_posix()
        except ValueError as exc:
            raise ValueError("lithography_config 必须位于锁定 OpenILT clone 内") from exc
        if not config_path.is_file():
            raise FileNotFoundError(f"OpenILT 光刻配置不存在：{config_path}")
        source_paths = (
            "pyilt/simpleopc.py",
            "pyilt/evaluation.py",
            "utils/polygon.py",
            "pycommon/glp.py",
            "pycommon/settings.py",
            "pycommon/utils.py",
            f"pylitho/{simulator}.py",
            config_relative,
        )
        missing_sources = [
            relative for relative in source_paths if not (self.openilt_dir / relative).is_file()
        ]
        if missing_sources:
            raise RuntimeError("OpenILT 缺少实际运行依赖：" + ", ".join(missing_sources))
        self.source_sha256 = {
            relative: hashlib.sha256((self.openilt_dir / relative).read_bytes()).hexdigest()
            for relative in source_paths
        }
        self._torch = torch
        openilt_text = str(self.openilt_dir)
        if openilt_text not in sys.path:
            sys.path.insert(0, openilt_text)
        previous_cwd = Path.cwd()
        try:
            os.chdir(str(self.openilt_dir))
            self._glp = importlib.import_module("pycommon.glp")
            self._polygon = importlib.import_module("utils.polygon")
            self._evaluation = importlib.import_module("pyilt.evaluation")
            self._settings = importlib.import_module("pycommon.settings")
            lithosim = importlib.import_module(f"pylitho.{simulator}")
            for imported, relative in (
                (self._glp, "pycommon/glp.py"),
                (self._polygon, "utils/polygon.py"),
                (self._evaluation, "pyilt/evaluation.py"),
                (self._settings, "pycommon/settings.py"),
                (lithosim, f"pylitho/{simulator}.py"),
            ):
                actual_source = Path(imported.__file__).resolve()
                expected_source = (self.openilt_dir / relative).resolve()
                if actual_source != expected_source:
                    raise RuntimeError(
                        f"OpenILT 模块来源漂移：{relative} 实际加载自 {actual_source}"
                    )
            self._litho = lithosim.LithoSim(str(config_path))
        finally:
            os.chdir(str(previous_cwd))
        self.layout_sha256 = hashlib.sha256(self.layout_path.read_bytes()).hexdigest()
        self._polygons = self._load_polygons()
        self._target_array = self._render_target()
        self.target_image = self._target_array.copy()
        self._target = torch.as_tensor(
            self._target_array, dtype=torch.float32, device="cuda"
        )
        self._vposes, self._hposes = self._evaluation.boundaries(self._target)
        geometry = dissect_global_fragments(
            polygons=self._polygons,
            target_image=self._target_array,
            fragment_parameters=self.fragment_parameters,
            nm_per_coordinate=self.nm_per_coordinate,
            dissect_fn=self._polygon.dissect,
            normal_probe_coordinate=normal_probe_coordinate,
            raster_scale=raster_scale,
            raster_offset_xy=raster_offset_xy,
        )
        self.epe_points = geometry.epe_points
        self._segments_by_polygon = geometry.segments_by_polygon
        self.fragmentation_sha256 = geometry.fragmentation_sha256

    def _load_polygons(self):
        """读取、居中并规范化 GLP 正交多边形。"""
        design = self._glp.Design(str(self.layout_path), down=1)
        design.center(self.image_size[0], self.image_size[1], 0, 0)
        polygons = tuple(
            tuple((int(point[0]), int(point[1])) for point in polygon)
            for polygon in design.polygons
        )
        if not polygons:
            raise RuntimeError("GLP 不包含 target polygon")
        return polygons

    def _render_target(self) -> np.ndarray:
        """栅格化固定原始 target。"""
        image = self._polygon.poly2img(
            [list(polygon) for polygon in self._polygons],
            self.image_size[0],
            self.image_size[1],
            scale=1,
        )
        return (np.asarray(image, dtype=np.float32) / 255.0).astype(np.float32)

    def _render_mask(self, mask_offsets_nm: np.ndarray) -> np.ndarray:
        """仅沿各 segment 的冻结外法向移动 mask 并重建 polygon。"""
        moved_by_polygon = []
        flat_index = 0
        for polygon_segments in self._segments_by_polygon:
            moved_segments = []
            for start, end in polygon_segments:
                point = self.epe_points[flat_index]
                delta = int(np.rint(mask_offsets_nm[flat_index] / self.nm_per_coordinate))
                dx = point.normal_xy[0] * delta
                dy = point.normal_xy[1] * delta
                moved_segments.append([
                    [start[0] + dx, start[1] + dy],
                    [end[0] + dx, end[1] + dy],
                ])
                flat_index += 1
            moved_by_polygon.append(moved_segments)
        polygons = [self._polygon.segs2poly(segments) for segments in moved_by_polygon]
        image = self._polygon.poly2img(
            polygons, self.image_size[0], self.image_size[1], scale=1
        )
        return (np.asarray(image, dtype=np.float32) / 255.0).astype(np.float32)

    def _simulate(self, mask_array: np.ndarray):
        """运行一次真实 OpenILT 光刻并返回二值 nominal/max/min 与 Golden 原始指标。"""
        torch = self._torch
        mask = torch.as_tensor(mask_array, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            nominal, maximum, minimum = self._litho(mask)
            nominal = (nominal >= self.threshold).to(dtype=torch.float32)
            maximum = (maximum >= self.threshold).to(dtype=torch.float32)
            minimum = (minimum >= self.threshold).to(dtype=torch.float32)
            l2 = torch.nn.functional.mse_loss(nominal, self._target, reduction="sum")
            pvb = torch.sum(maximum != minimum)
            epe_in, epe_out, _ = self._evaluation.epecheck(
                nominal, self._target, self._vposes, self._hposes
            )
        metrics = GoldenMetrics(
            l2=_number(l2), epe=_number(epe_in) + _number(epe_out), pvb=_number(pvb)
        )
        return (
            nominal.detach().cpu().numpy().astype(np.float32),
            maximum.detach().cpu().numpy().astype(np.float32),
            minimum.detach().cpu().numpy().astype(np.float32),
            metrics,
        )

    def _measure_recipe_signs(
        self, normal_offsets_nm: Mapping[str, float], nominal: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[dict, ...]]:
        """测量逐点移动符号，并把双侧同时违规作为显式冲突返回。"""
        printed = np.asarray(nominal, dtype=bool)
        if printed.shape != self._target_array.shape:
            raise ValueError("printed nominal 与冻结 target 形状不一致")
        signs = np.zeros(len(self.epe_points), dtype=np.float32)
        conflicts = []
        for index, point in enumerate(self.epe_points):
            probe = build_probe_window(
                point=point,
                normal_offset_nm=normal_offsets_nm[point.point_id],
                probe_distance_nm=self.control_probe_distance_nm,
                nm_per_coordinate=self.nm_per_coordinate,
                target_image=self._target_array,
            )
            if not probe.target_valid:
                raise ValueError(
                    f"EPE point {point.point_id} 的动作不能形成一内一外合法 probe"
                )
            inner = probe.inner_xy
            outer = probe.outer_xy
            underprint = not bool(printed[inner[1], inner[0]])
            overprint = bool(printed[outer[1], outer[0]])
            if underprint and overprint:
                conflicts.append({
                    "point_id": point.point_id,
                    "normal_offset_nm": float(normal_offsets_nm[point.point_id]),
                    "inner_xy": [int(inner[0]), int(inner[1])],
                    "outer_xy": [int(outer[0]), int(outer[1])],
                    "underprint": True,
                    "overprint": True,
                })
            elif underprint:
                signs[index] = 1.0
            elif overprint:
                signs[index] = -1.0
        return signs, tuple(conflicts)

    def _recipe_signs(
        self, normal_offsets_nm: Mapping[str, float], nominal: np.ndarray
    ) -> np.ndarray:
        """保留严格唯一方向诊断；正式 solver 使用可追溯的 stay 协议。"""
        signs, conflicts = self._measure_recipe_signs(normal_offsets_nm, nominal)
        if conflicts:
            point_ids = [item["point_id"] for item in conflicts]
            raise RuntimeError(
                "EPE point 的 inner/outer 同时违规，移动方向不唯一："
                + ", ".join(point_ids)
            )
        return signs

    def _solve(self, normal_offsets_nm: Mapping[str, float]) -> V2SolverResult:
        """运行内部 OPC；双侧冲突按冻结协议记录并令该点 stay。"""
        expected = {point.point_id for point in self.epe_points}
        if set(normal_offsets_nm) != expected:
            raise ValueError("v2 solver 要求完整且无额外 point_id 的 Recipe")
        offsets = {point_id: float(value) for point_id, value in normal_offsets_nm.items()}
        if any(not np.isfinite(value) for value in offsets.values()):
            raise ValueError("v2 Recipe offset 必须是有限数")
        mask_offsets = np.zeros(len(self.epe_points), dtype=np.float64)
        best = None
        trace = []
        previous_signs = np.zeros(len(self.epe_points), dtype=np.float32)
        for inner_step, step_size in enumerate((0.0,) + self.inner_step_sizes_nm):
            if inner_step:
                mask_offsets = np.clip(
                    mask_offsets + previous_signs * step_size,
                    -self.mask_displacement_limit_nm,
                    self.mask_displacement_limit_nm,
                )
            mask = self._render_mask(mask_offsets)
            nominal, maximum, minimum, metrics = self._simulate(mask)
            signs, conflicts = self._measure_recipe_signs(offsets, nominal)
            loss = metrics.weighted_loss(self.reward_weights)
            record = {
                "inner_step": inner_step,
                "step_size_nm": float(step_size),
                "raw_metrics": metrics.as_dict(),
                "raw_weighted_loss": loss,
                "active_sign_count": int(np.count_nonzero(signs)),
                "active_point_ids": tuple(
                    point.point_id
                    for index, point in enumerate(self.epe_points)
                    if float(signs[index]) != 0.0
                ),
                "control_conflict_count": len(conflicts),
                "control_conflicts": conflicts,
                "control_conflict_policy": self.control_conflict_policy,
                "conflicts_forced_to_stay_by_policy": bool(conflicts),
                "mask_displacement_min_nm": float(mask_offsets.min()),
                "mask_displacement_max_nm": float(mask_offsets.max()),
            }
            trace.append(record)
            if best is None or loss < best[0]:
                best = (loss, mask.copy(), nominal.copy(), maximum.copy(), minimum.copy(), signs.copy())
            previous_signs = signs
        assert best is not None
        _loss, mask, nominal, maximum, minimum, signs = best
        return V2SolverResult(
            mask_image=mask,
            printed_nominal=nominal,
            printed_max=maximum,
            printed_min=minimum,
            recipe_epe_signs=tuple(
                (point.point_id, float(signs[index]))
                for index, point in enumerate(self.epe_points)
            ),
            mask_sha256=array_sha256(mask),
            internal_trace=tuple(trace),
        )

    def solve(self, normal_offsets_nm: Mapping[str, float]) -> V2SolverResult:
        """按正式保守 stay 协议运行完整 Recipe，同时保留冲突证据。"""
        return self._solve(normal_offsets_nm)

    def solve_diagnostic(self, normal_offsets_nm: Mapping[str, float]) -> V2SolverResult:
        """保留 preflight 语义别名；运算规则与正式 solver 完全一致。"""
        return self._solve(normal_offsets_nm)


class OpenILTGoldenEvaluator:
    """使用锁定 ``pyilt/evaluation.py::epecheck`` 和固定 target 评价 solver 结果。"""

    evaluator_version = OPENILT_GOLDEN_EVALUATOR_VERSION

    def __init__(self, solver: OpenILTV2Solver, reward_weights: Mapping[str, float]):
        self._solver = solver
        self._weights = {name: float(value) for name, value in reward_weights.items()}
        GoldenMetrics(0, 0, 0).weighted_loss(self._weights)
        self.nm_per_coordinate = solver.nm_per_coordinate
        self.frozen_target_sha256 = array_sha256(
            np.asarray(solver.target_image >= solver.threshold, dtype=bool)
        )
        source = solver.openilt_dir / "pyilt" / "evaluation.py"
        self.evaluator_source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        self.sampling_state_sha256 = _sha256_json({
            "vposes": solver._vposes.detach().cpu().tolist(),
            "hposes": solver._hposes.detach().cpu().tolist(),
        })
        self.coordinate_system_sha256 = identity_raster_coordinate_system_sha256(
            self.frozen_target_sha256, self.nm_per_coordinate
        )
        self.epe_constraint_coordinate = int(solver._evaluation.EPE_CONSTRAINT)
        self.evaluator_contract_sha256 = _sha256_json({
            "version": self.evaluator_version,
            "source_sha256": self.evaluator_source_sha256,
            "target_sha256": self.frozen_target_sha256,
            "sampling_state_sha256": self.sampling_state_sha256,
            "coordinate_system_sha256": self.coordinate_system_sha256,
            "epe_constraint_coordinate": self.epe_constraint_coordinate,
            "reward_weights": dict(sorted(self._weights.items())),
        })

    def evaluate(self, result: V2SolverResult) -> GoldenEvaluation:
        """只使用固定 target/boundaries 计算 L2、EPE 和 PVB。"""
        torch = self._solver._torch
        # V2SolverResult 会把 NumPy 工件冻结为只读；显式复制后再交给 Torch，避免未定义写行为警告。
        nominal = torch.as_tensor(
            np.array(result.printed_nominal, dtype=np.float32, copy=True, order="C"),
            dtype=torch.float32,
            device="cuda",
        )
        maximum = torch.as_tensor(
            np.array(result.printed_max, dtype=np.float32, copy=True, order="C"),
            dtype=torch.float32,
            device="cuda",
        )
        minimum = torch.as_tensor(
            np.array(result.printed_min, dtype=np.float32, copy=True, order="C"),
            dtype=torch.float32,
            device="cuda",
        )
        with torch.no_grad():
            l2 = torch.nn.functional.mse_loss(
                nominal, self._solver._target, reduction="sum"
            )
            pvb = torch.sum(maximum != minimum)
            epe_in, epe_out, _ = self._solver._evaluation.epecheck(
                nominal,
                self._solver._target,
                self._solver._vposes,
                self._solver._hposes,
            )
        metrics = GoldenMetrics(
            l2=_number(l2), epe=_number(epe_in) + _number(epe_out), pvb=_number(pvb)
        )
        return GoldenEvaluation(
            metrics=metrics,
            raw_weighted_loss=metrics.weighted_loss(self._weights),
            evaluator_version=self.evaluator_version,
            evaluator_source_sha256=self.evaluator_source_sha256,
            evaluator_contract_sha256=self.evaluator_contract_sha256,
            evaluator_parameters=(
                ("epe_constraint_coordinate", self.epe_constraint_coordinate),
                ("nm_per_coordinate", self.nm_per_coordinate),
                ("reward_weights", dict(sorted(self._weights.items()))),
            ),
        )


def _build_v2_openilt_solver_and_evaluator(
    config: dict,
    layout_parent: str,
    require_layout_contract: bool = False,
) -> Tuple[OpenILTV2Solver, OpenILTGoldenEvaluator, Tuple[str, ...]]:
    """从同一冻结配置构造 preflight/smoke 共用的 solver 与 Golden evaluator。"""
    data = config["data"]
    openilt = config["openilt"]
    recipe = config["recipe_v2"]
    reward_weights = recipe["reward"]["weights"]
    fragments = recipe["fragment_parameters_nm"]
    solver_config = recipe["solver"]
    geometry_config = recipe["geometry_adapter"]
    expected_geometry_versions = {
        "version": GEOMETRY_ADAPTER_VERSION,
        "raster_mapping_version": RASTER_MAPPING_VERSION,
        "normal_probe_semantics_version": NORMAL_PROBE_SEMANTICS_VERSION,
        "minimum_fragment_rule_version": UPSTREAM_MIN_FRAGMENT_RULE_VERSION,
    }
    for name, expected in expected_geometry_versions.items():
        if geometry_config.get(name) != expected:
            raise ValueError(
                f"recipe_v2.geometry_adapter.{name} 必须为冻结值 {expected}"
            )
    solver = OpenILTV2Solver(
        openilt_dir=Path(data["openilt_dir"]),
        expected_commit=openilt["commit"],
        layout_path=Path(data["iccad13_dir"]) / f"{layout_parent}.glp",
        fragment_parameters=FragmentParameters(fragments["corner"], fragments["uniform"]),
        reward_weights=reward_weights,
        control_probe_distance_nm=recipe["epe_probe_distance_nm"],
        lithography_config=Path(solver_config["lithography_config"]),
        simulator=str(solver_config["simulator"]),
        image_size=tuple(int(value) for value in solver_config["image_size"]),
        nm_per_coordinate=recipe["nm_per_coordinate"],
        inner_step_sizes_nm=tuple(
            float(value) for value in solver_config["inner_step_sizes_nm"]
        ),
        mask_displacement_limit_nm=float(
            solver_config["mask_displacement_limit_nm"]
        ),
        threshold=float(solver_config["threshold"]),
        normal_probe_coordinate=geometry_config["normal_probe_coordinate"],
        raster_scale=geometry_config["raster_scale"],
        raster_offset_xy=geometry_config["raster_offset_xy"],
        control_conflict_policy=recipe["control_conflict_policy"],
    )
    evaluator = OpenILTGoldenEvaluator(solver, reward_weights)
    verified_golden_fields = _validate_configured_golden_identity(
        config.get("golden", {}),
        evaluator,
        layout_parent,
        require_layout_contract=require_layout_contract,
    )
    return solver, evaluator, verified_golden_fields


def run_v2_openilt_preflight(config: dict, layout_parent: str = "M1_test1") -> dict:
    """运行单图几何扫描和两点真实 sensitivity，返回可序列化诊断证据。"""
    started = time.monotonic()
    openilt = config["openilt"]
    recipe = config["recipe_v2"]
    reward_weights = recipe["reward"]["weights"]
    solver, evaluator, verified_golden_fields = (
        _build_v2_openilt_solver_and_evaluator(config, layout_parent)
    )
    action_offsets = tuple(float(value) for value in recipe["epe_normal_offsets_nm"])
    preflight = config.get("preflight", {})
    if preflight.get("control_conflict_policy") != PREFLIGHT_CONFLICT_POLICY:
        raise ValueError(
            f"preflight.control_conflict_policy 必须为 {PREFLIGHT_CONFLICT_POLICY}"
        )
    probe_distances = tuple(
        float(value) for value in preflight.get("probe_distance_candidates_nm", [16, 24, 32, 48, 56, 64])
    )
    geometry_scan = []
    for distance in probe_distances:
        valid = 0
        total = len(solver.epe_points) * len(action_offsets)
        per_action = {str(value): 0 for value in action_offsets}
        for point in solver.epe_points:
            candidates = build_action_candidates(
                point, action_offsets, distance, solver.nm_per_coordinate, solver.target_image
            )
            for candidate in candidates:
                if candidate.valid:
                    valid += 1
                    per_action[str(candidate.probe.normal_offset_nm)] += 1
        geometry_scan.append({
            "probe_distance_nm": distance,
            "valid_candidates": valid,
            "total_candidates": total,
            "valid_fraction": valid / total,
            "valid_points_by_action": per_action,
            "all_actions_valid_for_all_points": valid == total,
        })
    zero_recipe = {point.point_id: 0.0 for point in solver.epe_points}
    baseline_result = solver.solve_diagnostic(zero_recipe)
    baseline = evaluator.evaluate(baseline_result)
    baseline_conflicts = _result_control_conflicts(baseline_result)
    all_conflicts = [
        {**item, "recipe_case": "baseline"}
        for item in baseline_conflicts
    ]
    sensitivity = []
    point_limit = int(preflight.get("sensitivity_point_limit", 8))
    if point_limit <= 0:
        raise ValueError("preflight.sensitivity_point_limit 必须为正整数")
    selected_actions = _sensitivity_actions(preflight, action_offsets)
    all_point_ids = {point.point_id for point in solver.epe_points}
    baseline_active_ids = set(_baseline_active_point_ids(baseline_result))
    unknown_active_ids = baseline_active_ids - all_point_ids
    if unknown_active_ids:
        raise RuntimeError(
            "baseline solver 返回未知 active point_id："
            + ", ".join(sorted(unknown_active_ids))
        )
    baseline_conflict_ids = {item["point_id"] for item in baseline_conflicts}
    eligible_active_points = []
    for point in solver.epe_points:
        if point.point_id not in baseline_active_ids:
            continue
        if point.point_id in baseline_conflict_ids:
            continue
        candidates = build_action_candidates(
            point,
            selected_actions,
            recipe["epe_probe_distance_nm"],
            solver.nm_per_coordinate,
            solver.target_image,
        )
        if len(candidates) == len(selected_actions) and all(item.valid for item in candidates):
            eligible_active_points.append(point)
    selected_points = _select_sensitivity_points(eligible_active_points, point_limit)
    for point in selected_points:
        for offset in selected_actions:
            candidate_recipe = dict(zero_recipe)
            candidate_recipe[point.point_id] = offset
            result = solver.solve_diagnostic(candidate_recipe)
            golden = evaluator.evaluate(result)
            conflicts = _result_control_conflicts(result)
            all_conflicts.extend(
                {
                    **item,
                    "recipe_case": "single_point_action",
                    "action_point_id": point.point_id,
                    "action_offset_nm": float(offset),
                }
                for item in conflicts
            )
            sensitivity.append({
                "point_id": point.point_id,
                "offset_nm": offset,
                "status": "evaluated",
                "mask_sha256": result.mask_sha256,
                "mask_changed_from_baseline": result.mask_sha256 != baseline_result.mask_sha256,
                "raw_metrics": golden.metrics.as_dict(),
                "raw_weighted_loss": golden.raw_weighted_loss,
                "loss_delta_from_baseline": golden.raw_weighted_loss - baseline.raw_weighted_loss,
                "control_conflict_count": len(conflicts),
                "control_conflict_point_ids": sorted({
                    item["point_id"] for item in conflicts
                }),
            })
    baseline_signature = (
        baseline_result.mask_sha256,
        _sha256_json(baseline.metrics.as_dict()),
    )
    response_signatures = {
        (
            item["mask_sha256"],
            _sha256_json(item["raw_metrics"]),
        )
        for item in sensitivity
        if item["status"] == "evaluated"
    }
    point_response_summary = []
    responsive_points = 0
    for point in selected_points:
        point_results = [
            item
            for item in sensitivity
            if item["point_id"] == point.point_id and item["status"] == "evaluated"
        ]
        signature_offsets = {baseline_signature: [0.0]}
        for item in point_results:
            signature = (item["mask_sha256"], _sha256_json(item["raw_metrics"]))
            signature_offsets.setdefault(signature, []).append(float(item["offset_nm"]))
        changed_actions = [
            float(item["offset_nm"])
            for item in point_results
            if (
                item["mask_sha256"],
                _sha256_json(item["raw_metrics"]),
            ) != baseline_signature
        ]
        responsive = bool(changed_actions)
        responsive_points += int(responsive)
        response_classes = []
        for class_id, (signature, offsets) in enumerate(signature_offsets.items()):
            response_classes.append({
                "class_id": class_id,
                "offsets_nm": offsets,
                "matches_baseline": signature == baseline_signature,
                "mask_sha256": signature[0],
                "raw_metrics_sha256": signature[1],
            })
        point_response_summary.append({
            "point_id": point.point_id,
            "polygon_index": point.polygon_index,
            "source_edge_index": point.source_edge_index,
            "segment_index": point.segment_index,
            "normal_xy": list(point.normal_xy),
            "start_corner_type": point.start_corner_type,
            "end_corner_type": point.end_corner_type,
            "responsive": responsive,
            "changed_action_count": len(changed_actions),
            "changed_actions_nm": changed_actions,
            "distinct_response_count_including_baseline": len(signature_offsets),
            "response_classes": response_classes,
        })
    required_sensitivity_results = point_limit * len(selected_actions)
    sensitivity_pass = (
        len(selected_actions) >= 2
        and len(selected_points) == point_limit
        and len(sensitivity) == required_sensitivity_results
        and responsive_points == point_limit
    )
    unique_conflict_point_ids = sorted({
        item["point_id"] for item in all_conflicts
    })
    post_revision = _validate_openilt(solver.openilt_dir, openilt["commit"])
    return {
        "status": "diagnostic_only",
        "environment": config["environment"],
        "layout_parent": layout_parent,
        "layout_sha256": solver.layout_sha256,
        "openilt_revision": solver.revision,
        "openilt_source_sha256": solver.source_sha256,
        "fragmentation_sha256": solver.fragmentation_sha256,
        "epe_point_count": len(solver.epe_points),
        "baseline": {
            "mask_sha256": baseline_result.mask_sha256,
            "raw_metrics": baseline.metrics.as_dict(),
            "raw_weighted_loss": baseline.raw_weighted_loss,
            "active_control_point_count": len(baseline_active_ids),
            "control_conflict_count": len(baseline_conflicts),
            "control_conflict_point_ids": sorted({
                item["point_id"] for item in baseline_conflicts
            }),
        },
        "golden": {
            "evaluator_version": evaluator.evaluator_version,
            "constraint_coordinate": evaluator.epe_constraint_coordinate,
            "source_sha256": evaluator.evaluator_source_sha256,
            "sampling_state_sha256": evaluator.sampling_state_sha256,
            "coordinate_system_sha256": evaluator.coordinate_system_sha256,
            "evaluator_contract_sha256": evaluator.evaluator_contract_sha256,
            "configured_identity_verified_fields": verified_golden_fields,
        },
        "geometry_scan": geometry_scan,
        "sensitivity": sensitivity,
        "sensitivity_summary": {
            "selection_policy": PREFLIGHT_POINT_SELECTION_POLICY,
            "requested_point_count": point_limit,
            "baseline_active_point_count": len(baseline_active_ids),
            "baseline_conflict_point_count": len(baseline_conflict_ids),
            "eligible_active_point_count": len(eligible_active_points),
            "selected_point_count": len(selected_points),
            "selected_point_ids": [point.point_id for point in selected_points],
            "selected_actions_nm": selected_actions,
            "evaluated_result_count": len(sensitivity),
            "distinct_response_count": len(response_signatures),
            "distinct_response_count_including_baseline": len(
                response_signatures | {baseline_signature}
            ),
            "responsive_point_count": responsive_points,
            "point_response_summary": point_response_summary,
            "pass": sensitivity_pass,
        },
        "control_conflict_summary": {
            "policy": PREFLIGHT_CONFLICT_POLICY,
            "occurrence_count": len(all_conflicts),
            "unique_point_count": len(unique_conflict_point_ids),
            "point_ids": unique_conflict_point_ids,
            "resolved_by_frozen_policy": bool(all_conflicts),
            "training_compatible": True,
            "records": all_conflicts,
        },
        "solver_calls": 1 + sum(item["status"] == "evaluated" for item in sensitivity),
        "wall_time_seconds": time.monotonic() - started,
        "post_run_openilt_revision": post_revision,
        "post_run_openilt_tracked_diff_clean": True,
        "training_enabled": False,
    }


def _run_v2_episode_smoke_once(
    config: dict,
    layout_parent: str,
    training_protocol: str,
    repeat_index: int,
    input_example_dir: Optional[Path] = None,
) -> dict:
    """用 128 observation 执行一个完整确定性 episode 并核对 final replay。"""
    started = time.monotonic()
    recipe = config["recipe_v2"]
    observation_config = recipe["observation"]
    patch_size = int(observation_config["patch_size"])
    if patch_size != 128:
        raise ValueError("v2 episode smoke 当前只允许冻结的 128x128 observation")
    if observation_config.get("legacy_64_status") != (
        "excluded_insufficient_control_context"
    ):
        raise ValueError("v2 必须显式排除上下文覆盖不足的 64x64 输入")
    solver, evaluator, verified_golden_fields = (
        _build_v2_openilt_solver_and_evaluator(config, layout_parent)
    )
    episode = LocalEPEEpisode(
        solver=solver,
        golden_evaluator=evaluator,
        reward_weights=recipe["reward"]["weights"],
        training_protocol=training_protocol,
        patch_size=patch_size,
        action_offsets_nm=recipe["epe_normal_offsets_nm"],
        control_probe_distance_nm=recipe["epe_probe_distance_nm"],
        training_reward_scale=recipe["reward"]["scale"],
        shuffle_points=False,
    )
    episode.require_plain_ppo_compatible()
    point_order = tuple(sorted(episode.point_ids))
    observation, reset_info = episode.reset(
        seed=int(repeat_index), point_order=point_order
    )
    input_examples = None
    if input_example_dir is not None:
        report_config = observation_config["report_examples"]
        input_examples = save_v2_ppo_input_examples(
            episode,
            input_example_dir,
            example_count=int(report_config["example_count_per_layout"]),
            selection_policy=str(report_config["selection_policy"]),
        )
    observation_batch = episode.observation_cache.batch(point_order)
    observation_valid = (
        observation["image"].shape == (5, patch_size, patch_size)
        and observation_batch["image"].shape
        == (len(point_order), 5, patch_size, patch_size)
        and observation_batch["vector"].shape[0] == len(point_order)
        and np.all(np.isfinite(observation_batch["image"]))
        and np.all(np.isfinite(observation_batch["vector"]))
    )
    action_count = len(episode.action_offsets_nm)
    actions_by_point_id = {
        point_id: index % action_count
        for index, point_id in enumerate(point_order)
    }
    rewards = []
    terminated = False
    for point_id in point_order:
        observation, reward, terminated, info = episode.step(
            actions_by_point_id[point_id]
        )
        if not np.all(np.isfinite(observation["image"])) or not np.all(
            np.isfinite(observation["vector"])
        ):
            raise RuntimeError("v2 episode smoke 产生非有限 observation")
        rewards.append(float(reward))
    if not terminated or not info["final_recipe_complete"]:
        raise RuntimeError("v2 episode smoke 未完成全部 EPE point schedule")
    final = episode.final_golden_evaluation
    final_result = episode.final_result
    offsets, replay_result, replay_golden, replay_recipe_sha256 = (
        episode.replay_complete_action_map(actions_by_point_id)
    )
    replay_equal = (
        offsets == episode.final_recipe_offsets_nm
        and replay_recipe_sha256 == episode.final_recipe_sha256
        and replay_result.mask_sha256 == final_result.mask_sha256
        and replay_golden.metrics.as_dict() == final.metrics.as_dict()
        and np.isclose(
            replay_golden.raw_weighted_loss,
            final.raw_weighted_loss,
            rtol=0.0,
            atol=0.0,
        )
    )
    initial_loss = float(reset_info["initial_raw_weighted_loss"])
    expected_reward_sum = episode.training_reward_scale * (
        initial_loss - float(final.raw_weighted_loss)
    )
    reward_sum = float(sum(rewards))
    telescoping_error = abs(reward_sum - expected_reward_sum)
    expected_candidate_calls = (
        len(point_order) if training_protocol == EPE_DENSE_PROTOCOL else 1
    )
    call_counts = episode.solver_call_counts
    calls_valid = call_counts == {
        "baseline_solver": 1,
        "candidate_solver": expected_candidate_calls,
        "final_replay_solver": 1,
    }
    conflicts = _result_control_conflicts(final_result)
    result_pass = bool(
        observation_valid
        and replay_equal
        and calls_valid
        and np.isfinite(reward_sum)
        and telescoping_error <= 1e-6
    )
    return {
        "training_protocol": training_protocol,
        "repeat_index": int(repeat_index),
        "patch_size": patch_size,
        "observation_version": episode.observation_version,
        "observation_valid": bool(observation_valid),
        "point_count": len(point_order),
        "point_order_sha256": episode.point_order_sha256,
        "baseline_state_sha256": reset_info["baseline_state_sha256"],
        "action_map_sha256": _sha256_json(actions_by_point_id),
        "action_class_counts": {
            str(action_class): sum(
                value == action_class for value in actions_by_point_id.values()
            )
            for action_class in range(action_count)
        },
        "initial_raw_metrics": reset_info["initial_raw_metrics"],
        "initial_raw_weighted_loss": initial_loss,
        "final_raw_metrics": final.metrics.as_dict(),
        "final_raw_weighted_loss": float(final.raw_weighted_loss),
        "final_mask_sha256": final_result.mask_sha256,
        "final_recipe_sha256": episode.final_recipe_sha256,
        "scaled_reward_sum": reward_sum,
        "expected_scaled_reward_sum": expected_reward_sum,
        "reward_telescoping_error": telescoping_error,
        "solver_call_counts": call_counts,
        "final_control_conflict_occurrence_count": len(conflicts),
        "final_control_conflict_unique_point_count": len({
            item["point_id"] for item in conflicts
        }),
        "control_conflict_policy": solver.control_conflict_policy,
        "final_replay_equal": bool(replay_equal),
        "configured_golden_identity_verified_fields": verified_golden_fields,
        "ppo_input_examples": input_examples,
        "pass": result_pass,
        "wall_time_seconds": time.monotonic() - started,
    }


def run_v2_openilt_episode_smoke(
    config: dict,
    artifact_root: Path,
    layout_parent: str = "M1_test4",
) -> dict:
    """对 dense/terminal 各运行两个 128 完整 episode并保存 Actor 输入样例。"""
    started = time.monotonic()
    if config.get("status") not in {
        "protocol_preflight_and_episode_smoke_only",
        "ppo_smoke_ready_long_training_disabled",
    }:
        raise ValueError("v2 episode smoke 要求显式的 smoke-only 配置状态")
    if bool(config.get("training", {}).get("enabled", False)):
        raise ValueError("v2 episode smoke 要求 training.enabled=false")
    recipe = config["recipe_v2"]
    smoke_config = config.get("episode_smoke", {})
    repeats = int(smoke_config.get("repeats_per_protocol", 2))
    if repeats != 2:
        raise ValueError("episode_smoke.repeats_per_protocol 必须为 2")
    protocols = (
        str(recipe["reward"]["dense_protocol"]),
        str(recipe["reward"]["terminal_protocol"]),
    )
    if protocols != (EPE_DENSE_PROTOCOL, EPE_TERMINAL_PROTOCOL):
        raise ValueError("dense/terminal 协议名与 v2 contract 不一致")
    report_config = recipe["observation"].get("report_examples", {})
    if report_config.get("enabled") is not True:
        raise ValueError("v2 smoke 必须启用 observation.report_examples")
    if report_config.get("selection_policy") != PPO_INPUT_SELECTION_POLICY:
        raise ValueError("v2 smoke 的汇报样例抽样协议与实现不一致")
    example_count = int(report_config.get("example_count_per_layout", 0))
    if not 1 <= example_count <= 16:
        raise ValueError("example_count_per_layout 必须在 1..16")
    variants = []
    for protocol in protocols:
        for repeat_index in range(repeats):
            variants.append(_run_v2_episode_smoke_once(
                config,
                layout_parent=layout_parent,
                training_protocol=protocol,
                repeat_index=repeat_index,
                input_example_dir=(
                    Path(artifact_root) / "ppo-input-examples"
                    if not variants else None
                ),
            ))
    input_examples = variants[0].pop("ppo_input_examples")
    for variant in variants[1:]:
        if variant.pop("ppo_input_examples") is not None:
            raise RuntimeError("仅允许首个共享 baseline 写入一份 PPO 输入样例")
    final_signatures = {
        (
            item["final_recipe_sha256"],
            item["final_mask_sha256"],
            _sha256_json(item["final_raw_metrics"]),
        )
        for item in variants
    }
    baseline_signatures = {
        (
            item["baseline_state_sha256"],
            _sha256_json(item["initial_raw_metrics"]),
        )
        for item in variants
    }
    cross_protocol_final_equal = len(final_signatures) == 1
    repeat_baseline_equal = len(baseline_signatures) == 1
    overall_pass = bool(
        all(item["pass"] for item in variants)
        and cross_protocol_final_equal
        and repeat_baseline_equal
    )
    openilt = config["openilt"]
    openilt_dir = Path(config["data"]["openilt_dir"])
    post_revision = _validate_openilt(openilt_dir, openilt["commit"])
    return {
        "status": "diagnostic_only",
        "environment": config["environment"],
        "layout_parent": layout_parent,
        "patch_size": 128,
        "repeats_per_protocol": repeats,
        "action_pattern": "cycle-all-classes-by-sorted-point-id-v1",
        "ppo_input_examples": {
            **input_examples,
            "shared_by_all_smoke_variants": True,
        },
        "variants": variants,
        "cross_protocol_final_equal": cross_protocol_final_equal,
        "repeat_baseline_equal": repeat_baseline_equal,
        "pass": overall_pass,
        "solver_calls": sum(
            sum(item["solver_call_counts"].values()) for item in variants
        ),
        "wall_time_seconds": time.monotonic() - started,
        "post_run_openilt_revision": post_revision,
        "post_run_openilt_tracked_diff_clean": True,
        "training_enabled": False,
    }


def run_v2_openilt_input_examples(
    config: dict,
    artifact_root: Path,
    layout_parent: str = "M1_test4",
) -> dict:
    """只求解一次冻结基线并导出与 v2 Actor 完全相同的输入样例。"""
    started = time.monotonic()
    if bool(config.get("training", {}).get("enabled", False)):
        raise ValueError("PPO 输入样例导出要求 training.enabled=false")
    recipe = config["recipe_v2"]
    observation_config = recipe["observation"]
    if int(observation_config["patch_size"]) != 128:
        raise ValueError("PPO 输入样例导出当前只允许 128x128 observation")
    report_config = observation_config.get("report_examples", {})
    if report_config.get("enabled") is not True:
        raise ValueError("PPO 输入样例导出要求 report_examples.enabled=true")
    if report_config.get("selection_policy") != PPO_INPUT_SELECTION_POLICY:
        raise ValueError("PPO 输入样例抽样协议与实现不一致")
    example_count = int(report_config.get("example_count_per_layout", 0))
    if not 1 <= example_count <= 16:
        raise ValueError("example_count_per_layout 必须在 1..16")
    solver, evaluator, verified_golden_fields = (
        _build_v2_openilt_solver_and_evaluator(config, layout_parent)
    )
    episode = LocalEPEEpisode(
        solver=solver,
        golden_evaluator=evaluator,
        reward_weights=recipe["reward"]["weights"],
        training_protocol=EPE_TERMINAL_PROTOCOL,
        patch_size=128,
        action_offsets_nm=recipe["epe_normal_offsets_nm"],
        control_probe_distance_nm=recipe["epe_probe_distance_nm"],
        training_reward_scale=recipe["reward"]["scale"],
        shuffle_points=False,
    )
    episode.require_plain_ppo_compatible()
    point_order = tuple(sorted(episode.point_ids))
    _, reset_info = episode.reset(seed=0, point_order=point_order)
    input_examples = save_v2_ppo_input_examples(
        episode,
        Path(artifact_root) / "ppo-input-examples",
        example_count=example_count,
        selection_policy=str(report_config["selection_policy"]),
    )
    openilt = config["openilt"]
    openilt_dir = Path(config["data"]["openilt_dir"])
    post_revision = _validate_openilt(openilt_dir, openilt["commit"])
    return {
        "status": "diagnostic_only",
        "environment": config["environment"],
        "layout_parent": layout_parent,
        "patch_size": episode.patch_size,
        "point_count": len(point_order),
        "observation_version": episode.observation_version,
        "baseline_state_sha256": reset_info["baseline_state_sha256"],
        "ppo_input_examples": input_examples,
        "solver_calls": episode.solver_call_counts["baseline_solver"],
        "configured_golden_identity_verified_fields": verified_golden_fields,
        "wall_time_seconds": time.monotonic() - started,
        "post_run_openilt_revision": post_revision,
        "post_run_openilt_tracked_diff_clean": True,
        "training_enabled": False,
    }
