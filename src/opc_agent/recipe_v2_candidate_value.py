"""本模块构建并评估 v2 候选动作价值排序的离线基线。

输入为冻结的六图坐标搜索工件与预测时可获得的视觉特征工作簿。构建阶段按搜索轨迹重建
每个点动作前的 incumbent L2/EPE/PVB/J，再将四个候选的真实物理结果展开为一行一个候选；
工作簿中的最终 Recipe 标签只用于来源一致性检查，不进入模型。评估阶段按版图留一训练
ExtraTrees 回归器预测候选有效改善率，并与随机排序和仅由训练版图统计得到的固定动作顺序
比较。缺特征点固定退回四动作完整搜索。所有输出均为 diagnostic_only，不调用 Solver，
也不能把离线单步排序结果解释为最终 Recipe 质量。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .recipe_v2_action_audit import _consecutive_groups, _load_json, _load_jsonl
from .recipe_v2_vision_prompt import FEATURES, GEOMETRY_FEATURES
from .recipe_v2_vision_tree import binary, integer, read_sheets, records


ACTION_OFFSETS_NM = [-20.0, -10.0, 0.0, 10.0, 20.0]
TARGET_COLUMNS = [
    "candidate_l2", "candidate_epe", "candidate_pvb", "candidate_j",
    "delta_l2", "delta_epe", "delta_pvb", "delta_j",
    "feasible", "beneficial", "effective_gain", "effective_gain_fraction",
]
IDENTIFIER_COLUMNS = [
    "layout_parent", "group_index", "point_id", "epe_id", "has_features",
    "has_geometry_features",
    "candidate_call", "candidate_action", "candidate_offset_nm",
    "incumbent_action", "incumbent_offset_nm",
]
STATE_FEATURE_COLUMNS = [
    "candidate_offset_scaled", "candidate_direction", "candidate_magnitude_scaled",
    "current_l2_ratio", "current_epe_ratio", "current_pvb_ratio", "current_j_ratio",
    "baseline_l2_share", "baseline_epe_weighted_share", "baseline_pvb_share",
]
GEOMETRY_CONTINUOUS_COLUMNS = [
    "normal_x", "normal_y", "segment_length_nm", "distance_to_start_nm",
    "distance_to_end_nm", "nearest_endpoint_distance_nm", "farthest_endpoint_distance_nm",
    "start_corner_type", "end_corner_type", "corner_count", "convex_corner_count",
    "concave_corner_count", "nearest_point_distance_nm", "point_count_within_32nm",
    "point_count_within_64nm", "point_count_within_128nm", "point_count_within_256nm",
    "same_orientation_count_within_128nm", "opposite_normal_count_within_128nm",
    "front_corridor_count_within_128nm", "back_corridor_count_within_128nm",
    "nearest_same_orientation_distance_nm", "nearest_opposite_normal_distance_nm",
    "nearest_front_corridor_distance_nm", "nearest_back_corridor_distance_nm",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _load_feature_map(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    content = Path(path).read_bytes()
    sheets = read_sheets(content)
    samples = records(sheets["samples"])
    provenance = records(sheets["provenance"])
    summary = {row["field"]: row["value"] for row in records(sheets["summary"])}
    expected_header = {"epe_id", "result", *FEATURES}
    if set(sheets["samples"][0]) != expected_header:
        raise ValueError("samples 列名与冻结 28 特征协议不一致")
    if summary.get("teacher") != "coordinate_search_not_ppo":
        raise ValueError("视觉特征工作簿教师来源不符")
    sample_by_id: dict[int, dict[str, Any]] = {}
    for row in samples:
        epe_id = integer(row["epe_id"], "samples.epe_id")
        if epe_id in sample_by_id:
            raise ValueError("samples epe_id 重复")
        sample_by_id[epe_id] = row
    feature_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in provenance:
        epe_id = integer(row["epe_id"], "provenance.epe_id")
        if epe_id not in sample_by_id:
            raise ValueError("provenance 与 samples 未完整匹配")
        layout = row["layout_parent"]
        point_id = row["point_id"]
        if not isinstance(layout, str) or not isinstance(point_id, str) or not point_id:
            raise ValueError("provenance 版图或 point_id 非法")
        key = (layout, point_id)
        if key in feature_map:
            raise ValueError("layout_parent + point_id 重复")
        sample = sample_by_id[epe_id]
        feature_values = {name: binary(sample[name]) for name in FEATURES}
        # result 是历史最终 Recipe 标签，只核对 provenance 身份，绝不放入模型特征。
        result = integer(sample["result"], "samples.result")
        if float(row["normal_offset_nm"]) != result * 10:
            raise ValueError("samples.result 与 provenance.normal_offset_nm 不一致")
        feature_map[key] = {"epe_id": epe_id, **feature_values}
    if integer(summary["training_rows"], "training_rows") != len(feature_map):
        raise ValueError("工作簿汇总训练行数不一致")
    return feature_map, {
        "source_xlsx_sha256": hashlib.sha256(content).hexdigest(),
        "training_rows": len(feature_map),
        "total_points": integer(summary["total"], "total"),
        "complete": bool(summary.get("complete")),
        "teacher": summary["teacher"],
    }


def _euclidean(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _nearest_or_sentinel(values: list[float]) -> float:
    # 冻结版图画布小于 4096nm；显式哨兵同时配合对应计数列，避免把缺失类别伪装成零距离。
    return min(values) if values else 4096.0


def _load_geometry_map(path: Path | None) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, Any] | None]:
    if path is None:
        return {}, None
    path = Path(path)
    manifest = _load_json(path)
    if manifest.get("teacher") != "coordinate_search_not_ppo" or manifest.get("status") != "diagnostic_only":
        raise ValueError("几何 manifest 来源或状态不符")
    source_rows = manifest.get("rows")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("几何 manifest 缺少 rows")
    by_layout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        layout = row.get("layout_parent")
        point_id = row.get("point_id")
        if not isinstance(layout, str) or not isinstance(point_id, str) or not point_id:
            raise ValueError("几何 manifest 版图或 point_id 非法")
        by_layout[layout].append(row)
    output: dict[tuple[str, str], dict[str, float]] = {}
    for layout, layout_rows in by_layout.items():
        points = [tuple(float(value) for value in row["base_xy"]) for row in layout_rows]
        normals = [tuple(float(value) for value in row["normal_xy"]) for row in layout_rows]
        for index, row in enumerate(layout_rows):
            point = points[index]
            normal = normals[index]
            evidence = row["geometry_evidence"]
            start = tuple(float(value) for value in evidence["segment_start_xy"])
            end = tuple(float(value) for value in evidence["segment_end_xy"])
            distances: list[float] = []
            same_orientation: list[float] = []
            opposite_normal: list[float] = []
            front_corridor: list[float] = []
            back_corridor: list[float] = []
            for other_index, other in enumerate(points):
                if other_index == index:
                    continue
                dx = other[0] - point[0]
                dy = other[1] - point[1]
                distance = math.hypot(dx, dy)
                distances.append(distance)
                other_normal = normals[other_index]
                if (abs(normal[0]), abs(normal[1])) == (abs(other_normal[0]), abs(other_normal[1])):
                    same_orientation.append(distance)
                if other_normal == (-normal[0], -normal[1]):
                    opposite_normal.append(distance)
                normal_projection = dx * normal[0] + dy * normal[1]
                tangent_projection = dx * (-normal[1]) + dy * normal[0]
                if abs(tangent_projection) <= 16 and 0 < normal_projection <= 128:
                    front_corridor.append(distance)
                if abs(tangent_projection) <= 16 and -128 <= normal_projection < 0:
                    back_corridor.append(distance)
            geometry = row["geometry_features"]
            start_corner = int(evidence["start_corner_type"])
            end_corner = int(evidence["end_corner_type"])
            continuous = {
                "normal_x": normal[0],
                "normal_y": normal[1],
                "segment_length_nm": _euclidean(start, end),
                "distance_to_start_nm": _euclidean(point, start),
                "distance_to_end_nm": _euclidean(point, end),
                "nearest_endpoint_distance_nm": min(_euclidean(point, start), _euclidean(point, end)),
                "farthest_endpoint_distance_nm": max(_euclidean(point, start), _euclidean(point, end)),
                "start_corner_type": start_corner,
                "end_corner_type": end_corner,
                "corner_count": int(start_corner != 0) + int(end_corner != 0),
                "convex_corner_count": int(start_corner == 1) + int(end_corner == 1),
                "concave_corner_count": int(start_corner == -1) + int(end_corner == -1),
                "nearest_point_distance_nm": _nearest_or_sentinel(distances),
                "point_count_within_32nm": sum(value <= 32 for value in distances),
                "point_count_within_64nm": sum(value <= 64 for value in distances),
                "point_count_within_128nm": sum(value <= 128 for value in distances),
                "point_count_within_256nm": sum(value <= 256 for value in distances),
                "same_orientation_count_within_128nm": sum(value <= 128 for value in same_orientation),
                "opposite_normal_count_within_128nm": sum(value <= 128 for value in opposite_normal),
                "front_corridor_count_within_128nm": len(front_corridor),
                "back_corridor_count_within_128nm": len(back_corridor),
                "nearest_same_orientation_distance_nm": _nearest_or_sentinel(same_orientation),
                "nearest_opposite_normal_distance_nm": _nearest_or_sentinel(opposite_normal),
                "nearest_front_corridor_distance_nm": _nearest_or_sentinel(front_corridor),
                "nearest_back_corridor_distance_nm": _nearest_or_sentinel(back_corridor),
            }
            fixed = {name: int(bool(geometry[name])) for name in GEOMETRY_FEATURES}
            key = (layout, row["point_id"])
            if key in output:
                raise ValueError("几何 manifest 中 layout_parent + point_id 重复")
            output[key] = {**fixed, **continuous}
    return output, {
        "source_geometry_manifest": str(path),
        "source_geometry_manifest_sha256": _sha256(path),
        "geometry_points": len(output),
        "geometry_feature_columns": list(GEOMETRY_FEATURES) + GEOMETRY_CONTINUOUS_COLUMNS,
        "missing_distance_sentinel_nm": 4096.0,
        "absolute_xy_excluded": True,
        "final_recipe_labels_excluded": ["action_index", "normal_offset_nm", "result", "recipe_sha256"],
    }


def _ratio(value: float, baseline: float) -> float:
    return float(value) / max(abs(float(baseline)), 1.0)


def _candidate_rows_for_layout(
    layout: str,
    result: dict[str, Any],
    candidates: list[dict[str, Any]],
    feature_map: dict[tuple[str, str], dict[str, Any]],
    geometry_map: dict[tuple[str, str], dict[str, float]],
) -> list[dict[str, Any]]:
    groups = _consecutive_groups(candidates)
    if len(groups) != len(result["point_order"]):
        raise ValueError(f"{layout} 候选组数与 point_order 不一致")
    baseline_metrics = {key: float(result["baseline"]["metrics"][key]) for key in ("l2", "epe", "pvb")}
    baseline_j = float(result["baseline"]["j"])
    current_metrics = dict(baseline_metrics)
    current_j = baseline_j
    output: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        point_id = group[0]["point_id"]
        if len(group) != 4 or len({int(row["action"]) for row in group}) != 4:
            raise ValueError(f"{layout}/{point_id} 不是四个互异候选动作")
        missing = sorted(set(range(5)) - {int(row["action"]) for row in group})
        if len(missing) != 1:
            raise ValueError(f"{layout}/{point_id} 无法恢复 incumbent 动作")
        incumbent_action = missing[0]
        after_values = {float(row["incumbent_j_after_group"]) for row in group}
        if len(after_values) != 1:
            raise ValueError(f"{layout}/{point_id} 组内 incumbent 终点不一致")
        accepted_rows = [row for row in group if bool(row["accepted"])]
        if len(accepted_rows) > 1:
            raise ValueError(f"{layout}/{point_id} 存在多个 accepted 候选")
        feature_record = feature_map.get((layout, point_id))
        geometry_record = geometry_map.get((layout, point_id))
        for candidate in group:
            action = int(candidate["action"])
            offset = ACTION_OFFSETS_NM[action]
            metrics = {key: float(candidate["metrics"][key]) for key in ("l2", "epe", "pvb")}
            candidate_j = float(candidate["j"])
            feasible = bool(candidate["feasible"])
            beneficial = feasible and candidate_j < current_j
            effective_gain = current_j - candidate_j if beneficial else 0.0
            row: dict[str, Any] = {
                "layout_parent": layout,
                "group_index": group_index,
                "point_id": point_id,
                "epe_id": None if feature_record is None else feature_record["epe_id"],
                "has_features": feature_record is not None,
                "has_geometry_features": geometry_record is not None,
                "candidate_call": int(candidate["candidate_call"]),
                "candidate_action": action,
                "candidate_offset_nm": offset,
                "incumbent_action": incumbent_action,
                "incumbent_offset_nm": ACTION_OFFSETS_NM[incumbent_action],
                "baseline_l2": baseline_metrics["l2"],
                "baseline_epe": baseline_metrics["epe"],
                "baseline_pvb": baseline_metrics["pvb"],
                "baseline_j": baseline_j,
                "current_l2": current_metrics["l2"],
                "current_epe": current_metrics["epe"],
                "current_pvb": current_metrics["pvb"],
                "current_j": current_j,
                "candidate_l2": metrics["l2"],
                "candidate_epe": metrics["epe"],
                "candidate_pvb": metrics["pvb"],
                "candidate_j": candidate_j,
                "delta_l2": metrics["l2"] - current_metrics["l2"],
                "delta_epe": metrics["epe"] - current_metrics["epe"],
                "delta_pvb": metrics["pvb"] - current_metrics["pvb"],
                "delta_j": candidate_j - current_j,
                "feasible": feasible,
                "beneficial": beneficial,
                "effective_gain": effective_gain,
                "effective_gain_fraction": effective_gain / max(abs(current_j), 1.0),
                "candidate_offset_scaled": offset / 20.0,
                "candidate_direction": -1 if offset < 0 else 1,
                "candidate_magnitude_scaled": abs(offset) / 20.0,
                "current_l2_ratio": _ratio(current_metrics["l2"], baseline_metrics["l2"]),
                "current_epe_ratio": _ratio(current_metrics["epe"], baseline_metrics["epe"]),
                "current_pvb_ratio": _ratio(current_metrics["pvb"], baseline_metrics["pvb"]),
                "current_j_ratio": _ratio(current_j, baseline_j),
                "baseline_l2_share": baseline_metrics["l2"] / max(abs(baseline_j), 1.0),
                "baseline_epe_weighted_share": 100.0 * baseline_metrics["epe"] / max(abs(baseline_j), 1.0),
                "baseline_pvb_share": baseline_metrics["pvb"] / max(abs(baseline_j), 1.0),
            }
            for name in FEATURES:
                row[name] = None if feature_record is None else feature_record[name]
            for name in GEOMETRY_CONTINUOUS_COLUMNS:
                row[name] = None if geometry_record is None else geometry_record[name]
            # 八个确定性几何布尔字段来自 manifest 时覆盖视觉工作簿；两者都存在时必须一致。
            if geometry_record is not None:
                for name in GEOMETRY_FEATURES:
                    if feature_record is not None and row[name] != geometry_record[name]:
                        raise ValueError(f"{layout}/{point_id} 确定性几何特征与工作簿不一致：{name}")
                    row[name] = geometry_record[name]
            output.append(row)
        incumbent_after = after_values.pop()
        if accepted_rows:
            accepted = accepted_rows[0]
            current_metrics = {key: float(accepted["metrics"][key]) for key in ("l2", "epe", "pvb")}
            current_j = float(accepted["j"])
        if current_j != incumbent_after:
            raise ValueError(f"{layout}/{point_id} incumbent J 重建失败")
    if current_j != float(result["best"]["j"]):
        raise ValueError(f"{layout} 重建轨迹终点与 result.json 不一致")
    return output


def build_candidate_dataset(
    source_run: Path,
    feature_xlsx: Path,
    output_dir: Path,
    geometry_manifest: Path | None = None,
) -> dict[str, Any]:
    """构建包含全部候选和显式缺特征标记的平面 CSV 数据集。"""
    source_run = Path(source_run)
    feature_xlsx = Path(feature_xlsx)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{output_dir}")
    search_summary_path = source_run / "recipe-v2-search.json"
    search_summary = _load_json(search_summary_path)
    if search_summary.get("status") != "diagnostic_only" or search_summary.get("accepted") is not False:
        raise ValueError("候选价值数据只接受 diagnostic_only 且未 accepted 的搜索源")
    feature_map, workbook_audit = _load_feature_map(feature_xlsx)
    geometry_map, geometry_audit = _load_geometry_map(geometry_manifest)
    all_rows: list[dict[str, Any]] = []
    layouts: list[str] = []
    for arm in search_summary.get("arms", []):
        layout = str(arm["layout_parent"])
        if layout in layouts or arm.get("method") != "coordinate" or int(arm.get("seed")) != 0:
            raise ValueError("每张版图必须恰好一个 coordinate/seed-0 臂")
        layouts.append(layout)
        arm_root = source_run / layout / "seed-0" / "coordinate"
        result = _load_json(arm_root / "result.json")
        candidates = _load_jsonl(arm_root / "candidates.jsonl")
        all_rows.extend(_candidate_rows_for_layout(layout, result, candidates, feature_map, geometry_map))
    if layouts != [f"M1_test{i}" for i in range(1, 7)]:
        raise ValueError("候选价值数据固定使用 M1_test1–6")
    expected_points = sum(len(arm["point_order"]) for arm in search_summary["arms"])
    if len(all_rows) != expected_points * 4:
        raise ValueError("候选总行数不等于点数乘四")
    covered_points = len({(row["layout_parent"], row["point_id"]) for row in all_rows if row["has_features"]})
    missing_points = expected_points - covered_points
    if covered_points != workbook_audit["training_rows"]:
        raise ValueError("工作簿特征点没有与搜索点一一匹配")
    geometry_covered_points = len({
        (row["layout_parent"], row["point_id"]) for row in all_rows if row["has_geometry_features"]
    })
    if geometry_manifest is not None and geometry_covered_points != expected_points:
        raise ValueError("几何 manifest 没有覆盖全部搜索点")

    output_dir.mkdir(parents=True, exist_ok=False)
    columns = (
        IDENTIFIER_COLUMNS
        + ["baseline_l2", "baseline_epe", "baseline_pvb", "baseline_j",
           "current_l2", "current_epe", "current_pvb", "current_j"]
        + STATE_FEATURE_COLUMNS
        + GEOMETRY_CONTINUOUS_COLUMNS
        + list(FEATURES)
        + TARGET_COLUMNS
    )
    with (output_dir / "candidate-values.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)
    layout_counts = {
        layout: {
            "points": len({row["point_id"] for row in all_rows if row["layout_parent"] == layout}),
            "candidates": sum(row["layout_parent"] == layout for row in all_rows),
            "feature_points": len({row["point_id"] for row in all_rows if row["layout_parent"] == layout and row["has_features"]}),
            "beneficial_candidates": sum(row["layout_parent"] == layout and row["beneficial"] for row in all_rows),
        }
        for layout in layouts
    }
    feature_columns: dict[str, list[str]] = {
        "geometry_action": list(FEATURES) + STATE_FEATURE_COLUMNS[:3],
        "geometry_action_state": list(FEATURES) + STATE_FEATURE_COLUMNS,
    }
    feature_availability = {name: "has_features" for name in feature_columns}
    if geometry_manifest is not None:
        deterministic = list(GEOMETRY_FEATURES) + GEOMETRY_CONTINUOUS_COLUMNS
        feature_columns.update({
            "deterministic_geometry_action": deterministic + STATE_FEATURE_COLUMNS[:3],
            "deterministic_geometry_action_state": deterministic + STATE_FEATURE_COLUMNS,
            "vision_plus_geometry_action_state": list(FEATURES) + GEOMETRY_CONTINUOUS_COLUMNS + STATE_FEATURE_COLUMNS,
        })
        feature_availability.update({
            "deterministic_geometry_action": "has_geometry_features",
            "deterministic_geometry_action_state": "has_geometry_features",
            "vision_plus_geometry_action_state": "has_features",
        })
    manifest = {
        "status": "diagnostic_only",
        "accepted": False,
        "dataset_version": "recipe-v2-candidate-value-v1",
        "source_run": str(source_run),
        "source_search_sha256": _sha256(search_summary_path),
        "source_feature_xlsx": str(feature_xlsx),
        "source_feature_xlsx_sha256": _sha256(feature_xlsx),
        "solver_calls": 0,
        "layouts": layouts,
        "points": expected_points,
        "candidate_rows": len(all_rows),
        "points_with_features": covered_points,
        "points_without_features": missing_points,
        "candidate_rows_with_features": sum(bool(row["has_features"]) for row in all_rows),
        "candidate_rows_without_features": sum(not bool(row["has_features"]) for row in all_rows),
        "points_with_geometry_features": geometry_covered_points,
        "beneficial_candidate_rows": sum(bool(row["beneficial"]) for row in all_rows),
        "feature_columns": feature_columns,
        "feature_availability": feature_availability,
        "excluded_from_model": IDENTIFIER_COLUMNS + TARGET_COLUMNS + [
            "baseline_l2", "baseline_epe", "baseline_pvb", "baseline_j",
            "current_l2", "current_epe", "current_pvb", "current_j",
            "samples.result", "provenance.normal_offset_nm", "search_order",
        ],
        "target": "effective_gain_fraction",
        "missing_feature_policy": "full_four_action_search",
        "workbook_audit": workbook_audit,
        "geometry_audit": geometry_audit,
        "layout_counts": layout_counts,
        "interpretation_boundary": (
            "每行目标来自一条顺序相关的坐标搜索轨迹；数据集只支持记录状态上的单步排序。"
        ),
    }
    _write_json(output_dir / "dataset-manifest.json", manifest)
    return manifest


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"非法布尔 CSV 值：{value}")


def load_candidate_dataset(dataset_path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _load_json(Path(manifest_path))
    rows: list[dict[str, Any]] = []
    with Path(dataset_path).open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            row["group_index"] = int(raw["group_index"])
            row["epe_id"] = None if raw["epe_id"] == "" else int(raw["epe_id"])
            row["candidate_action"] = int(raw["candidate_action"])
            row["candidate_call"] = int(raw["candidate_call"])
            row["candidate_offset_nm"] = float(raw["candidate_offset_nm"])
            row["incumbent_action"] = int(raw["incumbent_action"])
            row["incumbent_offset_nm"] = float(raw["incumbent_offset_nm"])
            row["has_features"] = _parse_bool(raw["has_features"])
            row["has_geometry_features"] = _parse_bool(raw["has_geometry_features"])
            row["feasible"] = _parse_bool(raw["feasible"])
            row["beneficial"] = _parse_bool(raw["beneficial"])
            numeric_columns = [
                "baseline_l2", "baseline_epe", "baseline_pvb", "baseline_j",
                "current_l2", "current_epe", "current_pvb", "current_j",
            ] + STATE_FEATURE_COLUMNS + GEOMETRY_CONTINUOUS_COLUMNS + list(FEATURES) + TARGET_COLUMNS
            for name in numeric_columns:
                if name in ("feasible", "beneficial"):
                    continue
                row[name] = None if raw[name] == "" else float(raw[name])
            rows.append(row)
    if len(rows) != int(manifest["candidate_rows"]):
        raise ValueError("候选 CSV 行数与 manifest 不一致")
    return rows, manifest


def _group_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["layout_parent"], int(row["group_index"]))].append(row)
    groups = [grouped[key] for key in sorted(grouped)]
    if any(len(group) != 4 for group in groups):
        raise ValueError("候选数据中存在非四动作组")
    return groups


def _prediction_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["layout_parent"]), int(row["candidate_call"])


def _ranking_by_prediction(
    group: list[dict[str, Any]], predictions: dict[tuple[str, int], float]
) -> list[dict[str, Any]]:
    return sorted(
        group,
        key=lambda row: (
            -predictions[_prediction_key(row)],
            abs(float(row["candidate_offset_nm"])),
            float(row["candidate_offset_nm"]),
        ),
    )


def policy_metrics(
    groups: list[list[dict[str, Any]]],
    rankings: dict[tuple[str, int], list[dict[str, Any]]],
    top_k: int,
) -> dict[str, Any]:
    """按真实 Solver veto 语义计算固定 Top-k 的单步质量与调用预算。"""
    if top_k not in (1, 2, 4):
        raise ValueError("离线策略只支持 Top-1/Top-2/Full")
    oracle_gain = 0.0
    captured_gain = 0.0
    calls = 0
    improving_groups = 0
    hit_groups = 0
    selected_beneficial = 0
    selected_candidates = 0
    regrets: list[float] = []
    missing_fallback_groups = 0
    for group in groups:
        key = (group[0]["layout_parent"], int(group[0]["group_index"]))
        gains = [float(row["effective_gain"]) for row in group]
        oracle = max(gains)
        oracle_gain += oracle
        if oracle > 0:
            improving_groups += 1
        if key not in rankings:
            selected = group
            missing_fallback_groups += 1
        else:
            selected = rankings[key][:top_k]
        selected_gain = max(float(row["effective_gain"]) for row in selected)
        captured_gain += selected_gain
        calls += len(selected)
        selected_candidates += len(selected)
        selected_beneficial += sum(float(row["effective_gain"]) > 0 for row in selected)
        regret = oracle - selected_gain
        regrets.append(regret)
        if oracle > 0 and any(float(row["effective_gain"]) == oracle for row in selected):
            hit_groups += 1
    full_calls = len(groups) * 4
    regrets_array = np.asarray(regrets, dtype=float)
    return {
        "top_k": top_k,
        "group_count": len(groups),
        "improving_group_count": improving_groups,
        "missing_feature_full_search_groups": missing_fallback_groups,
        "candidate_calls": calls,
        "full_search_candidate_calls": full_calls,
        "candidate_call_reduction": 1.0 - calls / full_calls,
        "oracle_gain": oracle_gain,
        "captured_gain": captured_gain,
        "improvement_capture_ratio": captured_gain / oracle_gain if oracle_gain else 1.0,
        "best_action_group_hit_rate": hit_groups / improving_groups if improving_groups else 1.0,
        "selected_beneficial_candidate_precision": selected_beneficial / selected_candidates if selected_candidates else 0.0,
        "mean_group_regret": float(regrets_array.mean()) if len(regrets_array) else 0.0,
        "p95_group_regret": float(np.percentile(regrets_array, 95)) if len(regrets_array) else 0.0,
        "max_group_regret": float(regrets_array.max()) if len(regrets_array) else 0.0,
    }


def random_expected_metrics(
    groups: list[list[dict[str, Any]]],
    top_k: int,
    available_keys: set[tuple[str, int]],
) -> dict[str, Any]:
    """枚举每组等概率动作子集，计算随机 Top-k 的精确期望而非单次随机种子。"""
    oracle_gain = captured_gain = expected_hits = 0.0
    improving_groups = 0
    calls = 0
    missing_fallback_groups = 0
    for group in groups:
        gains = [float(row["effective_gain"]) for row in group]
        oracle = max(gains)
        oracle_gain += oracle
        key = (group[0]["layout_parent"], int(group[0]["group_index"]))
        if key not in available_keys:
            captured_gain += oracle
            calls += 4
            missing_fallback_groups += 1
            if oracle > 0:
                improving_groups += 1
                expected_hits += 1.0
            continue
        combinations = list(itertools.combinations(range(4), top_k))
        captured_gain += sum(max(gains[index] for index in subset) for subset in combinations) / len(combinations)
        calls += top_k
        if oracle > 0:
            improving_groups += 1
            expected_hits += sum(any(gains[index] == oracle for index in subset) for subset in combinations) / len(combinations)
    full_calls = len(groups) * 4
    return {
        "top_k": top_k,
        "group_count": len(groups),
        "improving_group_count": improving_groups,
        "missing_feature_full_search_groups": missing_fallback_groups,
        "candidate_calls": calls,
        "full_search_candidate_calls": full_calls,
        "candidate_call_reduction": 1.0 - calls / full_calls,
        "oracle_gain": oracle_gain,
        "expected_captured_gain": captured_gain,
        "expected_improvement_capture_ratio": captured_gain / oracle_gain if oracle_gain else 1.0,
        "expected_best_action_group_hit_rate": expected_hits / improving_groups if improving_groups else 1.0,
    }


def oracle_top2_full_upper_bound(
    groups: list[list[dict[str, Any]]],
    rankings: dict[tuple[str, int], list[dict[str, Any]]],
    target_capture_ratio: float = 0.95,
) -> dict[str, Any]:
    """事后按真实 regret 选择 Top-2→Full 回退组，仅作为不可部署的理论上界。"""
    if not 0 < target_capture_ratio <= 1:
        raise ValueError("target_capture_ratio 必须位于 (0,1]")
    base = policy_metrics(groups, rankings, 2)
    required_gain = target_capture_ratio * base["oracle_gain"]
    recovered_gain = 0.0
    regrets: list[float] = []
    for group in groups:
        key = (group[0]["layout_parent"], int(group[0]["group_index"]))
        if key not in rankings:
            continue
        oracle = max(float(row["effective_gain"]) for row in group)
        selected = rankings[key][:2]
        top2_gain = max(float(row["effective_gain"]) for row in selected)
        regrets.append(oracle - top2_gain)
    fallback_groups = 0
    for regret in sorted(regrets, reverse=True):
        if base["captured_gain"] + recovered_gain >= required_gain:
            break
        recovered_gain += regret
        fallback_groups += 1
    candidate_calls = base["candidate_calls"] + 2 * fallback_groups
    captured_gain = base["captured_gain"] + recovered_gain
    return {
        "status": "non_deployable_ex_post_upper_bound",
        "target_capture_ratio": target_capture_ratio,
        "fallback_group_count": fallback_groups,
        "candidate_calls": candidate_calls,
        "candidate_call_reduction": 1.0 - candidate_calls / base["full_search_candidate_calls"],
        "captured_gain": captured_gain,
        "improvement_capture_ratio": (
            captured_gain / base["oracle_gain"] if base["oracle_gain"] else 1.0
        ),
        "warning": "使用留出数据真实 regret 选择回退组，只能证明潜在空间，不能作为模型结果。",
    }


def _action_mean_rankings(
    train_rows: list[dict[str, Any]],
    test_groups: list[list[dict[str, Any]]],
    availability_field: str,
) -> tuple[list[int], dict[tuple[str, int], list[dict[str, Any]]]]:
    by_action: dict[int, list[float]] = defaultdict(list)
    for row in train_rows:
        if row[availability_field]:
            by_action[int(row["candidate_action"])].append(float(row["effective_gain_fraction"]))
    actions = sorted(
        (action for action in by_action),
        key=lambda action: (-float(np.mean(by_action[action])), abs(ACTION_OFFSETS_NM[action]), ACTION_OFFSETS_NM[action]),
    )
    order = {action: index for index, action in enumerate(actions)}
    rankings = {
        (group[0]["layout_parent"], int(group[0]["group_index"])): sorted(
            group, key=lambda row: order[int(row["candidate_action"])]
        )
        for group in test_groups if group[0][availability_field]
    }
    return actions, rankings


def evaluate_candidate_value(
    dataset_path: Path,
    manifest_path: Path,
    output_dir: Path,
    n_estimators: int = 300,
) -> dict[str, Any]:
    """运行固定 ExtraTrees 参数的六图留一单步候选排序诊断。"""
    from sklearn.ensemble import ExtraTreesRegressor
    import sklearn

    if n_estimators < 1:
        raise ValueError("n_estimators 必须为正")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{output_dir}")
    rows, manifest = load_candidate_dataset(dataset_path, manifest_path)
    layouts = list(manifest["layouts"])
    groups_all = _group_rows(rows)
    experiments: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for feature_set_name, feature_names in manifest["feature_columns"].items():
        availability_field = manifest.get("feature_availability", {}).get(
            feature_set_name, "has_features"
        )
        folds: list[dict[str, Any]] = []
        all_predictions: dict[tuple[str, int], float] = {}
        all_fixed_rankings: dict[tuple[str, int], list[dict[str, Any]]] = {}
        fixed_orders: dict[str, list[int]] = {}
        for held_out in layouts:
            train_rows = [
                row for row in rows
                if row["layout_parent"] != held_out and row[availability_field]
            ]
            test_rows = [row for row in rows if row["layout_parent"] == held_out]
            test_feature_rows = [row for row in test_rows if row[availability_field]]
            test_groups = _group_rows(test_rows)
            X_train = np.asarray([[float(row[name]) for name in feature_names] for row in train_rows], dtype=float)
            y_train = np.asarray([float(row["effective_gain_fraction"]) for row in train_rows], dtype=float)
            X_test = np.asarray([[float(row[name]) for name in feature_names] for row in test_feature_rows], dtype=float)
            model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=0,
                n_jobs=1,
            ).fit(X_train, y_train)
            predicted = model.predict(X_test)
            fold_predictions = {
                _prediction_key(row): max(0.0, float(value))
                for row, value in zip(test_feature_rows, predicted)
            }
            all_predictions.update(fold_predictions)
            model_rankings = {
                (group[0]["layout_parent"], int(group[0]["group_index"])): _ranking_by_prediction(group, fold_predictions)
                for group in test_groups if group[0][availability_field]
            }
            fixed_order, fixed_rankings = _action_mean_rankings(
                train_rows, test_groups, availability_field
            )
            available_keys = set(model_rankings)
            fixed_orders[held_out] = fixed_order
            all_fixed_rankings.update(fixed_rankings)
            fold = {
                "held_out_layout": held_out,
                "train_layouts": [layout for layout in layouts if layout != held_out],
                "train_candidate_rows": len(train_rows),
                "test_candidate_rows": len(test_rows),
                "test_feature_candidate_rows": len(test_feature_rows),
                "training_mean_action_order": fixed_order,
                "model": {str(k): policy_metrics(test_groups, model_rankings, k) for k in (1, 2)},
                "training_mean_action": {str(k): policy_metrics(test_groups, fixed_rankings, k) for k in (1, 2)},
                "random_expected": {
                    str(k): random_expected_metrics(test_groups, k, available_keys)
                    for k in (1, 2)
                },
            }
            folds.append(fold)
            ranks_by_call = {
                _prediction_key(row): rank
                for ranking in model_rankings.values()
                for rank, row in enumerate(ranking, 1)
            }
            for row in test_rows:
                prediction_rows.append({
                    "feature_set": feature_set_name,
                    "held_out_layout": held_out,
                    "group_index": row["group_index"],
                    "point_id": row["point_id"],
                    "candidate_call": row["candidate_call"],
                    "candidate_action": row["candidate_action"],
                    "candidate_offset_nm": row["candidate_offset_nm"],
                    "has_features": row["has_features"],
                    "has_model_features": row[availability_field],
                    "effective_gain": row["effective_gain"],
                    "effective_gain_fraction": row["effective_gain_fraction"],
                    "predicted_effective_gain_fraction": all_predictions.get(_prediction_key(row)),
                    "predicted_rank": ranks_by_call.get(_prediction_key(row)),
                })
        model_rankings_all = {
            (group[0]["layout_parent"], int(group[0]["group_index"])): _ranking_by_prediction(group, all_predictions)
            for group in groups_all if group[0][availability_field]
        }
        available_keys_all = set(model_rankings_all)
        experiments[feature_set_name] = {
            "features": feature_names,
            "feature_count": len(feature_names),
            "availability_field": availability_field,
            "model": "ExtraTreesRegressor",
            "parameters": {
                "n_estimators": n_estimators,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "random_state": 0,
                "n_jobs": 1,
            },
            "folds": folds,
            "aggregate": {
                "model": {str(k): policy_metrics(groups_all, model_rankings_all, k) for k in (1, 2)},
                "training_mean_action": {str(k): policy_metrics(groups_all, all_fixed_rankings, k) for k in (1, 2)},
                "random_expected": {
                    str(k): random_expected_metrics(groups_all, k, available_keys_all)
                    for k in (1, 2)
                },
                "oracle_top2_full_upper_bound": oracle_top2_full_upper_bound(
                    groups_all, model_rankings_all, target_capture_ratio=0.95
                ),
                "fold_training_mean_action_orders": fixed_orders,
            },
        }
    report = {
        "status": "diagnostic_only",
        "accepted": False,
        "evaluation_version": "recipe-v2-candidate-value-lolo-v1",
        "source_dataset": str(Path(dataset_path)),
        "source_dataset_sha256": _sha256(Path(dataset_path)),
        "source_manifest_sha256": _sha256(Path(manifest_path)),
        "solver_calls": 0,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "environment": {"python_executable": sys.executable},
        "missing_feature_policy": "full_four_action_search",
        "experiments": experiments,
        "interpretation_boundary": (
            "LOLO 指标只评价日志状态上的单步候选排序；策略改变前序动作后，"
            "后续状态会偏离日志，必须通过真实 Proposal–Veto 搜索验证。"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "metrics.json", report)
    fields = list(prediction_rows[0])
    with (output_dir / "out-of-fold-predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction_rows)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="构建一行一个候选的价值数据集")
    build_parser.add_argument("--source-run", type=Path, required=True)
    build_parser.add_argument("--features-xlsx", type=Path, required=True)
    build_parser.add_argument("--geometry-manifest", type=Path)
    build_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = sub.add_parser("evaluate", help="运行固定参数的六图留一候选排序诊断")
    evaluate_parser.add_argument("--dataset", type=Path, required=True)
    evaluate_parser.add_argument("--manifest", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_candidate_dataset(
            args.source_run, args.features_xlsx, args.output, args.geometry_manifest
        )
        keys = ("points", "candidate_rows", "points_with_features", "points_without_features", "beneficial_candidate_rows")
        print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False, indent=2))
    else:
        result = evaluate_candidate_value(
            args.dataset, args.manifest, args.output, n_estimators=args.n_estimators
        )
        aggregate = {
            name: experiment["aggregate"]
            for name, experiment in result["experiments"].items()
        }
        print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
