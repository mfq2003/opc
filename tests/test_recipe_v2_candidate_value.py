"""本文件验证 v2 候选价值数据的轨迹重建、特征隔离和 Top-k 预算计算。

测试使用临时 OOXML 工作簿与合成候选日志，不调用 OpenILT。它确保最终 Recipe 标签不会
进入模型字段、动作前 incumbent 指标会随已接受候选更新、缺特征点按四动作完整搜索计费，
并在可用 sklearn 环境中检查六图留一入口不混入留出版图。
"""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from opc_agent.recipe_v2_vision import write_excel
from opc_agent.recipe_v2_vision_prompt import FEATURES
from opc_agent.recipe_v2_candidate_value import (
    build_candidate_dataset,
    load_candidate_dataset,
    oracle_top2_full_upper_bound,
    policy_metrics,
)
from opc_agent.recipe_v2_candidate_value_models import (
    MODEL_NAMES,
    _group_gain_sample_weights,
    _pairwise_training_data,
    evaluate_candidate_value_models,
)


def _candidate(point_id, call, action, j, after, accepted=False, mask="m"):
    return {
        "point_id": point_id,
        "candidate_call": call,
        "action": action,
        "j": float(j),
        "metrics": {"l2": float(j), "epe": 0.0, "pvb": 0.0},
        "mask_sha256": mask,
        "feasible": True,
        "accepted": accepted,
        "incumbent_j_after_group": float(after),
    }


def _write_source(root):
    rows = [
        _candidate("p1", 1, 0, 100, 90),
        _candidate("p1", 2, 1, 90, 90, accepted=True),
        _candidate("p1", 3, 3, 95, 90),
        _candidate("p1", 4, 4, 110, 90),
        _candidate("p2", 5, 0, 90, 90),
        _candidate("p2", 6, 1, 90, 90),
        _candidate("p2", 7, 3, 90, 90),
        _candidate("p2", 8, 4, 90, 90),
    ]
    arm = root / "M1_test1" / "seed-0" / "coordinate"
    arm.mkdir(parents=True)
    result = {
        "baseline": {"metrics": {"l2": 100.0, "epe": 0.0, "pvb": 0.0}, "j": 100.0},
        "best": {"j": 90.0},
        "point_order": ["p1", "p2"],
    }
    (arm / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (arm / "candidates.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    arms = []
    for index in range(1, 7):
        layout = f"M1_test{index}"
        if index == 1:
            arms.append({"layout_parent": layout, "method": "coordinate", "seed": 0, "point_order": ["p1", "p2"]})
            continue
        other = root / layout / "seed-0" / "coordinate"
        other.mkdir(parents=True)
        (other / "result.json").write_text(json.dumps(result), encoding="utf-8")
        copied = [dict(row, point_id=f"p{index}a" if row["point_id"] == "p1" else f"p{index}b") for row in rows]
        (other / "candidates.jsonl").write_text(
            "\n".join(json.dumps(row) for row in copied) + "\n", encoding="utf-8"
        )
        arms.append({"layout_parent": layout, "method": "coordinate", "seed": 0, "point_order": [f"p{index}a", f"p{index}b"]})
    summary = {"status": "diagnostic_only", "accepted": False, "arms": arms}
    (root / "recipe-v2-search.json").write_text(json.dumps(summary), encoding="utf-8")


def _write_features(path, match_geometry=False):
    samples = [["epe_id", *FEATURES, "result"]]
    provenance = [["epe_id", "layout_parent", "point_id", "normal_offset_nm", "recipe_sha256"]]
    epe_id = 0
    for index in range(1, 7):
        point_ids = ["p1"] if index == 1 else [f"p{index}a"]
        for point_id in point_ids:
            values = {name: bool((epe_id + offset) % 2) for offset, name in enumerate(FEATURES)}
            if match_geometry:
                values.update({
                    "type_CV": False, "type_CH": False, "type_H": True, "type_V": False,
                    "on_horizontal_edge": True, "on_vertical_edge": False,
                    "on_start_corner_seg": False, "on_end_corner_seg": False,
                })
            samples.append([epe_id, *[values[name] for name in FEATURES], 0])
            provenance.append([epe_id, f"M1_test{index}", point_id, 0, "source"])
            epe_id += 1
    sheets = {
        "samples": samples,
        "provenance": provenance,
        "summary": [["field", "value"], ["complete", False], ["total", 12],
                    ["training_rows", 6], ["teacher", "coordinate_search_not_ppo"]],
    }
    write_excel(path, sheets)


def _write_geometry_manifest(path):
    rows = []
    epe_id = 0
    for index in range(1, 7):
        point_ids = ["p1", "p2"] if index == 1 else [f"p{index}a", f"p{index}b"]
        for point_index, point_id in enumerate(point_ids):
            x = float(index * 100 + point_index * 20)
            rows.append({
                "epe_id": epe_id,
                "layout_parent": f"M1_test{index}",
                "point_id": point_id,
                "base_xy": [x, 100.0],
                "normal_xy": [0, 1],
                "geometry_features": {
                    "type_CV": False, "type_CH": False, "type_H": True, "type_V": False,
                    "on_horizontal_edge": True, "on_vertical_edge": False,
                    "on_start_corner_seg": False, "on_end_corner_seg": False,
                },
                "geometry_evidence": {
                    "segment_start_xy": [x - 8, 100], "segment_end_xy": [x + 8, 100],
                    "start_corner_type": 0, "end_corner_type": 0,
                },
            })
            epe_id += 1
    path.write_text(json.dumps({
        "teacher": "coordinate_search_not_ppo",
        "status": "diagnostic_only",
        "rows": rows,
    }), encoding="utf-8")


def test_build_reconstructs_incumbent_and_preserves_missing_points(tmp_path):
    source = tmp_path / "source"
    features = tmp_path / "training.xlsx"
    output = tmp_path / "dataset"
    _write_source(source)
    _write_features(features, match_geometry=True)

    manifest = build_candidate_dataset(source, features, output)
    rows, loaded = load_candidate_dataset(output / "candidate-values.csv", output / "dataset-manifest.json")

    assert manifest == loaded
    assert manifest["points"] == 12 and manifest["candidate_rows"] == 48
    assert manifest["points_with_features"] == 6 and manifest["points_without_features"] == 6
    assert "samples.result" in manifest["excluded_from_model"]
    assert "result" not in manifest["feature_columns"]["geometry_action_state"]
    first_layout = [row for row in rows if row["layout_parent"] == "M1_test1"]
    assert {row["current_j"] for row in first_layout[:4]} == {100.0}
    assert {row["current_j"] for row in first_layout[4:]} == {90.0}
    assert all(row["has_features"] for row in first_layout[:4])
    assert all(not row["has_features"] for row in first_layout[4:])


def test_policy_metrics_uses_full_search_for_missing_features():
    covered = [
        {"layout_parent": "L", "group_index": 0, "has_features": True,
         "candidate_offset_nm": offset, "effective_gain": gain}
        for offset, gain in zip((-20, -10, 10, 20), (0, 5, 2, 0))
    ]
    missing = [
        {"layout_parent": "L", "group_index": 1, "has_features": False,
         "candidate_offset_nm": offset, "effective_gain": gain}
        for offset, gain in zip((-20, -10, 10, 20), (0, 0, 7, 0))
    ]
    rankings = {("L", 0): [covered[1], covered[2], covered[0], covered[3]]}

    metrics = policy_metrics([covered, missing], rankings, 1)

    assert metrics["candidate_calls"] == 5
    assert metrics["missing_feature_full_search_groups"] == 1
    assert metrics["oracle_gain"] == 12
    assert metrics["captured_gain"] == 12
    assert metrics["improvement_capture_ratio"] == 1

    upper = oracle_top2_full_upper_bound([covered, missing], rankings, 0.95)
    assert upper["status"] == "non_deployable_ex_post_upper_bound"
    assert upper["fallback_group_count"] == 0
    assert upper["candidate_call_reduction"] == 0.25


def test_geometry_manifest_covers_missing_visual_features_without_absolute_xy(tmp_path):
    source = tmp_path / "source"
    features = tmp_path / "training.xlsx"
    geometry = tmp_path / "manifest.json"
    output = tmp_path / "dataset"
    _write_source(source)
    _write_features(features, match_geometry=True)
    _write_geometry_manifest(geometry)

    manifest = build_candidate_dataset(source, features, output, geometry)
    rows, _ = load_candidate_dataset(output / "candidate-values.csv", output / "dataset-manifest.json")

    assert manifest["points_with_geometry_features"] == 12
    assert "deterministic_geometry_action_state" in manifest["feature_columns"]
    assert "base_x" not in manifest["feature_columns"]["deterministic_geometry_action_state"]
    missing_visual = [row for row in rows if not row["has_features"]]
    assert missing_visual and all(row["has_geometry_features"] for row in missing_visual)
    assert all(row["segment_length_nm"] == 16 for row in missing_visual)


def test_group_gain_weights_are_train_group_local_and_bounded():
    rows = []
    for group_index, gain in enumerate((0.0, 0.1, 1.0)):
        for candidate in range(4):
            rows.append({
                "layout_parent": "L",
                "group_index": group_index,
                "effective_gain_fraction": gain if candidate == 0 else 0.0,
            })

    weights, details = _group_gain_sample_weights(rows)

    assert np.all(weights[:4] == 1.0)
    assert np.all(weights[4:8] == weights[4])
    assert np.all(weights[8:12] == weights[8])
    assert 1.0 < weights[4] < weights[8] <= 10.0
    assert details["positive_group_count"] == 2


def test_pairwise_training_skips_ties_and_balances_directions():
    rows = []
    gains = (0.0, 0.2, 0.2, 0.8)
    for candidate, gain in enumerate(gains):
        rows.append({
            "layout_parent": "L",
            "group_index": 0,
            "f": float(candidate),
            "effective_gain_fraction": gain,
        })

    X, y, weights, details = _pairwise_training_data(rows, ["f"])

    assert details["skipped_tie_pairs"] == 1
    assert details["unordered_non_tie_pairs"] == 5
    assert X.shape == (10, 1)
    assert y.tolist().count(0) == y.tolist().count(1) == 5
    assert np.all(weights >= 1.0) and np.all(weights <= 10.0)


def test_model_benchmark_runs_all_fixed_models_without_solver(tmp_path):
    pytest.importorskip("sklearn")
    source = tmp_path / "source"
    features = tmp_path / "training.xlsx"
    geometry = tmp_path / "manifest.json"
    dataset = tmp_path / "dataset"
    output = tmp_path / "benchmark"
    _write_source(source)
    _write_features(features, match_geometry=True)
    _write_geometry_manifest(geometry)
    build_candidate_dataset(source, features, dataset, geometry)

    result = evaluate_candidate_value_models(
        dataset / "candidate-values.csv",
        dataset / "dataset-manifest.json",
        output,
        n_estimators=5,
        max_iter=5,
    )

    assert result["solver_calls"] == 0
    assert result["accepted"] is False
    assert set(result["models"]) == set(MODEL_NAMES)
    assert result["best_screening_model"] in MODEL_NAMES
    for model in result["models"].values():
        assert len(model["folds"]) == 6
        assert model["aggregate"]["model"]["2"]["candidate_calls"] == 36
        for fold in model["folds"]:
            assert fold["held_out_layout"] not in fold["train_layouts"]
    with (output / "out-of-fold-predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    assert len(predictions) == len(MODEL_NAMES) * 48
