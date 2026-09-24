"""验证高精度二分门控、嵌套按版图留一、缺特征 stay 回退和完整 Recipe 双回放。

测试使用合成 OOXML 和 Fake episode，不调用 API、GPU 或 OpenILT；重点防止留出版图
标签泄漏到阈值选择，并保证输出 Recipe 恰好覆盖源点集。
"""
from types import SimpleNamespace
import json

import numpy as np

from opc_agent.recipe_v2_vision import write_excel
from opc_agent.recipe_v2_vision_prompt import FEATURES
from opc_agent import recipe_v2_vision_gate as gate


def _workbook_and_source(tmp_path):
    samples = [["epe_id"] + list(FEATURES) + ["result"]]
    provenance = [["epe_id", "layout_parent", "point_id", "normal_offset_nm", "recipe_sha256"]]
    labels = gate.LABELS * 6
    by_layout = {"L0": [], "L1": [], "L2": []}
    for eid, label in enumerate(labels):
        layout = "L" + str(eid // 10)
        samples.append([eid] + [bool((eid + j) % 3) for j in range(len(FEATURES))] + [label])
        provenance.append([eid, layout, "p" + str(eid), label * 10, "source"])
        by_layout[layout].append({"point_id": "p" + str(eid), "action_index": label + 2,
                                  "normal_offset_nm": label * 10})
    # 两个没有特征的点仍必须出现在完整 Recipe，且预测强制 stay。
    by_layout["L0"].append({"point_id": "missing-0", "action_index": 2, "normal_offset_nm": 0})
    by_layout["L1"].append({"point_id": "missing-1", "action_index": 4, "normal_offset_nm": 20})
    workbook = tmp_path / "training.xlsx"
    write_excel(workbook, {
        "samples": samples,
        "provenance": provenance,
        "summary": [["field", "value"], ["teacher", "coordinate_search_not_ppo"],
                    ["training_rows", 30], ["total", 32]],
    })
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "schema_version": "v2-coordinate-recipe-labels-v1",
        "status": "diagnostic_only",
        "teacher": "coordinate_search_not_ppo",
        "accepted": False,
        "action_offsets_nm": [-20, -10, 0, 10, 20],
        "fragment_parameters_nm": {"corner": 16, "uniform": 32},
        "recipes": [{"layout_parent": layout, "labels": rows} for layout, rows in by_layout.items()],
    }), encoding="utf-8")
    return workbook, source


def _args(workbook, source, output):
    return SimpleNamespace(
        input=str(workbook), source_recipes=str(source), output=str(output), minimum_move_precision=0.8,
        gate_n_estimators=7, gate_max_depth=4, gate_min_samples_leaf=1,
        gate_max_features="sqrt", gate_class_weight="none",
        move_n_estimators=7, move_max_depth=4, move_min_samples_leaf=1,
        move_max_features="sqrt", move_class_weight="balanced_subsample",
    )


def test_threshold_prefers_recall_subject_to_precision_and_has_safe_fallback():
    selected, curve = gate.choose_threshold(np.array([1, 1, 0, 0]), np.array([.9, .8, .7, .1]), .8)
    assert selected["threshold"] == .8
    assert selected["precision"] == 1 and selected["recall"] == 1
    assert curve[-1]["predicted_move"] == 0
    fallback, _ = gate.choose_threshold(np.array([1, 0]), np.array([.1, .9]), 1.0)
    assert fallback["threshold"] > 1 and fallback["predicted_move"] == 0
    assert fallback["selection_reason"].endswith("fallback_all_stay")
    fixed, _ = gate.resolve_threshold(np.array([1, 0]), np.array([.8, .6]), fixed_threshold=.5)
    assert fixed["threshold"] == .5 and fixed["predicted_move"] == 2
    assert fixed["selection_reason"] == "fixed_probability_threshold"


def test_build_is_nested_complete_and_missing_features_fallback_to_stay(tmp_path):
    workbook, source = _workbook_and_source(tmp_path)
    output = tmp_path / "output"
    report = gate.build(_args(workbook, source, output))
    assert report["status"] == "diagnostic_only" and not report["golden_replay_performed"]
    assert report["samples_with_features"] == 30
    assert report["complete_recipe_points"] == 32
    assert report["missing_feature_fallback_count"] == 2
    assert len(report["folds"]) == 3
    for fold in report["folds"]:
        assert fold["held_out_layout"] not in fold["train_layouts"]
        assert fold["threshold_selection"]["minimum_precision"] == .8
    payload = json.loads((output / "predicted-recipes.json").read_text(encoding="utf-8"))
    rows = [row for recipe in payload["recipes"] for row in recipe["labels"]]
    missing = [row for row in rows if row["source"] == "missing_feature_fallback_stay"]
    assert len(rows) == 32 and len(missing) == 2
    assert all(row["action_index"] == 2 and row["normal_offset_nm"] == 0 for row in missing)
    assert (output / "two_stage_models.joblib").exists()
    assert (output / "threshold-curve.csv").exists()


class _Metrics:
    def __init__(self, l2, epe, pvb):
        self.value = {"l2": l2, "epe": epe, "pvb": pvb}

    def as_dict(self):
        return dict(self.value)


class _FakeEpisode:
    point_ids = ("p0", "p1")
    action_offsets_nm = (-20, -10, 0, 10, 20)

    def reset(self, point_order):
        assert point_order == ["p0", "p1"]
        return None, {"initial_raw_metrics": {"l2": 10, "epe": 2, "pvb": 20},
                      "initial_raw_weighted_loss": 230}

    def replay_complete_action_map(self, actions):
        assert set(actions) == set(self.point_ids)
        offsets = {point: (actions[point] - 2) * 10 for point in self.point_ids}
        result = SimpleNamespace(mask_sha256="mask")
        golden = SimpleNamespace(metrics=_Metrics(9, 1, 19), raw_weighted_loss=128)
        return offsets, result, golden, "recipe"


def test_replay_requires_full_point_set_and_records_independent_equality(tmp_path):
    predicted = tmp_path / "predicted.json"
    predicted.write_text(json.dumps({
        "schema_version": "v2-vision-two-stage-oof-recipes-v1",
        "status": "diagnostic_only", "accepted": False,
        "recipes": [
            {"layout_parent": "L0", "labels": [{"point_id": "p0", "action_index": 2}, {"point_id": "p1", "action_index": 4}]},
            {"layout_parent": "L1", "labels": [{"point_id": "p0", "action_index": 0}, {"point_id": "p1", "action_index": 2}]},
        ],
    }), encoding="utf-8")
    config = {"data": {"train_parents": ["L0", "L1"], "openilt_dir": "unused"},
              "training": {"enabled": False}, "openilt": {"commit": "frozen"}}
    result = gate.replay_predicted_recipes(
        config, predicted, tmp_path / "replay",
        episode_builder=lambda config, layout, shuffle_points: (_FakeEpisode(), ("contract",)),
        openilt_validator=lambda path, commit: commit,
    )
    assert result["status"] == "diagnostic_only" and not result["accepted"]
    assert result["all_final_replays_equal"]
    assert result["all_layouts_meet_metric_guardrail"]
    assert result["strictly_improved_layout_count"] == 2
    assert result["solver_calls"] == 6
