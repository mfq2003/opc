"""用高精度 stay/move 门控和移动档位森林组装 v2 EPE 完整 Recipe。

训练使用嵌套的按版图留一协议：外层留出版图只用于评估，门控阈值仅从
外层训练版图的内层留一概率中选择。低置信度点和没有视觉特征的点回退为
stay；只有通过门控的点才由四分类森林选择 -20/-10/+10/+20 nm。

默认 build 只产生 diagnostic_only 的折外预测和完整 Recipe，不调用 OpenILT。
replay 子命令才使用冻结 v2 solver/Golden 对每张训练版图做两次独立回放；
分类指标不能代替该回放，任何输出均不是 accepted 或 PPO 成功。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from .recipe_v2_vision_tree import LABELS, load_data, read_sheets, records, scores, write_csv, write_json


MOVE_LABELS = [-2, -1, 1, 2]
STAY_ACTION_INDEX = 2


def binary_scores(y_true, predicted_move):
    """返回 move 为正类的精度/召回率和可直接审计的计数。"""
    truth = np.asarray(y_true, dtype=int) != 0
    pred = np.asarray(predicted_move, dtype=bool)
    if truth.shape != pred.shape:
        raise ValueError("二分指标的标签与预测形状不一致")
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    tn = int(np.sum(~truth & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_move": tp,
        "false_move": fp,
        "missed_move": fn,
        "true_stay": tn,
        "predicted_move": tp + fp,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def threshold_curve(y_true, move_probability):
    """列出有限预测集上所有会改变决策的阈值，不做插值。"""
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(move_probability, dtype=float)
    if y.shape != probability.shape or not np.all(np.isfinite(probability)):
        raise ValueError("门控概率必须与标签同形且全部有限")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("门控概率必须位于 [0,1]")
    rows = []
    for threshold in sorted(set(probability.tolist()), reverse=True):
        metrics = binary_scores(y, probability >= threshold)
        rows.append({"threshold": float(threshold), **metrics})
    # 显式保留全 stay 安全回退；它不伪造 precision=1。
    rows.append({"threshold": 1.0000000000000002, **binary_scores(y, np.zeros(len(y), dtype=bool))})
    return rows


def choose_threshold(y_true, move_probability, minimum_precision):
    """在满足精度下限的阈值中最大化召回；无可行点时全 stay。"""
    if not 0 < minimum_precision <= 1:
        raise ValueError("minimum_move_precision 必须位于 (0,1]")
    curve = threshold_curve(y_true, move_probability)
    feasible = [row for row in curve if row["predicted_move"] > 0 and row["precision"] >= minimum_precision]
    if not feasible:
        selected = curve[-1]
        reason = "no_threshold_meets_minimum_precision_fallback_all_stay"
    else:
        selected = max(feasible, key=lambda row: (row["recall"], row["precision"], row["threshold"]))
        reason = "maximum_recall_subject_to_minimum_precision"
    return {**selected, "minimum_precision": float(minimum_precision), "selection_reason": reason}, curve


def resolve_threshold(y_true, move_probability, *, fixed_threshold=None, minimum_precision=None):
    """选择固定概率阈值或训练折内的精度约束阈值，两者不得混用。"""
    if fixed_threshold is not None and minimum_precision is not None:
        raise ValueError("固定门控阈值与最小精度阈值不得同时设置")
    if fixed_threshold is not None:
        if not 0 <= fixed_threshold <= 1:
            raise ValueError("gate_threshold 必须位于 [0,1]")
        metrics = binary_scores(y_true, np.asarray(move_probability) >= fixed_threshold)
        return {"threshold": float(fixed_threshold), **metrics,
                "selection_reason": "fixed_probability_threshold"}, threshold_curve(y_true, move_probability)
    if minimum_precision is None:
        raise ValueError("必须指定固定阈值或最小 move precision")
    return choose_threshold(y_true, move_probability, minimum_precision)


def _forest_probability(model, X, positive_class=1):
    classes = list(model.classes_)
    if positive_class not in classes:
        return np.zeros(len(X), dtype=float)
    return model.predict_proba(X)[:, classes.index(positive_class)]


def _params(args, prefix):
    class_weight = getattr(args, prefix + "_class_weight")
    if class_weight == "none":
        class_weight = None
    return {
        "n_estimators": int(getattr(args, prefix + "_n_estimators")),
        "max_depth": getattr(args, prefix + "_max_depth"),
        "min_samples_leaf": int(getattr(args, prefix + "_min_samples_leaf")),
        "max_features": getattr(args, prefix + "_max_features"),
        "class_weight": class_weight,
        "random_state": 0,
        "criterion": "gini",
        "bootstrap": True,
        "n_jobs": 1,
    }


def _validate_params(params, name):
    if params["n_estimators"] < 1 or params["min_samples_leaf"] < 1:
        raise ValueError(name + " 的森林数量和叶节样本数必须为正")
    if params["max_depth"] is not None and params["max_depth"] < 1:
        raise ValueError(name + " 的 max_depth 必须为正或 None")
    if params["max_features"] not in ("sqrt", 0.5, 1.0):
        raise ValueError(name + " 的 max_features 非法")
    if params["class_weight"] not in (None, "balanced", "balanced_subsample"):
        raise ValueError(name + " 的 class_weight 非法")


def _inner_gate_probabilities(X, binary_y, groups, train_indices, params):
    """只在外层训练集内产生留一版图概率，供阈值选择。"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut

    train_indices = np.asarray(train_indices, dtype=int)
    inner_groups = groups[train_indices]
    if len(set(inner_groups.tolist())) < 2:
        raise ValueError("外层训练集至少需要两张版图才能选门控阈值")
    probability = np.full(len(train_indices), np.nan, dtype=float)
    seen = np.zeros(len(train_indices), dtype=int)
    for inner_train, inner_test in LeaveOneGroupOut().split(X[train_indices], binary_y[train_indices], inner_groups):
        model = RandomForestClassifier(**params).fit(X[train_indices][inner_train], binary_y[train_indices][inner_train])
        probability[inner_test] = _forest_probability(model, X[train_indices][inner_test])
        seen[inner_test] += 1
    if not np.all(seen == 1) or not np.all(np.isfinite(probability)):
        raise RuntimeError("内层留一门控概率不完整")
    return probability


def _source_recipes(path):
    content = Path(path).read_bytes()
    payload = json.loads(content)
    if payload.get("teacher") != "coordinate_search_not_ppo" or payload.get("status") != "diagnostic_only":
        raise ValueError("完整点源必须是 diagnostic_only 坐标搜索工件")
    if payload.get("action_offsets_nm") != [-20, -10, 0, 10, 20]:
        raise ValueError("完整点源的动作表不符")
    layouts = {}
    for recipe in payload.get("recipes", []):
        layout = recipe["layout_parent"]
        if layout in layouts:
            raise ValueError("完整点源中版图重复")
        labels = recipe["labels"]
        point_ids = [item["point_id"] for item in labels]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("完整点源中 point_id 重复")
        for item in labels:
            if item["action_index"] not in range(5) or item["normal_offset_nm"] != (item["action_index"] - 2) * 10:
                raise ValueError("完整点源的动作与位移不一致")
        layouts[layout] = labels
    if not layouts:
        raise ValueError("完整点源不含版图")
    return payload, layouts, hashlib.sha256(content).hexdigest()


def _provenance_by_id(path):
    content = Path(path).read_bytes()
    sheets = read_sheets(content)
    provenance = records(sheets["provenance"])
    result = {}
    for row in provenance:
        eid = int(row["epe_id"])
        if eid in result:
            raise ValueError("provenance epe_id 重复")
        point_id = row["point_id"]
        if not isinstance(point_id, str) or not point_id:
            raise ValueError("provenance point_id 为空")
        result[eid] = row
    return result


def build(args):
    """训练嵌套折外两阶段模型，并生成全部点的保守预测 Recipe。"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut
    import joblib
    import sklearn

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录须为空，避免覆盖模型")
    gate_params = _params(args, "gate")
    move_params = _params(args, "move")
    _validate_params(gate_params, "gate")
    _validate_params(move_params, "move")
    X, y, groups, ids, names, audit = load_data(args.input, drop_stay=False)
    provenance = _provenance_by_id(args.input)
    if set(ids) != set(provenance):
        raise ValueError("训练样本与 provenance 不完整匹配")
    source_payload, source_layouts, source_sha = _source_recipes(args.source_recipes)
    expected_layouts = sorted(source_layouts)
    if sorted(set(groups.tolist())) != expected_layouts:
        raise ValueError("训练样本版图与完整点源不一致")
    source_label = {
        (layout, item["point_id"]): item["action_index"] - 2
        for layout, items in source_layouts.items() for item in items
    }
    for eid, label, layout in zip(ids, y.tolist(), groups.tolist()):
        key = (layout, provenance[eid]["point_id"])
        if source_label.get(key) != int(label):
            raise ValueError("训练标签与完整坐标搜索源不一致")

    binary_y = (y != 0).astype(np.int8)
    gate_probability = np.full(len(y), np.nan, dtype=float)
    prediction = np.zeros(len(y), dtype=int)
    conditional_move_prediction = np.zeros(len(y), dtype=int)
    thresholds = np.full(len(y), np.nan, dtype=float)
    seen = np.zeros(len(y), dtype=int)
    folds = []
    for fold_index, (train_indices, test_indices) in enumerate(LeaveOneGroupOut().split(X, y, groups), 1):
        held_out = str(groups[test_indices][0])
        print(f"折 {fold_index}: 留出版图 {held_out}，训练 {len(train_indices)}，预测 {len(test_indices)}", flush=True)
        inner_probability = _inner_gate_probabilities(X, binary_y, groups, train_indices, gate_params)
        selection, _ = resolve_threshold(
            binary_y[train_indices], inner_probability,
            fixed_threshold=getattr(args, "gate_threshold", None),
            minimum_precision=getattr(args, "minimum_move_precision", None),
        )
        threshold = selection["threshold"]
        gate = RandomForestClassifier(**gate_params).fit(X[train_indices], binary_y[train_indices])
        probability = _forest_probability(gate, X[test_indices])
        selected = probability >= threshold
        fold_prediction = np.zeros(len(test_indices), dtype=int)
        moving_train = train_indices[y[train_indices] != 0]
        if len(moving_train) == 0:
            raise ValueError("外层训练集没有移动样本")
        mover = RandomForestClassifier(**move_params).fit(X[moving_train], y[moving_train])
        all_move_prediction = mover.predict(X[test_indices]).astype(int)
        if np.any(selected):
            fold_prediction[selected] = all_move_prediction[selected]
        gate_probability[test_indices] = probability
        prediction[test_indices] = fold_prediction
        conditional_move_prediction[test_indices] = all_move_prediction
        thresholds[test_indices] = threshold
        seen[test_indices] += 1
        folds.append({
            "held_out_layout": held_out,
            "train_layouts": sorted(set(groups[train_indices].tolist())),
            "train_count": int(len(train_indices)),
            "test_count": int(len(test_indices)),
            "threshold_selection": selection,
            "held_out_gate": binary_scores(y[test_indices], selected),
            "held_out_move_classifier_on_true_moves": scores(
                y[test_indices][y[test_indices] != 0],
                all_move_prediction[y[test_indices] != 0],
                MOVE_LABELS,
            ) if np.any(y[test_indices] != 0) else None,
            "held_out_end_to_end": scores(y[test_indices], fold_prediction, LABELS),
        })
    if not np.all(seen == 1) or not np.all(np.isfinite(gate_probability)):
        raise RuntimeError("外层留一预测不完整")

    # 最终模型用全部训练样本拟合，部署阈值仍仅用六折折外概率选择。
    full_indices = np.arange(len(y))
    final_inner_probability = _inner_gate_probabilities(X, binary_y, groups, full_indices, gate_params)
    deployment_selection, curve = resolve_threshold(
        binary_y, final_inner_probability,
        fixed_threshold=getattr(args, "gate_threshold", None),
        minimum_precision=getattr(args, "minimum_move_precision", None),
    )
    gate_model = RandomForestClassifier(**gate_params).fit(X, binary_y)
    move_model = RandomForestClassifier(**move_params).fit(X[y != 0], y[y != 0])

    prediction_by_key = {}
    diagnostic_by_key = {}
    for eid, layout, label, pred, move_pred, probability, threshold in zip(
            ids, groups.tolist(), y.tolist(), prediction.tolist(), conditional_move_prediction.tolist(),
            gate_probability.tolist(), thresholds.tolist()):
        point_id = provenance[eid]["point_id"]
        key = (layout, point_id)
        if key in prediction_by_key:
            raise ValueError("预测 point_id 重复")
        prediction_by_key[key] = int(pred)
        diagnostic_by_key[key] = {
            "epe_id": int(eid),
            "teacher_label": int(label),
            "prediction": int(pred),
            "move_class_prediction": int(move_pred),
            "move_probability": float(probability),
            "gate_threshold": float(threshold),
        }

    recipes = []
    all_truth = []
    all_prediction = []
    missing = []
    for layout in expected_layouts:
        labels = []
        for source in source_layouts[layout]:
            point_id = source["point_id"]
            key = (layout, point_id)
            predicted_label = prediction_by_key.get(key, 0)
            fallback = key not in prediction_by_key
            if fallback:
                missing.append({"layout_parent": layout, "point_id": point_id})
            labels.append({
                "point_id": point_id,
                "action_index": int(predicted_label + 2),
                "normal_offset_nm": int(predicted_label * 10),
                "source": "missing_feature_fallback_stay" if fallback else "nested_leave_one_layout_out_prediction",
            })
            all_truth.append(int(source["action_index"] - 2))
            all_prediction.append(int(predicted_label))
        recipes.append({
            "layout_parent": layout,
            "prediction_scope": "outer_fold_model_did_not_train_on_this_layout",
            "point_count": len(labels),
            "action_counts": dict(sorted(Counter(item["action_index"] for item in labels).items())),
            "labels": labels,
        })
    if len(all_truth) != sum(len(items) for items in source_layouts.values()):
        raise RuntimeError("完整 Recipe 点数不闭合")

    output.mkdir(parents=True, exist_ok=True)
    recipe_payload = {
        "schema_version": "v2-vision-two-stage-oof-recipes-v1",
        "status": "diagnostic_only",
        "accepted": False,
        "teacher": "coordinate_search_not_ppo",
        "prediction_protocol": ("nested-leave-one-layout-out-fixed-gate-v1"
                                if deployment_selection["selection_reason"] == "fixed_probability_threshold"
                                else "nested-leave-one-layout-out-high-precision-gate-v1"),
        "golden_replay_performed": False,
        "action_offsets_nm": source_payload["action_offsets_nm"],
        "fragment_parameters_nm": source_payload["fragment_parameters_nm"],
        "source_recipes_sha256": source_sha,
        "source_xlsx_sha256": audit["source_xlsx_sha256"],
        "gate_threshold_policy": deployment_selection["selection_reason"],
        "gate_threshold": float(deployment_selection["threshold"]),
        "minimum_move_precision": getattr(args, "minimum_move_precision", None),
        "missing_feature_fallback_count": len(missing),
        "recipes": recipes,
    }
    write_json(output / "predicted-recipes.json", recipe_payload)
    write_csv(
        output / "out_of_fold_predictions.csv",
        ["epe_id", "layout_parent", "point_id", "teacher_label", "move_probability", "gate_threshold", "gate_move", "move_class_prediction", "prediction"],
        [[item["epe_id"], layout, point_id, item["teacher_label"], item["move_probability"], item["gate_threshold"],
          int(item["prediction"] != 0), item["move_class_prediction"], item["prediction"]]
         for (layout, point_id), item in sorted(diagnostic_by_key.items(), key=lambda pair: pair[1]["epe_id"])],
    )
    write_csv(
        output / "threshold-curve.csv",
        ["threshold", "predicted_move", "true_move", "false_move", "missed_move", "true_stay", "precision", "recall", "f1"],
        [[row[key] for key in ("threshold", "predicted_move", "true_move", "false_move", "missed_move", "true_stay", "precision", "recall", "f1")] for row in curve],
    )
    write_json(output / "missing-feature-fallbacks.json", {"count": len(missing), "points": missing})
    joblib.dump({
        "gate_model": gate_model,
        "move_model": move_model,
        "feature_names": names,
        "deployment_gate_threshold": deployment_selection["threshold"],
        "gate_threshold_policy": deployment_selection["selection_reason"],
        "minimum_move_precision": getattr(args, "minimum_move_precision", None),
        "source_xlsx_sha256": audit["source_xlsx_sha256"],
        "status": "diagnostic_only",
    }, output / "two_stage_models.joblib")

    report = {
        "status": "diagnostic_only",
        "accepted": False,
        "teacher": "coordinate_search_not_ppo",
        "golden_replay_performed": False,
        "source": str(Path(args.input).resolve()),
        "source_recipes": str(Path(args.source_recipes).resolve()),
        "audit": audit,
        "samples_with_features": int(len(y)),
        "complete_recipe_points": int(len(all_truth)),
        "missing_feature_fallback_count": len(missing),
        "features": names,
        "gate_parameters": gate_params,
        "move_parameters": move_params,
        "gate_threshold_policy": deployment_selection["selection_reason"],
        "gate_threshold": float(deployment_selection["threshold"]),
        "minimum_move_precision": getattr(args, "minimum_move_precision", None),
        "deployment_threshold_selection": deployment_selection,
        "nested_leave_one_layout_out_gate": binary_scores(y, prediction != 0),
        "nested_leave_one_layout_out_move_classifier_on_true_moves": scores(
            y[y != 0], conditional_move_prediction[y != 0], MOVE_LABELS),
        "nested_leave_one_layout_out_end_to_end": scores(y, prediction, LABELS),
        "complete_recipe_gate_including_missing_fallback": binary_scores(np.array(all_truth), np.array(all_prediction) != 0),
        "complete_recipe_end_to_end_including_missing_fallback": scores(np.array(all_truth), np.array(all_prediction), LABELS),
        "all_stay_complete_recipe": scores(np.array(all_truth), np.zeros(len(all_truth), dtype=int), LABELS),
        "folds": folds,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "environment": {"python_executable": sys.executable, "numpy_module": np.__file__, "sklearn_module": sklearn.__file__},
        "evaluation_note": "外层每折留出一张版图，阈值仅由其余版图的内层留一概率选择；同一六图仍用于模型设计，不是独立 validation。",
        "replay_note": "predicted-recipes.json 已覆盖全部点，但 build 未调用 Golden；必须执行 replay 才能判断完整 Recipe 质量。",
    }
    write_json(output / "metrics.json", report)
    gate = report["complete_recipe_gate_including_missing_fallback"]
    end = report["complete_recipe_end_to_end_including_missing_fallback"]
    baseline = report["all_stay_complete_recipe"]
    lines = [
        "# v2 EPE 高精度两阶段折外诊断",
        "",
        f"完整 Recipe 共 {len(all_truth)} 点；{len(y)} 点有特征，{len(missing)} 点缺特征并保守回退 stay。",
        (f"门控使用固定概率阈值 {deployment_selection['threshold']:.4f}。"
         if deployment_selection["selection_reason"] == "fixed_probability_threshold" else
         f"门控最小 move precision 设计值为 {args.minimum_move_precision:.2%}，它是诊断参数，不是 v2 acceptance 标准。"),
        "",
        "| 完整 1306 点指标 | 两阶段折外预测 | 全 stay 对照 |",
        "| --- | ---: | ---: |",
        f"| Accuracy | {end['accuracy']:.4f} | {baseline['accuracy']:.4f} |",
        f"| Macro-F1 | {end['macro_f1_fixed_five_classes']:.4f} | {baseline['macro_f1_fixed_five_classes']:.4f} |",
        f"| Balanced accuracy | {end['balanced_accuracy_present_classes']:.4f} | {baseline['balanced_accuracy_present_classes']:.4f} |",
        "",
        f"move 门控：precision={gate['precision']:.4f}，recall={gate['recall']:.4f}，false move={gate['false_move']}，missed move={gate['missed_move']}。",
        "",
        "该运行仍是 diagnostic_only，且与阈值设计共用六张训练版图。分类指标不能代替完整 Recipe 的 Golden 回放。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if hashlib.sha256(Path(args.input).read_bytes()).hexdigest() != audit["source_xlsx_sha256"]:
        raise RuntimeError("训练期间源 Excel 发生变化，请在新目录重新运行")
    if hashlib.sha256(Path(args.source_recipes).read_bytes()).hexdigest() != source_sha:
        raise RuntimeError("训练期间完整 Recipe 源发生变化，请在新目录重新运行")
    print(json.dumps({"output": str(output), "gate": gate, "end_to_end": end,
                      "deployment_threshold": deployment_selection}, ensure_ascii=False, indent=2))
    return report


def replay_predicted_recipes(config, predicted_path, output, *, episode_builder=None, openilt_validator=None):
    """对六张训练版图的完整折外 Recipe 做冻结 Golden 双回放。"""
    if episode_builder is None:
        from .recipe_v2_small_train import _build_episode
        episode_builder = _build_episode
    if openilt_validator is None:
        from .recipe_v2_openilt import _validate_openilt
        openilt_validator = _validate_openilt

    predicted_path = Path(predicted_path)
    payload = json.loads(predicted_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "v2-vision-two-stage-oof-recipes-v1":
        raise ValueError("仅接受两阶段折外 Recipe v1")
    if payload.get("status") != "diagnostic_only" or payload.get("accepted") is not False:
        raise ValueError("折外 Recipe 状态边界不符")
    layouts = [recipe["layout_parent"] for recipe in payload["recipes"]]
    if layouts != config["data"]["train_parents"]:
        raise ValueError("折外回放必须严格覆盖六张训练版图并保持顺序")
    if config["training"]["enabled"] is not False:
        raise ValueError("两阶段回放不得开启通用训练")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("回放输出目录须为空")
    output.mkdir(parents=True, exist_ok=True)
    arms = []
    for recipe in payload["recipes"]:
        layout = recipe["layout_parent"]
        episode, fields = episode_builder(config, layout, shuffle_points=False)
        expected = set(episode.point_ids)
        rows = recipe["labels"]
        actions = {row["point_id"]: int(row["action_index"]) for row in rows}
        if len(actions) != len(rows) or set(actions) != expected:
            raise ValueError(layout + " 的预测 Recipe 未恰好覆盖 solver 点集")
        if any(action not in range(len(episode.action_offsets_nm)) for action in actions.values()):
            raise ValueError(layout + " 的预测动作越界")
        _, reset = episode.reset(point_order=sorted(episode.point_ids))
        baseline = {"metrics": reset["initial_raw_metrics"], "j": float(reset["initial_raw_weighted_loss"])}
        offsets1, result1, golden1, hash1 = episode.replay_complete_action_map(actions)
        offsets2, result2, golden2, hash2 = episode.replay_complete_action_map(actions)
        first = {"metrics": golden1.metrics.as_dict(), "j": float(golden1.raw_weighted_loss),
                 "mask_sha256": result1.mask_sha256, "recipe_sha256": hash1}
        second = {"metrics": golden2.metrics.as_dict(), "j": float(golden2.raw_weighted_loss),
                  "mask_sha256": result2.mask_sha256, "recipe_sha256": hash2}
        equal = offsets1 == offsets2 and first == second
        guardrail = all(first["metrics"][key] <= baseline["metrics"][key] for key in ("l2", "epe", "pvb"))
        arms.append({
            "layout_parent": layout,
            "golden_verified_fields": list(fields),
            "baseline": baseline,
            "predicted": first,
            "independent_replay": second,
            "final_replay_equal": bool(equal),
            "all_metrics_le_zero_baseline": bool(guardrail),
            "strict_j_improvement": bool(first["j"] < baseline["j"]),
            "solver_call_counts": {"baseline": 1, "predicted_replay": 2, "total": 3},
        })
        if not equal:
            raise RuntimeError(layout + " 的完整 Recipe 两次回放不一致")
    revision = openilt_validator(Path(config["data"]["openilt_dir"]), config["openilt"]["commit"])
    result = {
        "schema_version": "v2-vision-two-stage-golden-replay-v1",
        "status": "diagnostic_only",
        "accepted": False,
        "quality_accepted": False,
        "source_predicted_recipes": str(predicted_path.resolve()),
        "source_predicted_recipes_sha256": hashlib.sha256(predicted_path.read_bytes()).hexdigest(),
        "arms": arms,
        "all_final_replays_equal": all(arm["final_replay_equal"] for arm in arms),
        "all_layouts_meet_metric_guardrail": all(arm["all_metrics_le_zero_baseline"] for arm in arms),
        "strictly_improved_layout_count": sum(arm["strict_j_improvement"] for arm in arms),
        "mean_j_improvement_fraction": float(np.mean([
            (arm["baseline"]["j"] - arm["predicted"]["j"]) / arm["baseline"]["j"] for arm in arms
        ])),
        "solver_calls": sum(arm["solver_call_counts"]["total"] for arm in arms),
        "post_run_openilt_revision": revision,
        "post_run_openilt_tracked_diff_clean": True,
    }
    write_json(output / "golden-replay.json", result)
    return result


def _depth(value):
    return None if value.lower() == "none" else int(value)


def _max_features(value):
    return value if value == "sqrt" else float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="训练嵌套折外门控/档位模型并输出完整 Recipe")
    build_parser.add_argument("--input", default="runs/v2-vision-dataset-004/training.xlsx")
    build_parser.add_argument("--source-recipes", default="docs/v2_search_20260915_recipes.json")
    build_parser.add_argument("--output", required=True)
    threshold_group = build_parser.add_mutually_exclusive_group()
    threshold_group.add_argument("--gate-threshold", type=float,
                                 help="固定 stay/move 概率阈值；两种阈值参数均未指定时默认 0.5")
    threshold_group.add_argument("--minimum-move-precision", type=float,
                                 help="改用折内精度约束选阈值；与 --gate-threshold 互斥")
    for prefix, default_weight in (("gate", "none"), ("move", "balanced_subsample")):
        build_parser.add_argument(f"--{prefix}-n-estimators", type=int, default=500)
        build_parser.add_argument(f"--{prefix}-max-depth", type=_depth, default=None)
        build_parser.add_argument(f"--{prefix}-min-samples-leaf", type=int, default=1)
        build_parser.add_argument(f"--{prefix}-max-features", type=_max_features, default="sqrt")
        build_parser.add_argument(f"--{prefix}-class-weight", choices=["none", "balanced", "balanced_subsample"], default=default_weight)
    replay_parser = sub.add_parser("replay", help="用冻结 OpenILT/Golden 对完整预测 Recipe 做双回放")
    replay_parser.add_argument("--config", default="configs/recipe_ppo_v2.yaml")
    replay_parser.add_argument("--predicted-recipes", required=True)
    replay_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "build":
        if args.gate_threshold is None and args.minimum_move_precision is None:
            args.gate_threshold = 0.5
        build(args)
    else:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        result = replay_predicted_recipes(config, args.predicted_recipes, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
