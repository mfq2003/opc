"""本模块补齐 v2 候选价值排序的零 Solver 模型族对照实验。

输入沿用 ``recipe_v2_candidate_value`` 已冻结的候选 CSV 与 manifest，只在六张训练版图上
执行 Leave-One-Layout-Out。实验固定同一特征集、缺失特征 Full 回退、Top-k 计费与真实
Solver veto 离线口径，比较原始 ExtraTrees、训练折内组收益加权 ExtraTrees、RandomForest、
三种 HistGradientBoosting、两阶段 beneficial×gain 模型和组内 Pairwise Ranking。所有标签、
样本权重和模型均只由训练版图产生；输出仍是 ``diagnostic_only``，不能替代嵌套 LOLO
风险回退、真实 Proposal-Veto、Golden 回放或最终 Recipe 验收。
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .recipe_v2_candidate_value import (
    _action_mean_rankings,
    _group_rows,
    _prediction_key,
    _ranking_by_prediction,
    _sha256,
    _write_json,
    load_candidate_dataset,
    oracle_top2_full_upper_bound,
    policy_metrics,
    random_expected_metrics,
)


DEFAULT_FEATURE_SET = "vision_plus_geometry_action_state"
MODEL_NAMES = (
    "extra_trees_baseline",
    "extra_trees_group_gain_weighted",
    "random_forest",
    "hist_gradient_boosting_squared",
    "hist_gradient_boosting_absolute",
    "hist_gradient_boosting_quantile90",
    "two_stage_beneficial_gain",
    "pairwise_gain_difference",
)


def _matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    """把冻结的数值字段转为模型矩阵。"""
    return np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=float,
    )


def _group_gain_sample_weights(rows: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    """仅用训练折组内 oracle gain 构造截断权重，强调高代价排序组。"""
    groups = _group_rows(rows)
    potential_by_key: dict[tuple[str, int], float] = {}
    positive: list[float] = []
    for group in groups:
        key = (str(group[0]["layout_parent"]), int(group[0]["group_index"]))
        potential = max(float(row["effective_gain_fraction"]) for row in group)
        potential_by_key[key] = potential
        if potential > 0:
            positive.append(potential)
    scale = float(np.percentile(np.asarray(positive, dtype=float), 90)) if positive else 1.0
    scale = max(scale, np.finfo(float).eps)
    weights = np.asarray([
        1.0 + 9.0 * min(
            potential_by_key[(str(row["layout_parent"]), int(row["group_index"]))] / scale,
            1.0,
        )
        for row in rows
    ], dtype=float)
    return weights, {
        "formula": "1 + 9 * min(train_group_oracle_gain_fraction / train_p90_positive_group_gain, 1)",
        "positive_group_count": len(positive),
        "train_p90_positive_group_gain_fraction": scale,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
    }


def _pairwise_training_data(
    rows: list[dict[str, Any]],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """构造对称组内动作差分样本；真实收益平局不强行指定顺序。"""
    raw_pairs: list[tuple[np.ndarray, int, float]] = []
    non_tie_differences: list[float] = []
    skipped_ties = 0
    for group in _group_rows(rows):
        features = _matrix(group, feature_names)
        gains = np.asarray([float(row["effective_gain_fraction"]) for row in group], dtype=float)
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                difference = float(gains[left] - gains[right])
                if difference == 0.0:
                    skipped_ties += 1
                    continue
                magnitude = abs(difference)
                non_tie_differences.append(magnitude)
                label = int(difference > 0)
                delta = features[left] - features[right]
                raw_pairs.append((delta, label, magnitude))
                raw_pairs.append((-delta, 1 - label, magnitude))
    if not raw_pairs:
        raise ValueError("Pairwise Ranking 训练折没有非平局动作对")
    scale = float(np.percentile(np.asarray(non_tie_differences, dtype=float), 90))
    scale = max(scale, np.finfo(float).eps)
    X = np.asarray([item[0] for item in raw_pairs], dtype=float)
    y = np.asarray([item[1] for item in raw_pairs], dtype=int)
    weights = np.asarray([
        1.0 + 9.0 * min(item[2] / scale, 1.0)
        for item in raw_pairs
    ], dtype=float)
    return X, y, weights, {
        "pair_rows": len(raw_pairs),
        "unordered_non_tie_pairs": len(non_tie_differences),
        "skipped_tie_pairs": skipped_ties,
        "weight_formula": "1 + 9 * min(abs(train_gain_difference) / train_p90_difference, 1)",
        "train_p90_gain_difference": scale,
    }


def _pairwise_scores(
    model: Any,
    rows: list[dict[str, Any]],
    feature_names: list[str],
) -> np.ndarray:
    """把两两胜率汇总成每个候选的组内排序分数。"""
    output: dict[tuple[str, int], float] = {}
    for group in _group_rows(rows):
        features = _matrix(group, feature_names)
        scores = np.zeros(len(group), dtype=float)
        for left in range(len(group)):
            for right in range(len(group)):
                if left == right:
                    continue
                probability = float(model.predict_proba((features[left] - features[right]).reshape(1, -1))[0, 1])
                scores[left] += probability
        for row, score in zip(group, scores):
            output[_prediction_key(row)] = float(score)
    return np.asarray([output[_prediction_key(row)] for row in rows], dtype=float)


def _fit_predict(
    model_name: str,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    feature_names: list[str],
    n_estimators: int,
    max_iter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """按冻结模型定义拟合一个 LOLO 折并返回非负候选分数。"""
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )

    X_train = _matrix(train_rows, feature_names)
    X_test = _matrix(test_rows, feature_names)
    y_train = np.asarray([float(row["effective_gain_fraction"]) for row in train_rows], dtype=float)
    common_forest = {
        "n_estimators": n_estimators,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 0,
        "n_jobs": 1,
    }
    common_boosting = {
        "learning_rate": 0.05,
        "max_iter": max_iter,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1e-4,
        "early_stopping": False,
        "random_state": 0,
    }
    details: dict[str, Any] = {
        "train_candidate_rows": len(train_rows),
        "test_candidate_rows": len(test_rows),
        "positive_train_candidate_rows": int(np.count_nonzero(y_train > 0)),
    }
    if model_name == "extra_trees_baseline":
        model = ExtraTreesRegressor(**common_forest)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        details["parameters"] = common_forest
    elif model_name == "extra_trees_group_gain_weighted":
        sample_weight, weight_details = _group_gain_sample_weights(train_rows)
        model = ExtraTreesRegressor(**common_forest)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        predicted = model.predict(X_test)
        details["parameters"] = common_forest
        details["sample_weight"] = weight_details
    elif model_name == "random_forest":
        model = RandomForestRegressor(**common_forest)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        details["parameters"] = common_forest
    elif model_name.startswith("hist_gradient_boosting_"):
        loss_name = {
            "hist_gradient_boosting_squared": "squared_error",
            "hist_gradient_boosting_absolute": "absolute_error",
            "hist_gradient_boosting_quantile90": "quantile",
        }[model_name]
        parameters = {"loss": loss_name, **common_boosting}
        if loss_name == "quantile":
            parameters["quantile"] = 0.9
        model = HistGradientBoostingRegressor(**parameters)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        details["parameters"] = parameters
    elif model_name == "two_stage_beneficial_gain":
        beneficial = np.asarray([bool(row["beneficial"]) for row in train_rows], dtype=bool)
        if len(np.unique(beneficial)) != 2:
            raise ValueError("两阶段模型训练折必须同时包含 beneficial 与非 beneficial 候选")
        classifier_parameters = {
            **common_forest,
            "class_weight": "balanced_subsample",
        }
        classifier = RandomForestClassifier(**classifier_parameters)
        classifier.fit(X_train, beneficial.astype(int))
        positive = beneficial
        regressor = ExtraTreesRegressor(**common_forest)
        regressor.fit(X_train[positive], y_train[positive])
        beneficial_probability = classifier.predict_proba(X_test)[:, 1]
        conditional_gain = np.maximum(0.0, regressor.predict(X_test))
        predicted = beneficial_probability * conditional_gain
        details["classifier"] = {
            "estimator": "RandomForestClassifier",
            "parameters": classifier_parameters,
        }
        details["positive_regressor"] = {
            "estimator": "ExtraTreesRegressor",
            "parameters": common_forest,
        }
    elif model_name == "pairwise_gain_difference":
        X_pair, y_pair, pair_weight, pair_details = _pairwise_training_data(train_rows, feature_names)
        parameters = {
            "loss": "log_loss",
            **common_boosting,
        }
        model = HistGradientBoostingClassifier(**parameters)
        model.fit(X_pair, y_pair, sample_weight=pair_weight)
        predicted = _pairwise_scores(model, test_rows, feature_names)
        details["parameters"] = parameters
        details["pairwise"] = pair_details
    else:
        raise ValueError(f"未知候选价值模型：{model_name}")
    return np.maximum(0.0, np.asarray(predicted, dtype=float)), details


def evaluate_candidate_value_models(
    dataset_path: Path,
    manifest_path: Path,
    output_dir: Path,
    feature_set: str = DEFAULT_FEATURE_SET,
    n_estimators: int = 300,
    max_iter: int = 200,
    progress: bool = False,
) -> dict[str, Any]:
    """运行固定模型族的六图 LOLO 对照，结果写入全新目录。"""
    import sklearn

    if n_estimators < 1 or max_iter < 1:
        raise ValueError("n_estimators 与 max_iter 必须为正")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{output_dir}")
    rows, manifest = load_candidate_dataset(dataset_path, manifest_path)
    if feature_set not in manifest["feature_columns"]:
        raise ValueError(f"manifest 不包含特征集：{feature_set}")
    feature_names = list(manifest["feature_columns"][feature_set])
    availability_field = manifest.get("feature_availability", {}).get(feature_set, "has_features")
    layouts = list(manifest["layouts"])
    groups_all = _group_rows(rows)
    models: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []

    for model_name in MODEL_NAMES:
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
            predicted, fit_details = _fit_predict(
                model_name,
                train_rows,
                test_feature_rows,
                feature_names,
                n_estimators,
                max_iter,
            )
            fold_predictions = {
                _prediction_key(row): float(value)
                for row, value in zip(test_feature_rows, predicted)
            }
            all_predictions.update(fold_predictions)
            model_rankings = {
                (group[0]["layout_parent"], int(group[0]["group_index"])): _ranking_by_prediction(
                    group, fold_predictions
                )
                for group in test_groups if group[0][availability_field]
            }
            fixed_order, fixed_rankings = _action_mean_rankings(
                train_rows, test_groups, availability_field
            )
            fixed_orders[held_out] = fixed_order
            all_fixed_rankings.update(fixed_rankings)
            available_keys = set(model_rankings)
            fold = {
                "held_out_layout": held_out,
                "train_layouts": [layout for layout in layouts if layout != held_out],
                "fit": fit_details,
                "model": {str(k): policy_metrics(test_groups, model_rankings, k) for k in (1, 2)},
                "training_mean_action": {
                    str(k): policy_metrics(test_groups, fixed_rankings, k) for k in (1, 2)
                },
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
                    "model": model_name,
                    "feature_set": feature_set,
                    "held_out_layout": held_out,
                    "group_index": row["group_index"],
                    "point_id": row["point_id"],
                    "candidate_call": row["candidate_call"],
                    "candidate_action": row["candidate_action"],
                    "candidate_offset_nm": row["candidate_offset_nm"],
                    "has_model_features": row[availability_field],
                    "effective_gain": row["effective_gain"],
                    "effective_gain_fraction": row["effective_gain_fraction"],
                    "predicted_score": all_predictions.get(_prediction_key(row)),
                    "predicted_rank": ranks_by_call.get(_prediction_key(row)),
                })
            if progress:
                print(f"model={model_name} held_out={held_out} complete", flush=True)
        rankings_all = {
            (group[0]["layout_parent"], int(group[0]["group_index"])): _ranking_by_prediction(
                group, all_predictions
            )
            for group in groups_all if group[0][availability_field]
        }
        available_keys_all = set(rankings_all)
        aggregate_model = {str(k): policy_metrics(groups_all, rankings_all, k) for k in (1, 2)}
        top2 = aggregate_model["2"]
        models[model_name] = {
            "folds": folds,
            "aggregate": {
                "model": aggregate_model,
                "training_mean_action": {
                    str(k): policy_metrics(groups_all, all_fixed_rankings, k) for k in (1, 2)
                },
                "random_expected": {
                    str(k): random_expected_metrics(groups_all, k, available_keys_all)
                    for k in (1, 2)
                },
                "oracle_top2_full_upper_bound": oracle_top2_full_upper_bound(
                    groups_all, rankings_all, target_capture_ratio=0.95
                ),
                "fold_training_mean_action_orders": fixed_orders,
            },
            "screening_gate": {
                "improvement_capture_at_least_95_percent": (
                    top2["improvement_capture_ratio"] >= 0.95
                ),
                "candidate_call_reduction_at_least_40_percent": (
                    top2["candidate_call_reduction"] >= 0.40
                ),
                "passed": (
                    top2["improvement_capture_ratio"] >= 0.95
                    and top2["candidate_call_reduction"] >= 0.40
                ),
            },
        }

    best_name = max(
        MODEL_NAMES,
        key=lambda name: (
            models[name]["aggregate"]["model"]["2"]["improvement_capture_ratio"],
            models[name]["aggregate"]["model"]["1"]["improvement_capture_ratio"],
            name,
        ),
    )
    report = {
        "status": "diagnostic_only",
        "accepted": False,
        "evaluation_version": "recipe-v2-candidate-value-model-benchmark-v1",
        "benchmark_protocol": "fixed-model-family-six-layout-lolo-v1",
        "source_dataset": str(Path(dataset_path)),
        "source_dataset_sha256": _sha256(Path(dataset_path)),
        "source_manifest_sha256": _sha256(Path(manifest_path)),
        "solver_calls": 0,
        "feature_set": feature_set,
        "feature_count": len(feature_names),
        "availability_field": availability_field,
        "missing_feature_policy": "full_four_action_search",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "environment": {"python_executable": sys.executable},
        "models": models,
        "best_screening_model": best_name,
        "selection_boundary": (
            "best_screening_model 是同一六图 LOLO 上的事后模型族比较摘要；"
            "不能直接作为冻结部署模型或 WP2b 外层测试成绩。"
        ),
        "interpretation_boundary": (
            "模型对照只评价既有顺序搜索日志状态上的单步候选排序；不调用 Solver，"
            "不证明策略改变前序动作后的最终 Recipe 质量。"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "metrics.json", report)
    fields = list(prediction_rows[0])
    with (output_dir / "out-of-fold-predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction_rows)
    return report


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args(argv)
    result = evaluate_candidate_value_models(
        args.dataset,
        args.manifest,
        args.output,
        feature_set=args.feature_set,
        n_estimators=args.n_estimators,
        max_iter=args.max_iter,
        progress=True,
    )
    summary = {
        name: {
            "top1_capture": value["aggregate"]["model"]["1"]["improvement_capture_ratio"],
            "top2_capture": value["aggregate"]["model"]["2"]["improvement_capture_ratio"],
            "top2_call_reduction": value["aggregate"]["model"]["2"]["candidate_call_reduction"],
            "screening_gate_passed": value["screening_gate"]["passed"],
        }
        for name, value in result["models"].items()
    }
    print(json.dumps({
        "best_screening_model": result["best_screening_model"],
        "models": summary,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
