"""本模块用完整九动作 Oracle 真值评估逐 clip PPO 模型是否真正学会位移动作。

输入为候选索引、对应 train-oracle 运行目录、已合并的点级标签和奖励权重配置；输出为逐 clip、
逐 seed 以及全局汇总的准确率、九分类 macro-F1、最优集合命中率和加权损失遗憾，并给出零位移
与均匀随机基线。模块只读取已生成的 NPZ、JSON、缓存和模型，不调用 OpenILT、不训练模型，
默认在 CPU 上执行推理；测试可以注入替身模型加载器而不依赖 CUDA 或 stable-baselines3。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from .metrics import DISPLACEMENT_CLASSES_NM
from .oracle_batch_labels import load_candidate_index
from .recipe_tree import PointTrainingDataset, PointTrainingRow, validate_training_dataset


ModelLoader = Callable[[Path], object]


def _sha256(path: Path) -> str:
    """计算输入文件的稳定内容哈希。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _macro_f1_all_nine(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """按全部九个动作类别计算 macro-F1，缺失类别按零分计入。"""
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    scores = []
    for label in range(len(DISPLACEMENT_CLASSES_NM)):
        true_positive = int(np.count_nonzero((true == label) & (pred == label)))
        false_positive = int(np.count_nonzero((true != label) & (pred == label)))
        false_negative = int(np.count_nonzero((true == label) & (pred != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(np.mean(scores))


def _prediction_metrics(
    rows: Sequence[PointTrainingRow],
    predictions: Sequence[int],
    losses: Sequence[Sequence[float]],
) -> Dict[str, object]:
    """计算一个模型或确定性基线相对 Oracle 真值的分类与奖励质量。"""
    if not rows or len(rows) != len(predictions) or len(rows) != len(losses):
        raise ValueError("PPO 评估的标签、预测和九动作损失数量必须相同且非空")
    predicted = [int(value) for value in predictions]
    if any(value < 0 or value >= len(DISPLACEMENT_CLASSES_NM) for value in predicted):
        raise ValueError("PPO 预测动作必须位于 0 到 8")
    truth = [int(row.displacement_class) for row in rows]
    optimal = [set(row.optimal_classes or [row.displacement_class]) for row in rows]
    regret = []
    relative_regret = []
    for row_losses, action in zip(losses, predicted):
        values = np.asarray(row_losses, dtype=np.float64)
        if values.shape != (len(DISPLACEMENT_CLASSES_NM),) or not np.all(np.isfinite(values)):
            raise ValueError("每个点必须具有九个有限的加权 Oracle 损失")
        best = float(np.min(values))
        difference = max(0.0, float(values[action]) - best)
        regret.append(difference)
        relative_regret.append(difference / max(abs(best), 1.0))
    action_counts = {
        str(action): predicted.count(action) for action in range(len(DISPLACEMENT_CLASSES_NM))
    }
    return {
        "rows": len(rows),
        "accuracy": float(np.mean([left == right for left, right in zip(predicted, truth)])),
        "optimal_set_accuracy": float(np.mean([
            action in accepted for action, accepted in zip(predicted, optimal)
        ])),
        "macro_f1_all_nine_classes": _macro_f1_all_nine(truth, predicted),
        "mean_absolute_displacement_error_nm": float(np.mean([
            abs(float(DISPLACEMENT_CLASSES_NM[action]) - float(DISPLACEMENT_CLASSES_NM[target]))
            for action, target in zip(predicted, truth)
        ])),
        "mean_weighted_loss_regret": float(np.mean(regret)),
        "mean_relative_weighted_loss_regret": float(np.mean(relative_regret)),
        "predicted_action_counts": action_counts,
    }


def _uniform_random_expectation(
    rows: Sequence[PointTrainingRow], losses: Sequence[Sequence[float]]
) -> Dict[str, float]:
    """解析计算均匀随机九动作策略的期望准确率和期望遗憾。"""
    if not rows or len(rows) != len(losses):
        raise ValueError("随机基线的标签和损失数量必须相同且非空")
    action_count = len(DISPLACEMENT_CLASSES_NM)
    expected_regret = []
    expected_relative_regret = []
    expected_abs_error = []
    for row, row_losses in zip(rows, losses):
        values = np.asarray(row_losses, dtype=np.float64)
        best = float(np.min(values))
        mean_regret = max(0.0, float(np.mean(values)) - best)
        expected_regret.append(mean_regret)
        expected_relative_regret.append(mean_regret / max(abs(best), 1.0))
        target_nm = float(DISPLACEMENT_CLASSES_NM[row.displacement_class])
        expected_abs_error.append(float(np.mean([
            abs(float(value) - target_nm) for value in DISPLACEMENT_CLASSES_NM
        ])))
    return {
        "canonical_accuracy": 1.0 / action_count,
        "optimal_set_accuracy": float(np.mean([
            len(set(row.optimal_classes or [row.displacement_class])) / action_count for row in rows
        ])),
        "mean_absolute_displacement_error_nm": float(np.mean(expected_abs_error)),
        "mean_weighted_loss_regret": float(np.mean(expected_regret)),
        "mean_relative_weighted_loss_regret": float(np.mean(expected_relative_regret)),
    }


def _majority_class_baseline(
    rows: Sequence[PointTrainingRow], losses: Sequence[Sequence[float]]
) -> Dict[str, object]:
    """计算始终预测当前评估集合规范标签多数类的确定性基线。"""
    if not rows:
        raise ValueError("多数类基线不能为空")
    counts = [sum(row.displacement_class == action for row in rows) for action in range(9)]
    action = max(range(9), key=lambda value: (counts[value], -value))
    return {"action": action, **_prediction_metrics(rows, [action] * len(rows), losses)}


def _default_model_loader(path: Path) -> object:
    """延迟导入 SB3，并在 CPU 上加载模型用于确定性推理。"""
    from stable_baselines3 import PPO

    return PPO.load(str(path), device="cpu")


def _model_seed(path: Path, expected_source_hash: str) -> int:
    """读取训练端模型元数据并校验候选数据身份。"""
    metadata_path = path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"PPO 模型缺少元数据：{metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("candidate_source_sha256") != expected_source_hash:
        raise ValueError(f"PPO 模型与候选数据哈希不一致：{path}")
    seed = int(metadata["seed"])
    name_match = re.search(r"seed-(\d+)", path.stem)
    if name_match is not None and int(name_match.group(1)) != seed:
        raise ValueError(f"PPO 模型文件名与元数据 seed 不一致：{path}")
    return seed


def _clip_inputs(
    entry,
    clip_stage: dict,
    rows_by_sample: Dict[str, PointTrainingRow],
    reward_weights: Dict[str, float],
) -> Tuple[np.ndarray, List[PointTrainingRow], List[List[float]]]:
    """按候选点顺序对齐 observation、点级标签和缓存中的九动作损失。"""
    with np.load(str(entry.dataset_path), allow_pickle=False) as payload:
        observations = np.asarray(payload["observations"], dtype=np.float32)
    metadata = json.loads(Path(entry.metadata_path).read_text(encoding="utf-8"))
    point_ids = list(metadata.get("point_ids", []))
    if observations.ndim != 2 or observations.shape[0] != len(point_ids):
        raise ValueError(f"{entry.clip_id} observation 与 point_ids 数量不一致")
    rows = []
    for point_id in point_ids:
        sample_id = f"{entry.clip_id}:{point_id}"
        row = rows_by_sample.get(sample_id)
        if row is None:
            raise ValueError(f"正式标签缺少候选点：{sample_id}")
        rows.append(row)
    cache_path = Path(clip_stage["cache_path"])
    if not cache_path.is_file():
        raise FileNotFoundError(f"{entry.clip_id} 缺少 Oracle 指标缓存：{cache_path}")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("source_sha256") != entry.dataset_sha256:
        raise ValueError(f"{entry.clip_id} Oracle 缓存与候选数据哈希不一致")
    metrics = cache.get("metrics", {})
    losses = []
    for point_index in range(len(point_ids)):
        point_losses = []
        for action_index in range(len(DISPLACEMENT_CLASSES_NM)):
            item = metrics.get(f"{point_index}:{action_index}")
            if item is None:
                raise ValueError(f"{entry.clip_id} 点 {point_index} 缺少动作 {action_index} 指标")
            point_losses.append(sum(
                float(reward_weights[name]) * float(item[name]) for name in ("l2", "epe", "pvb")
            ))
        losses.append(point_losses)
    return observations, rows, losses


def evaluate_ppo_run(
    index_path: Path,
    oracle_run_dir: Path,
    labels_path: Path,
    reward_weights: Dict[str, float],
    model_loader: Optional[ModelLoader] = None,
) -> Dict[str, object]:
    """校验同版本输入，并评估所有 clip、所有 seed 的 PPO 确定性动作。"""
    if set(reward_weights) != {"l2", "epe", "pvb"}:
        raise ValueError("reward_weights 必须恰好包含 l2、epe、pvb")
    index = load_candidate_index(Path(index_path))
    run_root = Path(oracle_run_dir)
    stage_path = run_root / "stage-result.json"
    if not stage_path.is_file():
        raise FileNotFoundError(f"PPO 运行缺少 stage-result.json：{stage_path}")
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    if stage.get("candidate_index_sha256") != index.index_sha256:
        raise ValueError("PPO 运行与候选索引版本不一致")
    seeds = [int(value) for value in stage.get("seeds", [])]
    if not seeds:
        raise ValueError("PPO 运行摘要没有 seeds")
    label_source = Path(labels_path)
    if not label_source.is_file():
        raise FileNotFoundError(f"正式标签不存在：{label_source}")
    dataset = PointTrainingDataset.parse_obj(json.loads(label_source.read_text(encoding="utf-8")))
    validate_training_dataset(dataset)
    rows_by_sample = {row.sample_id: row for row in dataset.rows}
    if len(rows_by_sample) != len(dataset.rows):
        raise ValueError("正式标签包含重复 sample_id")
    expected_rows = sum(entry.epe_points + entry.frag_points for entry in index.entries)
    if len(dataset.rows) != expected_rows:
        raise ValueError(f"正式标签行数 {len(dataset.rows)} 与候选点数 {expected_rows} 不一致")
    stage_clips = {item["clip_id"]: item for item in stage.get("clips", [])}
    loader = model_loader or _default_model_loader
    per_clip = []
    aggregate: Dict[int, Dict[str, list]] = {
        seed: {"rows": [], "predictions": [], "losses": []} for seed in seeds
    }
    all_rows: List[PointTrainingRow] = []
    all_losses: List[List[float]] = []
    for entry in index.entries:
        clip_stage = stage_clips.get(entry.clip_id)
        if clip_stage is None:
            raise ValueError(f"PPO 运行缺少 clip：{entry.clip_id}")
        observations, rows, losses = _clip_inputs(
            entry, clip_stage, rows_by_sample, reward_weights
        )
        all_rows.extend(rows)
        all_losses.extend(losses)
        models_by_seed = {}
        for raw_path in clip_stage.get("models", []):
            model_path = Path(raw_path)
            if not model_path.is_file():
                raise FileNotFoundError(f"PPO 模型不存在：{model_path}")
            seed = _model_seed(model_path, entry.dataset_sha256)
            if seed in models_by_seed:
                raise ValueError(f"{entry.clip_id} 存在重复 seed 模型：{seed}")
            models_by_seed[seed] = model_path
        if set(models_by_seed) != set(seeds):
            raise ValueError(
                f"{entry.clip_id} 模型 seeds={sorted(models_by_seed)} 与运行 seeds={sorted(seeds)} 不一致"
            )
        seed_results = []
        for seed in seeds:
            model = loader(models_by_seed[seed])
            raw_predictions, _ = model.predict(observations, deterministic=True)
            predictions = np.asarray(raw_predictions, dtype=np.int64).reshape(-1).tolist()
            metrics = _prediction_metrics(rows, predictions, losses)
            seed_results.append({"seed": seed, "model_path": str(models_by_seed[seed]), **metrics})
            aggregate[seed]["rows"].extend(rows)
            aggregate[seed]["predictions"].extend(predictions)
            aggregate[seed]["losses"].extend(losses)
        zero_predictions = [4] * len(rows)
        per_clip.append({
            "clip_id": entry.clip_id,
            "split": entry.split,
            "rows": len(rows),
            "seeds": seed_results,
            "zero_displacement_baseline": _prediction_metrics(rows, zero_predictions, losses),
            "majority_class_baseline": _majority_class_baseline(rows, losses),
            "uniform_random_expectation": _uniform_random_expectation(rows, losses),
        })
    aggregate_seeds = []
    for seed in seeds:
        values = aggregate[seed]
        aggregate_seeds.append({
            "seed": seed,
            **_prediction_metrics(values["rows"], values["predictions"], values["losses"]),
        })
    result = {
        "schema_version": "1.0",
        "evaluation_scope": "same-clip deterministic PPO inference; not cross-layout generalization",
        "candidate_index": str(index_path),
        "candidate_index_sha256": index.index_sha256,
        "oracle_run": str(oracle_run_dir),
        "labels": str(labels_path),
        "labels_sha256": _sha256(label_source),
        "reward_weights": {name: float(reward_weights[name]) for name in ("l2", "epe", "pvb")},
        "clips": len(index.entries),
        "rows": len(dataset.rows),
        "seeds": seeds,
        "aggregate_seeds": aggregate_seeds,
        "zero_displacement_baseline": _prediction_metrics(
            all_rows, [4] * len(all_rows), all_losses
        ),
        "majority_class_baseline": _majority_class_baseline(all_rows, all_losses),
        "uniform_random_expectation": _uniform_random_expectation(all_rows, all_losses),
        "per_clip": per_clip,
    }
    return result


def _write_versioned(path: Path, payload: Dict[str, object]) -> None:
    """相同内容允许重复执行，不同内容拒绝覆盖。"""
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"拒绝覆盖已有不同版本：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(encoded, encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    """从命令行评估一个多 clip、多 seed PPO Oracle 运行。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.ppo_evaluation")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--oracle-run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = evaluate_ppo_run(
        args.index,
        args.oracle_run,
        args.labels,
        dict(config["oracle"]["reward_weights"]),
    )
    _write_versioned(args.output, result)
    print(args.output)
    print(f"clips={result['clips']} rows={result['rows']} seeds={len(result['seeds'])}")
    for item in result["aggregate_seeds"]:
        print(
            f"seed={item['seed']} accuracy={item['accuracy']:.6f} "
            f"optimal_accuracy={item['optimal_set_accuracy']:.6f} "
            f"macro_f1={item['macro_f1_all_nine_classes']:.6f} "
            f"mean_regret={item['mean_weighted_loss_regret']:.6f}"
        )
    zero = result["zero_displacement_baseline"]
    majority = result["majority_class_baseline"]
    random = result["uniform_random_expectation"]
    print(
        f"zero_accuracy={zero['accuracy']:.6f} zero_mean_regret={zero['mean_weighted_loss_regret']:.6f}"
    )
    print(
        f"majority_action={majority['action']} majority_accuracy={majority['accuracy']:.6f} "
        f"majority_mean_regret={majority['mean_weighted_loss_regret']:.6f}"
    )
    print(
        f"uniform_expected_accuracy={random['canonical_accuracy']:.6f} "
        f"uniform_expected_mean_regret={random['mean_weighted_loss_regret']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
