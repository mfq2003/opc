"""本模块审核云端 SimpleOPC 多步 PPO 是否有资格成为决策树 Recipe 教师。

输入为完整 train-oracle 运行目录和论文配置；输出为按 clip/seed 比较初始掩模、SimpleOPC 启发式与
PPO 历史最优 Recipe 的质量报告。关键依赖仅为 JSON/哈希与 NumPy；模块不运行 OpenILT、不训练模型，
并且只用 validation 决定 accepted/rejected，test 仅在报告中展示。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from .metrics import DISPLACEMENT_CLASSES_NM, SIMPLEOPC_LOSS_VERSION, weighted_opc_loss


PPO_RECIPE_LABEL_VERSION = "ppo-simpleopc-multistep-v3"


def _weighted_loss(metrics: dict, weights: Dict[str, float]) -> float:
    """复用训练环境的论文原始加权和实现，禁止验收另建目标。"""
    return weighted_opc_loss(metrics, weights)


def _verify_trajectory_loss(
    trajectory: list,
    weights: Dict[str, float],
    loss_scale: float,
    metric_epsilon: float,
    source: Path,
) -> float:
    """验证轨迹原始/归一化损失与共享公式一致，并返回初始原始加权损失。"""
    if not trajectory or "metrics" not in trajectory[0]:
        raise ValueError(f"Recipe 缺少初始轨迹指标：{source}")
    initial = _weighted_loss(trajectory[0]["metrics"], weights)
    expected_scale = max(initial, metric_epsilon)
    if not np.isclose(float(loss_scale), expected_scale):
        raise ValueError(f"轨迹整体损失缩放值不一致：{source}")
    for item in trajectory:
        if item.get("loss_version") != SIMPLEOPC_LOSS_VERSION:
            raise ValueError(f"轨迹损失版本不一致：{source}")
        raw = _weighted_loss(item["metrics"], weights)
        if not np.isclose(float(item.get("raw_weighted_loss", np.nan)), raw):
            raise ValueError(f"轨迹原始加权损失不一致：{source}")
        if not np.isclose(float(item.get("loss", np.nan)), raw / expected_scale):
            raise ValueError(f"轨迹归一化损失不一致：{source}")
    return initial


def _read_json(path: Path) -> dict:
    """读取 JSON 对象并给出包含路径的明确错误。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"PPO 质量审核输入不存在：{source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"PPO 质量审核输入不是 JSON 对象：{source}")
    return payload


def evaluate_simpleopc_ppo_run(run_dir: Path, config: dict) -> dict:
    """只用 validation 门槛审核完整运行，并保留 train/test 的分项证据。"""
    root = Path(run_dir)
    stage = _read_json(root / "stage-result.json")
    if stage.get("environment") != "simpleopc-multistep-v3":
        raise ValueError("质量审核只接受 simpleopc-multistep-v3 运行")
    if stage.get("loss_version") != SIMPLEOPC_LOSS_VERSION:
        raise ValueError("PPO stage 的损失版本不一致")
    if stage.get("mode") != "full":
        raise ValueError("smoke 运行不能通过 PPO 教师质量门槛")
    if tuple(stage.get("displacement_classes_nm", [])) != DISPLACEMENT_CLASSES_NM:
        raise ValueError("PPO stage 的九分类位移代表值不一致")
    weights = {name: float(value) for name, value in config["oracle"]["reward_weights"].items()}
    if set(weights) != {"l2", "epe", "pvb"}:
        raise ValueError("oracle.reward_weights 必须恰好包含 l2、epe、pvb")
    metric_epsilon = float(config.get("simpleopc", {}).get("metric_epsilon", 1.0))
    if metric_epsilon <= 0:
        raise ValueError("simpleopc.metric_epsilon 必须大于零")
    gate = config.get("ppo_acceptance", {})
    max_vs_heuristic = float(gate.get("max_relative_loss_vs_heuristic", 0.0))
    min_vs_initial = float(gate.get("min_relative_improvement_vs_initial", 0.0))
    max_seed_cv = float(gate.get("max_seed_loss_cv", 0.20))
    max_action_fraction = float(gate.get("max_single_action_fraction", 0.95))
    if max_vs_heuristic < 0 or min_vs_initial < 0 or max_seed_cv < 0:
        raise ValueError("PPO acceptance 损失/稳定性门槛不能为负")
    if not 0 < max_action_fraction <= 1:
        raise ValueError("max_single_action_fraction 必须在 (0, 1] 内")
    clip_reports = []
    validation_classes: List[int] = []
    validation_checks = []
    for clip in stage.get("clips", []):
        heuristic = _read_json(Path(clip["heuristic_path"]))
        if heuristic.get("loss_version") != SIMPLEOPC_LOSS_VERSION:
            raise ValueError(f"启发式损失版本不一致：{clip['heuristic_path']}")
        heuristic_initial_loss = _verify_trajectory_loss(
            heuristic.get("trajectory", []),
            weights,
            float(heuristic.get("reset", {}).get("loss_scale", np.nan)),
            metric_epsilon,
            Path(clip["heuristic_path"]),
        )
        heuristic_loss = _weighted_loss(heuristic["best_metrics"], weights)
        seed_reports = []
        model_paths = list(clip.get("models", []))
        recipe_paths = list(clip.get("ppo_recipes", []))
        if len(model_paths) != len(recipe_paths):
            raise ValueError(f"clip {clip.get('clip_id')} 的模型数与 Recipe 数不一致")
        for model_path_raw, recipe_path_raw in zip(model_paths, recipe_paths):
            model_path = Path(model_path_raw)
            recipe_path = Path(recipe_path_raw)
            recipe = _read_json(recipe_path)
            if recipe.get("label_version") != PPO_RECIPE_LABEL_VERSION:
                raise ValueError(f"Recipe 不是 PPO SimpleOPC 标签：{recipe_path}")
            if recipe.get("loss_version") != SIMPLEOPC_LOSS_VERSION:
                raise ValueError(f"Recipe 损失版本不一致：{recipe_path}")
            if tuple(recipe.get("displacement_classes_nm", [])) != DISPLACEMENT_CLASSES_NM:
                raise ValueError(f"Recipe 的九分类位移代表值不一致：{recipe_path}")
            if recipe.get("layout_sha256") != clip.get("layout_sha256"):
                raise ValueError(f"Recipe 与 stage 版图哈希不一致：{recipe_path}")
            actual_model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if actual_model_hash != recipe.get("model_sha256"):
                raise ValueError(f"Recipe 与 PPO 模型哈希不一致：{recipe_path}")
            trajectory = recipe.get("trajectory", [])
            initial_loss = _verify_trajectory_loss(
                trajectory,
                weights,
                float(recipe.get("loss_scale", np.nan)),
                metric_epsilon,
                recipe_path,
            )
            if not np.isclose(initial_loss, heuristic_initial_loss):
                raise ValueError(f"PPO 与启发式初始加权损失不一致：{recipe_path}")
            if not np.isclose(float(recipe.get("initial_weighted_loss", np.nan)), initial_loss):
                raise ValueError(f"Recipe 初始加权损失不一致：{recipe_path}")
            ppo_loss = _weighted_loss(recipe["best_metrics"], weights)
            if len(recipe.get("labels", [])) != int(clip["segment_count"]):
                raise ValueError(f"Recipe 标签数与边段数不一致：{recipe_path}")
            action_classes = [int(label["displacement_class"]) for label in recipe["labels"]]
            if any(value < 0 or value > 8 for value in action_classes):
                raise ValueError(f"Recipe 动作类别越界：{recipe_path}")
            for label, action_class in zip(recipe["labels"], action_classes):
                exact = float(label["ppo_displacement_nm"])
                if exact != float(DISPLACEMENT_CLASSES_NM[action_class]):
                    raise ValueError(f"Recipe 精确位移与类别代表值不一致：{recipe_path}")
                if float(label["quantization_error_nm"]) != 0.0:
                    raise ValueError(f"Recipe 含有非零位移量化误差：{recipe_path}")
            seed_reports.append({
                "seed": int(recipe["seed"]),
                "model_path": str(model_path),
                "recipe_path": str(recipe_path),
                "initial_loss": initial_loss,
                "heuristic_loss": heuristic_loss,
                "ppo_loss": ppo_loss,
                "relative_improvement_vs_initial": (initial_loss - ppo_loss) / max(initial_loss, 1e-12),
                "relative_loss_vs_heuristic": (ppo_loss - heuristic_loss) / max(heuristic_loss, 1e-12),
                "action_classes": action_classes,
            })
        if not seed_reports:
            raise ValueError(f"clip {clip.get('clip_id')} 没有 PPO Recipe")
        expected_seeds = {int(value) for value in config["oracle"]["seeds"]}
        actual_seeds = {item["seed"] for item in seed_reports}
        if actual_seeds != expected_seeds:
            raise ValueError(
                f"clip {clip.get('clip_id')} seed 不完整："
                f"expected={sorted(expected_seeds)} actual={sorted(actual_seeds)}"
            )
        losses = np.asarray([item["ppo_loss"] for item in seed_reports], dtype=np.float64)
        best = min(seed_reports, key=lambda item: (item["ppo_loss"], item["seed"]))
        seed_cv = float(losses.std() / max(abs(float(losses.mean())), 1e-12))
        report = {
            "clip_id": clip["clip_id"],
            "split": clip["split"],
            "segment_count": int(clip["segment_count"]),
            "heuristic_loss": heuristic_loss,
            "best_seed": best["seed"],
            "best_recipe_path": best["recipe_path"],
            "best_ppo_loss": best["ppo_loss"],
            "best_relative_improvement_vs_initial": best["relative_improvement_vs_initial"],
            "best_relative_loss_vs_heuristic": best["relative_loss_vs_heuristic"],
            "seed_loss_cv": seed_cv,
            "seeds": seed_reports,
        }
        clip_reports.append(report)
        if clip["split"] == "validation":
            validation_classes.extend(best["action_classes"])
            validation_checks.append(
                best["relative_loss_vs_heuristic"] <= max_vs_heuristic
                and best["relative_improvement_vs_initial"] >= min_vs_initial
                and seed_cv <= max_seed_cv
            )
    if not validation_checks:
        raise ValueError("完整运行缺少 validation clip，不能选择 PPO 教师")
    expected_validation = set(config.get("data", {}).get("validation_parents", []))
    actual_validation = {
        item["clip_id"] for item in clip_reports if item["split"] == "validation"
    }
    if expected_validation and actual_validation != expected_validation:
        raise ValueError(
            f"validation clip 不完整：expected={sorted(expected_validation)} actual={sorted(actual_validation)}"
        )
    configured_splits = {
        split: set(config.get("data", {}).get(f"{split}_parents", []))
        for split in ("train", "validation", "test")
    }
    for split, expected in configured_splits.items():
        if not expected:
            continue
        actual = {item["clip_id"] for item in clip_reports if item["split"] == split}
        if actual != expected:
            raise ValueError(
                f"{split} clip 不完整：expected={sorted(expected)} actual={sorted(actual)}"
            )
    class_counts = np.bincount(np.asarray(validation_classes, dtype=np.int64), minlength=9)
    dominant_fraction = float(class_counts.max() / max(class_counts.sum(), 1))
    actions_ok = dominant_fraction <= max_action_fraction
    accepted = all(validation_checks) and actions_ok
    report = {
        "schema_version": "1.0",
        "environment_version": "simpleopc-multistep-v3",
        "loss_version": SIMPLEOPC_LOSS_VERSION,
        "label_version": PPO_RECIPE_LABEL_VERSION,
        "status": "accepted" if accepted else "rejected",
        "gate_split": "validation",
        "thresholds": {
            "max_relative_loss_vs_heuristic": max_vs_heuristic,
            "min_relative_improvement_vs_initial": min_vs_initial,
            "max_seed_loss_cv": max_seed_cv,
            "max_single_action_fraction": max_action_fraction,
        },
        "validation": {
            "clip_checks": validation_checks,
            "action_class_counts": class_counts.tolist(),
            "dominant_action_fraction": dominant_fraction,
            "actions_not_collapsed": actions_ok,
        },
        "clips": clip_reports,
        "tree_training_allowed": bool(accepted),
    }
    encoded_identity = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(encoded_identity.encode("utf-8")).hexdigest()
    return report


def main(argv: Optional[List[str]] = None) -> int:
    """从命令行审核既有云端运行，不重新启动 OpenILT 或 PPO。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.simpleopc_quality")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = evaluate_simpleopc_ppo_run(args.run_dir, config)
    output = args.output or (args.run_dir / "ppo-quality.json")
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"拒绝覆盖已有不同质量报告：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_text(encoded, encoding="utf-8")
    print(output)
    print(report["status"])
    return 0 if report["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
