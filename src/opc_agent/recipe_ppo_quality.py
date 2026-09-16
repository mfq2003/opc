"""本模块只读审核点级 Recipe PPO 的训练前基线、训练后 Recipe 与 validation 质量。

输入为 train-oracle 运行目录和同一份论文配置；输出为可追溯的 JSON 质量报告。审核会验证共享模型
哈希、固定九分类、每个 EPE/FRAG 点一次决策、原始加权损失、数据切分与动作塌缩。smoke 仅返回
diagnostic_only，不会被误判为正式 accepted；本模块不运行 OpenILT、不训练模型、也不调用 MLLM。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from .metrics import DISPLACEMENT_CLASSES_NM, RECIPE_OPC_LOSS_VERSION, weighted_opc_loss
from .recipe_contract import (
    PPO_RECIPE_LABEL_VERSION,
    RECIPE_ENV_VERSION,
    RECIPE_OBSERVATION_VERSION,
    RECIPE_POINT_VERSION,
)


def _read_json(path: Path) -> dict:
    """读取 JSON 对象并在错误中保留产物路径。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Recipe PPO 质量审核输入不存在：{source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Recipe PPO 质量审核输入不是 JSON 对象：{source}")
    return payload


def _loss(metrics: dict, weights: Dict[str, float]) -> float:
    """复用训练环境的原始论文加权和。"""
    return weighted_opc_loss(metrics, weights)


def _verify_raw_trajectory(trajectory: list, weights: Dict[str, float], source: Path) -> float:
    """验证轨迹没有重新引入 per-episode 归一化，并返回最佳原始损失。"""
    if not trajectory:
        raise ValueError(f"轨迹为空：{source}")
    losses = []
    for item in trajectory:
        if item.get("loss_version") != RECIPE_OPC_LOSS_VERSION:
            raise ValueError(f"轨迹损失版本错误：{source}")
        expected = _loss(item["metrics"], weights)
        if not np.isclose(float(item.get("raw_weighted_loss", np.nan)), expected):
            raise ValueError(f"轨迹原始加权损失错误：{source}")
        if not np.isclose(float(item.get("loss", np.nan)), expected):
            raise ValueError(f"轨迹不应使用 episode 归一化损失：{source}")
        losses.append(expected)
    return float(min(losses))


def _dominant_fraction(classes: List[int]) -> tuple[list, float]:
    """返回固定九类计数与最大单类占比。"""
    counts = np.bincount(np.asarray(classes, dtype=np.int64), minlength=9)
    return counts.tolist(), float(counts.max() / max(int(counts.sum()), 1))


def evaluate_recipe_ppo_run(run_dir: Path, config: dict) -> dict:
    """审核一个完整或 smoke 运行；只有 full validation 能决定 accepted。"""
    root = Path(run_dir)
    stage_path = root / "stage-result.json"
    stage = _read_json(stage_path)
    expected_stage = {
        "environment": RECIPE_ENV_VERSION,
        "observation_version": RECIPE_OBSERVATION_VERSION,
        "point_version": RECIPE_POINT_VERSION,
        "loss_version": RECIPE_OPC_LOSS_VERSION,
        "label_version": PPO_RECIPE_LABEL_VERSION,
        "reward_mode": "paper_raw",
        "policy_scope": "shared_train_clips",
        "action_space": "Discrete(9)",
        "action_semantics": "one_absolute_nine_class_decision_per_recipe_point",
    }
    for field, expected in expected_stage.items():
        if stage.get(field) != expected:
            raise ValueError(f"stage {field} 错误：expected={expected!r} actual={stage.get(field)!r}")
    if stage.get("patch_shape") != [5, 64, 64]:
        raise ValueError("stage 必须记录 5×64×64 observation")
    if stage.get("vector_shape") != [14]:
        raise ValueError("stage 必须记录包含点坐标的 14 维辅助向量")
    if tuple(stage.get("displacement_classes_nm", [])) != DISPLACEMENT_CLASSES_NM:
        raise ValueError("stage 九分类位移错误")
    if stage.get("openilt_mutation") != "none":
        raise ValueError("stage 未证明 OpenILT tracked 源码保持不变")
    mode = str(stage.get("mode"))
    if mode not in {"smoke", "full"}:
        raise ValueError("stage mode 必须是 smoke 或 full")

    weights = {name: float(value) for name, value in config["oracle"]["reward_weights"].items()}
    if set(weights) != {"l2", "epe", "pvb"}:
        raise ValueError("oracle.reward_weights 必须恰好包含 l2、epe、pvb")
    gate = config.get("ppo_acceptance", {})
    max_vs_default = float(gate.get("max_relative_loss_vs_default_recipe", 0.0))
    min_vs_initial = float(gate.get("min_relative_improvement_vs_initial", 0.0))
    max_seed_cv = float(gate.get("max_seed_loss_cv", 0.20))
    max_action_fraction = float(gate.get("max_single_action_fraction", 0.95))
    if min(max_vs_default, min_vs_initial, max_seed_cv) < 0:
        raise ValueError("PPO 损失与 seed 门槛不能为负")
    if not 0 < max_action_fraction <= 1:
        raise ValueError("max_single_action_fraction 必须在 (0, 1] 内")

    clip_reports = []
    validation_checks = []
    validation_actions = {"EPE": [], "FRAG": []}
    expected_seeds = {int(value) for value in stage.get("seeds", [])}
    if not expected_seeds:
        raise ValueError("stage 缺少训练 seed")
    shared_models = [Path(value) for value in stage.get("shared_models", [])]
    if len(shared_models) != len(expected_seeds):
        raise ValueError("stage 的共享模型数量与 seed 数不一致")
    metadata_seeds = set()
    for model_path in shared_models:
        metadata_path = model_path.with_suffix(".metadata.json")
        metadata = _read_json(metadata_path)
        if metadata.get("environment_version") != RECIPE_ENV_VERSION:
            raise ValueError(f"共享模型环境版本错误：{metadata_path}")
        if metadata.get("observation_version") != RECIPE_OBSERVATION_VERSION:
            raise ValueError(f"共享模型 observation 版本错误：{metadata_path}")
        if metadata.get("loss_version") != RECIPE_OPC_LOSS_VERSION:
            raise ValueError(f"共享模型损失版本错误：{metadata_path}")
        if metadata.get("patch_shape") != [5, 64, 64] or metadata.get("vector_shape") != [14]:
            raise ValueError(f"共享模型 observation shape 错误：{metadata_path}")
        actual_model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if metadata.get("model_sha256") != actual_model_hash:
            raise ValueError(f"共享模型 metadata 哈希错误：{metadata_path}")
        metadata_seeds.add(int(metadata["seed"]))
    if metadata_seeds != expected_seeds:
        raise ValueError("共享模型 metadata 的 seed 集合与 stage 不一致")
    for clip in stage.get("clips", []):
        baseline_path = Path(clip["default_recipe_path"])
        baseline = _read_json(baseline_path)
        if baseline.get("environment_version") != RECIPE_ENV_VERSION:
            raise ValueError(f"默认 Recipe 环境版本错误：{baseline_path}")
        if baseline.get("loss_version") != RECIPE_OPC_LOSS_VERSION:
            raise ValueError(f"默认 Recipe 损失版本错误：{baseline_path}")
        default_loss = _verify_raw_trajectory(baseline.get("trajectory", []), weights, baseline_path)
        model_paths = [Path(value) for value in clip.get("models", [])]
        recipe_paths = [Path(value) for value in clip.get("ppo_recipes", [])]
        if len(model_paths) != len(recipe_paths) or not recipe_paths:
            raise ValueError(f"clip {clip.get('clip_id')} 的模型与 Recipe 数量不一致或为空")
        seeds = []
        for model_path, recipe_path in zip(model_paths, recipe_paths):
            recipe = _read_json(recipe_path)
            expected_recipe = {
                "environment_version": RECIPE_ENV_VERSION,
                "observation_version": RECIPE_OBSERVATION_VERSION,
                "point_version": RECIPE_POINT_VERSION,
                "loss_version": RECIPE_OPC_LOSS_VERSION,
                "label_version": PPO_RECIPE_LABEL_VERSION,
                "reward_mode": "paper_raw",
                "action_semantics": "one_absolute_nine_class_decision_per_recipe_point",
            }
            for field, expected in expected_recipe.items():
                if recipe.get(field) != expected:
                    raise ValueError(f"Recipe {field} 错误：{recipe_path}")
            if recipe.get("patch_shape") != [5, 64, 64]:
                raise ValueError(f"Recipe observation 不是 5×64×64：{recipe_path}")
            if recipe.get("vector_shape") != [14]:
                raise ValueError(f"Recipe 辅助向量不是 14 维：{recipe_path}")
            if recipe.get("layout_sha256") != clip.get("layout_sha256"):
                raise ValueError(f"Recipe 与版图哈希不一致：{recipe_path}")
            actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if recipe.get("model_sha256") != actual_hash:
                raise ValueError(f"Recipe 与共享 PPO 模型哈希不一致：{recipe_path}")
            labels = recipe.get("labels", [])
            if len(labels) != int(clip["point_count"]):
                raise ValueError(f"Recipe 点数不一致：{recipe_path}")
            task_counts = Counter(str(label.get("task_type")) for label in labels)
            if task_counts != Counter({
                "EPE": int(clip["epe_point_count"]),
                "FRAG": int(clip["frag_point_count"]),
            }):
                raise ValueError(f"Recipe EPE/FRAG 点数不一致：{recipe_path}")
            point_ids = [str(label["point"]["point_id"]) for label in labels]
            if len(set(point_ids)) != len(point_ids):
                raise ValueError(f"Recipe 含重复 point_id：{recipe_path}")
            action_classes = []
            task_actions = {"EPE": [], "FRAG": []}
            for label in labels:
                action_class = int(label["displacement_class"])
                if not 0 <= action_class <= 8:
                    raise ValueError(f"Recipe 动作类别越界：{recipe_path}")
                displacement = float(label["ppo_displacement_nm"])
                if displacement != float(DISPLACEMENT_CLASSES_NM[action_class]):
                    raise ValueError(f"Recipe 位移并非九类代表值：{recipe_path}")
                if float(label.get("quantization_error_nm", np.nan)) != 0.0:
                    raise ValueError(f"Recipe 含量化误差：{recipe_path}")
                action_classes.append(action_class)
                task_actions[label["task_type"]].append(action_class)
            trajectory = recipe.get("trajectory", [])
            if len(trajectory) != int(clip["episode_horizon"]) + 1:
                raise ValueError(f"确定性回放没有做到每点一次：{recipe_path}")
            visited_ids = [item.get("point_id") for item in trajectory[1:]]
            if len(set(visited_ids)) != len(point_ids) or set(visited_ids) != set(point_ids):
                raise ValueError(f"确定性回放点集合不完整或重复：{recipe_path}")
            best_trace_loss = _verify_raw_trajectory(trajectory, weights, recipe_path)
            ppo_loss = _loss(recipe["best_metrics"], weights)
            if not np.isclose(best_trace_loss, ppo_loss):
                raise ValueError(f"Recipe best_metrics 不是轨迹历史最佳：{recipe_path}")
            if not np.isclose(float(recipe.get("initial_weighted_loss", np.nan)), default_loss):
                raise ValueError(f"PPO 与默认 Recipe 初始损失不一致：{recipe_path}")
            seed = int(recipe["seed"])
            seeds.append({
                "seed": seed,
                "model_path": str(model_path),
                "recipe_path": str(recipe_path),
                "default_loss": default_loss,
                "ppo_loss": ppo_loss,
                "relative_improvement_vs_initial": (default_loss - ppo_loss) / max(default_loss, 1e-12),
                "relative_loss_vs_default_recipe": (ppo_loss - default_loss) / max(default_loss, 1e-12),
                "action_classes": action_classes,
                "task_actions": task_actions,
            })
        actual_seeds = {item["seed"] for item in seeds}
        if actual_seeds != expected_seeds:
            raise ValueError(
                f"clip {clip.get('clip_id')} seed 不完整："
                f"expected={sorted(expected_seeds)} actual={sorted(actual_seeds)}"
            )
        losses = np.asarray([item["ppo_loss"] for item in seeds], dtype=np.float64)
        best = min(seeds, key=lambda item: (item["ppo_loss"], item["seed"]))
        seed_cv = float(losses.std() / max(abs(float(losses.mean())), 1e-12))
        clip_ok = (
            best["relative_loss_vs_default_recipe"] <= max_vs_default
            and best["relative_improvement_vs_initial"] >= min_vs_initial
            and seed_cv <= max_seed_cv
        )
        clip_reports.append({
            "clip_id": clip["clip_id"],
            "split": clip["split"],
            "point_count": int(clip["point_count"]),
            "epe_point_count": int(clip["epe_point_count"]),
            "frag_point_count": int(clip["frag_point_count"]),
            "default_loss": default_loss,
            "best_seed": best["seed"],
            "best_ppo_loss": best["ppo_loss"],
            "best_relative_improvement_vs_initial": best["relative_improvement_vs_initial"],
            "best_relative_loss_vs_default_recipe": best["relative_loss_vs_default_recipe"],
            "seed_loss_cv": seed_cv,
            "quality_check": bool(clip_ok),
            "seeds": seeds,
        })
        if clip["split"] == "validation":
            validation_checks.append(bool(clip_ok))
            for task_type in ("EPE", "FRAG"):
                validation_actions[task_type].extend(best["task_actions"][task_type])

    split_keys = {"train": "train_parents", "validation": "validation_parents", "test": "test_parents"}
    if mode == "full":
        for split, key in split_keys.items():
            expected = {str(value) for value in config["data"].get(key, [])}
            actual = {item["clip_id"] for item in clip_reports if item["split"] == split}
            if actual != expected:
                raise ValueError(
                    f"{split} clip 不完整：expected={sorted(expected)} actual={sorted(actual)}"
                )
        if not validation_checks:
            raise ValueError("full 运行缺少 validation，不能验收 PPO")

    validation_summary = {}
    actions_ok = True
    for task_type in ("EPE", "FRAG"):
        counts, dominant = _dominant_fraction(validation_actions[task_type])
        task_ok = bool(validation_actions[task_type]) and dominant <= max_action_fraction
        actions_ok = actions_ok and task_ok
        validation_summary[task_type.lower()] = {
            "action_class_counts": counts,
            "dominant_action_fraction": dominant,
            "actions_not_collapsed": task_ok,
        }
    accepted = mode == "full" and all(validation_checks) and actions_ok
    status = "accepted" if accepted else ("rejected" if mode == "full" else "diagnostic_only")
    report = {
        "schema_version": "1.0",
        "run_dir": str(root),
        "stage_result_path": str(stage_path),
        "environment_version": RECIPE_ENV_VERSION,
        "observation_version": RECIPE_OBSERVATION_VERSION,
        "point_version": RECIPE_POINT_VERSION,
        "loss_version": RECIPE_OPC_LOSS_VERSION,
        "label_version": PPO_RECIPE_LABEL_VERSION,
        "mode": mode,
        "status": status,
        "gate_split": "validation",
        "thresholds": {
            "max_relative_loss_vs_default_recipe": max_vs_default,
            "min_relative_improvement_vs_initial": min_vs_initial,
            "max_seed_loss_cv": max_seed_cv,
            "max_single_action_fraction": max_action_fraction,
        },
        "validation": {
            "clip_checks": validation_checks,
            "actions_not_collapsed": actions_ok,
            **validation_summary,
        },
        "clips": clip_reports,
        "tree_training_allowed": bool(accepted),
    }
    identity = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return report


def main(argv: Optional[List[str]] = None) -> int:
    """从命令行审核既有训练产物；smoke 诊断成功返回 0，full rejected 返回 2。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.recipe_ppo_quality")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = evaluate_recipe_ppo_run(args.run_dir, config)
    output = args.output or (args.run_dir / "ppo-quality.json")
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"拒绝覆盖已有不同质量报告：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_text(encoded, encoding="utf-8")
    print(output)
    print(report["status"])
    return 2 if report["status"] == "rejected" else 0


if __name__ == "__main__":
    raise SystemExit(main())
