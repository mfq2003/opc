"""本文件验证新 Recipe PPO 只读质量审核不会把 smoke 或动作塌缩误判为正式通过。

测试用最小 JSON 产物模拟共享模型、固定默认 Recipe、EPE/FRAG 九分类标签和逐点一次回放；
不运行 OpenILT、CUDA 或 PPO，因此可在本地 CPU 回归中验证协议、损失与 validation 门槛。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from opc_agent.metrics import RECIPE_OPC_LOSS_VERSION
from opc_agent.recipe_contract import (
    PPO_RECIPE_LABEL_VERSION,
    RECIPE_ENV_VERSION,
    RECIPE_OBSERVATION_VERSION,
    RECIPE_POINT_VERSION,
)
from opc_agent.recipe_ppo_quality import evaluate_recipe_ppo_run


def _dump(path: Path, payload: dict) -> None:
    """把最小测试产物写成 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _trajectory(losses: list[float]) -> list[dict]:
    """生成与 L2-only 权重严格一致的逐点轨迹。"""
    point_ids = [None, "epe-0", "frag-0"]
    task_types = [None, "EPE", "FRAG"]
    return [
        {
            "step": index,
            "point_id": point_ids[index],
            "task_type": task_types[index],
            "loss": loss,
            "raw_weighted_loss": loss,
            "loss_version": RECIPE_OPC_LOSS_VERSION,
            "metrics": {"l2": loss, "epe": 0.0, "pvb": 0.0},
        }
        for index, loss in enumerate(losses)
    ]


def _make_run(tmp_path: Path, mode: str = "full", collapsed: bool = False):
    """建立一个可通过或可触发动作塌缩门槛的最小运行。"""
    run = tmp_path / "run"
    model = run / "models" / "shared.zip"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"shared-model")
    _dump(model.with_suffix(".metadata.json"), {
        "environment_version": RECIPE_ENV_VERSION,
        "observation_version": RECIPE_OBSERVATION_VERSION,
        "loss_version": RECIPE_OPC_LOSS_VERSION,
        "patch_shape": [5, 64, 64],
        "vector_shape": [14],
        "seed": 0,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    })
    baseline = run / "clips" / "M1_test7" / "default-recipe.json"
    _dump(baseline, {
        "environment_version": RECIPE_ENV_VERSION,
        "loss_version": RECIPE_OPC_LOSS_VERSION,
        "trajectory": _trajectory([100.0])[:1],
    })
    recipe_path = run / "recipes" / "M1_test7-seed-0.recipe.json"
    epe_class = 4 if collapsed else 3
    frag_class = 4 if collapsed else 5
    labels = [
        {
            "task_type": "EPE",
            "point": {"point_id": "epe-0"},
            "displacement_class": epe_class,
            "ppo_displacement_nm": float((-40, -30, -20, -10, 0, 10, 20, 30, 40)[epe_class]),
            "quantization_error_nm": 0.0,
        },
        {
            "task_type": "FRAG",
            "point": {"point_id": "frag-0"},
            "displacement_class": frag_class,
            "ppo_displacement_nm": float((-40, -30, -20, -10, 0, 10, 20, 30, 40)[frag_class]),
            "quantization_error_nm": 0.0,
        },
    ]
    _dump(recipe_path, {
        "environment_version": RECIPE_ENV_VERSION,
        "observation_version": RECIPE_OBSERVATION_VERSION,
        "point_version": RECIPE_POINT_VERSION,
        "loss_version": RECIPE_OPC_LOSS_VERSION,
        "label_version": PPO_RECIPE_LABEL_VERSION,
        "reward_mode": "paper_raw",
        "action_semantics": "one_absolute_nine_class_decision_per_recipe_point",
        "patch_shape": [5, 64, 64],
        "vector_shape": [14],
        "layout_sha256": "layout-hash",
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "seed": 0,
        "initial_weighted_loss": 100.0,
        "trajectory": _trajectory([100.0, 95.0, 90.0]),
        "best_metrics": {"l2": 90.0, "epe": 0.0, "pvb": 0.0},
        "labels": labels,
    })
    split = "validation" if mode == "full" else "train"
    clip_id = "M1_test7"
    _dump(run / "stage-result.json", {
        "environment": RECIPE_ENV_VERSION,
        "observation_version": RECIPE_OBSERVATION_VERSION,
        "point_version": RECIPE_POINT_VERSION,
        "loss_version": RECIPE_OPC_LOSS_VERSION,
        "label_version": PPO_RECIPE_LABEL_VERSION,
        "reward_mode": "paper_raw",
        "policy_scope": "shared_train_clips",
        "action_space": "Discrete(9)",
        "action_semantics": "one_absolute_nine_class_decision_per_recipe_point",
        "patch_shape": [5, 64, 64],
        "vector_shape": [14],
        "displacement_classes_nm": [-40, -30, -20, -10, 0, 10, 20, 30, 40],
        "openilt_mutation": "none",
        "mode": mode,
        "seeds": [0],
        "shared_models": [str(model)],
        "clips": [{
            "clip_id": clip_id,
            "split": split,
            "layout_sha256": "layout-hash",
            "point_count": 2,
            "epe_point_count": 1,
            "frag_point_count": 1,
            "episode_horizon": 2,
            "default_recipe_path": str(baseline),
            "models": [str(model)],
            "ppo_recipes": [str(recipe_path)],
        }],
    })
    config = {
        "oracle": {"reward_weights": {"l2": 1.0, "epe": 0.0, "pvb": 0.0}},
        "data": {
            "train_parents": [] if mode == "full" else [clip_id],
            "validation_parents": [clip_id] if mode == "full" else [],
            "test_parents": [],
        },
        "ppo_acceptance": {
            "max_relative_loss_vs_default_recipe": 0.0,
            "min_relative_improvement_vs_initial": 0.01,
            "max_seed_loss_cv": 0.20,
            "max_single_action_fraction": 1.0 if not collapsed else 0.95,
        },
    }
    return run, config


def test_full_recipe_run_can_be_accepted(tmp_path: Path) -> None:
    """full 运行只有同时通过损失、seed 与 EPE/FRAG 动作门槛才 accepted。"""
    run, config = _make_run(tmp_path)
    report = evaluate_recipe_ppo_run(run, config)
    assert report["status"] == "accepted"
    assert report["tree_training_allowed"] is True
    assert report["clips"][0]["best_relative_improvement_vs_initial"] == 0.1


def test_smoke_is_diagnostic_only(tmp_path: Path) -> None:
    """smoke 即使产物完整且损失改善，也不能成为论文质量结论。"""
    run, config = _make_run(tmp_path, mode="smoke")
    report = evaluate_recipe_ppo_run(run, config)
    assert report["status"] == "diagnostic_only"
    assert report["tree_training_allowed"] is False


def test_task_action_collapse_rejects_full_run(tmp_path: Path) -> None:
    """EPE 或 FRAG 单类动作占比超过门槛时必须拒绝。"""
    run, config = _make_run(tmp_path, collapsed=True)
    report = evaluate_recipe_ppo_run(run, config)
    assert report["status"] == "rejected"
    assert report["validation"]["actions_not_collapsed"] is False
