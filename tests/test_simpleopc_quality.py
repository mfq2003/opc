"""本模块验证只有通过 validation 物理质量、种子稳定性和动作分布门槛的 PPO 才能训练树。

输入为临时 stage、启发式基线、PPO 模型字节和 Recipe JSON；输出为 accepted/rejected 及模型哈希
防混用断言。关键依赖为 NumPy 与 pytest；测试不导入 Gymnasium、CUDA、OpenILT 或真实 PPO。
"""
import hashlib
import json
from pathlib import Path

import pytest

from opc_agent.simpleopc_quality import evaluate_simpleopc_ppo_run


def _run(tmp_path: Path, action_classes=(0, 8), mode="full") -> Path:
    """建立一个 validation clip、三个稳定 seed 的最小可审计运行。"""
    clip_root = tmp_path / "clips" / "M1_test7"
    clip_root.mkdir(parents=True)
    heuristic = clip_root / "simpleopc-heuristic.json"
    heuristic.write_text(json.dumps({
        "loss_version": "paper-weighted-sum-initial-normalized-v1",
        "reset": {"loss_scale": 121},
        "best_metrics": {"l2": 80, "epe": 1, "pvb": 20},
        "trajectory": [{
            "loss_version": "paper-weighted-sum-initial-normalized-v1",
            "metrics": {"l2": 100, "epe": 1, "pvb": 20},
            "raw_weighted_loss": 121,
            "loss": 1.0,
        }],
    }), encoding="utf-8")
    models = []
    recipes = []
    for seed, l2 in enumerate((69, 70, 71)):
        model = tmp_path / f"seed-{seed}.zip"
        model.write_bytes(f"model-{seed}".encode("ascii"))
        recipe = tmp_path / f"seed-{seed}.recipe.json"
        recipe.write_text(json.dumps({
            "label_version": "ppo-simpleopc-multistep-v3",
            "loss_version": "paper-weighted-sum-initial-normalized-v1",
            "initial_weighted_loss": 121,
            "loss_scale": 121,
            "layout_sha256": "a" * 64,
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "displacement_classes_nm": [-40, -30, -20, -10, 0, 10, 20, 30, 40],
            "seed": seed,
            "best_metrics": {"l2": l2, "epe": 1, "pvb": 20},
            "trajectory": [{
                "loss_version": "paper-weighted-sum-initial-normalized-v1",
                "metrics": {"l2": 100, "epe": 1, "pvb": 20},
                "raw_weighted_loss": 121,
                "loss": 1.0,
            }],
            "labels": [{
                "displacement_class": value,
                "ppo_displacement_nm": (-40, -30, -20, -10, 0, 10, 20, 30, 40)[value],
                "quantization_error_nm": 0,
            } for value in action_classes],
        }), encoding="utf-8")
        models.append(str(model))
        recipes.append(str(recipe))
    stage = {
        "environment": "simpleopc-multistep-v3",
        "loss_version": "paper-weighted-sum-initial-normalized-v1",
        "mode": mode,
        "displacement_classes_nm": [-40, -30, -20, -10, 0, 10, 20, 30, 40],
        "clips": [{
            "clip_id": "M1_test7", "split": "validation", "layout_sha256": "a" * 64,
            "segment_count": len(action_classes), "heuristic_path": str(heuristic),
            "models": models, "ppo_recipes": recipes,
        }],
    }
    (tmp_path / "stage-result.json").write_text(json.dumps(stage), encoding="utf-8")
    return tmp_path


def _config():
    """使用可读的等权测试损失和严格动作塌缩门槛。"""
    return {
        "oracle": {
            "reward_weights": {"l2": 1, "epe": 1, "pvb": 1},
            "seeds": [0, 1, 2],
        },
        "ppo_acceptance": {
            "max_relative_loss_vs_heuristic": 0,
            "min_relative_improvement_vs_initial": 0.01,
            "max_seed_loss_cv": 0.20,
            "max_single_action_fraction": 0.95,
        },
    }


def test_quality_gate_accepts_stable_validation_recipe_better_than_heuristic(tmp_path: Path):
    """稳定、改善且动作不塌缩的 validation PPO 应成为合格教师。"""
    report = evaluate_simpleopc_ppo_run(_run(tmp_path), _config())
    assert report["status"] == "accepted"
    assert report["tree_training_allowed"] is True
    assert report["clips"][0]["best_seed"] == 0
    assert len(report["report_sha256"]) == 64


def test_quality_gate_rejects_single_action_collapse(tmp_path: Path):
    """即使物理损失改善，全部边段同一类别仍不能成为树标签。"""
    report = evaluate_simpleopc_ppo_run(_run(tmp_path, action_classes=(4, 4)), _config())
    assert report["status"] == "rejected"
    assert report["validation"]["actions_not_collapsed"] is False


def test_quality_gate_rejects_smoke_as_teacher(tmp_path: Path):
    """冒烟运行只能验证可执行性，不能训练决策树。"""
    with pytest.raises(ValueError, match="smoke"):
        evaluate_simpleopc_ppo_run(_run(tmp_path, mode="smoke"), _config())
