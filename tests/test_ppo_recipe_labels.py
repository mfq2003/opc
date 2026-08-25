"""本模块验证决策树 EPE 标签只能来自质量报告选中的 PPO 最优 Recipe。

输入为一个带模型来源、边段特征和 accepted 报告哈希的最小运行；输出为标签版本、特征版本、
精确/量化位移和质量来源断言。关键依赖为 Pydantic 与 pytest；测试不调用 OpenILT、CUDA 或 PPO。
"""
import hashlib
import json
from pathlib import Path

from opc_agent.ppo_recipe_labels import build_epe_labels_from_ppo


def test_build_epe_labels_uses_quality_selected_ppo_recipe(tmp_path: Path):
    """转换器应保留 PPO 模型哈希和精确位移，且不能伪造 FRAG 行。"""
    recipe_path = tmp_path / "best.recipe.json"
    recipe = {
        "label_version": "ppo-simpleopc-multistep-v3",
        "loss_version": "paper-weighted-sum-initial-normalized-v1",
        "layout_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "displacement_classes_nm": [-40, -30, -20, -10, 0, 10, 20, 30, 40],
        "labels": [{
            "task_type": "EPE",
            "segment": {"segment_id": "p0:s0:f0"},
            "features": {
                "midpoint_x_nm": 10,
                "midpoint_y_nm": 20,
                "segment_length_nm": 32,
                "is_horizontal": 1,
                "normal_x": 0,
                "normal_y": 1,
                "initial_epe_sign": -1,
            },
            "displacement_class": 6,
            "ppo_displacement_nm": 20,
            "quantization_error_nm": 0,
        }],
    }
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    stage = {
        "environment": "simpleopc-multistep-v3",
        "loss_version": "paper-weighted-sum-initial-normalized-v1",
        "mode": "full",
        "clips": [{
            "clip_id": "M1_test1", "split": "train", "layout_sha256": "a" * 64,
            "ppo_recipes": [str(recipe_path)],
        }],
    }
    stage_path = tmp_path / "stage-result.json"
    stage_path.write_text(json.dumps(stage), encoding="utf-8")
    quality = {
        "status": "accepted",
        "tree_training_allowed": True,
        "loss_version": "paper-weighted-sum-initial-normalized-v1",
        "clips": [{"clip_id": "M1_test1", "best_recipe_path": str(recipe_path)}],
    }
    identity = json.dumps(quality, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    quality["report_sha256"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    quality_path = tmp_path / "ppo-quality.json"
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    dataset = build_epe_labels_from_ppo(stage_path, quality_path)
    assert dataset.feature_version == "simpleopc-segment-v1"
    assert dataset.label_version == "ppo-simpleopc-multistep-v3-epe-only"
    assert dataset.ppo_quality_status == "accepted"
    assert len(dataset.rows) == 1
    assert dataset.rows[0].ppo_displacement_nm == 20
    assert dataset.rows[0].quantization_error_nm == 0
    assert dataset.rows[0].ppo_model_sha256 == "b" * 64
