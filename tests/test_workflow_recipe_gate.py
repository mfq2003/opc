"""本模块验证正式决策树不能绕过 PPO 教师质量门槛或用 EPE-only 数据提前训练。

输入为旧 Oracle 标签和 accepted 但缺 FRAG 的 PPO 标签；输出为 build-recipe 在调用 sklearn 前的
明确拒绝断言。关键依赖为 Pydantic 与 pytest；测试不训练模型、不调用 OpenILT、GPU 或网络。
"""
import json
from pathlib import Path

import pytest

from opc_agent.features import FEATURE_NAMES, SIMPLEOPC_SEGMENT_FEATURE_NAMES
from opc_agent.workflow import build_recipe_stage


def _row(features, task="EPE"):
    """构造一个最小点级标签行。"""
    return {
        "sample_id": f"sample-{task}",
        "clip_id": "M1_test1",
        "parent_layout": "M1_test1",
        "split": "train",
        "task_type": task,
        "features": features,
        "displacement_class": 4,
    }


def test_build_recipe_rejects_legacy_oracle_labels(tmp_path: Path):
    """即使旧标签可被 Schema 读取，也不能再成为正式决策树教师。"""
    dataset = tmp_path / "legacy.json"
    dataset.write_text(json.dumps({
        "feature_version": "geometry-v1",
        "label_version": "oracle-weighted-loss-v2",
        "rows": [_row({name: 0.0 for name in FEATURE_NAMES})],
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="只接受.*PPO SimpleOPC"):
        build_recipe_stage({"workflow": {"point_training_dataset": str(dataset)}}, tmp_path / "run")


def test_build_recipe_rejects_accepted_epe_only_before_frag_exists(tmp_path: Path):
    """EPE PPO 通过后仍必须等待 FRAG PPO，不能提前训练单树冒充双树。"""
    dataset = tmp_path / "epe-only.json"
    dataset.write_text(json.dumps({
        "feature_version": "simpleopc-segment-v1",
        "label_version": "ppo-simpleopc-multistep-v3-epe-only",
        "ppo_quality_status": "accepted",
        "ppo_quality_report_sha256": "a" * 64,
        "rows": [_row({name: 0.0 for name in SIMPLEOPC_SEGMENT_FEATURE_NAMES})],
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="同时具有.*EPE.*FRAG"):
        build_recipe_stage({"workflow": {"point_training_dataset": str(dataset)}}, tmp_path / "run")
