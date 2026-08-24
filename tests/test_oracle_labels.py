"""本模块验证九动作 OpenILT 缓存到决策树标签的奖励最优选择和完整性检查。

输入为临时 NPZ、候选元数据和九动作指标缓存；输出为最佳位移类别及缺失动作失败断言。
关键依赖为 NumPy、Pydantic 与 pytest；测试不调用 GPU、OpenILT、PPO、Qwen 或网络。
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from opc_agent.oracle_labels import build_training_rows


def _inputs(tmp_path: Path):
    """写入一个点及九个动作的完整缓存，其中动作 6 加权损失最低。"""
    dataset = tmp_path / "candidate.npz"
    np.savez_compressed(
        str(dataset),
        observations=np.asarray([[0.1, 0.2, 0.3, 0.4, 1.0, 0, 0, 1, 0, 0]], dtype=np.float32),
        target=np.zeros((4, 4), dtype=np.float32),
        candidate_masks=np.zeros((1, 9, 4, 4), dtype=np.float32),
    )
    metadata = tmp_path / "candidate.metadata.json"
    metadata.write_text(json.dumps({
        "adapter_version": "raster-fragment-v1", "clip_id": "clip-1",
        "parent_layout": "M1_test1", "split": "train",
        "point_ids": ["p1"], "task_types": ["EPE"],
    }), encoding="utf-8")
    metrics = {f"0:{index}": {"l2": 100 + index, "epe": 2, "pvb": 3} for index in range(9)}
    metrics["0:6"] = {"l2": 1, "epe": 0, "pvb": 1}
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({
        "source_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(), "metrics": metrics,
    }), encoding="utf-8")
    return dataset, metadata, cache


def test_oracle_labels_select_minimum_weighted_loss(tmp_path: Path):
    """应选择加权 L2+100*EPE+PVB 最小的动作 6。"""
    dataset, metadata, cache = _inputs(tmp_path)
    result = build_training_rows(dataset, metadata, cache, {"l2": 1, "epe": 100, "pvb": 1})
    assert len(result.rows) == 1
    assert result.schema_version == "2.0"
    assert result.label_version == "oracle-weighted-loss-v2"
    assert result.rows[0].schema_version == "2.0"
    assert result.rows[0].displacement_class == 6
    assert result.rows[0].features["fill_ratio"] == pytest.approx(0.1)


def test_oracle_labels_break_ties_by_smallest_absolute_displacement(tmp_path: Path):
    """等损失动作不得按数组顺序偏向 -40nm，并应保留歧义与候选碰撞证据。"""
    dataset, metadata, cache = _inputs(tmp_path)
    payload = json.loads(cache.read_text(encoding="utf-8"))
    for index in range(9):
        payload["metrics"][f"0:{index}"] = {
            "l2": float(20 + index), "epe": 1.0, "pvb": 2.0,
            "mask_sha256": f"mask-{index}",
        }
    for index in (0, 1, 2, 3):
        payload["metrics"][f"0:{index}"] = {
            "l2": 1.0, "epe": 0.0, "pvb": 1.0,
            "mask_sha256": "same-mask",
        }
    cache.write_text(json.dumps(payload), encoding="utf-8")
    result = build_training_rows(dataset, metadata, cache, {"l2": 1, "epe": 100, "pvb": 1})
    row = result.rows[0]
    assert row.displacement_class == 3
    assert row.optimal_classes == [0, 1, 2, 3]
    assert row.ambiguous is True
    assert row.candidate_collision is True
    assert row.loss_margin == pytest.approx(124.0)


def test_oracle_labels_reject_missing_action_metric(tmp_path: Path):
    """九动作任一精算值缺失时不得生成标签。"""
    dataset, metadata, cache = _inputs(tmp_path)
    payload = json.loads(cache.read_text(encoding="utf-8"))
    del payload["metrics"]["0:8"]
    cache.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="缺少动作 8"):
        build_training_rows(dataset, metadata, cache, {"l2": 1, "epe": 100, "pvb": 1})
