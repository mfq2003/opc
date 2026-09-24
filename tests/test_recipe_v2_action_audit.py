"""本文件验证 v2 动作可辨识性审计的分组、轨迹重建与失败门禁。

测试只构造最小 JSON 工件，不调用 OpenILT 或真实 Solver。重点保证每个点必须恰好包含
四个互异候选动作、incumbent 轨迹能够由候选结果重建，以及输出不会把诊断提升为正式验收。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from opc_agent.recipe_v2_action_audit import audit_search_run


def _candidate(point_id, call, action, j, mask, after, accepted=False):
    return {
        "point_id": point_id,
        "candidate_call": call,
        "action": action,
        "j": float(j),
        "metrics": {"l2": float(j), "epe": 0.0, "pvb": 0.0},
        "mask_sha256": mask,
        "feasible": True,
        "accepted": accepted,
        "incumbent_j_after_group": float(after),
    }


def _write_source(root: Path, rows):
    arm_root = root / "M1_test1" / "seed-0" / "coordinate"
    arm_root.mkdir(parents=True)
    result = {
        "baseline": {"j": 100.0},
        "best": {"j": 90.0},
        "candidate_calls": len(rows),
        "point_order": ["p1", "p2"],
    }
    (arm_root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (arm_root / "candidates.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "diagnostic_only",
        "accepted": False,
        "training_enabled": False,
        "arms": [{"layout_parent": "M1_test1", "method": "coordinate", "seed": 0}],
    }
    (root / "recipe-v2-search.json").write_text(json.dumps(summary), encoding="utf-8")
    config = {"recipe_v2": {"epe_normal_offsets_nm": [-20, -10, 0, 10, 20]}}
    (root / "config.snapshot.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _valid_rows():
    return [
        _candidate("p1", 1, 0, 100, "m0", 90),
        _candidate("p1", 2, 1, 90, "m1", 90, accepted=True),
        _candidate("p1", 3, 3, 95, "m3", 90),
        _candidate("p1", 4, 4, 110, "m4", 90),
        _candidate("p2", 5, 0, 90, "same", 90),
        _candidate("p2", 6, 2, 90, "same", 90),
        _candidate("p2", 7, 3, 90, "same", 90),
        _candidate("p2", 8, 4, 90, "same", 90),
    ]


def test_audit_reconstructs_rankable_and_invariant_groups(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source, _valid_rows())

    report = audit_search_run(source, output)

    assert report["status"] == "diagnostic_only"
    assert report["accepted"] is False
    assert report["solver_calls"] == 0
    assert report["overall"]["group_count"] == 2
    assert report["overall"]["candidate_count"] == 8
    assert report["overall"]["value_rankable_group_count"] == 1
    assert report["overall"]["fully_invariant_group_count"] == 1
    assert report["overall"]["improving_group_count"] == 1
    assert report["overall"]["total_realized_improvement"] == 10.0
    assert report["overall"]["value_rankable_improvement_share"] == 1.0
    assert report["overall"]["decision_signal"] == "proceed_to_offline_candidate_value_baseline"
    assert (output / "action-identifiability.json").exists()
    assert (output / "layout-summary.csv").exists()
    assert (output / "groups.csv").exists()


def test_audit_rejects_incomplete_candidate_group(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    rows = _valid_rows()[:-1]
    _write_source(source, rows)

    with pytest.raises(ValueError, match="候选数"):
        audit_search_run(source, output)
