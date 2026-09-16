"""本模块验证 Recipe PPO v2 的逐点法向几何、全局 FRAG、冻结 observation 与回报协议。

测试输入为正交矩形/L 形 target、内存 dissect 和 Fake solver；输出覆盖 q=p+delta*n、probe
合法性、FRAG 拓扑/ID、Golden 独立性、dense/terminal solver 调用和 final-only 产物。测试不导入
Gymnasium、CUDA、PyTorch 或 OpenILT，不能替代锁定提交上的真实光刻 preflight/smoke。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pytest
import yaml

import opc_agent.recipe_v2_runner as runner_module
import opc_agent.recipe_v2_small_train as small_train_module
from opc_agent.recipe_v2_search import search_recipe, search_budget
from opc_agent.recipe_v2 import (
    FixedProbeGoldenEvaluator,
    FrozenObservationCache,
    LocalEPEEpisode,
    build_action_candidates,
    build_golden_point_set,
    build_probe_window,
    build_v2_recipe_payload,
    candidate_action_mask,
    control_move_signs,
    dissect_global_fragments,
    outward_normal_from_target,
)
from opc_agent.recipe_v2_contract import (
    ACTOR_GEOMETRY_FIELDS,
    EPE_ACTION_OFFSETS_NM,
    EPE_DENSE_PROTOCOL,
    EPE_TERMINAL_PROTOCOL,
    EPEControlPoint,
    FragmentParameters,
    GoldenControlPoint,
    GoldenEvaluation,
    V2SolverResult,
    array_sha256,
    observation_version,
)
from opc_agent.recipe_v2_runner import (
    _deterministic_final_replay,
    _stability_numeric_summary,
    _training_numerics,
)
from opc_agent.recipe_v2_small_train import (
    _random_terminal_recipe_baseline,
    _rollout_observation_env_major,
    _rollout_scalar_matrix,
    _shared_training_numerics,
    _small_train_contract,
    _small_train_gate,
)


def _target() -> np.ndarray:
    """建立非方形矩形 target，用于同时检查 x/y 索引方向。"""
    target = np.zeros((200, 220), dtype=np.uint8)
    target[40:161, 40:181] = 1
    return target


@pytest.mark.parametrize("method", ["coordinate", "random"])
def test_search_budget_monotonicity_and_replay(method, tmp_path):
    """搜索保持全零单项护栏、严格下降及独立回放，并保留完整候选日志。"""
    import json

    _, episode = _episode(EPE_TERMINAL_PROTOCOL, patch_size=128,
                          action_offsets_nm=(-10, 0, 10))
    result = search_recipe(episode, method, 0, 4, tmp_path / method)
    assert result["candidate_calls"] == 4
    assert result["final_replay_equal"]
    assert result["best"]["j"] <= result["baseline"]["j"]
    assert result["solver_call_counts"]["total"] == 6
    rows = [json.loads(line) for line in (tmp_path / method / "candidates.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert all(len(row["actions"]) == 2 for row in rows)
    losses = [row["incumbent_j_after_group"] for row in rows]
    assert losses == sorted(losses, reverse=True)
    assert all(result["best"]["metrics"][key] <= result["baseline"]["metrics"][key]
               for key in ("l2", "epe", "pvb"))


def test_search_does_not_start_partial_coordinate_group(tmp_path):
    """不足以比较当前点全部候选时不消耗部分预算。"""
    _, episode = _episode(EPE_TERMINAL_PROTOCOL, patch_size=128,
                          action_offsets_nm=(-10, 0, 10))
    result = search_recipe(episode, "coordinate", 0, 1, tmp_path / "partial")
    assert result["candidate_calls"] == 0
    assert result["final_replay_equal"]


def test_coordinate_resume_reuses_complete_groups(tmp_path):
    """迁移完整候选组不重新求解，旧日志保持不变，最终仍独立回放。"""
    _, episode = _episode(EPE_TERMINAL_PROTOCOL, patch_size=128,
                          action_offsets_nm=(-10, 0, 10))
    source = tmp_path / "old"
    first = search_recipe(episode, "coordinate", 0, 2, source)
    original = (source / "candidates.jsonl").read_bytes()
    result = search_recipe(episode, "coordinate", 0, 4, tmp_path / "new",
                           resume_from=source, baseline_calls=0)
    assert result["reused_candidate_calls"] == 2
    assert result["solver_call_counts"]["total"] == 3
    assert result["final_replay_equal"]
    assert result["best"]["j"] <= first["best"]["j"]
    assert (source / "candidates.jsonl").read_bytes() == original


def test_coordinate_resume_rejects_wrong_actions(tmp_path):
    """旧候选与当前动作身份不匹配时拒绝复用。"""
    import json
    _, episode = _episode(EPE_TERMINAL_PROTOCOL, patch_size=128,
                          action_offsets_nm=(-10, 0, 10))
    source = tmp_path / "old"
    search_recipe(episode, "coordinate", 0, 2, source)
    path = source / "candidates.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["action"] = 99
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="冻结点顺序"):
        search_recipe(episode, "coordinate", 0, 4, tmp_path / "new", resume_from=source)


@pytest.mark.parametrize("points,expected", [(242, 968), (208, 832), (260, 1040), (104, 416), (246, 984)])
def test_search_budget_uses_actual_layout_point_count(points, expected):
    """不同版图均完整扫描一轮，不沿用 test5/6 的固定点数。"""
    episode = SimpleNamespace(episode_horizon=points, action_offsets_nm=(-20,-10,0,10,20))
    assert search_budget(episode) == expected


def test_search_config_covers_train_six_only():
    """六图搜索配置保留验证与测试隔离，预算由点数推导。"""
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/recipe_ppo_v2.yaml").read_text(encoding="utf-8"))
    search = config["search"]
    assert search["layout_parents"] == config["data"]["train_parents"]
    assert len(search["layout_parents"]) == 6
    assert search["seeds"] == [0]
    assert search["budget_policy"] == "one_full_coordinate_sweep"
    assert "candidate_budget" not in search
    assert not set(search["layout_parents"]) & set(config["data"]["validation_parents"] + config["data"]["test_parents"])


@pytest.mark.parametrize("seeds", [[0, 1, 2], [1], []])
def test_search_rejects_nonzero_or_multiple_seeds(seeds, tmp_path):
    """入口拒绝旧多种子设置，避免意外放大真实求解预算。"""
    from opc_agent.recipe_v2_search import run_v2_search
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/recipe_ppo_v2.yaml").read_text(encoding="utf-8"))
    config["search"]["seeds"] = seeds
    with pytest.raises(ValueError, match="单种子"):
        run_v2_search(config, tmp_path)


def _points():
    """返回一个水平上边点和一个垂直右边点，法线均指向 target 外部。"""
    return (
        EPEControlPoint(
            point_id="top",
            polygon_index=0,
            source_edge_index=0,
            segment_index=0,
            base_xy=(80, 40),
            segment_start_xy=(60, 40),
            segment_end_xy=(100, 40),
            normal_xy=(0, -1),
            start_corner_type=0,
            end_corner_type=0,
        ),
        EPEControlPoint(
            point_id="right",
            polygon_index=0,
            source_edge_index=1,
            segment_index=0,
            base_xy=(180, 100),
            segment_start_xy=(180, 80),
            segment_end_xy=(180, 120),
            normal_xy=(1, 0),
            start_corner_type=0,
            end_corner_type=0,
        ),
    )


def _golden_points():
    """返回从原始 target 冻结、且类型上不含任何 FRAG 字段的独立 Golden 点。"""
    return (
        GoldenControlPoint("golden-top", (80, 40), (0, -1)),
        GoldenControlPoint("golden-right", (180, 100), (1, 0)),
    )


class _FakeSolver:
    """以逐点 offset 总和确定性改变固定内部像素错误数，并记录完整 Recipe 调用。"""

    layout_sha256 = "a" * 64
    revision = "fake-openilt-not-formal"
    nm_per_coordinate = 1.0

    def __init__(self):
        self.target_image = _target()
        self.epe_points = _points()
        self.fragment_parameters = FragmentParameters(16, 32)
        self.calls = []

    def solve(self, normal_offsets_nm: Mapping[str, float]) -> V2SolverResult:
        if set(normal_offsets_nm) != {point.point_id for point in self.epe_points}:
            raise ValueError("Fake solver 只接受完整 Recipe")
        offsets = {point_id: float(value) for point_id, value in normal_offsets_nm.items()}
        self.calls.append(dict(sorted(offsets.items())))
        improvement = int(round(sum(offsets.values()) / 10.0))
        error_count = 20 - improvement
        if not 0 <= error_count <= 30:
            raise ValueError("Fake solver 测试动作超出预设范围")
        nominal = self.target_image.copy()
        for index in range(error_count):
            nominal[80, 60 + index] = 0
        digest = array_sha256(self.target_image)
        return V2SolverResult(
            mask_image=self.target_image.copy(),
            printed_nominal=nominal,
            printed_max=nominal.copy(),
            printed_min=nominal.copy(),
            recipe_epe_signs=tuple((point.point_id, 0.0) for point in self.epe_points),
            mask_sha256=digest,
            internal_trace=({"fake": True, "error_count": error_count},),
        )


def _episode(
    protocol: str,
    patch_size: int = 64,
    shuffle_points: bool = False,
    action_offsets_nm=EPE_ACTION_OFFSETS_NM,
):
    """建立使用独立固定 Golden evaluator 的最小 v2 episode。"""
    solver = _FakeSolver()
    golden_point_set = build_golden_point_set(
        target_image=solver.target_image,
        points=_golden_points(),
        nm_per_coordinate=solver.nm_per_coordinate,
        source_sha256="e" * 64,
    )
    evaluator = FixedProbeGoldenEvaluator(
        target_image=solver.target_image,
        golden_point_set=golden_point_set,
        reward_weights={"l2": 1, "epe": 100, "pvb": 1},
        diagnostic_probe_distance_nm=15,
        evaluator_source_sha256="b" * 64,
    )
    episode = LocalEPEEpisode(
        solver=solver,
        golden_evaluator=evaluator,
        reward_weights={"l2": 1, "epe": 100, "pvb": 1},
        training_protocol=protocol,
        patch_size=patch_size,
        action_offsets_nm=action_offsets_nm,
        control_probe_distance_nm=16,
        shuffle_points=shuffle_points,
    )
    return solver, episode


def test_normal_offset_moves_horizontal_and_vertical_points_on_normal_only():
    """水平 edge 只能改变 y，垂直 edge 只能改变 x。"""
    horizontal, vertical = _points()
    assert horizontal.moved_xy(10, 1) == (80, 30)
    assert horizontal.moved_xy(-10, 1) == (80, 50)
    assert vertical.moved_xy(10, 1) == (190, 100)
    assert vertical.moved_xy(-10, 1) == (170, 100)
    assert horizontal.as_recipe_dict(10, 1)["normal_offset_nm"] == 10.0
    assert horizontal.as_recipe_dict(10, 1)["moved_xy"] == [80, 30]


def test_probe_preflight_exposes_default_16nm_vs_40nm_conflict():
    """固定 16nm probe 下，直边上的 ±20/30/40nm 不能静默作为有效九类动作。"""
    point = _points()[0]
    candidates = build_action_candidates(point, EPE_ACTION_OFFSETS_NM, 16, 1, _target())
    mask = candidate_action_mask(candidates)
    assert mask.tolist() == [False, False, False, True, True, True, False, False, False]
    assert candidates[0].probe.in_bounds is True
    assert candidates[0].probe.outer_target_valid is False
    assert candidates[6].probe.in_bounds is True
    assert candidates[6].probe.inner_target_valid is False
    assert candidates[6].probe.outer_target_valid is True
    assert candidates[8].probe.in_bounds is False
    _, episode = _episode(EPE_DENSE_PROTOCOL)
    with pytest.raises(ValueError, match="普通 stable-baselines3 PPO"):
        episode.require_plain_ppo_compatible()


def test_probe_checks_bounds_before_numpy_indexing_and_marks_action_aliases():
    """负坐标不能借 NumPy 负索引回绕；粗数据库栅格上的重合动作必须标 alias。"""
    target = np.zeros((32, 64), dtype=np.uint8)
    target[2:30, 10:50] = 1
    point = EPEControlPoint(
        "near-top", 0, 0, 0, (20, 2), (10, 2), (30, 2), (0, -1)
    )
    probe = build_probe_window(point, 0, 16, 1, target)
    assert probe.in_bounds is False
    assert probe.inner_target_valid is False
    assert probe.outer_target_valid is False

    aliases = build_action_candidates(_points()[0], (-10, 0, 10), 25, 25, _target())
    assert aliases[0].alias_of_action_class is None
    assert aliases[1].alias_of_action_class == 0
    assert aliases[2].alias_of_action_class == 0


def test_control_adapter_rejects_invalid_probe_and_double_violation():
    """移动 crossing 无法一内一外或两侧同时违规时，solver 方向必须显式失败。"""
    point = _points()[0]
    with pytest.raises(ValueError, match="不能形成一内一外"):
        control_move_signs((point,), {"top": 40}, _target(), _target(), 16, 1)

    printed = _target()
    probe = build_probe_window(point, 0, 16, 1, _target())
    printed[probe.inner_xy[1], probe.inner_xy[0]] = 0
    printed[probe.outer_xy[1], probe.outer_xy[0]] = 1
    with pytest.raises(ValueError, match="同时违规"):
        control_move_signs((point,), {"top": 0}, printed, _target(), 16, 1)


def test_outward_normal_is_winding_independent_on_same_target_edge():
    """同一条 top edge 反向遍历后，栅格语义得到的外法线仍指向上方。"""
    target = _target()
    assert outward_normal_from_target((40, 40), (180, 40), target) == (0, -1)
    assert outward_normal_from_target((180, 40), (40, 40), target) == (0, -1)
    assert outward_normal_from_target((180, 40), (180, 160), target) == (1, 0)
    assert outward_normal_from_target((180, 160), (180, 40), target) == (1, 0)


def test_global_fragment_adapter_passes_integer_lengths_and_versions_ids():
    """每个 polygon 必须收到整数全局参数；同拓扑可识别 alias，但参数变化仍重建 point ID。"""
    calls = []

    def fake_dissect(polygon, lenCorner, lenUniform):
        calls.append((lenCorner, lenUniform))
        return [
            (tuple(polygon[index]), tuple(polygon[(index + 1) % len(polygon)]))
            for index in range(len(polygon))
        ]

    polygon = ((40, 40), (180, 40), (180, 160), (40, 160))
    first = dissect_global_fragments(
        (polygon,), _target(), FragmentParameters(16, 32), 1, fake_dissect
    )
    second = dissect_global_fragments(
        (polygon,), _target(), FragmentParameters(20, 40), 1, fake_dissect
    )
    assert calls == [(16, 32), (20, 40)]
    assert all(isinstance(value, int) for call in calls for value in call)
    assert first.topology_sha256 == second.topology_sha256
    assert first.fragmentation_sha256 != second.fragmentation_sha256
    assert {point.point_id for point in first.epe_points}.isdisjoint(
        point.point_id for point in second.epe_points
    )
    assert {point.normal_xy for point in first.epe_points} == {
        (0, -1), (1, 0), (0, 1), (-1, 0)
    }
    assert {point.start_corner_type for point in first.epe_points} == {1}

    with pytest.raises(ValueError, match="不能被 nm_per_coordinate"):
        dissect_global_fragments(
            (polygon,), _target(), FragmentParameters(16, 32), 3, fake_dissect
        )


def test_concave_polygon_corner_type_and_parent_segment_binding_survive_winding():
    """L 形凹角在 CW/CCW 输入中均保持凹语义，且 point 仍绑定唯一父 edge。"""
    target = np.zeros((220, 220), dtype=np.uint8)
    target[40:181, 40:91] = 1
    target[130:181, 40:181] = 1
    polygon = ((40, 40), (90, 40), (90, 130), (180, 130), (180, 180), (40, 180))

    def fake_dissect(points, lenCorner, lenUniform):
        del lenCorner, lenUniform
        return [
            (tuple(points[index]), tuple(points[(index + 1) % len(points)]))
            for index in range(len(points))
        ]

    forward = dissect_global_fragments((polygon,), target, FragmentParameters(), 1, fake_dissect)
    reverse = dissect_global_fragments((tuple(reversed(polygon)),), target, FragmentParameters(), 1, fake_dissect)
    assert -1 in {
        kind
        for geometry in (forward, reverse)
        for point in geometry.epe_points
        for kind in (point.start_corner_type, point.end_corner_type)
    }
    assert all(point.source_edge_index >= 0 and point.segment_index == 0 for point in forward.epe_points)


def test_frozen_observation_is_prefix_and_schedule_independent_for_same_point():
    """同一点在不同 prefix 和 schedule 下必须读取字节完全一致的 baseline observation。"""
    _, episode = _episode(EPE_DENSE_PROTOCOL, patch_size=64, shuffle_points=False)
    top_first, reset_info = episode.reset(seed=1, point_order=("top", "right"))
    top_hash = episode.observation_cache.get("top").sha256
    episode.step(5)
    episode.step(5)
    right_first, _ = episode.reset(seed=2, point_order=("right", "top"))
    episode.step(5)
    top_after_prefix = episode.observation_cache.get("top").as_dict()
    assert np.array_equal(top_first["image"], top_after_prefix["image"])
    assert np.array_equal(top_first["vector"], top_after_prefix["vector"])
    assert top_hash == episode.observation_cache.get("top").sha256
    assert reset_info["frag_point_count"] == 0
    assert top_first["image"].shape == (5, 64, 64)
    assert top_first["vector"].shape == (len(ACTOR_GEOMETRY_FIELDS),)
    assert not np.array_equal(top_first["image"], right_first["image"])


def test_64_and_128_observations_have_distinct_versions_and_hashes():
    """64 smoke 与 128 正式候选不能静默共享 observation 版本或内容哈希。"""
    solver = _FakeSolver()
    baseline = solver.solve({"top": 0, "right": 0})
    cache64 = FrozenObservationCache(
        solver.epe_points, solver.fragment_parameters, solver.target_image, baseline, 64, 1
    )
    cache128 = FrozenObservationCache(
        solver.epe_points, solver.fragment_parameters, solver.target_image, baseline, 128, 1
    )
    assert observation_version(64) != observation_version(128)
    assert cache64.cache_sha256 != cache128.cache_sha256
    assert cache64.get("top").image.shape == (5, 64, 64)
    assert cache128.get("top").image.shape == (5, 128, 128)
    with pytest.raises(ValueError, match="只版本化"):
        observation_version(96)


def test_baseline_state_hash_binds_epe_signs_and_cache_metadata_is_read_only():
    """同 raster 不同 baseline sign 必须改变 Actor state/hash，且公开哈希不可重绑定。"""
    solver = _FakeSolver()
    baseline = solver.solve({"top": 0, "right": 0})
    changed_signs = V2SolverResult(
        mask_image=baseline.mask_image,
        printed_nominal=baseline.printed_nominal,
        printed_max=baseline.printed_max,
        printed_min=baseline.printed_min,
        recipe_epe_signs=(("top", 1.0), ("right", -1.0)),
        mask_sha256=baseline.mask_sha256,
    )
    zero_cache = FrozenObservationCache(
        solver.epe_points, solver.fragment_parameters, solver.target_image, baseline, 64, 1
    )
    signed_cache = FrozenObservationCache(
        solver.epe_points, solver.fragment_parameters, solver.target_image, changed_signs, 64, 1
    )
    assert zero_cache.baseline_state_sha256 != signed_cache.baseline_state_sha256
    assert zero_cache.cache_sha256 != signed_cache.cache_sha256
    assert zero_cache.get("top").vector[9] == 0.0
    assert signed_cache.get("top").vector[9] == 1.0
    with pytest.raises(AttributeError):
        zero_cache.baseline_state_sha256 = "not-a-sha"
    with pytest.raises(AttributeError):
        zero_cache.cache_sha256 = "not-a-sha"


def test_dense_reward_telescopes_and_each_action_updates_one_point():
    """dense 每步只改当前 point，solver 每步一次，reward 总和等于 final raw 改善。"""
    solver, episode = _episode(EPE_DENSE_PROTOCOL)
    _, reset_info = episode.reset(point_order=("top", "right"))
    offsets_before = {"top": 0.0, "right": 0.0}
    _, reward1, done1, info1 = episode.step(5)
    _, reward2, done2, info2 = episode.step(5)
    assert done1 is False and done2 is True
    assert info1["recipe_offsets_nm"] == {"right": 0.0, "top": 10.0}
    assert info2["recipe_offsets_nm"] == {"right": 10.0, "top": 10.0}
    assert offsets_before == {"top": 0.0, "right": 0.0}
    assert len(solver.calls) == 3  # baseline + 两个候选
    assert episode.solver_call_counts == {
        "baseline_solver": 1,
        "candidate_solver": 2,
        "final_replay_solver": 0,
    }
    initial = reset_info["initial_raw_weighted_loss"]
    final = episode.final_golden_evaluation.raw_weighted_loss
    assert reward1 + reward2 == pytest.approx(1e-5 * (initial - final), abs=1e-12)
    assert episode.final_recipe_offsets_nm == {"right": 10.0, "top": 10.0}


def test_terminal_protocol_calls_candidate_solver_only_at_final_step():
    """terminal 前 N-1 步 reward/候选调用均为零，终点只调用一次完整 Recipe。"""
    solver, episode = _episode(EPE_TERMINAL_PROTOCOL)
    episode.reset(point_order=("top", "right"))
    _, reward1, done1, info1 = episode.step(5)
    assert reward1 == 0.0 and done1 is False
    assert info1["candidate_solver_called"] is False
    assert len(solver.calls) == 1
    _, reward2, done2, info2 = episode.step(5)
    assert done2 is True and reward2 > 0
    assert info2["candidate_solver_called"] is True
    assert len(solver.calls) == 2
    assert episode.solver_call_counts["candidate_solver"] == 1


def test_batch_final_replay_is_point_id_based_and_order_independent():
    """完整 batch replay 不受映射存储顺序影响，且每个候选只调用一次 solver。"""
    _, episode = _episode(EPE_TERMINAL_PROTOCOL)
    first = episode.replay_complete_action_map({"top": 5, "right": 3})
    second = episode.replay_complete_action_map({"right": 3, "top": 5})
    assert first[0] == second[0] == {"right": -10.0, "top": 10.0}
    assert first[3] == second[3]
    assert first[2].raw_weighted_loss == second[2].raw_weighted_loss
    assert episode.solver_call_counts == {
        "baseline_solver": 1,
        "candidate_solver": 0,
        "final_replay_solver": 2,
    }


def test_frag_change_invalidates_episode_and_golden_api_cannot_receive_offsets():
    """FRAG 改变后必须整体重建；Golden evaluator 的 evaluate API 没有 Recipe offset 参数。"""
    solver, episode = _episode(EPE_DENSE_PROTOCOL)
    episode.reset(point_order=("top", "right"))
    solver.fragment_parameters = FragmentParameters(20, 40)
    with pytest.raises(RuntimeError, match="必须重建 solver"):
        episode.step(5)

    assert tuple(episode.golden_evaluator.evaluate.__code__.co_varnames[:2]) == ("self", "result")


def test_v2_payload_uses_final_recipe_and_forbids_fake_golden_distance():
    """产物保存逐点 normal 字段、独立 Golden 身份和 diagnostic best，不得导出历史 best Recipe。"""
    _, episode = _episode(EPE_DENSE_PROTOCOL)
    episode.reset(point_order=("top", "right"))
    episode.step(5)
    episode.step(5)
    payload = build_v2_recipe_payload(
        episode,
        epe_model_sha256="c" * 64,
        frag_model_sha256="",
        source_hashes={"recipe_v2.py": "d" * 64},
    )
    assert payload["environment"] == "simpleopc-recipe-local-epe-global-frag-v2"
    assert payload["accepted_metrics_source"] is None
    assert payload["required_accepted_metrics_source"] == "batched_final_replay"
    assert payload["raw_metrics_source"] == "ppo_dense_sequential_final"
    assert payload["full_recipe_replay_sha256"] is None
    assert payload["sequential_final_replay_gap"] is None
    assert payload["final_recipe_complete"] is True
    assert payload["status"] == "diagnostic_only"
    assert payload["golden_evaluator"]["version"] == "fixed-target-probe-golden-diagnostic-v1"
    assert payload["golden_evaluator"]["parameters"]["diagnostic_probe_distance_nm"] == 15.0
    assert payload["golden_evaluator"]["parameters"]["nm_per_coordinate"] == 1.0
    assert payload["golden_evaluator"]["parameters"]["raster_mapping_version"] == (
        "db-coordinate-equals-raster-pixel-v1"
    )
    assert payload["golden_evaluator"]["parameters"]["threshold"] == 0.5
    assert payload["golden_evaluator"]["parameters"]["formal_openilt_epecheck"] is False
    assert payload["golden_evaluator"]["parameters"]["golden_point_set_sha256"]
    assert payload["golden_evaluator"]["frozen_target_sha256"]
    assert payload["golden_evaluator"]["sampling_state_sha256"]
    assert payload["golden_evaluator"]["nm_per_coordinate"] == 1.0
    assert payload["golden_evaluator"]["coordinate_system_sha256"]
    assert payload["golden_evaluator"]["contract_sha256"]
    assert payload["optional_extended_metrics"] == {
        "epe_n": None,
        "epe_d": None,
        "mrc_violations": None,
    }
    assert {point["point_id"] for point in payload["epe_points"]} == {"top", "right"}
    assert {point["normal_offset_nm"] for point in payload["epe_points"]} == {10.0}
    assert payload["frag_model_sha256"] is None
    text = repr(payload)
    assert "epe_control_distance_nm" not in text
    assert "golden_epe_distance_nm" not in text
    assert "best_recipe_offsets_nm" not in text


def test_epe_point_rejects_fractional_coordinates_and_freezes_input_lists():
    """frozen dataclass 必须真正冻结坐标，不能截断小数或保留可变 list 引用。"""
    base = [80, 40]
    start = [60, 40]
    point = EPEControlPoint(
        "immutable", 0, 0, 0, base, start, [100, 40], [0, -1]
    )
    base[0] = 999
    start[0] = 999
    assert point.base_xy == (80, 40)
    assert point.segment_start_xy == (60, 40)
    assert isinstance(point.base_xy, tuple)
    with pytest.raises(TypeError, match="禁止静默截断"):
        EPEControlPoint(
            "fractional", 0, 0, 0, (80.5, 40), (60, 40), (100, 40), (0, -1)
        )


def test_solver_result_rejects_nonbinary_raster_nan_sign_and_fake_mask_hash():
    """solver contract 必须在 observation/evaluator 前拒绝未归一化栅格、NaN sign 和伪哈希。"""
    mask = np.zeros((8, 8), dtype=np.uint8)
    common = {
        "mask_image": mask,
        "printed_nominal": mask,
        "printed_max": mask,
        "printed_min": mask,
        "recipe_epe_signs": (("point", 0.0),),
        "mask_sha256": array_sha256(mask),
    }
    bad_raster = dict(common)
    image_255 = np.full((8, 8), 255, dtype=np.uint8)
    bad_raster["mask_image"] = image_255
    bad_raster["mask_sha256"] = array_sha256(image_255)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        V2SolverResult(**bad_raster)
    bad_sign = dict(common)
    bad_sign["recipe_epe_signs"] = (("point", float("nan")),)
    with pytest.raises(ValueError, match="-1/0/1"):
        V2SolverResult(**bad_sign)
    bad_hash = dict(common)
    bad_hash["mask_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="内容不一致"):
        V2SolverResult(**bad_hash)


def test_step_failure_rolls_back_cursor_recipe_and_trajectory(monkeypatch):
    """solver/evaluator 异常只能增加调用尝试计数，不得留下半提交 episode 状态。"""
    solver, episode = _episode(EPE_DENSE_PROTOCOL)
    episode.reset(point_order=("top", "right"))
    before = episode.trajectory
    original_solve = solver.solve

    def fail_solve(_offsets):
        raise RuntimeError("injected solver failure")

    monkeypatch.setattr(solver, "solve", fail_solve)
    with pytest.raises(RuntimeError, match="injected solver failure"):
        episode.step(5)
    assert episode.trajectory == before
    with pytest.raises(RuntimeError, match="尚不完整"):
        _ = episode.final_recipe_offsets_nm

    monkeypatch.setattr(solver, "solve", original_solve)
    _, _, done, info = episode.step(5)
    assert done is False
    assert info["step"] == 1 and info["point_id"] == "top"
    assert info["recipe_offsets_nm"] == {"right": 0.0, "top": 10.0}


def test_episode_rejects_golden_loss_computed_with_different_weights():
    """evaluator 即使返回结构合法，也不能绕过 episode 冻结的 reward 权重。"""
    solver = _FakeSolver()
    point_set = build_golden_point_set(
        solver.target_image,
        _golden_points(),
        solver.nm_per_coordinate,
        "e" * 64,
    )
    base = FixedProbeGoldenEvaluator(
        solver.target_image,
        point_set,
        {"l2": 1, "epe": 100, "pvb": 1},
        15,
        "b" * 64,
    )

    class BadLossEvaluator:
        evaluator_version = base.evaluator_version
        evaluator_source_sha256 = base.evaluator_source_sha256
        frozen_target_sha256 = base.frozen_target_sha256
        sampling_state_sha256 = base.sampling_state_sha256
        nm_per_coordinate = base.nm_per_coordinate
        coordinate_system_sha256 = base.coordinate_system_sha256
        evaluator_contract_sha256 = base.evaluator_contract_sha256

        def evaluate(self, result):
            correct = base.evaluate(result)
            return GoldenEvaluation(
                metrics=correct.metrics,
                raw_weighted_loss=correct.raw_weighted_loss + 1.0,
                evaluator_version=correct.evaluator_version,
                evaluator_source_sha256=correct.evaluator_source_sha256,
                evaluator_contract_sha256=correct.evaluator_contract_sha256,
                evaluator_parameters=correct.evaluator_parameters,
            )

    episode = LocalEPEEpisode(
        solver,
        BadLossEvaluator(),
        {"l2": 1, "epe": 100, "pvb": 1},
        EPE_DENSE_PROTOCOL,
        shuffle_points=False,
    )
    with pytest.raises(ValueError, match="reward_weights 不一致"):
        episode.reset(point_order=("top", "right"))


def test_stateful_evaluator_cannot_mutate_contract_during_evaluation():
    """evaluator 在一次 evaluate 内改变 sampling/contract 身份时必须在状态提交前失败。"""
    solver = _FakeSolver()
    point_set = build_golden_point_set(
        solver.target_image,
        _golden_points(),
        1,
        "e" * 64,
    )
    base = FixedProbeGoldenEvaluator(
        solver.target_image,
        point_set,
        {"l2": 1, "epe": 100, "pvb": 1},
        15,
        "b" * 64,
    )

    class StatefulEvaluator:
        evaluator_version = base.evaluator_version
        evaluator_source_sha256 = base.evaluator_source_sha256
        frozen_target_sha256 = base.frozen_target_sha256
        sampling_state_sha256 = base.sampling_state_sha256
        nm_per_coordinate = base.nm_per_coordinate
        coordinate_system_sha256 = base.coordinate_system_sha256
        evaluator_contract_sha256 = base.evaluator_contract_sha256

        def evaluate(self, result):
            evaluation = base.evaluate(result)
            self.evaluator_contract_sha256 = "f" * 64
            return evaluation

    episode = LocalEPEEpisode(
        solver,
        StatefulEvaluator(),
        {"l2": 1, "epe": 100, "pvb": 1},
        EPE_DENSE_PROTOCOL,
        shuffle_points=False,
    )
    with pytest.raises(RuntimeError, match="contract 已改变"):
        episode.reset(point_order=("top", "right"))
    assert len(solver.calls) == 1
    with pytest.raises(RuntimeError, match="先 reset"):
        episode.step(5)


def test_golden_coordinate_scale_must_match_solver_scale():
    """Golden probe 的 nm/coordinate 换算必须与 solver 相同，不能仅比较 target hash。"""
    solver = _FakeSolver()
    mismatched = build_golden_point_set(
        solver.target_image,
        _golden_points(),
        3,
        "e" * 64,
    )
    evaluator = FixedProbeGoldenEvaluator(
        solver.target_image,
        mismatched,
        {"l2": 1, "epe": 100, "pvb": 1},
        15,
        "b" * 64,
    )
    with pytest.raises(ValueError, match="nm_per_coordinate 不一致"):
        LocalEPEEpisode(
            solver,
            evaluator,
            {"l2": 1, "epe": 100, "pvb": 1},
            EPE_DENSE_PROTOCOL,
        )


def test_solver_identity_drift_rolls_back_before_evaluator_or_step_commit(monkeypatch):
    """solver 回调内篡改 revision 时，post-solve invariant 必须阻止轨迹提交。"""
    solver, episode = _episode(EPE_DENSE_PROTOCOL)
    episode.reset(point_order=("top", "right"))
    before = episode.trajectory
    original_solve = solver.solve

    def drift_revision(offsets):
        result = original_solve(offsets)
        solver.revision = "drifted-revision"
        return result

    monkeypatch.setattr(solver, "solve", drift_revision)
    with pytest.raises(RuntimeError, match="revision 已改变"):
        episode.step(5)
    assert episode.trajectory == before

    solver.revision = "fake-openilt-not-formal"
    monkeypatch.setattr(solver, "solve", original_solve)
    _, _, done, info = episode.step(5)
    assert done is False and info["step"] == 1


def test_solver_duck_object_cannot_bypass_v2_result_validation(monkeypatch):
    """Protocol 类型注解不是运行时校验；episode 必须拒绝未构造 V2SolverResult 的 duck。"""
    solver, episode = _episode(EPE_DENSE_PROTOCOL)
    image = solver.target_image.copy()
    duck = SimpleNamespace(
        mask_image=image,
        printed_nominal=image,
        printed_max=image,
        printed_min=image,
        recipe_epe_signs=(("top", 0.0), ("right", 0.0)),
        mask_sha256=array_sha256(image),
        internal_trace=(),
    )
    monkeypatch.setattr(solver, "solve", lambda _offsets: duck)
    with pytest.raises(TypeError, match="V2SolverResult"):
        episode.reset(point_order=("top", "right"))


def test_golden_parameters_and_episode_metadata_are_mutation_isolated():
    """公开 accessor 返回副本；关键协议/哈希属性不可在 final 后被重绑定。"""
    _, episode = _episode(EPE_DENSE_PROTOCOL)
    episode.reset(point_order=("top", "right"))
    episode.step(5)
    episode.step(3)
    final = episode.final_golden_evaluation
    cached = episode.observation_cache.get("top")
    original_pixel = float(cached.image[0, 0, 0])
    cached.image.setflags(write=True)
    cached.image[0, 0, 0] = 1.0 - original_pixel
    assert episode.observation_cache.get("top").image[0, 0, 0] == original_pixel

    result_copy = episode.final_result
    original_mask_pixel = int(result_copy.mask_image[0, 0])
    result_copy.mask_image.setflags(write=True)
    result_copy.mask_image[0, 0] = 1 - original_mask_pixel
    assert episode.final_result.mask_image[0, 0] == original_mask_pixel

    external = final.parameters_dict()
    external["reward_weights"]["l2"] = 999
    assert final.parameters_dict()["reward_weights"]["l2"] == 1.0
    frozen_weights = dict(final.evaluator_parameters)["reward_weights"]
    with pytest.raises(TypeError):
        frozen_weights["l2"] = 999

    original_hash = episode.final_recipe_sha256
    for name, value in (
        ("training_protocol", EPE_TERMINAL_PROTOCOL),
        ("action_table_sha256", "0" * 64),
        ("point_set_sha256", "0" * 64),
        ("geometry_identity_sha256", "0" * 64),
    ):
        with pytest.raises(AttributeError):
            setattr(episode, name, value)
    with pytest.raises(AttributeError):
        episode.golden_evaluator.golden_point_set = build_golden_point_set(
            _target(), _golden_points(), 1, "e" * 64
        )
    assert episode.final_recipe_sha256 == original_hash

    payload = build_v2_recipe_payload(
        episode,
        epe_model_sha256="c" * 64,
        frag_model_sha256="",
        source_hashes={"recipe_v2.py": "d" * 64},
    )
    assert payload["training_protocol"] == EPE_DENSE_PROTOCOL
    assert payload["golden_evaluator"]["parameters"]["reward_weights"]["l2"] == 1.0


def test_golden_point_set_is_order_stable_and_cannot_accept_frag_points():
    """Golden 点集身份只绑定冻结 target，不接收含 FRAG 字段的 Recipe 控制点。"""
    solver = _FakeSolver()
    forward = build_golden_point_set(
        solver.target_image,
        _golden_points(),
        1,
        "e" * 64,
    )
    reverse = build_golden_point_set(
        solver.target_image,
        tuple(reversed(_golden_points())),
        1,
        "e" * 64,
    )
    assert forward.point_set_sha256 == reverse.point_set_sha256
    with pytest.raises(TypeError, match="GoldenPointSet"):
        FixedProbeGoldenEvaluator(
            solver.target_image,
            solver.epe_points,
            {"l2": 1, "epe": 100, "pvb": 1},
            15,
            "b" * 64,
        )

    first = FixedProbeGoldenEvaluator(
        solver.target_image,
        forward,
        {"l2": 1, "epe": 100, "pvb": 1},
        15,
        "b" * 64,
    )
    solver.fragment_parameters = FragmentParameters(20, 40)
    second = FixedProbeGoldenEvaluator(
        solver.target_image,
        reverse,
        {"l2": 1, "epe": 100, "pvb": 1},
        15,
        "b" * 64,
    )
    result = solver.solve({"top": 0, "right": 0})
    assert first.evaluator_contract_sha256 == second.evaluator_contract_sha256
    assert first.evaluate(result).metrics == second.evaluate(result).metrics


def test_payload_uses_final_not_better_intermediate_prefix():
    """第一步改善、末步回退时，产物仍必须保存完整 final，而不能泄漏 best-prefix。"""
    _, episode = _episode(EPE_DENSE_PROTOCOL)
    episode.reset(point_order=("top", "right"))
    episode.step(5)  # top +10，成为诊断 best-prefix
    episode.step(3)  # right -10，final 回到基准 loss
    payload = build_v2_recipe_payload(
        episode,
        epe_model_sha256="c" * 64,
        frag_model_sha256="",
        source_hashes={"recipe_v2.py": "d" * 64},
    )
    offsets = {point["point_id"]: point["normal_offset_nm"] for point in payload["epe_points"]}
    assert offsets == {"right": -10.0, "top": 10.0}
    assert payload["diagnostic_best_prefix"]["raw_weighted_loss"] < payload["raw_metrics"]["weighted_loss"]
    assert payload["raw_metrics"]["weighted_loss"] == episode.final_golden_evaluation.raw_weighted_loss


def test_sequential_final_and_batch_replay_have_same_recipe_metrics_and_mask():
    """相同完整 point action map 的 sequential final 与 batch replay 必须逐项一致。"""
    _, episode = _episode(EPE_DENSE_PROTOCOL)
    episode.reset(point_order=("top", "right"))
    episode.step(5)
    episode.step(3)
    sequential_hash = episode.final_recipe_sha256
    sequential_golden = episode.final_golden_evaluation
    sequential_mask_sha256 = episode.final_result.mask_sha256
    _, replay_result, replay_golden, replay_hash = episode.replay_complete_action_map(
        {"right": 3, "top": 5}
    )
    assert replay_hash == sequential_hash
    assert replay_golden.metrics == sequential_golden.metrics
    assert replay_golden.raw_weighted_loss == sequential_golden.raw_weighted_loss
    assert replay_result.mask_sha256 == sequential_mask_sha256


def test_episode_and_batch_replay_reject_fractional_actions():
    """离散动作 API 不得把 1.9 静默截断为 action 1。"""
    _, episode = _episode(EPE_DENSE_PROTOCOL)
    episode.reset(point_order=("top", "right"))
    with pytest.raises(TypeError, match="整数离散编号"):
        episode.step(5.0)
    assert len(episode.trajectory) == 1
    with pytest.raises(TypeError, match="整数离散编号"):
        episode.replay_complete_action_map({"top": 5, "right": 3.2})


def test_plain_gym_wrapper_hard_gates_invalid_actions_and_contains_observation():
    """普通 PPO wrapper 构造期硬门控；全合法小动作表才允许 reset。"""
    pytest.importorskip("gymnasium")
    from opc_agent.recipe_ppo_v2 import LocalEPEPPOEnv

    solver, blocked = _episode(EPE_DENSE_PROTOCOL)
    assert solver.calls == []
    with pytest.raises(ValueError, match="普通 stable-baselines3 PPO"):
        LocalEPEPPOEnv(blocked)
    assert solver.calls == []

    point_set = build_golden_point_set(
        solver.target_image,
        _golden_points(),
        1,
        "e" * 64,
    )
    evaluator = FixedProbeGoldenEvaluator(
        solver.target_image,
        point_set,
        {"l2": 1, "epe": 100, "pvb": 1},
        15,
        "b" * 64,
    )
    allowed = LocalEPEEpisode(
        solver,
        evaluator,
        {"l2": 1, "epe": 100, "pvb": 1},
        EPE_DENSE_PROTOCOL,
        action_offsets_nm=(-10, 0, 10),
        shuffle_points=False,
    )
    env = LocalEPEPPOEnv(allowed)
    observation, _ = env.reset(options={"point_order": ("top", "right")})
    assert env.observation_space.contains(observation)
    with pytest.raises(ValueError, match="Discrete action_space"):
        env.step(1.9)


def test_v2_config_records_shared_terminal_small_training_contract():
    """配置必须冻结单共享模型的小训练预算，同时继续阻断 dense 和长训练。"""
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "recipe_ppo_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["status"] == (
        "shared_terminal_small_training_rollout_shape_fix_cloud_rerun_pending_"
        "long_training_disabled"
    )
    assert config["training"]["enabled"] is False
    assert "128_dense_terminal_episode_smoke_not_completed" not in (
        config["training"]["blocked_by"]
    )
    assert "real_128_observation_smoke_not_completed" not in (
        config["training"]["blocked_by"]
    )
    report_examples = config["recipe_v2"]["observation"]["report_examples"]
    assert report_examples == {
        "enabled": True,
        "example_count_per_layout": 4,
        "selection_policy": "geometry-diverse-by-point-id-v1",
    }
    assert "ppo_input_examples_cloud_artifact_not_completed" not in (
        config["training"]["blocked_by"]
    )
    assert "per_layout_golden_contract_hashes_not_frozen" not in (
        config["training"]["blocked_by"]
    )
    assert config["golden"]["verification_status"] == (
        "train_layout_contracts_frozen_from_two_matching_preflights"
    )
    assert set(config["golden"]["layout_contracts"]) == {
        f"M1_test{index}" for index in range(1, 7)
    }
    assert all(
        len(contract) == 3
        and all(len(str(value)) == 64 for value in contract.values())
        for contract in config["golden"]["layout_contracts"].values()
    )
    assert config["preflight"]["sensitivity_point_limit"] == 8
    assert config["recipe_v2"]["epe_normal_offsets_nm"] == [-20, -10, 0, 10, 20]
    assert config["recipe_v2"]["epe_probe_distance_nm"] == 24
    assert config["recipe_v2"]["control_conflict_policy"] == (
        "both-sides-conflict-stay-v1"
    )
    assert config["recipe_v2"]["observation"]["patch_size"] == 128
    assert config["recipe_v2"]["observation"]["legacy_64_status"] == (
        "excluded_insufficient_control_context"
    )
    assert config["preflight"]["sensitivity_actions_nm"] == [-20, -10, 10, 20]
    assert config["golden"]["constraint_coordinate"] == 15
    assert config["golden"]["source_sha256"] == (
        "cc2c111993491f9f0123e0bbad5e9815acb849e0f9d36b3ccefe0e12589f3d8c"
    )
    assert config["golden"]["sampling_state_sha256"] is None
    assert config["golden"]["coordinate_system_sha256"] is None
    assert config["golden"]["evaluator_contract_sha256"] is None
    assert config["training"]["smoke"] == {
        "enabled": True,
        "layout_parent": "M1_test4",
        "protocols": [EPE_DENSE_PROTOCOL, EPE_TERMINAL_PROTOCOL],
        "seed": 0,
        "total_timesteps": 104,
        "n_steps": 104,
        "batch_size": 52,
        "n_epochs": 1,
        "learning_rate": 0.0003,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_return_p99_abs": 10.0,
        "max_approx_kl": 0.1,
        "max_clip_fraction": 0.8,
    }
    assert config["training"]["pilot"] == {
        "enabled": True,
        "layout_parent": "M1_test4",
        "protocols": [EPE_DENSE_PROTOCOL, EPE_TERMINAL_PROTOCOL],
        "seed": 0,
        "rollout_count": 3,
        "consecutive_failure_updates": 3,
        "max_value_target_normalized_rmse": 10.0,
        "max_return_p99_abs": 10.0,
        "max_approx_kl": 0.1,
        "max_clip_fraction": 0.8,
    }
    assert config["training"]["small_train"] == {
        "enabled": True,
        "protocols": [EPE_TERMINAL_PROTOCOL],
        "layout_parents": ["M1_test5", "M1_test6"],
        "seed": 0,
        "episodes_per_layout_per_update": 2,
        "update_count": 5,
        "episodes_per_layout": 10,
        "n_steps_per_env": 492,
        "total_timesteps": 4920,
        "batch_size": 123,
        "n_epochs": 1,
        "learning_rate": 0.0003,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "reward_scale": 0.00001,
        "random_recipe_count_per_layout": 10,
        "consecutive_failure_updates": 3,
        "max_post_update_value_target_normalized_rmse": 10.0,
        "max_return_p99_abs": 10.0,
        "max_approx_kl": 0.1,
        "max_clip_fraction": 0.8,
    }
    assert config["training"]["blocked_by"] == [
        "shared_terminal_small_training_not_completed",
        "dense_critic_value_target_rmse_failed_three_consecutive_updates",
    ]
    assert config["recipe_v2"]["geometry_adapter"] == {
        "version": "openilt-dissect-parent-edge-adapter-v2",
        "raster_mapping_version": "db-coordinate-equals-raster-pixel-v1",
        "raster_scale": 1.0,
        "raster_offset_xy": [0, 0],
        "normal_probe_semantics_version": "target-two-sided-axis-probe-v1",
        "normal_probe_coordinate": 2,
        "minimum_fragment_rule_version": "min-corner-uniform-coordinate-v1",
    }
    assert all(value is None for value in config["acceptance"].values())


def test_v2_runner_deterministic_final_replay_uses_complete_recipe():
    """PPO runner 的确定性动作必须覆盖全部点，并与独立 final replay 一致。"""

    class StayModel:
        """始终输出零偏移动作类的最小确定性模型替身。"""

        def predict(self, _observation, deterministic=True):
            assert deterministic is True
            return np.asarray(4), None

    _, episode = _episode(EPE_TERMINAL_PROTOCOL, patch_size=128)
    result = _deterministic_final_replay(StayModel(), episode, seed=0)

    assert result["point_count"] == 2
    assert result["action_class_counts"]["4"] == 2
    assert result["dominant_action_fraction"] == 1.0
    assert result["final_replay_equal"] is True
    assert result["solver_call_counts"] == {
        "baseline_solver": 1,
        "candidate_solver": 1,
        "final_replay_solver": 1,
    }


def test_v2_runner_training_numerics_reports_finite_rollout_and_update_metrics():
    """训练 smoke 必须保存有限 return/value 诊断与 KL、clip 指标。"""
    model = SimpleNamespace(
        rollout_buffer=SimpleNamespace(
            returns=np.asarray([[0.2], [-0.1]], dtype=np.float32),
            values=np.asarray([[0.1], [0.0]], dtype=np.float32),
            rewards=np.asarray([[0.2], [-0.1]], dtype=np.float32),
        ),
        logger=SimpleNamespace(name_to_value={
            "train/approx_kl": 0.01,
            "train/clip_fraction": 0.2,
            "train/value_loss": 0.03,
        }),
        _n_updates=1,
    )

    result = _training_numerics(model)

    assert result["sample_count"] == 2
    assert result["all_finite"] is True
    assert result["return_p99_abs"] < 1.0
    assert result["logger"]["approx_kl"] == 0.01
    assert result["logger"]["clip_fraction"] == 0.2
    assert result["update_count"] == 1


def test_v2_runner_marks_negligible_variance_explained_variance_as_undefined():
    """terminal 近常量 return 的有限病态 EV 也必须标记为未定义。"""
    model = SimpleNamespace(
        rollout_buffer=SimpleNamespace(
            returns=np.asarray([[0.2], [0.20000002]], dtype=np.float32),
            values=np.asarray([[0.1], [0.0]], dtype=np.float32),
            rewards=np.asarray([[0.0], [0.2]], dtype=np.float32),
        ),
        logger=SimpleNamespace(name_to_value={
            "train/approx_kl": 0.01,
            "train/clip_fraction": 0.2,
            "train/explained_variance": -3.7e14,
        }),
        _n_updates=1,
    )

    result = _training_numerics(model)

    assert result["all_finite"] is True
    assert 0.0 < result["return_variance"] <= 1e-12
    assert result["return_std"] > 0.0
    assert result["explained_variance_defined"] is False
    assert result["logger"]["explained_variance"] is None
    assert result["explained_variance_undefined_reason"] == (
        "single-rollout-return-variance-is-zero-or-negligible"
    )


def test_v2_stability_summary_requires_three_consecutive_breaches():
    """间断超限不能误报；连续三次 Critic RMSE 超限必须令 pilot 失败。"""
    pilot = {
        "rollout_count": 3,
        "consecutive_failure_updates": 3,
        "max_value_target_normalized_rmse": 10.0,
        "max_return_p99_abs": 10.0,
        "max_approx_kl": 0.1,
        "max_clip_fraction": 0.8,
    }

    def history(rmses):
        return [
            {
                "update_index": index,
                "numerics": {
                    "all_finite": True,
                    "return_p99_abs": 0.1,
                    "value_target_normalized_rmse": rmse,
                    "logger": {"approx_kl": 0.01, "clip_fraction": 0.0},
                },
            }
            for index, rmse in enumerate(rmses, start=1)
        ]

    intermittent = _stability_numeric_summary(history([11.0, 9.0, 11.0]), pilot)
    assert intermittent["pass"] is True
    assert intermittent["max_consecutive_breaches"][
        "value_target_normalized_rmse"
    ] == 1

    consecutive = _stability_numeric_summary(history([11.0, 12.0, 13.0]), pilot)
    assert consecutive["pass"] is False
    assert consecutive["failure_reasons"] == [
        "value-target-normalized-rmse-high-for-three-updates"
    ]


def test_v2_pilot_model_continues_exactly_two_additional_rollouts(
    monkeypatch, tmp_path
):
    """pilot 必须在首个 smoke 更新后续跑两次，而不是重建三个独立模型。"""

    class FakeModel:
        """记录 learn 调用和累计 timestep 的最小连续训练替身。"""

        def __init__(self):
            self.num_timesteps = 2
            self.update_count = 1
            self.learn_calls = []
            self.policy = object()

        def learn(self, total_timesteps, reset_num_timesteps):
            self.learn_calls.append((total_timesteps, reset_num_timesteps))
            self.num_timesteps += int(total_timesteps)
            self.update_count += 1
            return self

        def save(self, model_base):
            Path(str(model_base) + ".zip").write_bytes(b"pilot-model")

    model = FakeModel()

    def numerics(current_model):
        return {
            "all_finite": True,
            "return_p99_abs": 0.1,
            "value_target_normalized_rmse": 1.0,
            "update_count": current_model.update_count,
            "logger": {"approx_kl": 0.01, "clip_fraction": 0.0},
        }

    def first_update(_env, _smoke, model_base):
        model.save(model_base)
        return model, {
            "model": Path(str(model_base) + ".zip").name,
            "model_sha256": "0" * 64,
            "initial_policy_sha256": "1" * 64,
            "trained_policy_sha256": "2" * 64,
            "total_timesteps_requested": 2,
            "total_timesteps_actual": 2,
            "elapsed_seconds": 0.1,
            "python_version": "test",
            "torch_version": "test",
            "cuda_device_name": "test",
            "stable_baselines3_version": "2.0.0",
            "gymnasium_version": "0.28.1",
            "numerics": numerics(model),
        }

    monkeypatch.setattr(runner_module, "_train_sb3_model", first_update)
    monkeypatch.setattr(runner_module, "_training_numerics", numerics)
    monkeypatch.setattr(
        runner_module,
        "_state_dict_sha256",
        lambda _policy: str(model.update_count) * 64,
    )
    smoke = {"n_steps": 2, "total_timesteps": 2}
    pilot = {
        "rollout_count": 3,
        "consecutive_failure_updates": 3,
        "max_value_target_normalized_rmse": 10.0,
        "max_return_p99_abs": 10.0,
        "max_approx_kl": 0.1,
        "max_clip_fraction": 0.8,
    }

    trained_model, result = runner_module._train_sb3_pilot_model(
        object(), smoke, pilot, tmp_path / "model-pilot-terminal"
    )

    assert trained_model is model
    assert model.learn_calls == [(2, False), (2, False)]
    assert result["total_timesteps_requested"] == 6
    assert result["total_timesteps_actual"] == 6
    assert [item["numerics"]["update_count"] for item in result["update_history"]] == [
        1,
        2,
        3,
    ]
    assert result["stability"]["pass"] is True


def test_v2_ppo_smoke_orchestration_keeps_terminal_arm_diagnostic(
    monkeypatch, tmp_path
):
    """训练编排必须保存输入、模型诊断和 final replay，但绝不产生 accepted。"""
    pytest.importorskip("gymnasium")

    class FakeModel:
        """为编排测试输出中间的零偏移动作类。"""

        def predict(self, _observation, deterministic=True):
            assert deterministic is True
            return np.asarray(1), None

    def build_fake(_config, _layout_parent, **_kwargs):
        _, episode = _episode(
            EPE_TERMINAL_PROTOCOL,
            patch_size=128,
            shuffle_points=False,
            action_offsets_nm=(-10, 0, 10),
        )
        return episode.solver, episode.golden_evaluator, ("fake-layout-contract",)

    def train_fake(_env, smoke, model_base):
        model_path = Path(str(model_base) + ".zip")
        model_path.write_bytes(b"fake-model")
        return FakeModel(), {
            "model": model_path.name,
            "model_sha256": array_sha256(np.frombuffer(b"fake-model", dtype=np.uint8)),
            "initial_policy_sha256": "1" * 64,
            "trained_policy_sha256": "2" * 64,
            "total_timesteps_requested": int(smoke["total_timesteps"]),
            "total_timesteps_actual": int(smoke["total_timesteps"]),
            "elapsed_seconds": 0.1,
            "python_version": "test",
            "torch_version": "test",
            "cuda_device_name": "test",
            "stable_baselines3_version": "2.0.0",
            "gymnasium_version": "0.28.1",
            "numerics": {
                "sample_count": 2,
                "return_p99_abs": 0.1,
                "value_target_normalized_rmse": 1.0,
                "update_count": 1,
                "logger": {"approx_kl": 0.01, "clip_fraction": 0.1},
                "all_finite": True,
            },
        }

    config = {
        "environment": "simpleopc-recipe-local-epe-global-frag-v2",
        "data": {"openilt_dir": "fake"},
        "openilt": {"commit": "fixed"},
        "recipe_v2": {
            "epe_normal_offsets_nm": [-10, 0, 10],
            "epe_probe_distance_nm": 16,
            "observation": {
                "patch_size": 128,
                "report_examples": {
                    "example_count_per_layout": 2,
                    "selection_policy": "geometry-diverse-by-point-id-v1",
                },
            },
            "reward": {
                "weights": {"l2": 1, "epe": 100, "pvb": 1},
                "scale": 0.00001,
            },
        },
        "training": {
            "enabled": False,
            "smoke": {
                "enabled": True,
                "layout_parent": "fake",
                "protocols": [EPE_DENSE_PROTOCOL, EPE_TERMINAL_PROTOCOL],
                "seed": 0,
                "total_timesteps": 2,
                "n_steps": 2,
                "batch_size": 2,
                "max_return_p99_abs": 10,
                "max_approx_kl": 0.1,
                "max_clip_fraction": 0.8,
            },
        },
    }
    monkeypatch.setattr(
        runner_module, "_build_v2_openilt_solver_and_evaluator", build_fake
    )
    monkeypatch.setattr(runner_module, "_train_sb3_model", train_fake)
    monkeypatch.setattr(runner_module, "_validate_openilt", lambda *_args: "fixed")

    result = runner_module.run_v2_ppo_smoke(
        config,
        artifact_root=tmp_path,
        protocol_alias="terminal",
        layout_parent="fake",
    )

    assert result["status"] == "diagnostic_only"
    assert result["training_protocol"] == EPE_TERMINAL_PROTOCOL
    assert result["pass"] is True
    assert result["accepted"] is False
    assert result["long_training_enabled"] is False
    assert result["deterministic_final_replay"]["final_replay_equal"] is True
    assert (tmp_path / "ppo-input-examples" / "manifest.json").is_file()


def test_v2_ppo_pilot_orchestration_keeps_three_updates_diagnostic(
    monkeypatch, tmp_path
):
    """三次更新 pilot 必须保存逐次诊断和 final replay，但不得开启长训练。"""
    pytest.importorskip("gymnasium")

    class FakeModel:
        """为 pilot 编排测试始终输出零偏移动作。"""

        def predict(self, _observation, deterministic=True):
            assert deterministic is True
            return np.asarray(1), None

    def build_fake(_config, _layout_parent, **_kwargs):
        _, episode = _episode(
            EPE_TERMINAL_PROTOCOL, patch_size=128, shuffle_points=False
        )
        return episode.solver, episode.golden_evaluator, ("fake-layout-contract",)

    def train_fake(_env, _smoke, pilot, model_base):
        model_path = Path(str(model_base) + ".zip")
        model_path.write_bytes(b"fake-pilot-model")
        history = [
            {
                "update_index": index,
                "total_timesteps": index * 2,
                "policy_sha256": str(index) * 64,
                "numerics": {
                    "sample_count": 2,
                    "return_p99_abs": 0.1,
                    "value_target_normalized_rmse": 1.0,
                    "update_count": index,
                    "logger": {"approx_kl": 0.01, "clip_fraction": 0.1},
                    "all_finite": True,
                },
            }
            for index in range(1, 4)
        ]
        return FakeModel(), {
            "model": model_path.name,
            "model_sha256": "a" * 64,
            "initial_policy_sha256": "0" * 64,
            "trained_policy_sha256": "3" * 64,
            "total_timesteps_requested": 6,
            "total_timesteps_actual": 6,
            "elapsed_seconds": 0.3,
            "python_version": "test",
            "torch_version": "test",
            "cuda_device_name": "test",
            "stable_baselines3_version": "2.0.0",
            "gymnasium_version": "0.28.1",
            "numerics": history[-1]["numerics"],
            "update_history": history,
            "stability": _stability_numeric_summary(history, pilot),
        }

    config = {
        "environment": "simpleopc-recipe-local-epe-global-frag-v2",
        "data": {"openilt_dir": "fake"},
        "openilt": {"commit": "fixed"},
        "recipe_v2": {
            "epe_normal_offsets_nm": [-10, 0, 10],
            "epe_probe_distance_nm": 16,
            "observation": {
                "patch_size": 128,
                "report_examples": {
                    "example_count_per_layout": 2,
                    "selection_policy": "geometry-diverse-by-point-id-v1",
                },
            },
            "reward": {
                "weights": {"l2": 1, "epe": 100, "pvb": 1},
                "scale": 0.00001,
            },
        },
        "training": {
            "enabled": False,
            "blocked_by": ["three_update_stability_pilot_not_completed"],
            "smoke": {
                "enabled": True,
                "layout_parent": "fake",
                "protocols": [EPE_DENSE_PROTOCOL, EPE_TERMINAL_PROTOCOL],
                "seed": 0,
                "total_timesteps": 2,
                "n_steps": 2,
                "batch_size": 2,
            },
            "pilot": {
                "enabled": True,
                "layout_parent": "fake",
                "protocols": [EPE_DENSE_PROTOCOL, EPE_TERMINAL_PROTOCOL],
                "seed": 0,
                "rollout_count": 3,
                "consecutive_failure_updates": 3,
                "max_value_target_normalized_rmse": 10.0,
                "max_return_p99_abs": 10.0,
                "max_approx_kl": 0.1,
                "max_clip_fraction": 0.8,
            },
        },
    }
    monkeypatch.setattr(
        runner_module, "_build_v2_openilt_solver_and_evaluator", build_fake
    )
    monkeypatch.setattr(runner_module, "_train_sb3_pilot_model", train_fake)
    monkeypatch.setattr(runner_module, "_validate_openilt", lambda *_args: "fixed")

    result = runner_module.run_v2_ppo_pilot(
        config,
        artifact_root=tmp_path,
        protocol_alias="terminal",
        layout_parent="fake",
    )

    assert result["status"] == "diagnostic_only"
    assert result["rollout_count"] == 3
    assert result["training"]["total_timesteps_actual"] == 6
    assert len(result["training"]["update_history"]) == 3
    assert result["stability_pass"] is True
    assert result["pass"] is True
    assert result["accepted"] is False
    assert result["long_training_enabled"] is False
    assert result["deterministic_final_replay"]["final_replay_equal"] is True
    assert (tmp_path / "ppo-input-examples" / "manifest.json").is_file()


def _small_train_test_config():
    """返回适配两点 Fake episode 的共享 terminal 小训练配置。"""
    return {
        "environment": "simpleopc-recipe-local-epe-global-frag-v2",
        "data": {
            "openilt_dir": "fake",
            "train_parents": ["layout-a", "layout-b"],
            "validation_parents": [],
            "test_parents": [],
        },
        "openilt": {"commit": "fixed"},
        "recipe_v2": {
            "epe_normal_offsets_nm": [-20, -10, 0, 10, 20],
            "epe_probe_distance_nm": 24,
            "observation": {
                "patch_size": 128,
                "report_examples": {
                    "enabled": True,
                    "example_count_per_layout": 2,
                    "selection_policy": "geometry-diverse-by-point-id-v1",
                },
            },
            "reward": {
                "weights": {"l2": 1, "epe": 100, "pvb": 1},
                "scale": 0.00001,
            },
        },
        "training": {
            "enabled": False,
            "small_train": {
                "enabled": True,
                "protocols": [EPE_TERMINAL_PROTOCOL],
                "layout_parents": ["layout-a", "layout-b"],
                "seed": 0,
                "episodes_per_layout_per_update": 2,
                "update_count": 5,
                "episodes_per_layout": 10,
                "n_steps_per_env": 4,
                "total_timesteps": 40,
                "batch_size": 4,
                "n_epochs": 1,
                "learning_rate": 0.0003,
                "gamma": 1.0,
                "gae_lambda": 1.0,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "reward_scale": 0.00001,
                "random_recipe_count_per_layout": 10,
                "consecutive_failure_updates": 3,
                "max_post_update_value_target_normalized_rmse": 10.0,
                "max_return_p99_abs": 10.0,
                "max_approx_kl": 0.1,
                "max_clip_fraction": 0.8,
            },
        },
    }


def test_v2_small_train_contract_balances_two_layouts_and_rejects_drift():
    """共享 buffer 必须每轮等量包含两张图，且预算漂移立即失败。"""
    small_train = _small_train_test_config()["training"]["small_train"]

    contract = _small_train_contract(small_train, [2, 2])

    assert contract == {
        "episode_horizon": 2,
        "n_envs": 2,
        "n_steps_per_env": 4,
        "rollout_buffer_size": 8,
        "update_count": 5,
        "episodes_per_layout_per_update": 2,
        "episodes_per_layout": 10,
        "total_episodes": 20,
        "total_timesteps": 40,
        "batch_size": 4,
    }
    with pytest.raises(ValueError, match="相同且为正"):
        _small_train_contract(small_train, [2, 3])
    changed = dict(small_train, total_timesteps=41)
    with pytest.raises(ValueError, match="total_timesteps"):
        _small_train_contract(changed, [2, 2])


def test_v2_small_train_gate_uses_post_update_critic_for_three_step_stop():
    """小训练只按更新后 Critic 连续三次超限熔断，间断超限不能误停。"""
    small_train = _small_train_test_config()["training"]["small_train"]

    def history(rmses):
        return [
            {
                "numerics": {
                    "all_finite": True,
                    "return_p99_abs": 0.2,
                    "post_update_value_target": {"normalized_rmse": rmse},
                    "logger": {"approx_kl": 0.01, "clip_fraction": 0.0},
                }
            }
            for rmse in rmses
        ]

    intermittent = _small_train_gate(history([11.0, 9.0, 11.0]), small_train)
    assert intermittent["should_stop"] is False
    assert intermittent["max_consecutive_breaches"][
        "post_update_value_target_normalized_rmse"
    ] == 1

    consecutive = _small_train_gate(history([11.0, 12.0, 13.0]), small_train)
    assert consecutive["should_stop"] is True
    assert consecutive["failure_reasons"] == [
        "post-update-value-target-normalized-rmse-high-for-three-updates"
    ]


def test_v2_small_train_numerics_separates_pre_post_critic_and_layouts():
    """共享诊断必须恢复 SB3 更新后的 env-major buffer 并区分两张版图。"""
    torch = pytest.importorskip("torch")

    class FakePolicy:
        """返回固定更新后 value 和三类动作概率。"""

        def predict_values(self, observations):
            assert set(observations) == {"image", "vector"}
            return torch.tensor([0.1, 0.3, 0.2, 0.4], dtype=torch.float32)

        def get_distribution(self, _observations):
            probabilities = torch.tensor(
                [
                    [0.7, 0.2, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.2, 0.6],
                    [0.2, 0.6, 0.2],
                ],
                dtype=torch.float32,
            )
            return SimpleNamespace(
                distribution=SimpleNamespace(probs=probabilities)
            )

    model = SimpleNamespace(
        rollout_buffer=SimpleNamespace(
            buffer_size=2,
            n_envs=2,
            # SB3 swap_and_flatten 后按 env0 的全部 step、env1 的全部 step 排列。
            returns=np.asarray([[0.1], [0.3], [0.2], [0.4]], dtype=np.float32),
            values=np.zeros((4, 1), dtype=np.float32),
            rewards=np.asarray([[0.0], [0.3], [0.0], [0.4]], dtype=np.float32),
            advantages=np.asarray(
                [[-0.2], [0.1], [-0.1], [0.2]], dtype=np.float32
            ),
            actions=np.asarray([[0], [2], [1], [1]], dtype=np.int64),
            observations={
                "image": np.zeros((4, 5, 4, 4), dtype=np.float32),
                "vector": np.zeros((4, 12), dtype=np.float32),
            },
        ),
        policy=FakePolicy(),
        device="cpu",
        logger=SimpleNamespace(name_to_value={
            "train/approx_kl": 0.01,
            "train/clip_fraction": 0.0,
        }),
        _n_updates=1,
    )

    result = _shared_training_numerics(model, ["layout-a", "layout-b"])

    assert result["sample_count"] == 4
    assert result["pre_update_value_target"]["rmse"] > 0.0
    assert result["post_update_value_target"]["rmse"] < 1e-6
    assert result["layout_summaries"]["layout-a"]["sample_count"] == 2
    assert result["layout_summaries"]["layout-b"]["sample_count"] == 2
    assert result["action_distribution"]["sampled_action_class_counts"] == {
        "0": 1, "1": 2, "2": 1
    }
    assert result["all_finite"] is True


def test_v2_small_train_rollout_shape_helpers_cover_pre_and_post_update_forms():
    """形状适配器对原始 step×env 与 SB3 env-major 展平结果必须等价。"""
    step_env = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    env_major = np.asarray([[1.0], [3.0], [2.0], [4.0]], dtype=np.float32)

    restored_before = _rollout_scalar_matrix(
        step_env, n_steps=2, n_envs=2, field_name="returns"
    )
    restored_after = _rollout_scalar_matrix(
        env_major, n_steps=2, n_envs=2, field_name="returns"
    )

    assert restored_before.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert restored_after.tolist() == restored_before.tolist()

    observation = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    flattened = _rollout_observation_env_major(
        observation, n_steps=2, n_envs=2, field_name="vector"
    )
    assert flattened.shape == (4, 3)
    assert flattened.tolist() == observation.swapaxes(0, 1).reshape(4, 3).tolist()
    assert _rollout_observation_env_major(
        flattened, n_steps=2, n_envs=2, field_name="vector"
    ).tolist() == flattened.tolist()


def test_v2_small_train_random_baseline_is_fixed_budget_and_not_selection():
    """随机对照必须完整求解固定数量 Recipe，且明确不参与训练或选模。"""
    _, episode = _episode(
        EPE_TERMINAL_PROTOCOL,
        patch_size=128,
        action_offsets_nm=(-10, 0, 10),
    )

    result = _random_terminal_recipe_baseline(episode, seed=17, sample_count=3)

    assert result["sample_count"] == 3
    assert len(result["samples"]) == 3
    assert result["selection_used_for_training_or_checkpoint"] is False
    assert result["solver_call_counts"] == {
        "baseline_solver": 1,
        "candidate_solver": 3,
        "final_replay_solver": 0,
    }
    assert all(len(item["action_map_sha256"]) == 64 for item in result["samples"])


def test_v2_small_train_orchestration_uses_one_shared_model_and_two_layouts(
    monkeypatch, tmp_path
):
    """编排必须只训练一个共享模型，同时保留两图评价、随机对照和输入样例。"""
    class FakeModel:
        """为两张图的确定性评价统一输出零偏移动作。"""

        def predict(self, _observation, deterministic=True):
            assert deterministic is True
            return np.asarray(1), None

    build_calls = []
    train_calls = []

    def build_fake(_config, layout_parent, **_kwargs):
        build_calls.append(layout_parent)
        _, episode = _episode(
            EPE_TERMINAL_PROTOCOL,
            patch_size=128,
            shuffle_points=False,
            action_offsets_nm=(-10, 0, 10),
        )
        return episode, (f"{layout_parent}-golden-contract",)

    def train_fake(
        envs,
        layout_parents,
        small_train,
        contract,
        _model_base,
        _checkpoint_root,
        evaluation_callback,
    ):
        train_calls.append((len(envs), tuple(layout_parents), dict(contract)))
        model = FakeModel()
        initial = dict(evaluation_callback(model, 0))
        history = [
            {
                "update_index": update_index,
                "deterministic_evaluation": dict(
                    evaluation_callback(model, update_index)
                ),
            }
            for update_index in range(1, int(small_train["update_count"]) + 1)
        ]
        return model, {
            "model": "model-shared-terminal-final.zip",
            "model_sha256": "a" * 64,
            "initial_policy_sha256": "b" * 64,
            "trained_policy_sha256": "c" * 64,
            "total_timesteps_requested": int(contract["total_timesteps"]),
            "total_timesteps_actual": int(contract["total_timesteps"]),
            "completed_update_count": int(contract["update_count"]),
            "completed_episodes_per_layout": int(contract["episodes_per_layout"]),
            "training_completed": True,
            "stopped_early": False,
            "stop_decision": {"failure_reasons": [], "should_stop": False},
            "initial_evaluation": initial,
            "update_history": history,
            "elapsed_seconds": 0.5,
            "python_version": "test",
            "torch_version": "test",
            "cuda_device_name": "test",
            "stable_baselines3_version": "2.0.0",
            "gymnasium_version": "0.28.1",
        }

    monkeypatch.setattr(small_train_module, "_build_episode", build_fake)
    monkeypatch.setattr(
        small_train_module,
        "_build_ppo_env",
        lambda episode: SimpleNamespace(episode=episode),
    )
    monkeypatch.setattr(
        small_train_module, "_train_shared_terminal_model", train_fake
    )
    monkeypatch.setattr(
        small_train_module, "_validate_openilt", lambda *_args: "fixed"
    )

    result = small_train_module.run_v2_ppo_small_train(
        _small_train_test_config(),
        artifact_root=tmp_path,
        protocol_alias="terminal",
        layout_parents=["layout-a", "layout-b"],
    )

    assert len(train_calls) == 1
    assert train_calls[0][0:2] == (2, ("layout-a", "layout-b"))
    assert build_calls == [
        "layout-a", "layout-b",
        "layout-a", "layout-b",
        "layout-a", "layout-b",
    ]
    assert result["shared_model"] is True
    assert result["actor_layout_id_included"] is False
    assert result["balanced_layout_sampling"] is True
    assert result["layout_parents"] == ["layout-a", "layout-b"]
    assert set(result["ppo_input_examples"]) == {"layout-a", "layout-b"}
    assert all(
        value["saved_example_count"] == 2
        for value in result["ppo_input_examples"].values()
    )
    assert all(
        value["sample_count"] == 10
        for value in result["random_baselines"].values()
    )
    assert set(result["zero_offset_baselines"]) == {"layout-a", "layout-b"}
    assert set(result["untrained_deterministic_evaluation"]) == {
        "layout-a", "layout-b"
    }
    assert result["all_final_replays_equal"] is True
    assert result["episode_contract_equal"] is True
    assert result["execution_pass"] is True
    assert result["quality_accepted"] is False
    assert result["accepted"] is False
    assert result["long_training_enabled"] is False
