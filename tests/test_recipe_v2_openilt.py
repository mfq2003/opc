"""本模块验证 Recipe PPO v2 OpenILT 预检的 CPU 编排与证据口径。

测试使用内存 solver/evaluator 替身，不导入 CUDA 或 OpenILT；输出覆盖 probe 全点统计、两点真实调用
编排、Golden 身份字段、diagnostic-only 状态和训练禁用门禁，不能替代云端真实光刻运行。
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import opc_agent.recipe_v2_openilt as module
from opc_agent.recipe_v2_contract import (
    EPE_DENSE_PROTOCOL,
    EPE_TERMINAL_PROTOCOL,
    EPEControlPoint,
    FragmentParameters,
    GoldenEvaluation,
    GoldenMetrics,
    V2SolverResult,
    array_sha256,
)


def _point(point_id: str, base_x: int) -> EPEControlPoint:
    """建立远离边界和角点的水平上边控制点。"""
    return EPEControlPoint(
        point_id=point_id,
        polygon_index=0,
        source_edge_index=0,
        segment_index=base_x,
        base_xy=(base_x, 100),
        segment_start_xy=(base_x - 20, 100),
        segment_end_xy=(base_x + 20, 100),
        normal_xy=(0, -1),
    )


class _FakeSolver:
    """返回随完整 Recipe 确定变化的栅格，并记录预检 solver 调用。"""

    calls = []
    active_point_ids = ("p0", "p1")

    def __init__(self, **_kwargs):
        type(self).calls = []
        self.openilt_dir = Path("fake-openilt")
        self.threshold = 0.5
        self.nm_per_coordinate = 1.0
        self.target_image = np.zeros((320, 280), dtype=np.float32)
        self.target_image[100:241, 20:261] = 1
        self.epe_points = (_point("p0", 80), _point("p1", 160), _point("p2", 200))
        self.layout_sha256 = "a" * 64
        self.revision = "fixed"
        self.source_sha256 = {
            "pyilt/simpleopc.py": "1" * 64,
            "pyilt/evaluation.py": "2" * 64,
            "utils/polygon.py": "3" * 64,
        }
        self.fragmentation_sha256 = "b" * 64

    def solve(self, recipe):
        normalized = dict(sorted((key, float(value)) for key, value in recipe.items()))
        type(self).calls.append(normalized)
        marker = int(sum((index + 1) * value for index, value in enumerate(normalized.values())))
        mask = self.target_image.copy()
        mask[100, 100 + marker // 10] = 1 - mask[100, 100 + marker // 10]
        return V2SolverResult(
            mask_image=mask,
            printed_nominal=mask.copy(),
            printed_max=mask.copy(),
            printed_min=mask.copy(),
            recipe_epe_signs=tuple(
                (point.point_id, 1.0 if point.point_id in self.active_point_ids else 0.0)
                for point in self.epe_points
            ),
            mask_sha256=array_sha256(mask),
            internal_trace=({
                "inner_step": 0,
                "active_point_ids": self.active_point_ids,
                "control_conflicts": (),
            },),
        )

    def solve_diagnostic(self, recipe):
        """CPU 替身无控制冲突，诊断路径与严格路径返回相同结果。"""
        return self.solve(recipe)


class _FakeEvaluator:
    """提供固定 Golden 身份，并以 nominal 与 target 差异数评分。"""

    evaluator_version = "fake-golden"
    epe_constraint_coordinate = 15
    evaluator_source_sha256 = "c" * 64
    sampling_state_sha256 = "d" * 64
    coordinate_system_sha256 = "e" * 64
    evaluator_contract_sha256 = "f" * 64

    def __init__(self, solver, _weights):
        self.solver = solver

    def evaluate(self, result):
        l2 = float(np.count_nonzero(result.printed_nominal != self.solver.target_image))
        metrics = GoldenMetrics(l2=l2, epe=0, pvb=0)
        return GoldenEvaluation(
            metrics=metrics,
            raw_weighted_loss=l2,
            evaluator_version=self.evaluator_version,
            evaluator_source_sha256=self.evaluator_source_sha256,
            evaluator_contract_sha256=self.evaluator_contract_sha256,
        )


class _ConflictFakeSolver(_FakeSolver):
    """在每次 diagnostic 调用中注入一个显式双侧冲突记录。"""

    def solve_diagnostic(self, recipe):
        base = self.solve(recipe)
        return V2SolverResult(
            mask_image=base.mask_image,
            printed_nominal=base.printed_nominal,
            printed_max=base.printed_max,
            printed_min=base.printed_min,
            recipe_epe_signs=base.recipe_epe_signs,
            mask_sha256=base.mask_sha256,
            internal_trace=({
                "inner_step": 0,
                "active_point_ids": self.active_point_ids,
                "control_conflicts": ({
                    "point_id": "p2",
                    "normal_offset_nm": 0.0,
                    "inner_xy": [200, 116],
                    "outer_xy": [200, 84],
                    "underprint": True,
                    "overprint": True,
                },),
            },),
        )


def _config():
    """返回包含九动作、两点 sensitivity 和多 probe 扫描的最小配置。"""
    return {
        "status": "protocol_preflight_only",
        "environment": "simpleopc-recipe-local-epe-global-frag-v2",
        "data": {"openilt_dir": "third_party/OpenILT", "iccad13_dir": "benchmark"},
        "openilt": {"commit": "fixed"},
        "golden": {
            "evaluator": "openilt_evaluation_epecheck",
            "constraint_coordinate": 15,
            "source_sha256": "c" * 64,
            "sampling_state_sha256": None,
            "coordinate_system_sha256": None,
            "evaluator_contract_sha256": None,
        },
        "preflight": {
            "probe_distance_candidates_nm": [16, 56],
            "sensitivity_point_limit": 2,
            "sensitivity_actions_nm": [-10, 10],
            "control_conflict_policy": "both-sides-conflict-stay-v1",
        },
        "recipe_v2": {
            "nm_per_coordinate": 1.0,
            "control_conflict_policy": "both-sides-conflict-stay-v1",
            "solver": {
                "simulator": "simple",
                "lithography_config": "config/lithosimple.txt",
                "image_size": [2048, 2048],
                "inner_step_sizes_nm": [8, 8, 8, 8, 4, 4, 4, 4],
                "mask_displacement_limit_nm": 24,
                "threshold": 0.5,
            },
            "fragment_parameters_nm": {"corner": 16, "uniform": 32},
            "geometry_adapter": {
                "version": "openilt-dissect-parent-edge-adapter-v2",
                "raster_mapping_version": "db-coordinate-equals-raster-pixel-v1",
                "raster_scale": 1.0,
                "raster_offset_xy": [0, 0],
                "normal_probe_semantics_version": "target-two-sided-axis-probe-v1",
                "normal_probe_coordinate": 2,
                "minimum_fragment_rule_version": "min-corner-uniform-coordinate-v1",
            },
            "epe_normal_offsets_nm": [-40, -30, -20, -10, 0, 10, 20, 30, 40],
            "epe_probe_distance_nm": 16,
            "reward": {"weights": {"l2": 1.0, "epe": 100.0, "pvb": 1.0}},
        },
    }


def test_preflight_scans_geometry_and_keeps_training_disabled(monkeypatch):
    """CPU 编排必须保存全点合法率、两点四次响应及 post-run 干净证据。"""
    monkeypatch.setattr(module, "OpenILTV2Solver", _FakeSolver)
    monkeypatch.setattr(module, "OpenILTGoldenEvaluator", _FakeEvaluator)
    monkeypatch.setattr(module, "_validate_openilt", lambda *_args: "fixed")
    result = module.run_v2_openilt_preflight(_config())

    assert result["status"] == "diagnostic_only"
    assert result["training_enabled"] is False
    assert result["solver_calls"] == 5
    assert len(_FakeSolver.calls) == 5
    assert result["baseline"]["active_control_point_count"] == 2
    assert result["sensitivity_summary"]["selection_policy"] == (
        "baseline-any-step-active-nonconflict-edge-stratified-v2"
    )
    assert result["sensitivity_summary"]["selected_point_ids"] == ["p0", "p1"]
    assert result["sensitivity_summary"]["selected_point_count"] == 2
    assert result["sensitivity_summary"]["evaluated_result_count"] == 4
    assert result["sensitivity_summary"]["pass"] is True
    assert result["sensitivity_summary"]["responsive_point_count"] == 2
    assert len(result["sensitivity_summary"]["point_response_summary"]) == 2
    assert result["post_run_openilt_tracked_diff_clean"] is True
    assert result["golden"]["configured_identity_verified_fields"] == (
        "constraint_coordinate",
        "source_sha256",
    )
    assert result["control_conflict_summary"]["occurrence_count"] == 0
    assert result["control_conflict_summary"]["resolved_by_frozen_policy"] is False
    assert result["control_conflict_summary"]["training_compatible"] is True
    assert len(result["geometry_scan"]) == 2
    assert result["geometry_scan"][0]["all_actions_valid_for_all_points"] is False
    assert result["geometry_scan"][1]["all_actions_valid_for_all_points"] is True


def test_preflight_records_conflicts_as_resolved_by_frozen_stay_policy(monkeypatch):
    """正式 stay 协议必须保留冲突证据，但不再把方向歧义伪装成运行时崩溃。"""
    monkeypatch.setattr(module, "OpenILTV2Solver", _ConflictFakeSolver)
    monkeypatch.setattr(module, "OpenILTGoldenEvaluator", _FakeEvaluator)
    monkeypatch.setattr(module, "_validate_openilt", lambda *_args: "fixed")
    result = module.run_v2_openilt_preflight(_config())

    summary = result["control_conflict_summary"]
    assert result["training_enabled"] is False
    assert result["sensitivity_summary"]["pass"] is True
    assert summary["occurrence_count"] == 5
    assert summary["unique_point_count"] == 1
    assert summary["point_ids"] == ["p2"]
    assert summary["resolved_by_frozen_policy"] is True
    assert summary["training_compatible"] is True
    assert len(summary["records"]) == 5


def test_formal_solver_and_diagnostic_alias_share_the_same_policy_path():
    """正式 solver 与 preflight 必须共用同一冻结路径，禁止训练/诊断语义漂移。"""
    solver = object.__new__(module.OpenILTV2Solver)
    calls = []
    sentinel = object()

    def fake_solve(recipe):
        calls.append(dict(recipe))
        return sentinel

    solver._solve = fake_solve

    assert solver.solve({"p0": 0.0}) is sentinel
    assert solver.solve_diagnostic({"p0": 10.0}) is sentinel
    assert calls == [{"p0": 0.0}, {"p0": 10.0}]


def test_preflight_skips_geometrically_legal_but_inactive_first_point(monkeypatch):
    """sensitivity 必须抽取基线实际参与控制的点，不能再直接取前两个合法点。"""

    class InactiveFirstFakeSolver(_FakeSolver):
        """把首个几何合法点标为 inactive，后两个点标为 active。"""

        active_point_ids = ("p1", "p2")

    monkeypatch.setattr(module, "OpenILTV2Solver", InactiveFirstFakeSolver)
    monkeypatch.setattr(module, "OpenILTGoldenEvaluator", _FakeEvaluator)
    monkeypatch.setattr(module, "_validate_openilt", lambda *_args: "fixed")
    result = module.run_v2_openilt_preflight(_config())

    summary = result["sensitivity_summary"]
    assert summary["baseline_active_point_count"] == 2
    assert summary["eligible_active_point_count"] == 2
    assert summary["selected_point_ids"] == ["p1", "p2"]
    assert {item["point_id"] for item in result["sensitivity"]} == {"p1", "p2"}


def test_sensitivity_selection_covers_distinct_edges_before_adjacent_segments():
    """分层抽样必须先选不同原始边，不能再取同一边的前两个 segment。"""
    points = (
        _point("edge0-segment0", 60),
        _point("edge0-segment1", 100),
        EPEControlPoint(
            point_id="edge1-segment0",
            polygon_index=0,
            source_edge_index=1,
            segment_index=0,
            base_xy=(140, 100),
            segment_start_xy=(120, 100),
            segment_end_xy=(160, 100),
            normal_xy=(0, -1),
        ),
    )

    selected = module._select_sensitivity_points(points, 2)

    assert [point.point_id for point in selected] == [
        "edge0-segment0",
        "edge1-segment0",
    ]


def test_response_summary_compares_actions_with_baseline(monkeypatch):
    """即使所有非零动作彼此别名，只要它们异于基线就必须判为有响应。"""

    class AliasedButResponsiveSolver(_FakeSolver):
        """让每个非零 Recipe 都生成同一个、但与基线不同的 mask。"""

        def solve(self, recipe):
            normalized = dict(sorted((key, float(value)) for key, value in recipe.items()))
            type(self).calls.append(normalized)
            mask = self.target_image.copy()
            if any(value != 0.0 for value in normalized.values()):
                mask[100, 100] = 1 - mask[100, 100]
            return V2SolverResult(
                mask_image=mask,
                printed_nominal=mask.copy(),
                printed_max=mask.copy(),
                printed_min=mask.copy(),
                recipe_epe_signs=tuple(
                    (point.point_id, 1.0 if point.point_id in self.active_point_ids else 0.0)
                    for point in self.epe_points
                ),
                mask_sha256=array_sha256(mask),
                internal_trace=({
                    "inner_step": 0,
                    "active_point_ids": self.active_point_ids,
                    "control_conflicts": (),
                },),
            )

    monkeypatch.setattr(module, "OpenILTV2Solver", AliasedButResponsiveSolver)
    monkeypatch.setattr(module, "OpenILTGoldenEvaluator", _FakeEvaluator)
    monkeypatch.setattr(module, "_validate_openilt", lambda *_args: "fixed")

    result = module.run_v2_openilt_preflight(_config())
    summaries = result["sensitivity_summary"]["point_response_summary"]

    assert result["sensitivity_summary"]["responsive_point_count"] == 2
    assert all(item["responsive"] for item in summaries)
    assert all(item["changed_action_count"] == 2 for item in summaries)
    assert all(item["distinct_response_count_including_baseline"] == 2 for item in summaries)
    assert all(
        sorted(response_class["offsets_nm"])
        in ([0.0], [-10.0, 10.0])
        for item in summaries
        for response_class in item["response_classes"]
    )


def test_episode_smoke_runs_two_dense_and_two_terminal_128_replays(
    monkeypatch, tmp_path
):
    """128-only smoke 必须完成四个 episode，并严格核对 final replay。"""

    class SmokeFakeSolver(_FakeSolver):
        """为 LocalEPEEpisode 补齐冻结 FRAG 协议字段。"""

        def __init__(self):
            super().__init__()
            self.fragment_parameters = FragmentParameters(16, 32)
            self.control_conflict_policy = "both-sides-conflict-stay-v1"

    class SmokeFakeEvaluator(_FakeEvaluator):
        """为 episode smoke 提供完整 Golden 身份字段。"""

        def __init__(self, solver):
            super().__init__(solver, {"l2": 1.0, "epe": 100.0, "pvb": 1.0})
            self.frozen_target_sha256 = array_sha256(
                np.asarray(solver.target_image >= solver.threshold, dtype=bool)
            )
            self.nm_per_coordinate = solver.nm_per_coordinate
            self.coordinate_system_sha256 = (
                module.identity_raster_coordinate_system_sha256(
                    self.frozen_target_sha256, self.nm_per_coordinate
                )
            )

    def build_fake(_config_value, _layout_parent):
        solver = SmokeFakeSolver()
        return solver, SmokeFakeEvaluator(solver), ("source_sha256",)

    config = _config()
    config["status"] = "protocol_preflight_and_episode_smoke_only"
    config["training"] = {"enabled": False}
    config["recipe_v2"]["epe_normal_offsets_nm"] = [-20, -10, 0, 10, 20]
    config["recipe_v2"]["epe_probe_distance_nm"] = 24
    config["recipe_v2"]["observation"] = {
        "patch_size": 128,
        "legacy_64_status": "excluded_insufficient_control_context",
        "report_examples": {
            "enabled": True,
            "example_count_per_layout": 2,
            "selection_policy": "geometry-diverse-by-point-id-v1",
        },
    }
    config["recipe_v2"]["reward"].update({
        "scale": 0.00001,
        "dense_protocol": EPE_DENSE_PROTOCOL,
        "terminal_protocol": EPE_TERMINAL_PROTOCOL,
    })
    config["episode_smoke"] = {"repeats_per_protocol": 2}
    monkeypatch.setattr(
        module, "_build_v2_openilt_solver_and_evaluator", build_fake
    )
    monkeypatch.setattr(module, "_validate_openilt", lambda *_args: "fixed")

    result = module.run_v2_openilt_episode_smoke(
        config, artifact_root=tmp_path / "smoke", layout_parent="fake"
    )

    assert result["status"] == "diagnostic_only"
    assert result["patch_size"] == 128
    assert len(result["variants"]) == 4
    assert [item["training_protocol"] for item in result["variants"]] == [
        EPE_DENSE_PROTOCOL,
        EPE_DENSE_PROTOCOL,
        EPE_TERMINAL_PROTOCOL,
        EPE_TERMINAL_PROTOCOL,
    ]
    assert result["solver_calls"] == 16
    assert result["cross_protocol_final_equal"] is True
    assert result["repeat_baseline_equal"] is True
    assert result["pass"] is True
    assert result["training_enabled"] is False
    examples = result["ppo_input_examples"]
    assert examples["saved_example_count"] == 2
    assert examples["selection_depends_on_reward_or_result"] is False
    assert examples["shared_by_all_smoke_variants"] is True
    manifest_path = tmp_path / "smoke" / examples["manifest"]
    assert manifest_path.is_file()
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        examples["manifest_sha256"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["channel_names"] == [
        "target",
        "baseline-mask",
        "printed-nominal",
        "point-marker",
        "segment-normal",
    ]
    for item in manifest["examples"]:
        png_path = manifest_path.parent / item["png"]
        npz_path = manifest_path.parent / item["npz"]
        assert png_path.is_file()
        assert npz_path.is_file()
        assert hashlib.sha256(png_path.read_bytes()).hexdigest() == item["png_sha256"]
        assert hashlib.sha256(npz_path.read_bytes()).hexdigest() == item["npz_sha256"]
        with np.load(npz_path) as payload:
            assert payload["image"].shape == (5, 128, 128)
            assert payload["vector"].shape == (12,)
    assert all(item["observation_valid"] for item in result["variants"])
    assert all(item["final_replay_equal"] for item in result["variants"])
    assert all(item["reward_telescoping_error"] <= 1e-6 for item in result["variants"])

    export = module.run_v2_openilt_input_examples(
        config, artifact_root=tmp_path / "examples", layout_parent="fake"
    )
    assert export["status"] == "diagnostic_only"
    assert export["solver_calls"] == 1
    assert export["point_count"] == 3
    assert export["ppo_input_examples"]["saved_example_count"] == 2
    assert (tmp_path / "examples" / "ppo-input-examples" / "manifest.json").is_file()


@pytest.mark.parametrize("actions", ([0, 10], [-10, -10], [-10, 15]))
def test_preflight_rejects_invalid_explicit_sensitivity_actions(monkeypatch, actions):
    """显式 sensitivity 动作必须非零、无重复且属于当前 EPE 动作表。"""
    config = _config()
    config["preflight"]["sensitivity_actions_nm"] = actions
    monkeypatch.setattr(module, "OpenILTV2Solver", _FakeSolver)
    monkeypatch.setattr(module, "OpenILTGoldenEvaluator", _FakeEvaluator)
    monkeypatch.setattr(module, "_validate_openilt", lambda *_args: "fixed")

    with pytest.raises(ValueError, match="sensitivity_actions_nm"):
        module.run_v2_openilt_preflight(config)


def test_preflight_rejects_configured_golden_identity_mismatch(monkeypatch):
    """已写入配置的 Golden source/constraint 必须与真实 evaluator 严格一致。"""
    config = _config()
    config["golden"]["source_sha256"] = "0" * 64
    monkeypatch.setattr(module, "OpenILTV2Solver", _FakeSolver)
    monkeypatch.setattr(module, "OpenILTGoldenEvaluator", _FakeEvaluator)
    monkeypatch.setattr(module, "_validate_openilt", lambda *_args: "fixed")

    with pytest.raises(RuntimeError, match="Golden 身份不匹配"):
        module.run_v2_openilt_preflight(config)


def test_preflight_requires_matching_layout_specific_golden_contract(monkeypatch):
    """冻结逐版图 Golden contract 后，缺失或错配字段必须在 solver 构造期失败。"""
    config = _config()
    config["golden"]["layout_contracts"] = {
        "M1_test1": {
            "sampling_state_sha256": "d" * 64,
            "coordinate_system_sha256": "e" * 64,
            "evaluator_contract_sha256": "0" * 64,
        }
    }
    monkeypatch.setattr(module, "OpenILTV2Solver", _FakeSolver)
    monkeypatch.setattr(module, "OpenILTGoldenEvaluator", _FakeEvaluator)
    monkeypatch.setattr(module, "_validate_openilt", lambda *_args: "fixed")

    with pytest.raises(RuntimeError, match="Golden 逐版图身份不匹配"):
        module.run_v2_openilt_preflight(config, layout_parent="M1_test1")

    evaluator = _FakeEvaluator(_FakeSolver(), {"l2": 1, "epe": 100, "pvb": 1})
    with pytest.raises(RuntimeError, match="缺少版图 M1_test2"):
        module._validate_configured_golden_identity(
            config["golden"],
            evaluator,
            layout_parent="M1_test2",
            require_layout_contract=True,
        )


def _sign_solver() -> module.OpenILTV2Solver:
    """绕过 CUDA 构造，仅建立 `_recipe_signs` 所需的冻结字段。"""
    solver = module.OpenILTV2Solver.__new__(module.OpenILTV2Solver)
    solver.threshold = 0.5
    solver.nm_per_coordinate = 1.0
    solver.control_probe_distance_nm = 16.0
    solver._target_array = np.zeros((240, 240), dtype=np.float32)
    solver._target_array[100:201, 20:221] = 1
    solver.epe_points = (_point("p0", 120),)
    return solver


def test_recipe_signs_rejects_invalid_probe_instead_of_silent_noop():
    """超过 16nm 窗口的动作不能在真实 solver 内静默变成 sign=0。"""
    solver = _sign_solver()
    with pytest.raises(ValueError, match="一内一外合法 probe"):
        solver._recipe_signs({"p0": 40.0}, solver._target_array.copy())


def test_recipe_signs_rejects_ambiguous_two_sided_violation():
    """inner 缺印且 outer 多印时方向不唯一，必须失败而不是偏向外移。"""
    solver = _sign_solver()
    nominal = solver._target_array.copy()
    point = solver.epe_points[0]
    probe = module.build_probe_window(
        point, 0.0, 16.0, solver.nm_per_coordinate, solver._target_array
    )
    nominal[probe.inner_xy[1], probe.inner_xy[0]] = 0
    nominal[probe.outer_xy[1], probe.outer_xy[0]] = 1
    with pytest.raises(RuntimeError, match="同时违规"):
        solver._recipe_signs({"p0": 0.0}, nominal)

    signs, conflicts = solver._measure_recipe_signs({"p0": 0.0}, nominal)
    assert signs.tolist() == [0.0]
    assert [item["point_id"] for item in conflicts] == ["p0"]


def test_openilt_revision_and_tracked_diff_are_both_required(tmp_path, monkeypatch):
    """只读依赖门禁必须同时核对锁定提交和 tracked diff。"""
    openilt = tmp_path / "OpenILT"
    for relative in ("pyilt/simpleopc.py", "pyilt/evaluation.py", "utils/polygon.py"):
        path = openilt / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# upstream\n", encoding="utf-8")

    def clean_run(command, **_kwargs):
        if "rev-parse" in command:
            return SimpleNamespace(stdout="fixed\n", returncode=0)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(module.subprocess, "run", clean_run)
    assert module._validate_openilt(openilt, "fixed") == "fixed"

    def dirty_run(command, **_kwargs):
        if "rev-parse" in command:
            return SimpleNamespace(stdout="fixed\n", returncode=0)
        return SimpleNamespace(stdout="", returncode=1)

    monkeypatch.setattr(module.subprocess, "run", dirty_run)
    with pytest.raises(RuntimeError, match="tracked 文件存在修改"):
        module._validate_openilt(openilt, "fixed")


def test_preflight_rejects_unversioned_geometry_before_solver(monkeypatch):
    """YAML 版本与代码冻结协议不一致时不得继续构造真实 solver。"""
    config = _config()
    config["recipe_v2"]["geometry_adapter"]["version"] = "unknown"
    monkeypatch.setattr(
        module,
        "OpenILTV2Solver",
        lambda **_kwargs: pytest.fail("版本不一致时不应构造 solver"),
    )
    with pytest.raises(ValueError, match="必须为冻结值"):
        module.run_v2_openilt_preflight(config)
