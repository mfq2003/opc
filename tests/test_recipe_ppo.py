"""本模块验证点级 Recipe PPO 的 64×64 图像、九分类单步动作和 EPE/FRAG Recipe 导出。

输入为两个原始 target recipe 点和确定性内存 solver；输出为每点一次绝对位移、原始论文 reward、
episode 终止及模型哈希断言。测试不导入 CUDA、OpenILT、网络、Qwen、DQN 或决策树。
"""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium", reason="Recipe PPO 环境测试需要 Gymnasium")

from opc_agent.metrics import RECIPE_OPC_LOSS_VERSION
from opc_agent.recipe_ppo import (
    RECIPE_ENV_VERSION,
    RecipeEvaluation,
    RecipePoint,
    RecipePointPPOEnv,
)
from opc_agent.recipe_ppo_runner import build_recipe_payload
from opc_agent.simpleopc import SimpleOPCMetrics


class _RecipeSolver:
    """以 recipe 位移绝对值确定性改善指标，并记录完整 recipe 调用。"""

    revision = "fixed-commit"
    layout_sha256 = "a" * 64
    image_size = (128, 128)
    recipe_points = (
        RecipePoint("p0:e0:epe0", "EPE", 0, 0, 0, 64, 64, 1, 0, 0, 1, -40, 40),
        RecipePoint("p0:e0:frag1", "FRAG", 0, 0, 1, 80, 64, 1, 0, 0, 1, -40, 40),
    )

    def __init__(self):
        self.calls = []

    def solve(self, recipe_offsets_nm):
        offsets = np.asarray(recipe_offsets_nm, dtype=np.float64)
        self.calls.append(offsets.copy())
        improvement = float(np.abs(offsets).sum())
        target = np.zeros((128, 128), dtype=np.uint8)
        target[48:96, 48:96] = 1
        mask = target.copy()
        printed = target.copy()
        return RecipeEvaluation(
            metrics=SimpleOPCMetrics(100.0 - improvement, 1.0, 20.0),
            recipe_epe_signs=np.asarray([1.0, 0.0], dtype=np.float32),
            mask_sha256=f"mask-{len(self.calls)}",
            target_image=target,
            mask_image=mask,
            printed_image=printed,
            internal_trace=({"inner_step": 0},),
        )

    def local_image_stack(self, point_index, recipe_offsets_nm, evaluation, patch_size):
        del recipe_offsets_nm
        channels = np.zeros((5, patch_size, patch_size), dtype=np.float32)
        channels[0:3, 16:48, 16:48] = 1.0
        marker = 3 if self.recipe_points[point_index].task_type == "EPE" else 4
        channels[marker, 31:34, 31:34] = 1.0
        return channels


def _env():
    """建立不打乱点顺序的两点最小环境。"""
    solver = _RecipeSolver()
    env = RecipePointPPOEnv(
        solver=solver,
        reward_weights={"l2": 1, "epe": 100, "pvb": 1},
        patch_size=64,
        shuffle_points=False,
    )
    return solver, env


def test_recipe_env_uses_one_direct_nine_class_action_per_point():
    """每个点只应决策一次，动作直接对应 -40..40nm，不能再累计四步。"""
    solver, env = _env()
    observation, info = env.reset(seed=3)
    assert env.action_space.n == 9
    assert env.episode_horizon == 2
    assert observation["image"].shape == (5, 64, 64)
    assert observation["vector"].shape == (14,)
    assert info["environment_version"] == RECIPE_ENV_VERSION
    assert info["epe_point_count"] == 1
    assert info["frag_point_count"] == 1

    _, reward1, terminated1, truncated1, info1 = env.step(0)
    _, reward2, terminated2, truncated2, info2 = env.step(8)
    assert terminated1 is False and terminated2 is True
    assert truncated1 is False and truncated2 is False
    assert np.array_equal(solver.calls[-2], np.asarray([-40.0, 0.0]))
    assert np.array_equal(solver.calls[-1], np.asarray([-40.0, 40.0]))
    assert info1["displacement_nm"] == -40.0
    assert info2["best_recipe_offsets_nm"] == [-40.0, 40.0]
    assert reward1 == pytest.approx(-(60.0 + 100.0 + 20.0))
    assert reward2 == pytest.approx(-(20.0 + 100.0 + 20.0))
    assert env.trajectory[0]["loss_version"] == RECIPE_OPC_LOSS_VERSION
    assert len(env.trajectory) == 3


def test_recipe_env_requires_both_epe_and_frag_and_full_action_range():
    """缺少 FRAG 或不能支持完整 ±40nm 的点不得进入论文主环境。"""
    solver = _RecipeSolver()
    solver.recipe_points = (solver.recipe_points[0],)
    with pytest.raises(ValueError, match="同时具有 EPE 与 FRAG"):
        RecipePointPPOEnv(solver, {"l2": 1, "epe": 100, "pvb": 1})

    solver = _RecipeSolver()
    restricted = solver.recipe_points[1]
    solver.recipe_points = (
        solver.recipe_points[0],
        RecipePoint(
            restricted.point_id,
            restricted.task_type,
            restricted.polygon_index,
            restricted.edge_index,
            restricted.anchor_index,
            restricted.base_x,
            restricted.base_y,
            restricted.tangent_x,
            restricted.tangent_y,
            restricted.normal_x,
            restricted.normal_y,
            -30,
            40,
        ),
    )
    with pytest.raises(ValueError, match="完整支持"):
        RecipePointPPOEnv(solver, {"l2": 1, "epe": 100, "pvb": 1})


def test_recipe_payload_contains_epe_and_frag_exact_actions(tmp_path: Path):
    """PPO Recipe 应同时保留 EPE/FRAG 点、精确九档动作和共享模型哈希。"""
    _, env = _env()
    _, reset_info = env.reset(seed=7)
    env.step(0)
    env.step(8)
    model_path = tmp_path / "shared.zip"
    model_path.write_bytes(b"shared-ppo")
    rollout = {
        "reset": reset_info,
        "trajectory": list(env.trajectory),
        "best_step": env.best_step,
        "best_metrics": env.best_evaluation.metrics.as_dict(),
        "best_recipe_offsets_nm": env.best_recipe_offsets_nm.tolist(),
        "best_internal_solver_trace": list(env.best_evaluation.internal_trace),
    }
    payload = build_recipe_payload(env, rollout, model_path, seed=7)
    assert payload["label_version"] == "ppo-recipe-point-v1"
    assert payload["loss_version"] == RECIPE_OPC_LOSS_VERSION
    assert payload["patch_shape"] == [5, 64, 64]
    assert payload["epe_point_count"] == 1
    assert payload["frag_point_count"] == 1
    assert [label["task_type"] for label in payload["labels"]] == ["EPE", "FRAG"]
    assert [label["displacement_class"] for label in payload["labels"]] == [0, 8]
    assert [label["ppo_displacement_nm"] for label in payload["labels"]] == [-40.0, 40.0]
    assert len(payload["model_sha256"]) == 64
