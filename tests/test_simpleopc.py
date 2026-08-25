"""本模块验证 SimpleOPC 多步状态转移、批量边段动作、历史最优 Recipe 和启发式基线。

输入为两个合成边段和内存后端；输出为跨步累积位移、终止条件、九分类量化及来源哈希断言。
关键依赖为 NumPy、Gymnasium 与 pytest；测试不导入 CUDA、OpenILT、网络或真实 PPO。
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gymnasium", reason="SimpleOPC 环境测试需要云端锁定的 Gymnasium 依赖")

from opc_agent.simpleopc import (
    OpenILTSimpleOPCBackend,
    SIMPLEOPC_ENV_VERSION,
    SimpleOPCEvaluation,
    SimpleOPCMetrics,
    SimpleOPCMultiStepEnv,
    SimpleOPCSegment,
)
from opc_agent.simpleopc_runner import build_recipe_payload, run_simpleopc_heuristic
from opc_agent.metrics import SIMPLEOPC_LOSS_VERSION


class _Backend:
    """用位移绝对值改善合成指标，记录环境传入的完整 Recipe。"""

    revision = "fixed-commit"
    layout_sha256 = "a" * 64
    segments = (
        SimpleOPCSegment("p0:s0:f0", 0, 0, 0, 0, 10, 0, 0, 1, 10.0),
        SimpleOPCSegment("p0:s1:f1", 0, 1, 10, 0, 10, 10, -1, 0, 10.0),
    )

    def __init__(self):
        self.calls = []

    def reset(self):
        return self.evaluate(np.zeros(2, dtype=np.float64))

    def evaluate(self, displacements_nm):
        values = np.asarray(displacements_nm, dtype=np.float64)
        self.calls.append(values.copy())
        improvement = float(np.abs(values).sum())
        return SimpleOPCEvaluation(
            metrics=SimpleOPCMetrics(100.0 - improvement, 10.0, 50.0),
            epe_signs=np.asarray([1.0, -1.0], dtype=np.float32),
            mask_sha256=f"mask-{len(self.calls)}",
        )


def _env(step_sizes=(10, 10)):
    """建立使用论文权重的两段最小多步环境。"""
    backend = _Backend()
    env = SimpleOPCMultiStepEnv(
        backend,
        reward_weights={"l2": 1, "epe": 100, "pvb": 1},
        step_sizes_nm=step_sizes,
        displacement_limit_nm=40,
    )
    return backend, env


def test_multistep_env_persists_mask_recipe_and_batches_all_segments():
    """每轮动作应基于上一轮累积，且一次 evaluate 同时接收全部边段位移。"""
    backend, env = _env()
    observation, info = env.reset(seed=3)
    assert observation.shape == env.observation_space.shape
    assert info["environment_version"] == SIMPLEOPC_ENV_VERSION
    _, reward1, terminated1, truncated1, _ = env.step(np.asarray([2, 0]))
    _, reward2, terminated2, truncated2, info2 = env.step(np.asarray([2, 0]))
    assert reward1 < 0 and reward2 < 0
    assert terminated1 is False and terminated2 is True
    assert truncated1 is False and truncated2 is False
    assert np.array_equal(backend.calls[-2], np.asarray([10.0, -10.0]))
    assert np.array_equal(backend.calls[-1], np.asarray([20.0, -20.0]))
    assert env.best_step == 2
    assert info2["best_displacements_nm"] == [20.0, -20.0]
    assert env.trajectory[0]["loss_version"] == SIMPLEOPC_LOSS_VERSION
    assert env.trajectory[0]["raw_weighted_loss"] == pytest.approx(1150.0)
    assert env.trajectory[0]["loss"] == pytest.approx(1.0)
    assert env.trajectory[1]["raw_weighted_loss"] == pytest.approx(1130.0)
    assert env.trajectory[1]["loss"] == pytest.approx(1130.0 / 1150.0)
    assert env.trajectory[2]["raw_weighted_loss"] == pytest.approx(1110.0)
    assert env.trajectory[2]["loss"] == pytest.approx(1110.0 / 1150.0)


def test_multistep_env_rejects_offsets_outside_equal_ten_nm_grid():
    """8/4nm 旧调度会产生非九分类代表值，环境必须在仿真前拒绝。"""
    with pytest.raises(ValueError, match="10nm 位移代表值间隔"):
        _env(step_sizes=(8, 4))


def test_multistep_env_clips_cumulative_displacement_to_recipe_limit():
    """重复向外移动不得突破论文 Recipe 的 ±40nm 边界。"""
    backend, env = _env(step_sizes=(30, 30))
    env.reset()
    env.step(np.asarray([2, 2]))
    env.step(np.asarray([2, 2]))
    assert np.array_equal(backend.calls[-1], np.asarray([40.0, 40.0]))


def test_recipe_payload_is_from_model_and_keeps_exact_and_quantized_offsets(tmp_path: Path):
    """PPO Recipe 必须携带模型哈希，并同时保存精确累积位移与九分类标签。"""
    _, env = _env()
    _, reset_info = env.reset()
    env.step(np.asarray([2, 0]))
    env.step(np.asarray([2, 0]))
    model = tmp_path / "model.zip"
    model.write_bytes(b"ppo-model")
    rollout = {
        "reset": reset_info,
        "best_displacements_nm": env.best_displacements_nm.tolist(),
        "best_step": env.best_step,
        "best_metrics": env.best_evaluation.metrics.as_dict(),
        "trajectory": list(env.trajectory),
    }
    payload = build_recipe_payload(env, rollout, model, seed=7)
    assert payload["label_version"] == "ppo-simpleopc-multistep-v3"
    assert payload["loss_version"] == SIMPLEOPC_LOSS_VERSION
    assert payload["initial_weighted_loss"] == pytest.approx(1150.0)
    assert payload["loss_scale"] == pytest.approx(1150.0)
    assert payload["openilt_commit"] == "fixed-commit"
    assert payload["labels"][0]["ppo_displacement_nm"] == 20.0
    assert payload["labels"][0]["quantized_displacement_nm"] == 20.0
    assert payload["labels"][0]["quantization_error_nm"] == 0.0
    assert payload["displacement_classes_nm"] == [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    assert len(payload["model_sha256"]) == 64


def test_simpleopc_heuristic_follows_current_epe_directions():
    """启发式基线应把 +1/-1 EPE 建议映射为外移/内移动作并完整跑完 episode。"""
    _, env = _env()
    result = run_simpleopc_heuristic(env)
    assert result["best_step"] == 2
    assert result["best_displacements_nm"] == [20.0, -20.0]
    assert len(result["trajectory"]) == 3


def test_openilt_validation_rejects_tracked_local_changes(tmp_path: Path, monkeypatch):
    """固定提交正确但已跟踪文件被改动时，也不能生成不可复现实验。"""
    openilt = tmp_path / "OpenILT"
    (openilt / "pyilt").mkdir(parents=True)
    (openilt / "utils").mkdir()
    (openilt / "pyilt" / "simpleopc.py").write_text("# test\n", encoding="utf-8")
    (openilt / "utils" / "polygon.py").write_text("# test\n", encoding="utf-8")
    backend = OpenILTSimpleOPCBackend.__new__(OpenILTSimpleOPCBackend)
    backend.openilt_dir = openilt

    def fake_run(command, **kwargs):
        if "rev-parse" in command:
            return SimpleNamespace(stdout="fixed-commit\n", returncode=0)
        return SimpleNamespace(stdout="", returncode=1)

    monkeypatch.setattr("opc_agent.simpleopc.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="已跟踪文件存在本地修改"):
        backend._validate_openilt("fixed-commit")
