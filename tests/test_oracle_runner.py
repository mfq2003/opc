"""本模块验证 GPU Oracle 训练前的数据契约、九分类动作映射和论文奖励公式。

输入为临时 NPZ 候选掩模与内存评价器；输出为形状校验失败、单步环境奖励和动作信息断言。
关键依赖为 NumPy、pytest 与 Gymnasium；测试不启动 CUDA、OpenILT、PPO、网络或 API。
"""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

pytest.importorskip("gymnasium", reason="GPU Oracle 测试需要云端锁定的 Gymnasium 依赖")

import opc_agent.oracle_runner as oracle_runner
from opc_agent.oracle_runner import CandidatePointEnv, load_candidate_point_set, ppo_rollout_schedule


class _Evaluator:
    """返回固定指标，隔离环境逻辑与昂贵 OpenILT 后端。"""

    def evaluate(self, point_index: int, action_index: int):
        return 10.0 + point_index, 2.0, 3.0 + action_index


def _write_dataset(path: Path, action_count: int = 9) -> None:
    """写入两个点、四维状态和小尺寸候选掩模。"""
    np.savez_compressed(
        str(path),
        observations=np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
        target=np.zeros((6, 6), dtype=np.float32),
        candidate_masks=np.zeros((2, action_count, 6, 6), dtype=np.float32),
    )


def test_candidate_point_env_uses_weighted_reward_and_displacement(tmp_path: Path):
    """动作 8 应映射到 +40nm，奖励应严格使用 1/100/1 权重。"""
    path = tmp_path / "points.npz"
    _write_dataset(path)
    dataset = load_candidate_point_set(path)
    env = CandidatePointEnv(dataset, _Evaluator(), {"l2": 1, "epe": 100, "pvb": 1})
    _, reset_info = env.reset(seed=42)
    _, reward, terminated, truncated, info = env.step(8)
    expected = -((10 + reset_info["point_index"]) + 100 * 2 + (3 + 8))
    assert reward == expected
    assert terminated is True and truncated is False
    assert info["displacement_nm"] == 40


def test_candidate_point_set_requires_exactly_nine_masks_per_point(tmp_path: Path):
    """候选掩模不是九分类时必须在训练前失败。"""
    path = tmp_path / "bad-points.npz"
    _write_dataset(path, action_count=8)
    with pytest.raises(ValueError, match="candidate_masks"):
        load_candidate_point_set(path)
def test_compact_dataset_dispatches_recorded_v2_adapter(tmp_path: Path):
    """紧凑 NPZ 必须按自身 v2 标记生成九个互异且面积单调的边界条带动作。"""
    path = tmp_path / "compact-v2.npz"
    base = np.zeros((20, 20), dtype=np.float32)
    base[6:14, 6:14] = 1
    np.savez_compressed(
        str(path),
        observations=np.asarray([[1, 2, 3, 4]], dtype=np.float32),
        target=base,
        base_mask=base,
        point_geometry=np.asarray([[13, 10, 1, 0, 2]], dtype=np.int32),
        scale_nm_per_pixel=np.asarray(10.0),
        adapter_version=np.asarray("raster-boundary-strip-v2"),
    )
    dataset = load_candidate_point_set(path)
    candidates = [dataset.mask_for(0, index) for index in range(9)]
    assert dataset.adapter_version == "raster-boundary-strip-v2"
    assert len({candidate.tobytes() for candidate in candidates}) == 9
    assert [int(candidate.sum()) for candidate in candidates] == sorted(
        int(candidate.sum()) for candidate in candidates
    )


def test_compact_dataset_dispatches_recorded_v3_segment_adapter(tmp_path: Path):
    """v3 九列紧凑几何必须在 Oracle 读取后恢复为完整边段动作。"""
    path = tmp_path / "compact-v3.npz"
    base = np.zeros((64, 64), dtype=np.float32)
    base[20:44, 20:44] = 1
    np.savez_compressed(
        str(path),
        observations=np.asarray([[1, 2, 3, 4]], dtype=np.float32),
        target=base,
        base_mask=base,
        point_geometry=np.asarray([[43, 31, 1, 0, 3, 43, 20, 43, 43]], dtype=np.int32),
        scale_nm_per_pixel=np.asarray(10.0),
        adapter_version=np.asarray("raster-edge-segment-v3"),
    )
    dataset = load_candidate_point_set(path)
    candidates = [dataset.mask_for(0, index) for index in range(9)]
    assert dataset.adapter_version == "raster-edge-segment-v3"
    assert len({candidate.tobytes() for candidate in candidates}) == 9
    assert [int(candidate.sum()) for candidate in candidates] == sorted(
        int(candidate.sum()) for candidate in candidates
    )


def test_openilt_import_runs_from_openilt_root(tmp_path: Path, monkeypatch):
    """evaluation 的默认相对配置必须在 OpenILT 根目录下解析，并在初始化后恢复 cwd。"""
    dataset_path = tmp_path / "points.npz"
    _write_dataset(dataset_path)
    dataset = load_candidate_point_set(dataset_path)
    openilt_dir = tmp_path / "OpenILT"
    config_path = openilt_dir / "config" / "lithosimple.txt"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("placeholder", encoding="utf-8")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        float32=np.float32,
        as_tensor=lambda value, **kwargs: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        oracle_runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="fixed-commit\n"),
    )

    import_directories = []

    class _Metric:
        def __init__(self, *args, **kwargs):
            pass

    def fake_import(name: str):
        import_directories.append(Path.cwd())
        if name.startswith("pylitho."):
            return SimpleNamespace(LithoSim=lambda config: object())
        return SimpleNamespace(Basic=_Metric, EPEChecker=_Metric)

    monkeypatch.setattr(oracle_runner.importlib, "import_module", fake_import)
    original_cwd = Path.cwd()
    oracle_runner.OpenILTCandidateEvaluator(
        dataset=dataset,
        openilt_dir=openilt_dir,
        expected_commit="fixed-commit",
        lithography_config=Path("config/lithosimple.txt"),
        cache_path=tmp_path / "cache.json",
    )
    assert import_directories == [openilt_dir.resolve(), openilt_dir.resolve()]
    assert Path.cwd() == original_cwd

def test_ppo_rollout_schedule_preserves_requested_steps():
    """冒烟和正式配置都应选择可整除总步数的 rollout，避免 SB3 多跑。"""
    assert ppo_rollout_schedule(256) == (256, 64)
    assert ppo_rollout_schedule(10000) == (2000, 50)

