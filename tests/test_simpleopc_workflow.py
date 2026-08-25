"""本模块验证统一 CLI 在新配置下只走 GLP SimpleOPC 多步 PPO，而不读取旧候选索引。

输入为临时 ICCAD13 路径和替身 SimpleOPC/PPO 模块；输出为 smoke 父版图、种子、基线和 PPO
Recipe 的 stage-result 断言。关键依赖为 pytest；测试不调用 CUDA、OpenILT、Stable-Baselines3 或网络。
"""
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml

from opc_agent.workflow import train_oracle_stage


def test_paper_config_uses_published_range_and_exact_equal_grid_adapter():
    """正式配置应固定论文 ±40nm 范围，并让九个适配代表值无需近似量化即可到达。"""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "paper_repro.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["oracle"]["environment"] == "simpleopc-multistep-v3"
    assert config["oracle"]["displacement_nm_limit"] == 40
    assert config["simpleopc"]["displacement_classes_nm"] == [
        -40, -30, -20, -10, 0, 10, 20, 30, 40,
    ]
    assert config["simpleopc"]["step_sizes_nm"] == [10, 10, 10, 10]


def test_train_oracle_routes_to_simpleopc_multistep_without_candidate_data(tmp_path: Path, monkeypatch):
    """新环境冒烟只跑首个 train GLP，并明确声明决策树标签只能来自 PPO Recipe。"""
    calls = []
    fake_simpleopc = types.ModuleType("opc_agent.simpleopc")

    class _Backend:
        revision = "fixed"
        layout_sha256 = "b" * 64
        segments = (SimpleNamespace(segment_id="s0"),)

        def __init__(self, **kwargs):
            calls.append(("backend", kwargs))

    fake_simpleopc.OpenILTSimpleOPCBackend = _Backend
    fake_simpleopc.SimpleOPCMultiStepEnv = lambda **kwargs: SimpleNamespace(
        backend=kwargs["backend"], settings=kwargs
    )
    fake_runner = types.ModuleType("opc_agent.simpleopc_runner")
    fake_runner.run_simpleopc_heuristic = lambda env, seed=0: {
        "best_step": 1, "best_metrics": {"l2": 1, "epe": 1, "pvb": 1}
    }

    def _train(**kwargs):
        calls.append(("train", kwargs))
        output = Path(str(kwargs["output_path"]) + ".zip")
        recipe = output.with_suffix(".recipe.json")
        return output, recipe

    fake_runner.train_simpleopc_ppo = _train
    monkeypatch.setitem(sys.modules, "opc_agent.simpleopc", fake_simpleopc)
    monkeypatch.setitem(sys.modules, "opc_agent.simpleopc_runner", fake_runner)
    data_root = tmp_path / "ICCAD2013"
    data_root.mkdir()
    (data_root / "M1_test1.glp").write_text("layout", encoding="utf-8")
    run_root = tmp_path / "run"
    run_root.mkdir()
    config = {
        "data": {
            "openilt_dir": str(tmp_path / "OpenILT"),
            "iccad13_dir": str(data_root),
            "train_parents": ["M1_test1"],
            "validation_parents": ["M1_test7"],
            "test_parents": ["M1_test9"],
        },
        "openilt": {"commit": "fixed"},
        "oracle": {
            "environment": "simpleopc-multistep-v3",
            "seeds": [0, 1, 2],
            "smoke_timesteps": 24,
            "total_timesteps": 120,
            "lithography_config": "config/lithosimple.txt",
            "simulator": "simple",
            "openilt_scale": 1,
            "reward_weights": {"l2": 1, "epe": 100, "pvb": 1},
            "displacement_nm_limit": 40,
            "learning_rate": 0.0003,
        },
        "simpleopc": {
            "image_size": [2048, 2048],
            "len_corner_nm": 16,
            "len_uniform_nm": 32,
            "epe_sample_distance_nm": 16,
            "displacement_classes_nm": [-40, -30, -20, -10, 0, 10, 20, 30, 40],
            "step_sizes_nm": [10, 10, 10, 10],
        },
    }
    train_oracle_stage(config, run_root, smoke=True)
    result = json.loads((run_root / "stage-result.json").read_text(encoding="utf-8"))
    assert result["environment"] == "simpleopc-multistep-v3"
    assert result["loss_version"] == "paper-weighted-sum-initial-normalized-v1"
    assert result["episode_step_sizes_nm"] == [10.0, 10.0, 10.0, 10.0]
    assert result["displacement_classes_nm"] == [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    assert result["openilt_mutation"] == "none"
    assert result["seeds"] == [0]
    assert [item["clip_id"] for item in result["clips"]] == ["M1_test1"]
    assert result["decision_tree_labels"] == "ppo_recipe_only"
    assert any(kind == "train" for kind, _ in calls)
