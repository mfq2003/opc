"""本模块验证统一 CLI 默认进入共享的 Recipe point CNN-PPO，而不读取旧候选索引。

输入为临时 ICCAD13 路径及替身 solver/PPO 模块；输出为 64×64 observation、单步九分类、
EPE/FRAG 点数、共享模型和 stage-result 断言。测试不调用 CUDA、OpenILT、SB3 或网络。
"""
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml

from opc_agent.workflow import train_oracle_stage


def test_paper_config_uses_recipe_point_single_step_and_64_patch():
    """正式配置应删除四步累计动作，并固定每点一次九分类和 64×64 局部图像。"""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "paper_repro.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["oracle"]["environment"] == "simpleopc-recipe-point-v1"
    assert config["oracle"]["reward_mode"] == "paper_raw"
    assert config["oracle"]["displacement_nm_limit"] == 40
    assert config["simpleopc"]["local_patch_size"] == 64
    assert config["simpleopc"]["displacement_classes_nm"] == [
        -40, -30, -20, -10, 0, 10, 20, 30, 40,
    ]
    assert "step_sizes_nm" not in config["simpleopc"]
    assert config["simpleopc"]["inner_step_sizes_nm"] == [8, 8, 8, 8, 4, 4, 4, 4]


def test_train_oracle_routes_to_shared_recipe_point_ppo(tmp_path: Path, monkeypatch):
    """冒烟应只用首个 train GLP 训练共享模型并导出同时含 EPE/FRAG 的 Recipe。"""
    calls = []
    fake_recipe = types.ModuleType("opc_agent.recipe_ppo")
    fake_recipe.RECIPE_ENV_VERSION = "simpleopc-recipe-point-v1"
    fake_recipe.RECIPE_OBSERVATION_VERSION = "local-raster-64-v1"
    fake_recipe.RECIPE_POINT_VERSION = "target-recipe-point-v1"

    class _Solver:
        revision = "fixed"
        layout_sha256 = "b" * 64
        recipe_points = (
            SimpleNamespace(task_type="EPE"),
            SimpleNamespace(task_type="FRAG"),
        )

        def __init__(self, **kwargs):
            calls.append(("solver", kwargs))

    class _Env:
        def __init__(self, **kwargs):
            self.solver = kwargs["solver"]
            self.recipe_points = self.solver.recipe_points
            self.episode_horizon = len(self.recipe_points)
            self.patch_size = kwargs["patch_size"]
            calls.append(("env", kwargs))

        def reset(self, seed=0):
            return {
                "image": SimpleNamespace(shape=(5, 64, 64)),
                "vector": SimpleNamespace(shape=(14,)),
            }, {
                "point_count": 2,
                "epe_point_count": 1,
                "frag_point_count": 1,
                "seed": seed,
            }

    fake_recipe.OpenILTRecipeAwareSolver = _Solver
    fake_recipe.RecipePointPPOEnv = _Env
    fake_runner = types.ModuleType("opc_agent.recipe_ppo_runner")
    fake_runner.PPO_RECIPE_LABEL_VERSION = "ppo-recipe-point-v1"
    fake_runner.run_default_recipe = lambda env, seed=0: {
        "loss_version": "paper-weighted-sum-raw-v1",
        "best_metrics": {"l2": 1, "epe": 1, "pvb": 1},
    }

    def _train(**kwargs):
        calls.append(("train", kwargs))
        output = Path(str(kwargs["output_path"]) + ".zip")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"model")
        return SimpleNamespace(), output

    fake_runner.train_shared_recipe_ppo = _train
    fake_runner.deterministic_recipe_rollout = lambda model, env, seed: {"seed": seed}
    fake_runner.build_recipe_payload = lambda env, rollout, model_path, seed: {
        "label_version": "ppo-recipe-point-v1",
        "labels": [{"task_type": "EPE"}, {"task_type": "FRAG"}],
    }
    monkeypatch.setitem(sys.modules, "opc_agent.recipe_ppo", fake_recipe)
    monkeypatch.setitem(sys.modules, "opc_agent.recipe_ppo_runner", fake_runner)

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
            "environment": "simpleopc-recipe-point-v1",
            "reward_mode": "paper_raw",
            "seeds": [0, 1, 2],
            "smoke_timesteps": 24,
            "total_timesteps": 120,
            "lithography_config": "config/lithosimple.txt",
            "simulator": "simple",
            "openilt_scale": 1,
            "reward_weights": {"l2": 1, "epe": 100, "pvb": 1},
            "displacement_nm_limit": 40,
            "learning_rate": 0.0003,
            "ppo_n_steps": 8,
            "ppo_batch_size": 4,
        },
        "simpleopc": {
            "image_size": [2048, 2048],
            "local_patch_size": 64,
            "base_fragment_length_nm": 96,
            "min_fragment_length_nm": 8,
            "epe_sample_distance_nm": 16,
            "displacement_classes_nm": [-40, -30, -20, -10, 0, 10, 20, 30, 40],
            "inner_step_sizes_nm": [8, 8, 8, 8, 4, 4, 4, 4],
            "mask_displacement_limit_nm": 24,
        },
    }
    train_oracle_stage(config, run_root, smoke=True)
    result = json.loads((run_root / "stage-result.json").read_text(encoding="utf-8"))
    progress = json.loads((run_root / "stage-progress.json").read_text(encoding="utf-8"))
    assert result["environment"] == "simpleopc-recipe-point-v1"
    assert result["loss_version"] == "paper-weighted-sum-raw-v1"
    assert result["action_space"] == "Discrete(9)"
    assert result["action_semantics"] == "one_absolute_nine_class_decision_per_recipe_point"
    assert result["patch_shape"] == [5, 64, 64]
    assert result["vector_shape"] == [14]
    assert result["openilt_mutation"] == "none"
    assert result["seeds"] == [0]
    assert result["shared_models"]
    assert [item["clip_id"] for item in result["clips"]] == ["M1_test1"]
    assert result["clips"][0]["epe_point_count"] == 1
    assert result["clips"][0]["frag_point_count"] == 1
    assert progress["progress_status"] == "complete"
    assert any(kind == "train" for kind, _ in calls)

    train_calls = sum(kind == "train" for kind, _ in calls)
    preflight_root = tmp_path / "preflight"
    preflight_root.mkdir()
    train_oracle_stage(config, preflight_root, preflight=True)
    preflight = json.loads((preflight_root / "stage-result.json").read_text(encoding="utf-8"))
    assert preflight["mode"] == "preflight"
    assert preflight["preflight"]["image_shape"] == [5, 64, 64]
    assert preflight["preflight"]["vector_shape"] == [14]
    assert preflight["preflight"]["training_started"] is False
    assert sum(kind == "train" for kind, _ in calls) == train_calls
