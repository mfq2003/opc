"""本模块运行 Recipe PPO v2 的单版图、单协议 CUDA 数值训练 smoke。

输入为已冻结 Golden contract 的 OpenILT 版图、128×128 五通道 observation、显式 smoke
超参数和 dense 或 terminal 协议；输出为独立 PPO 模型、训练数值诊断、确定性完整 Recipe
回放、final replay 一致性与 PPO 输入样例。dense/terminal 必须分开运行，结果始终标记为
``diagnostic_only``，不得用于 accepted、收敛或跨版图泛化结论。
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
import time
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np

from .recipe_v2 import LocalEPEEpisode
from .recipe_v2_contract import (
    ACTOR_GEOMETRY_FIELDS,
    EPE_DENSE_PROTOCOL,
    EPE_TERMINAL_PROTOCOL,
)
from .recipe_v2_openilt import (
    _build_v2_openilt_solver_and_evaluator,
    _sha256_json,
    _validate_openilt,
)
from .recipe_v2_visualization import (
    PPO_INPUT_SELECTION_POLICY,
    save_v2_ppo_input_examples,
)


PPO_SMOKE_VERSION = "recipe-v2-single-layout-single-rollout-ppo-smoke-v2"
PPO_PILOT_VERSION = "recipe-v2-single-layout-three-update-stability-pilot-v1"
RETURN_VARIANCE_FLOOR = 1e-12
PROTOCOL_ALIASES = {
    "dense": EPE_DENSE_PROTOCOL,
    "terminal": EPE_TERMINAL_PROTOCOL,
}


def _feature_extractor_class():
    """延迟构造 128 输入 CNN，避免非训练命令强制导入 Torch/SB3。"""
    import torch
    from torch import nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class LocalEPEFeatureExtractor(BaseFeaturesExtractor):
        """融合五通道局部图像和 12 维无绝对坐标几何向量。"""

        def __init__(self, observation_space, features_dim: int = 256):
            super().__init__(observation_space, features_dim)
            image_space = observation_space.spaces["image"]
            vector_space = observation_space.spaces["vector"]
            self.cnn = nn.Sequential(
                nn.Conv2d(int(image_space.shape[0]), 32, 5, 2, 2),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 2, 1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                sample = torch.zeros((1,) + image_space.shape, dtype=torch.float32)
                cnn_dim = int(self.cnn(sample).shape[1])
            self.vector = nn.Sequential(
                nn.Linear(int(np.prod(vector_space.shape)), 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(cnn_dim + 64, int(features_dim)),
                nn.ReLU(),
            )

        def forward(self, observations):
            image = self.cnn(observations["image"].float())
            vector = self.vector(observations["vector"].float())
            return self.fusion(torch.cat((image, vector), dim=1))

    return LocalEPEFeatureExtractor


def _state_dict_sha256(policy) -> str:
    """对策略参数名称、dtype、shape 与原始字节计算稳定 SHA256。"""
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _training_numerics(model) -> Dict[str, object]:
    """提取最后 rollout 的 return/value 和 SB3 更新日志。"""
    returns = np.asarray(model.rollout_buffer.returns, dtype=np.float64).reshape(-1)
    values = np.asarray(model.rollout_buffer.values, dtype=np.float64).reshape(-1)
    rewards = np.asarray(model.rollout_buffer.rewards, dtype=np.float64).reshape(-1)
    if not all(np.all(np.isfinite(item)) for item in (returns, values, rewards)):
        raise RuntimeError("PPO rollout 的 reward/return/value 包含非有限数值")
    logger = getattr(model.logger, "name_to_value", {})
    names = (
        "train/approx_kl",
        "train/clip_fraction",
        "train/entropy_loss",
        "train/explained_variance",
        "train/policy_gradient_loss",
        "train/value_loss",
    )
    metrics = {}
    explained_variance_defined = True
    return_variance = float(np.var(returns))
    for name in names:
        if name not in logger:
            continue
        metric_name = name.split("/", 1)[1]
        raw_value = float(np.asarray(logger[name]).item())
        if (
            metric_name == "explained_variance"
            and return_variance <= RETURN_VARIANCE_FLOOR
        ):
            metrics[metric_name] = None
            explained_variance_defined = False
            continue
        if np.isfinite(raw_value):
            metrics[metric_name] = raw_value
            continue
        raise RuntimeError(f"PPO 训练日志 {metric_name} 包含非有限数值")
    if "approx_kl" not in metrics or "clip_fraction" not in metrics:
        raise RuntimeError("PPO smoke 未产生 approx_kl/clip_fraction 更新日志")
    rmse = float(np.sqrt(np.mean(np.square(values - returns))))
    return_rms = float(np.sqrt(np.mean(np.square(returns))))
    return {
        "sample_count": int(returns.size),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "return_variance": return_variance,
        "return_std": float(np.sqrt(return_variance)),
        "return_p99_abs": float(np.quantile(np.abs(returns), 0.99)),
        "value_target_rmse": rmse,
        "value_target_normalized_rmse": rmse / max(return_rms, 1e-3),
        "explained_variance_defined": explained_variance_defined,
        "explained_variance_undefined_reason": (
            None
            if explained_variance_defined
            else "single-rollout-return-variance-is-zero-or-negligible"
        ),
        "update_count": int(getattr(model, "_n_updates", 0)),
        "logger": metrics,
        "all_finite": True,
    }


def _train_sb3_model(env, smoke: Mapping[str, object], model_base: Path):
    """用一个完整 rollout 运行一次普通 SB3 PPO 更新并保存独立模型。"""
    import gymnasium
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    if stable_baselines3.__version__ != "2.0.0" or gymnasium.__version__ != "0.28.1":
        raise RuntimeError(
            "v2 PPO 要求 stable-baselines3==2.0.0 与 gymnasium==0.28.1；"
            f"当前为 {stable_baselines3.__version__}/{gymnasium.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("v2 PPO smoke 要求 CUDA")
    seed = int(smoke["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    n_steps = int(smoke["n_steps"])
    if n_steps != env.episode.episode_horizon:
        raise ValueError("PPO smoke n_steps 必须恰好等于当前版图完整 episode horizon")
    total_timesteps = int(smoke["total_timesteps"])
    if total_timesteps != n_steps:
        raise ValueError("PPO smoke 只允许一个完整 rollout，不允许伪装成长训练")
    batch_size = int(smoke["batch_size"])
    if batch_size <= 1 or n_steps % batch_size != 0:
        raise ValueError("PPO smoke batch_size 必须大于 1 且整除 n_steps")
    vec_env = DummyVecEnv([lambda: env])
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        device="cuda",
        seed=seed,
        learning_rate=float(smoke["learning_rate"]),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=int(smoke["n_epochs"]),
        gamma=float(smoke["gamma"]),
        gae_lambda=float(smoke["gae_lambda"]),
        clip_range=float(smoke["clip_range"]),
        ent_coef=float(smoke["ent_coef"]),
        vf_coef=float(smoke["vf_coef"]),
        policy_kwargs={
            "features_extractor_class": _feature_extractor_class(),
            "features_extractor_kwargs": {"features_dim": 256},
            "normalize_images": False,
        },
        verbose=1,
    )
    initial_policy_sha256 = _state_dict_sha256(model.policy)
    started = time.monotonic()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.monotonic() - started
    numerics = _training_numerics(model)
    model_base.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_base))
    model_path = Path(str(model_base) + ".zip")
    return model, {
        "model": model_path.name,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "initial_policy_sha256": initial_policy_sha256,
        "trained_policy_sha256": _state_dict_sha256(model.policy),
        "total_timesteps_requested": total_timesteps,
        "total_timesteps_actual": int(model.num_timesteps),
        "elapsed_seconds": elapsed,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_version": gymnasium.__version__,
        "numerics": numerics,
    }


def _max_consecutive_true(values) -> int:
    """返回布尔序列中连续为真的最大长度。"""
    maximum = 0
    current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        maximum = max(maximum, current)
    return maximum


def _stability_numeric_summary(
    update_history, pilot: Mapping[str, object]
) -> Dict[str, object]:
    """按既定连续三次规则汇总多次 PPO 更新的数值稳定性。"""
    expected_updates = int(pilot["rollout_count"])
    if len(update_history) != expected_updates:
        raise ValueError("PPO stability pilot 的更新记录数量与 rollout_count 不一致")
    consecutive_limit = int(pilot["consecutive_failure_updates"])
    if consecutive_limit != 3 or expected_updates != 3:
        raise ValueError("PPO stability pilot 当前只允许恰好三次更新")
    rmse_limit = float(pilot["max_value_target_normalized_rmse"])
    kl_limit = float(pilot["max_approx_kl"])
    clip_limit = float(pilot["max_clip_fraction"])
    return_limit = float(pilot["max_return_p99_abs"])
    numerics = [item["numerics"] for item in update_history]
    rmse_streak = _max_consecutive_true(
        item["value_target_normalized_rmse"] > rmse_limit for item in numerics
    )
    kl_streak = _max_consecutive_true(
        item["logger"]["approx_kl"] > kl_limit for item in numerics
    )
    clip_streak = _max_consecutive_true(
        item["logger"]["clip_fraction"] > clip_limit for item in numerics
    )
    reasons = []
    if rmse_streak >= consecutive_limit:
        reasons.append("value-target-normalized-rmse-high-for-three-updates")
    if kl_streak >= consecutive_limit:
        reasons.append("approx-kl-high-for-three-updates")
    if clip_streak >= consecutive_limit:
        reasons.append("clip-fraction-high-for-three-updates")
    basic_finite_pass = all(
        item["all_finite"] and item["return_p99_abs"] <= return_limit
        for item in numerics
    )
    if not basic_finite_pass:
        reasons.append("nonfinite-or-return-p99-limit-failed")
    return {
        "expected_update_count": expected_updates,
        "observed_update_count": len(update_history),
        "consecutive_failure_updates": consecutive_limit,
        "thresholds": {
            "max_value_target_normalized_rmse": rmse_limit,
            "max_approx_kl": kl_limit,
            "max_clip_fraction": clip_limit,
            "max_return_p99_abs": return_limit,
        },
        "max_consecutive_breaches": {
            "value_target_normalized_rmse": rmse_streak,
            "approx_kl": kl_streak,
            "clip_fraction": clip_streak,
        },
        "failure_reasons": reasons,
        "pass": bool(basic_finite_pass and not reasons),
    }


def _train_sb3_pilot_model(
    env,
    smoke: Mapping[str, object],
    pilot: Mapping[str, object],
    model_base: Path,
):
    """从同一初始化连续运行三次完整 rollout，并逐次保存数值诊断。"""
    one_rollout = dict(smoke)
    one_rollout["total_timesteps"] = int(smoke["n_steps"])
    started = time.monotonic()
    model, result = _train_sb3_model(env, one_rollout, model_base)
    update_history = [{
        "update_index": 1,
        "total_timesteps": int(model.num_timesteps),
        "policy_sha256": _state_dict_sha256(model.policy),
        "numerics": result["numerics"],
    }]
    for update_index in range(2, int(pilot["rollout_count"]) + 1):
        model.learn(
            total_timesteps=int(smoke["n_steps"]),
            reset_num_timesteps=False,
        )
        update_history.append({
            "update_index": update_index,
            "total_timesteps": int(model.num_timesteps),
            "policy_sha256": _state_dict_sha256(model.policy),
            "numerics": _training_numerics(model),
        })
    model.save(str(model_base))
    model_path = Path(str(model_base) + ".zip")
    result.update({
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "trained_policy_sha256": _state_dict_sha256(model.policy),
        "total_timesteps_requested": (
            int(smoke["n_steps"]) * int(pilot["rollout_count"])
        ),
        "total_timesteps_actual": int(model.num_timesteps),
        "elapsed_seconds": time.monotonic() - started,
        "numerics": update_history[-1]["numerics"],
        "update_history": update_history,
        "stability": _stability_numeric_summary(update_history, pilot),
    })
    return model, result


def _deterministic_final_replay(model, episode: LocalEPEEpisode, seed: int) -> dict:
    """确定性逐点生成完整 Recipe，并与一次独立 batched final replay 严格比较。"""
    point_order = tuple(sorted(episode.point_ids))
    observation, reset_info = episode.reset(seed=int(seed), point_order=point_order)
    actions_by_point_id = {}
    rewards = []
    terminated = False
    for point_id in point_order:
        action, _ = model.predict(observation, deterministic=True)
        action_class = int(np.asarray(action).item())
        actions_by_point_id[point_id] = action_class
        observation, reward, terminated, info = episode.step(action_class)
        rewards.append(float(reward))
    if not terminated or not info["final_recipe_complete"]:
        raise RuntimeError("PPO 确定性回放未完成全部 EPE 点")
    final = episode.final_golden_evaluation
    final_result = episode.final_result
    offsets, replay_result, replay_golden, replay_recipe_sha256 = (
        episode.replay_complete_action_map(actions_by_point_id)
    )
    replay_equal = bool(
        offsets == episode.final_recipe_offsets_nm
        and replay_recipe_sha256 == episode.final_recipe_sha256
        and replay_result.mask_sha256 == final_result.mask_sha256
        and replay_golden.metrics.as_dict() == final.metrics.as_dict()
        and np.isclose(
            replay_golden.raw_weighted_loss,
            final.raw_weighted_loss,
            rtol=0.0,
            atol=0.0,
        )
    )
    counts = {
        str(index): sum(value == index for value in actions_by_point_id.values())
        for index in range(len(episode.action_offsets_nm))
    }
    dominant = max(counts.values()) / len(actions_by_point_id)
    return {
        "point_count": len(point_order),
        "point_order_sha256": episode.point_order_sha256,
        "baseline_state_sha256": reset_info["baseline_state_sha256"],
        "initial_raw_metrics": reset_info["initial_raw_metrics"],
        "initial_raw_weighted_loss": reset_info["initial_raw_weighted_loss"],
        "action_map_sha256": _sha256_json(actions_by_point_id),
        "action_class_counts": counts,
        "dominant_action_fraction": dominant,
        "scaled_reward_sum": float(sum(rewards)),
        "final_raw_metrics": final.metrics.as_dict(),
        "final_raw_weighted_loss": float(final.raw_weighted_loss),
        "final_recipe_sha256": episode.final_recipe_sha256,
        "final_mask_sha256": final_result.mask_sha256,
        "final_replay_equal": replay_equal,
        "solver_call_counts": episode.solver_call_counts,
    }


def run_v2_ppo_smoke(
    config: dict,
    artifact_root: Path,
    protocol_alias: str,
    layout_parent: str = "M1_test4",
) -> dict:
    """运行一个独立协议的一次 PPO 更新、模型保存和确定性 final replay。"""
    started = time.monotonic()
    if protocol_alias not in PROTOCOL_ALIASES:
        raise ValueError("protocol 必须为 dense 或 terminal")
    training = config["training"]
    smoke = training.get("smoke", {})
    if training.get("enabled") is not False or smoke.get("enabled") is not True:
        raise ValueError("只允许 training.enabled=false 且 training.smoke.enabled=true")
    if str(smoke.get("layout_parent")) != str(layout_parent):
        raise ValueError("PPO smoke 当前只允许配置冻结的单张版图")
    protocol = PROTOCOL_ALIASES[protocol_alias]
    if protocol not in tuple(smoke.get("protocols", ())):
        raise ValueError("请求协议不在 training.smoke.protocols 冻结列表中")
    recipe = config["recipe_v2"]
    observation = recipe["observation"]
    if int(observation["patch_size"]) != 128:
        raise ValueError("v2 PPO smoke 只允许 128x128 observation")
    from .recipe_ppo_v2 import LocalEPEPPOEnv

    train_solver, train_evaluator, verified_fields = (
        _build_v2_openilt_solver_and_evaluator(
            config, layout_parent, require_layout_contract=True
        )
    )
    train_episode = LocalEPEEpisode(
        solver=train_solver,
        golden_evaluator=train_evaluator,
        reward_weights=recipe["reward"]["weights"],
        training_protocol=protocol,
        patch_size=128,
        action_offsets_nm=recipe["epe_normal_offsets_nm"],
        control_probe_distance_nm=recipe["epe_probe_distance_nm"],
        training_reward_scale=recipe["reward"]["scale"],
        shuffle_points=False,
    )
    train_episode.require_plain_ppo_compatible()
    train_episode.reset(
        seed=int(smoke["seed"]),
        point_order=tuple(sorted(train_episode.point_ids)),
    )
    report = observation["report_examples"]
    if report.get("selection_policy") != PPO_INPUT_SELECTION_POLICY:
        raise ValueError("PPO 输入汇报样例抽样协议不一致")
    input_examples = save_v2_ppo_input_examples(
        train_episode,
        Path(artifact_root) / "ppo-input-examples",
        int(report["example_count_per_layout"]),
        str(report["selection_policy"]),
    )
    env = LocalEPEPPOEnv(train_episode)
    model, training_result = _train_sb3_model(
        env,
        smoke,
        Path(artifact_root) / f"model-{protocol_alias}",
    )
    eval_solver, eval_evaluator, eval_verified_fields = (
        _build_v2_openilt_solver_and_evaluator(
            config, layout_parent, require_layout_contract=True
        )
    )
    eval_episode = LocalEPEEpisode(
        solver=eval_solver,
        golden_evaluator=eval_evaluator,
        reward_weights=recipe["reward"]["weights"],
        training_protocol=protocol,
        patch_size=128,
        action_offsets_nm=recipe["epe_normal_offsets_nm"],
        control_probe_distance_nm=recipe["epe_probe_distance_nm"],
        training_reward_scale=recipe["reward"]["scale"],
        shuffle_points=False,
    )
    final_replay = _deterministic_final_replay(
        model, eval_episode, seed=int(smoke["seed"])
    )
    numerics = training_result["numerics"]
    numeric_pass = bool(
        numerics["all_finite"]
        and numerics["return_p99_abs"] <= float(smoke["max_return_p99_abs"])
        and numerics["logger"]["approx_kl"] <= float(smoke["max_approx_kl"])
        and numerics["logger"]["clip_fraction"] <= float(smoke["max_clip_fraction"])
    )
    baseline_equal = (
        train_episode.observation_cache.baseline_state_sha256
        == final_replay["baseline_state_sha256"]
    )
    post_revision = _validate_openilt(
        Path(config["data"]["openilt_dir"]), config["openilt"]["commit"]
    )
    return {
        "status": "diagnostic_only",
        "smoke_version": PPO_SMOKE_VERSION,
        "environment": config["environment"],
        "layout_parent": layout_parent,
        "training_protocol": protocol,
        "seed": int(smoke["seed"]),
        "patch_size": 128,
        "configured_golden_identity_verified_fields": verified_fields,
        "evaluation_golden_identity_verified_fields": eval_verified_fields,
        "ppo_input_examples": input_examples,
        "training": training_result,
        "deterministic_final_replay": final_replay,
        "train_evaluation_baseline_equal": baseline_equal,
        "numeric_pass": numeric_pass,
        "pass": bool(numeric_pass and baseline_equal and final_replay["final_replay_equal"]),
        "solver_call_counts": {
            "training": train_episode.solver_call_counts,
            "evaluation": eval_episode.solver_call_counts,
        },
        "wall_time_seconds": time.monotonic() - started,
        "post_run_openilt_revision": post_revision,
        "post_run_openilt_tracked_diff_clean": True,
        "ppo_updates_performed": True,
        "long_training_enabled": False,
        "accepted": False,
    }


def run_v2_ppo_pilot(
    config: dict,
    artifact_root: Path,
    protocol_alias: str,
    layout_parent: str = "M1_test4",
) -> dict:
    """运行单图单 seed 的三次更新稳定性 pilot，禁止长训练和 accepted。"""
    started = time.monotonic()
    if protocol_alias not in PROTOCOL_ALIASES:
        raise ValueError("protocol 必须为 dense 或 terminal")
    training = config["training"]
    smoke = training.get("smoke", {})
    pilot = training.get("pilot", {})
    if training.get("enabled") is not False or pilot.get("enabled") is not True:
        raise ValueError("pilot 只允许 training.enabled=false 且 training.pilot.enabled=true")
    unfinished_smoke = {
        "dense_ppo_cuda_smoke_not_completed",
        "terminal_ppo_cuda_smoke_not_completed",
        "terminal_ppo_cuda_smoke_artifact_refresh_after_ev_fix",
    }.intersection(training.get("blocked_by", ()))
    if unfinished_smoke:
        raise ValueError("dense/terminal CUDA smoke 尚未全部关闭，禁止运行 stability pilot")
    if str(pilot.get("layout_parent")) != str(layout_parent):
        raise ValueError("PPO stability pilot 当前只允许配置冻结的单张版图")
    if (
        int(pilot.get("rollout_count", 0)) != 3
        or int(pilot.get("consecutive_failure_updates", 0)) != 3
    ):
        raise ValueError("PPO stability pilot 当前只允许恰好三次更新")
    protocol = PROTOCOL_ALIASES[protocol_alias]
    if protocol not in tuple(pilot.get("protocols", ())):
        raise ValueError("请求协议不在 training.pilot.protocols 冻结列表中")
    if int(pilot.get("seed")) != int(smoke.get("seed")):
        raise ValueError("PPO stability pilot 必须与 smoke 使用相同 seed")
    recipe = config["recipe_v2"]
    observation = recipe["observation"]
    if int(observation["patch_size"]) != 128:
        raise ValueError("v2 PPO stability pilot 只允许 128x128 observation")
    from .recipe_ppo_v2 import LocalEPEPPOEnv

    train_solver, train_evaluator, verified_fields = (
        _build_v2_openilt_solver_and_evaluator(
            config, layout_parent, require_layout_contract=True
        )
    )
    train_episode = LocalEPEEpisode(
        solver=train_solver,
        golden_evaluator=train_evaluator,
        reward_weights=recipe["reward"]["weights"],
        training_protocol=protocol,
        patch_size=128,
        action_offsets_nm=recipe["epe_normal_offsets_nm"],
        control_probe_distance_nm=recipe["epe_probe_distance_nm"],
        training_reward_scale=recipe["reward"]["scale"],
        shuffle_points=False,
    )
    train_episode.require_plain_ppo_compatible()
    train_episode.reset(
        seed=int(pilot["seed"]),
        point_order=tuple(sorted(train_episode.point_ids)),
    )
    report = observation["report_examples"]
    if report.get("selection_policy") != PPO_INPUT_SELECTION_POLICY:
        raise ValueError("PPO 输入汇报样例抽样协议不一致")
    input_examples = save_v2_ppo_input_examples(
        train_episode,
        Path(artifact_root) / "ppo-input-examples",
        int(report["example_count_per_layout"]),
        str(report["selection_policy"]),
    )
    env = LocalEPEPPOEnv(train_episode)
    model, training_result = _train_sb3_pilot_model(
        env,
        smoke,
        pilot,
        Path(artifact_root) / f"model-pilot-{protocol_alias}",
    )
    eval_solver, eval_evaluator, eval_verified_fields = (
        _build_v2_openilt_solver_and_evaluator(
            config, layout_parent, require_layout_contract=True
        )
    )
    eval_episode = LocalEPEEpisode(
        solver=eval_solver,
        golden_evaluator=eval_evaluator,
        reward_weights=recipe["reward"]["weights"],
        training_protocol=protocol,
        patch_size=128,
        action_offsets_nm=recipe["epe_normal_offsets_nm"],
        control_probe_distance_nm=recipe["epe_probe_distance_nm"],
        training_reward_scale=recipe["reward"]["scale"],
        shuffle_points=False,
    )
    final_replay = _deterministic_final_replay(
        model, eval_episode, seed=int(pilot["seed"])
    )
    baseline_equal = (
        train_episode.observation_cache.baseline_state_sha256
        == final_replay["baseline_state_sha256"]
    )
    stability_pass = bool(training_result["stability"]["pass"])
    post_revision = _validate_openilt(
        Path(config["data"]["openilt_dir"]), config["openilt"]["commit"]
    )
    return {
        "status": "diagnostic_only",
        "pilot_version": PPO_PILOT_VERSION,
        "environment": config["environment"],
        "layout_parent": layout_parent,
        "training_protocol": protocol,
        "seed": int(pilot["seed"]),
        "patch_size": 128,
        "rollout_count": int(pilot["rollout_count"]),
        "configured_golden_identity_verified_fields": verified_fields,
        "evaluation_golden_identity_verified_fields": eval_verified_fields,
        "ppo_input_examples": input_examples,
        "training": training_result,
        "deterministic_final_replay": final_replay,
        "train_evaluation_baseline_equal": baseline_equal,
        "stability_pass": stability_pass,
        "pass": bool(
            stability_pass
            and baseline_equal
            and final_replay["final_replay_equal"]
        ),
        "solver_call_counts": {
            "training": train_episode.solver_call_counts,
            "evaluation": eval_episode.solver_call_counts,
        },
        "wall_time_seconds": time.monotonic() - started,
        "post_run_openilt_revision": post_revision,
        "post_run_openilt_tracked_diff_clean": True,
        "ppo_updates_performed": True,
        "long_training_enabled": False,
        "accepted": False,
    }
