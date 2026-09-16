"""本模块实现 Recipe PPO v2 的双版图单模型 terminal 受控小训练。

输入是已经冻结 Golden contract 的 M1_test5/M1_test6、128×128 五通道局部 observation、
五类法向动作和显式小训练预算；两个版图各自保留独立 OpenILT solver/Gym 环境，但共同写入
一个 SB3 PPO rollout buffer 并只更新一个共享 Actor-Critic。输出包含随机对照、每次更新前后
Critic/return/advantage 诊断、逐版图确定性完整 Recipe 回放、checkpoint、模型哈希和每图 PPO
输入可视化。该入口始终是 diagnostic_only，不改变既有 smoke/pilot，也不产生 accepted。
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
import time
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from .recipe_v2 import LocalEPEEpisode
from .recipe_v2_contract import EPE_TERMINAL_PROTOCOL
from .recipe_v2_openilt import (
    _build_v2_openilt_solver_and_evaluator,
    _sha256_json,
    _validate_openilt,
)
from .recipe_v2_runner import (
    _deterministic_final_replay,
    _feature_extractor_class,
    _max_consecutive_true,
    _state_dict_sha256,
    _training_numerics,
)
from .recipe_v2_visualization import (
    PPO_INPUT_SELECTION_POLICY,
    save_v2_ppo_input_examples,
)


PPO_SMALL_TRAIN_VERSION = "recipe-v2-shared-two-layout-terminal-small-train-v1"


def _array_summary(values) -> Dict[str, float]:
    """返回训练数组的有限基础统计，空数组和非有限值立即失败。"""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise RuntimeError("小训练诊断数组必须非空且全部有限")
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _value_error_summary(values, returns) -> Dict[str, float]:
    """计算 value 相对同批 return target 的绝对和归一化 RMSE。"""
    value_array = np.asarray(values, dtype=np.float64).reshape(-1)
    return_array = np.asarray(returns, dtype=np.float64).reshape(-1)
    if value_array.shape != return_array.shape:
        raise ValueError("value 与 return 的形状必须一致")
    if not np.all(np.isfinite(value_array)) or not np.all(np.isfinite(return_array)):
        raise RuntimeError("value/return 误差诊断包含非有限数值")
    rmse = float(np.sqrt(np.mean(np.square(value_array - return_array))))
    return_rms = float(np.sqrt(np.mean(np.square(return_array))))
    return {
        "rmse": rmse,
        "normalized_rmse": rmse / max(return_rms, 1e-3),
        "return_rms": return_rms,
    }


def _rollout_scalar_matrix(
    values,
    *,
    n_steps: int,
    n_envs: int,
    field_name: str,
) -> np.ndarray:
    """把 SB3 更新前二维或更新后环境优先展平的标量字段恢复成 step×env。"""
    array = np.asarray(values)
    expected_size = int(n_steps) * int(n_envs)
    if array.size != expected_size:
        raise RuntimeError(
            f"共享 rollout 字段 {field_name} 元素数 {array.size} != {expected_size}"
        )
    if array.ndim >= 2 and array.shape[:2] == (n_steps, n_envs):
        return np.asarray(array, dtype=np.float64).reshape(n_steps, n_envs)
    if array.shape[0] == expected_size:
        # RolloutBuffer.get() 使用 swap_and_flatten：先 env、后 step。
        env_major = np.asarray(array, dtype=np.float64).reshape(n_envs, n_steps)
        return env_major.T
    raise RuntimeError(
        f"共享 rollout 字段 {field_name} 形状 {array.shape} 既不是更新前 step×env，"
        "也不是 SB3 更新后的 env-major 展平格式"
    )


def _rollout_observation_env_major(
    values,
    *,
    n_steps: int,
    n_envs: int,
    field_name: str,
) -> np.ndarray:
    """把 observation 统一成 SB3 更新后使用的 env-major 批次且保留特征维度。"""
    array = np.asarray(values)
    expected_size = int(n_steps) * int(n_envs)
    if array.ndim >= 2 and array.shape[:2] == (n_steps, n_envs):
        return array.swapaxes(0, 1).reshape(
            (expected_size,) + tuple(array.shape[2:])
        )
    if array.ndim >= 1 and array.shape[0] == expected_size:
        return array
    raise RuntimeError(
        f"共享 rollout observation {field_name} 形状 {array.shape} 无法按 "
        f"n_steps={n_steps}, n_envs={n_envs} 解释"
    )


def _small_train_contract(
    small_train: Mapping[str, object], episode_horizons: Sequence[int]
) -> Dict[str, int]:
    """冻结双版图平衡采样预算，并拒绝不能整除的 PPO buffer。"""
    horizons = tuple(int(value) for value in episode_horizons)
    if len(horizons) != 2 or len(set(horizons)) != 1 or horizons[0] <= 0:
        raise ValueError("共享小训练当前要求两张版图具有相同且为正的 episode horizon")
    per_layout_per_update = int(small_train["episodes_per_layout_per_update"])
    update_count = int(small_train["update_count"])
    episodes_per_layout = int(small_train["episodes_per_layout"])
    if per_layout_per_update != 2 or update_count != 5 or episodes_per_layout != 10:
        raise ValueError("共享小训练当前冻结为每图每次 2 episode、5 次更新、每图共 10 episode")
    n_envs = len(horizons)
    n_steps_per_env = horizons[0] * per_layout_per_update
    rollout_buffer_size = n_steps_per_env * n_envs
    total_timesteps = rollout_buffer_size * update_count
    if int(small_train["n_steps_per_env"]) != n_steps_per_env:
        raise ValueError("small_train.n_steps_per_env 与版图 horizon/episode 预算不一致")
    if int(small_train["total_timesteps"]) != total_timesteps:
        raise ValueError("small_train.total_timesteps 与共享 rollout 预算不一致")
    batch_size = int(small_train["batch_size"])
    if batch_size <= 1 or rollout_buffer_size % batch_size != 0:
        raise ValueError("small_train.batch_size 必须大于 1 且整除共享 rollout buffer")
    if int(small_train["random_recipe_count_per_layout"]) != episodes_per_layout:
        raise ValueError("随机 Recipe 对照必须与每图训练 episode 数相同")
    return {
        "episode_horizon": horizons[0],
        "n_envs": n_envs,
        "n_steps_per_env": n_steps_per_env,
        "rollout_buffer_size": rollout_buffer_size,
        "update_count": update_count,
        "episodes_per_layout_per_update": per_layout_per_update,
        "episodes_per_layout": episodes_per_layout,
        "total_episodes": episodes_per_layout * n_envs,
        "total_timesteps": total_timesteps,
        "batch_size": batch_size,
    }


def _small_train_gate(
    update_history: Sequence[Mapping[str, object]],
    small_train: Mapping[str, object],
) -> Dict[str, object]:
    """按更新后 Critic RMSE、KL、clip 和有限性生成受控训练熔断判定。"""
    limit = int(small_train["consecutive_failure_updates"])
    if limit != 3:
        raise ValueError("共享小训练当前只允许连续三次失败熔断")
    rmse_limit = float(small_train["max_post_update_value_target_normalized_rmse"])
    kl_limit = float(small_train["max_approx_kl"])
    clip_limit = float(small_train["max_clip_fraction"])
    return_limit = float(small_train["max_return_p99_abs"])
    numerics = [item["numerics"] for item in update_history]
    rmse_streak = _max_consecutive_true(
        item["post_update_value_target"]["normalized_rmse"] > rmse_limit
        for item in numerics
    )
    kl_streak = _max_consecutive_true(
        item["logger"]["approx_kl"] > kl_limit for item in numerics
    )
    clip_streak = _max_consecutive_true(
        item["logger"]["clip_fraction"] > clip_limit for item in numerics
    )
    reasons = []
    if any(
        not bool(item["all_finite"])
        or float(item["return_p99_abs"]) > return_limit
        for item in numerics
    ):
        reasons.append("nonfinite-or-return-p99-limit-failed")
    if rmse_streak >= limit:
        reasons.append("post-update-value-target-normalized-rmse-high-for-three-updates")
    if kl_streak >= limit:
        reasons.append("approx-kl-high-for-three-updates")
    if clip_streak >= limit:
        reasons.append("clip-fraction-high-for-three-updates")
    return {
        "consecutive_failure_updates": limit,
        "thresholds": {
            "max_post_update_value_target_normalized_rmse": rmse_limit,
            "max_approx_kl": kl_limit,
            "max_clip_fraction": clip_limit,
            "max_return_p99_abs": return_limit,
        },
        "max_consecutive_breaches": {
            "post_update_value_target_normalized_rmse": rmse_streak,
            "approx_kl": kl_streak,
            "clip_fraction": clip_streak,
        },
        "failure_reasons": reasons,
        "should_stop": bool(reasons),
    }


def _shared_training_numerics(model, layout_parents: Sequence[str]) -> Dict[str, object]:
    """兼容 SB3 更新后展平 buffer，记录共享训练数值并按环境拆分。"""
    import torch

    legacy = _training_numerics(model)
    buffer = model.rollout_buffer
    n_steps = int(buffer.buffer_size)
    n_envs = int(buffer.n_envs)
    if n_steps <= 0 or n_envs != len(layout_parents):
        raise RuntimeError("共享 rollout 的环境列数与 layout_parents 不一致")
    scalar_fields = {
        "returns": buffer.returns,
        "values": buffer.values,
        "rewards": buffer.rewards,
        "advantages": buffer.advantages,
        "actions": buffer.actions,
    }
    matrices = {
        name: _rollout_scalar_matrix(
            values,
            n_steps=n_steps,
            n_envs=n_envs,
            field_name=name,
        )
        for name, values in scalar_fields.items()
    }
    returns = matrices["returns"]
    pre_values = matrices["values"]
    rewards = matrices["rewards"]
    advantages = matrices["advantages"]
    sampled_actions = np.asarray(matrices["actions"], dtype=np.int64)
    observations = {}
    for name, values in buffer.observations.items():
        array = _rollout_observation_env_major(
            values,
            n_steps=n_steps,
            n_envs=n_envs,
            field_name=name,
        )
        observations[name] = torch.as_tensor(
            array,
            device=model.device,
        )
    with torch.no_grad():
        post_values_env_major = (
            model.policy.predict_values(observations).detach().cpu().numpy().reshape(-1)
        )
        distribution = model.policy.get_distribution(observations).distribution
        probabilities = distribution.probs.detach().cpu().numpy()
    expected_size = n_steps * n_envs
    if post_values_env_major.size != expected_size:
        raise RuntimeError("更新后 Critic 输出数量与共享 rollout 不一致")
    if probabilities.ndim != 2 or probabilities.shape[0] != expected_size:
        raise RuntimeError("更新后动作概率数量与共享 rollout 不一致")
    post_values = post_values_env_major.reshape(n_envs, n_steps).T
    finite_arrays = (returns, pre_values, post_values, rewards, advantages, probabilities)
    all_finite = all(np.all(np.isfinite(item)) for item in finite_arrays)
    if not all_finite:
        raise RuntimeError("共享小训练的 return/value/advantage/action probability 包含非有限数值")
    sorted_probabilities = np.sort(probabilities, axis=1)
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1
    )
    margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    deterministic_actions = np.argmax(probabilities, axis=1)
    action_count = probabilities.shape[1]
    layout_summaries = {}
    for env_index, layout_parent in enumerate(layout_parents):
        layout_returns = returns[:, env_index]
        layout_advantages = advantages[:, env_index]
        layout_summaries[str(layout_parent)] = {
            "sample_count": int(layout_returns.size),
            "reward": _array_summary(rewards[:, env_index]),
            "return": _array_summary(layout_returns),
            "advantage": {
                **_array_summary(layout_advantages),
                "positive_fraction": float(np.mean(layout_advantages > 0.0)),
                "negative_fraction": float(np.mean(layout_advantages < 0.0)),
                "zero_fraction": float(np.mean(layout_advantages == 0.0)),
            },
            "pre_update_value": _array_summary(pre_values[:, env_index]),
            "post_update_value": _array_summary(post_values[:, env_index]),
            "pre_update_value_target": _value_error_summary(
                pre_values[:, env_index], layout_returns
            ),
            "post_update_value_target": _value_error_summary(
                post_values[:, env_index], layout_returns
            ),
        }
    return {
        "sample_count": int(returns.size),
        "return_p99_abs": float(np.quantile(np.abs(returns.reshape(-1)), 0.99)),
        "reward": _array_summary(rewards),
        "return": _array_summary(returns),
        "advantage": {
            **_array_summary(advantages),
            "positive_fraction": float(np.mean(advantages > 0.0)),
            "negative_fraction": float(np.mean(advantages < 0.0)),
            "zero_fraction": float(np.mean(advantages == 0.0)),
        },
        "pre_update_value": _array_summary(pre_values),
        "post_update_value": _array_summary(post_values),
        "pre_update_value_target": _value_error_summary(pre_values, returns),
        "post_update_value_target": _value_error_summary(post_values, returns),
        "action_distribution": {
            "sampled_action_class_counts": {
                str(index): int(np.sum(sampled_actions == index))
                for index in range(action_count)
            },
            "deterministic_argmax_class_counts": {
                str(index): int(np.sum(deterministic_actions == index))
                for index in range(action_count)
            },
            "mean_probability_by_action_class": [
                float(value) for value in np.mean(probabilities, axis=0)
            ],
            "entropy": _array_summary(entropy),
            "argmax_margin": _array_summary(margins),
        },
        "layout_summaries": layout_summaries,
        "update_count": int(getattr(model, "_n_updates", 0)),
        "logger": legacy["logger"],
        "all_finite": True,
    }


def _random_terminal_recipe_baseline(
    episode: LocalEPEEpisode, seed: int, sample_count: int
) -> Dict[str, object]:
    """用固定预算生成不参与训练/选模的随机完整 Recipe 对照。"""
    if episode.training_protocol != EPE_TERMINAL_PROTOCOL:
        raise ValueError("随机完整 Recipe 对照当前只允许 terminal 协议")
    count = int(sample_count)
    if count <= 0:
        raise ValueError("随机 Recipe 样本数必须为正")
    rng = np.random.default_rng(int(seed))
    point_order = tuple(sorted(episode.point_ids))
    samples = []
    initial_loss = None
    initial_metrics = None
    baseline_state_sha256 = None
    for index in range(count):
        _, reset_info = episode.reset(seed=int(seed) + index, point_order=point_order)
        if initial_loss is None:
            initial_loss = float(reset_info["initial_raw_weighted_loss"])
            initial_metrics = reset_info["initial_raw_metrics"]
            baseline_state_sha256 = reset_info["baseline_state_sha256"]
        actions = {}
        rewards = []
        terminated = False
        for point_id in point_order:
            action = int(rng.integers(0, len(episode.action_offsets_nm)))
            actions[point_id] = action
            _, reward, terminated, info = episode.step(action)
            rewards.append(float(reward))
        if not terminated or not info["final_recipe_complete"]:
            raise RuntimeError("随机完整 Recipe 未覆盖全部 EPE 点")
        final = episode.final_golden_evaluation
        samples.append({
            "sample_index": index,
            "action_map_sha256": _sha256_json(actions),
            "action_class_counts": {
                str(action): sum(value == action for value in actions.values())
                for action in range(len(episode.action_offsets_nm))
            },
            "scaled_reward_sum": float(sum(rewards)),
            "final_raw_metrics": final.metrics.as_dict(),
            "final_raw_weighted_loss": float(final.raw_weighted_loss),
            "final_recipe_sha256": episode.final_recipe_sha256,
            "final_mask_sha256": episode.final_result.mask_sha256,
        })
    losses = np.asarray(
        [item["final_raw_weighted_loss"] for item in samples], dtype=np.float64
    )
    best_index = int(np.argmin(losses))
    return {
        "seed": int(seed),
        "sample_count": count,
        "selection_used_for_training_or_checkpoint": False,
        "baseline_state_sha256": baseline_state_sha256,
        "initial_raw_metrics": initial_metrics,
        "initial_raw_weighted_loss": initial_loss,
        "weighted_loss_median": float(np.median(losses)),
        "weighted_loss_best": float(losses[best_index]),
        "best_sample_index": best_index,
        "improved_over_zero_offset_count": int(np.sum(losses < float(initial_loss))),
        "samples": samples,
        "solver_call_counts": dict(episode.solver_call_counts),
    }


def _build_episode(
    config: dict,
    layout_parent: str,
    *,
    shuffle_points: bool,
) -> Tuple[LocalEPEEpisode, Tuple[str, ...]]:
    """为一张冻结 train 版图建立独立 terminal episode。"""
    solver, evaluator, verified_fields = _build_v2_openilt_solver_and_evaluator(
        config, layout_parent, require_layout_contract=True
    )
    recipe = config["recipe_v2"]
    episode = LocalEPEEpisode(
        solver=solver,
        golden_evaluator=evaluator,
        reward_weights=recipe["reward"]["weights"],
        training_protocol=EPE_TERMINAL_PROTOCOL,
        patch_size=128,
        action_offsets_nm=recipe["epe_normal_offsets_nm"],
        control_probe_distance_nm=recipe["epe_probe_distance_nm"],
        training_reward_scale=recipe["reward"]["scale"],
        shuffle_points=bool(shuffle_points),
    )
    episode.require_plain_ppo_compatible()
    return episode, tuple(verified_fields)


def _build_ppo_env(episode: LocalEPEEpisode):
    """延迟导入 Gym adapter，使纯编排测试不依赖本机 CUDA/Gym 环境。"""
    from .recipe_ppo_v2 import LocalEPEPPOEnv

    return LocalEPEPPOEnv(episode)


def _train_shared_terminal_model(
    envs,
    layout_parents: Sequence[str],
    small_train: Mapping[str, object],
    contract: Mapping[str, int],
    model_base: Path,
    checkpoint_root: Path,
    evaluation_callback: Callable[[object, int], Mapping[str, object]],
):
    """在两个环境上更新一个 PPO 模型，逐次评价、存档并按新门禁熔断。"""
    import gymnasium
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    if stable_baselines3.__version__ != "2.0.0" or gymnasium.__version__ != "0.28.1":
        raise RuntimeError(
            "v2 共享小训练要求 stable-baselines3==2.0.0 与 gymnasium==0.28.1；"
            f"当前为 {stable_baselines3.__version__}/{gymnasium.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("v2 共享小训练要求 CUDA")
    seed = int(small_train["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    vec_env = DummyVecEnv([lambda env=env: env for env in envs])
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        device="cuda",
        seed=seed,
        learning_rate=float(small_train["learning_rate"]),
        n_steps=int(contract["n_steps_per_env"]),
        batch_size=int(contract["batch_size"]),
        n_epochs=int(small_train["n_epochs"]),
        gamma=float(small_train["gamma"]),
        gae_lambda=float(small_train["gae_lambda"]),
        clip_range=float(small_train["clip_range"]),
        ent_coef=float(small_train["ent_coef"]),
        vf_coef=float(small_train["vf_coef"]),
        policy_kwargs={
            "features_extractor_class": _feature_extractor_class(),
            "features_extractor_kwargs": {"features_dim": 256},
            "normalize_images": False,
        },
        verbose=1,
    )
    initial_policy_sha256 = _state_dict_sha256(model.policy)
    initial_evaluation = dict(evaluation_callback(model, 0))
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    update_history = []
    stop_decision = {
        "failure_reasons": [],
        "should_stop": False,
    }
    started = time.monotonic()
    steps_per_update = int(contract["rollout_buffer_size"])
    for update_index in range(1, int(contract["update_count"]) + 1):
        update_started = time.monotonic()
        model.learn(
            total_timesteps=steps_per_update,
            reset_num_timesteps=(update_index == 1),
        )
        numerics = _shared_training_numerics(model, layout_parents)
        checkpoint_base = checkpoint_root / f"update-{update_index:03d}"
        model.save(str(checkpoint_base))
        checkpoint_path = Path(str(checkpoint_base) + ".zip")
        evaluation = dict(evaluation_callback(model, update_index))
        update_history.append({
            "update_index": update_index,
            "total_timesteps": int(model.num_timesteps),
            "completed_episodes_per_layout": (
                update_index * int(contract["episodes_per_layout_per_update"])
            ),
            "policy_sha256": _state_dict_sha256(model.policy),
            "checkpoint": str(checkpoint_path.relative_to(model_base.parent)),
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "elapsed_seconds": time.monotonic() - update_started,
            "numerics": numerics,
            "deterministic_evaluation": evaluation,
        })
        stop_decision = _small_train_gate(update_history, small_train)
        if not all(
            bool(result["final_replay_equal"]) for result in evaluation.values()
        ):
            stop_decision = {
                **stop_decision,
                "failure_reasons": [
                    *stop_decision["failure_reasons"],
                    "deterministic-final-replay-mismatch",
                ],
                "should_stop": True,
            }
        if stop_decision["should_stop"]:
            break
    model_base.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_base))
    model_path = Path(str(model_base) + ".zip")
    completed_updates = len(update_history)
    training_completed = bool(
        completed_updates == int(contract["update_count"])
        and not stop_decision["should_stop"]
    )
    return model, {
        "model": model_path.name,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "initial_policy_sha256": initial_policy_sha256,
        "trained_policy_sha256": _state_dict_sha256(model.policy),
        "total_timesteps_requested": int(contract["total_timesteps"]),
        "total_timesteps_actual": int(model.num_timesteps),
        "completed_update_count": completed_updates,
        "completed_episodes_per_layout": (
            completed_updates * int(contract["episodes_per_layout_per_update"])
        ),
        "training_completed": training_completed,
        "stopped_early": not training_completed,
        "stop_decision": stop_decision,
        "initial_evaluation": initial_evaluation,
        "update_history": update_history,
        "elapsed_seconds": time.monotonic() - started,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_version": gymnasium.__version__,
    }


def _sum_solver_calls(groups: Mapping[str, Mapping[str, Mapping[str, int]]]) -> int:
    """汇总 training/evaluation/random 三组逐版图 solver 账目。"""
    return int(sum(
        int(value)
        for layouts in groups.values()
        for counts in layouts.values()
        for value in counts.values()
    ))


def run_v2_ppo_small_train(
    config: dict,
    artifact_root: Path,
    protocol_alias: str,
    layout_parents: Sequence[str],
) -> dict:
    """运行 M1_test5/6 双环境、单共享模型的 terminal 受控小训练。"""
    started = time.monotonic()
    training = config["training"]
    small_train = training.get("small_train", {})
    if training.get("enabled") is not False or small_train.get("enabled") is not True:
        raise ValueError("small train 只允许 training.enabled=false 且 small_train.enabled=true")
    if protocol_alias != "terminal":
        raise ValueError("共享小训练 v1 当前只允许 terminal 协议")
    if EPE_TERMINAL_PROTOCOL not in tuple(small_train.get("protocols", ())):
        raise ValueError("terminal 协议不在 small_train.protocols 冻结列表中")
    requested_layouts = tuple(str(value) for value in layout_parents)
    configured_layouts = tuple(str(value) for value in small_train["layout_parents"])
    if requested_layouts != configured_layouts or len(set(requested_layouts)) != 2:
        raise ValueError("共享小训练必须按配置顺序使用两张且不重复的冻结版图")
    train_parents = set(str(value) for value in config["data"]["train_parents"])
    if not set(requested_layouts).issubset(train_parents):
        raise ValueError("共享小训练只能使用 train_parents，禁止读取 validation/test")
    recipe = config["recipe_v2"]
    observation = recipe["observation"]
    if int(observation["patch_size"]) != 128:
        raise ValueError("v2 共享小训练只允许 128x128 observation")
    if float(recipe["reward"]["scale"]) != float(small_train["reward_scale"]):
        raise ValueError("共享小训练不得静默覆盖首轮 reward scale")
    training_episodes = {}
    configured_fields = {}
    input_examples = {}
    report = observation["report_examples"]
    if report.get("selection_policy") != PPO_INPUT_SELECTION_POLICY:
        raise ValueError("PPO 输入汇报样例抽样协议不一致")
    for layout_parent in requested_layouts:
        episode, fields = _build_episode(
            config, layout_parent, shuffle_points=True
        )
        episode.reset(
            seed=int(small_train["seed"]),
            point_order=tuple(sorted(episode.point_ids)),
        )
        training_episodes[layout_parent] = episode
        configured_fields[layout_parent] = fields
        saved = save_v2_ppo_input_examples(
            episode,
            Path(artifact_root) / "ppo-input-examples" / layout_parent,
            int(report["example_count_per_layout"]),
            str(report["selection_policy"]),
        )
        input_examples[layout_parent] = {
            **saved,
            "directory": f"ppo-input-examples/{layout_parent}",
            "manifest": f"ppo-input-examples/{layout_parent}/manifest.json",
        }
    contract = _small_train_contract(
        small_train,
        [training_episodes[name].episode_horizon for name in requested_layouts],
    )

    random_episodes = {}
    random_fields = {}
    random_baselines = {}
    for layout_index, layout_parent in enumerate(requested_layouts):
        episode, fields = _build_episode(
            config, layout_parent, shuffle_points=False
        )
        random_episodes[layout_parent] = episode
        random_fields[layout_parent] = fields
        random_baselines[layout_parent] = _random_terminal_recipe_baseline(
            episode,
            seed=int(small_train["seed"]) + 1000 + layout_index,
            sample_count=int(small_train["random_recipe_count_per_layout"]),
        )

    evaluation_episodes = {}
    evaluation_fields = {}
    for layout_parent in requested_layouts:
        episode, fields = _build_episode(config, layout_parent, shuffle_points=False)
        evaluation_episodes[layout_parent] = episode
        evaluation_fields[layout_parent] = fields

    def evaluate(model, update_index: int) -> Dict[str, object]:
        """对共享 checkpoint 在两张训练图上分别执行确定性完整回放。"""
        return {
            layout_parent: _deterministic_final_replay(
                model,
                evaluation_episodes[layout_parent],
                seed=int(small_train["seed"]) + int(update_index),
            )
            for layout_parent in requested_layouts
        }

    envs = [_build_ppo_env(training_episodes[name]) for name in requested_layouts]
    _, training_result = _train_shared_terminal_model(
        envs,
        requested_layouts,
        small_train,
        contract,
        Path(artifact_root) / "model-shared-terminal-final",
        Path(artifact_root) / "checkpoints",
        evaluate,
    )
    final_evaluation = (
        training_result["update_history"][-1]["deterministic_evaluation"]
        if training_result["update_history"]
        else training_result["initial_evaluation"]
    )
    zero_offset_baselines = {
        name: {
            "baseline_state_sha256": random_baselines[name][
                "baseline_state_sha256"
            ],
            "raw_metrics": random_baselines[name]["initial_raw_metrics"],
            "raw_weighted_loss": random_baselines[name][
                "initial_raw_weighted_loss"
            ],
        }
        for name in requested_layouts
    }
    baseline_identity_equal = all(
        training_episodes[name].observation_cache.baseline_state_sha256
        == final_evaluation[name]["baseline_state_sha256"]
        == random_baselines[name]["baseline_state_sha256"]
        for name in requested_layouts
    )
    episode_contract_equal = all(
        training_episodes[name].geometry_identity_sha256
        == random_episodes[name].geometry_identity_sha256
        == evaluation_episodes[name].geometry_identity_sha256
        and training_episodes[name].action_table_sha256
        == random_episodes[name].action_table_sha256
        == evaluation_episodes[name].action_table_sha256
        and configured_fields[name]
        == random_fields[name]
        == evaluation_fields[name]
        for name in requested_layouts
    )
    all_replays_equal = all(
        result["final_replay_equal"]
        for result in training_result["initial_evaluation"].values()
    ) and all(
        result["final_replay_equal"]
        for update in training_result["update_history"]
        for result in update["deterministic_evaluation"].values()
    )
    solver_call_counts = {
        "training": {
            name: dict(training_episodes[name].solver_call_counts)
            for name in requested_layouts
        },
        "evaluation": {
            name: dict(evaluation_episodes[name].solver_call_counts)
            for name in requested_layouts
        },
        "random_baseline": {
            name: dict(random_episodes[name].solver_call_counts)
            for name in requested_layouts
        },
    }
    post_revision = _validate_openilt(
        Path(config["data"]["openilt_dir"]), config["openilt"]["commit"]
    )
    execution_pass = bool(
        training_result["training_completed"]
        and baseline_identity_equal
        and episode_contract_equal
        and all_replays_equal
    )
    return {
        "status": "diagnostic_only",
        "small_train_version": PPO_SMALL_TRAIN_VERSION,
        "environment": config["environment"],
        "training_protocol": EPE_TERMINAL_PROTOCOL,
        "layout_parents": list(requested_layouts),
        "shared_model": True,
        "actor_layout_id_included": False,
        "balanced_layout_sampling": True,
        "seed": int(small_train["seed"]),
        "patch_size": 128,
        "contract": contract,
        "configured_golden_identity_verified_fields": configured_fields,
        "random_golden_identity_verified_fields": random_fields,
        "evaluation_golden_identity_verified_fields": evaluation_fields,
        "ppo_input_examples": input_examples,
        "zero_offset_baselines": zero_offset_baselines,
        "untrained_deterministic_evaluation": training_result[
            "initial_evaluation"
        ],
        "random_baselines": random_baselines,
        "training": training_result,
        "final_deterministic_evaluation": final_evaluation,
        "baseline_identity_equal": baseline_identity_equal,
        "episode_contract_equal": episode_contract_equal,
        "all_final_replays_equal": all_replays_equal,
        "execution_pass": execution_pass,
        "pass": execution_pass,
        "solver_call_counts": {
            **solver_call_counts,
            "total": _sum_solver_calls(solver_call_counts),
        },
        "wall_time_seconds": time.monotonic() - started,
        "post_run_openilt_revision": post_revision,
        "post_run_openilt_tracked_diff_clean": True,
        "ppo_updates_performed": bool(training_result["completed_update_count"]),
        "quality_accepted": False,
        "long_training_enabled": False,
        "accepted": False,
    }
