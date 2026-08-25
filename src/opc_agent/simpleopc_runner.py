"""本模块训练 SimpleOPC 多步 PPO 并导出只能由该 PPO 轨迹产生的 Recipe。

输入为已经连接固定 OpenILT 后端的多步环境、训练步数和随机种子；输出为 PPO 模型、训练元数据、
确定性回放轨迹及历史最优逐段位移 Recipe。关键依赖为 Stable-Baselines3 与 PyTorch；模块不会修改
OpenILT，也不会把启发式九动作最小值冒充 PPO 标签。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .metrics import DISPLACEMENT_CLASSES_NM, SIMPLEOPC_LOSS_VERSION
from .oracle_runner import ppo_rollout_schedule
from .simpleopc import SIMPLEOPC_ENV_VERSION, SimpleOPCMultiStepEnv


PPO_RECIPE_LABEL_VERSION = "ppo-simpleopc-multistep-v3"


def _displacement_class(value_nm: float) -> Tuple[int, float]:
    """把连续累积位移映射到决策树九分类，同时返回量化误差。"""
    values = np.asarray(DISPLACEMENT_CLASSES_NM, dtype=np.float64)
    index = int(np.argmin(np.abs(values - float(value_nm))))
    return index, float(value_nm - values[index])


def deterministic_rollout(model, env: SimpleOPCMultiStepEnv, seed: int) -> Dict[str, Any]:
    """用冻结策略完整跑一个 episode，并保留历史最优而非最后一步 Recipe。"""
    observation, reset_info = env.reset(seed=int(seed))
    terminated = False
    final_info: Dict[str, Any] = {}
    while not terminated:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, final_info = env.step(action)
        if truncated:
            raise RuntimeError("SimpleOPC 确定性回放不允许被截断")
    return {
        "reset": reset_info,
        "final": final_info,
        "trajectory": list(env.trajectory),
        "best_step": env.best_step,
        "best_metrics": env.best_evaluation.metrics.as_dict(),
        "best_displacements_nm": env.best_displacements_nm.tolist(),
    }


def build_recipe_payload(
    env: SimpleOPCMultiStepEnv,
    rollout: Dict[str, Any],
    model_path: Path,
    seed: int,
) -> Dict[str, Any]:
    """把 PPO 最优轨迹转换为带模型和版图来源哈希的可审计 Recipe。"""
    source = Path(model_path)
    if not source.is_file():
        raise FileNotFoundError(f"PPO 模型不存在：{source}")
    displacements = np.asarray(rollout["best_displacements_nm"], dtype=np.float64)
    if displacements.shape != (len(env.segments),):
        raise ValueError("PPO Recipe 位移数量与 SimpleOPC 边段数不一致")
    labels = []
    initial_epe_signs = env.initial_epe_signs
    nm_per_coordinate = float(getattr(env.backend, "nm_per_coordinate", 1.0))
    for index, (segment, displacement) in enumerate(zip(env.segments, displacements)):
        category, error = _displacement_class(float(displacement))
        if not np.isclose(error, 0.0):
            raise RuntimeError("PPO Recipe 位移不在九分类代表值上，拒绝近似量化")
        midpoint_x = (segment.start_x + segment.end_x) * nm_per_coordinate / 2.0
        midpoint_y = (segment.start_y + segment.end_y) * nm_per_coordinate / 2.0
        labels.append({
            "task_type": "EPE",
            "segment": asdict(segment),
            "feature_version": "simpleopc-segment-v1",
            "features": {
                "midpoint_x_nm": float(midpoint_x),
                "midpoint_y_nm": float(midpoint_y),
                "segment_length_nm": float(segment.length_nm),
                "is_horizontal": 1.0 if segment.horizontal else 0.0,
                "normal_x": float(segment.normal_x),
                "normal_y": float(segment.normal_y),
                "initial_epe_sign": float(initial_epe_signs[index]),
            },
            "ppo_displacement_nm": float(displacement),
            "displacement_class": category,
            "quantized_displacement_nm": float(DISPLACEMENT_CLASSES_NM[category]),
            "quantization_error_nm": error,
        })
    return {
        "schema_version": "1.0",
        "label_version": PPO_RECIPE_LABEL_VERSION,
        "feature_version": "simpleopc-segment-v1",
        "environment_version": SIMPLEOPC_ENV_VERSION,
        "loss_version": SIMPLEOPC_LOSS_VERSION,
        "seed": int(seed),
        "openilt_commit": env.backend.revision,
        "layout_sha256": env.backend.layout_sha256,
        "model_path": str(source),
        "model_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reward_weights": dict(env.reward_weights),
        "initial_weighted_loss": float(rollout["trajectory"][0]["raw_weighted_loss"]),
        "loss_scale": float(rollout["reset"]["loss_scale"]),
        "step_sizes_nm": list(env.step_sizes_nm),
        "displacement_classes_nm": list(DISPLACEMENT_CLASSES_NM),
        "displacement_limit_nm": env.displacement_limit_nm,
        "segment_count": len(env.segments),
        "best_step": int(rollout["best_step"]),
        "best_metrics": dict(rollout["best_metrics"]),
        "trajectory": list(rollout["trajectory"]),
        "labels": labels,
        "frag_status": "not_implemented_requires_nested_resegmentation",
    }


def train_simpleopc_ppo(
    env: SimpleOPCMultiStepEnv,
    output_path: Path,
    total_timesteps: int,
    seed: int,
    learning_rate: float = 3e-4,
) -> Tuple[Path, Path]:
    """在 CUDA 上训练多步 PPO，随后冻结回放并导出历史最优 Recipe。"""
    import torch
    import gymnasium
    import stable_baselines3
    from stable_baselines3 import PPO

    if stable_baselines3.__version__ != "2.0.0" or gymnasium.__version__ != "0.28.1":
        raise RuntimeError(
            "SimpleOPC PPO 要求 stable-baselines3==2.0.0 与 gymnasium==0.28.1；"
            f"当前为 {stable_baselines3.__version__}/{gymnasium.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("SimpleOPC PPO 要求 CUDA；当前 torch.cuda.is_available() 为 False")
    if total_timesteps < len(env.step_sizes_nm):
        raise ValueError("total_timesteps 不能少于一个完整 SimpleOPC episode")
    n_steps, batch_size = ppo_rollout_schedule(int(total_timesteps))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model = PPO(
        "MlpPolicy",
        env,
        device="cuda",
        seed=int(seed),
        learning_rate=float(learning_rate),
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=1,
    )
    model.learn(total_timesteps=int(total_timesteps))
    model.save(str(destination))
    model_path = destination if destination.suffix == ".zip" else Path(str(destination) + ".zip")
    rollout = deterministic_rollout(model, env, int(seed))
    recipe = build_recipe_payload(env, rollout, model_path, int(seed))
    recipe_path = model_path.with_suffix(".recipe.json")
    recipe_path.write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "environment_version": SIMPLEOPC_ENV_VERSION,
        "loss_version": SIMPLEOPC_LOSS_VERSION,
        "label_version": PPO_RECIPE_LABEL_VERSION,
        "seed": int(seed),
        "total_timesteps_requested": int(total_timesteps),
        "total_timesteps_actual": int(model.num_timesteps),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "loss_scale": recipe["loss_scale"],
        "episode_horizon": len(env.step_sizes_nm),
        "segment_count": len(env.segments),
        "device": "cuda",
        "elapsed_seconds": time.monotonic() - started,
        "openilt_commit": env.backend.revision,
        "layout_sha256": env.backend.layout_sha256,
        "model_sha256": recipe["model_sha256"],
        "recipe_path": str(recipe_path),
        "best_step": recipe["best_step"],
        "best_metrics": recipe["best_metrics"],
    }
    model_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return model_path, recipe_path


def run_simpleopc_heuristic(env: SimpleOPCMultiStepEnv, seed: int = 0) -> Dict[str, Any]:
    """按当前逐段 EPE 符号运行与上游一致的内移/不动/外移启发式基线。"""
    observation, reset_info = env.reset(seed=int(seed))
    del observation
    terminated = False
    final_info: Dict[str, Any] = {}
    while not terminated:
        signs = env.current_epe_signs
        action = np.where(signs > 0, 2, np.where(signs < 0, 0, 1)).astype(np.int64)
        _, _, terminated, truncated, final_info = env.step(action)
        if truncated:
            raise RuntimeError("SimpleOPC 启发式基线不允许被截断")
    return {
        "environment_version": SIMPLEOPC_ENV_VERSION,
        "loss_version": SIMPLEOPC_LOSS_VERSION,
        "reset": reset_info,
        "final": final_info,
        "trajectory": list(env.trajectory),
        "best_step": env.best_step,
        "best_metrics": env.best_evaluation.metrics.as_dict(),
        "best_displacements_nm": env.best_displacements_nm.tolist(),
    }
