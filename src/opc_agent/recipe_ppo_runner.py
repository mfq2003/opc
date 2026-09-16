"""本模块训练共享的点级 Recipe PPO，并导出逐 clip、逐点可追溯 Recipe。

输入为多个 train clip 的 RecipePointPPOEnv、九分类动作和 64×64 五通道局部图像；输出为共享
PPO 模型、训练元数据、确定性回放轨迹及同时包含 EPE/FRAG 的最佳 Recipe。CNN 只编码当前点
局部图像，轻量向量编码坐标、点类型、切向/法向、进度和物理损失；模块不会调用 Qwen、DQN 或决策树。
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from .metrics import DISPLACEMENT_CLASSES_NM, RECIPE_OPC_LOSS_VERSION
from .recipe_contract import PPO_RECIPE_LABEL_VERSION
from .recipe_ppo import (
    RECIPE_ENV_VERSION,
    RECIPE_OBSERVATION_VERSION,
    RECIPE_POINT_VERSION,
    RecipePointPPOEnv,
)


def _feature_extractor_class():
    """延迟构造 SB3 自定义 CNN，避免 CPU 数据命令强制导入 PyTorch。"""
    import torch
    from torch import nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class RecipePointFeatureExtractor(BaseFeaturesExtractor):
        """融合 64×64 五通道 CNN 特征和 14 维 recipe 点向量。"""

        def __init__(self, observation_space, features_dim: int = 256):
            super().__init__(observation_space, features_dim)
            image_space = observation_space.spaces["image"]
            vector_space = observation_space.spaces["vector"]
            channels = int(image_space.shape[0])
            self.cnn = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
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
            image_features = self.cnn(observations["image"].float())
            vector_features = self.vector(observations["vector"].float())
            return self.fusion(torch.cat((image_features, vector_features), dim=1))

    return RecipePointFeatureExtractor


def deterministic_recipe_rollout(model, env: RecipePointPPOEnv, seed: int) -> Dict[str, Any]:
    """用冻结策略让每个 recipe 点恰好决策一次，并保留历史最佳 recipe。"""
    observation, reset_info = env.reset(seed=int(seed))
    terminated = False
    final_info: Dict[str, Any] = {}
    while not terminated:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, final_info = env.step(int(np.asarray(action).item()))
        if truncated:
            raise RuntimeError("Recipe PPO 确定性回放不允许被截断")
    return {
        "reset": reset_info,
        "final": final_info,
        "trajectory": list(env.trajectory),
        "best_step": env.best_step,
        "best_metrics": env.best_evaluation.metrics.as_dict(),
        "best_recipe_offsets_nm": env.best_recipe_offsets_nm.tolist(),
        "best_internal_solver_trace": list(env.best_evaluation.internal_trace),
    }


def build_recipe_payload(
    env: RecipePointPPOEnv,
    rollout: Dict[str, Any],
    model_path: Path,
    seed: int,
) -> Dict[str, Any]:
    """把确定性点级回放转换为同时包含 EPE/FRAG 的模型绑定 Recipe。"""
    source = Path(model_path)
    if not source.is_file():
        raise FileNotFoundError(f"PPO 模型不存在：{source}")
    offsets = np.asarray(rollout["best_recipe_offsets_nm"], dtype=np.float64)
    if offsets.shape != (len(env.recipe_points),):
        raise ValueError("PPO Recipe 位移数量与 recipe 点数不一致")
    classes = np.asarray(DISPLACEMENT_CLASSES_NM, dtype=np.float64)
    labels = []
    for point, displacement in zip(env.recipe_points, offsets):
        matches = np.flatnonzero(np.isclose(classes, displacement))
        if matches.size != 1:
            raise RuntimeError("PPO Recipe 位移不在九分类代表值上")
        action_class = int(matches[0])
        labels.append({
            "task_type": point.task_type,
            "point": asdict(point),
            "point_version": RECIPE_POINT_VERSION,
            "ppo_displacement_nm": float(displacement),
            "displacement_class": action_class,
            "quantized_displacement_nm": float(classes[action_class]),
            "quantization_error_nm": 0.0,
        })
    epe_count = sum(label["task_type"] == "EPE" for label in labels)
    frag_count = len(labels) - epe_count
    initial_loss = float(rollout["trajectory"][0]["raw_weighted_loss"])
    return {
        "schema_version": "2.0",
        "label_version": PPO_RECIPE_LABEL_VERSION,
        "point_version": RECIPE_POINT_VERSION,
        "observation_version": RECIPE_OBSERVATION_VERSION,
        "environment_version": RECIPE_ENV_VERSION,
        "loss_version": RECIPE_OPC_LOSS_VERSION,
        "reward_mode": "paper_raw",
        "seed": int(seed),
        "openilt_commit": env.solver.revision,
        "layout_sha256": env.solver.layout_sha256,
        "model_path": str(source),
        "model_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reward_weights": dict(env.reward_weights),
        "initial_weighted_loss": initial_loss,
        "displacement_classes_nm": list(DISPLACEMENT_CLASSES_NM),
        "paper_displacement_range_nm": [-40, 40],
        "action_semantics": "one_absolute_nine_class_decision_per_recipe_point",
        "patch_shape": [5, 64, 64],
        "vector_shape": [14],
        "point_count": len(labels),
        "epe_point_count": epe_count,
        "frag_point_count": frag_count,
        "best_step": int(rollout["best_step"]),
        "best_metrics": dict(rollout["best_metrics"]),
        "trajectory": list(rollout["trajectory"]),
        "best_internal_solver_trace": list(rollout["best_internal_solver_trace"]),
        "labels": labels,
        "frag_status": "implemented_as_recipe_segmentation_points",
    }


def run_default_recipe(env: RecipePointPPOEnv, seed: int = 0) -> Dict[str, Any]:
    """只评价零位移默认 recipe，作为 PPO 前的真实 solver 基线。"""
    _, reset_info = env.reset(seed=int(seed))
    evaluation = env.best_evaluation
    initial = env.trajectory[0]
    return {
        "environment_version": RECIPE_ENV_VERSION,
        "loss_version": RECIPE_OPC_LOSS_VERSION,
        "reward_mode": "paper_raw",
        "reset": reset_info,
        "best_step": 0,
        "best_metrics": evaluation.metrics.as_dict(),
        "best_recipe_offsets_nm": [0.0] * len(env.recipe_points),
        "trajectory": [initial],
        "internal_solver_trace": list(evaluation.internal_trace),
    }


def _valid_batch_size(rollout_size: int, requested: int) -> int:
    """选择不超过请求值且能整除 rollout 的 batch size。"""
    upper = min(int(requested), int(rollout_size))
    return next(value for value in range(upper, 0, -1) if rollout_size % value == 0)


def train_shared_recipe_ppo(
    train_envs: Sequence[RecipePointPPOEnv],
    output_path: Path,
    total_timesteps: int,
    seed: int,
    learning_rate: float = 3e-4,
    n_steps: int = 256,
    batch_size: int = 64,
):
    """在全部 train clip 上训练一个共享 CNN-PPO，并保存模型与训练元数据。"""
    import gymnasium
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    if stable_baselines3.__version__ != "2.0.0" or gymnasium.__version__ != "0.28.1":
        raise RuntimeError(
            "Recipe PPO 要求 stable-baselines3==2.0.0 与 gymnasium==0.28.1；"
            f"当前为 {stable_baselines3.__version__}/{gymnasium.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Recipe PPO 要求 CUDA；当前 torch.cuda.is_available() 为 False")
    environments = tuple(train_envs)
    if not environments:
        raise ValueError("共享 PPO 至少需要一个 train environment")
    if any(env.patch_size != 64 for env in environments):
        raise ValueError("共享 PPO 的所有环境都必须使用 64×64 局部图像")
    if int(total_timesteps) <= 0:
        raise ValueError("total_timesteps 必须大于零")
    n_steps = int(n_steps)
    if n_steps < 2:
        raise ValueError("ppo_n_steps 至少为 2")
    vec_env = DummyVecEnv([lambda env=env: env for env in environments])
    rollout_size = n_steps * len(environments)
    actual_batch_size = _valid_batch_size(rollout_size, int(batch_size))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    extractor = _feature_extractor_class()
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        device="cuda",
        seed=int(seed),
        learning_rate=float(learning_rate),
        n_steps=n_steps,
        batch_size=actual_batch_size,
        policy_kwargs={
            "features_extractor_class": extractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "normalize_images": False,
        },
        verbose=1,
    )
    model.learn(total_timesteps=int(total_timesteps))
    model.save(str(destination))
    model_path = destination if destination.suffix == ".zip" else Path(str(destination) + ".zip")
    metadata = {
        "environment_version": RECIPE_ENV_VERSION,
        "observation_version": RECIPE_OBSERVATION_VERSION,
        "point_version": RECIPE_POINT_VERSION,
        "loss_version": RECIPE_OPC_LOSS_VERSION,
        "label_version": PPO_RECIPE_LABEL_VERSION,
        "reward_mode": "paper_raw",
        "action_space": "Discrete(9)",
        "action_semantics": "one_absolute_nine_class_decision_per_recipe_point",
        "patch_shape": [5, 64, 64],
        "vector_shape": [14],
        "seed": int(seed),
        "total_timesteps_requested": int(total_timesteps),
        "total_timesteps_actual": int(model.num_timesteps),
        "n_envs": len(environments),
        "n_steps": n_steps,
        "batch_size": actual_batch_size,
        "learning_rate": float(learning_rate),
        "episode_horizons": [env.episode_horizon for env in environments],
        "complete_episode_covered_during_training": bool(
            int(total_timesteps) >= max(env.episode_horizon for env in environments)
        ),
        "train_layout_sha256": [env.solver.layout_sha256 for env in environments],
        "openilt_commits": sorted({env.solver.revision for env in environments}),
        "reward_weights": [dict(env.reward_weights) for env in environments],
        "displacement_classes_nm": list(DISPLACEMENT_CLASSES_NM),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_version": gymnasium.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "device": "cuda",
        "elapsed_seconds": time.monotonic() - started,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    model_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return model, model_path
