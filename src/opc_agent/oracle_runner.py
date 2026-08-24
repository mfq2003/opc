"""本模块在云端 CUDA 环境中训练逐点 PPO Oracle，并用 OpenILT 评价九个候选掩模。

输入为 NPZ 点数据（observations、target、candidate_masks）、固定提交的 OpenILT 目录和训练参数；输出为
Stable-Baselines3 PPO 模型、逐点动作指标缓存和训练元数据。关键依赖为 NumPy、Gymnasium、PyTorch、
Stable-Baselines3 与 OpenILT；模块不会下载数据或修改 OpenILT，上游点移动几何未公开，因此候选掩模必须由
独立预处理阶段明确提供，不能在这里猜测生成。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .metrics import DISPLACEMENT_CLASSES_NM


@dataclass(frozen=True)
class CandidatePointSet:
    """保存同一 clip 的状态、目标图和完整或紧凑候选描述。"""

    observations: np.ndarray
    target: np.ndarray
    source_sha256: str
    candidate_masks: Optional[np.ndarray] = None
    base_mask: Optional[np.ndarray] = None
    point_geometry: Optional[np.ndarray] = None
    scale_nm_per_pixel: Optional[float] = None
    adapter_version: str = "raster-fragment-v1"

    @property
    def point_count(self) -> int:
        return int(self.observations.shape[0])

    def mask_for(self, point_index: int, action_index: int) -> np.ndarray:
        """兼容完整候选数组；正式紧凑格式按点参数即时生成一个候选。"""
        if self.candidate_masks is not None:
            return self.candidate_masks[point_index, action_index]
        if self.base_mask is None or self.point_geometry is None or self.scale_nm_per_pixel is None:
            raise RuntimeError("候选数据既没有完整掩模，也没有紧凑点几何")
        from .candidate_masks import FragmentPoint, generate_candidate_mask

        geometry = [int(value) for value in self.point_geometry[point_index]]
        x, y, normal_x, normal_y, radius = geometry[:5]
        segment = {}
        if len(geometry) == 9:
            segment = dict(zip(
                ("segment_start_x", "segment_start_y", "segment_end_x", "segment_end_y"),
                geometry[5:],
            ))
        point = FragmentPoint(
            point_id=str(point_index), task_type="EPE", x=x, y=y,
            normal_x=normal_x, normal_y=normal_y, support_radius_px=radius,
            **segment,
        )
        displacement_px = int(round(DISPLACEMENT_CLASSES_NM[action_index] / self.scale_nm_per_pixel))
        return generate_candidate_mask(self.base_mask, point, displacement_px, self.adapter_version)


class PointMetricEvaluator(Protocol):
    """定义环境所需的点动作评价接口，便于测试和真实 OpenILT 后端复用。"""

    def evaluate(self, point_index: int, action_index: int) -> Tuple[float, float, float]:
        """返回动作对应的 L2、总 EPE 和 PVBand。"""


def load_candidate_point_set(path: Path) -> CandidatePointSet:
    """读取无 pickle 的 NPZ，并校验完整或节省磁盘的紧凑候选格式。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"候选掩模数据不存在：{source}")
    with np.load(str(source), allow_pickle=False) as payload:
        required = {"observations", "target"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"候选掩模 NPZ 缺少字段：{sorted(missing)}")
        observations = np.asarray(payload["observations"], dtype=np.float32)
        target = np.asarray(payload["target"], dtype=np.float32)
        candidate_masks = np.asarray(payload["candidate_masks"], dtype=np.float32) if "candidate_masks" in payload.files else None
        base_mask = np.asarray(payload["base_mask"], dtype=np.float32) if "base_mask" in payload.files else None
        point_geometry = np.asarray(payload["point_geometry"], dtype=np.int32) if "point_geometry" in payload.files else None
        scale_nm_per_pixel = float(payload["scale_nm_per_pixel"]) if "scale_nm_per_pixel" in payload.files else None
        adapter_version = (
            str(payload["adapter_version"].item())
            if "adapter_version" in payload.files else "raster-fragment-v1"
        )
    if observations.ndim != 2 or observations.shape[0] == 0 or observations.shape[1] == 0:
        raise ValueError("observations 必须是非空的 [point, feature] 二维数组")
    if target.ndim != 2:
        raise ValueError("target 必须是 [height, width] 二维数组")
    expected = (observations.shape[0], len(DISPLACEMENT_CLASSES_NM), target.shape[0], target.shape[1])
    if candidate_masks is not None and candidate_masks.shape != expected:
        raise ValueError(f"candidate_masks 应为 {expected}，实际为 {candidate_masks.shape}")
    expected_geometry_width = 9 if adapter_version == "raster-edge-segment-v3" else 5
    compact_valid = (
        base_mask is not None and base_mask.shape == target.shape
        and point_geometry is not None
        and point_geometry.shape == (observations.shape[0], expected_geometry_width)
        and scale_nm_per_pixel is not None and scale_nm_per_pixel > 0
    )
    if candidate_masks is None and not compact_valid:
        raise ValueError(
            f"紧凑候选格式必须包含 base_mask、[point,{expected_geometry_width}] point_geometry 和正比例"
        )
    if adapter_version not in {
        "raster-fragment-v1", "raster-boundary-strip-v2", "raster-edge-segment-v3"
    }:
        raise ValueError(f"未知候选适配器版本：{adapter_version}")
    if not np.isfinite(observations).all():
        raise ValueError("observations 含 NaN 或无穷值")
    if np.any((target < 0) | (target > 1)):
        raise ValueError("target 必须归一化到 [0, 1]")
    if candidate_masks is not None and np.any((candidate_masks < 0) | (candidate_masks > 1)):
        raise ValueError("candidate_masks 必须归一化到 [0, 1]")
    if base_mask is not None and np.any((base_mask < 0) | (base_mask > 1)):
        raise ValueError("base_mask 必须归一化到 [0, 1]")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return CandidatePointSet(
        observations=observations, target=target, source_sha256=digest,
        candidate_masks=candidate_masks, base_mask=base_mask,
        point_geometry=point_geometry, scale_nm_per_pixel=scale_nm_per_pixel,
        adapter_version=adapter_version,
    )

class CandidatePointEnv(gym.Env):
    """将一个 clip 的逐点九分类选择包装为单步 PPO 环境。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: CandidatePointSet,
        evaluator: PointMetricEvaluator,
        reward_weights: Dict[str, float],
    ):
        super().__init__()
        if set(reward_weights) != {"l2", "epe", "pvb"}:
            raise ValueError("reward_weights 必须恰好包含 l2、epe、pvb")
        if any(float(value) < 0 for value in reward_weights.values()):
            raise ValueError("reward_weights 不能为负数")
        self.dataset = dataset
        self.evaluator = evaluator
        self.reward_weights = {name: float(value) for name, value in reward_weights.items()}
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(dataset.observations.shape[1],),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(DISPLACEMENT_CLASSES_NM))
        self._point_index = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """按 Gymnasium 随机种子选择一个点，保证同种子可复现。"""
        super().reset(seed=seed)
        self._point_index = int(self.np_random.integers(0, self.dataset.point_count))
        return self.dataset.observations[self._point_index].copy(), {"point_index": self._point_index}

    def step(self, action: int):
        """调用真实评价器并按论文权重返回单步奖励。"""
        if not self.action_space.contains(action):
            raise ValueError("动作不在九分类位移集合中")
        l2, epe, pvb = self.evaluator.evaluate(self._point_index, int(action))
        reward = -(
            self.reward_weights["l2"] * l2
            + self.reward_weights["epe"] * epe
            + self.reward_weights["pvb"] * pvb
        )
        info = {
            "point_index": self._point_index,
            "action_index": int(action),
            "displacement_nm": DISPLACEMENT_CLASSES_NM[int(action)],
            "l2": float(l2),
            "epe": float(epe),
            "pvb": float(pvb),
        }
        return self.dataset.observations[self._point_index].copy(), float(reward), True, False, info


class OpenILTCandidateEvaluator:
    """在 CUDA 上调用固定提交 OpenILT 的 Basic 与 EPEChecker，并缓存昂贵评价。"""

    def __init__(
        self,
        dataset: CandidatePointSet,
        openilt_dir: Path,
        expected_commit: str,
        lithography_config: Path,
        cache_path: Path,
        simulator: str = "simple",
        scale: int = 1,
        threshold: float = 0.5,
    ):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("OpenILT Oracle 要求 CUDA；请在 GPU 实例中运行")
        self.dataset = dataset
        self.openilt_dir = Path(openilt_dir).resolve()
        self.cache_path = Path(cache_path)
        self.scale = int(scale)
        if self.scale <= 0:
            raise ValueError("OpenILT scale 必须为正整数")
        revision = subprocess.run(
            ["git", "-C", str(self.openilt_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != expected_commit:
            raise RuntimeError("OpenILT 提交与配置不一致；拒绝训练不可复现 Oracle")
        config_path = Path(lithography_config)
        if not config_path.is_absolute():
            config_path = self.openilt_dir / config_path
        if not config_path.is_file():
            raise FileNotFoundError(f"OpenILT 光刻配置不存在：{config_path}")
        if simulator not in {"simple", "exact"}:
            raise ValueError("OpenILT simulator 必须是 simple 或 exact")
        openilt_text = str(self.openilt_dir)
        if openilt_text not in sys.path:
            sys.path.insert(0, openilt_text)
        self._torch = torch
        previous_cwd = Path.cwd()
        try:
            os.chdir(str(self.openilt_dir))
            lithosim = importlib.import_module(f"pylitho.{simulator}")
            evaluation = importlib.import_module("pyilt.evaluation")
            self._litho = lithosim.LithoSim(str(config_path))
            self._basic = evaluation.Basic(self._litho, float(threshold))
            self._epe = evaluation.EPEChecker(self._litho, float(threshold))
        finally:
            os.chdir(str(previous_cwd))
        self._target = torch.as_tensor(dataset.target, dtype=torch.float32, device="cuda")
        self._cache = self._read_cache()

    def _read_cache(self) -> Dict[str, Dict[str, float]]:
        """只接收与当前候选数据哈希一致的缓存。"""
        if not self.cache_path.is_file():
            return {}
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        if payload.get("source_sha256") != self.dataset.source_sha256:
            raise RuntimeError("Oracle 指标缓存来自不同候选数据；拒绝混用")
        return dict(payload.get("metrics", {}))

    def _write_cache(self) -> None:
        """原子替换缓存文件，降低训练中断造成 JSON 损坏的风险。"""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        payload = {"source_sha256": self.dataset.source_sha256, "metrics": self._cache}
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.cache_path)

    @staticmethod
    def _number(value) -> float:
        """把上游返回的 Python 数值或单元素张量统一转换为 float。"""
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return float(value)

    def evaluate(self, point_index: int, action_index: int) -> Tuple[float, float, float]:
        """评价一个候选掩模；命中缓存时不重复调用 OpenILT。"""
        if not 0 <= point_index < self.dataset.point_count:
            raise IndexError("point_index 越界")
        if not 0 <= action_index < len(DISPLACEMENT_CLASSES_NM):
            raise IndexError("action_index 越界")
        key = f"{point_index}:{action_index}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached["l2"], cached["epe"], cached["pvb"]
        mask_array = self.dataset.mask_for(point_index, action_index)
        mask_sha256 = hashlib.sha256(np.ascontiguousarray(mask_array).tobytes()).hexdigest()
        mask = self._torch.as_tensor(
            mask_array, dtype=self._torch.float32, device="cuda"
        )
        with self._torch.no_grad():
            l2, pvb = self._basic.run(mask, self._target, scale=self.scale)
            epe_in, epe_out = self._epe.run(mask, self._target, scale=self.scale)
        metrics = {
            "l2": self._number(l2),
            "epe": self._number(epe_in) + self._number(epe_out),
            "pvb": self._number(pvb),
            "mask_sha256": mask_sha256,
        }
        self._cache[key] = metrics
        self._write_cache()
        return metrics["l2"], metrics["epe"], metrics["pvb"]


def complete_metric_cache(dataset: CandidatePointSet, evaluator: PointMetricEvaluator) -> int:
    """确保每个点的九个动作都有真实指标；评价器自行跳过已缓存项。"""
    total = dataset.point_count * len(DISPLACEMENT_CLASSES_NM)
    completed = 0
    for point_index in range(dataset.point_count):
        for action_index in range(len(DISPLACEMENT_CLASSES_NM)):
            evaluator.evaluate(point_index, action_index)
            completed += 1
            if completed == total or completed % 25 == 0:
                print(f"Oracle 指标完整性：{completed}/{total}", flush=True)
    return completed


def ppo_rollout_schedule(total_timesteps: int) -> Tuple[int, int]:
    """选择能整除总步数的 rollout/batch，避免 SB3 静默向上补齐 timestep。"""
    if total_timesteps < 2:
        raise ValueError("PPO total_timesteps 至少为 2")
    upper = min(2048, int(total_timesteps))
    n_steps = next((value for value in range(upper, 1, -1) if total_timesteps % value == 0), upper)
    batch_upper = min(64, n_steps)
    batch_size = next(value for value in range(batch_upper, 0, -1) if n_steps % value == 0)
    return n_steps, batch_size

def train_ppo(
    env: CandidatePointEnv,
    output_path: Path,
    total_timesteps: int,
    seed: int,
    learning_rate: float = 3e-4,
) -> Path:
    """强制使用 CUDA 训练 PPO，并保存模型和最小可审计元数据。"""
    import torch
    from stable_baselines3 import PPO

    if not torch.cuda.is_available():
        raise RuntimeError("PPO Oracle 配置为 GPU 模式，但当前 torch.cuda.is_available() 为 False")
    if total_timesteps < 2:
        raise ValueError("total_timesteps 至少为 2")
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
    metadata = {
        "seed": int(seed),
        "total_timesteps_requested": int(total_timesteps),
        "total_timesteps_actual": int(model.num_timesteps),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "device": "cuda",
        "elapsed_seconds": time.monotonic() - started,
        "candidate_source_sha256": env.dataset.source_sha256,
    }
    model_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return model_path


def main(argv: Optional[List[str]] = None) -> int:
    """从命令行启动单 clip 的 CUDA PPO 冒烟或正式训练。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.oracle_runner")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--openilt-dir", type=Path, required=True)
    parser.add_argument("--openilt-commit", required=True)
    parser.add_argument("--simulator", choices=("simple", "exact"), default="simple")
    parser.add_argument("--lithography-config", type=Path, default=Path("config/lithosimple.txt"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    dataset = load_candidate_point_set(args.dataset)
    evaluator = OpenILTCandidateEvaluator(
        dataset=dataset,
        openilt_dir=args.openilt_dir,
        expected_commit=args.openilt_commit,
        lithography_config=args.lithography_config,
        cache_path=args.cache,
        simulator=args.simulator,
    )
    env = CandidatePointEnv(dataset, evaluator, {"l2": 1.0, "epe": 100.0, "pvb": 1.0})
    path = train_ppo(env, args.output, args.timesteps, args.seed)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())










