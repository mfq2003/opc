"""本模块实现不修改 OpenILT 上游仓库的 SimpleOPC 多步 PPO 环境。

输入为固定提交的 OpenILT、单个 ICCAD13 GLP 版图、分段参数和逐轮移动步长；输出为
Gymnasium 多步环境状态、一次批量边段动作对应的 L2/EPE/PVBand，以及可追溯的历史最优
Recipe。关键依赖为 NumPy、Gymnasium 和云端 OpenILT/PyTorch；模块只导入上游函数，所有
实验产物由调用方写入本项目 runs/，不会改写 OpenILT 源码或向其 tmp/ 写文件。
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .metrics import DISPLACEMENT_CLASSES_NM, SIMPLEOPC_LOSS_VERSION, weighted_opc_loss


SIMPLEOPC_ENV_VERSION = "simpleopc-multistep-v3"
SIMPLEOPC_ACTION_NAMES = ("inward", "stay", "outward")


@dataclass(frozen=True)
class SimpleOPCMetrics:
    """保存一次掩模仿真的三个论文目标指标。"""

    l2: float
    epe: float
    pvb: float

    def as_dict(self) -> Dict[str, float]:
        """返回可直接写入 JSON 的指标映射。"""
        return {"l2": float(self.l2), "epe": float(self.epe), "pvb": float(self.pvb)}


@dataclass(frozen=True)
class SimpleOPCSegment:
    """记录一个固定分段边的原始几何及指向版图外部的法向。"""

    segment_id: str
    polygon_index: int
    segment_index: int
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    normal_x: int
    normal_y: int
    length_nm: float

    @property
    def horizontal(self) -> bool:
        """判断边段是否为水平边。"""
        return self.start_y == self.end_y


@dataclass(frozen=True)
class SimpleOPCEvaluation:
    """保存一次全局仿真的指标、逐段 EPE 方向和掩模哈希。"""

    metrics: SimpleOPCMetrics
    epe_signs: np.ndarray
    mask_sha256: str


class SimpleOPCBackend(Protocol):
    """定义多步环境所需的最小 SimpleOPC 后端，便于 CPU 测试替换。"""

    segments: Sequence[SimpleOPCSegment]
    layout_sha256: str
    revision: str

    def reset(self) -> SimpleOPCEvaluation:
        """评价未移动的初始掩模。"""

    def evaluate(self, displacements_nm: np.ndarray) -> SimpleOPCEvaluation:
        """按每个边段相对原始位置的绝对法向位移评价完整掩模。"""


def _validate_reward_weights(reward_weights: Dict[str, float]) -> Dict[str, float]:
    """校验并规范化 L2/EPE/PVBand 权重。"""
    if set(reward_weights) != {"l2", "epe", "pvb"}:
        raise ValueError("reward_weights 必须恰好包含 l2、epe、pvb")
    result = {name: float(value) for name, value in reward_weights.items()}
    if any(value < 0 for value in result.values()) or sum(result.values()) <= 0:
        raise ValueError("reward_weights 必须非负且至少一个大于零")
    return result


class SimpleOPCMultiStepEnv(gym.Env):
    """每轮同时移动全部边段、每轮只做一次光刻仿真的多步 PPO 环境。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        backend: SimpleOPCBackend,
        reward_weights: Dict[str, float],
        step_sizes_nm: Sequence[float],
        displacement_limit_nm: float = 40.0,
        metric_epsilon: float = 1.0,
    ):
        super().__init__()
        self.backend = backend
        self.reward_weights = _validate_reward_weights(reward_weights)
        self.step_sizes_nm = tuple(float(value) for value in step_sizes_nm)
        if not self.step_sizes_nm or any(value <= 0 for value in self.step_sizes_nm):
            raise ValueError("step_sizes_nm 必须是非空正数序列")
        self.displacement_limit_nm = float(displacement_limit_nm)
        if self.displacement_limit_nm <= 0:
            raise ValueError("displacement_limit_nm 必须大于零")
        class_values = np.asarray(DISPLACEMENT_CLASSES_NM, dtype=np.float64)
        class_steps = np.diff(class_values)
        if not np.allclose(class_steps, class_steps[0]):
            raise RuntimeError("决策树位移代表值必须等间隔")
        if not np.isclose(self.displacement_limit_nm, np.max(np.abs(class_values))):
            raise ValueError("displacement_limit_nm 必须与九分类位移范围一致")
        grid_step = float(class_steps[0])
        ratios = np.asarray(self.step_sizes_nm, dtype=np.float64) / grid_step
        if not np.allclose(ratios, np.round(ratios)):
            raise ValueError("step_sizes_nm 必须是 10nm 位移代表值间隔的整数倍")
        self.metric_epsilon = float(metric_epsilon)
        if self.metric_epsilon <= 0:
            raise ValueError("metric_epsilon 必须大于零")
        self.segments = tuple(backend.segments)
        if not self.segments:
            raise ValueError("SimpleOPC 环境至少需要一个可移动边段")
        self.action_space = spaces.MultiDiscrete(
            np.full(len(self.segments), len(SIMPLEOPC_ACTION_NAMES), dtype=np.int64)
        )
        self._features_per_segment = 12
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.segments) * self._features_per_segment,),
            dtype=np.float32,
        )
        lengths = np.asarray([segment.length_nm for segment in self.segments], dtype=np.float64)
        self._length_scale = max(float(lengths.max()), self.metric_epsilon)
        self._step_index = 0
        self._displacements = np.zeros(len(self.segments), dtype=np.float64)
        self._initial: Optional[SimpleOPCEvaluation] = None
        self._current: Optional[SimpleOPCEvaluation] = None
        self._best: Optional[SimpleOPCEvaluation] = None
        self._best_displacements = self._displacements.copy()
        self._best_step = 0
        self._loss_scale = self.metric_epsilon
        self._trajectory = []

    def _raw_weighted_loss(self, metrics: SimpleOPCMetrics) -> float:
        """返回与论文公式及质量门槛完全一致的原始加权和。"""
        return weighted_opc_loss(metrics.as_dict(), self.reward_weights)

    def _loss(self, metrics: SimpleOPCMetrics) -> float:
        """将论文原始加权总损失整体除以初始总损失，只缩放数值而不改变相对权重。"""
        return float(self._raw_weighted_loss(metrics) / self._loss_scale)

    def _observation(self) -> np.ndarray:
        """拼接静态边段几何、当前位移/EPE 和全局指标形成固定长度状态。"""
        if self._current is None or self._initial is None:
            raise RuntimeError("环境必须先 reset")
        current_metrics = self._current.metrics.as_dict()
        initial_metrics = self._initial.metrics.as_dict()
        step_fraction = self._step_index / len(self.step_sizes_nm)
        rows = []
        for index, segment in enumerate(self.segments):
            rows.append([
                1.0 if segment.horizontal else 0.0,
                0.0 if segment.horizontal else 1.0,
                float(segment.normal_x),
                float(segment.normal_y),
                float(segment.length_nm / self._length_scale),
                float(self._displacements[index] / self.displacement_limit_nm),
                float(self._current.epe_signs[index]),
                float(step_fraction),
                float(current_metrics["l2"] / max(initial_metrics["l2"], self.metric_epsilon)),
                float(current_metrics["epe"] / max(initial_metrics["epe"], self.metric_epsilon)),
                float(current_metrics["pvb"] / max(initial_metrics["pvb"], self.metric_epsilon)),
                float(self._loss(self._current.metrics)),
            ])
        return np.asarray(rows, dtype=np.float32).reshape(-1)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """恢复原始分段掩模，并用初始原始加权总损失建立唯一缩放尺度。"""
        super().reset(seed=seed)
        self._step_index = 0
        self._displacements = np.zeros(len(self.segments), dtype=np.float64)
        self._initial = self.backend.reset()
        if self._initial.epe_signs.shape != (len(self.segments),):
            raise ValueError("后端 epe_signs 数量必须与边段数一致")
        self._current = self._initial
        self._best = self._initial
        self._best_displacements = self._displacements.copy()
        self._best_step = 0
        initial_weighted_loss = self._raw_weighted_loss(self._initial.metrics)
        self._loss_scale = max(initial_weighted_loss, self.metric_epsilon)
        self._trajectory = [{
            "step": 0,
            "step_size_nm": 0.0,
            "loss": self._loss(self._initial.metrics),
            "raw_weighted_loss": self._raw_weighted_loss(self._initial.metrics),
            "loss_version": SIMPLEOPC_LOSS_VERSION,
            "metrics": self._initial.metrics.as_dict(),
            "mask_sha256": self._initial.mask_sha256,
        }]
        return self._observation(), {
            "environment_version": SIMPLEOPC_ENV_VERSION,
            "loss_version": SIMPLEOPC_LOSS_VERSION,
            "initial_weighted_loss": initial_weighted_loss,
            "loss_scale": self._loss_scale,
            "segment_count": len(self.segments),
            "initial_metrics": self._initial.metrics.as_dict(),
        }

    def step(self, action: np.ndarray):
        """批量应用所有边段动作，持久化新掩模状态并返回非单步奖励。"""
        if self._current is None or self._initial is None:
            raise RuntimeError("环境必须先 reset")
        action_array = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(action_array):
            raise ValueError("动作必须为每个边段各一个 inward/stay/outward 类别")
        if self._step_index >= len(self.step_sizes_nm):
            raise RuntimeError("episode 已结束，请先 reset")
        previous_loss = self._loss(self._current.metrics)
        step_size = self.step_sizes_nm[self._step_index]
        directions = action_array.astype(np.float64) - 1.0
        self._displacements = np.clip(
            self._displacements + directions * step_size,
            -self.displacement_limit_nm,
            self.displacement_limit_nm,
        )
        evaluation = self.backend.evaluate(self._displacements.copy())
        if evaluation.epe_signs.shape != (len(self.segments),):
            raise ValueError("后端 epe_signs 数量必须与边段数一致")
        self._current = evaluation
        self._step_index += 1
        current_loss = self._loss(evaluation.metrics)
        if self._best is None or current_loss < self._loss(self._best.metrics):
            self._best = evaluation
            self._best_displacements = self._displacements.copy()
            self._best_step = self._step_index
        self._trajectory.append({
            "step": self._step_index,
            "step_size_nm": step_size,
            "loss": current_loss,
            "raw_weighted_loss": self._raw_weighted_loss(evaluation.metrics),
            "loss_version": SIMPLEOPC_LOSS_VERSION,
            "metrics": evaluation.metrics.as_dict(),
            "mask_sha256": evaluation.mask_sha256,
            "displacements_nm": self._displacements.tolist(),
        })
        terminated = self._step_index >= len(self.step_sizes_nm)
        info = {
            "step": self._step_index,
            "step_size_nm": step_size,
            "loss": current_loss,
            "raw_weighted_loss": self._raw_weighted_loss(evaluation.metrics),
            "loss_version": SIMPLEOPC_LOSS_VERSION,
            "loss_improvement": previous_loss - current_loss,
            "metrics": evaluation.metrics.as_dict(),
            "mask_sha256": evaluation.mask_sha256,
        }
        if terminated:
            info["best_step"] = self._best_step
            info["best_metrics"] = self.best_evaluation.metrics.as_dict()
            info["best_displacements_nm"] = self._best_displacements.tolist()
        return self._observation(), float(-current_loss), terminated, False, info

    @property
    def best_evaluation(self) -> SimpleOPCEvaluation:
        """返回当前 episode 历史加权损失最低的仿真结果。"""
        if self._best is None:
            raise RuntimeError("环境必须先 reset")
        return self._best

    @property
    def best_displacements_nm(self) -> np.ndarray:
        """返回历史最优 Recipe 的逐段绝对位移副本。"""
        if self._best is None:
            raise RuntimeError("环境必须先 reset")
        return self._best_displacements.copy()

    @property
    def best_step(self) -> int:
        """返回历史最优 Recipe 出现的迭代编号。"""
        if self._best is None:
            raise RuntimeError("环境必须先 reset")
        return int(self._best_step)

    @property
    def current_epe_signs(self) -> np.ndarray:
        """返回当前掩模逐段 EPE 建议方向的副本。"""
        if self._current is None:
            raise RuntimeError("环境必须先 reset")
        return self._current.epe_signs.copy()

    @property
    def initial_epe_signs(self) -> np.ndarray:
        """返回 episode 初始掩模的逐段 EPE 方向副本。"""
        if self._initial is None:
            raise RuntimeError("环境必须先 reset")
        return self._initial.epe_signs.copy()

    @property
    def trajectory(self) -> Tuple[dict, ...]:
        """返回不允许调用方原地修改的轨迹快照。"""
        return tuple(copy.deepcopy(self._trajectory))


class OpenILTSimpleOPCBackend:
    """直接复用 OpenILT 光刻与 polygon 函数的只读 SimpleOPC 后端。"""

    def __init__(
        self,
        openilt_dir: Path,
        expected_commit: str,
        layout_path: Path,
        lithography_config: Path = Path("config/lithosimple.txt"),
        simulator: str = "simple",
        image_size: Tuple[int, int] = (2048, 2048),
        openilt_scale: int = 1,
        nm_per_coordinate: float = 1.0,
        len_corner_nm: float = 16.0,
        len_uniform_nm: float = 32.0,
        epe_sample_distance_nm: float = 16.0,
        threshold: float = 0.5,
    ):
        import torch

        self.openilt_dir = Path(openilt_dir).resolve()
        self.layout_path = Path(layout_path).resolve()
        if not self.layout_path.is_file():
            raise FileNotFoundError(f"SimpleOPC GLP 不存在：{self.layout_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("SimpleOPC 多步环境要求 CUDA；请在云端 GPU 实例运行")
        if simulator not in {"simple", "exact"}:
            raise ValueError("simulator 必须是 simple 或 exact")
        self.openilt_scale = int(openilt_scale)
        if self.openilt_scale != 1:
            raise ValueError("simpleopc-multistep-v3 当前只支持 openilt_scale=1")
        self.nm_per_coordinate = float(nm_per_coordinate)
        if self.nm_per_coordinate <= 0:
            raise ValueError("nm_per_coordinate 必须大于零")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        if min(self.image_size) <= 0:
            raise ValueError("image_size 必须为正整数")
        self.epe_sample_distance_nm = float(epe_sample_distance_nm)
        if self.epe_sample_distance_nm <= 0:
            raise ValueError("epe_sample_distance_nm 必须大于零")
        self.threshold = float(threshold)
        self.revision = self._validate_openilt(expected_commit)
        config_path = Path(lithography_config)
        if not config_path.is_absolute():
            config_path = self.openilt_dir / config_path
        if not config_path.is_file():
            raise FileNotFoundError(f"OpenILT 光刻配置不存在：{config_path}")
        self._torch = torch
        openilt_text = str(self.openilt_dir)
        if openilt_text not in sys.path:
            sys.path.insert(0, openilt_text)
        previous_cwd = Path.cwd()
        try:
            os.chdir(str(self.openilt_dir))
            self._glp = importlib.import_module("pycommon.glp")
            self._polygon = importlib.import_module("utils.polygon")
            self._evaluation = importlib.import_module("pyilt.evaluation")
            lithosim = importlib.import_module(f"pylitho.{simulator}")
            self._litho = lithosim.LithoSim(str(config_path))
        finally:
            os.chdir(str(previous_cwd))
        self.layout_sha256 = hashlib.sha256(self.layout_path.read_bytes()).hexdigest()
        self._reference_by_polygon, self._reference_flat = self._load_segments(
            len_corner_nm=float(len_corner_nm), len_uniform_nm=float(len_uniform_nm)
        )
        self._target_array = self._render(np.zeros(len(self._reference_flat), dtype=np.float64))
        self._target = torch.as_tensor(self._target_array, dtype=torch.float32, device="cuda")
        self._vposes, self._hposes = self._evaluation.boundaries(self._target)
        normals = self._outward_normals(self._reference_flat, self._target_array)
        self.segments = tuple(
            self._segment_record(polygon_index, segment_index, flat_index, segment, normals[flat_index])
            for polygon_index, polygon_segments in enumerate(self._reference_by_polygon)
            for segment_index, (flat_index, segment) in enumerate(polygon_segments)
        )
        self._normals = np.asarray(
            [[segment.normal_x, segment.normal_y] for segment in self.segments], dtype=np.int32
        )
        self._cache: Dict[Tuple[int, ...], SimpleOPCEvaluation] = {}

    def _validate_openilt(self, expected_commit: str) -> str:
        """核验上游入口、提交号与已跟踪文件，不改写其工作树状态。"""
        required = (
            self.openilt_dir / "pyilt" / "simpleopc.py",
            self.openilt_dir / "utils" / "polygon.py",
        )
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"未找到完整 OpenILT SimpleOPC：{self.openilt_dir}")
        revision = subprocess.run(
            ["git", "-C", str(self.openilt_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != expected_commit:
            raise RuntimeError("OpenILT 提交与配置不一致；拒绝生成不可复现实验")
        diff = subprocess.run(
            ["git", "-C", str(self.openilt_dir), "diff", "--quiet", "HEAD", "--"],
            check=False,
        )
        if diff.returncode == 1:
            raise RuntimeError("OpenILT 已跟踪文件存在本地修改；请改用固定提交的干净 clone")
        if diff.returncode != 0:
            raise RuntimeError("无法核验 OpenILT 已跟踪文件是否干净")
        return revision

    def _load_segments(self, len_corner_nm: float, len_uniform_nm: float):
        """按上游 SimpleOPC 参数读取、居中并固定分段。"""
        len_corner = max(1, int(round(len_corner_nm / self.nm_per_coordinate)))
        len_uniform = max(1, int(round(len_uniform_nm / self.nm_per_coordinate)))
        design = self._glp.Design(str(self.layout_path), down=self.openilt_scale)
        design.center(self.image_size[0], self.image_size[1], 0, 0)
        polygons = []
        for polygon in design.polygons:
            polygons.append([[int(point[0]), int(point[1])] for point in polygon])
        by_polygon = []
        flat = []
        flat_index = 0
        for polygon in polygons:
            dissected = self._polygon.dissect(
                polygon, lenCorner=len_corner, lenUniform=len_uniform
            )
            records = []
            for segment in dissected:
                normalized = (
                    (int(segment[0][0]), int(segment[0][1])),
                    (int(segment[1][0]), int(segment[1][1])),
                )
                records.append((flat_index, normalized))
                flat.append(normalized)
                flat_index += 1
            by_polygon.append(records)
        if not flat:
            raise RuntimeError(f"版图没有可分段正交边：{self.layout_path}")
        return by_polygon, tuple(flat)

    def _outward_normals(self, segments, target: np.ndarray) -> np.ndarray:
        """通过边段两侧目标像素确定唯一外法向，拒绝含糊或越界边。"""
        result = []
        probe = 2
        height, width = target.shape
        binary = target >= self.threshold
        for index, segment in enumerate(segments):
            start, end = segment
            mid_x = int(round((start[0] + end[0]) / 2))
            mid_y = int(round((start[1] + end[1]) / 2))
            if start[0] == end[0]:
                samples = ((mid_x + probe, mid_y), (mid_x - probe, mid_y))
                candidates = ((-1, 0), (1, 0))
            elif start[1] == end[1]:
                samples = ((mid_x, mid_y + probe), (mid_x, mid_y - probe))
                candidates = ((0, -1), (0, 1))
            else:
                raise ValueError(f"边段 {index} 不是正交边")
            if any(x < 0 or x >= width or y < 0 or y >= height for x, y in samples):
                raise ValueError(f"边段 {index} 法向探针越出图像")
            first_inside = bool(binary[samples[0][1], samples[0][0]])
            second_inside = bool(binary[samples[1][1], samples[1][0]])
            if first_inside == second_inside:
                raise ValueError(f"边段 {index} 无法从目标掩模确定唯一外法向")
            result.append(candidates[0] if first_inside else candidates[1])
        return np.asarray(result, dtype=np.int32)

    def _segment_record(self, polygon_index, segment_index, flat_index, segment, normal):
        """构造可写入 Recipe 的稳定边段记录。"""
        start, end = segment
        length_coord = abs(start[0] - end[0]) + abs(start[1] - end[1])
        return SimpleOPCSegment(
            segment_id=f"p{polygon_index}:s{segment_index}:f{flat_index}",
            polygon_index=int(polygon_index),
            segment_index=int(segment_index),
            start_x=int(start[0]),
            start_y=int(start[1]),
            end_x=int(end[0]),
            end_y=int(end[1]),
            normal_x=int(normal[0]),
            normal_y=int(normal[1]),
            length_nm=float(length_coord * self.nm_per_coordinate),
        )

    def _render(self, displacements_nm: np.ndarray) -> np.ndarray:
        """移动完整边段并用上游 segs2poly 法化转角后栅格化。"""
        values = np.asarray(displacements_nm, dtype=np.float64)
        if values.shape != (len(self._reference_flat),):
            raise ValueError("逐段位移数量与 SimpleOPC 分段数不一致")
        moved_flat = []
        normals = getattr(self, "_normals", None)
        if normals is None:
            normals = self._outward_normals(self._reference_flat, self._render_reference_only())
        for index, segment in enumerate(self._reference_flat):
            delta = int(round(values[index] / self.nm_per_coordinate))
            dx = int(normals[index][0]) * delta
            dy = int(normals[index][1]) * delta
            moved_flat.append([
                (segment[0][0] + dx, segment[0][1] + dy),
                (segment[1][0] + dx, segment[1][1] + dy),
            ])
        polygons = []
        for records in self._reference_by_polygon:
            polygons.append(self._polygon.segs2poly([moved_flat[index] for index, _ in records]))
        image = self._polygon.poly2img(
            polygons, self.image_size[0], self.image_size[1], scale=self.openilt_scale
        )
        return np.asarray(image, dtype=np.float32) / 255.0

    def _render_reference_only(self) -> np.ndarray:
        """在外法向尚未建立时栅格化零位移参考多边形。"""
        polygons = []
        for records in self._reference_by_polygon:
            polygons.append(self._polygon.segs2poly([list(segment) for _, segment in records]))
        image = self._polygon.poly2img(
            polygons, self.image_size[0], self.image_size[1], scale=self.openilt_scale
        )
        return np.asarray(image, dtype=np.float32) / 255.0

    @staticmethod
    def _number(value) -> float:
        """把上游 Python 数值或单元素张量统一转为 float。"""
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return float(value)

    def _segment_epe_signs(self, binary_nominal: np.ndarray) -> np.ndarray:
        """按 SimpleOPC 内外采样定义返回 +1 外移、-1 内移、0 无动作。"""
        target = self._target_array >= self.threshold
        printed = np.asarray(binary_nominal, dtype=bool)
        height, width = target.shape
        distance = int(round(self.epe_sample_distance_nm / self.nm_per_coordinate))
        signs = np.zeros(len(self.segments), dtype=np.float32)
        for index, segment in enumerate(self.segments):
            mid_x = int(round((segment.start_x + segment.end_x) / 2))
            mid_y = int(round((segment.start_y + segment.end_y) / 2))
            inner_x = mid_x - segment.normal_x * distance
            inner_y = mid_y - segment.normal_y * distance
            outer_x = mid_x + segment.normal_x * distance
            outer_y = mid_y + segment.normal_y * distance
            if not (0 <= inner_x < width and 0 <= inner_y < height):
                continue
            if not (0 <= outer_x < width and 0 <= outer_y < height):
                continue
            if target[inner_y, inner_x] and not printed[inner_y, inner_x]:
                signs[index] = 1.0
            if not target[outer_y, outer_x] and printed[outer_y, outer_x]:
                signs[index] = -1.0
        return signs

    def reset(self) -> SimpleOPCEvaluation:
        """评价目标多边形直接作为初始掩模的状态。"""
        return self.evaluate(np.zeros(len(self.segments), dtype=np.float64))

    def evaluate(self, displacements_nm: np.ndarray) -> SimpleOPCEvaluation:
        """一次光刻仿真同时计算 L2、EPE、PVBand 和逐段 EPE 方向。"""
        values = np.asarray(displacements_nm, dtype=np.float64)
        if values.shape != (len(self.segments),):
            raise ValueError("逐段位移数量与 SimpleOPC 分段数不一致")
        key = tuple(int(round(value * 1000.0)) for value in values)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        mask_array = self._render(values)
        mask_sha256 = hashlib.sha256(np.ascontiguousarray(mask_array).tobytes()).hexdigest()
        mask = self._torch.as_tensor(mask_array, dtype=self._torch.float32, device="cuda")
        with self._torch.no_grad():
            printed_nom, printed_max, printed_min = self._litho(mask)
            binary_nom = (printed_nom >= self.threshold).to(dtype=self._torch.float32)
            binary_max = printed_max >= self.threshold
            binary_min = printed_min >= self.threshold
            l2 = self._torch.nn.functional.mse_loss(binary_nom, self._target, reduction="sum")
            pvb = self._torch.sum(binary_max != binary_min)
            epe_in, epe_out, _ = self._evaluation.epecheck(
                binary_nom, self._target, self._vposes, self._hposes
            )
        evaluation = SimpleOPCEvaluation(
            metrics=SimpleOPCMetrics(
                l2=self._number(l2),
                epe=self._number(epe_in) + self._number(epe_out),
                pvb=self._number(pvb),
            ),
            epe_signs=self._segment_epe_signs(binary_nom.detach().cpu().numpy()),
            mask_sha256=mask_sha256,
        )
        self._cache[key] = evaluation
        return evaluation
