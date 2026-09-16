"""本模块实现论文主线的点级 OPC Recipe PPO 环境与只读 OpenILT 求解器。

输入为原始 target 多边形、EPE/FRAG recipe 点、九档绝对位移和固定 OpenILT 光刻模型；输出为
以单个 recipe 点为中心的 64×64 多通道 observation、一次九分类动作、solver 内部优化后的 mask
及固定评价点上的 L2/EPE/PVBand。PPO 只能修改 recipe 点，任何 mask 边段位移都封装在 solver
内部；模块不修改 OpenILT 源码，也不向 OpenILT 工作树写实验产物。
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import os
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .metrics import DISPLACEMENT_CLASSES_NM, RECIPE_OPC_LOSS_VERSION, weighted_opc_loss
from .recipe_contract import (
    RECIPE_ENV_VERSION,
    RECIPE_OBSERVATION_VERSION,
    RECIPE_POINT_VERSION,
)
from .simpleopc import SimpleOPCMetrics


@dataclass(frozen=True)
class RecipePoint:
    """记录绑定到原始 target 边的稳定 EPE 或 FRAG recipe 点。"""

    point_id: str
    task_type: str
    polygon_index: int
    edge_index: int
    anchor_index: int
    base_x: int
    base_y: int
    tangent_x: int
    tangent_y: int
    normal_x: int
    normal_y: int
    lower_delta_nm: float
    upper_delta_nm: float

    def __post_init__(self) -> None:
        if self.task_type not in {"EPE", "FRAG"}:
            raise ValueError("Recipe point task_type 必须是 EPE 或 FRAG")


@dataclass(frozen=True)
class RecipeEvaluation:
    """保存 solver 最佳内部 mask、固定评价指标和生成 observation 所需图像。"""

    metrics: SimpleOPCMetrics
    recipe_epe_signs: np.ndarray
    mask_sha256: str
    target_image: np.ndarray
    mask_image: np.ndarray
    printed_image: np.ndarray
    internal_trace: Tuple[dict, ...]


class RecipeSolver(Protocol):
    """定义点级 PPO 所需的 recipe-aware solver 最小接口。"""

    recipe_points: Sequence[RecipePoint]
    layout_sha256: str
    revision: str
    image_size: Tuple[int, int]

    def solve(self, recipe_offsets_nm: np.ndarray) -> RecipeEvaluation:
        """根据完整 recipe 在内部优化 mask，并用固定评价点返回结果。"""

    def local_image_stack(
        self,
        point_index: int,
        recipe_offsets_nm: np.ndarray,
        evaluation: RecipeEvaluation,
        patch_size: int,
    ) -> np.ndarray:
        """返回当前 recipe 点周围的固定尺寸多通道图像。"""


def _validate_reward_weights(reward_weights: Dict[str, float]) -> Dict[str, float]:
    """校验并规范化论文 L2/EPE/PVBand 权重。"""
    if set(reward_weights) != {"l2", "epe", "pvb"}:
        raise ValueError("reward_weights 必须恰好包含 l2、epe、pvb")
    result = {name: float(value) for name, value in reward_weights.items()}
    if any(value < 0 for value in result.values()) or sum(result.values()) <= 0:
        raise ValueError("reward_weights 必须非负且至少一个大于零")
    return result


def _center_crop(image: np.ndarray, center_x: int, center_y: int, size: int) -> np.ndarray:
    """以零填充方式截取以指定像素为中心的正方形图像。"""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("局部图像裁剪只接受二维数组")
    if size <= 0 or size % 2 != 0:
        raise ValueError("patch_size 必须是正偶数")
    radius = size // 2
    output = np.zeros((size, size), dtype=np.float32)
    source_x0 = max(0, int(center_x) - radius)
    source_y0 = max(0, int(center_y) - radius)
    source_x1 = min(array.shape[1], int(center_x) + radius)
    source_y1 = min(array.shape[0], int(center_y) + radius)
    target_x0 = source_x0 - (int(center_x) - radius)
    target_y0 = source_y0 - (int(center_y) - radius)
    target_x1 = target_x0 + (source_x1 - source_x0)
    target_y1 = target_y0 + (source_y1 - source_y0)
    if source_x1 > source_x0 and source_y1 > source_y0:
        output[target_y0:target_y1, target_x0:target_x1] = array[
            source_y0:source_y1, source_x0:source_x1
        ]
    return output


class RecipePointPPOEnv(gym.Env):
    """每个 recipe 点仅决策一次、直接选择九档绝对位移的 PPO 环境。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        solver: RecipeSolver,
        reward_weights: Dict[str, float],
        displacement_classes_nm: Sequence[float] = DISPLACEMENT_CLASSES_NM,
        patch_size: int = 64,
        shuffle_points: bool = True,
        metric_epsilon: float = 1.0,
    ):
        super().__init__()
        self.solver = solver
        self.reward_weights = _validate_reward_weights(reward_weights)
        self.displacement_classes_nm = tuple(float(value) for value in displacement_classes_nm)
        if self.displacement_classes_nm != tuple(float(value) for value in DISPLACEMENT_CLASSES_NM):
            raise ValueError("Recipe PPO 动作必须是 -40 到 40nm 的九个等距类别")
        self.patch_size = int(patch_size)
        if self.patch_size != 64:
            raise ValueError("论文主线当前固定使用 64×64 像素局部图像")
        self.shuffle_points = bool(shuffle_points)
        self.metric_epsilon = float(metric_epsilon)
        if self.metric_epsilon <= 0:
            raise ValueError("metric_epsilon 必须大于零")
        self.recipe_points = tuple(solver.recipe_points)
        if not self.recipe_points:
            raise ValueError("Recipe PPO 至少需要一个 EPE 或 FRAG 点")
        if {point.task_type for point in self.recipe_points} != {"EPE", "FRAG"}:
            raise ValueError("论文主线必须同时具有 EPE 与 FRAG recipe 点")
        for point in self.recipe_points:
            if point.lower_delta_nm > -40 or point.upper_delta_nm < 40:
                raise ValueError(f"Recipe 点 {point.point_id} 不能完整支持 ±40nm 九分类动作")
        self.action_space = spaces.Discrete(len(self.displacement_classes_nm))
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(5, self.patch_size, self.patch_size),
                dtype=np.float32,
            ),
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32),
        })
        self._schedule = np.arange(len(self.recipe_points), dtype=np.int64)
        self._cursor = 0
        self._offsets = np.zeros(len(self.recipe_points), dtype=np.float64)
        self._initial: Optional[RecipeEvaluation] = None
        self._current: Optional[RecipeEvaluation] = None
        self._best: Optional[RecipeEvaluation] = None
        self._best_offsets = self._offsets.copy()
        self._best_step = 0
        self._trajectory: List[dict] = []

    @property
    def episode_horizon(self) -> int:
        """返回一个 episode 中需要各决策一次的 recipe 点数量。"""
        return len(self.recipe_points)

    def _raw_loss(self, metrics: SimpleOPCMetrics) -> float:
        """返回论文明确给出的原始加权 OPC loss。"""
        return weighted_opc_loss(metrics.as_dict(), self.reward_weights)

    def _active_index(self) -> int:
        """返回当前待决策点在完整 recipe 中的索引。"""
        if self._cursor >= len(self._schedule):
            return int(self._schedule[-1])
        return int(self._schedule[self._cursor])

    def _observation(self) -> Dict[str, np.ndarray]:
        """构造当前点的 64×64 CNN 图像和轻量几何/物理向量。"""
        if self._current is None or self._initial is None:
            raise RuntimeError("环境必须先 reset")
        point_index = self._active_index()
        point = self.recipe_points[point_index]
        image = self.solver.local_image_stack(
            point_index,
            self._offsets,
            self._current,
            self.patch_size,
        )
        if image.shape != (5, self.patch_size, self.patch_size):
            raise ValueError("solver 返回的局部图像必须是 5×64×64")
        initial_loss = max(self._raw_loss(self._initial.metrics), self.metric_epsilon)
        signs = np.asarray(self._current.recipe_epe_signs, dtype=np.float32)
        if signs.shape != (len(self.recipe_points),):
            raise ValueError("solver recipe_epe_signs 数量与 recipe 点数不一致")
        vector = np.asarray([
            1.0 if point.task_type == "EPE" else 0.0,
            1.0 if point.task_type == "FRAG" else 0.0,
            float(point.base_x / max(self.solver.image_size[0] - 1, 1)),
            float(point.base_y / max(self.solver.image_size[1] - 1, 1)),
            float(point.tangent_x),
            float(point.tangent_y),
            float(point.normal_x),
            float(point.normal_y),
            float(point.lower_delta_nm / 40.0),
            float(point.upper_delta_nm / 40.0),
            float(self._offsets[point_index] / 40.0),
            float(self._cursor / max(len(self._schedule), 1)),
            float(self._raw_loss(self._current.metrics) / initial_loss),
            float(signs[point_index]),
        ], dtype=np.float32)
        return {"image": image.astype(np.float32, copy=False), "vector": vector}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """恢复默认 recipe，并为本 episode 安排每个点恰好一次的决策顺序。"""
        super().reset(seed=seed)
        self._cursor = 0
        self._offsets = np.zeros(len(self.recipe_points), dtype=np.float64)
        self._schedule = np.arange(len(self.recipe_points), dtype=np.int64)
        if self.shuffle_points:
            self.np_random.shuffle(self._schedule)
        self._initial = self.solver.solve(self._offsets.copy())
        self._current = self._initial
        self._best = self._initial
        self._best_offsets = self._offsets.copy()
        self._best_step = 0
        initial_loss = self._raw_loss(self._initial.metrics)
        self._trajectory = [{
            "step": 0,
            "point_index": None,
            "point_id": None,
            "task_type": None,
            "action_class": 4,
            "displacement_nm": 0.0,
            "loss": initial_loss,
            "raw_weighted_loss": initial_loss,
            "loss_version": RECIPE_OPC_LOSS_VERSION,
            "metrics": self._initial.metrics.as_dict(),
            "mask_sha256": self._initial.mask_sha256,
        }]
        epe_count = sum(point.task_type == "EPE" for point in self.recipe_points)
        frag_count = len(self.recipe_points) - epe_count
        return self._observation(), {
            "environment_version": RECIPE_ENV_VERSION,
            "observation_version": RECIPE_OBSERVATION_VERSION,
            "point_version": RECIPE_POINT_VERSION,
            "loss_version": RECIPE_OPC_LOSS_VERSION,
            "reward_mode": "paper_raw",
            "point_count": len(self.recipe_points),
            "epe_point_count": epe_count,
            "frag_point_count": frag_count,
            "episode_horizon": len(self.recipe_points),
            "initial_weighted_loss": initial_loss,
            "initial_metrics": self._initial.metrics.as_dict(),
        }

    def step(self, action: int):
        """为当前点直接设置一个九档绝对位移，然后让 solver 内部重新优化 mask。"""
        if self._current is None or self._initial is None:
            raise RuntimeError("环境必须先 reset")
        if self._cursor >= len(self._schedule):
            raise RuntimeError("episode 已结束，请先 reset")
        action_value = int(np.asarray(action).item())
        if not self.action_space.contains(action_value):
            raise ValueError("动作必须是 0 到 8 的单个九分类编号")
        point_index = self._active_index()
        point = self.recipe_points[point_index]
        displacement = self.displacement_classes_nm[action_value]
        if displacement < point.lower_delta_nm or displacement > point.upper_delta_nm:
            raise ValueError(f"动作超出 recipe 点 {point.point_id} 的合法位移范围")
        self._offsets[point_index] = displacement
        evaluation = self.solver.solve(self._offsets.copy())
        self._current = evaluation
        self._cursor += 1
        current_loss = self._raw_loss(evaluation.metrics)
        if self._best is None or current_loss < self._raw_loss(self._best.metrics):
            self._best = evaluation
            self._best_offsets = self._offsets.copy()
            self._best_step = self._cursor
        item = {
            "step": self._cursor,
            "point_index": point_index,
            "point_id": point.point_id,
            "task_type": point.task_type,
            "action_class": action_value,
            "displacement_nm": displacement,
            "loss": current_loss,
            "raw_weighted_loss": current_loss,
            "loss_version": RECIPE_OPC_LOSS_VERSION,
            "metrics": evaluation.metrics.as_dict(),
            "mask_sha256": evaluation.mask_sha256,
            "recipe_offsets_nm": self._offsets.tolist(),
        }
        self._trajectory.append(item)
        terminated = self._cursor >= len(self._schedule)
        info = dict(item)
        if terminated:
            info.update({
                "best_step": self._best_step,
                "best_metrics": self.best_evaluation.metrics.as_dict(),
                "best_recipe_offsets_nm": self._best_offsets.tolist(),
            })
        return self._observation(), float(-current_loss), terminated, False, info

    @property
    def best_evaluation(self) -> RecipeEvaluation:
        """返回 episode 中固定评价损失最低的 solver 结果。"""
        if self._best is None:
            raise RuntimeError("环境必须先 reset")
        return self._best

    @property
    def best_recipe_offsets_nm(self) -> np.ndarray:
        """返回最佳 EPE/FRAG recipe 的逐点绝对位移。"""
        if self._best is None:
            raise RuntimeError("环境必须先 reset")
        return self._best_offsets.copy()

    @property
    def best_step(self) -> int:
        """返回最佳 recipe 首次出现的点级决策步。"""
        if self._best is None:
            raise RuntimeError("环境必须先 reset")
        return int(self._best_step)

    @property
    def trajectory(self) -> Tuple[dict, ...]:
        """返回不可由调用方原地修改的轨迹快照。"""
        return tuple(copy.deepcopy(self._trajectory))


@dataclass(frozen=True)
class _EdgeGeometry:
    """保存一条原始 target 边、固定分段位置和外法向。"""

    polygon_index: int
    edge_index: int
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    tangent_x: int
    tangent_y: int
    normal_x: int
    normal_y: int
    length_coord: int
    base_cuts_coord: Tuple[int, ...]


class OpenILTRecipeAwareSolver:
    """在项目侧重建 recipe-aware SimpleOPC 循环，OpenILT 仅作为只读函数库。"""

    def __init__(
        self,
        openilt_dir: Path,
        expected_commit: str,
        layout_path: Path,
        reward_weights: Dict[str, float],
        lithography_config: Path = Path("config/lithosimple.txt"),
        simulator: str = "simple",
        image_size: Tuple[int, int] = (2048, 2048),
        openilt_scale: int = 1,
        nm_per_coordinate: float = 1.0,
        base_fragment_length_nm: float = 96.0,
        min_fragment_length_nm: float = 8.0,
        recipe_displacement_limit_nm: float = 40.0,
        epe_sample_distance_nm: float = 16.0,
        inner_step_sizes_nm: Sequence[float] = (8, 8, 8, 8, 4, 4, 4, 4),
        mask_displacement_limit_nm: float = 24.0,
        threshold: float = 0.5,
        cache_entries: int = 8,
    ):
        import torch

        self.openilt_dir = Path(openilt_dir).resolve()
        self.layout_path = Path(layout_path).resolve()
        if not self.layout_path.is_file():
            raise FileNotFoundError(f"Recipe PPO GLP 不存在：{self.layout_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("Recipe-aware OpenILT solver 要求 CUDA；请在云端 GPU 实例运行")
        if simulator not in {"simple", "exact"}:
            raise ValueError("simulator 必须是 simple 或 exact")
        self.reward_weights = _validate_reward_weights(reward_weights)
        self.openilt_scale = int(openilt_scale)
        if self.openilt_scale != 1:
            raise ValueError("simpleopc-recipe-point-v1 当前只支持 openilt_scale=1")
        self.nm_per_coordinate = float(nm_per_coordinate)
        if self.nm_per_coordinate <= 0:
            raise ValueError("nm_per_coordinate 必须大于零")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size 必须包含两个正整数")
        self.base_fragment_length_nm = float(base_fragment_length_nm)
        self.min_fragment_length_nm = float(min_fragment_length_nm)
        self.recipe_displacement_limit_nm = float(recipe_displacement_limit_nm)
        required_base = 2 * self.recipe_displacement_limit_nm + self.min_fragment_length_nm
        if self.base_fragment_length_nm < required_base:
            raise ValueError(
                "base_fragment_length_nm 必须至少为 "
                "2×recipe_displacement_limit_nm+min_fragment_length_nm"
            )
        self.epe_sample_distance_nm = float(epe_sample_distance_nm)
        if self.epe_sample_distance_nm <= 0:
            raise ValueError("epe_sample_distance_nm 必须大于零")
        self.inner_step_sizes_nm = tuple(float(value) for value in inner_step_sizes_nm)
        if not self.inner_step_sizes_nm or any(value <= 0 for value in self.inner_step_sizes_nm):
            raise ValueError("inner_step_sizes_nm 必须是非空正数序列")
        self.mask_displacement_limit_nm = float(mask_displacement_limit_nm)
        if self.mask_displacement_limit_nm <= 0:
            raise ValueError("mask_displacement_limit_nm 必须大于零")
        self.threshold = float(threshold)
        self.cache_entries = int(cache_entries)
        if self.cache_entries < 0:
            raise ValueError("cache_entries 不能为负")
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
        self._polygons = self._load_polygons()
        self._target_array = self._render_target()
        self._target = torch.as_tensor(self._target_array, dtype=torch.float32, device="cuda")
        self._vposes, self._hposes = self._evaluation.boundaries(self._target)
        self._edges, self.recipe_points, self._frag_point_by_edge_cut = self._build_recipe_geometry()
        self._cache: "OrderedDict[Tuple[int, ...], RecipeEvaluation]" = OrderedDict()

    @property
    def target_polygons(self) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        """返回原始 target 的只读多边形快照，供当前主线可视化复用。"""
        return tuple(tuple((int(x), int(y)) for x, y in polygon) for polygon in self._polygons)

    def _validate_openilt(self, expected_commit: str) -> str:
        """核验上游入口、固定提交和 tracked diff，不修改 OpenILT。"""
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

    def _load_polygons(self) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        """从原始 GLP 读取并居中 target 多边形，不使用 OPC 后 mask 打点。"""
        design = self._glp.Design(str(self.layout_path), down=self.openilt_scale)
        design.center(self.image_size[0], self.image_size[1], 0, 0)
        polygons = []
        for polygon in design.polygons:
            points = tuple((int(point[0]), int(point[1])) for point in polygon)
            if len(points) < 4:
                raise ValueError("target polygon 至少需要四个顶点")
            polygons.append(points)
        if not polygons:
            raise RuntimeError(f"版图不包含 target 多边形：{self.layout_path}")
        return tuple(polygons)

    def _render_target(self) -> np.ndarray:
        """直接栅格化原始 target 多边形作为固定评价基准。"""
        image = self._polygon.poly2img(
            [list(polygon) for polygon in self._polygons],
            self.image_size[0],
            self.image_size[1],
            scale=self.openilt_scale,
        )
        return (np.asarray(image, dtype=np.float32) / 255.0).astype(np.float32)

    def _normal_for_edge(self, start: Tuple[int, int], end: Tuple[int, int]) -> Tuple[int, int]:
        """通过原始 target 边两侧像素确定指向图形外部的法向。"""
        mid_x = int(round((start[0] + end[0]) / 2))
        mid_y = int(round((start[1] + end[1]) / 2))
        probe = 2
        binary = self._target_array >= self.threshold
        height, width = binary.shape
        if start[0] == end[0]:
            samples = ((mid_x + probe, mid_y), (mid_x - probe, mid_y))
            candidates = ((-1, 0), (1, 0))
        elif start[1] == end[1]:
            samples = ((mid_x, mid_y + probe), (mid_x, mid_y - probe))
            candidates = ((0, -1), (0, 1))
        else:
            raise ValueError("Recipe solver 当前只支持正交 target 边")
        if any(x < 0 or x >= width or y < 0 or y >= height for x, y in samples):
            raise ValueError("target 边法向探针越出图像")
        first_inside = bool(binary[samples[0][1], samples[0][0]])
        second_inside = bool(binary[samples[1][1], samples[1][0]])
        if first_inside == second_inside:
            raise ValueError("无法从原始 target 唯一确定边的外法向")
        return candidates[0] if first_inside else candidates[1]

    def _base_cut_positions(self, length_coord: int) -> Tuple[int, ...]:
        """生成保证每个内部 FRAG 点均可独立选择 ±40nm 的初始均匀分段。"""
        target = max(1, int(round(self.base_fragment_length_nm / self.nm_per_coordinate)))
        minimum = max(1, int(round(self.min_fragment_length_nm / self.nm_per_coordinate)))
        limit = max(1, int(round(self.recipe_displacement_limit_nm / self.nm_per_coordinate)))
        required = 2 * limit + minimum
        fragment_count = max(1, int(round(length_coord / target)))
        fragment_count = min(fragment_count, max(1, length_coord // required))
        while fragment_count > 1:
            positions = tuple(int(round(index * length_coord / fragment_count)) for index in range(fragment_count + 1))
            if min(np.diff(np.asarray(positions))) >= required:
                return positions
            fragment_count -= 1
        return (0, int(length_coord))

    @staticmethod
    def _point_on_edge(edge: _EdgeGeometry, position_coord: int) -> Tuple[int, int]:
        """把沿边弧长坐标转换为 target 图像像素坐标。"""
        return (
            int(edge.start_x + edge.tangent_x * position_coord),
            int(edge.start_y + edge.tangent_y * position_coord),
        )

    def _build_recipe_geometry(self):
        """在原始 target 上生成 EPE 测量点和可移动 FRAG 分段点。"""
        edges: List[_EdgeGeometry] = []
        points: List[RecipePoint] = []
        frag_lookup: Dict[Tuple[int, int], int] = {}
        limit = self.recipe_displacement_limit_nm
        for polygon_index, polygon in enumerate(self._polygons):
            for edge_index in range(len(polygon)):
                start = polygon[edge_index]
                end = polygon[(edge_index + 1) % len(polygon)]
                dx = int(end[0] - start[0])
                dy = int(end[1] - start[1])
                length = abs(dx) + abs(dy)
                if length <= 0 or (dx != 0 and dy != 0):
                    raise ValueError("Recipe solver 当前只接受非零正交 target 边")
                tangent = (
                    0 if dx == 0 else (1 if dx > 0 else -1),
                    0 if dy == 0 else (1 if dy > 0 else -1),
                )
                normal = self._normal_for_edge(start, end)
                edge = _EdgeGeometry(
                    polygon_index=polygon_index,
                    edge_index=edge_index,
                    start_x=start[0],
                    start_y=start[1],
                    end_x=end[0],
                    end_y=end[1],
                    tangent_x=tangent[0],
                    tangent_y=tangent[1],
                    normal_x=normal[0],
                    normal_y=normal[1],
                    length_coord=length,
                    base_cuts_coord=self._base_cut_positions(length),
                )
                global_edge_index = len(edges)
                edges.append(edge)
                cuts = edge.base_cuts_coord
                for fragment_index, (left, right) in enumerate(zip(cuts[:-1], cuts[1:])):
                    middle = int(round((left + right) / 2))
                    lower_nm = -middle * self.nm_per_coordinate
                    upper_nm = (length - middle) * self.nm_per_coordinate
                    if lower_nm <= -limit and upper_nm >= limit:
                        x, y = self._point_on_edge(edge, middle)
                        points.append(RecipePoint(
                            point_id=f"p{polygon_index}:e{edge_index}:epe{fragment_index}",
                            task_type="EPE",
                            polygon_index=polygon_index,
                            edge_index=global_edge_index,
                            anchor_index=fragment_index,
                            base_x=x,
                            base_y=y,
                            tangent_x=edge.tangent_x,
                            tangent_y=edge.tangent_y,
                            normal_x=edge.normal_x,
                            normal_y=edge.normal_y,
                            lower_delta_nm=-limit,
                            upper_delta_nm=limit,
                        ))
                for cut_index, position in enumerate(cuts[1:-1], start=1):
                    x, y = self._point_on_edge(edge, position)
                    point_index = len(points)
                    points.append(RecipePoint(
                        point_id=f"p{polygon_index}:e{edge_index}:frag{cut_index}",
                        task_type="FRAG",
                        polygon_index=polygon_index,
                        edge_index=global_edge_index,
                        anchor_index=cut_index,
                        base_x=x,
                        base_y=y,
                        tangent_x=edge.tangent_x,
                        tangent_y=edge.tangent_y,
                        normal_x=edge.normal_x,
                        normal_y=edge.normal_y,
                        lower_delta_nm=-limit,
                        upper_delta_nm=limit,
                    ))
                    frag_lookup[(global_edge_index, cut_index)] = point_index
        if not points or {point.task_type for point in points} != {"EPE", "FRAG"}:
            raise RuntimeError(
                "当前 base_fragment_length 无法在该版图同时生成完整 ±40nm 的 EPE/FRAG 点"
            )
        return tuple(edges), tuple(points), frag_lookup

    def _recipe_point_position(self, point_index: int, offsets_nm: np.ndarray) -> Tuple[int, int]:
        """计算应用绝对 recipe 位移后的点坐标。"""
        point = self.recipe_points[point_index]
        delta_coord = int(round(float(offsets_nm[point_index]) / self.nm_per_coordinate))
        return (
            int(point.base_x + point.tangent_x * delta_coord),
            int(point.base_y + point.tangent_y * delta_coord),
        )

    def _fragments_from_recipe(self, offsets_nm: np.ndarray):
        """用 FRAG recipe 点重建分段，并把每个 EPE 点映射到当前 fragment。"""
        offsets = np.asarray(offsets_nm, dtype=np.float64)
        if offsets.shape != (len(self.recipe_points),):
            raise ValueError("Recipe 位移数量与点数不一致")
        if np.any(~np.isin(offsets, np.asarray(DISPLACEMENT_CLASSES_NM, dtype=np.float64))):
            raise ValueError("Recipe 位移必须精确属于九分类代表值")
        minimum = max(1, int(round(self.min_fragment_length_nm / self.nm_per_coordinate)))
        by_polygon: List[List[List[Tuple[int, int]]]] = [[] for _ in self._polygons]
        flat = []
        edge_ranges: Dict[int, Tuple[int, Tuple[int, ...]]] = {}
        for edge_index, edge in enumerate(self._edges):
            cuts = list(edge.base_cuts_coord)
            for cut_index in range(1, len(cuts) - 1):
                point_index = self._frag_point_by_edge_cut[(edge_index, cut_index)]
                cuts[cut_index] += int(round(offsets[point_index] / self.nm_per_coordinate))
            if min(np.diff(np.asarray(cuts, dtype=np.int64))) < minimum:
                raise ValueError(f"FRAG recipe 使 edge {edge_index} 出现小于最小长度的 fragment")
            start_flat = len(flat)
            for left, right in zip(cuts[:-1], cuts[1:]):
                start = self._point_on_edge(edge, int(left))
                end = self._point_on_edge(edge, int(right))
                segment = [start, end]
                by_polygon[edge.polygon_index].append(segment)
                flat.append((edge_index, int(left), int(right), segment))
            edge_ranges[edge_index] = (start_flat, tuple(int(value) for value in cuts))
        epe_to_fragment = np.full(len(self.recipe_points), -1, dtype=np.int64)
        for point_index, point in enumerate(self.recipe_points):
            if point.task_type != "EPE":
                continue
            edge = self._edges[point.edge_index]
            x, y = self._recipe_point_position(point_index, offsets)
            position = (
                (x - edge.start_x) * edge.tangent_x
                + (y - edge.start_y) * edge.tangent_y
            )
            start_flat, cuts = edge_ranges[point.edge_index]
            local = int(np.searchsorted(np.asarray(cuts), position, side="right") - 1)
            local = min(max(local, 0), len(cuts) - 2)
            epe_to_fragment[point_index] = start_flat + local
        return by_polygon, flat, epe_to_fragment

    def _render_mask(self, by_polygon, flat, mask_offsets_nm: np.ndarray) -> np.ndarray:
        """仅在 solver 内沿各 fragment 法向移动 mask，并法化连接点。"""
        values = np.asarray(mask_offsets_nm, dtype=np.float64)
        if values.shape != (len(flat),):
            raise ValueError("mask fragment 位移数量不一致")
        moved_by_polygon: List[List[List[Tuple[int, int]]]] = [[] for _ in self._polygons]
        for flat_index, (edge_index, _left, _right, segment) in enumerate(flat):
            edge = self._edges[edge_index]
            delta = int(round(values[flat_index] / self.nm_per_coordinate))
            dx = edge.normal_x * delta
            dy = edge.normal_y * delta
            moved_by_polygon[edge.polygon_index].append([
                (segment[0][0] + dx, segment[0][1] + dy),
                (segment[1][0] + dx, segment[1][1] + dy),
            ])
        polygons = [self._polygon.segs2poly(segments) for segments in moved_by_polygon]
        image = self._polygon.poly2img(
            polygons,
            self.image_size[0],
            self.image_size[1],
            scale=self.openilt_scale,
        )
        return (np.asarray(image, dtype=np.float32) / 255.0).astype(np.float32)

    @staticmethod
    def _number(value) -> float:
        """把单元素张量或普通数值转换为 float。"""
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return float(value)

    def _simulate(self, mask_array: np.ndarray):
        """执行一次固定 OpenILT 光刻仿真并返回固定评价指标。"""
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
        metrics = SimpleOPCMetrics(
            l2=self._number(l2),
            epe=self._number(epe_in) + self._number(epe_out),
            pvb=self._number(pvb),
        )
        return (
            metrics,
            binary_nom.detach().cpu().numpy().astype(np.uint8),
        )

    def _recipe_epe_signs(self, offsets_nm: np.ndarray, printed_image: np.ndarray) -> np.ndarray:
        """只在可移动 recipe EPE 点上测量内部 solver 的移动方向。"""
        printed = np.asarray(printed_image, dtype=bool)
        target = self._target_array >= self.threshold
        height, width = target.shape
        distance = int(round(self.epe_sample_distance_nm / self.nm_per_coordinate))
        signs = np.zeros(len(self.recipe_points), dtype=np.float32)
        for point_index, point in enumerate(self.recipe_points):
            if point.task_type != "EPE":
                continue
            x, y = self._recipe_point_position(point_index, offsets_nm)
            inner_x = x - point.normal_x * distance
            inner_y = y - point.normal_y * distance
            outer_x = x + point.normal_x * distance
            outer_y = y + point.normal_y * distance
            if not (0 <= inner_x < width and 0 <= inner_y < height):
                continue
            if not (0 <= outer_x < width and 0 <= outer_y < height):
                continue
            if target[inner_y, inner_x] and not printed[inner_y, inner_x]:
                signs[point_index] = 1.0
            elif not target[outer_y, outer_x] and printed[outer_y, outer_x]:
                signs[point_index] = -1.0
        return signs

    def solve(self, recipe_offsets_nm: np.ndarray) -> RecipeEvaluation:
        """根据 EPE/FRAG recipe 完成内部 mask OPC，并按固定 target 评价最佳结果。"""
        offsets = np.asarray(recipe_offsets_nm, dtype=np.float64)
        key = tuple(int(round(value)) for value in offsets)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        by_polygon, flat, epe_to_fragment = self._fragments_from_recipe(offsets)
        mask_offsets = np.zeros(len(flat), dtype=np.float64)
        best_loss = float("inf")
        best_metrics: Optional[SimpleOPCMetrics] = None
        best_mask: Optional[np.ndarray] = None
        best_printed: Optional[np.ndarray] = None
        best_signs: Optional[np.ndarray] = None
        trace = []
        step_sizes = (0.0,) + self.inner_step_sizes_nm
        for inner_step, step_size in enumerate(step_sizes):
            if inner_step > 0:
                previous_signs = trace[-1]["recipe_epe_signs"]
                votes = np.zeros(len(flat), dtype=np.float64)
                for point_index, sign in enumerate(previous_signs):
                    fragment_index = int(epe_to_fragment[point_index])
                    if fragment_index >= 0:
                        votes[fragment_index] += float(sign)
                directions = np.sign(votes)
                mask_offsets = np.clip(
                    mask_offsets + directions * float(step_size),
                    -self.mask_displacement_limit_nm,
                    self.mask_displacement_limit_nm,
                )
            mask = self._render_mask(by_polygon, flat, mask_offsets)
            metrics, printed = self._simulate(mask)
            signs = self._recipe_epe_signs(offsets, printed)
            raw_loss = weighted_opc_loss(metrics.as_dict(), self.reward_weights)
            mask_hash = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
            trace.append({
                "inner_step": inner_step,
                "mask_step_size_nm": float(step_size),
                "raw_weighted_loss": raw_loss,
                "metrics": metrics.as_dict(),
                "recipe_epe_violations": int(np.count_nonzero(signs)),
                "recipe_epe_signs": signs.tolist(),
                "mask_sha256": mask_hash,
            })
            if raw_loss < best_loss:
                best_loss = raw_loss
                best_metrics = metrics
                best_mask = (mask >= self.threshold).astype(np.uint8)
                best_printed = printed.astype(np.uint8)
                best_signs = signs.copy()
        if best_metrics is None or best_mask is None or best_printed is None or best_signs is None:
            raise RuntimeError("Recipe-aware solver 未产生任何内部 OPC 结果")
        evaluation = RecipeEvaluation(
            metrics=best_metrics,
            recipe_epe_signs=best_signs,
            mask_sha256=hashlib.sha256(np.ascontiguousarray(best_mask).tobytes()).hexdigest(),
            target_image=(self._target_array >= self.threshold).astype(np.uint8),
            mask_image=best_mask,
            printed_image=best_printed,
            internal_trace=tuple(copy.deepcopy(trace)),
        )
        if self.cache_entries > 0:
            self._cache[key] = evaluation
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_entries:
                self._cache.popitem(last=False)
        return evaluation

    def local_image_stack(
        self,
        point_index: int,
        recipe_offsets_nm: np.ndarray,
        evaluation: RecipeEvaluation,
        patch_size: int,
    ) -> np.ndarray:
        """生成 target/mask/printed/EPE-marker/FRAG-marker 五通道局部图像。"""
        if not 0 <= point_index < len(self.recipe_points):
            raise IndexError("Recipe point index 越界")
        x, y = self._recipe_point_position(point_index, np.asarray(recipe_offsets_nm))
        channels = [
            _center_crop(evaluation.target_image, x, y, patch_size),
            _center_crop(evaluation.mask_image, x, y, patch_size),
            _center_crop(evaluation.printed_image, x, y, patch_size),
            np.zeros((patch_size, patch_size), dtype=np.float32),
            np.zeros((patch_size, patch_size), dtype=np.float32),
        ]
        marker_channel = 3 if self.recipe_points[point_index].task_type == "EPE" else 4
        center = patch_size // 2
        channels[marker_channel][center - 1:center + 2, center - 1:center + 2] = 1.0
        return np.stack(channels, axis=0).astype(np.float32)
