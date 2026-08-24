"""本模块用版本化、公开可审计的栅格规则生成逐点九分类候选掩模。

输入为目标图、基准掩模、像素物理比例和包含点坐标/轴向法线/局部半径的 JSON 清单；输出为 PPO Oracle
所需 observations、target、candidate_masks NPZ 及元数据 JSON。关键依赖为 OpenCV、NumPy、Pydantic 和
geometry-v1 特征；论文未公开其点移动几何代码，本实现是明确标记的兼容适配，不宣称与论文逐像素一致。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field, root_validator, validator

from .features import FEATURE_NAMES, extract_geometry_features
from .metrics import DISPLACEMENT_CLASSES_NM
from .models import TaskType


ADAPTER_V1 = "raster-fragment-v1"
ADAPTER_V2 = "raster-boundary-strip-v2"
ADAPTER_V3 = "raster-edge-segment-v3"
DEFAULT_ADAPTER_VERSION = ADAPTER_V2


class FragmentPoint(BaseModel):
    """保存测量点；v3 额外保存该点所属完整轴向边段的两个端点。"""

    schema_version: str = "1.0"
    point_id: str
    task_type: TaskType
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    normal_x: int = Field(ge=-1, le=1)
    normal_y: int = Field(ge=-1, le=1)
    support_radius_px: int = Field(gt=0)
    segment_start_x: Optional[int] = Field(default=None, ge=0)
    segment_start_y: Optional[int] = Field(default=None, ge=0)
    segment_end_x: Optional[int] = Field(default=None, ge=0)
    segment_end_y: Optional[int] = Field(default=None, ge=0)

    @validator("normal_y")
    def _axis_aligned_normal(cls, value: int, values) -> int:
        normal_x = values.get("normal_x")
        if normal_x is not None and abs(normal_x) + abs(value) != 1:
            raise ValueError("法线必须是 (±1,0) 或 (0,±1)")
        return value

    @root_validator(skip_on_failure=True)
    def _valid_optional_segment(cls, values):
        names = ("segment_start_x", "segment_start_y", "segment_end_x", "segment_end_y")
        segment_values = [values.get(name) for name in names]
        if all(value is None for value in segment_values):
            return values
        if any(value is None for value in segment_values):
            raise ValueError("边段四个端点坐标必须同时提供")
        start_x, start_y, end_x, end_y = segment_values
        if start_x != end_x and start_y != end_y:
            raise ValueError("边段必须水平或垂直")
        if values.get("x") != (start_x + end_x) // 2 or values.get("y") != (start_y + end_y) // 2:
            raise ValueError("v3 测量点必须位于边段整数中点")
        normal_x, normal_y = values.get("normal_x"), values.get("normal_y")
        if start_x == end_x and start_y != end_y and normal_x == 0:
            raise ValueError("垂直边段的法线必须沿 x 轴")
        if start_y == end_y and start_x != end_x and normal_y == 0:
            raise ValueError("水平边段的法线必须沿 y 轴")
        return values

    @property
    def has_segment_geometry(self) -> bool:
        """返回该点是否携带完整的 v3 边段几何。"""
        return self.segment_start_x is not None


class CandidateManifest(BaseModel):
    """保存同一 clip 的候选生成输入和物理比例。"""

    schema_version: str = "1.0"
    adapter_version: str = DEFAULT_ADAPTER_VERSION
    clip_id: str
    parent_layout: str
    split: str
    target_path: str
    base_mask_path: str
    scale_nm_per_pixel: float = Field(gt=0)
    points: List[FragmentPoint]
    sampling_audit: Optional[Dict[str, int]] = None
    sampling_exclusions: List[dict] = Field(default_factory=list)

    @validator("adapter_version")
    def _valid_adapter(cls, value: str) -> str:
        if value not in {ADAPTER_V1, ADAPTER_V2, ADAPTER_V3}:
            raise ValueError(f"未知候选适配器版本：{value}")
        return value

    @validator("split")
    def _valid_split(cls, value: str) -> str:
        if value not in {"train", "validation", "test"}:
            raise ValueError("split 必须是 train、validation 或 test")
        return value

    @validator("points")
    def _nonempty_unique_points(cls, value: List[FragmentPoint]) -> List[FragmentPoint]:
        if not value:
            raise ValueError("points 不能为空")
        identifiers = [point.point_id for point in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("points 存在重复 point_id")
        return value

    @root_validator(skip_on_failure=True)
    def _v3_requires_epe_segments(cls, values):
        if values.get("adapter_version") != ADAPTER_V3:
            return values
        for point in values.get("points") or []:
            if point.task_type != TaskType.EPE:
                raise ValueError("v3 第一阶段只允许 EPE 边段，不生成伪 FRAG 标签")
            if not point.has_segment_geometry:
                raise ValueError("v3 的每个测量点都必须包含完整边段端点")
        return values


def _load_binary(path: Path) -> np.ndarray:
    """兼容中文路径读取灰度图，并转换为 0/1 float32。"""
    if not path.is_file():
        raise FileNotFoundError(f"候选掩模输入不存在：{path}")
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    return (image > 127).astype(np.float32)


def _sha256(path: Path) -> str:
    """计算输入文件哈希。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bounds(center: int, radius: int, limit: int):
    """返回裁剪到图像范围的半开区间。"""
    return max(0, center - radius), min(limit, center + radius + 1)


def move_local_fragment(mask: np.ndarray, point: FragmentPoint, displacement_px: int) -> np.ndarray:
    """清除点附近局部块并沿法线平移，越界部分被裁剪。"""
    source = np.asarray(mask, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("mask 必须是二维数组")
    height, width = source.shape
    if point.x >= width or point.y >= height:
        raise ValueError(f"点 {point.point_id} 超出掩模范围")
    if displacement_px == 0:
        return source.copy()
    y0, y1 = _bounds(point.y, point.support_radius_px, height)
    x0, x1 = _bounds(point.x, point.support_radius_px, width)
    patch = source[y0:y1, x0:x1].copy()
    result = source.copy()
    result[y0:y1, x0:x1] = 0
    shift_y = point.normal_y * displacement_px
    shift_x = point.normal_x * displacement_px
    destination_y0, destination_x0 = y0 + shift_y, x0 + shift_x
    destination_y1, destination_x1 = y1 + shift_y, x1 + shift_x
    clipped_y0, clipped_y1 = max(0, destination_y0), min(height, destination_y1)
    clipped_x0, clipped_x1 = max(0, destination_x0), min(width, destination_x1)
    if clipped_y0 < clipped_y1 and clipped_x0 < clipped_x1:
        patch_y0 = clipped_y0 - destination_y0
        patch_x0 = clipped_x0 - destination_x0
        patch_y1 = patch_y0 + (clipped_y1 - clipped_y0)
        patch_x1 = patch_x0 + (clipped_x1 - clipped_x0)
        result[clipped_y0:clipped_y1, clipped_x0:clipped_x1] = np.maximum(
            result[clipped_y0:clipped_y1, clipped_x0:clipped_x1],
            patch[patch_y0:patch_y1, patch_x0:patch_x1],
        )
    return result


def move_boundary_strip(mask: np.ndarray, point: FragmentPoint, displacement_px: int) -> np.ndarray:
    """沿外法线扩张或向内收缩局部边界条带，避免方块平移产生动作塌缩。"""
    source = np.asarray(mask, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("mask 必须是二维数组")
    height, width = source.shape
    if point.x >= width or point.y >= height:
        raise ValueError(f"点 {point.point_id} 超出掩模范围")
    result = source.copy()
    if displacement_px == 0:
        return result
    radius = point.support_radius_px
    magnitude = abs(int(displacement_px))
    outward = displacement_px > 0
    value = 1.0 if outward else 0.0
    if point.normal_x:
        y0, y1 = _bounds(point.y, radius, height)
        offsets = range(1, magnitude + 1) if outward else range(magnitude)
        for offset in offsets:
            x = point.x + point.normal_x * offset if outward else point.x - point.normal_x * offset
            if 0 <= x < width:
                result[y0:y1, x] = value
    else:
        x0, x1 = _bounds(point.x, radius, width)
        offsets = range(1, magnitude + 1) if outward else range(magnitude)
        for offset in offsets:
            y = point.y + point.normal_y * offset if outward else point.y - point.normal_y * offset
            if 0 <= y < height:
                result[y, x0:x1] = value
    return result


def move_edge_segment(mask: np.ndarray, point: FragmentPoint, displacement_px: int) -> np.ndarray:
    """沿法线整体扩张或收缩一条 v3 边段，边段上的像素使用同一个动作。"""
    source = np.asarray(mask, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("mask 必须是二维数组")
    if not point.has_segment_geometry:
        raise ValueError(f"点 {point.point_id} 缺少 v3 边段端点")
    height, width = source.shape
    coordinates = (
        point.segment_start_x, point.segment_start_y,
        point.segment_end_x, point.segment_end_y,
    )
    if not (0 <= coordinates[0] < width and 0 <= coordinates[2] < width):
        raise ValueError(f"点 {point.point_id} 的边段 x 坐标超出掩模范围")
    if not (0 <= coordinates[1] < height and 0 <= coordinates[3] < height):
        raise ValueError(f"点 {point.point_id} 的边段 y 坐标超出掩模范围")
    result = source.copy()
    if displacement_px == 0:
        return result
    magnitude = abs(int(displacement_px))
    outward = displacement_px > 0
    value = 1.0 if outward else 0.0
    if point.normal_x:
        y0, y1 = sorted((point.segment_start_y, point.segment_end_y))
        offsets = range(1, magnitude + 1) if outward else range(magnitude)
        for offset in offsets:
            x = point.x + point.normal_x * offset if outward else point.x - point.normal_x * offset
            if 0 <= x < width:
                result[y0:y1 + 1, x] = value
    else:
        x0, x1 = sorted((point.segment_start_x, point.segment_end_x))
        offsets = range(1, magnitude + 1) if outward else range(magnitude)
        for offset in offsets:
            y = point.y + point.normal_y * offset if outward else point.y - point.normal_y * offset
            if 0 <= y < height:
                result[y, x0:x1 + 1] = value
    return result


def generate_candidate_mask(
    mask: np.ndarray,
    point: FragmentPoint,
    displacement_px: int,
    adapter_version: str,
) -> np.ndarray:
    """按数据内记录的适配器版本生成候选，保证旧数据不会被新几何静默重解释。"""
    if adapter_version == ADAPTER_V1:
        return move_local_fragment(mask, point, displacement_px)
    if adapter_version == ADAPTER_V2:
        return move_boundary_strip(mask, point, displacement_px)
    if adapter_version == ADAPTER_V3:
        return move_edge_segment(mask, point, displacement_px)
    raise ValueError(f"未知候选适配器版本：{adapter_version}")

def validate_distinct_candidate_actions(
    mask: np.ndarray,
    point: FragmentPoint,
    scale_nm_per_pixel: float,
    adapter_version: str,
) -> None:
    """要求 v2/v3 的九个位移生成九张不同掩模，阻止候选碰撞进入昂贵 OpenILT 阶段。"""
    if adapter_version not in {ADAPTER_V2, ADAPTER_V3}:
        return
    hashes = set()
    for displacement_nm in DISPLACEMENT_CLASSES_NM:
        displacement_px = int(round(displacement_nm / scale_nm_per_pixel))
        candidate = generate_candidate_mask(mask, point, displacement_px, adapter_version)
        packed = np.packbits(candidate > 0.5)
        hashes.add(hashlib.sha256(packed.tobytes()).hexdigest())
    if len(hashes) != len(DISPLACEMENT_CLASSES_NM):
        raise ValueError(
            f"点 {point.point_id} 的 {adapter_version} 九动作仅生成 {len(hashes)} 张不同掩模；拒绝候选碰撞"
        )


def _observation(mask: np.ndarray, point: FragmentPoint) -> np.ndarray:
    """组合局部 geometry-v1、归一化坐标、法线和任务类型。"""
    height, width = mask.shape
    radius = point.support_radius_px
    y0, y1 = _bounds(point.y, radius, height)
    x0, x1 = _bounds(point.x, radius, width)
    patch = mask[y0:y1, x0:x1]
    if min(patch.shape) < 2:
        raise ValueError(f"点 {point.point_id} 的局部窗口过小")
    features = extract_geometry_features(patch)
    values = [features[name] for name in FEATURE_NAMES]
    values.extend([
        point.x / max(1, width - 1),
        point.y / max(1, height - 1),
        float(point.normal_x),
        float(point.normal_y),
        0.0 if point.task_type == TaskType.EPE else 1.0,
    ])
    return np.asarray(values, dtype=np.float32)


def build_candidate_arrays(manifest: CandidateManifest):
    """按点清单顺序生成状态矩阵和 [point, 9, height, width] 候选掩模。"""
    target_path = Path(manifest.target_path)
    mask_path = Path(manifest.base_mask_path)
    target = _load_binary(target_path)
    base_mask = _load_binary(mask_path)
    if target.shape != base_mask.shape:
        raise ValueError("目标图与基准掩模尺寸不一致")
    observations = []
    candidates = []
    for point in manifest.points:
        observations.append(_observation(base_mask, point))
        point_candidates = []
        for displacement_nm in DISPLACEMENT_CLASSES_NM:
            displacement_px = int(round(displacement_nm / manifest.scale_nm_per_pixel))
            point_candidates.append(generate_candidate_mask(
                base_mask, point, displacement_px, manifest.adapter_version
            ))
        candidates.append(point_candidates)
    metadata = {
        "schema_version": "1.0",
        "adapter_version": manifest.adapter_version,
        "clip_id": manifest.clip_id,
        "parent_layout": manifest.parent_layout,
        "split": manifest.split,
        "scale_nm_per_pixel": manifest.scale_nm_per_pixel,
        "target_sha256": _sha256(target_path),
        "base_mask_sha256": _sha256(mask_path),
        "point_ids": [point.point_id for point in manifest.points],
        "task_types": [point.task_type.value for point in manifest.points],
        "observation_fields": list(FEATURE_NAMES) + ["x_norm", "y_norm", "normal_x", "normal_y", "task_is_frag"],
        "displacement_classes_nm": list(DISPLACEMENT_CLASSES_NM),
        "sampling_audit": manifest.sampling_audit,
        "sampling_exclusions": manifest.sampling_exclusions,
    }
    return np.stack(observations), target, np.asarray(candidates, dtype=np.float32), metadata


def build_compact_candidate_data(manifest: CandidateManifest):
    """仅保存基准掩模和点参数，避免为每个点复制九张全尺寸图。"""
    target_path = Path(manifest.target_path)
    mask_path = Path(manifest.base_mask_path)
    target = _load_binary(target_path)
    base_mask = _load_binary(mask_path)
    if target.shape != base_mask.shape:
        raise ValueError("目标图与基准掩模尺寸不一致")
    for point in manifest.points:
        validate_distinct_candidate_actions(
            base_mask, point, manifest.scale_nm_per_pixel, manifest.adapter_version
        )
    observations = np.stack([_observation(base_mask, point) for point in manifest.points])
    if manifest.adapter_version == ADAPTER_V3:
        point_geometry = np.asarray([
            [
                point.x, point.y, point.normal_x, point.normal_y, point.support_radius_px,
                point.segment_start_x, point.segment_start_y, point.segment_end_x, point.segment_end_y,
            ]
            for point in manifest.points
        ], dtype=np.int32)
        storage_format = "compact-edge-geometry-v1"
    else:
        point_geometry = np.asarray([
            [point.x, point.y, point.normal_x, point.normal_y, point.support_radius_px]
            for point in manifest.points
        ], dtype=np.int32)
        storage_format = "compact-point-geometry-v1"
    metadata = {
        "schema_version": "1.0",
        "adapter_version": manifest.adapter_version,
        "storage_format": storage_format,
        "action_uniqueness": (
            "nine-distinct-required"
            if manifest.adapter_version in {ADAPTER_V2, ADAPTER_V3}
            else "legacy-not-required"
        ),
        "clip_id": manifest.clip_id,
        "parent_layout": manifest.parent_layout,
        "split": manifest.split,
        "scale_nm_per_pixel": manifest.scale_nm_per_pixel,
        "target_sha256": _sha256(target_path),
        "base_mask_sha256": _sha256(mask_path),
        "point_ids": [point.point_id for point in manifest.points],
        "task_types": [point.task_type.value for point in manifest.points],
        "observation_fields": list(FEATURE_NAMES) + ["x_norm", "y_norm", "normal_x", "normal_y", "task_is_frag"],
        "displacement_classes_nm": list(DISPLACEMENT_CLASSES_NM),
        "sampling_audit": manifest.sampling_audit,
        "sampling_exclusions": manifest.sampling_exclusions,
    }
    return observations, target, base_mask, point_geometry, metadata

def main(argv: Optional[List[str]] = None) -> int:
    """从 JSON manifest 生成不可静默覆盖的 NPZ 和元数据文件。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.candidate_masks")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = CandidateManifest.parse_obj(json.loads(args.manifest.read_text(encoding="utf-8")))
    observations, target, base_mask, point_geometry, metadata = build_compact_candidate_data(manifest)
    metadata_path = args.output.with_suffix(".metadata.json")
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError("候选掩模输出已存在；请使用新的数据版本路径")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(args.output), observations=observations, target=target, base_mask=base_mask,
        point_geometry=point_geometry, scale_nm_per_pixel=np.asarray(manifest.scale_nm_per_pixel),
        adapter_version=np.asarray(manifest.adapter_version),
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



