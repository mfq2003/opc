"""本模块从公开 OPC 二值图确定性采样 v2 点或 v3 可移动边段，并生成多 clip 紧凑候选索引。

输入为 paper_repro.yaml 的父版图 6/2/2 划分、OpenILT tmp 中的目标/掩模图和采样参数；输出为每个
父版图的 CandidateManifest、v2 点几何或 v3 边段几何 NPZ/元数据及带哈希的 candidate-index.json。
关键依赖为 OpenCV、NumPy、Pydantic 与 candidate_masks；v2 保留原有 EPE/FRAG 点规则，v3 使用自适应
正交边段和每段中心一个 EPE 测量点。两者都属于论文缺失逻辑的可审计适配，不调用 GPU、OpenILT 或 PPO。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from pydantic import BaseModel, Field, root_validator, validator

from .candidate_masks import (
    ADAPTER_V3,
    CandidateManifest,
    FragmentPoint,
    _load_binary,
    build_compact_candidate_data,
    validate_distinct_candidate_actions,
)
from .models import TaskType


SAMPLER_V2 = "axis-boundary-sampler-v2"
SAMPLER_V3 = "adaptive-edge-segment-sampler-v3"
SAMPLER_VERSION = SAMPLER_V2


class ClipImageSource(BaseModel):
    """保存一个父版图对应的公开目标图、基准掩模和固定数据划分。"""

    schema_version: str = "1.0"
    clip_id: str
    parent_layout: str
    split: str
    target_path: str
    base_mask_path: str
    scale_nm_per_pixel: float = Field(gt=0)

    @validator("split")
    def _valid_split(cls, value: str) -> str:
        if value not in {"train", "validation", "test"}:
            raise ValueError("split 必须是 train、validation 或 test")
        return value


class SamplingSettings(BaseModel):
    """保存 v2 点采样参数以及 v3 自适应边段长度参数。"""

    schema_version: str = "1.0"
    sampler_version: str = SAMPLER_VERSION
    epe_spacing_px: int = Field(default=128, gt=0)
    frag_spacing_px: int = Field(default=192, gt=1)
    support_radius_px: int = Field(default=8, gt=0)
    min_segment_length_px: int = Field(default=24, gt=1)
    max_epe_points: int = Field(default=8, gt=0)
    max_frag_points: int = Field(default=8, gt=0)
    max_displacement_nm: float = Field(default=40, gt=0, le=40)
    v3_target_segment_length_px: int = Field(default=128, gt=1)
    v3_min_segment_length_px: int = Field(default=32, gt=1)
    v3_corner_segment_length_px: int = Field(default=64, gt=1)

    @validator("sampler_version")
    def _valid_sampler(cls, value: str) -> str:
        if value not in {SAMPLER_V2, SAMPLER_V3}:
            raise ValueError(f"未知采样器版本：{value}")
        return value

    @root_validator(skip_on_failure=True)
    def _valid_v3_lengths(cls, values):
        minimum = values.get("v3_min_segment_length_px")
        corner = values.get("v3_corner_segment_length_px")
        target = values.get("v3_target_segment_length_px")
        if None not in {minimum, corner, target} and not minimum <= corner <= target:
            raise ValueError("v3 边段长度必须满足 min <= corner <= target")
        return values


class CandidateIndexEntry(BaseModel):
    """保存一个 clip 的紧凑候选路径、哈希和 EPE/FRAG 点数。"""

    schema_version: str = "1.0"
    clip_id: str
    parent_layout: str
    split: str
    dataset_path: str
    metadata_path: str
    manifest_path: str
    dataset_sha256: str = Field(min_length=64, max_length=64)
    epe_points: int = Field(ge=1)
    frag_points: int = Field(ge=0)
    excluded_segments: int = Field(default=0, ge=0)
    excluded_boundary_units: int = Field(default=0, ge=0)


class CandidateIndex(BaseModel):
    """保存无父版图泄漏的多 clip 候选数据版本。"""

    schema_version: str = "1.0"
    sampler_version: str = SAMPLER_VERSION
    settings: SamplingSettings
    entries: List[CandidateIndexEntry]
    index_sha256: str = Field(min_length=64, max_length=64)


@dataclass(frozen=True)
class BoundarySegment:
    """表示一个含端点的水平或垂直正交边界段。"""

    orientation: str
    fixed: int
    start: int
    end: int
    normal_x: int
    normal_y: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def _group_consecutive(values: Sequence[int]) -> List[Tuple[int, int]]:
    """把已排序整数合并为含端点的连续区间。"""
    if not values:
        return []
    groups = []
    start = previous = int(values[0])
    for raw in values[1:]:
        value = int(raw)
        if value != previous + 1:
            groups.append((start, previous))
            start = value
        previous = value
    groups.append((start, previous))
    return groups


def extract_axis_boundary_segments(binary: np.ndarray) -> List[BoundarySegment]:
    """从二值图的水平/垂直像素跃迁提取带外法线的连续正交边界段。"""
    image = np.asarray(binary, dtype=bool)
    if image.ndim != 2 or min(image.shape) < 3:
        raise ValueError("边界采样输入必须是边长至少 3 的二维图")
    segments: List[BoundarySegment] = []
    vertical = image[:, :-1] != image[:, 1:]
    ys, x_lefts = np.nonzero(vertical)
    vertical_groups: Dict[Tuple[int, int], List[int]] = {}
    for y, x_left in zip(ys.tolist(), x_lefts.tolist()):
        left_inside = bool(image[y, x_left])
        point_x = x_left if left_inside else x_left + 1
        normal_x = 1 if left_inside else -1
        vertical_groups.setdefault((point_x, normal_x), []).append(y)
    for (fixed, normal_x), values in sorted(vertical_groups.items()):
        for start, end in _group_consecutive(sorted(values)):
            segments.append(BoundarySegment("vertical", fixed, start, end, normal_x, 0))

    horizontal = image[:-1, :] != image[1:, :]
    y_tops, xs = np.nonzero(horizontal)
    horizontal_groups: Dict[Tuple[int, int], List[int]] = {}
    for y_top, x in zip(y_tops.tolist(), xs.tolist()):
        top_inside = bool(image[y_top, x])
        point_y = y_top if top_inside else y_top + 1
        normal_y = 1 if top_inside else -1
        horizontal_groups.setdefault((point_y, normal_y), []).append(x)
    for (fixed, normal_y), values in sorted(horizontal_groups.items()):
        for start, end in _group_consecutive(sorted(values)):
            segments.append(BoundarySegment("horizontal", fixed, start, end, 0, normal_y))
    return sorted(
        segments,
        key=lambda item: (item.orientation, item.fixed, item.start, item.end, item.normal_x, item.normal_y),
    )


def _positions(start: int, end: int, spacing: int, internal: bool) -> List[int]:
    """沿含端点区间确定性取样；FRAG 使用内部间隔点，EPE 使用区间中心网格。"""
    length = end - start + 1
    if internal:
        result = list(range(start + spacing, end, spacing))
        return result or ([start + length // 2] if length >= 2 * spacing else [])
    count = max(1, int(math.ceil(length / spacing)))
    return [int(round(value)) for value in np.linspace(start, end, count + 2)[1:-1]]


def _point_on_segment(segment: BoundarySegment, position: int, task_type: TaskType, radius: int) -> FragmentPoint:
    """把边界段上的一维位置转换为带稳定临时 id 的二维点。"""
    if segment.orientation == "vertical":
        x, y = segment.fixed, position
    else:
        x, y = position, segment.fixed
    return FragmentPoint(
        point_id=f"temporary-{task_type.value}-{x}-{y}-{segment.normal_x}-{segment.normal_y}",
        task_type=task_type,
        x=x,
        y=y,
        normal_x=segment.normal_x,
        normal_y=segment.normal_y,
        support_radius_px=radius,
    )


def _mask_has_local_edge(mask: np.ndarray, point: FragmentPoint) -> bool:
    """要求基准掩模局部窗口同时包含前景和背景，避免移动纯空白块。"""
    radius = point.support_radius_px
    height, width = mask.shape
    patch = mask[
        max(0, point.y - radius):min(height, point.y + radius + 1),
        max(0, point.x - radius):min(width, point.x + radius + 1),
    ]
    return patch.size > 0 and bool(np.any(patch > 0.5)) and bool(np.any(patch <= 0.5))


def _cap_and_relabel(points: List[FragmentPoint], maximum: int, prefix: str) -> List[FragmentPoint]:
    """按空间顺序均匀下采样并重写稳定连续 id。"""
    ordered = sorted(points, key=lambda item: (item.y, item.x, item.normal_y, item.normal_x))
    if len(ordered) > maximum:
        indices = np.linspace(0, len(ordered) - 1, maximum, dtype=np.int64).tolist()
        ordered = [ordered[index] for index in indices]
    return [point.copy(update={"point_id": f"{prefix}-{index:04d}"}) for index, point in enumerate(ordered)]


def _balanced_lengths(total: int, count: int) -> List[int]:
    """把总长度稳定地均分为 count 段，前面的段最多多一个像素。"""
    quotient, remainder = divmod(total, count)
    return [quotient + (1 if index < remainder else 0) for index in range(count)]


def _adaptive_segment_lengths(length: int, settings: SamplingSettings) -> List[int]:
    """生成兼顾转角短段和长直边覆盖的 v3 自适应分段长度。"""
    target = settings.v3_target_segment_length_px
    minimum = settings.v3_min_segment_length_px
    corner = settings.v3_corner_segment_length_px
    if length <= target:
        return [length]
    if length >= 2 * corner + minimum:
        interior = length - 2 * corner
        interior_count = max(1, int(math.ceil(interior / target)))
        lengths = [corner, *_balanced_lengths(interior, interior_count), corner]
    else:
        lengths = _balanced_lengths(length, int(math.ceil(length / target)))
    if min(lengths) < minimum or max(lengths) > target:
        lengths = _balanced_lengths(length, int(math.ceil(length / target)))
    return lengths


def _split_boundary_segment(segment: BoundarySegment, settings: SamplingSettings) -> List[BoundarySegment]:
    """按 v3 自适应长度把最大连续边界切成互不重叠、无间隙的子边段。"""
    cursor = segment.start
    result = []
    for length in _adaptive_segment_lengths(segment.length, settings):
        end = cursor + length - 1
        result.append(BoundarySegment(
            segment.orientation, segment.fixed, cursor, end, segment.normal_x, segment.normal_y
        ))
        cursor = end + 1
    return result


def _v3_point_on_segment(segment: BoundarySegment, radius: int) -> FragmentPoint:
    """在一个 v3 子边段中心建立唯一 EPE 测量点，并携带完整边段端点。"""
    midpoint = (segment.start + segment.end) // 2
    if segment.orientation == "vertical":
        x, y = segment.fixed, midpoint
        start_x, start_y, end_x, end_y = segment.fixed, segment.start, segment.fixed, segment.end
    else:
        x, y = midpoint, segment.fixed
        start_x, start_y, end_x, end_y = segment.start, segment.fixed, segment.end, segment.fixed
    return FragmentPoint(
        point_id=f"temporary-epe-segment-{start_x}-{start_y}-{end_x}-{end_y}",
        task_type=TaskType.EPE,
        x=x,
        y=y,
        normal_x=segment.normal_x,
        normal_y=segment.normal_y,
        support_radius_px=radius,
        segment_start_x=start_x,
        segment_start_y=start_y,
        segment_end_x=end_x,
        segment_end_y=end_y,
    )


def _sample_v3_segments(
    source: ClipImageSource, settings: SamplingSettings, target: np.ndarray, mask: np.ndarray
) -> CandidateManifest:
    """从基准掩模生成 v3 动作边段；target 仅作为后续 OpenILT EPE 优化目标。"""
    height, width = mask.shape
    displacement_px = int(math.ceil(settings.max_displacement_nm / source.scale_nm_per_pixel))
    points: List[FragmentPoint] = []
    unsafe_segments = 0
    invalid_mask_edges = 0
    exclusions = []
    raw_segments = extract_axis_boundary_segments(mask)
    raw_boundary_units = sum(segment.length for segment in raw_segments)
    excluded_short_units = 0
    excluded_collision_units = 0
    for segment in raw_segments:
        if segment.orientation == "vertical":
            fixed_valid = displacement_px <= segment.fixed <= width - displacement_px - 1
        else:
            fixed_valid = displacement_px <= segment.fixed <= height - displacement_px - 1
        if not fixed_valid:
            unsafe_segments += 1
            continue
        if segment.length < settings.v3_min_segment_length_px:
            excluded_short_units += segment.length
            exclusions.append({
                "reason": "shorter-than-v3-min-segment",
                "orientation": segment.orientation,
                "fixed": segment.fixed,
                "start": segment.start,
                "end": segment.end,
                "normal_x": segment.normal_x,
                "normal_y": segment.normal_y,
                "length_px": segment.length,
            })
            continue
        for child in _split_boundary_segment(segment, settings):
            point = _v3_point_on_segment(child, settings.support_radius_px)
            if not _mask_has_local_edge(mask, point):
                invalid_mask_edges += 1
                continue
            try:
                validate_distinct_candidate_actions(
                    mask, point, source.scale_nm_per_pixel, ADAPTER_V3
                )
            except ValueError as exc:
                excluded_collision_units += child.length
                exclusions.append({
                    "reason": "non-unique-nine-actions",
                    "orientation": child.orientation,
                    "fixed": child.fixed,
                    "start": child.start,
                    "end": child.end,
                    "normal_x": child.normal_x,
                    "normal_y": child.normal_y,
                    "length_px": child.length,
                    "detail": str(exc),
                })
                continue
            points.append(point)
    if unsafe_segments or invalid_mask_edges:
        raise ValueError(
            f"{source.clip_id} 的 v3 基准掩模边界完整覆盖失败：越界风险边段={unsafe_segments}，"
            f"异常局部边界子段={invalid_mask_edges}；拒绝静默丢边"
        )
    ordered = sorted(
        points,
        key=lambda item: (
            item.y, item.x, item.normal_y, item.normal_x,
            item.segment_start_y, item.segment_start_x,
        ),
    )
    if not ordered:
        raise ValueError(f"{source.clip_id} 未采到可移动 v3 EPE 边段；请核验图像或参数")
    points = [point.copy(update={"point_id": f"epe-{index:04d}"}) for index, point in enumerate(ordered)]
    eligible_boundary_units = sum(
        max(
            abs(point.segment_end_x - point.segment_start_x),
            abs(point.segment_end_y - point.segment_start_y),
        ) + 1
        for point in points
    )
    accounted_boundary_units = (
        eligible_boundary_units + excluded_short_units + excluded_collision_units
    )
    if accounted_boundary_units != raw_boundary_units:
        raise RuntimeError(
            f"{source.clip_id} v3 边界核算不守恒：raw={raw_boundary_units}，"
            f"accounted={accounted_boundary_units}"
        )
    sampling_audit = {
        "raw_segments": len(raw_segments),
        "raw_boundary_units": raw_boundary_units,
        "eligible_segments": len(points),
        "eligible_boundary_units": eligible_boundary_units,
        "excluded_short_segments": sum(
            item["reason"] == "shorter-than-v3-min-segment" for item in exclusions
        ),
        "excluded_short_boundary_units": excluded_short_units,
        "excluded_collision_segments": sum(
            item["reason"] == "non-unique-nine-actions" for item in exclusions
        ),
        "excluded_collision_boundary_units": excluded_collision_units,
        "accounted_boundary_units": accounted_boundary_units,
    }
    return CandidateManifest(
        adapter_version=ADAPTER_V3,
        clip_id=source.clip_id,
        parent_layout=source.parent_layout,
        split=source.split,
        target_path=source.target_path,
        base_mask_path=source.base_mask_path,
        scale_nm_per_pixel=source.scale_nm_per_pixel,
        points=points,
        sampling_audit=sampling_audit,
        sampling_exclusions=exclusions,
    )


def sample_clip_points(source: ClipImageSource, settings: SamplingSettings) -> CandidateManifest:
    """按显式采样器版本生成 v2 EPE/FRAG 点或 v3 EPE 边段清单。"""
    target = _load_binary(Path(source.target_path))
    mask = _load_binary(Path(source.base_mask_path))
    if target.shape != mask.shape:
        raise ValueError(f"{source.clip_id} 的目标图和掩模尺寸不一致")
    if settings.sampler_version == SAMPLER_V3:
        return _sample_v3_segments(source, settings, target, mask)
    height, width = target.shape
    displacement_px = int(math.ceil(settings.max_displacement_nm / source.scale_nm_per_pixel))
    margin = displacement_px + settings.support_radius_px + 1
    epe_points: List[FragmentPoint] = []
    frag_points: List[FragmentPoint] = []
    for segment in extract_axis_boundary_segments(target):
        if segment.length < settings.min_segment_length_px:
            continue
        if segment.orientation == "vertical":
            clipped_start, clipped_end = max(segment.start, margin), min(segment.end, height - margin - 1)
            fixed_valid = margin <= segment.fixed < width - margin
        else:
            clipped_start, clipped_end = max(segment.start, margin), min(segment.end, width - margin - 1)
            fixed_valid = margin <= segment.fixed < height - margin
        if not fixed_valid or clipped_start > clipped_end:
            continue
        clipped = BoundarySegment(
            segment.orientation, segment.fixed, clipped_start, clipped_end, segment.normal_x, segment.normal_y
        )
        for position in _positions(clipped.start, clipped.end, settings.epe_spacing_px, internal=False):
            point = _point_on_segment(clipped, position, TaskType.EPE, settings.support_radius_px)
            if _mask_has_local_edge(mask, point):
                epe_points.append(point)
        for position in _positions(clipped.start, clipped.end, settings.frag_spacing_px, internal=True):
            point = _point_on_segment(clipped, position, TaskType.FRAG, settings.support_radius_px)
            if _mask_has_local_edge(mask, point):
                frag_points.append(point)
    epe_points = _cap_and_relabel(epe_points, settings.max_epe_points, "epe")
    frag_points = _cap_and_relabel(frag_points, settings.max_frag_points, "frag")
    if not epe_points or not frag_points:
        raise ValueError(
            f"{source.clip_id} 采样不足：EPE={len(epe_points)}，FRAG={len(frag_points)}；请核验图像或参数"
        )
    return CandidateManifest(
        clip_id=source.clip_id,
        parent_layout=source.parent_layout,
        split=source.split,
        target_path=source.target_path,
        base_mask_path=source.base_mask_path,
        scale_nm_per_pixel=source.scale_nm_per_pixel,
        points=epe_points + frag_points,
    )


def _sha256(path: Path) -> str:
    """计算候选文件哈希。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_sources(sources: Sequence[ClipImageSource]) -> None:
    """拒绝重复 clip 和父版图跨集合泄漏。"""
    if not sources:
        raise ValueError("多 clip 源不能为空")
    clip_ids = [source.clip_id for source in sources]
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError("多 clip 源存在重复 clip_id")
    parent_splits: Dict[str, str] = {}
    for source in sources:
        previous = parent_splits.setdefault(source.parent_layout, source.split)
        if previous != source.split:
            raise ValueError(f"父版图 {source.parent_layout} 跨数据集泄漏")


def build_candidate_index(
    sources: Sequence[ClipImageSource],
    settings: SamplingSettings,
    output_dir: Path,
) -> CandidateIndex:
    """逐 clip 生成紧凑候选文件，并建立可恢复且不可混用的哈希索引。"""
    _validate_sources(sources)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for source in sorted(sources, key=lambda item: item.clip_id):
        manifest = sample_clip_points(source, settings)
        manifest_path = root / f"{source.clip_id}.manifest.json"
        dataset_path = root / f"{source.clip_id}.npz"
        metadata_path = root / f"{source.clip_id}.metadata.json"
        present = [path.exists() for path in (manifest_path, dataset_path, metadata_path)]
        if any(present) and not all(present):
            raise RuntimeError(f"{source.clip_id} 候选文件仅部分存在；拒绝静默续写")
        manifest_text = json.dumps(manifest.dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if all(present):
            if manifest_path.read_text(encoding="utf-8") != manifest_text:
                raise FileExistsError(f"{source.clip_id} 已有 manifest 与当前采样参数不同")
        else:
            observations, target, base_mask, geometry, metadata = build_compact_candidate_data(manifest)
            manifest_path.write_text(manifest_text, encoding="utf-8")
            np.savez_compressed(
                str(dataset_path), observations=observations, target=target, base_mask=base_mask,
                point_geometry=geometry, scale_nm_per_pixel=np.asarray(manifest.scale_nm_per_pixel),
                adapter_version=np.asarray(manifest.adapter_version),
            )
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        epe_count = sum(point.task_type == TaskType.EPE for point in manifest.points)
        frag_count = sum(point.task_type == TaskType.FRAG for point in manifest.points)
        entries.append(CandidateIndexEntry(
            clip_id=source.clip_id,
            parent_layout=source.parent_layout,
            split=source.split,
            dataset_path=str(dataset_path),
            metadata_path=str(metadata_path),
            manifest_path=str(manifest_path),
            dataset_sha256=_sha256(dataset_path),
            epe_points=epe_count,
            frag_points=frag_count,
            excluded_segments=(
                manifest.sampling_audit.get("excluded_short_segments", 0)
                + manifest.sampling_audit.get("excluded_collision_segments", 0)
                + manifest.sampling_audit.get("excluded_pilot_cap_segments", 0)
                if manifest.sampling_audit else 0
            ),
            excluded_boundary_units=(
                manifest.sampling_audit.get("excluded_short_boundary_units", 0)
                + manifest.sampling_audit.get("excluded_collision_boundary_units", 0)
                + manifest.sampling_audit.get("excluded_pilot_cap_boundary_units", 0)
                if manifest.sampling_audit else 0
            ),
        ))
    identity = {
        "sampler_version": settings.sampler_version,
        "settings": settings.dict(),
        "entries": [entry.dict() for entry in entries],
    }
    index_hash = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    index = CandidateIndex(
        sampler_version=settings.sampler_version,
        settings=settings,
        entries=entries,
        index_sha256=index_hash,
    )
    index_path = root / "candidate-index.json"
    encoded = json.dumps(index.dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if index_path.exists() and index_path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError("candidate-index.json 已存在不同版本")
    if not index_path.exists():
        index_path.write_text(encoded, encoding="utf-8")
    return index


def sources_from_config(
    config: dict, image_dir: Path, prefix: str, parents: Optional[List[str]], scale_nm_per_pixel: float
) -> List[ClipImageSource]:
    """按 paper_repro.yaml 6/2/2 划分和 SimpleOPC 文件名生成数据源。"""
    data = config["data"]
    split_map = {
        **{parent: "train" for parent in data["train_parents"]},
        **{parent: "validation" for parent in data["validation_parents"]},
        **{parent: "test" for parent in data["test_parents"]},
    }
    selected = parents if parents else list(split_map)
    unknown = sorted(set(selected).difference(split_map))
    if unknown:
        raise ValueError(f"请求了配置外父版图：{unknown}")
    sources = []
    for parent in selected:
        match = re.fullmatch(r"M1_test(\d+)", parent)
        if match is None:
            raise ValueError(f"无法从父版图名提取编号：{parent}")
        number = match.group(1)
        sources.append(ClipImageSource(
            clip_id=parent,
            parent_layout=parent,
            split=split_map[parent],
            target_path=str(Path(image_dir) / f"{prefix}_target{number}.png"),
            base_mask_path=str(Path(image_dir) / f"{prefix}_mask{number}.png"),
            scale_nm_per_pixel=scale_nm_per_pixel,
        ))
    return sources


def main(argv: Optional[List[str]] = None) -> int:
    """从固定划分和 OpenILT 输出批量生成候选索引。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.point_sampling")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="SimpleOPC")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parents", nargs="*")
    parser.add_argument("--sampler-version", choices=[SAMPLER_V2, SAMPLER_V3], default=SAMPLER_V2)
    parser.add_argument("--epe-spacing", type=int, default=128)
    parser.add_argument("--frag-spacing", type=int, default=192)
    parser.add_argument("--support-radius", type=int, default=8)
    parser.add_argument("--scale-nm-per-pixel", type=float, default=1.0)
    parser.add_argument("--max-epe", type=int, default=8)
    parser.add_argument("--max-frag", type=int, default=8)
    parser.add_argument("--v3-target-segment-length", type=int, default=128)
    parser.add_argument("--v3-min-segment-length", type=int, default=32)
    parser.add_argument("--v3-corner-segment-length", type=int, default=64)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = SamplingSettings(
        sampler_version=args.sampler_version,
        epe_spacing_px=args.epe_spacing,
        frag_spacing_px=args.frag_spacing,
        support_radius_px=args.support_radius,
        max_epe_points=args.max_epe,
        max_frag_points=args.max_frag,
        v3_target_segment_length_px=args.v3_target_segment_length,
        v3_min_segment_length_px=args.v3_min_segment_length,
        v3_corner_segment_length_px=args.v3_corner_segment_length,
    )
    sources = sources_from_config(
        config, args.image_dir, args.prefix, args.parents, args.scale_nm_per_pixel
    )
    index = build_candidate_index(sources, settings, args.output_dir)
    print(args.output_dir / "candidate-index.json")
    print(f"clips={len(index.entries)} epe={sum(item.epe_points for item in index.entries)} "
          f"frag={sum(item.frag_points for item in index.entries)} "
          f"excluded={sum(item.excluded_segments for item in index.entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
