"""本模块把版本化候选点、外法线和九个位移动作渲染为可审计 PNG，并输出机器可读质量摘要。

输入为 point_sampling 生成且通过哈希校验的 candidate-index.json、配套 manifest/NPZ/metadata；输出为
每个 clip 的整图点位总览、逐点局部窗口、九动作差异网格及 visualization-summary.json。关键依赖仅为
OpenCV、NumPy 和项目既有候选数据模型；模块只读取已有数据，不重新采样，不调用 GPU、OpenILT、PPO 或 API。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .candidate_masks import CandidateManifest, FragmentPoint, generate_candidate_mask
from .metrics import DISPLACEMENT_CLASSES_NM
from .models import TaskType
from .oracle_batch_labels import load_candidate_index
from .point_sampling import extract_axis_boundary_segments


EPE_COLOR = (0, 0, 255)
FRAG_COLOR = (255, 170, 0)
NORMAL_COLOR = (0, 255, 0)
SEGMENT_COLOR = (0, 165, 255)
MASK_CONTOUR_COLOR = (0, 255, 255)
TARGET_CONTOUR_COLOR = (255, 255, 0)


def _encode_png(image: np.ndarray) -> bytes:
    """编码 PNG，避免 OpenCV 在中文路径上直接写文件失败。"""
    ok, encoded = cv2.imencode(".png", np.asarray(image, dtype=np.uint8))
    if not ok:
        raise RuntimeError("OpenCV 无法编码可视化 PNG")
    return encoded.tobytes()


def _write_derived(path: Path, content: bytes) -> None:
    """派生文件允许同内容幂等执行，不同内容要求使用新输出目录。"""
    destination = Path(path)
    if destination.exists():
        if destination.read_bytes() != content:
            raise FileExistsError(f"拒绝覆盖已有不同可视化：{destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def _binary_contours(binary: np.ndarray):
    """提取二值前景轮廓，兼容不同 OpenCV 返回签名。"""
    result = cv2.findContours(
        (np.asarray(binary) > 0.5).astype(np.uint8) * 255,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return result[-2]


def _base_canvas(target: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    """生成灰色 target 填充与黄色 base mask 轮廓的 BGR 底图。"""
    canvas = np.zeros((*target.shape, 3), dtype=np.uint8)
    canvas[np.asarray(target) > 0.5] = (75, 75, 75)
    cv2.drawContours(canvas, _binary_contours(base_mask), -1, MASK_CONTOUR_COLOR, 1)
    return canvas


def _point_color(point: FragmentPoint) -> Tuple[int, int, int]:
    """为 EPE/FRAG 点返回稳定且高对比度的 BGR 颜色。"""
    return EPE_COLOR if point.task_type == TaskType.EPE else FRAG_COLOR


def _draw_point(
    image: np.ndarray,
    point: FragmentPoint,
    origin: Tuple[int, int] = (0, 0),
    with_label: bool = True,
) -> None:
    """在给定裁剪原点下绘制 v3 边段、中心测量点、编号和外法线箭头。"""
    origin_x, origin_y = origin
    x, y = int(point.x - origin_x), int(point.y - origin_y)
    color = _point_color(point)
    if point.has_segment_geometry:
        cv2.line(
            image,
            (int(point.segment_start_x - origin_x), int(point.segment_start_y - origin_y)),
            (int(point.segment_end_x - origin_x), int(point.segment_end_y - origin_y)),
            SEGMENT_COLOR,
            2,
            lineType=cv2.LINE_AA,
        )
    marker_radius = max(3, int(round(min(image.shape[:2]) / 256)))
    if point.task_type == TaskType.EPE:
        cv2.circle(image, (x, y), marker_radius, color, -1, lineType=cv2.LINE_AA)
    else:
        cv2.rectangle(
            image,
            (x - marker_radius, y - marker_radius),
            (x + marker_radius, y + marker_radius),
            color,
            -1,
            lineType=cv2.LINE_AA,
        )
    arrow_length = max(12, marker_radius * 5)
    endpoint = (x + point.normal_x * arrow_length, y + point.normal_y * arrow_length)
    cv2.arrowedLine(image, (x, y), endpoint, NORMAL_COLOR, 2, tipLength=0.35)
    if with_label:
        cv2.putText(
            image,
            point.point_id,
            (x + 5, max(12, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )


def render_overview(target: np.ndarray, base_mask: np.ndarray, points: List[FragmentPoint]) -> np.ndarray:
    """渲染一个 clip 的整图 target、mask 轮廓、EPE/FRAG 点和外法线。"""
    canvas = _base_canvas(target, base_mask)
    for point in points:
        _draw_point(canvas, point)
    cv2.putText(canvas, "EPE: red circle", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, EPE_COLOR, 1)
    cv2.putText(canvas, "FRAG: blue square", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FRAG_COLOR, 1)
    cv2.putText(canvas, "normal: green arrow", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, NORMAL_COLOR, 1)
    if any(point.has_segment_geometry for point in points):
        cv2.putText(canvas, "v3 segment: orange line", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SEGMENT_COLOR, 1)
    return canvas


def _crop_bounds(
    point: FragmentPoint,
    shape: Tuple[int, int],
    scale_nm_per_pixel: float,
) -> Tuple[int, int, int, int]:
    """返回覆盖 support 区域和最大位移的局部裁剪范围。"""
    height, width = shape
    max_displacement_px = int(round(max(abs(value) for value in DISPLACEMENT_CLASSES_NM) / scale_nm_per_pixel))
    radius = max(point.support_radius_px * 3, point.support_radius_px + max_displacement_px + 4)
    anchor_x = [point.x]
    anchor_y = [point.y]
    if point.has_segment_geometry:
        anchor_x.extend([point.segment_start_x, point.segment_end_x])
        anchor_y.extend([point.segment_start_y, point.segment_end_y])
    x0, x1 = max(0, min(anchor_x) - radius), min(width, max(anchor_x) + radius + 1)
    y0, y1 = max(0, min(anchor_y) - radius), min(height, max(anchor_y) + radius + 1)
    return x0, y0, x1, y1


def _resize_nearest(image: np.ndarray, minimum: int = 320) -> np.ndarray:
    """以最近邻放大局部栅格，保证像素变化清晰可见。"""
    height, width = image.shape[:2]
    scale = max(1, int(np.ceil(minimum / max(1, min(height, width)))))
    return cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)


def render_local_patch(
    target: np.ndarray,
    base_mask: np.ndarray,
    point: FragmentPoint,
    scale_nm_per_pixel: float,
) -> np.ndarray:
    """渲染单点局部 target/mask、支持窗口、坐标和外法线。"""
    x0, y0, x1, y1 = _crop_bounds(point, target.shape, scale_nm_per_pixel)
    canvas = _base_canvas(target[y0:y1, x0:x1], base_mask[y0:y1, x0:x1])
    local_x, local_y = point.x - x0, point.y - y0
    radius = point.support_radius_px
    cv2.rectangle(
        canvas,
        (max(0, local_x - radius), max(0, local_y - radius)),
        (min(canvas.shape[1] - 1, local_x + radius), min(canvas.shape[0] - 1, local_y + radius)),
        (255, 0, 255),
        1,
    )
    _draw_point(canvas, point, origin=(x0, y0), with_label=False)
    enlarged = _resize_nearest(canvas)
    header = np.zeros((42, enlarged.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"{point.point_id} ({point.x},{point.y}) normal=({point.normal_x},{point.normal_y})",
        (6, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, enlarged])


def _action_tile(
    target_crop: np.ndarray,
    base_crop: np.ndarray,
    candidate_crop: np.ndarray,
    displacement_nm: int,
) -> np.ndarray:
    """用绿色新增、红色删除和灰白前景渲染一个动作差异块。"""
    base = np.asarray(base_crop) > 0.5
    candidate = np.asarray(candidate_crop) > 0.5
    tile = np.zeros((*base.shape, 3), dtype=np.uint8)
    tile[candidate] = (180, 180, 180)
    tile[(~base) & candidate] = (0, 255, 0)
    tile[base & (~candidate)] = (0, 0, 255)
    cv2.drawContours(tile, _binary_contours(target_crop), -1, TARGET_CONTOUR_COLOR, 1)
    enlarged = _resize_nearest(tile, minimum=210)
    header = np.zeros((30, enlarged.shape[1], 3), dtype=np.uint8)
    label = f"{displacement_nm:+d} nm" if displacement_nm else "0 nm"
    cv2.putText(header, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, enlarged])


def render_action_grid(
    target: np.ndarray,
    base_mask: np.ndarray,
    point: FragmentPoint,
    scale_nm_per_pixel: float,
    adapter_version: str,
) -> np.ndarray:
    """渲染单点九个位移动作的 3×3 差异网格。"""
    x0, y0, x1, y1 = _crop_bounds(point, target.shape, scale_nm_per_pixel)
    target_crop = target[y0:y1, x0:x1]
    base_crop = base_mask[y0:y1, x0:x1]
    tiles = []
    for displacement_nm in DISPLACEMENT_CLASSES_NM:
        displacement_px = int(round(displacement_nm / scale_nm_per_pixel))
        candidate = generate_candidate_mask(base_mask, point, displacement_px, adapter_version)
        tiles.append(_action_tile(
            target_crop,
            base_crop,
            candidate[y0:y1, x0:x1],
            displacement_nm,
        ))
    cell_height = max(tile.shape[0] for tile in tiles)
    cell_width = max(tile.shape[1] for tile in tiles)
    normalized = []
    for tile in tiles:
        cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
        cell[:tile.shape[0], :tile.shape[1]] = tile
        normalized.append(cell)
    rows = [np.hstack(normalized[index:index + 3]) for index in range(0, 9, 3)]
    grid = np.vstack(rows)
    title = np.zeros((42, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        title,
        f"{point.point_id}: green=added red=removed cyan=target",
        (8, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([title, grid])


def _boundary_status(binary: np.ndarray, point: FragmentPoint) -> Tuple[bool, bool]:
    """检查点是否在给定二值图四邻域边界上，以及法线是否由前景指向背景。"""
    height, width = binary.shape
    if not (0 <= point.x < width and 0 <= point.y < height):
        return False, False
    center = bool(binary[point.y, point.x] > 0.5)
    neighbors = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        x, y = point.x + dx, point.y + dy
        if 0 <= x < width and 0 <= y < height:
            neighbors.append(bool(binary[y, x] > 0.5))
    on_boundary = any(value != center for value in neighbors)
    outward_x, outward_y = point.x + point.normal_x, point.y + point.normal_y
    normal_outward = (
        center
        and 0 <= outward_x < width
        and 0 <= outward_y < height
        and not bool(binary[outward_y, outward_x] > 0.5)
    )
    return on_boundary, normal_outward


def _segment_boundary_status(binary: np.ndarray, point: FragmentPoint) -> Tuple[bool, bool]:
    """检查 v3 整条边段是否都在给定二值图边界且法线逐像素指向背景。"""
    if not point.has_segment_geometry:
        return True, True
    if point.normal_x:
        positions = [
            (point.x, y)
            for y in range(min(point.segment_start_y, point.segment_end_y), max(point.segment_start_y, point.segment_end_y) + 1)
        ]
    else:
        positions = [
            (x, point.y)
            for x in range(min(point.segment_start_x, point.segment_end_x), max(point.segment_start_x, point.segment_end_x) + 1)
        ]
    height, width = binary.shape
    on_boundary = True
    normal_outward = True
    for x, y in positions:
        outward_x, outward_y = x + point.normal_x, y + point.normal_y
        if not (0 <= x < width and 0 <= y < height and binary[y, x] > 0.5):
            on_boundary = False
        if not (
            0 <= outward_x < width
            and 0 <= outward_y < height
            and binary[y, x] > 0.5
            and binary[outward_y, outward_x] <= 0.5
        ):
            normal_outward = False
    return on_boundary, normal_outward


def _mask_edge_in_support(base_mask: np.ndarray, point: FragmentPoint) -> bool:
    """检查点的支持窗口内是否同时包含基准掩模前景和背景。"""
    radius = point.support_radius_px
    y0, y1 = max(0, point.y - radius), min(base_mask.shape[0], point.y + radius + 1)
    x0, x1 = max(0, point.x - radius), min(base_mask.shape[1], point.x + radius + 1)
    patch = base_mask[y0:y1, x0:x1]
    return patch.size > 0 and bool(np.any(patch > 0.5)) and bool(np.any(patch <= 0.5))


def inspect_point(
    target: np.ndarray,
    base_mask: np.ndarray,
    point: FragmentPoint,
    scale_nm_per_pixel: float,
    adapter_version: str,
) -> Dict[str, object]:
    """检查点位、法线、动作唯一性、面积单调性和修改局部性。"""
    on_target_boundary, target_normal_outward = _boundary_status(target, point)
    action_reference = base_mask if point.has_segment_geometry else target
    on_action_boundary, action_normal_outward = _boundary_status(action_reference, point)
    segment_on_target, _ = _segment_boundary_status(target, point)
    segment_on_action, segment_normal_outward = _segment_boundary_status(action_reference, point)
    candidates = []
    hashes = set()
    for displacement_nm in DISPLACEMENT_CLASSES_NM:
        displacement_px = int(round(displacement_nm / scale_nm_per_pixel))
        candidate = generate_candidate_mask(base_mask, point, displacement_px, adapter_version)
        candidates.append(candidate)
        hashes.add(hashlib.sha256(np.packbits(candidate > 0.5).tobytes()).hexdigest())
    areas = [int(np.count_nonzero(candidate > 0.5)) for candidate in candidates]
    zero_index = DISPLACEMENT_CLASSES_NM.index(0)
    zero_area = areas[zero_index]
    monotonic = areas == sorted(areas)
    negative_shrink = monotonic and all(area < zero_area for area in areas[:zero_index])
    positive_expand = monotonic and all(area > zero_area for area in areas[zero_index + 1:])

    max_displacement_px = int(round(max(abs(value) for value in DISPLACEMENT_CLASSES_NM) / scale_nm_per_pixel))
    if point.normal_x:
        if point.has_segment_geometry:
            allowed_y0, allowed_y1 = sorted((point.segment_start_y, point.segment_end_y))
        else:
            allowed_y0, allowed_y1 = point.y - point.support_radius_px, point.y + point.support_radius_px
        allowed_x0, allowed_x1 = point.x - max_displacement_px, point.x + max_displacement_px
    else:
        allowed_y0, allowed_y1 = point.y - max_displacement_px, point.y + max_displacement_px
        if point.has_segment_geometry:
            allowed_x0, allowed_x1 = sorted((point.segment_start_x, point.segment_end_x))
        else:
            allowed_x0, allowed_x1 = point.x - point.support_radius_px, point.x + point.support_radius_px
    local_changes_only = True
    base_binary = base_mask > 0.5
    for candidate in candidates:
        changed_y, changed_x = np.nonzero((candidate > 0.5) != base_binary)
        if changed_y.size and not (
            np.all((allowed_y0 <= changed_y) & (changed_y <= allowed_y1))
            and np.all((allowed_x0 <= changed_x) & (changed_x <= allowed_x1))
        ):
            local_changes_only = False
            break

    result: Dict[str, object] = {
        "point_id": point.point_id,
        "task_type": point.task_type.value,
        "x": point.x,
        "y": point.y,
        "normal_x": point.normal_x,
        "normal_y": point.normal_y,
        "on_target_boundary": on_target_boundary,
        "target_normal_points_outward": target_normal_outward,
        "on_action_boundary": on_action_boundary,
        "mask_edge_in_support": _mask_edge_in_support(base_mask, point),
        "normal_points_outward": action_normal_outward,
        "has_segment_geometry": point.has_segment_geometry,
        "segment_on_target_boundary": segment_on_target,
        "segment_on_action_boundary": segment_on_action,
        "segment_normal_points_outward": segment_normal_outward,
        "unique_action_masks": len(hashes),
        "action_areas_px": areas,
        "negative_actions_shrink": negative_shrink,
        "positive_actions_expand": positive_expand,
        "local_changes_only": local_changes_only,
    }
    result["passed"] = all((
        result["on_action_boundary"],
        result["mask_edge_in_support"],
        result["normal_points_outward"],
        result["segment_on_action_boundary"],
        result["segment_normal_points_outward"],
        result["unique_action_masks"] == len(DISPLACEMENT_CLASSES_NM),
        result["negative_actions_shrink"],
        result["positive_actions_expand"],
        result["local_changes_only"],
    ))
    return result


def _load_clip(entry) -> Tuple[CandidateManifest, np.ndarray, np.ndarray, str, float]:
    """读取并交叉校验 manifest、metadata 和紧凑 NPZ 的点顺序与几何。"""
    manifest = CandidateManifest.parse_obj(json.loads(Path(entry.manifest_path).read_text(encoding="utf-8")))
    metadata = json.loads(Path(entry.metadata_path).read_text(encoding="utf-8"))
    with np.load(str(entry.dataset_path), allow_pickle=False) as payload:
        required = {"target", "base_mask", "point_geometry", "adapter_version", "scale_nm_per_pixel"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{entry.clip_id} 候选 NPZ 缺少字段：{sorted(missing)}")
        target = np.asarray(payload["target"], dtype=np.float32)
        base_mask = np.asarray(payload["base_mask"], dtype=np.float32)
        geometry = np.asarray(payload["point_geometry"], dtype=np.int32)
        adapter_version = str(np.asarray(payload["adapter_version"]).item())
        scale_nm_per_pixel = float(np.asarray(payload["scale_nm_per_pixel"]).item())
    if adapter_version == "raster-edge-segment-v3":
        expected_geometry = np.asarray([
            [
                point.x, point.y, point.normal_x, point.normal_y, point.support_radius_px,
                point.segment_start_x, point.segment_start_y, point.segment_end_x, point.segment_end_y,
            ]
            for point in manifest.points
        ], dtype=np.int32)
    else:
        expected_geometry = np.asarray([
            [point.x, point.y, point.normal_x, point.normal_y, point.support_radius_px]
            for point in manifest.points
        ], dtype=np.int32)
    if target.ndim != 2 or target.shape != base_mask.shape:
        raise ValueError(f"{entry.clip_id} target/base_mask 必须是同尺寸二维图")
    if not np.array_equal(geometry, expected_geometry):
        raise ValueError(f"{entry.clip_id} manifest 与 NPZ 点几何不一致")
    if metadata.get("point_ids") != [point.point_id for point in manifest.points]:
        raise ValueError(f"{entry.clip_id} metadata 与 manifest 点顺序不一致")
    if metadata.get("sampling_audit") != manifest.sampling_audit:
        raise ValueError(f"{entry.clip_id} metadata 与 manifest 采样审计不一致")
    if adapter_version != manifest.adapter_version:
        raise ValueError(f"{entry.clip_id} adapter_version 不一致")
    if not np.isclose(scale_nm_per_pixel, manifest.scale_nm_per_pixel):
        raise ValueError(f"{entry.clip_id} scale_nm_per_pixel 不一致")
    return manifest, target, base_mask, adapter_version, scale_nm_per_pixel


def _axis_boundary_coverage(
    reference: np.ndarray, points: List[FragmentPoint], reference_name: str
) -> Dict[str, object]:
    """计算 v3 边段对动作参考图全部轴向边界单元的覆盖率。"""
    if not any(point.has_segment_geometry for point in points):
        return {
            "reference_axis_boundary_units": None,
            "covered_axis_boundary_units": None,
            "unexpected_axis_boundary_units": None,
            "axis_boundary_coverage": None,
            "boundary_reference": None,
        }
    target_units = set()
    for segment in extract_axis_boundary_segments(reference):
        for position in range(segment.start, segment.end + 1):
            if segment.orientation == "vertical":
                unit = (segment.fixed, position, segment.normal_x, segment.normal_y)
            else:
                unit = (position, segment.fixed, segment.normal_x, segment.normal_y)
            target_units.add(unit)
    covered_units = set()
    for point in points:
        if not point.has_segment_geometry:
            continue
        if point.normal_x:
            start, end = sorted((point.segment_start_y, point.segment_end_y))
            for y in range(start, end + 1):
                covered_units.add((point.x, y, point.normal_x, point.normal_y))
        else:
            start, end = sorted((point.segment_start_x, point.segment_end_x))
            for x in range(start, end + 1):
                covered_units.add((x, point.y, point.normal_x, point.normal_y))
    matched = target_units.intersection(covered_units)
    return {
        "reference_axis_boundary_units": len(target_units),
        "covered_axis_boundary_units": len(matched),
        "unexpected_axis_boundary_units": len(covered_units.difference(target_units)),
        "axis_boundary_coverage": len(matched) / len(target_units) if target_units else 1.0,
        "boundary_reference": reference_name,
    }


def visualize_candidate_index(
    index_path: Path,
    output_dir: Path,
    overview: bool = True,
    local_patches: bool = True,
    action_grids: bool = True,
) -> Dict[str, object]:
    """验证候选索引并生成所选图片与汇总，返回与磁盘一致的摘要对象。"""
    if not any((overview, local_patches, action_grids)):
        raise ValueError("至少选择一种可视化输出")
    index = load_candidate_index(Path(index_path))
    root = Path(output_dir)
    clip_summaries = []
    all_points = []
    for entry in index.entries:
        manifest, target, base_mask, adapter_version, scale_nm_per_pixel = _load_clip(entry)
        clip_root = root / entry.clip_id
        if overview:
            _write_derived(clip_root / "overview.png", _encode_png(
                render_overview(target, base_mask, manifest.points)
            ))
        point_results = []
        for point in manifest.points:
            result = inspect_point(target, base_mask, point, scale_nm_per_pixel, adapter_version)
            point_results.append(result)
            all_points.append(result)
            if local_patches:
                _write_derived(clip_root / f"{point.point_id}-local.png", _encode_png(
                    render_local_patch(target, base_mask, point, scale_nm_per_pixel)
                ))
            if action_grids:
                _write_derived(clip_root / f"{point.point_id}-actions.png", _encode_png(
                    render_action_grid(target, base_mask, point, scale_nm_per_pixel, adapter_version)
                ))
        if adapter_version == "raster-edge-segment-v3":
            coverage = _axis_boundary_coverage(base_mask, manifest.points, "base_mask")
        else:
            coverage = _axis_boundary_coverage(target, manifest.points, "target")
        reference_units = coverage["reference_axis_boundary_units"]
        if manifest.sampling_audit is not None and reference_units:
            accounted_coverage = (
                manifest.sampling_audit["accounted_boundary_units"] / reference_units
            )
        else:
            accounted_coverage = None
        clip_summaries.append({
            "clip_id": entry.clip_id,
            "split": entry.split,
            "points": len(point_results),
            "passed_points": sum(bool(item["passed"]) for item in point_results),
            "failed_points": [item["point_id"] for item in point_results if not item["passed"]],
            "sampling_audit": manifest.sampling_audit,
            "sampling_exclusions": manifest.sampling_exclusions,
            "accounted_axis_boundary_coverage": accounted_coverage,
            **coverage,
        })
    summary: Dict[str, object] = {
        "schema_version": "1.0",
        "candidate_index": str(index_path),
        "candidate_index_sha256": index.index_sha256,
        "clips": len(index.entries),
        "points": len(all_points),
        "passed_points": sum(bool(item["passed"]) for item in all_points),
        "segments": sum(bool(item["has_segment_geometry"]) for item in all_points),
        "passed_segments": sum(
            bool(item["has_segment_geometry"] and item["passed"]) for item in all_points
        ),
        "minimum_axis_boundary_coverage": min(
            (
                float(item["axis_boundary_coverage"])
                for item in clip_summaries
                if item["axis_boundary_coverage"] is not None
            ),
            default=None,
        ),
        "boundary_references": sorted({
            str(item["boundary_reference"])
            for item in clip_summaries
            if item["boundary_reference"] is not None
        }),
        "minimum_accounted_axis_boundary_coverage": min(
            (
                float(item["accounted_axis_boundary_coverage"])
                for item in clip_summaries
                if item["accounted_axis_boundary_coverage"] is not None
            ),
            default=None,
        ),
        "excluded_segments": sum(
            len(item["sampling_exclusions"]) for item in clip_summaries
        ),
        "failed_points": [
            {"point_id": item["point_id"], "task_type": item["task_type"], "x": item["x"], "y": item["y"]}
            for item in all_points if not item["passed"]
        ],
        "clip_summaries": clip_summaries,
        "point_checks": all_points,
    }
    encoded = (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_derived(root / "visualization-summary.json", encoded)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    """从命令行生成候选点可视化；未指定图片类别时默认全部生成。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.point_visualization")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overview", action="store_true")
    parser.add_argument("--local-patches", action="store_true")
    parser.add_argument("--action-grids", action="store_true")
    args = parser.parse_args(argv)
    selected = any((args.overview, args.local_patches, args.action_grids))
    summary = visualize_candidate_index(
        args.index,
        args.output_dir,
        overview=args.overview or not selected,
        local_patches=args.local_patches or not selected,
        action_grids=args.action_grids or not selected,
    )
    print(f"clips={summary['clips']} points={summary['points']} "
          f"passed={summary['passed_points']} failed={len(summary['failed_points'])}")
    print(Path(args.output_dir) / "visualization-summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
