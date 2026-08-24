"""本模块验证正交边界段提取、EPE/FRAG 确定性采样、点数限制和父版图防泄漏。

输入为临时矩形目标/掩模图与小型多 clip 源；输出为稳定点坐标、紧凑候选索引和非法划分断言。
关键依赖为 OpenCV、NumPy、Pydantic 与 pytest；测试不调用 GPU、OpenILT 仿真、PPO、网络或 API。
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from opc_agent.point_sampling import (
    ClipImageSource,
    SamplingSettings,
    build_candidate_index,
    extract_axis_boundary_segments,
    sample_clip_points,
    SAMPLER_V3,
)


def _write_png(path: Path, image: np.ndarray) -> None:
    """通过 Python 文件 API 写 PNG，兼容中文临时目录。"""
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())


def _source(tmp_path: Path, clip_id: str, parent: str, split: str) -> ClipImageSource:
    """构造有足够安全边距的矩形目标和一致掩模。"""
    image = np.zeros((128, 128), dtype=np.uint8)
    image[24:104, 28:100] = 255
    target = tmp_path / f"{clip_id}-target.png"
    mask = tmp_path / f"{clip_id}-mask.png"
    _write_png(target, image)
    _write_png(mask, image)
    return ClipImageSource(
        clip_id=clip_id,
        parent_layout=parent,
        split=split,
        target_path=str(target),
        base_mask_path=str(mask),
        scale_nm_per_pixel=10,
    )


def _settings() -> SamplingSettings:
    """使用小图可覆盖的采样间距和位移比例。"""
    return SamplingSettings(
        epe_spacing_px=16,
        frag_spacing_px=20,
        support_radius_px=3,
        min_segment_length_px=12,
        max_epe_points=6,
        max_frag_points=4,
    )


def test_extract_segments_and_sample_both_tasks_deterministically(tmp_path: Path):
    """矩形必须产生四向边界，重复采样应得到相同 EPE/FRAG 点。"""
    source = _source(tmp_path, "clip-1", "M1_test1", "train")
    encoded = np.fromfile(source.target_path, dtype=np.uint8)
    binary = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE) > 127
    segments = extract_axis_boundary_segments(binary)
    assert {segment.orientation for segment in segments} == {"horizontal", "vertical"}
    first = sample_clip_points(source, _settings())
    second = sample_clip_points(source, _settings())
    assert [point.dict() for point in first.points] == [point.dict() for point in second.points]
    assert sum(point.task_type.value == "EPE" for point in first.points) == 6
    assert sum(point.task_type.value == "FRAG" for point in first.points) == 4
    assert all(abs(point.normal_x) + abs(point.normal_y) == 1 for point in first.points)


def test_candidate_index_writes_compact_files_and_rejects_split_leakage(tmp_path: Path):
    """索引应写入两个紧凑数据集，并拒绝同一父版图跨集合。"""
    first = _source(tmp_path, "clip-a", "M1_test1", "train")
    second = _source(tmp_path, "clip-b", "M1_test7", "validation")
    index = build_candidate_index([first, second], _settings(), tmp_path / "candidates")
    assert len(index.entries) == 2
    assert all(Path(entry.dataset_path).is_file() for entry in index.entries)
    with np.load(index.entries[0].dataset_path, allow_pickle=False) as payload:
        assert "candidate_masks" not in payload.files
        assert payload["point_geometry"].shape[1] == 5
        assert str(payload["adapter_version"].item()) == "raster-boundary-strip-v2"

    leaked = second.copy(update={"parent_layout": "M1_test1"})
    with pytest.raises(ValueError, match="跨数据集泄漏"):
        build_candidate_index([first, leaked], _settings(), tmp_path / "leaked")


def test_v3_adaptive_segments_have_one_center_point_and_no_frag_cap(tmp_path: Path):
    """v3 应覆盖全部自适应子边段，每段一个中心 EPE 点，且不套用 v2 的全局点数上限。"""
    source = _source(tmp_path, "clip-v3", "M1_test1", "train")
    settings = SamplingSettings(
        sampler_version=SAMPLER_V3,
        support_radius_px=3,
        max_epe_points=1,
        max_frag_points=1,
        v3_target_segment_length_px=32,
        v3_min_segment_length_px=12,
        v3_corner_segment_length_px=16,
    )
    manifest = sample_clip_points(source, settings)
    assert manifest.adapter_version == "raster-edge-segment-v3"
    assert len(manifest.points) > settings.max_epe_points
    assert all(point.task_type.value == "EPE" for point in manifest.points)
    assert all(point.has_segment_geometry for point in manifest.points)
    lengths = []
    for point in manifest.points:
        length = max(
            abs(point.segment_end_x - point.segment_start_x),
            abs(point.segment_end_y - point.segment_start_y),
        ) + 1
        lengths.append(length)
        assert point.x == (point.segment_start_x + point.segment_end_x) // 2
        assert point.y == (point.segment_start_y + point.segment_end_y) // 2
    assert max(lengths) <= settings.v3_target_segment_length_px

    index = build_candidate_index([source], settings, tmp_path / "v3-candidates")
    assert index.sampler_version == SAMPLER_V3
    assert index.entries[0].frag_points == 0
    with np.load(index.entries[0].dataset_path, allow_pickle=False) as payload:
        assert payload["point_geometry"].shape == (len(manifest.points), 9)
        assert str(payload["adapter_version"].item()) == "raster-edge-segment-v3"


def test_v3_audits_short_edges_and_rejects_unaccounted_boundary_loss(tmp_path: Path):
    """短边可排除但必须完整核算；无法安全移动的贴边图形仍应显式失败。"""
    image = np.zeros((128, 128), dtype=np.uint8)
    image[24:104, 28:100] = 255
    target = tmp_path / "short-target.png"
    mask = tmp_path / "short-mask.png"
    _write_png(target, image)
    _write_png(mask, image)
    source = ClipImageSource(
        clip_id="short", parent_layout="M1_test1", split="train",
        target_path=str(target), base_mask_path=str(mask), scale_nm_per_pixel=10,
    )
    settings = SamplingSettings(
        sampler_version=SAMPLER_V3,
        support_radius_px=3,
        v3_target_segment_length_px=128,
        v3_min_segment_length_px=80,
        v3_corner_segment_length_px=80,
    )
    manifest = sample_clip_points(source, settings)
    lengths = [
        max(
            abs(point.segment_end_x - point.segment_start_x),
            abs(point.segment_end_y - point.segment_start_y),
        ) + 1
        for point in manifest.points
    ]
    assert 72 not in lengths
    assert manifest.sampling_audit["excluded_short_segments"] == 2
    assert manifest.sampling_audit["excluded_short_boundary_units"] == 144
    assert manifest.sampling_audit["accounted_boundary_units"] == manifest.sampling_audit["raw_boundary_units"]

    unsafe = np.zeros((128, 128), dtype=np.uint8)
    unsafe[20:80, 2:30] = 255
    unsafe_target = tmp_path / "unsafe-target.png"
    unsafe_mask = tmp_path / "unsafe-mask.png"
    _write_png(unsafe_target, unsafe)
    _write_png(unsafe_mask, unsafe)
    unsafe_source = source.copy(update={
        "clip_id": "unsafe",
        "target_path": str(unsafe_target),
        "base_mask_path": str(unsafe_mask),
    })
    with pytest.raises(ValueError, match="拒绝静默丢边"):
        sample_clip_points(unsafe_source, settings)


def test_v3_excludes_and_audits_non_unique_action_segments(tmp_path: Path):
    """法线内侧过薄导致动作饱和时，应排除该段并记录碰撞，而不是污染九分类标签。"""
    image = np.zeros((256, 256), dtype=np.uint8)
    image[50:170, 50:170] = 255
    image[190:210, 60:160] = 255
    target = tmp_path / "collision-target.png"
    mask = tmp_path / "collision-mask.png"
    _write_png(target, image)
    _write_png(mask, image)
    source = ClipImageSource(
        clip_id="collision", parent_layout="M1_test1", split="train",
        target_path=str(target), base_mask_path=str(mask), scale_nm_per_pixel=1,
    )
    settings = SamplingSettings(
        sampler_version=SAMPLER_V3,
        support_radius_px=8,
        v3_target_segment_length_px=128,
        v3_min_segment_length_px=32,
        v3_corner_segment_length_px=64,
    )
    manifest = sample_clip_points(source, settings)
    audit = manifest.sampling_audit
    assert audit["excluded_collision_segments"] >= 2
    assert audit["excluded_collision_boundary_units"] >= 200
    assert audit["accounted_boundary_units"] == audit["raw_boundary_units"]
    assert any(
        item["reason"] == "non-unique-nine-actions"
        for item in manifest.sampling_exclusions
    )


def test_v3_action_segments_follow_base_mask_not_shifted_target(tmp_path: Path):
    """target 与基准掩模错位时，v3 动作边段必须锚定真正被修改的基准掩模边界。"""
    target_image = np.zeros((128, 128), dtype=np.uint8)
    target_image[20:50, 20:50] = 255
    mask_image = np.zeros((128, 128), dtype=np.uint8)
    mask_image[70:100, 70:100] = 255
    target = tmp_path / "shifted-target.png"
    mask = tmp_path / "shifted-mask.png"
    _write_png(target, target_image)
    _write_png(mask, mask_image)
    source = ClipImageSource(
        clip_id="shifted", parent_layout="M1_test1", split="train",
        target_path=str(target), base_mask_path=str(mask), scale_nm_per_pixel=10,
    )
    settings = SamplingSettings(
        sampler_version=SAMPLER_V3,
        support_radius_px=3,
        v3_target_segment_length_px=32,
        v3_min_segment_length_px=12,
        v3_corner_segment_length_px=16,
    )
    manifest = sample_clip_points(source, settings)
    assert manifest.points
    assert all(mask_image[point.y, point.x] == 255 for point in manifest.points)
    assert all(target_image[point.y, point.x] == 0 for point in manifest.points)
