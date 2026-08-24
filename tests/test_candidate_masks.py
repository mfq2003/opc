"""本模块验证 raster-fragment-v1 的九分类候选形状、零位移动作和法线校验。

输入为小型二值图、一个轴向点和 10nm/像素比例；输出为十维状态、九个候选和非法法线断言。
关键依赖为 OpenCV、NumPy、Pydantic 与 pytest；测试不调用 GPU、OpenILT、Qwen 或网络。
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from opc_agent.candidate_masks import (
    ADAPTER_V1, ADAPTER_V2, ADAPTER_V3, CandidateManifest, FragmentPoint,
    build_candidate_arrays, build_compact_candidate_data, generate_candidate_mask,
    move_local_fragment,
)
from opc_agent.models import TaskType


def _write_png(path: Path, image: np.ndarray) -> None:
    """通过 Python 文件 API 写入 PNG，兼容中文临时路径。"""
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())


def test_candidate_builder_generates_nine_masks_and_keeps_zero_action(tmp_path: Path):
    """九分类维度必须固定，类别 4 的 0nm 候选必须等于基准掩模。"""
    target = np.zeros((20, 20), dtype=np.uint8)
    target[6:14, 6:14] = 255
    target_path = tmp_path / "target.png"
    mask_path = tmp_path / "mask.png"
    _write_png(target_path, target)
    _write_png(mask_path, target)
    manifest = CandidateManifest(
        clip_id="clip-1", parent_layout="M1_test1", split="train",
        target_path=str(target_path), base_mask_path=str(mask_path), scale_nm_per_pixel=10,
        points=[FragmentPoint(
            point_id="p1", task_type=TaskType.EPE, x=13, y=10,
            normal_x=1, normal_y=0, support_radius_px=2,
        )],
    )
    observations, normalized_target, candidates, metadata = build_candidate_arrays(manifest)
    assert observations.shape == (1, 10)
    assert candidates.shape == (1, 9, 20, 20)
    assert np.array_equal(candidates[0, 4], normalized_target)
    assert not np.array_equal(candidates[0, 8], normalized_target)
    assert metadata["adapter_version"] == ADAPTER_V2
    assert len({candidate.tobytes() for candidate in candidates[0]}) == 9
    areas = [int(candidate.sum()) for candidate in candidates[0]]
    assert areas == sorted(areas)
    compact_observations, _, base_mask, geometry, compact_metadata = build_compact_candidate_data(manifest)
    assert compact_observations.shape == (1, 10)
    assert base_mask.shape == (20, 20)
    assert geometry.shape == (1, 5)
    assert compact_metadata["storage_format"] == "compact-point-geometry-v1"


def test_compact_v2_rejects_colliding_actions(tmp_path: Path):
    """九个位移生成相同空白掩模时必须在 OpenILT 前失败。"""
    empty = np.zeros((20, 20), dtype=np.uint8)
    target_path = tmp_path / "empty-target.png"
    mask_path = tmp_path / "empty-mask.png"
    _write_png(target_path, empty)
    _write_png(mask_path, empty)
    manifest = CandidateManifest(
        clip_id="empty", parent_layout="M1_test1", split="train",
        target_path=str(target_path), base_mask_path=str(mask_path), scale_nm_per_pixel=10,
        points=[FragmentPoint(
            point_id="collision", task_type=TaskType.EPE, x=10, y=10,
            normal_x=1, normal_y=0, support_radius_px=2,
        )],
    )
    with pytest.raises(ValueError, match="拒绝候选碰撞"):
        build_compact_candidate_data(manifest)


def test_adapter_v1_dispatch_preserves_legacy_square_move():
    """旧适配器必须显式分派到原方块平移，避免升级后静默重解释旧 NPZ。"""
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[6:14, 6:14] = 1
    point = FragmentPoint(
        point_id="legacy", task_type=TaskType.EPE, x=13, y=10,
        normal_x=1, normal_y=0, support_radius_px=2,
    )
    expected = move_local_fragment(mask, point, -2)
    actual = generate_candidate_mask(mask, point, -2, ADAPTER_V1)
    assert np.array_equal(actual, expected)


def test_fragment_point_requires_axis_aligned_unit_normal():
    """斜向或零法线必须失败。"""
    with pytest.raises(ValueError, match="法线"):
        FragmentPoint(
            point_id="bad", task_type=TaskType.FRAG, x=1, y=1,
            normal_x=1, normal_y=1, support_radius_px=1,
        )


def test_v3_moves_whole_segment_and_uses_nine_column_geometry(tmp_path: Path):
    """v3 九动作必须移动完整边段，面积严格单调，并保存可恢复的九列边段几何。"""
    image = np.zeros((64, 64), dtype=np.uint8)
    image[20:44, 20:44] = 255
    target_path = tmp_path / "v3-target.png"
    mask_path = tmp_path / "v3-mask.png"
    _write_png(target_path, image)
    _write_png(mask_path, image)
    point = FragmentPoint(
        point_id="epe-0000", task_type=TaskType.EPE, x=43, y=31,
        normal_x=1, normal_y=0, support_radius_px=3,
        segment_start_x=43, segment_start_y=20,
        segment_end_x=43, segment_end_y=43,
    )
    manifest = CandidateManifest(
        adapter_version=ADAPTER_V3,
        clip_id="v3", parent_layout="M1_test1", split="train",
        target_path=str(target_path), base_mask_path=str(mask_path), scale_nm_per_pixel=10,
        points=[point],
    )
    _, _, base_mask, geometry, metadata = build_compact_candidate_data(manifest)
    candidates = [generate_candidate_mask(base_mask, point, offset, ADAPTER_V3) for offset in range(-4, 5)]
    assert geometry.shape == (1, 9)
    assert metadata["storage_format"] == "compact-edge-geometry-v1"
    assert [int(candidate.sum()) for candidate in candidates] == sorted(
        int(candidate.sum()) for candidate in candidates
    )
    changed_y, _ = np.nonzero(candidates[-1] != base_mask)
    assert (int(changed_y.min()), int(changed_y.max())) == (20, 43)


def test_v3_manifest_rejects_point_without_segment_geometry():
    """v3 清单不得把旧点几何误当成整段动作。"""
    point = FragmentPoint(
        point_id="old-point", task_type=TaskType.EPE, x=10, y=10,
        normal_x=1, normal_y=0, support_radius_px=2,
    )
    with pytest.raises(ValueError, match="完整边段端点"):
        CandidateManifest(
            adapter_version=ADAPTER_V3,
            clip_id="bad-v3", parent_layout="M1_test1", split="train",
            target_path="target.png", base_mask_path="mask.png", scale_nm_per_pixel=10,
            points=[point],
        )


def test_v3_allows_one_pixel_corner_segment():
    """栅格转角可能产生单像素轴向边段，数据模型必须保留而不是静默删掉。"""
    point = FragmentPoint(
        point_id="corner", task_type=TaskType.EPE, x=10, y=10,
        normal_x=1, normal_y=0, support_radius_px=2,
        segment_start_x=10, segment_start_y=10,
        segment_end_x=10, segment_end_y=10,
    )
    assert point.has_segment_geometry is True

