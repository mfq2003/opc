"""本模块验证候选点总览、局部窗口、九动作差异图和质量摘要的确定性生成与交叉校验。

输入为临时目录中的小型正交二值图、EPE/FRAG 点和带哈希候选索引；输出为 PNG/JSON 派生文件及
几何不一致异常断言。关键依赖为 OpenCV、NumPy、Pydantic 和 pytest；测试不调用 GPU、OpenILT、PPO 或网络。
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opc_agent.candidate_masks import CandidateManifest, FragmentPoint, build_compact_candidate_data
from opc_agent.models import TaskType
from opc_agent.point_sampling import (
    CandidateIndexEntry,
    ClipImageSource,
    SAMPLER_VERSION,
    SAMPLER_V3,
    SamplingSettings,
    build_candidate_index,
)
from opc_agent.point_visualization import visualize_candidate_index


def _write_png(path: Path, image: np.ndarray) -> None:
    """通过 Python 文件 API 写入 PNG，兼容中文临时路径。"""
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())


def _write_fixture(tmp_path: Path) -> Path:
    """生成两个方向正确、九动作唯一的点及完整候选索引。"""
    image = np.zeros((80, 80), dtype=np.uint8)
    image[20:60, 20:60] = 255
    target_path = tmp_path / "target.png"
    mask_path = tmp_path / "mask.png"
    _write_png(target_path, image)
    _write_png(mask_path, image)
    manifest = CandidateManifest(
        clip_id="clip-1",
        parent_layout="M1_test1",
        split="train",
        target_path=str(target_path),
        base_mask_path=str(mask_path),
        scale_nm_per_pixel=10.0,
        points=[
            FragmentPoint(
                point_id="epe-0000", task_type=TaskType.EPE, x=59, y=40,
                normal_x=1, normal_y=0, support_radius_px=4,
            ),
            FragmentPoint(
                point_id="frag-0000", task_type=TaskType.FRAG, x=40, y=59,
                normal_x=0, normal_y=1, support_radius_px=4,
            ),
        ],
    )
    manifest_path = tmp_path / "clip-1.manifest.json"
    dataset_path = tmp_path / "clip-1.npz"
    metadata_path = tmp_path / "clip-1.metadata.json"
    manifest_path.write_text(
        json.dumps(manifest.dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    observations, target, base_mask, geometry, metadata = build_compact_candidate_data(manifest)
    np.savez_compressed(
        str(dataset_path),
        observations=observations,
        target=target,
        base_mask=base_mask,
        point_geometry=geometry,
        scale_nm_per_pixel=np.asarray(manifest.scale_nm_per_pixel),
        adapter_version=np.asarray(manifest.adapter_version),
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    entry = CandidateIndexEntry(
        clip_id=manifest.clip_id,
        parent_layout=manifest.parent_layout,
        split=manifest.split,
        dataset_path=str(dataset_path),
        metadata_path=str(metadata_path),
        manifest_path=str(manifest_path),
        dataset_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        epe_points=1,
        frag_points=1,
    )
    settings = SamplingSettings(
        epe_spacing_px=16,
        frag_spacing_px=16,
        support_radius_px=4,
        min_segment_length_px=8,
        max_epe_points=1,
        max_frag_points=1,
        max_displacement_nm=40,
    )
    identity = {
        "sampler_version": SAMPLER_VERSION,
        "settings": settings.dict(),
        "entries": [entry.dict()],
    }
    index_hash = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    index_path = tmp_path / "candidate-index.json"
    index_path.write_text(
        json.dumps({**identity, "schema_version": "1.0", "index_sha256": index_hash},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index_path


def test_visualization_generates_images_summary_and_preserves_inputs(tmp_path: Path):
    """两点必须通过全部检查并生成可读 PNG，重复运行不得修改候选输入。"""
    index_path = _write_fixture(tmp_path)
    raw_index = json.loads(index_path.read_text(encoding="utf-8"))
    dataset_path = Path(raw_index["entries"][0]["dataset_path"])
    original_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    output_dir = tmp_path / "visualization"

    summary = visualize_candidate_index(index_path, output_dir)
    repeated = visualize_candidate_index(index_path, output_dir)

    assert summary == repeated
    assert summary["points"] == 2
    assert summary["passed_points"] == 2
    assert summary["failed_points"] == []
    assert all(item["unique_action_masks"] == 9 for item in summary["point_checks"])
    assert all(item["negative_actions_shrink"] for item in summary["point_checks"])
    assert all(item["positive_actions_expand"] for item in summary["point_checks"])
    expected = [
        output_dir / "clip-1" / "overview.png",
        output_dir / "clip-1" / "epe-0000-local.png",
        output_dir / "clip-1" / "epe-0000-actions.png",
        output_dir / "clip-1" / "frag-0000-local.png",
        output_dir / "clip-1" / "frag-0000-actions.png",
        output_dir / "visualization-summary.json",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)
    overview = cv2.imdecode(np.frombuffer(expected[0].read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    actions = cv2.imdecode(np.frombuffer(expected[2].read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert overview.shape == (80, 80, 3)
    assert actions.shape[0] > 600 and actions.shape[1] > 600
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == original_hash


def test_visualization_rejects_manifest_npz_geometry_mismatch(tmp_path: Path):
    """manifest 点坐标被篡改后必须在渲染前失败。"""
    index_path = _write_fixture(tmp_path)
    raw_index = json.loads(index_path.read_text(encoding="utf-8"))
    manifest_path = Path(raw_index["entries"][0]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["points"][0]["y"] += 1
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="点几何不一致"):
        visualize_candidate_index(index_path, tmp_path / "visualization")


def test_v3_visualization_checks_and_draws_complete_segments(tmp_path: Path):
    """v3 总览应按 base mask 画完整动作边段，target 错位不应误判动作几何。"""
    mask_image = np.zeros((160, 160), dtype=np.uint8)
    mask_image[40:120, 30:130] = 255
    target_image = np.zeros((160, 160), dtype=np.uint8)
    target_image[45:125, 35:135] = 255
    target_path = tmp_path / "v3-target.png"
    mask_path = tmp_path / "v3-mask.png"
    _write_png(target_path, target_image)
    _write_png(mask_path, mask_image)
    source = ClipImageSource(
        clip_id="clip-v3", parent_layout="M1_test1", split="train",
        target_path=str(target_path), base_mask_path=str(mask_path), scale_nm_per_pixel=10,
    )
    settings = SamplingSettings(
        sampler_version=SAMPLER_V3,
        support_radius_px=3,
        v3_target_segment_length_px=128,
        v3_min_segment_length_px=90,
        v3_corner_segment_length_px=90,
    )
    candidate_dir = tmp_path / "v3-candidates"
    index = build_candidate_index([source], settings, candidate_dir)
    output_dir = tmp_path / "v3-visualization"
    summary = visualize_candidate_index(
        candidate_dir / "candidate-index.json",
        output_dir,
        overview=True,
        local_patches=False,
        action_grids=False,
    )
    assert summary["segments"] == index.entries[0].epe_points
    assert summary["passed_segments"] == summary["segments"]
    assert 0 < summary["minimum_axis_boundary_coverage"] < 1.0
    assert summary["minimum_accounted_axis_boundary_coverage"] == 1.0
    assert summary["excluded_segments"] == 2
    assert summary["boundary_references"] == ["base_mask"]
    assert summary["clip_summaries"][0]["unexpected_axis_boundary_units"] == 0
    assert summary["clip_summaries"][0]["boundary_reference"] == "base_mask"
    assert summary["failed_points"] == []
    assert any(not item["segment_on_target_boundary"] for item in summary["point_checks"])
    assert all(item["segment_on_action_boundary"] for item in summary["point_checks"])
    assert all(item["segment_normal_points_outward"] for item in summary["point_checks"])
    assert (output_dir / "clip-v3" / "overview.png").is_file()
