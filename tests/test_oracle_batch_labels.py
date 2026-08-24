"""本模块验证多 clip Oracle 标签合并时的索引哈希、候选哈希、缓存映射与父版图切分。

输入为临时生成的两个矩形 clip、完整九动作缓存和运行摘要；输出为合并后的 EPE/FRAG 点级训练行，
以及候选文件被篡改时的失败断言。关键依赖为 NumPy、OpenCV、Pydantic 与 pytest；不调用 GPU、OpenILT 或网络。
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from opc_agent.oracle_batch_labels import load_candidate_index, merge_oracle_labels
from opc_agent.point_sampling import ClipImageSource, SamplingSettings, build_candidate_index


def _write_image(path: Path) -> None:
    """写入兼容中文路径的矩形二值图。"""
    image = np.zeros((128, 128), dtype=np.uint8)
    image[24:104, 28:100] = 255
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())


def _source(tmp_path: Path, clip_id: str, parent: str, split: str) -> ClipImageSource:
    """构造一个具有 EPE/FRAG 边界点的公开图像源描述。"""
    target = tmp_path / f"{clip_id}-target.png"
    mask = tmp_path / f"{clip_id}-mask.png"
    _write_image(target)
    _write_image(mask)
    return ClipImageSource(
        clip_id=clip_id, parent_layout=parent, split=split,
        target_path=str(target), base_mask_path=str(mask), scale_nm_per_pixel=10,
    )


def _prepare_run(tmp_path: Path):
    """生成候选索引、每个 clip 的完整缓存和匹配的 stage-result。"""
    settings = SamplingSettings(
        epe_spacing_px=32, frag_spacing_px=24, support_radius_px=3,
        min_segment_length_px=12, max_epe_points=2, max_frag_points=2,
    )
    index = build_candidate_index([
        _source(tmp_path, "clip-a", "M1_test1", "train"),
        _source(tmp_path, "clip-b", "M1_test7", "validation"),
    ], settings, tmp_path / "candidates")
    index_path = tmp_path / "candidates" / "candidate-index.json"
    run_root = tmp_path / "run"
    clips = []
    for entry in index.entries:
        cache = run_root / "clips" / entry.clip_id / "oracle-metrics.cache.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        point_count = entry.epe_points + entry.frag_points
        metrics = {}
        for point_index in range(point_count):
            for action_index in range(9):
                metrics[f"{point_index}:{action_index}"] = {
                    "l2": float(100 + action_index), "epe": 2.0, "pvb": 3.0,
                }
            metrics[f"{point_index}:4"] = {"l2": 1.0, "epe": 0.0, "pvb": 1.0}
        cache.write_text(json.dumps({
            "source_sha256": hashlib.sha256(Path(entry.dataset_path).read_bytes()).hexdigest(),
            "metrics": metrics,
        }), encoding="utf-8")
        clips.append({"clip_id": entry.clip_id, "cache_path": str(cache)})
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "stage-result.json").write_text(json.dumps({
        "candidate_index_sha256": index.index_sha256, "clips": clips,
    }), encoding="utf-8")
    return index_path, run_root, index


def test_merge_oracle_labels_keeps_both_tasks_and_splits(tmp_path: Path):
    """两个 clip 合并后必须保留任务、切分和最佳九分类动作。"""
    index_path, run_root, index = _prepare_run(tmp_path)
    result = merge_oracle_labels(
        index_path, run_root, {"l2": 1.0, "epe": 100.0, "pvb": 1.0}
    )
    assert len(result.rows) == sum(item.epe_points + item.frag_points for item in index.entries)
    assert {row.task_type.value for row in result.rows} == {"EPE", "FRAG"}
    assert {row.split for row in result.rows} == {"train", "validation"}
    assert {row.displacement_class for row in result.rows} == {4}


def test_load_candidate_index_rejects_tampered_npz(tmp_path: Path):
    """候选 NPZ 发生任何字节变化后都不能继续使用旧索引。"""
    index_path, _, index = _prepare_run(tmp_path)
    dataset = Path(index.entries[0].dataset_path)
    dataset.write_bytes(dataset.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="候选数据哈希不一致"):
        load_candidate_index(index_path)
