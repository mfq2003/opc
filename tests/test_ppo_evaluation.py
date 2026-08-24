"""本模块验证 PPO 模型评估时的 Oracle 标签对齐、九分类指标、奖励遗憾和版本身份门禁。

输入为临时生成的候选索引、完整九动作缓存、点级标签以及替身 PPO 模型；输出为逐 seed 与全局
评估摘要，并断言完美策略、零位移基线和错误索引身份得到预期结果。关键依赖为 NumPy、OpenCV、
Pydantic 与 pytest；不加载 stable-baselines3，不调用 CUDA、OpenILT 或网络。
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from opc_agent.oracle_batch_labels import merge_oracle_labels
from opc_agent.point_sampling import ClipImageSource, SamplingSettings, build_candidate_index
from opc_agent.ppo_evaluation import evaluate_ppo_run


def _write_image(path: Path) -> None:
    """写入一个具有足够长正交边界的二值矩形。"""
    image = np.zeros((128, 128), dtype=np.uint8)
    image[24:104, 28:100] = 255
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())


class _FakeModel:
    """返回测试预设动作，模拟 SB3 的确定性 predict 接口。"""

    def __init__(self, predictions):
        self.predictions = np.asarray(predictions, dtype=np.int64)

    def predict(self, observations, deterministic=True):
        assert deterministic is True
        assert len(observations) == len(self.predictions)
        return self.predictions.copy(), None


def _prepare_inputs(tmp_path: Path):
    """构造两个 clip、两个 seed、完整缓存和匹配标签。"""
    sources = []
    for clip_id, split in (("clip-a", "train"), ("clip-b", "test")):
        target = tmp_path / f"{clip_id}-target.png"
        mask = tmp_path / f"{clip_id}-mask.png"
        _write_image(target)
        _write_image(mask)
        sources.append(ClipImageSource(
            clip_id=clip_id,
            parent_layout=clip_id,
            split=split,
            target_path=str(target),
            base_mask_path=str(mask),
            scale_nm_per_pixel=10,
        ))
    index = build_candidate_index(
        sources,
        SamplingSettings(
            epe_spacing_px=32,
            frag_spacing_px=24,
            support_radius_px=3,
            min_segment_length_px=12,
            max_epe_points=2,
            max_frag_points=2,
        ),
        tmp_path / "candidates",
    )
    index_path = tmp_path / "candidates" / "candidate-index.json"
    run_root = tmp_path / "run"
    models_root = run_root / "models"
    models_root.mkdir(parents=True)
    stage_clips = []
    truth_by_clip = {}
    for entry in index.entries:
        point_count = entry.epe_points + entry.frag_points
        truth = [point_index % 9 for point_index in range(point_count)]
        truth_by_clip[entry.clip_id] = truth
        metrics = {}
        for point_index, best_action in enumerate(truth):
            for action_index in range(9):
                metrics[f"{point_index}:{action_index}"] = {
                    "l2": float(10 * abs(action_index - best_action)),
                    "epe": 0.0,
                    "pvb": 0.0,
                    "mask_sha256": f"{point_index:04d}-{action_index:02d}",
                }
        cache_path = run_root / "clips" / entry.clip_id / "oracle-metrics.cache.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text(json.dumps({
            "source_sha256": entry.dataset_sha256,
            "metrics": metrics,
        }), encoding="utf-8")
        model_paths = []
        for seed in (0, 1):
            model_path = models_root / f"{entry.clip_id}-seed-{seed}.zip"
            model_path.write_bytes(b"fake-model")
            model_path.with_suffix(".metadata.json").write_text(json.dumps({
                "seed": seed,
                "candidate_source_sha256": entry.dataset_sha256,
            }), encoding="utf-8")
            model_paths.append(str(model_path))
        stage_clips.append({
            "clip_id": entry.clip_id,
            "cache_path": str(cache_path),
            "models": model_paths,
        })
    run_root.mkdir(parents=True, exist_ok=True)
    stage_path = run_root / "stage-result.json"
    stage_path.write_text(json.dumps({
        "mode": "full",
        "candidate_index_sha256": index.index_sha256,
        "seeds": [0, 1],
        "timesteps_per_seed": 10000,
        "clips": stage_clips,
    }), encoding="utf-8")
    labels = merge_oracle_labels(index_path, run_root, {"l2": 1.0, "epe": 0.0, "pvb": 0.0})
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels.dict()), encoding="utf-8")
    return index_path, run_root, labels_path, truth_by_clip, stage_path


def test_evaluate_ppo_run_compares_seeds_and_baselines(tmp_path: Path):
    """完美 seed 必须零遗憾，固定零位移 seed 必须等于零位移基线。"""
    index_path, run_root, labels_path, truth_by_clip, _ = _prepare_inputs(tmp_path)

    def loader(path: Path):
        match = re.search(r"(.+)-seed-(\d+)$", path.stem)
        assert match is not None
        clip_id, seed = match.group(1), int(match.group(2))
        predictions = truth_by_clip[clip_id] if seed == 0 else [4] * len(truth_by_clip[clip_id])
        return _FakeModel(predictions)

    result = evaluate_ppo_run(
        index_path,
        run_root,
        labels_path,
        {"l2": 1.0, "epe": 0.0, "pvb": 0.0},
        model_loader=loader,
    )
    assert result["clips"] == 2
    assert result["rows"] == sum(len(values) for values in truth_by_clip.values())
    seed_zero, seed_one = result["aggregate_seeds"]
    assert seed_zero["seed"] == 0
    assert seed_zero["accuracy"] == 1.0
    assert seed_zero["optimal_set_accuracy"] == 1.0
    assert seed_zero["mean_weighted_loss_regret"] == 0.0
    assert seed_one["seed"] == 1
    assert seed_one["accuracy"] == result["zero_displacement_baseline"]["accuracy"]
    assert seed_one["mean_weighted_loss_regret"] == result["zero_displacement_baseline"][
        "mean_weighted_loss_regret"
    ]
    assert 0 <= result["majority_class_baseline"]["action"] <= 8
    assert result["majority_class_baseline"]["accuracy"] >= result[
        "uniform_random_expectation"
    ]["canonical_accuracy"]
    assert result["uniform_random_expectation"]["canonical_accuracy"] == pytest.approx(1 / 9)


def test_evaluate_ppo_run_rejects_stage_from_other_index(tmp_path: Path):
    """运行摘要的候选索引哈希不一致时必须在加载模型前失败。"""
    index_path, run_root, labels_path, _, stage_path = _prepare_inputs(tmp_path)
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage["candidate_index_sha256"] = "0" * 64
    stage_path.write_text(json.dumps(stage), encoding="utf-8")
    with pytest.raises(ValueError, match="候选索引版本不一致"):
        evaluate_ppo_run(
            index_path,
            run_root,
            labels_path,
            {"l2": 1.0, "epe": 0.0, "pvb": 0.0},
            model_loader=lambda _: None,
        )
