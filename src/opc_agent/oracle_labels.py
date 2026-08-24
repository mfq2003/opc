"""本模块把 OpenILT 九动作指标缓存转换为决策树所需的点级 Oracle 类别数据。

输入为候选掩模 NPZ、对应 metadata、完整 OpenILT 指标缓存和论文奖励权重；输出为带父版图划分、
geometry-v1 特征及最佳九分类动作的 JSON 数据集。关键依赖为 NumPy、Pydantic 与 recipe_tree Schema；
缺少任一动作指标时显式失败，不会用 PPO 预测或默认值填补真实精算结果。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from .features import FEATURE_NAMES
from .metrics import DISPLACEMENT_CLASSES_NM
from .models import TaskType
from .recipe_tree import PointTrainingDataset, PointTrainingRow


def build_training_rows(
    dataset_path: Path,
    metadata_path: Path,
    cache_path: Path,
    reward_weights: Dict[str, float],
) -> PointTrainingDataset:
    """选择每个点加权损失最低的动作，并保留确定性 geometry-v1 特征。"""
    if set(reward_weights) != {"l2", "epe", "pvb"}:
        raise ValueError("reward_weights 必须恰好包含 l2、epe、pvb")
    for path in (dataset_path, metadata_path, cache_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Oracle 标签输入不存在：{path}")
    source_hash = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    if cache.get("source_sha256") != source_hash:
        raise RuntimeError("OpenILT 指标缓存与候选掩模 NPZ 哈希不一致")
    if metadata.get("adapter_version") not in {
        "raster-fragment-v1", "raster-boundary-strip-v2", "raster-edge-segment-v3"
    }:
        raise ValueError("候选元数据缺少受支持的版本化适配器")
    with np.load(str(dataset_path), allow_pickle=False) as payload:
        observations = np.asarray(payload["observations"], dtype=np.float64)
    point_ids = list(metadata.get("point_ids", []))
    task_types = list(metadata.get("task_types", []))
    if observations.ndim != 2 or observations.shape[1] < len(FEATURE_NAMES):
        raise ValueError("候选 observations 缺少 geometry-v1 特征")
    if len(point_ids) != observations.shape[0] or len(task_types) != observations.shape[0]:
        raise ValueError("候选 metadata 的 point_ids/task_types 与 observations 点数不一致")
    metrics = cache.get("metrics", {})
    rows = []
    for point_index, point_id in enumerate(point_ids):
        losses = []
        for action_index in range(9):
            item = metrics.get(f"{point_index}:{action_index}")
            if item is None:
                raise ValueError(f"点 {point_id} 缺少动作 {action_index} 的 OpenILT 指标")
            loss = (
                float(reward_weights["l2"]) * float(item["l2"])
                + float(reward_weights["epe"]) * float(item["epe"])
                + float(reward_weights["pvb"]) * float(item["pvb"])
            )
            losses.append(loss)
        loss_array = np.asarray(losses, dtype=np.float64)
        best_loss = float(np.min(loss_array))
        tolerance = max(1e-6, abs(best_loss) * 1e-9)
        optimal_classes = [
            int(index) for index, value in enumerate(loss_array)
            if abs(float(value) - best_loss) <= tolerance
        ]
        best_class = min(
            optimal_classes,
            key=lambda index: (abs(float(DISPLACEMENT_CLASSES_NM[index])), index),
        )
        worse = [float(value) for value in loss_array if float(value) > best_loss + tolerance]
        loss_margin = min(worse) - best_loss if worse else None
        hashes = [
            metrics[f"{point_index}:{index}"].get("mask_sha256")
            for index in optimal_classes
        ]
        if len(optimal_classes) <= 1:
            candidate_collision = False
        elif all(value is not None for value in hashes):
            candidate_collision = len(set(hashes)) == 1
        else:
            candidate_collision = None
        features = {name: float(observations[point_index, index]) for index, name in enumerate(FEATURE_NAMES)}
        rows.append(PointTrainingRow(
            schema_version="2.0",
            sample_id=f"{metadata['clip_id']}:{point_id}",
            clip_id=metadata["clip_id"],
            parent_layout=metadata["parent_layout"],
            split=metadata["split"],
            task_type=TaskType(task_types[point_index]),
            features=features,
            displacement_class=best_class,
            optimal_classes=optimal_classes,
            loss_margin=loss_margin,
            ambiguous=len(optimal_classes) > 1,
            candidate_collision=candidate_collision,
        ))
    return PointTrainingDataset(
        schema_version="2.0", label_version="oracle-weighted-loss-v2", rows=rows
    )


def main(argv: Optional[List[str]] = None) -> int:
    """从命令行生成不可静默覆盖的点级训练 JSON。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.oracle_labels")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = build_training_rows(
        args.dataset, args.metadata, args.cache, dict(config["oracle"]["reward_weights"])
    )
    encoded = json.dumps(result.dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != encoded:
        raise FileExistsError("拒绝覆盖不同版本的点级 Oracle 标签")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        args.output.write_text(encoded, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
