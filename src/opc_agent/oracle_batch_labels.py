"""本模块把多 clip Oracle 指标缓存合并为一个版本化的点级训练数据集。

输入为候选索引、对应 train-oracle 运行目录、奖励权重配置和输出路径；输出为同时含 EPE/FRAG、
train/validation/test 父版图信息的 JSON 训练集及摘要。关键依赖为 Pydantic、NumPy、候选索引与
oracle_labels；模块不调用 GPU、OpenILT、PPO、Qwen 或网络，并会校验索引哈希、候选文件哈希和缓存完整性。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .oracle_labels import build_training_rows
from .point_sampling import CandidateIndex
from .recipe_tree import PointTrainingDataset, validate_training_dataset


def _index_identity(index: CandidateIndex) -> dict:
    """重建生成端使用的稳定索引身份。"""
    return {
        "sampler_version": index.sampler_version,
        "settings": index.settings.dict(),
        "entries": [entry.dict() for entry in index.entries],
    }


def load_candidate_index(path: Path) -> CandidateIndex:
    """读取并复核候选索引自身哈希以及每个 NPZ 的内容哈希。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"候选索引不存在：{source}")
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw_identity = {
        "sampler_version": raw.get("sampler_version"),
        "settings": raw.get("settings"),
        "entries": raw.get("entries"),
    }
    actual_index_hash = hashlib.sha256(
        json.dumps(raw_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if actual_index_hash != raw.get("index_sha256"):
        raise ValueError("候选索引身份哈希不一致")
    index = CandidateIndex.parse_obj(raw)
    for entry in index.entries:
        dataset_path = Path(entry.dataset_path)
        if not dataset_path.is_file():
            raise FileNotFoundError(f"候选数据不存在：{dataset_path}")
        actual_dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        if actual_dataset_hash != entry.dataset_sha256:
            raise ValueError(f"{entry.clip_id} 候选数据哈希不一致")
        for raw in (entry.metadata_path, entry.manifest_path):
            if not Path(raw).is_file():
                raise FileNotFoundError(f"候选配套文件不存在：{raw}")
    return index


def merge_oracle_labels(
    index_path: Path,
    oracle_run_dir: Path,
    reward_weights: Dict[str, float],
) -> PointTrainingDataset:
    """按索引顺序读取每个 clip 的独立缓存并合并，随后执行父版图防泄漏校验。"""
    index = load_candidate_index(index_path)
    run_root = Path(oracle_run_dir)
    stage_path = run_root / "stage-result.json"
    if not stage_path.is_file():
        raise FileNotFoundError(f"Oracle 运行缺少 stage-result.json：{stage_path}")
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    if stage.get("candidate_index_sha256") != index.index_sha256:
        raise ValueError("Oracle 运行与候选索引版本不一致")
    stage_clips = {item["clip_id"]: item for item in stage.get("clips", [])}
    rows = []
    for entry in index.entries:
        clip = stage_clips.get(entry.clip_id)
        if clip is None:
            raise ValueError(f"Oracle 运行缺少 clip：{entry.clip_id}")
        cache_path = Path(clip["cache_path"])
        part = build_training_rows(
            Path(entry.dataset_path), Path(entry.metadata_path), cache_path, reward_weights
        )
        rows.extend(part.rows)
    result = PointTrainingDataset(
        schema_version="2.0", label_version="oracle-weighted-loss-v2", rows=rows
    )
    validate_training_dataset(result)
    return result


def _write_versioned(path: Path, text: str) -> None:
    """相同内容允许重复执行，不同内容拒绝覆盖。"""
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"拒绝覆盖已有不同版本：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(text, encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    """从命令行合并多 clip Oracle 标签并写入统计摘要。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.oracle_batch_labels")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--oracle-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = merge_oracle_labels(
        args.index, args.oracle_run, dict(config["oracle"]["reward_weights"])
    )
    encoded = json.dumps(result.dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_versioned(args.output, encoded)
    split_counts = {
        split: sum(row.split == split for row in result.rows)
        for split in ("train", "validation", "test")
    }
    task_counts = {
        task: sum(row.task_type.value == task for row in result.rows)
        for task in ("EPE", "FRAG")
    }
    summary = {
        "schema_version": "1.0",
        "rows": len(result.rows),
        "split_counts": split_counts,
        "task_counts": task_counts,
        "label_version": result.label_version,
        "ambiguous_rows": sum(row.ambiguous for row in result.rows),
        "candidate_collision_rows": sum(row.candidate_collision is True for row in result.rows),
        "output_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }
    summary_path = args.output.with_suffix(".summary.json")
    _write_versioned(summary_path, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(f"rows={len(result.rows)} epe={task_counts['EPE']} frag={task_counts['FRAG']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
