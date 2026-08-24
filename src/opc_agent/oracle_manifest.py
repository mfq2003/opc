"""本模块从确定性 EPE 标签集生成可恢复、可复现的多种子 Oracle/PPO 任务清单。

输入为 epe_dataset 生成的标签 JSON、随机种子和奖励权重；输出为只包含训练/验证集的稳定任务清单，
每个任务具有内容哈希派生的 job_id。关键依赖为标准库与 Pydantic；本模块只规划任务，不调用 GPU、
OpenILT、Stable-Baselines3、网络或 API，测试集永不进入 Oracle 训练任务。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .epe_dataset import EpeLabel, EpeLabelDataset


class OracleJobStatus(str, Enum):
    """定义可恢复 Oracle 任务的生命周期状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OracleJob(BaseModel):
    """保存一个 clip、随机种子和奖励配置对应的稳定 Oracle 任务。"""

    schema_version: str = "1.0"
    job_id: str = Field(min_length=64, max_length=64)
    clip_id: str
    parent_layout: str
    split: str
    seed: int
    metric_version: str
    label_fingerprint: str = Field(min_length=64, max_length=64)
    reward_weights: Dict[str, float]
    displacement_nm_limit: float = Field(gt=0, le=40)
    status: OracleJobStatus = OracleJobStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    result_path: Optional[str] = None
    error: Optional[str] = None


class OracleManifest(BaseModel):
    """保存一个不可覆盖、可按 job_id 恢复的 Oracle 任务版本。"""

    schema_version: str = "1.0"
    manifest_hash: str = Field(min_length=64, max_length=64)
    jobs: List[OracleJob]


def _label_fingerprint(label: EpeLabel) -> str:
    """用输入图哈希与指标版本生成标签指纹。"""
    payload = {
        "clip_id": label.clip_id,
        "metric_version": label.metric_version,
        "target_sha256": label.target_sha256,
        "printed_sha256": label.printed_sha256,
        "epe_n": label.epe_n,
        "epe_d_nm": label.epe_d_nm,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_oracle_manifest(
    dataset: EpeLabelDataset,
    seeds: List[int],
    reward_weights: Dict[str, float],
    displacement_nm_limit: float = 40,
) -> OracleManifest:
    """为训练/验证标签生成稳定任务；测试标签被明确排除。"""
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Oracle seeds 必须非空且不能重复")
    required_weights = {"l2", "epe", "pvb"}
    if set(reward_weights) != required_weights or any(value < 0 for value in reward_weights.values()):
        raise ValueError("奖励权重必须恰好包含非负的 l2、epe、pvb")
    jobs = []
    for label in sorted(dataset.labels, key=lambda item: item.clip_id):
        if label.split == "test":
            continue
        fingerprint = _label_fingerprint(label)
        for seed in sorted(seeds):
            identity = {
                "clip_id": label.clip_id,
                "fingerprint": fingerprint,
                "seed": seed,
                "reward_weights": reward_weights,
                "displacement_nm_limit": displacement_nm_limit,
            }
            job_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            jobs.append(OracleJob(
                job_id=job_id,
                clip_id=label.clip_id,
                parent_layout=label.parent_layout,
                split=label.split,
                seed=seed,
                metric_version=label.metric_version,
                label_fingerprint=fingerprint,
                reward_weights=reward_weights,
                displacement_nm_limit=displacement_nm_limit,
            ))
    manifest_payload = [job.dict() for job in jobs]
    manifest_hash = hashlib.sha256(json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return OracleManifest(manifest_hash=manifest_hash, jobs=jobs)


def main(argv: List[str] | None = None) -> int:
    """读取 EPE 标签和论文配置，生成 CPU-only Oracle 任务清单。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.oracle_manifest")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import yaml

    dataset = EpeLabelDataset.parse_obj(json.loads(args.labels.read_text(encoding="utf-8")))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    oracle = config["oracle"]
    manifest = build_oracle_manifest(
        dataset,
        seeds=list(oracle["seeds"]),
        reward_weights=dict(oracle["reward_weights"]),
        displacement_nm_limit=float(oracle["displacement_nm_limit"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = OracleManifest.parse_obj(json.loads(args.output.read_text(encoding="utf-8")))
        if existing.manifest_hash != manifest.manifest_hash:
            raise FileExistsError("现有 Oracle manifest 内容不同；禁止覆盖旧数据版本")
    else:
        args.output.write_text(json.dumps(manifest.dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


