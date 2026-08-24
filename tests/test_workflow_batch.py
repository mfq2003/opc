"""本模块验证统一 train-oracle 阶段按候选索引拆分 clip、缓存、模型和冒烟种子。

输入为两个最小候选文件、带哈希索引和替身 Oracle 后端；输出为 stage-result.json 的逐 clip 记录。
关键依赖为 Pydantic 与 pytest；替身后端确保测试不导入 Gymnasium、PyTorch、CUDA 或 OpenILT。
"""
import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from opc_agent.point_sampling import CandidateIndex, CandidateIndexEntry, SamplingSettings
from opc_agent.workflow import train_oracle_stage


def _candidate_index(tmp_path: Path) -> Path:
    """创建通过字节哈希和配套文件校验的双 clip 索引。"""
    entries = []
    for clip_id, parent, split in (
        ("M1_test1", "M1_test1", "train"),
        ("M1_test7", "M1_test7", "validation"),
    ):
        dataset = tmp_path / f"{clip_id}.npz"
        metadata = tmp_path / f"{clip_id}.metadata.json"
        manifest = tmp_path / f"{clip_id}.manifest.json"
        dataset.write_bytes(f"dataset-{clip_id}".encode("ascii"))
        metadata.write_text("{}", encoding="utf-8")
        manifest.write_text("{}", encoding="utf-8")
        entries.append(CandidateIndexEntry(
            clip_id=clip_id, parent_layout=parent, split=split,
            dataset_path=str(dataset), metadata_path=str(metadata), manifest_path=str(manifest),
            dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
            epe_points=1, frag_points=1,
        ))
    settings = SamplingSettings()
    identity = {
        "sampler_version": "axis-boundary-sampler-v2",
        "settings": settings.dict(),
        "entries": [entry.dict() for entry in entries],
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    index = CandidateIndex(settings=settings, entries=entries, index_sha256=digest)
    path = tmp_path / "candidate-index.json"
    path.write_text(json.dumps(index.dict(), ensure_ascii=False), encoding="utf-8")
    return path


def test_train_oracle_index_smoke_separates_clip_artifacts(tmp_path: Path, monkeypatch):
    """多 clip 冒烟应遍历全部索引项，但只使用首个种子和 smoke_timesteps。"""
    index_path = _candidate_index(tmp_path)
    completed = []
    fake = types.ModuleType("opc_agent.oracle_runner")
    fake.load_candidate_point_set = lambda path: SimpleNamespace(
        source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    )
    fake.OpenILTCandidateEvaluator = lambda **kwargs: SimpleNamespace(**kwargs)
    fake.CandidatePointEnv = lambda dataset, evaluator, weights: SimpleNamespace(
        dataset=dataset, evaluator=evaluator, weights=weights
    )
    fake.complete_metric_cache = lambda dataset, evaluator: completed.append(evaluator.cache_path)

    def _train(**kwargs):
        return Path(str(kwargs["output_path"]) + ".zip")

    fake.train_ppo = _train
    monkeypatch.setitem(sys.modules, "opc_agent.oracle_runner", fake)
    config = {
        "data": {"openilt_dir": "third_party/OpenILT"},
        "openilt": {"commit": "fixed"},
        "oracle": {
            "seeds": [0, 1, 2], "smoke_timesteps": 256, "total_timesteps": 10000,
            "lithography_config": "config/lithosimple.txt", "simulator": "simple",
            "openilt_scale": 1, "reward_weights": {"l2": 1, "epe": 100, "pvb": 1},
            "learning_rate": 0.0003,
        },
    }
    run_root = tmp_path / "run"
    run_root.mkdir()
    train_oracle_stage(config, run_root, smoke=True, candidate_index=index_path)
    result = json.loads((run_root / "stage-result.json").read_text(encoding="utf-8"))
    assert result["seeds"] == [0]
    assert result["timesteps_per_seed"] == 256
    assert [item["clip_id"] for item in result["clips"]] == ["M1_test1", "M1_test7"]
    assert len(set(item["cache_path"] for item in result["clips"])) == 2
    assert all(item["models"][0].endswith(f"{item['clip_id']}-seed-0.zip") for item in result["clips"])
    assert len(completed) == 2
