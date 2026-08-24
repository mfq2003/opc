"""本模块验证 Oracle/PPO 任务清单的确定性、多随机种子展开和测试集隔离。

输入为合成 EPE 标签集、奖励权重与随机种子；输出为稳定 job_id、任务数量和非法配置失败断言。
关键依赖为 pytest 与 Pydantic；测试不调用 GPU、OpenILT、Stable-Baselines3、网络或 API。
"""
from opc_agent.epe_dataset import EpeLabel, EpeLabelDataset
from opc_agent.oracle_manifest import build_oracle_manifest


def _label(clip_id: str, split: str) -> EpeLabel:
    """构造具有稳定文件哈希的最小 EPE 标签。"""
    return EpeLabel(
        clip_id=clip_id, parent_layout=f"parent-{clip_id}", split=split,
        target_path="target.png", printed_path="printed.png",
        target_sha256="a" * 64, printed_sha256="b" * 64,
        scale_nm_per_pixel=1, tolerance_nm=1, sample_count=4,
        epe_n=1, epe_d_pixels=2, epe_d_nm=2, mean_distance_nm=0.5,
    )


def test_oracle_manifest_expands_seeds_and_excludes_test_labels():
    """两个可训练标签与三个种子应生成六个任务，测试标签不得进入训练。"""
    dataset = EpeLabelDataset(labels=[_label("train-1", "train"), _label("val-1", "validation"), _label("test-1", "test")])
    manifest = build_oracle_manifest(dataset, [0, 1, 2], {"l2": 1, "epe": 100, "pvb": 1})
    assert len(manifest.jobs) == 6
    assert {job.split for job in manifest.jobs} == {"train", "validation"}
    assert len({job.job_id for job in manifest.jobs}) == 6


def test_oracle_manifest_is_deterministic():
    """相同标签和配置重复构建必须得到相同 manifest_hash。"""
    dataset = EpeLabelDataset(labels=[_label("train-1", "train")])
    first = build_oracle_manifest(dataset, [2, 1, 0], {"l2": 1, "epe": 100, "pvb": 1})
    second = build_oracle_manifest(dataset, [0, 1, 2], {"pvb": 1, "epe": 100, "l2": 1})
    assert first.manifest_hash == second.manifest_hash

