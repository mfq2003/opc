"""本模块验证 EPE 图像对的数据哈希、物理单位转换和父版图防泄漏规则。

输入为临时二值 PNG 图像和版本化图像对；输出为确定性 EPE 标签、稳定哈希和跨集合失败断言。
关键依赖为 OpenCV、NumPy 与 pytest；测试不调用 GPU、OpenILT、网络或 API。
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from opc_agent.epe_dataset import EpeImagePair, build_epe_dataset


def _pair(tmp_path: Path, clip_id: str, parent: str, split: str, shift: int = 0) -> EpeImagePair:
    """创建一个目标矩形和可水平平移的印刷矩形图像对。"""
    target = np.zeros((12, 12), dtype=np.uint8)
    printed = np.zeros((12, 12), dtype=np.uint8)
    target[3:8, 3:8] = 255
    printed[3:8, 3 + shift:8 + shift] = 255
    target_path = tmp_path / f"{clip_id}-target.png"
    printed_path = tmp_path / f"{clip_id}-printed.png"
    target_ok, target_bytes = cv2.imencode(".png", target)
    assert target_ok
    target_path.write_bytes(target_bytes.tobytes())
    printed_ok, printed_bytes = cv2.imencode(".png", printed)
    assert printed_ok
    printed_path.write_bytes(printed_bytes.tobytes())
    return EpeImagePair(
        clip_id=clip_id, parent_layout=parent, split=split,
        target_path=str(target_path), printed_path=str(printed_path),
        scale_nm_per_pixel=2, tolerance_nm=1,
    )


def test_epe_dataset_hashes_inputs_and_converts_distance_to_nm(tmp_path: Path):
    """平移图形应生成非零 EPE，且纳米距离等于像素距离乘比例。"""
    dataset = build_epe_dataset([_pair(tmp_path, "clip-1", "M1_test1", "train", shift=1)])
    label = dataset.labels[0]
    assert len(label.target_sha256) == 64
    assert label.epe_n > 0
    assert label.epe_d_nm == label.epe_d_pixels * 2


def test_epe_dataset_rejects_parent_layout_split_leakage(tmp_path: Path):
    """同一父版图的不同 clip 不得跨训练集和验证集。"""
    pairs = [
        _pair(tmp_path, "clip-a", "M1_test1", "train"),
        _pair(tmp_path, "clip-b", "M1_test1", "validation"),
    ]
    with pytest.raises(ValueError, match="跨数据集合"):
        build_epe_dataset(pairs)



