"""本模块把目标/印刷二值图对转换为可追溯的逐点 EPE 标签数据集。

输入为版本化 JSON 图像对清单，包含 clip、父版图、数据划分、路径、比例和容差；输出为带文件哈希的
EPE N/EPE D JSON 标签集。关键依赖为 OpenCV、NumPy、Pydantic 与本项目 epe 模块；本模块仅在 CPU
读取已有图片并计算标签，不执行光刻仿真、GPU 优化、网络下载或 API 调用。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
from pydantic import BaseModel, Field, validator

from .epe import measure_boundary_epe


class EpeImagePair(BaseModel):
    """描述一个目标图和对应印刷图的路径、来源与物理比例。"""

    schema_version: str = "1.0"
    clip_id: str
    parent_layout: str
    split: str
    target_path: str
    printed_path: str
    scale_nm_per_pixel: float = Field(gt=0)
    tolerance_nm: float = Field(ge=0)

    @validator("split")
    def _valid_split(cls, value: str) -> str:
        if value not in {"train", "validation", "test"}:
            raise ValueError("split 必须是 train、validation 或 test")
        return value


class EpeLabel(BaseModel):
    """保存一个 clip 的确定性 EPE 标签及输入文件哈希。"""

    schema_version: str = "1.0"
    metric_version: str = "boundary-distance-v1"
    clip_id: str
    parent_layout: str
    split: str
    target_path: str
    printed_path: str
    target_sha256: str = Field(min_length=64, max_length=64)
    printed_sha256: str = Field(min_length=64, max_length=64)
    scale_nm_per_pixel: float = Field(gt=0)
    tolerance_nm: float = Field(ge=0)
    sample_count: int = Field(ge=0)
    epe_n: int = Field(ge=0)
    epe_d_pixels: float = Field(ge=0)
    epe_d_nm: float = Field(ge=0)
    mean_distance_nm: float = Field(ge=0)


class EpeLabelDataset(BaseModel):
    """保存无重复 clip 且可版本化的 EPE 标签集合。"""

    schema_version: str = "1.0"
    metric_version: str = "boundary-distance-v1"
    labels: List[EpeLabel]


def _sha256(path: Path) -> str:
    """流式计算输入图片 SHA-256，避免数据库保存图片二进制。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_binary(path: Path) -> np.ndarray:
    """以灰度读取图片并使用固定 127 阈值转换为二值图。"""
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    if image.ndim != 2:
        raise ValueError(f"图片必须是二维灰度图：{path}")
    return image > 127


def build_epe_label(pair: EpeImagePair) -> EpeLabel:
    """读取一个图像对，计算像素/纳米单位 EPE 并绑定输入哈希。"""
    target_path = Path(pair.target_path)
    printed_path = Path(pair.printed_path)
    if not target_path.is_file() or not printed_path.is_file():
        raise FileNotFoundError(f"EPE 图像对缺失：{target_path} 或 {printed_path}")
    target = _load_binary(target_path)
    printed = _load_binary(printed_path)
    tolerance_pixels = pair.tolerance_nm / pair.scale_nm_per_pixel
    measurement = measure_boundary_epe(target, printed, tolerance_pixels=tolerance_pixels)
    return EpeLabel(
        clip_id=pair.clip_id,
        parent_layout=pair.parent_layout,
        split=pair.split,
        target_path=str(target_path),
        printed_path=str(printed_path),
        target_sha256=_sha256(target_path),
        printed_sha256=_sha256(printed_path),
        scale_nm_per_pixel=pair.scale_nm_per_pixel,
        tolerance_nm=pair.tolerance_nm,
        sample_count=measurement.sample_count,
        epe_n=measurement.epe_n,
        epe_d_pixels=measurement.epe_d,
        epe_d_nm=measurement.epe_d * pair.scale_nm_per_pixel,
        mean_distance_nm=measurement.mean_distance_pixels * pair.scale_nm_per_pixel,
    )


def build_epe_dataset(pairs: List[EpeImagePair]) -> EpeLabelDataset:
    """按 clip_id 排序构建标签集，并拒绝重复 clip 或父版图跨集合泄漏。"""
    clip_ids = [pair.clip_id for pair in pairs]
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError("EPE 图像对中存在重复 clip_id")
    parent_splits = {}
    for pair in pairs:
        previous = parent_splits.setdefault(pair.parent_layout, pair.split)
        if previous != pair.split:
            raise ValueError(f"父版图 {pair.parent_layout} 跨数据集合")
    labels = [build_epe_label(pair) for pair in sorted(pairs, key=lambda item: item.clip_id)]
    return EpeLabelDataset(labels=labels)


def main(argv: List[str] | None = None) -> int:
    """从 JSON 图像对清单生成可审计的 EPE 标签 JSON。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.epe_dataset")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw_pairs = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(raw_pairs, list):
        raise ValueError("EPE manifest 根节点必须是数组")
    dataset = build_epe_dataset([EpeImagePair.parse_obj(item) for item in raw_pairs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset.dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



