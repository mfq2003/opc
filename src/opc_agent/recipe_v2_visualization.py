"""本模块把 Recipe PPO v2 实际使用的冻结 observation 保存为可追溯汇报样例。

输入是已经 reset 的 ``LocalEPEEpisode``，因此图像与 NPZ 直接来自送入 Actor 的
``FrozenObservationCache``，不会重新计算或替换任一通道。输出位于调用方指定的
``runs/<run_id>/ppo-input-examples``：每个样例包含五通道汇总 PNG、精确 image/vector
NPZ，以及记录点几何、observation 哈希和文件哈希的 manifest。本模块不运行 PPO、
不修改 OpenILT，也不把可视化结果用于训练或验收指标。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

from .recipe_v2 import LocalEPEEpisode
from .recipe_v2_contract import EPEControlPoint


PPO_INPUT_VISUALIZATION_VERSION = "recipe-v2-ppo-input-5ch-montage-v1"
PPO_INPUT_SELECTION_POLICY = "geometry-diverse-by-point-id-v1"
PPO_INPUT_CHANNEL_NAMES = (
    "target",
    "baseline-mask",
    "printed-nominal",
    "point-marker",
    "segment-normal",
)


def _sha256_bytes(payload: bytes) -> str:
    """计算文件内容 SHA256。"""
    return hashlib.sha256(payload).hexdigest()


def _select_report_points(
    points: Sequence[EPEControlPoint], example_count: int
) -> Tuple[EPEControlPoint, ...]:
    """按几何新颖度确定性抽样，避免按 reward 或结果挑选汇报样例。"""
    count = int(example_count)
    if count <= 0:
        raise ValueError("PPO 输入可视化 example_count 必须为正整数")
    remaining = sorted(points, key=lambda point: point.point_id)
    if not remaining:
        raise ValueError("PPO 输入可视化至少需要一个 EPE 控制点")
    selected = []
    seen_edges = set()
    seen_normals = set()
    seen_corners = set()
    while remaining and len(selected) < count:
        def novelty(point: EPEControlPoint) -> tuple:
            edge = (point.polygon_index, point.source_edge_index)
            corners = (point.start_corner_type, point.end_corner_type)
            return (
                int(edge not in seen_edges),
                int(point.normal_xy not in seen_normals),
                int(corners not in seen_corners),
            )

        best_score = max(novelty(point) for point in remaining)
        point = next(point for point in remaining if novelty(point) == best_score)
        remaining.remove(point)
        selected.append(point)
        seen_edges.add((point.polygon_index, point.source_edge_index))
        seen_normals.add(point.normal_xy)
        seen_corners.add((point.start_corner_type, point.end_corner_type))
    return tuple(selected)


def _channel_tile(channel: np.ndarray, title: str) -> np.ndarray:
    """把一个浮点通道转换为带英文标题的等比例灰度面板。"""
    value = np.asarray(channel, dtype=np.float32)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("PPO 输入图像通道必须是有限二维数组")
    clipped = np.clip(value, 0.0, 1.0)
    image = np.rint(clipped * 255.0).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    header = np.full((26, image.shape[1], 3), 28, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, image))


def _overlay_tile(image: np.ndarray, point: EPEControlPoint) -> np.ndarray:
    """由五个真实输入通道生成一张仅用于阅读的轮廓叠加面板。"""
    patch_size = int(image.shape[1])
    canvas = np.full((patch_size, patch_size, 3), 18, dtype=np.uint8)
    contour_specs = (
        (image[0], (210, 210, 210)),
        (image[1], (60, 210, 60)),
        (image[2], (230, 80, 210)),
    )
    for channel, color in contour_specs:
        binary = np.asarray(channel > 0.5, dtype=np.uint8) * 255
        contours, _ = cv2.findContours(
            binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, color, 1, lineType=cv2.LINE_AA)
    center = patch_size // 2
    cv2.circle(canvas, (center, center), 3, (40, 40, 255), -1, cv2.LINE_AA)
    endpoint = (
        center + int(point.normal_xy[0]) * 18,
        center + int(point.normal_xy[1]) * 18,
    )
    cv2.arrowedLine(
        canvas,
        (center, center),
        endpoint,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.35,
    )
    return np.vstack((
        _channel_tile(np.zeros_like(image[0]), "overlay")[:26],
        canvas,
    ))


def _render_montage(image: np.ndarray, point: EPEControlPoint) -> np.ndarray:
    """渲染五个原始通道及一张解释性 overlay，并添加几何说明。"""
    value = np.asarray(image, dtype=np.float32)
    if value.ndim != 3 or value.shape[0] != len(PPO_INPUT_CHANNEL_NAMES):
        raise ValueError("PPO 输入图像必须是 5xHxW")
    if value.shape[1] != value.shape[2]:
        raise ValueError("PPO 输入 patch 必须为正方形")
    tiles = [
        _channel_tile(value[index], name)
        for index, name in enumerate(PPO_INPUT_CHANNEL_NAMES)
    ]
    tiles.append(_overlay_tile(value, point))
    separator = np.full((tiles[0].shape[0], 4, 3), 8, dtype=np.uint8)
    row = tiles[0]
    for tile in tiles[1:]:
        row = np.hstack((row, separator, tile))
    footer = np.full((42, row.shape[1], 3), 28, dtype=np.uint8)
    label = (
        f"point={point.point_id[:28]}  base={point.base_xy}  "
        f"normal={point.normal_xy}  edge={point.polygon_index}:{point.source_edge_index}"
    )
    cv2.putText(
        footer,
        label,
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    montage = np.vstack((row, footer))
    return cv2.resize(
        montage,
        (montage.shape[1] * 2, montage.shape[0] * 2),
        interpolation=cv2.INTER_NEAREST,
    )


def save_v2_ppo_input_examples(
    episode: LocalEPEEpisode,
    output_dir: Path,
    example_count: int,
    selection_policy: str = PPO_INPUT_SELECTION_POLICY,
) -> Dict[str, object]:
    """保存少量真实 Actor 输入样例，并返回可嵌入运行 JSON 的追溯摘要。"""
    if selection_policy != PPO_INPUT_SELECTION_POLICY:
        raise ValueError(
            "PPO 输入可视化 selection_policy 与已实现的固定抽样协议不一致"
        )
    cache = episode.observation_cache
    selected = _select_report_points(episode.epe_points, example_count)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    examples = []
    for index, point in enumerate(selected):
        observation = cache.get(point.point_id)
        point_digest = hashlib.sha256(point.point_id.encode("utf-8")).hexdigest()
        stem = f"example-{index:02d}-{point_digest[:12]}"
        png_name = f"{stem}-5ch.png"
        npz_name = f"{stem}-input.npz"
        montage = _render_montage(observation.image, point)
        encoded_ok, encoded = cv2.imencode(".png", montage)
        if not encoded_ok:
            raise RuntimeError(f"PPO 输入可视化 PNG 编码失败：{point.point_id}")
        png_bytes = encoded.tobytes()
        (root / png_name).write_bytes(png_bytes)
        np.savez_compressed(
            root / npz_name,
            image=np.asarray(observation.image, dtype=np.float32),
            vector=np.asarray(observation.vector, dtype=np.float32),
        )
        npz_bytes = (root / npz_name).read_bytes()
        examples.append({
            "index": index,
            "point_id": point.point_id,
            "polygon_index": point.polygon_index,
            "source_edge_index": point.source_edge_index,
            "segment_index": point.segment_index,
            "base_xy": list(point.base_xy),
            "normal_xy": list(point.normal_xy),
            "observation_version": observation.version,
            "observation_sha256": observation.sha256,
            "png": png_name,
            "png_sha256": _sha256_bytes(png_bytes),
            "npz": npz_name,
            "npz_sha256": _sha256_bytes(npz_bytes),
        })
    manifest = {
        "visualization_version": PPO_INPUT_VISUALIZATION_VERSION,
        "selection_policy": selection_policy,
        "selection_depends_on_reward_or_result": False,
        "configured_example_count": int(example_count),
        "saved_example_count": len(examples),
        "channel_names": list(PPO_INPUT_CHANNEL_NAMES),
        "patch_size": cache.patch_size,
        "observation_cache_sha256": cache.cache_sha256,
        "baseline_state_sha256": cache.baseline_state_sha256,
        "examples": examples,
    }
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, indent=2
    ).encode("utf-8")
    (root / "manifest.json").write_bytes(manifest_bytes)
    return {
        "directory": root.name,
        "manifest": f"{root.name}/manifest.json",
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        **manifest,
    }
