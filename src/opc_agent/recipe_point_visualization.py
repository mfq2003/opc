"""本模块为当前点级 Recipe PPO 主线生成可复现的 ICCAD2013 打点总览图。

输入为 paper_repro YAML、训练主线使用的只读 OpenILT solver，以及可选的完整或 smoke 运行目录；
输出统一写入 outputs/recipe_point_visualization。训练前图展示原始 target 上的默认 EPE/FRAG 点，
训练后图从版本化 Recipe JSON 读取九分类位移，绘制默认点、PPO 点和切向箭头。本模块不训练 PPO、
不调用 solver.solve、不修改 OpenILT，也不复用语义不同的历史 mask 边段可视化。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .metrics import DISPLACEMENT_CLASSES_NM
from .recipe_contract import PPO_RECIPE_LABEL_VERSION, RECIPE_POINT_VERSION

if TYPE_CHECKING:
    from .recipe_ppo import RecipePoint


VISUALIZATION_VERSION = "recipe-point-overview-v1"
DEFAULT_OUTPUT_ROOT = Path("outputs/recipe_point_visualization")
TARGET_COLOR = (55, 55, 55)
EPE_COLOR = (40, 80, 220)
FRAG_COLOR = (210, 115, 35)
BASE_COLOR = (145, 145, 145)
ARROW_COLOR = (20, 165, 225)
TEXT_COLOR = (35, 35, 35)


def _encode_png(image: np.ndarray) -> bytes:
    """通过内存编码 PNG，兼容 Windows 中文输出路径。"""
    ok, encoded = cv2.imencode(".png", np.asarray(image, dtype=np.uint8))
    if not ok:
        raise RuntimeError("OpenCV 无法编码 Recipe 点可视化 PNG")
    return encoded.tobytes()


def _point_position(point: "RecipePoint", displacement_nm: float, nm_per_coordinate: float) -> Tuple[int, int]:
    """按训练环境的绝对切向位移语义计算可视化点坐标。"""
    delta = int(round(float(displacement_nm) / float(nm_per_coordinate)))
    return (
        int(point.base_x + point.tangent_x * delta),
        int(point.base_y + point.tangent_y * delta),
    )


def _draw_marker(
    image: np.ndarray,
    point: Tuple[int, int],
    task_type: str,
    color: Tuple[int, int, int],
    radius: int,
    filled: bool,
) -> None:
    """用圆形表示 EPE、方形表示 FRAG，避免只依赖颜色区分。"""
    thickness = -1 if filled else max(1, radius // 2)
    if task_type == "EPE":
        cv2.circle(image, point, radius, color, thickness, lineType=cv2.LINE_AA)
        return
    if task_type == "FRAG":
        cv2.rectangle(
            image,
            (point[0] - radius, point[1] - radius),
            (point[0] + radius, point[1] + radius),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        return
    raise ValueError(f"未知 Recipe 点类型：{task_type}")


def render_recipe_point_overview(
    polygons: Sequence[Sequence[Tuple[int, int]]],
    points: Sequence["RecipePoint"],
    offsets_nm: Sequence[float],
    nm_per_coordinate: float,
    output_path: Path,
    clip_id: str,
    split: str,
    mode: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """绘制单张 target 的默认点或 PPO 位移前后对比总览。"""
    if mode not in {"before-training", "after-training"}:
        raise ValueError("mode 必须是 before-training 或 after-training")
    point_items = tuple(points)
    polygon_items = tuple(tuple((int(x), int(y)) for x, y in polygon) for polygon in polygons)
    offsets = np.asarray(offsets_nm, dtype=np.float64)
    if not polygon_items:
        raise ValueError("Recipe 可视化至少需要一个 target 多边形")
    if offsets.shape != (len(point_items),):
        raise ValueError("可视化位移数量与 Recipe 点数不一致")
    classes = np.asarray(DISPLACEMENT_CLASSES_NM, dtype=np.float64)
    if np.any(~np.isin(offsets, classes)):
        raise ValueError("可视化位移必须精确属于九分类代表值")
    if nm_per_coordinate <= 0:
        raise ValueError("nm_per_coordinate 必须大于零")

    moved = [
        _point_position(point, float(offset), nm_per_coordinate)
        for point, offset in zip(point_items, offsets)
    ]
    coordinates = [coordinate for polygon in polygon_items for coordinate in polygon]
    coordinates.extend((point.base_x, point.base_y) for point in point_items)
    coordinates.extend(moved)
    min_x = min(value[0] for value in coordinates) - 72
    min_y = min(value[1] for value in coordinates) - 72
    max_x = max(value[0] for value in coordinates) + 72
    max_y = max(value[1] for value in coordinates) + 72
    header_height = 92
    width = max(640, int(max_x - min_x + 1))
    plot_height = max(280, int(max_y - min_y + 1))
    image = np.full((plot_height + header_height, width, 3), 250, dtype=np.uint8)

    def transform(coordinate: Tuple[int, int]) -> Tuple[int, int]:
        return int(coordinate[0] - min_x), int(coordinate[1] - min_y + header_height)

    for polygon in polygon_items:
        contour = np.asarray([transform(value) for value in polygon], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [contour], True, TARGET_COLOR, 2, lineType=cv2.LINE_AA)

    marker_radius = max(4, min(8, int(round(min(width, plot_height) / 180))))
    for index, (point, offset, moved_coordinate) in enumerate(zip(point_items, offsets, moved)):
        base = transform((point.base_x, point.base_y))
        destination = transform(moved_coordinate)
        color = EPE_COLOR if point.task_type == "EPE" else FRAG_COLOR
        if mode == "after-training":
            _draw_marker(image, base, point.task_type, BASE_COLOR, marker_radius, filled=False)
            if float(offset) != 0.0:
                cv2.arrowedLine(
                    image, base, destination, ARROW_COLOR, 2,
                    line_type=cv2.LINE_AA, tipLength=0.28,
                )
        _draw_marker(image, destination, point.task_type, color, marker_radius, filled=True)
        cv2.putText(
            image,
            str(index),
            (destination[0] + marker_radius + 2, max(header_height + 10, destination[1] - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            TEXT_COLOR,
            1,
            lineType=cv2.LINE_AA,
        )

    epe_count = sum(point.task_type == "EPE" for point in point_items)
    frag_count = len(point_items) - epe_count
    seed_text = "" if seed is None else f" | seed={seed}"
    title = f"{clip_id} | {mode}{seed_text}"
    subtitle = f"split={split} | EPE={epe_count} | FRAG={frag_count} | points={len(point_items)}"
    cv2.putText(image, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(image, subtitle, (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.47, TEXT_COLOR, 1, cv2.LINE_AA)
    legend_y = 74
    cv2.line(image, (18, legend_y), (43, legend_y), TARGET_COLOR, 2, cv2.LINE_AA)
    cv2.putText(image, "target", (49, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1, cv2.LINE_AA)
    _draw_marker(image, (125, legend_y), "EPE", EPE_COLOR, 5, filled=True)
    cv2.putText(image, "EPE", (135, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1, cv2.LINE_AA)
    _draw_marker(image, (188, legend_y), "FRAG", FRAG_COLOR, 5, filled=True)
    cv2.putText(image, "FRAG", (199, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1, cv2.LINE_AA)
    if mode == "after-training":
        cv2.arrowedLine(image, (255, legend_y), (283, legend_y), ARROW_COLOR, 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(image, "PPO shift", (290, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1, cv2.LINE_AA)

    destination_path = Path(output_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    content = _encode_png(image)
    destination_path.write_bytes(content)
    return {
        "clip_id": clip_id,
        "split": split,
        "mode": mode,
        "seed": seed,
        "point_count": len(point_items),
        "epe_point_count": epe_count,
        "frag_point_count": frag_count,
        "nonzero_displacement_count": int(np.count_nonzero(offsets)),
        "displacement_class_counts": [int(np.count_nonzero(offsets == value)) for value in classes],
        "image_path": str(destination_path),
        "image_sha256": hashlib.sha256(content).hexdigest(),
        "image_shape": list(image.shape),
    }


def _clip_jobs(config: Dict[str, Any]) -> List[Tuple[str, str]]:
    """按训练、验证、测试顺序返回十个不重叠父版图。"""
    data = config["data"]
    jobs = [
        (str(parent), split)
        for split, key in (
            ("train", "train_parents"),
            ("validation", "validation_parents"),
            ("test", "test_parents"),
        )
        for parent in data[key]
    ]
    clip_ids = [clip_id for clip_id, _split in jobs]
    if len(clip_ids) != 10 or len(set(clip_ids)) != 10:
        raise ValueError("Recipe 点可视化要求配置包含十个不重叠 ICCAD2013 父版图")
    return jobs


def _assert_openilt_clean(openilt_dir: Path) -> str:
    """返回当前提交，并拒绝 OpenILT tracked 文件发生任何变化。"""
    revision = subprocess.run(
        ["git", "-C", str(openilt_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "-C", str(openilt_dir), "diff", "--quiet", "HEAD", "--"],
        check=False,
    )
    if diff.returncode == 1:
        raise RuntimeError("Recipe 点可视化检测到 OpenILT tracked 文件变化")
    if diff.returncode != 0:
        raise RuntimeError("Recipe 点可视化无法核验 OpenILT tracked diff")
    return revision


def _release_cuda_cache() -> None:
    """在调用方删除逐版图 solver 后回收 CUDA 缓存，避免十图累计显存。"""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _write_summary(path: Path, payload: Dict[str, Any]) -> Path:
    """确定性写入带排序键的可视化摘要。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def visualize_before_training(
    config: Dict[str, Any],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    solver_factory: Optional[Callable[[Dict[str, Any], str], Any]] = None,
) -> Path:
    """使用训练主线的同一打点逻辑生成十张默认 Recipe 总览。"""
    if solver_factory is None:
        from .workflow import create_recipe_point_solver

        solver_factory = create_recipe_point_solver
    sys.dont_write_bytecode = True
    openilt_dir = Path(config["data"]["openilt_dir"])
    revision = _assert_openilt_clean(openilt_dir)
    destination = Path(output_root) / "before-training"
    clip_summaries = []
    for clip_id, split in _clip_jobs(config):
        solver = solver_factory(config, clip_id)
        try:
            clip_summaries.append(render_recipe_point_overview(
                polygons=solver.target_polygons,
                points=solver.recipe_points,
                offsets_nm=[0.0] * len(solver.recipe_points),
                nm_per_coordinate=float(solver.nm_per_coordinate),
                output_path=destination / f"{clip_id}-points.png",
                clip_id=clip_id,
                split=split,
                mode="before-training",
            ))
        finally:
            del solver
            _release_cuda_cache()
    post_revision = _assert_openilt_clean(openilt_dir)
    if post_revision != revision:
        raise RuntimeError("Recipe 点可视化期间 OpenILT 提交发生变化")
    summary = {
        "schema_version": "1.0",
        "visualization_version": VISUALIZATION_VERSION,
        "mode": "before-training",
        "output_root": str(destination),
        "openilt_commit": revision,
        "openilt_mutation": "none",
        "clip_count": len(clip_summaries),
        "clips": clip_summaries,
    }
    return _write_summary(destination / "visualization-summary.json", summary)


def _load_recipe_points(recipe: Dict[str, Any], solver: Any) -> Tuple[Tuple["RecipePoint", ...], List[float]]:
    """校验 Recipe 与当前 GLP 打点完全一致，并返回逐点位移。"""
    if recipe.get("label_version") != PPO_RECIPE_LABEL_VERSION:
        raise ValueError("训练后可视化只接受当前 ppo-recipe-point-v1 Recipe")
    if recipe.get("point_version") != RECIPE_POINT_VERSION:
        raise ValueError("训练后可视化 Recipe 点版本不一致")
    if recipe.get("layout_sha256") != solver.layout_sha256:
        raise ValueError("训练后可视化 Recipe 与当前 GLP 哈希不一致")
    labels = recipe.get("labels")
    if not isinstance(labels, list) or len(labels) != len(solver.recipe_points):
        raise ValueError("训练后可视化 Recipe 标签数量与当前打点不一致")
    points = []
    offsets = []
    for expected, label in zip(solver.recipe_points, labels):
        if not isinstance(label, dict) or label.get("point") != asdict(expected):
            raise ValueError("训练后可视化 Recipe 点几何与当前 solver 不一致")
        points.append(expected)
        offsets.append(float(label["ppo_displacement_nm"]))
    return tuple(points), offsets


def visualize_after_training(
    config: Dict[str, Any],
    run_dir: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    solver_factory: Optional[Callable[[Dict[str, Any], str], Any]] = None,
) -> Path:
    """从 smoke 或完整运行的 Recipe JSON 生成逐 seed 十图前后对比。"""
    if solver_factory is None:
        from .workflow import create_recipe_point_solver

        solver_factory = create_recipe_point_solver
    sys.dont_write_bytecode = True
    source_root = Path(run_dir)
    stage_path = source_root / "stage-result.json"
    if not stage_path.is_file():
        raise FileNotFoundError(f"训练后可视化缺少 stage-result.json：{stage_path}")
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    if stage.get("environment") != "simpleopc-recipe-point-v1":
        raise ValueError("训练后可视化只接受当前 Recipe 点级 PPO 运行")
    openilt_dir = Path(config["data"]["openilt_dir"])
    revision = _assert_openilt_clean(openilt_dir)
    destination = Path(output_root) / "after-training" / source_root.name
    split_by_clip = dict(_clip_jobs(config))
    clip_summaries = []
    for clip_entry in stage.get("clips", []):
        clip_id = str(clip_entry["clip_id"])
        if clip_id not in split_by_clip:
            raise ValueError(f"stage-result 包含配置外版图：{clip_id}")
        solver = solver_factory(config, clip_id)
        try:
            for raw_path in clip_entry.get("ppo_recipes", []):
                recipe_path = Path(raw_path)
                if not recipe_path.is_file():
                    raise FileNotFoundError(f"训练后可视化缺少 PPO Recipe：{recipe_path}")
                recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
                points, offsets = _load_recipe_points(recipe, solver)
                seed = int(recipe["seed"])
                clip_summary = render_recipe_point_overview(
                    polygons=solver.target_polygons,
                    points=points,
                    offsets_nm=offsets,
                    nm_per_coordinate=float(solver.nm_per_coordinate),
                    output_path=destination / f"seed-{seed}" / f"{clip_id}-points.png",
                    clip_id=clip_id,
                    split=split_by_clip[clip_id],
                    mode="after-training",
                    seed=seed,
                )
                clip_summary["recipe_path"] = str(recipe_path)
                clip_summary["recipe_sha256"] = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
                clip_summaries.append(clip_summary)
        finally:
            del solver
            _release_cuda_cache()
    if not clip_summaries:
        raise RuntimeError("训练后可视化未找到任何 PPO Recipe")
    post_revision = _assert_openilt_clean(openilt_dir)
    if post_revision != revision:
        raise RuntimeError("训练后可视化期间 OpenILT 提交发生变化")
    summary = {
        "schema_version": "1.0",
        "visualization_version": VISUALIZATION_VERSION,
        "mode": "after-training",
        "source_run_dir": str(source_root),
        "output_root": str(destination),
        "openilt_commit": revision,
        "openilt_mutation": "none",
        "image_count": len(clip_summaries),
        "clips": clip_summaries,
    }
    return _write_summary(destination / "visualization-summary.json", summary)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """解析训练前或训练后可视化命令，并打印摘要路径。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.recipe_point_visualization")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, help="可选；提供后生成该运行的逐 seed 训练后对比图")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("配置根节点必须是映射")
    if args.run_dir is None:
        summary_path = visualize_before_training(config, output_root=args.output_root)
    else:
        summary_path = visualize_after_training(config, args.run_dir, output_root=args.output_root)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
