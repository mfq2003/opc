"""本模块按 Recipe v2 的冻结 FRAG/EPE 点集离线计算 1 nm EPE N 与 EPE D。

输入为 v2 坐标搜索运行目录、运行配置快照、锁定的 OpenILT/ICCAD13 几何数据，以及每张版图
的 ``target.png``、``final-mask.png``、``final-printed.png`` 和 ``result.json``；输出为逐版图
JSON 汇总、逐版图 CSV 和逐 EPE 点 CSV。程序复用 v2 几何适配器：OpenILT ``dissect`` 按
``corner=16 nm``、``uniform=32 nm`` 分段，每个 segment 中点是一个固定 EPE 采样点；v2
不存在需要与 EPE 点合并统计的独立 FRAG 点集。重建点 ID 必须与结果中的完整 Recipe 点 ID
逐一相等，否则拒绝计算。距离定义为固定 target 采样点到最终印刷前景边界的最近欧氏距离；
EPE N 统计严格大于容差的点数，EPE D 只累加这些违规点的完整距离。本模块不调用 GPU、
Solver、网络或 API，也不修改原始运行工件。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field
from scipy.ndimage import distance_transform_edt

from .epe import foreground_boundary
from .recipe_v2_vision import rebuild


SAMPLING_VERSION = "recipe-v2-openilt-dissect-segment-midpoint-v1"
METRIC_VERSION = "fixed-v2-point-nearest-boundary-gt-tolerance-v1"


class PointMeasurement(BaseModel):
    """保存一个冻结 v2 EPE 点的几何身份、距离和违规状态。"""

    point_id: str
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    normal_x: int = Field(ge=-1, le=1)
    normal_y: int = Field(ge=-1, le=1)
    distance_nm: float = Field(ge=0)
    violation: bool


class MetricSummary(BaseModel):
    """保存一个冻结 v2 EPE 点集上的 EPE N、EPE D 和辅助统计。"""

    sample_count: int = Field(ge=0)
    epe_n: int = Field(ge=0)
    epe_d_nm: float = Field(ge=0)
    mean_violation_distance_nm: float = Field(ge=0)
    max_distance_nm: float = Field(ge=0)


class LayoutMeasurement(BaseModel):
    """保存一张版图的输入身份、v2 点集身份和新指标。"""

    layout: str
    seed: int = Field(ge=0)
    target_path: str
    final_mask_path: str
    final_printed_path: str
    result_path: str
    target_sha256: str = Field(min_length=64, max_length=64)
    final_mask_sha256: str = Field(min_length=64, max_length=64)
    final_printed_sha256: str = Field(min_length=64, max_length=64)
    result_sha256: str = Field(min_length=64, max_length=64)
    layout_glp_sha256: str = Field(min_length=64, max_length=64)
    recipe_sha256: str = Field(min_length=64, max_length=64)
    scale_nm_per_pixel: float = Field(gt=0)
    tolerance_nm: float = Field(ge=0)
    existing_golden_epe_15nm: float = Field(ge=0)
    metrics: MetricSummary
    points: List[PointMeasurement]


class SampledEpeReport(BaseModel):
    """保存一次多版图离线计算的完整 v2 协议和结果。"""

    schema_version: str = "2.0"
    sampling_version: str = SAMPLING_VERSION
    metric_version: str = METRIC_VERSION
    run_dir: str
    seed: int = Field(ge=0)
    tolerance_nm: float = Field(ge=0)
    fragment_parameters_nm: dict
    geometry_adapter: dict
    aggregate: MetricSummary
    layouts: List[LayoutMeasurement]


def _sha256(path: Path) -> str:
    """流式计算文件 SHA-256，绑定统计输入而不复制原工件。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_binary(path: Path) -> np.ndarray:
    """兼容中文路径读取二值 PNG，并固定使用 127 灰度阈值。"""
    if not path.is_file():
        raise FileNotFoundError(f"EPE 统计输入不存在：{path}")
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2:
        raise ValueError(f"无法读取二维灰度图：{path}")
    return image > 127


def _load_run_config(run_dir: Path) -> dict:
    """读取本次运行的配置快照，拒绝改用当前工作区中可能漂移的配置。"""
    config_path = run_dir / "config.snapshot.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少运行配置快照：{config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("运行配置快照必须是 YAML 对象")
    return config


def _apply_data_overrides(config: dict, openilt_dir: Path | None, iccad13_dir: Path | None) -> None:
    """仅覆盖云端数据路径，不改变冻结算法、FRAG 参数或评价协议。"""
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("配置快照缺少 data")
    if openilt_dir is not None:
        data["openilt_dir"] = str(openilt_dir)
    if iccad13_dir is not None:
        data["iccad13_dir"] = str(iccad13_dir)


def _v2_protocol(config: dict) -> Tuple[float, dict, dict]:
    """读取并校验 v2 比例、FRAG 参数和几何适配器，禁止回退到 v1 默认值。"""
    recipe = config.get("recipe_v2")
    if not isinstance(recipe, dict):
        raise ValueError("配置快照缺少 recipe_v2")
    scale = float(recipe.get("nm_per_coordinate", 0))
    fragment = recipe.get("fragment_parameters_nm")
    geometry = recipe.get("geometry_adapter")
    if scale <= 0:
        raise ValueError("recipe_v2.nm_per_coordinate 必须大于零")
    if not isinstance(fragment, dict) or set(fragment) != {"corner", "uniform"}:
        raise ValueError("recipe_v2.fragment_parameters_nm 必须明确包含 corner/uniform")
    if float(fragment["corner"]) <= 0 or float(fragment["uniform"]) <= 0:
        raise ValueError("v2 FRAG 参数必须大于零")
    if not isinstance(geometry, dict) or not geometry.get("version"):
        raise ValueError("配置快照缺少 v2 geometry_adapter 身份")
    return scale, dict(fragment), dict(geometry)


def _expected_layouts(config: dict) -> Tuple[str, ...]:
    """读取本次搜索声明的完整版图集合，避免对部分下载静默出报告。"""
    raw = config.get("search", {}).get("layout_parents")
    if not isinstance(raw, list) or not raw or any(not isinstance(item, str) for item in raw):
        raise ValueError("配置快照缺少 search.layout_parents，无法验证运行是否完整")
    if len(raw) != len(set(raw)):
        raise ValueError("search.layout_parents 存在重复版图")
    return tuple(raw)


def _validate_layout_coverage(found: Sequence[str], expected: Sequence[str]) -> None:
    """要求本地版图目录与运行配置完全一致。"""
    missing = sorted(set(expected) - set(found))
    unexpected = sorted(set(found) - set(expected))
    if missing or unexpected:
        raise RuntimeError(
            f"运行工件版图覆盖不完整或不一致：missing={missing}，unexpected={unexpected}"
        )


def _layout_directories(run_dir: Path, seed: int) -> List[Tuple[str, Path]]:
    """发现 v2 搜索的 ``layout/seed-N/coordinate`` 最终工件目录。"""
    result = []
    for layout_dir in sorted(run_dir.glob("M1_test*"), key=lambda path: path.name):
        coordinate_dir = layout_dir / f"seed-{seed}" / "coordinate"
        if coordinate_dir.is_dir():
            result.append((layout_dir.name, coordinate_dir))
    if not result:
        raise FileNotFoundError(f"未找到 seed-{seed} 的坐标搜索结果：{run_dir}")
    return result


def _validate_rebuilt_points(points: Sequence[object], result_payload: dict) -> str:
    """用 result 中的完整动作映射校验重建点 ID、顺序集合和 Recipe 身份。"""
    best = result_payload.get("best")
    if not isinstance(best, dict) or not isinstance(best.get("actions"), dict):
        raise ValueError("result.json 缺少 best.actions")
    rebuilt_ids = [str(point.point_id) for point in points]
    action_ids = list(best["actions"])
    point_order = result_payload.get("point_order")
    if not rebuilt_ids or len(rebuilt_ids) != len(set(rebuilt_ids)):
        raise ValueError("v2 几何重建未生成唯一非空点集")
    if set(rebuilt_ids) != set(action_ids):
        missing = sorted(set(action_ids) - set(rebuilt_ids))[:3]
        extra = sorted(set(rebuilt_ids) - set(action_ids))[:3]
        raise RuntimeError(f"v2 重建点与 result 动作点不一致：missing={missing}，extra={extra}")
    if not isinstance(point_order, list) or set(point_order) != set(rebuilt_ids):
        raise RuntimeError("result.point_order 与 v2 重建点集不一致")
    recipe_sha256 = best.get("recipe_sha256")
    if not isinstance(recipe_sha256, str) or len(recipe_sha256) != 64:
        raise ValueError("result.json 缺少有效 best.recipe_sha256")
    if result_payload.get("final_replay_equal") is not True:
        raise RuntimeError("最终独立回放未确认一致，不计算正式补充指标")
    return recipe_sha256


def measure_v2_points(
    target: np.ndarray,
    printed: np.ndarray,
    points: Sequence[object],
    scale_nm_per_pixel: float,
    tolerance_nm: float = 1.0,
) -> List[PointMeasurement]:
    """在冻结 v2 segment 中点测量到最终印刷前景边界的最近欧氏距离。"""
    target_binary = np.asarray(target, dtype=bool)
    printed_binary = np.asarray(printed, dtype=bool)
    if target_binary.shape != printed_binary.shape:
        raise ValueError("target 与 final-printed 图像尺寸必须一致")
    if tolerance_nm < 0 or scale_nm_per_pixel <= 0:
        raise ValueError("容差不能为负，像素物理比例必须为正")
    target_boundary = foreground_boundary(target_binary)
    printed_boundary = foreground_boundary(printed_binary)
    if not np.any(printed_boundary):
        raise ValueError("final-printed 不包含可测量的前景边界")
    nearest_printed = distance_transform_edt(~printed_boundary)
    height, width = target_binary.shape
    measurements = []
    for point in points:
        x, y = (int(point.base_xy[0]), int(point.base_xy[1]))
        normal_x, normal_y = (int(point.normal_xy[0]), int(point.normal_xy[1]))
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"v2 采样点越界：{point.point_id}")
        if not target_boundary[y, x]:
            raise ValueError(f"v2 采样点不在固定 target 前景边界上：{point.point_id}")
        if abs(normal_x) + abs(normal_y) != 1:
            raise ValueError(f"v2 采样点法线不是轴对齐单位向量：{point.point_id}")
        distance_nm = float(nearest_printed[y, x] * scale_nm_per_pixel)
        measurements.append(PointMeasurement(
            point_id=str(point.point_id), x=x, y=y,
            normal_x=normal_x, normal_y=normal_y,
            distance_nm=distance_nm,
            violation=bool(distance_nm > tolerance_nm),
        ))
    return measurements


def _summarize(measurements: Iterable[PointMeasurement]) -> MetricSummary:
    """按严格大于容差的违规点计算 EPE N，并只对违规完整距离求 EPE D。"""
    items = list(measurements)
    violations = [item.distance_nm for item in items if item.violation]
    return MetricSummary(
        sample_count=len(items),
        epe_n=len(violations),
        epe_d_nm=float(sum(violations)),
        mean_violation_distance_nm=float(np.mean(violations)) if violations else 0.0,
        max_distance_nm=max((item.distance_nm for item in items), default=0.0),
    )


def measure_layout(
    config: dict,
    layout: str,
    coordinate_dir: Path,
    seed: int,
    scale_nm_per_pixel: float,
    tolerance_nm: float,
) -> LayoutMeasurement:
    """精确重建一张版图的 v2 点集，完成工件与点身份门禁后计算指标。"""
    target_path = coordinate_dir / "target.png"
    final_mask_path = coordinate_dir / "final-mask.png"
    printed_path = coordinate_dir / "final-printed.png"
    result_path = coordinate_dir / "result.json"
    for path in (target_path, final_mask_path, printed_path, result_path):
        if not path.is_file():
            raise FileNotFoundError(f"{layout} 缺少完整最终工件：{path}")
    target = _load_binary(target_path)
    final_mask = _load_binary(final_mask_path)
    printed = _load_binary(printed_path)
    if target.shape != final_mask.shape or target.shape != printed.shape:
        raise ValueError(f"{layout} 的 target/mask/printed 尺寸不一致")
    rebuilt_target, points, layout_glp_sha256 = rebuild(config, layout)
    if not np.array_equal(target, np.asarray(rebuilt_target) >= 0.5):
        raise RuntimeError(f"{layout} 的 target.png 与锁定 GLP 重建 target 不一致")
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    recipe_sha256 = _validate_rebuilt_points(points, result_payload)
    measurements = measure_v2_points(target, printed, points, scale_nm_per_pixel, tolerance_nm)
    golden_epe = float(result_payload["best"]["metrics"]["epe"])
    return LayoutMeasurement(
        layout=layout, seed=seed,
        target_path=str(target_path), final_mask_path=str(final_mask_path),
        final_printed_path=str(printed_path), result_path=str(result_path),
        target_sha256=_sha256(target_path), final_mask_sha256=_sha256(final_mask_path),
        final_printed_sha256=_sha256(printed_path), result_sha256=_sha256(result_path),
        layout_glp_sha256=layout_glp_sha256, recipe_sha256=recipe_sha256,
        scale_nm_per_pixel=scale_nm_per_pixel, tolerance_nm=tolerance_nm,
        existing_golden_epe_15nm=golden_epe,
        metrics=_summarize(measurements), points=measurements,
    )


def build_report(
    run_dir: Path,
    seed: int = 0,
    tolerance_nm: float = 1.0,
    openilt_dir: Path | None = None,
    iccad13_dir: Path | None = None,
) -> SampledEpeReport:
    """对配置声明的全部 v2 搜索版图构建单一、不可部分成功的报告。"""
    run_dir = Path(run_dir)
    config = _load_run_config(run_dir)
    _apply_data_overrides(config, openilt_dir, iccad13_dir)
    scale, fragment, geometry = _v2_protocol(config)
    expected = _expected_layouts(config)
    directories = _layout_directories(run_dir, seed)
    _validate_layout_coverage([layout for layout, _ in directories], expected)
    by_layout = dict(directories)
    layouts = [
        measure_layout(config, layout, by_layout[layout], seed, scale, tolerance_nm)
        for layout in expected
    ]
    return SampledEpeReport(
        run_dir=str(run_dir), seed=seed, tolerance_nm=tolerance_nm,
        fragment_parameters_nm=fragment, geometry_adapter=geometry,
        aggregate=_summarize(point for item in layouts for point in item.points),
        layouts=layouts,
    )


def write_report(report: SampledEpeReport, output_dir: Path) -> Tuple[Path, Path, Path]:
    """写出完整 JSON、逐图摘要 CSV 和逐 v2 EPE 点审计 CSV。"""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "sampled-epe-metrics.json"
    summary_csv_path = root / "sampled-epe-summary.csv"
    points_csv_path = root / "sampled-epe-points.csv"
    json_path.write_text(
        json.dumps(report.dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "layout", "sample_count", "epe_n", "epe_d_nm",
            "mean_violation_distance_nm", "max_distance_nm", "existing_golden_epe_15nm",
        ])
        writer.writeheader()
        for layout in report.layouts:
            writer.writerow({
                "layout": layout.layout, **layout.metrics.dict(),
                "existing_golden_epe_15nm": layout.existing_golden_epe_15nm,
            })
        writer.writerow({
            "layout": "ALL", **report.aggregate.dict(), "existing_golden_epe_15nm": "",
        })
    with points_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "layout", "seed", "point_id", "x", "y", "normal_x", "normal_y",
            "distance_nm", "violation", "tolerance_nm",
        ])
        writer.writeheader()
        for layout in report.layouts:
            for point in layout.points:
                writer.writerow({
                    "layout": layout.layout, "seed": layout.seed, "point_id": point.point_id,
                    "x": point.x, "y": point.y, "normal_x": point.normal_x,
                    "normal_y": point.normal_y, "distance_nm": f"{point.distance_nm:.9f}",
                    "violation": str(point.violation).lower(),
                    "tolerance_nm": f"{layout.tolerance_nm:.9f}",
                })
    return json_path, summary_csv_path, points_csv_path


def main(argv: Sequence[str] | None = None) -> int:
    """提供可直接上传云端执行的只读 v2 EPE N/D 计算入口。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.sampled_epe_metrics")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance-nm", type=float, default=1.0)
    parser.add_argument("--openilt-dir", type=Path)
    parser.add_argument("--iccad13-dir", type=Path)
    args = parser.parse_args(argv)
    report = build_report(
        run_dir=args.run_dir, seed=args.seed, tolerance_nm=args.tolerance_nm,
        openilt_dir=args.openilt_dir, iccad13_dir=args.iccad13_dir,
    )
    for path in write_report(report, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
