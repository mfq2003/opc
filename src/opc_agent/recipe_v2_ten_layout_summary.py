"""本模块汇总 Recipe v2 十图启发式搜索的 PVB、1 nm EPE N 与 EPE D。

输入为互不重叠、合计覆盖 ``M1_test1`` 至 ``M1_test10`` 的搜索运行目录；输出逐图基线/最终
指标、十图宏平均和按采样点汇总的微观统计。EPE 点严格复用 v2 的 OpenILT ``dissect``
分段中点，重建点 ID 必须与各自 ``result.json`` 完整匹配。若旧运行没有保存
``baseline-printed.png``，程序会用同一冻结配置独立回放一次全零 Recipe，并先核对原有
L2、15 nm Golden EPE、PVB 和 J；回放图只写入新输出目录，不修改历史运行。1 nm EPE N/D
属于最终后处理指标，不参与原坐标搜索的候选接受护栏。本模块需要锁定 OpenILT/CUDA 环境，
不访问网络或 API，也不把逐图启发式结果解释为未见版图泛化。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import yaml

from .recipe_v2_search import _write_binary_png
from .recipe_v2_small_train import _build_episode
from .recipe_v2_vision import rebuild
from .sampled_epe_metrics import (
    METRIC_VERSION,
    SAMPLING_VERSION,
    _load_binary,
    _summarize,
    _validate_rebuilt_points,
    measure_v2_points,
)


SUMMARY_VERSION = "recipe-v2-ten-layout-pvb-epe-1nm-v1"
EXPECTED_LAYOUTS = tuple(f"M1_test{index}" for index in range(1, 11))


def _sha256(path: Path) -> str:
    """计算工件哈希，使合并报告可追溯到具体输入。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_config(run_dir: Path) -> Tuple[dict, Path]:
    """读取搜索运行自己的配置快照，拒绝用当前工作区配置替代历史协议。"""
    path = Path(run_dir) / "config.snapshot.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"搜索运行缺少配置快照：{path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"配置快照不是 YAML 对象：{path}")
    return config, path


def _core_protocol(config: Mapping[str, object]) -> dict:
    """提取跨运行必须完全相同的 solver、几何、动作和 Golden 全局协议。"""
    recipe = config["recipe_v2"]
    golden = config["golden"]
    return {
        "openilt": config["openilt"],
        "solver": recipe["solver"],
        "nm_per_coordinate": recipe["nm_per_coordinate"],
        "fragment_parameters_nm": recipe["fragment_parameters_nm"],
        "epe_normal_offsets_nm": recipe["epe_normal_offsets_nm"],
        "epe_probe_distance_nm": recipe["epe_probe_distance_nm"],
        "geometry_adapter": recipe["geometry_adapter"],
        "control_conflict_policy": recipe["control_conflict_policy"],
        "reward_weights": recipe["reward"]["weights"],
        "golden_global": {
            key: golden.get(key)
            for key in ("evaluator", "constraint_coordinate", "source_sha256")
        },
    }


def _discover_layouts(run_dirs: Sequence[Path], seed: int) -> Dict[str, dict]:
    """合并多个运行中的版图，要求十图完整且每图只有一个来源。"""
    discovered: Dict[str, dict] = {}
    reference_protocol = None
    for run_dir in map(Path, run_dirs):
        config, config_path = _read_config(run_dir)
        protocol = _core_protocol(config)
        if reference_protocol is None:
            reference_protocol = protocol
        elif protocol != reference_protocol:
            raise RuntimeError(f"搜索运行核心协议不一致：{run_dir}")
        for layout_dir in sorted(run_dir.glob("M1_test*"), key=lambda path: path.name):
            coordinate = layout_dir / f"seed-{seed}" / "coordinate"
            if not (coordinate / "result.json").is_file():
                continue
            if layout_dir.name in discovered:
                raise RuntimeError(f"版图在多个搜索运行中重复：{layout_dir.name}")
            discovered[layout_dir.name] = {
                "run_dir": run_dir,
                "coordinate_dir": coordinate,
                "config": config,
                "config_path": config_path,
            }
    missing = sorted(set(EXPECTED_LAYOUTS) - set(discovered))
    unexpected = sorted(set(discovered) - set(EXPECTED_LAYOUTS))
    if missing or unexpected:
        raise RuntimeError(f"十图覆盖不完整：missing={missing}，unexpected={unexpected}")
    return {layout: discovered[layout] for layout in EXPECTED_LAYOUTS}


def _same_number(left: float, right: float) -> bool:
    """对整数型Golden指标保持严格语义，同时容忍JSON浮点表示。"""
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-9))


def _validate_baseline_replay(reset_info: Mapping[str, object], result: dict, layout: str) -> None:
    """要求独立零Recipe回放精确复现历史搜索基线。"""
    expected = result.get("baseline")
    if not isinstance(expected, dict) or not isinstance(expected.get("metrics"), dict):
        raise ValueError(f"{layout} result.json 缺少 baseline")
    actual_metrics = reset_info["initial_raw_metrics"]
    for name in ("l2", "epe", "pvb"):
        if not _same_number(actual_metrics[name], expected["metrics"][name]):
            raise RuntimeError(
                f"{layout} 零Recipe回放基线不一致：{name} "
                f"历史={expected['metrics'][name]}，实际={actual_metrics[name]}"
            )
    if not _same_number(reset_info["initial_raw_weighted_loss"], expected["j"]):
        raise RuntimeError(f"{layout} 零Recipe回放 J 与历史基线不一致")


def _baseline_printed(
    source: dict,
    layout: str,
    result: dict,
    output_dir: Path,
) -> Tuple[np.ndarray, dict]:
    """优先读取搜索保存的基线图；旧运行缺失时进行一次独立全零回放。"""
    coordinate = source["coordinate_dir"]
    existing = coordinate / "baseline-printed.png"
    artifacts = result.get("baseline_artifacts")
    if isinstance(artifacts, dict):
        recorded_paths = {
            "mask": coordinate / str(artifacts.get("mask_path", "")),
            "printed": coordinate / str(artifacts.get("printed_path", "")),
        }
        for name, path in recorded_paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{layout} 缺少 result.json 声明的 baseline {name} 工件：{path}")
            if artifacts.get(f"{name}_sha256") != _sha256(path):
                raise RuntimeError(f"{layout} baseline {name} 工件哈希与 result.json 不一致")
    if existing.is_file():
        digest = _sha256(existing)
        return _load_binary(existing), {
            "source": (
                "search_artifact_hash_verified"
                if isinstance(artifacts, dict)
                else "unrecorded_search_artifact"
            ),
            "path": str(existing),
            "sha256": digest,
            "extra_solver_calls": 0,
        }
    require_contract = layout in source["config"].get("golden", {}).get(
        "layout_contracts", {}
    )
    episode, verified_fields = _build_episode(
        source["config"],
        layout,
        shuffle_points=False,
        require_layout_contract=require_contract,
    )
    _, reset_info = episode.reset(point_order=sorted(episode.point_ids))
    _validate_baseline_replay(reset_info, result, layout)
    baseline = episode.baseline_result
    replay_dir = Path(output_dir) / "baseline-replays" / layout
    replay_dir.mkdir(parents=True, exist_ok=False)
    mask_path = replay_dir / "baseline-mask.png"
    printed_path = replay_dir / "baseline-printed.png"
    mask_sha256 = _write_binary_png(mask_path, baseline.mask_image)
    printed_sha256 = _write_binary_png(printed_path, baseline.printed_nominal)
    return np.asarray(baseline.printed_nominal) >= 0.5, {
        "source": "independent_zero_recipe_replay",
        "path": str(printed_path),
        "sha256": printed_sha256,
        "mask_path": str(mask_path),
        "mask_sha256": mask_sha256,
        "extra_solver_calls": 1,
        "golden_verified_fields": list(verified_fields),
    }


def _metric_state(pvb: float, measurements: Iterable[object]) -> dict:
    """组合搜索PVB和同一点集上的1 nm EPE N/D。"""
    summary = _summarize(measurements)
    return {
        "pvb": float(pvb),
        "epe_n": int(summary.epe_n),
        "epe_d_nm": float(summary.epe_d_nm),
        "mean_violation_distance_nm": float(summary.mean_violation_distance_nm),
        "max_distance_nm": float(summary.max_distance_nm),
    }


def _relative_reduction(baseline: float, final: float) -> float | None:
    """计算相对下降率；零基线不伪造百分比。"""
    if baseline == 0:
        return None
    return float((baseline - final) / baseline * 100.0)


def _layout_row(
    layout: str,
    source: dict,
    output_dir: Path,
    tolerance_nm: float,
) -> dict:
    """读取一张图、重建v2点集、核对身份并计算基线与最终1 nm指标。"""
    coordinate = source["coordinate_dir"]
    required = {
        name: coordinate / name
        for name in ("target.png", "final-mask.png", "final-printed.png", "result.json")
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{layout} 缺少最终工件：{missing}")
    result = json.loads(required["result.json"].read_text(encoding="utf-8"))
    if result.get("final_replay_equal") is not True:
        raise RuntimeError(f"{layout} 最终独立回放不一致")
    source_artifacts = {
        name.removesuffix(".png").replace("-", "_"): {
            "path": str(path),
            "sha256": _sha256(path),
        }
        for name, path in required.items()
        if name != "result.json"
    }
    recorded_artifacts = result.get("final_artifacts")
    if isinstance(recorded_artifacts, dict):
        for name, artifact in source_artifacts.items():
            recorded = recorded_artifacts.get(name)
            if not isinstance(recorded, dict) or recorded.get("sha256") != artifact["sha256"]:
                raise RuntimeError(f"{layout} {name} 工件哈希与 result.json 不一致")
    target = _load_binary(required["target.png"])
    final_printed = _load_binary(required["final-printed.png"])
    rebuilt_target, points, layout_sha256 = rebuild(source["config"], layout)
    if not np.array_equal(target, np.asarray(rebuilt_target) >= 0.5):
        raise RuntimeError(f"{layout} target.png 与锁定GLP重建结果不一致")
    recipe_sha256 = _validate_rebuilt_points(points, result)
    baseline_printed, baseline_artifact = _baseline_printed(
        source, layout, result, output_dir
    )
    if baseline_printed.shape != target.shape or final_printed.shape != target.shape:
        raise RuntimeError(f"{layout} target/baseline/final 图像尺寸不一致")
    scale = float(source["config"]["recipe_v2"]["nm_per_coordinate"])
    baseline_measurements = measure_v2_points(
        target, baseline_printed, points, scale, tolerance_nm
    )
    final_measurements = measure_v2_points(
        target, final_printed, points, scale, tolerance_nm
    )
    baseline_state = _metric_state(result["baseline"]["metrics"]["pvb"], baseline_measurements)
    final_state = _metric_state(result["best"]["metrics"]["pvb"], final_measurements)
    reductions = {
        name: {
            "absolute": float(baseline_state[name] - final_state[name]),
            "relative_percent": _relative_reduction(
                float(baseline_state[name]), float(final_state[name])
            ),
        }
        for name in ("pvb", "epe_n", "epe_d_nm")
    }
    return {
        "layout": layout,
        "source_run": source["run_dir"].name,
        "source_result_path": str(required["result.json"]),
        "source_result_sha256": _sha256(required["result.json"]),
        "source_artifacts": source_artifacts,
        "config_sha256": _sha256(source["config_path"]),
        "layout_glp_sha256": layout_sha256,
        "recipe_sha256": recipe_sha256,
        "sample_count": len(points),
        "tolerance_nm": float(tolerance_nm),
        "baseline": baseline_state,
        "final": final_state,
        "reduction": reductions,
        "baseline_artifact": baseline_artifact,
        "existing_golden_15nm": {
            "baseline": float(result["baseline"]["metrics"]["epe"]),
            "final": float(result["best"]["metrics"]["epe"]),
        },
        "final_replay_equal": True,
    }


def aggregate_rows(rows: Sequence[Mapping[str, object]]) -> dict:
    """生成十图等权宏平均与按全部采样点汇总的微观统计。"""
    if [row["layout"] for row in rows] != list(EXPECTED_LAYOUTS):
        raise ValueError("聚合输入必须按 M1_test1–10 顺序且恰好十行")
    count = len(rows)
    macro = {"layout_count": count, "baseline_mean": {}, "final_mean": {}, "reduction": {}}
    for name in ("pvb", "epe_n", "epe_d_nm"):
        baseline = float(sum(float(row["baseline"][name]) for row in rows) / count)
        final = float(sum(float(row["final"][name]) for row in rows) / count)
        macro["baseline_mean"][name] = baseline
        macro["final_mean"][name] = final
        macro["reduction"][name] = {
            "absolute": baseline - final,
            "relative_percent": _relative_reduction(baseline, final),
        }
    samples = int(sum(int(row["sample_count"]) for row in rows))
    baseline_n = int(sum(int(row["baseline"]["epe_n"]) for row in rows))
    final_n = int(sum(int(row["final"]["epe_n"]) for row in rows))
    baseline_d = float(sum(float(row["baseline"]["epe_d_nm"]) for row in rows))
    final_d = float(sum(float(row["final"]["epe_d_nm"]) for row in rows))
    micro = {
        "sample_count": samples,
        "baseline_epe_n_total": baseline_n,
        "final_epe_n_total": final_n,
        "baseline_epe_d_nm_total": baseline_d,
        "final_epe_d_nm_total": final_d,
        "baseline_violation_rate": baseline_n / samples if samples else 0.0,
        "final_violation_rate": final_n / samples if samples else 0.0,
        "baseline_epe_d_nm_per_sample": baseline_d / samples if samples else 0.0,
        "final_epe_d_nm_per_sample": final_d / samples if samples else 0.0,
    }
    return {"macro_average": macro, "micro_total": micro}


def build_report(
    run_dirs: Sequence[Path],
    output_dir: Path,
    seed: int = 0,
    tolerance_nm: float = 1.0,
) -> dict:
    """完成十图覆盖与协议检查后计算报告；任一图失败则不输出部分总表。"""
    if seed != 0:
        raise ValueError("当前十图启发式汇总只接受 seed=0")
    if tolerance_nm < 0:
        raise ValueError("tolerance_nm 不能为负")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("十图汇总输出目录必须为空，避免覆盖已有审计工件")
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = _discover_layouts(run_dirs, seed)
    rows = [
        _layout_row(layout, sources[layout], output_dir, tolerance_nm)
        for layout in EXPECTED_LAYOUTS
    ]
    aggregate = aggregate_rows(rows)
    return {
        "schema_version": "1.0",
        "summary_version": SUMMARY_VERSION,
        "sampling_version": SAMPLING_VERSION,
        "metric_version": METRIC_VERSION,
        "search_runs": [str(Path(path)) for path in run_dirs],
        "seed": seed,
        "tolerance_nm": float(tolerance_nm),
        "distance_definition": "nearest_euclidean_printed_foreground_boundary",
        "epe_d_definition": "sum_full_distance_for_points_strictly_over_tolerance",
        "search_guardrail_note": (
            "1 nm EPE N/D is post-processing only; the heuristic accepted candidates using "
            "L2, 15 nm Golden EPE, PVB, and strict J improvement."
        ),
        "generalization_claim_allowed": False,
        "extra_baseline_solver_calls": sum(
            int(row["baseline_artifact"]["extra_solver_calls"]) for row in rows
        ),
        "layouts": rows,
        **aggregate,
    }


def write_report(report: Mapping[str, object], output_dir: Path) -> Tuple[Path, Path]:
    """写出完整JSON与便于查看的逐图/十图平均及总量CSV。"""
    output_dir = Path(output_dir)
    json_path = output_dir / "ten-layout-epe-summary.json"
    csv_path = output_dir / "ten-layout-epe-summary.csv"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "layout", "sample_count",
        "baseline_pvb", "final_pvb", "pvb_reduction_percent",
        "baseline_epe_n", "final_epe_n", "epe_n_reduction_percent",
        "baseline_epe_d_nm", "final_epe_d_nm", "epe_d_reduction_percent",
        "baseline_violation_rate", "final_violation_rate",
        "baseline_source", "final_replay_equal",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["layouts"]:
            writer.writerow({
                "layout": row["layout"],
                "sample_count": row["sample_count"],
                "baseline_pvb": row["baseline"]["pvb"],
                "final_pvb": row["final"]["pvb"],
                "pvb_reduction_percent": row["reduction"]["pvb"]["relative_percent"],
                "baseline_epe_n": row["baseline"]["epe_n"],
                "final_epe_n": row["final"]["epe_n"],
                "epe_n_reduction_percent": row["reduction"]["epe_n"]["relative_percent"],
                "baseline_epe_d_nm": row["baseline"]["epe_d_nm"],
                "final_epe_d_nm": row["final"]["epe_d_nm"],
                "epe_d_reduction_percent": row["reduction"]["epe_d_nm"]["relative_percent"],
                "baseline_violation_rate": row["baseline"]["epe_n"] / row["sample_count"],
                "final_violation_rate": row["final"]["epe_n"] / row["sample_count"],
                "baseline_source": row["baseline_artifact"]["source"],
                "final_replay_equal": row["final_replay_equal"],
            })
        macro = report["macro_average"]
        writer.writerow({
            "layout": "ALL_MEAN",
            "sample_count": report["micro_total"]["sample_count"] / 10,
            "baseline_pvb": macro["baseline_mean"]["pvb"],
            "final_pvb": macro["final_mean"]["pvb"],
            "pvb_reduction_percent": macro["reduction"]["pvb"]["relative_percent"],
            "baseline_epe_n": macro["baseline_mean"]["epe_n"],
            "final_epe_n": macro["final_mean"]["epe_n"],
            "epe_n_reduction_percent": macro["reduction"]["epe_n"]["relative_percent"],
            "baseline_epe_d_nm": macro["baseline_mean"]["epe_d_nm"],
            "final_epe_d_nm": macro["final_mean"]["epe_d_nm"],
            "epe_d_reduction_percent": macro["reduction"]["epe_d_nm"]["relative_percent"],
            "baseline_violation_rate": "",
            "final_violation_rate": "",
            "baseline_source": "mixed_and_audited",
            "final_replay_equal": True,
        })
        micro = report["micro_total"]
        writer.writerow({
            "layout": "ALL_TOTAL",
            "sample_count": micro["sample_count"],
            "baseline_pvb": "",
            "final_pvb": "",
            "pvb_reduction_percent": "",
            "baseline_epe_n": micro["baseline_epe_n_total"],
            "final_epe_n": micro["final_epe_n_total"],
            "epe_n_reduction_percent": _relative_reduction(
                micro["baseline_epe_n_total"], micro["final_epe_n_total"]
            ),
            "baseline_epe_d_nm": micro["baseline_epe_d_nm_total"],
            "final_epe_d_nm": micro["final_epe_d_nm_total"],
            "epe_d_reduction_percent": _relative_reduction(
                micro["baseline_epe_d_nm_total"], micro["final_epe_d_nm_total"]
            ),
            "baseline_violation_rate": micro["baseline_violation_rate"],
            "final_violation_rate": micro["final_violation_rate"],
            "baseline_source": "mixed_and_audited",
            "final_replay_equal": True,
        })
    return json_path, csv_path


def main(argv: Sequence[str] | None = None) -> int:
    """提供云端十图合并、缺失基线回放与统一指标计算入口。"""
    parser = argparse.ArgumentParser(
        prog="python -m opc_agent.recipe_v2_ten_layout_summary"
    )
    parser.add_argument("--search-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance-nm", type=float, default=1.0)
    args = parser.parse_args(argv)
    report = build_report(
        args.search_runs,
        args.output_dir,
        seed=args.seed,
        tolerance_nm=args.tolerance_nm,
    )
    for path in write_report(report, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
