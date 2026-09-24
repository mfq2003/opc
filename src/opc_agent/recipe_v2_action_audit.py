"""本模块审计 v2 坐标搜索候选动作是否具有可学习的价值差异。

输入是既有 ``v2-search`` 运行目录，模块只读取总表、冻结配置、逐版图结果和
``candidates.jsonl``，不调用 OpenILT、Solver 或训练模型。它按连续的四候选记录重建
每个 EPE 点在当时 incumbent 状态下的动作组，分别统计 mask 可区分性、指标可区分性、
候选 J 间隔、有效改善和改善量归属。输出 JSON 与 CSV 仅属于 ``diagnostic_only``，
用于决定是否值得进入候选动作价值排序，不能替代 Golden 回放或正式验收。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml


METRIC_KEYS = ("l2", "epe", "pvb")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"候选日志第 {line_number} 行不是对象：{path}")
        rows.append(row)
    return rows


def _consecutive_groups(rows: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for row in rows:
        point_id = row.get("point_id")
        if not isinstance(point_id, str) or not point_id:
            raise ValueError("候选记录缺少非空 point_id")
        if not groups or groups[-1][0]["point_id"] != point_id:
            groups.append([])
        groups[-1].append(row)
    return groups


def _metric_tuple(row: dict[str, Any]) -> tuple[float, float, float]:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("候选记录缺少 metrics")
    return tuple(float(metrics[key]) for key in METRIC_KEYS)


def _audit_group(
    layout: str,
    group_index: int,
    records: list[dict[str, Any]],
    incumbent_j_before: float,
    action_count: int,
    action_offsets_nm: list[float],
) -> dict[str, Any]:
    if len(records) != action_count - 1:
        raise ValueError(
            f"{layout} 第 {group_index} 组候选数为 {len(records)}，"
            f"预期 {action_count - 1}"
        )
    point_id = records[0]["point_id"]
    actions = [int(row["action"]) for row in records]
    if len(set(actions)) != len(actions) or any(action not in range(action_count) for action in actions):
        raise ValueError(f"{layout}/{point_id} 候选动作不唯一或越界")
    missing_actions = sorted(set(range(action_count)) - set(actions))
    if len(missing_actions) != 1:
        raise ValueError(f"{layout}/{point_id} 无法唯一确定 incumbent 动作")
    calls = [int(row["candidate_call"]) for row in records]
    if calls != list(range(calls[0], calls[0] + len(calls))):
        raise ValueError(f"{layout}/{point_id} candidate_call 不连续")
    after_values = {float(row["incumbent_j_after_group"]) for row in records}
    if len(after_values) != 1:
        raise ValueError(f"{layout}/{point_id} 组内 incumbent_j_after_group 不一致")
    incumbent_j_after = after_values.pop()
    candidate_js = [float(row["j"]) for row in records]
    feasible_records = [row for row in records if bool(row["feasible"])]
    feasible_js = [float(row["j"]) for row in feasible_records]
    expected_after = min([incumbent_j_before, *feasible_js])
    if incumbent_j_after != expected_after:
        raise ValueError(
            f"{layout}/{point_id} incumbent 轨迹不一致："
            f"记录 {incumbent_j_after}，按候选应为 {expected_after}"
        )
    accepted_records = [row for row in records if bool(row["accepted"])]
    if len(accepted_records) != int(incumbent_j_after < incumbent_j_before):
        raise ValueError(f"{layout}/{point_id} accepted 标记与 J 轨迹不一致")

    unique_candidate_js = sorted(set(candidate_js))
    sorted_candidate_js = sorted(candidate_js)
    best_j = sorted_candidate_js[0]
    best_actions = sorted(
        int(row["action"]) for row in records if float(row["j"]) == best_j
    )
    unique_mask_count = len({str(row["mask_sha256"]) for row in records})
    unique_metric_count = len({_metric_tuple(row) for row in records})
    oracle_gain = incumbent_j_before - incumbent_j_after
    beneficial_records = [
        row for row in feasible_records if float(row["j"]) < incumbent_j_before
    ]
    accepted_action = int(accepted_records[0]["action"]) if accepted_records else None
    return {
        "layout_parent": layout,
        "group_index": group_index,
        "point_id": point_id,
        "candidate_call_start": calls[0],
        "candidate_call_end": calls[-1],
        "incumbent_action": missing_actions[0],
        "incumbent_offset_nm": float(action_offsets_nm[missing_actions[0]]),
        "candidate_actions": actions,
        "candidate_offsets_nm": [float(action_offsets_nm[action]) for action in actions],
        "incumbent_j_before": incumbent_j_before,
        "incumbent_j_after": incumbent_j_after,
        "oracle_gain": oracle_gain,
        "realized_improvement": oracle_gain,
        "feasible_candidate_count": len(feasible_records),
        "beneficial_candidate_count": len(beneficial_records),
        "accepted_action": accepted_action,
        "accepted_offset_nm": (
            float(action_offsets_nm[accepted_action]) if accepted_action is not None else None
        ),
        "unique_mask_count": unique_mask_count,
        "unique_metric_count": unique_metric_count,
        "unique_j_count": len(unique_candidate_js),
        "j_span": max(candidate_js) - min(candidate_js),
        "best_second_gap": sorted_candidate_js[1] - sorted_candidate_js[0],
        "best_candidate_j": best_j,
        "best_candidate_actions": best_actions,
        "best_candidate_tie_count": len(best_actions),
        "mask_distinguishable": unique_mask_count > 1,
        "metric_distinguishable": unique_metric_count > 1,
        "value_rankable": len(unique_candidate_js) > 1,
        "fully_invariant": unique_mask_count == 1 and unique_metric_count == 1,
        "has_observed_improvement": oracle_gain > 0,
    }


def _fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _summarize_layout(
    layout: str,
    rows: list[dict[str, Any]],
    result: dict[str, Any],
    action_offsets_nm: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups = _consecutive_groups(rows)
    incumbent_j = float(result["baseline"]["j"])
    audited_groups: list[dict[str, Any]] = []
    for index, records in enumerate(groups):
        audited = _audit_group(
            layout,
            index,
            records,
            incumbent_j,
            len(action_offsets_nm),
            action_offsets_nm,
        )
        audited_groups.append(audited)
        incumbent_j = float(audited["incumbent_j_after"])

    final_j = float(result["best"]["j"])
    if incumbent_j != final_j:
        raise ValueError(f"{layout} 候选轨迹终点与 result.json best.j 不一致")
    if len(rows) != int(result["candidate_calls"]):
        raise ValueError(f"{layout} 候选日志行数与 result.json candidate_calls 不一致")
    if len(groups) != len(result["point_order"]):
        raise ValueError(f"{layout} 动作组数与 point_order 不一致")

    total_gain = sum(float(group["oracle_gain"]) for group in audited_groups)
    rankable_gain = sum(
        float(group["oracle_gain"]) for group in audited_groups if group["value_rankable"]
    )
    invariant_gain = sum(
        float(group["oracle_gain"]) for group in audited_groups if group["fully_invariant"]
    )
    group_count = len(audited_groups)
    mask_distinguishable_count = sum(bool(group["mask_distinguishable"]) for group in audited_groups)
    metric_distinguishable_count = sum(bool(group["metric_distinguishable"]) for group in audited_groups)
    value_rankable_count = sum(bool(group["value_rankable"]) for group in audited_groups)
    fully_invariant_count = sum(bool(group["fully_invariant"]) for group in audited_groups)
    improving_count = sum(bool(group["has_observed_improvement"]) for group in audited_groups)
    unique_mask_histogram = Counter(int(group["unique_mask_count"]) for group in audited_groups)
    summary = {
        "layout_parent": layout,
        "group_count": group_count,
        "candidate_count": len(rows),
        "baseline_j": float(result["baseline"]["j"]),
        "final_j": final_j,
        "total_realized_improvement": total_gain,
        "result_j_improvement": float(result["baseline"]["j"]) - final_j,
        "mask_distinguishable_group_count": mask_distinguishable_count,
        "mask_distinguishable_group_fraction": _fraction(mask_distinguishable_count, group_count),
        "metric_distinguishable_group_count": metric_distinguishable_count,
        "metric_distinguishable_group_fraction": _fraction(metric_distinguishable_count, group_count),
        "value_rankable_group_count": value_rankable_count,
        "value_rankable_group_fraction": _fraction(value_rankable_count, group_count),
        "fully_invariant_group_count": fully_invariant_count,
        "fully_invariant_group_fraction": _fraction(fully_invariant_count, group_count),
        "improving_group_count": improving_count,
        "improving_group_fraction": _fraction(improving_count, group_count),
        "improvement_on_value_rankable_groups": rankable_gain,
        "improvement_on_fully_invariant_groups": invariant_gain,
        "value_rankable_improvement_share": _fraction(rankable_gain, total_gain),
        "unique_mask_count_histogram": {
            str(key): unique_mask_histogram[key] for key in sorted(unique_mask_histogram)
        },
    }
    if total_gain <= 0:
        summary["decision_signal"] = "no_observed_pointwise_gain"
    elif rankable_gain <= 0:
        summary["decision_signal"] = "proposal_or_skip_signal_only"
    else:
        summary["decision_signal"] = "candidate_value_ranking_signal_present"
    return summary, audited_groups


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def audit_search_run(source_run: Path, output_dir: Path) -> dict[str, Any]:
    """审计完整六图坐标搜索工件并写入零 Solver 成本的诊断报告。"""
    source_run = Path(source_run)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{output_dir}")
    summary_path = source_run / "recipe-v2-search.json"
    config_path = source_run / "config.snapshot.yaml"
    summary = _load_json(summary_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if summary.get("status") != "diagnostic_only" or summary.get("accepted") is not False:
        raise ValueError("动作审计只接受 diagnostic_only 且未 accepted 的搜索源")
    if summary.get("training_enabled") is not False:
        raise ValueError("动作审计拒绝混入训练运行")
    action_offsets_nm = [float(value) for value in config["recipe_v2"]["epe_normal_offsets_nm"]]
    if action_offsets_nm != [-20.0, -10.0, 0.0, 10.0, 20.0]:
        raise ValueError("动作审计要求冻结五动作 [-20,-10,0,10,20]nm")

    layout_summaries: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    arms = summary.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("搜索总表缺少 arms")
    seen_layouts: set[str] = set()
    for arm in arms:
        layout = str(arm["layout_parent"])
        if layout in seen_layouts or arm.get("method") != "coordinate" or int(arm.get("seed")) != 0:
            raise ValueError("动作审计要求每张版图恰好一个 coordinate/seed-0 臂")
        seen_layouts.add(layout)
        arm_root = source_run / layout / "seed-0" / "coordinate"
        result = _load_json(arm_root / "result.json")
        rows = _load_jsonl(arm_root / "candidates.jsonl")
        layout_summary, audited_groups = _summarize_layout(
            layout, rows, result, action_offsets_nm
        )
        layout_summaries.append(layout_summary)
        group_rows.extend(audited_groups)

    layout_summaries.sort(key=lambda row: row["layout_parent"])
    group_rows.sort(key=lambda row: (row["layout_parent"], row["group_index"]))
    total_groups = len(group_rows)
    total_gain = sum(float(row["total_realized_improvement"]) for row in layout_summaries)
    rankable_gain = sum(float(row["improvement_on_value_rankable_groups"]) for row in layout_summaries)
    overall = {
        "layout_count": len(layout_summaries),
        "group_count": total_groups,
        "candidate_count": sum(int(row["candidate_count"]) for row in layout_summaries),
        "mask_distinguishable_group_count": sum(
            int(row["mask_distinguishable_group_count"]) for row in layout_summaries
        ),
        "metric_distinguishable_group_count": sum(
            int(row["metric_distinguishable_group_count"]) for row in layout_summaries
        ),
        "value_rankable_group_count": sum(
            int(row["value_rankable_group_count"]) for row in layout_summaries
        ),
        "fully_invariant_group_count": sum(
            int(row["fully_invariant_group_count"]) for row in layout_summaries
        ),
        "improving_group_count": sum(int(row["improving_group_count"]) for row in layout_summaries),
        "total_realized_improvement": total_gain,
        "improvement_on_value_rankable_groups": rankable_gain,
        "value_rankable_improvement_share": _fraction(rankable_gain, total_gain),
    }
    for key in (
        "mask_distinguishable",
        "metric_distinguishable",
        "value_rankable",
        "fully_invariant",
        "improving",
    ):
        overall[f"{key}_group_fraction"] = _fraction(
            overall[f"{key}_group_count"], total_groups
        )
    if total_gain <= 0:
        overall["decision_signal"] = "stop_pointwise_route_candidate"
    elif rankable_gain <= 0:
        overall["decision_signal"] = "proposal_or_skip_only_candidate"
    else:
        overall["decision_signal"] = "proceed_to_offline_candidate_value_baseline"

    report = {
        "status": "diagnostic_only",
        "accepted": False,
        "audit_version": "recipe-v2-action-identifiability-v1",
        "source_run": str(source_run),
        "source_summary_sha256": _sha256(summary_path),
        "source_config_sha256": _sha256(config_path),
        "solver_calls": 0,
        "action_offsets_nm": action_offsets_nm,
        "overall": overall,
        "layouts": layout_summaries,
        "interpretation_boundary": (
            "该报告只描述既有坐标搜索轨迹上的单步候选可辨识性；"
            "不能证明离线排序模型能保持最终 Recipe 质量。"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "action-identifiability.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    layout_fields = [
        "layout_parent", "group_count", "candidate_count", "baseline_j", "final_j",
        "total_realized_improvement", "mask_distinguishable_group_count",
        "mask_distinguishable_group_fraction", "metric_distinguishable_group_count",
        "metric_distinguishable_group_fraction", "value_rankable_group_count",
        "value_rankable_group_fraction", "fully_invariant_group_count",
        "fully_invariant_group_fraction", "improving_group_count", "improving_group_fraction",
        "improvement_on_value_rankable_groups", "value_rankable_improvement_share",
        "decision_signal",
    ]
    group_fields = [
        "layout_parent", "group_index", "point_id", "candidate_call_start",
        "candidate_call_end", "incumbent_action", "incumbent_offset_nm",
        "incumbent_j_before", "incumbent_j_after", "oracle_gain",
        "feasible_candidate_count", "beneficial_candidate_count", "accepted_action",
        "accepted_offset_nm", "unique_mask_count", "unique_metric_count", "unique_j_count",
        "j_span", "best_second_gap", "best_candidate_j", "best_candidate_tie_count",
        "mask_distinguishable", "metric_distinguishable", "value_rankable",
        "fully_invariant", "has_observed_improvement",
    ]
    _write_csv(output_dir / "layout-summary.csv", layout_summaries, layout_fields)
    _write_csv(output_dir / "groups.csv", group_rows, group_fields)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="只读审计既有 v2 坐标搜索候选动作的可辨识性"
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_search_run(args.source_run, args.output)
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
