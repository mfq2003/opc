"""本模块实现固定 FRAG 的 EPE 黑盒搜索诊断。

输入为冻结的 v2 配置和显式版图范围；默认只允许训练版图，只有单独授权的
``validation_test_diagnostic`` 范围才允许搜索 M1_test7–10。入口执行全零基线与离散坐标搜索，
随机函数分支仅保留历史回归兼容，不再安排随机实验。各方法共用
完整 terminal solver 与 Golden 评价。输出逐候选日志、完整动作、指标、独立回放、
全零/最终图像、输入样例和结果图片。搜索不训练 PPO；验证/测试版图搜索使用自身 solver
反馈，因而只能作为逐图启发式诊断，不能冒充未见版图泛化或正式验收。
"""
from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from .recipe_v2_small_train import _build_episode
from .recipe_v2_openilt import _validate_openilt
from .recipe_v2_visualization import save_v2_ppo_input_examples


def evaluate_recipe(episode, actions):
    """逐点提交完整动作并只在 terminal 调用 solver。"""
    order = sorted(episode.point_ids)
    episode.reset(point_order=order)
    for point in order:
        episode.step(int(actions[point]))
    evaluation = episode.final_golden_evaluation
    return {
        "actions": dict(actions),
        "offsets_nm": episode.final_recipe_offsets_nm,
        "metrics": evaluation.metrics.as_dict(),
        "j": float(evaluation.raw_weighted_loss),
        "recipe_sha256": episode.final_recipe_sha256,
        "mask_sha256": episode.final_result.mask_sha256,
    }


def _write_binary_png(path, array):
    """把冻结 solver 二值数组写为 PNG，并返回文件 SHA-256。"""
    pixels = np.clip(np.asarray(array) * 255, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise RuntimeError(f"搜索结果图片编码失败：{path}")
    payload = encoded.tobytes()
    Path(path).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _golden_identity(episode):
    """记录本次真实 evaluator 的逐版图身份；未预冻结字段也不能丢失。"""
    evaluator = episode.golden_evaluator
    return {
        "evaluator_version": str(evaluator.evaluator_version),
        "source_sha256": str(evaluator.evaluator_source_sha256),
        "target_sha256": str(evaluator.frozen_target_sha256),
        "sampling_state_sha256": str(evaluator.sampling_state_sha256),
        "coordinate_system_sha256": str(evaluator.coordinate_system_sha256),
        "evaluator_contract_sha256": str(evaluator.evaluator_contract_sha256),
        "epe_constraint_coordinate": int(evaluator.epe_constraint_coordinate),
        "nm_per_coordinate": float(evaluator.nm_per_coordinate),
    }


def _validate_search_scope(config):
    """冻结训练与验证/测试两种互斥范围，拒绝静默混合或只挑部分评估图。"""
    settings = config["search"]
    layouts = list(settings["layout_parents"])
    train = list(config["data"]["train_parents"])
    evaluation = list(config["data"]["validation_parents"] + config["data"]["test_parents"])
    scope = settings.get("scope", "train_only")
    if scope == "train_only":
        if layouts != train:
            raise ValueError("train_only 搜索必须恰好覆盖全部训练版图")
        return scope, True
    if scope == "validation_test_diagnostic":
        if layouts != evaluation:
            raise ValueError("validation_test_diagnostic 必须恰好覆盖 M1_test7–10")
        return scope, False
    raise ValueError("search.scope 只能是 train_only 或 validation_test_diagnostic")


def search_recipe(
    episode,
    method,
    seed,
    budget,
    root,
    resume_from=None,
    baseline_calls=1,
    save_baseline_artifacts=False,
):
    """运行一次等预算搜索；坐标候选从同一 incumbent 出发，平局保持旧值。"""
    if method not in ("random", "coordinate") or budget <= 0:
        raise ValueError("非法搜索方法或预算")
    episode.require_plain_ppo_compatible()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    points = sorted(episode.point_ids)
    zero = episode.action_offsets_nm.index(0.0)
    _, reset = episode.reset(point_order=points)
    baseline = {"metrics": reset["initial_raw_metrics"],
                "j": float(reset["initial_raw_weighted_loss"])}
    actions = {p: zero for p in points}
    baseline_artifacts = None
    if save_baseline_artifacts:
        baseline_result = episode.baseline_result
        baseline_artifacts = {
            "mask_path": "baseline-mask.png",
            "mask_sha256": _write_binary_png(root / "baseline-mask.png", baseline_result.mask_image),
            "printed_path": "baseline-printed.png",
            "printed_sha256": _write_binary_png(
                root / "baseline-printed.png", baseline_result.printed_nominal
            ),
        }
    # 基线已由 reset 求解；直接保存它的指标，避免额外零 Recipe 求解。
    best = {**baseline, "actions": dict(actions)}
    calls = 0
    accepted = 0
    curve = [(0, best["j"])]
    order = list(rng.permutation(points))
    started = time.monotonic()
    saved = []
    if resume_from is not None and (Path(resume_from) / "candidates.jsonl").exists():
        lines = (Path(resume_from) / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            try:
                saved.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise ValueError("续跑日志中间存在损坏")
        width = len(episode.action_offsets_nm) - 1
        saved = saved[:len(saved) // width * width]
        if len(saved) > budget or method != "coordinate":
            raise ValueError("续跑只支持预算内的坐标搜索")
    reused = len(saved)
    last_report = started
    with (root / "candidates.jsonl").open("w", encoding="utf-8") as log:
        groups = [[None]] * budget if method == "random" else [[p] for p in order]
        for group in groups:
            point = group[0]
            alternatives = [None] if point is None else [
                a for a in range(len(episode.action_offsets_nm)) if a != actions[point]
            ]
            if calls + len(alternatives) > budget:
                break
            winner = best
            records = []
            for action in alternatives:
                candidate = dict(actions)
                if point is None:
                    candidate = {p: int(rng.integers(len(episode.action_offsets_nm))) for p in points}
                else:
                    candidate[point] = action
                if calls < reused:
                    previous = saved[calls]
                    if (previous["actions"] != candidate or previous["point_id"] != point
                            or previous["action"] != action or previous["candidate_call"] != calls + 1):
                        raise ValueError("续跑日志与冻结点顺序/动作不一致")
                    result = {key: previous[key] for key in
                              ("actions", "offsets_nm", "metrics", "j", "recipe_sha256", "mask_sha256")}
                else:
                    result = evaluate_recipe(episode, candidate)
                calls += 1
                feasible = all(result["metrics"][k] <= baseline["metrics"][k]
                               for k in ("l2", "epe", "pvb"))
                reason = "guardrail_failed" if not feasible else "no_strict_improvement"
                if feasible and result["j"] < winner["j"]:
                    winner = result
                    reason = "eligible_improvement"
                records.append({**result, "candidate_call": calls, "point_id": point,
                                "action": action, "feasible": feasible, "reason": reason})
            changed = winner is not best
            if changed:
                best = winner
                actions = dict(best["actions"])
                accepted += 1
            curve.append((calls, best["j"]))
            for record in records:
                record["accepted"] = changed and record["actions"] == actions
                record["incumbent_j_after_group"] = best["j"]
                log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            now = time.monotonic()
            if calls > reused and (now - last_report >= 30 or calls == budget):
                seconds_per_call = (now - started) / (calls - reused)
                print(f"{root}: {calls}/{budget}, reused={reused}, J={best['j']:.1f}, "
                      f"arm ETA={(budget-calls)*seconds_per_call/60:.1f} min", flush=True)
                last_report = now
    # 独立完整求解，包括无改善时的零 Recipe；与选中候选核对。
    offsets, replay, golden, recipe_hash = episode.replay_complete_action_map(actions)
    replay_equal = golden.metrics.as_dict() == best["metrics"] and golden.raw_weighted_loss == best["j"]
    if "mask_sha256" in best:
        replay_equal = replay_equal and replay.mask_sha256 == best["mask_sha256"] and recipe_hash == best["recipe_sha256"]
    canvas = np.full((480, 900, 3), 255, dtype=np.uint8)
    low = min(value for _, value in curve)
    high = max(value for _, value in curve)
    coords = [(60 + int(780 * count / max(budget, 1)),
               410 - int(330 * (value - low) / max(high - low, 1)))
              for count, value in curve]
    cv2.line(canvas, (60, 70), (60, 410), (0, 0, 0), 1)
    cv2.line(canvas, (60, 410), (850, 410), (0, 0, 0), 1)
    for left, right in zip(coords, coords[1:]):
        cv2.line(canvas, left, (right[0], left[1]), (180, 70, 20), 2)
        cv2.line(canvas, (right[0], left[1]), right, (180, 70, 20), 2)
    cv2.putText(canvas, f"{method} seed={seed}: best feasible J vs candidate calls", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
    cv2.putText(canvas, f"J range [{low:.1f}, {high:.1f}]   calls [0, {budget}]", (60, 455), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
    (root / "search-curve.png").write_bytes(cv2.imencode(".png", canvas)[1].tobytes())
    final_artifacts = {}
    for name, array in (("target", episode.solver.target_image),
                        ("final-mask", replay.mask_image), ("final-printed", replay.printed_nominal)):
        file_name = name + ".png"
        final_artifacts[name.replace("-", "_")] = {
            "path": file_name,
            "sha256": _write_binary_png(root / file_name, array),
        }
    result = {"method": method, "seed": seed, "baseline": baseline,
              "best": {**best, "offsets_nm": offsets, "recipe_sha256": recipe_hash},
              "candidate_budget": budget, "candidate_calls": calls,
              "accepted_group_count": accepted, "point_order": order,
              "best_j_curve": curve,
              "action_counts": {str(a): sum(v == a for v in actions.values())
                                for a in range(len(episode.action_offsets_nm))},
              "stop_reason": "budget_exhausted" if calls == budget else "single_sweep_complete_or_insufficient_budget",
              "final_replay_equal": bool(replay_equal),
              "reused_candidate_calls": reused,
              "solver_call_counts": {"baseline": baseline_calls, "candidate": calls - reused,
                                     "final_replay": 1, "total": calls - reused + baseline_calls + 1},
              "baseline_artifacts": baseline_artifacts,
              "final_artifacts": final_artifacts,
              "elapsed_seconds": time.monotonic() - started}
    (root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def search_budget(episode):
    """按实际点数和非当前动作数计算一次完整扫描的候选预算。"""
    if episode.episode_horizon <= 0 or len(episode.action_offsets_nm) <= 1:
        raise ValueError("搜索需要非空点集及至少两个动作")
    return episode.episode_horizon * (len(episode.action_offsets_nm) - 1)


def run_v2_search(config, artifact_root):
    """在显式冻结范围内以 seed=0 执行坐标搜索；每图一次基线和独立回放。"""
    settings = config["search"]
    layouts = settings["layout_parents"]
    if settings["enabled"] is not True or config["training"]["enabled"] is not False:
        raise ValueError("搜索必须启用且通用训练禁用")
    scope, require_layout_contract = _validate_search_scope(config)
    if settings["seeds"] != [0]:
        raise ValueError("搜索固定使用单种子 seed=0")
    recipe = config["recipe_v2"]
    if recipe["fragment_parameters_nm"] != {"corner": 16, "uniform": 32} or recipe["epe_normal_offsets_nm"] != [-20, -10, 0, 10, 20]:
        raise ValueError("搜索必须保持冻结 FRAG 和五动作")
    if settings["budget_policy"] != "one_full_coordinate_sweep" or settings["guardrail"] != "all_metrics_le_zero_baseline":
        raise ValueError("搜索必须采用一轮完整扫描预算及零偏移单项护栏")
    arms = []
    root = Path(artifact_root)
    source = Path(settings["resume_from"]) if settings.get("resume_from") else None
    if source is not None:
        old_config = yaml.safe_load((source / "config.snapshot.yaml").read_text(encoding="utf-8"))
        if {k: v for k, v in old_config.items() if k != "search"} != {k: v for k, v in config.items() if k != "search"}:
            raise ValueError("续跑配置与旧快照不一致（search 设置之外必须完全相同）")
    for layout in layouts:
        episode, fields = _build_episode(
            config,
            layout,
            shuffle_points=False,
            require_layout_contract=require_layout_contract,
        )
        _, baseline_info = episode.reset()
        if source is not None:
            reference = source / layout / "seed-0" / "coordinate" / "result.json"
            if reference.exists():
                previous_baseline = json.loads(reference.read_text(encoding="utf-8"))["baseline"]
                if (previous_baseline["metrics"] != baseline_info["initial_raw_metrics"]
                        or previous_baseline["j"] != baseline_info["initial_raw_weighted_loss"]):
                    raise ValueError("续跑基线与历史基线不一致，拒绝混用候选")
        save_v2_ppo_input_examples(episode, root / "ppo-input-examples" / layout, 4)
        for seed in settings["seeds"]:
            for method in ("coordinate",):
                budget = search_budget(episode)
                old_arm = source / layout / f"seed-{seed}" / method if source else None
                result = search_recipe(episode, method, seed, budget,
                                       root / layout / f"seed-{seed}" / method,
                                       resume_from=old_arm, baseline_calls=int(seed == 0),
                                       save_baseline_artifacts=True)
                result["point_count"] = episode.episode_horizon
                arms.append({
                    "layout_parent": layout,
                    "golden_verified_fields": fields,
                    "golden_identity": _golden_identity(episode),
                    **result,
                })
                print(f"{layout} seed={seed} {method}: J={result['best']['j']} calls={result['candidate_calls']}", flush=True)
                if not result["final_replay_equal"]:
                    raise RuntimeError("搜索最终独立回放不一致；已保留该臂工件")
    revision = _validate_openilt(Path(config["data"]["openilt_dir"]), config["openilt"]["commit"])
    result = {"status": "diagnostic_only", "search_version": "coordinate-only-resumable-v4-evaluation-scope",
              "search_scope": scope,
              "uses_layout_specific_solver_feedback": True,
              "generalization_claim_allowed": False,
              "resume_from": str(source) if source else None,
              "budget_policy": settings["budget_policy"],
              "arms": arms, "accepted": False, "training_enabled": False,
              "post_run_openilt_revision": revision, "post_run_openilt_tracked_diff_clean": True,
              "solver_calls": sum(a["solver_call_counts"]["total"] for a in arms)}
    (root / "recipe-v2-search.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
