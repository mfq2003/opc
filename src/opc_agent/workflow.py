"""本模块把已实现的数据、GPU Oracle、双树 Recipe 与闭环评估接入统一 CLI 阶段。

输入为已解析 YAML、唯一 runs/<run_id> 目录和可选冒烟标志；输出为阶段产物及 stage-result.json。
关键依赖按阶段延迟导入，避免仅运行 CPU 命令时强制加载 CUDA/Gymnasium/scikit-learn；输入文件缺失会
显式失败，不会下载数据、调用 Qwen、修改 OpenILT 或生成伪实验结果。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _required_path(config: Dict[str, Any], name: str) -> Path:
    """读取 workflow 路径并在进入昂贵阶段前检查文件存在。"""
    workflow = config.get("workflow", {})
    raw = workflow.get(name)
    if not raw:
        raise ValueError(f"配置缺少 workflow.{name}")
    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(f"前置产物不存在：{path}")
    return path


def train_oracle_stage(
    config: Dict[str, Any],
    root: Path,
    smoke: bool = False,
    candidate_index: Path | None = None,
) -> None:
    """在 CUDA 上训练 PPO；支持单 clip 冒烟或按索引逐 clip、逐种子运行。"""
    environment = str(config.get("oracle", {}).get("environment", "legacy-candidate-point-v1"))
    if environment == "simpleopc-multistep-v3":
        if candidate_index is not None:
            raise ValueError("SimpleOPC 多步 PPO 直接读取 GLP，不接受旧 candidate-index")
        _train_simpleopc_stage(config, root, smoke=smoke)
        return
    if environment != "legacy-candidate-point-v1":
        raise ValueError(f"未知 oracle.environment：{environment}")
    from .oracle_runner import (
        CandidatePointEnv, OpenILTCandidateEvaluator, complete_metric_cache,
        load_candidate_point_set, train_ppo,
    )

    backend = config["openilt"]
    oracle = config["oracle"]
    seeds = [int(oracle["seeds"][0])] if smoke else [int(value) for value in oracle["seeds"]]
    timesteps = int(oracle["smoke_timesteps"] if smoke else oracle["total_timesteps"])
    index_path = Path(candidate_index) if candidate_index is not None else None
    if index_path is None and not smoke:
        configured = config.get("workflow", {}).get("oracle_candidate_index")
        if configured:
            index_path = Path(configured)

    if index_path is not None:
        from .oracle_batch_labels import load_candidate_index

        index = load_candidate_index(index_path)
        jobs = [(entry.clip_id, Path(entry.dataset_path)) for entry in index.entries]
        index_hash = index.index_sha256
    else:
        dataset_path = _required_path(config, "oracle_candidate_dataset")
        jobs = [("smoke", dataset_path)]
        index_hash = None

    clip_results = []
    for clip_id, dataset_path in jobs:
        dataset = load_candidate_point_set(dataset_path)
        clip_root = root if index_hash is None else root / "clips" / clip_id
        cache_path = clip_root / "oracle-metrics.cache.json"
        outputs = []
        for seed in seeds:
            evaluator = OpenILTCandidateEvaluator(
                dataset=dataset,
                openilt_dir=Path(config["data"]["openilt_dir"]),
                expected_commit=backend["commit"],
                lithography_config=Path(oracle["lithography_config"]),
                cache_path=cache_path,
                simulator=oracle["simulator"],
                scale=int(oracle.get("openilt_scale", 1)),
            )
            env = CandidatePointEnv(dataset, evaluator, dict(oracle["reward_weights"]))
            output = train_ppo(
                env=env,
                output_path=root / "models" / (
                    f"ppo-oracle-seed-{seed}" if index_hash is None else f"{clip_id}-seed-{seed}"
                ),
                total_timesteps=timesteps,
                seed=seed,
                learning_rate=float(oracle["learning_rate"]),
            )
            outputs.append(str(output))
            if seed == seeds[0]:
                complete_metric_cache(dataset, evaluator)
        clip_results.append({
            "clip_id": clip_id,
            "candidate_dataset": str(dataset_path),
            "candidate_source_sha256": dataset.source_sha256,
            "cache_path": str(cache_path),
            "models": outputs,
        })
    result = {
        "mode": "smoke" if smoke else "full",
        "candidate_index": str(index_path) if index_path is not None else None,
        "candidate_index_sha256": index_hash,
        "seeds": seeds,
        "timesteps_per_seed": timesteps,
        "clips": clip_results,
    }
    if index_hash is None:
        result.update({
            "candidate_dataset": clip_results[0]["candidate_dataset"],
            "candidate_source_sha256": clip_results[0]["candidate_source_sha256"],
            "models": clip_results[0]["models"],
        })
    (root / "stage-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def _train_simpleopc_stage(config: Dict[str, Any], root: Path, smoke: bool) -> None:
    """逐父版图、逐种子训练多步 SimpleOPC PPO，并把所有产物写入本项目 runs。"""
    from .simpleopc import OpenILTSimpleOPCBackend, SimpleOPCMultiStepEnv
    from .simpleopc_runner import run_simpleopc_heuristic, train_simpleopc_ppo

    data = config["data"]
    backend_config = config["openilt"]
    oracle = config["oracle"]
    simpleopc = config.get("simpleopc")
    if not isinstance(simpleopc, dict):
        raise ValueError("simpleopc-multistep-v3 需要 simpleopc 配置段")
    split_parents = {
        "train": list(data["train_parents"]),
        "validation": list(data["validation_parents"]),
        "test": list(data["test_parents"]),
    }
    jobs = [
        (str(parent), split)
        for split in ("train", "validation", "test")
        for parent in split_parents[split]
    ]
    if smoke:
        jobs = jobs[:1]
    seeds = [int(oracle["seeds"][0])] if smoke else [int(value) for value in oracle["seeds"]]
    timesteps = int(oracle["smoke_timesteps"] if smoke else oracle["total_timesteps"])
    from .metrics import DISPLACEMENT_CLASSES_NM, SIMPLEOPC_LOSS_VERSION

    displacement_limit = float(oracle["displacement_nm_limit"])
    if displacement_limit != 40.0:
        raise ValueError("论文公开的 PPO 位移范围要求 oracle.displacement_nm_limit=40")
    configured_classes = tuple(int(value) for value in simpleopc["displacement_classes_nm"])
    if configured_classes != DISPLACEMENT_CLASSES_NM:
        raise ValueError(
            "simpleopc.displacement_classes_nm 必须为 "
            f"{list(DISPLACEMENT_CLASSES_NM)}"
        )
    step_sizes = [float(value) for value in simpleopc["step_sizes_nm"]]
    if step_sizes != [10.0, 10.0, 10.0, 10.0]:
        raise ValueError(
            "paper_repro 的等距九分类适配要求 step_sizes_nm=[10, 10, 10, 10]"
        )
    image_size = tuple(int(value) for value in simpleopc.get("image_size", [2048, 2048]))
    if len(image_size) != 2:
        raise ValueError("simpleopc.image_size 必须包含 width、height 两个整数")
    clip_results = []
    for clip_id, split in jobs:
        layout_path = Path(data["iccad13_dir"]) / f"{clip_id}.glp"
        clip_root = root / "clips" / clip_id
        clip_root.mkdir(parents=True, exist_ok=True)
        backend = OpenILTSimpleOPCBackend(
            openilt_dir=Path(data["openilt_dir"]),
            expected_commit=backend_config["commit"],
            layout_path=layout_path,
            lithography_config=Path(oracle["lithography_config"]),
            simulator=oracle["simulator"],
            image_size=(image_size[0], image_size[1]),
            openilt_scale=int(oracle.get("openilt_scale", 1)),
            nm_per_coordinate=float(simpleopc.get("nm_per_coordinate", 1.0)),
            len_corner_nm=float(simpleopc["len_corner_nm"]),
            len_uniform_nm=float(simpleopc["len_uniform_nm"]),
            epe_sample_distance_nm=float(simpleopc["epe_sample_distance_nm"]),
            threshold=float(simpleopc.get("threshold", 0.5)),
        )

        def make_env() -> SimpleOPCMultiStepEnv:
            """为相同只读后端建立全新 episode 状态。"""
            return SimpleOPCMultiStepEnv(
                backend=backend,
                reward_weights=dict(oracle["reward_weights"]),
                step_sizes_nm=step_sizes,
                displacement_limit_nm=displacement_limit,
                metric_epsilon=float(simpleopc.get("metric_epsilon", 1.0)),
            )

        heuristic = run_simpleopc_heuristic(make_env(), seed=0)
        heuristic_path = clip_root / "simpleopc-heuristic.json"
        heuristic_path.write_text(
            json.dumps(heuristic, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        outputs = []
        recipes = []
        for seed in seeds:
            output, recipe = train_simpleopc_ppo(
                env=make_env(),
                output_path=root / "models" / f"{clip_id}-seed-{seed}",
                total_timesteps=timesteps,
                seed=seed,
                learning_rate=float(oracle["learning_rate"]),
            )
            outputs.append(str(output))
            recipes.append(str(recipe))
        clip_results.append({
            "clip_id": clip_id,
            "split": split,
            "layout_path": str(layout_path),
            "layout_sha256": backend.layout_sha256,
            "segment_count": len(backend.segments),
            "heuristic_path": str(heuristic_path),
            "models": outputs,
            "ppo_recipes": recipes,
        })
    result = {
        "mode": "smoke" if smoke else "full",
        "environment": "simpleopc-multistep-v3",
        "loss_version": SIMPLEOPC_LOSS_VERSION,
        "openilt_mutation": "none",
        "seeds": seeds,
        "timesteps_per_seed": timesteps,
        "episode_step_sizes_nm": step_sizes,
        "displacement_classes_nm": list(DISPLACEMENT_CLASSES_NM),
        "paper_displacement_range_nm": [-displacement_limit, displacement_limit],
        "clips": clip_results,
        "decision_tree_labels": "ppo_recipe_only",
        "frag_status": "not_implemented_requires_nested_resegmentation",
    }
    (root / "stage-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

def build_recipe_stage(config: Dict[str, Any], root: Path) -> None:
    """从版本化点级真值训练 EPE/FRAG 双树并导出确定性 Recipe。"""
    from .recipe_tree import PointTrainingDataset, train_both_trees

    dataset_path = _required_path(config, "point_training_dataset")
    dataset = PointTrainingDataset.parse_obj(json.loads(dataset_path.read_text(encoding="utf-8")))
    if not dataset.label_version.startswith("ppo-simpleopc-"):
        raise RuntimeError(
            "正式决策树只接受通过质量门槛的 PPO SimpleOPC Recipe 标签；"
            f"当前 label_version={dataset.label_version}"
        )
    if dataset.ppo_quality_status != "accepted":
        raise RuntimeError("PPO SimpleOPC 质量报告未 accepted，禁止训练正式决策树")
    if not dataset.ppo_quality_report_sha256 or len(dataset.ppo_quality_report_sha256) != 64:
        raise RuntimeError("PPO 决策树标签缺少 64 位质量报告哈希")
    tasks = {row.task_type.value for row in dataset.rows}
    if tasks != {"EPE", "FRAG"}:
        raise RuntimeError("正式决策树必须同时具有 accepted PPO 生成的 EPE 与 FRAG 标签")
    tree_config = config.get("decision_tree", {})
    max_depth = tree_config.get("max_depth")
    result = train_both_trees(
        dataset,
        seed=int(config.get("run", {}).get("seed", 42)),
        max_depth=int(max_depth) if max_depth is not None else None,
    )
    model_root = root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "epe.tree.json": result.epe_tree.dict(),
        "frag.tree.json": result.frag_tree.dict(),
        "recipe.raw.json": result.recipe.dict(),
    }
    for name, payload in artifacts.items():
        (model_root / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    stage_result = {
        "point_training_dataset": str(dataset_path),
        "epe_macro_f1_all_nine_classes": result.epe_tree.macro_f1_all_nine_classes,
        "frag_macro_f1_all_nine_classes": result.frag_tree.macro_f1_all_nine_classes,
        "recipe_path": str(model_root / "recipe.raw.json"),
        "qwen_interpretation_status": "not_run",
    }
    (root / "stage-result.json").write_text(json.dumps(stage_result, ensure_ascii=False, indent=2), encoding="utf-8")


def run_loop_stage(config: Dict[str, Any], root: Path) -> None:
    """扫描验证结果阈值并写出闭环质量/调用率报告。"""
    from .closed_loop_eval import ClosedLoopSample, build_closed_loop_report

    outcomes_path = _required_path(config, "validation_outcomes")
    raw = json.loads(outcomes_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("validation_outcomes 根节点必须是数组")
    samples = [ClosedLoopSample.parse_obj(item) for item in raw]
    routing = config["routing"]
    report = build_closed_loop_report(
        samples=samples,
        thresholds=[float(value) for value in routing["thresholds"]],
        max_relative_degradation=float(routing["max_relative_epe_d_degradation"]),
        max_refine_rate=float(routing["max_refine_rate"]),
    )
    (root / "closed-loop.thresholds.json").write_text(
        json.dumps(report.dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "stage-result.json").write_text(
        json.dumps({"status": report.status, "selected": report.selected.dict() if report.selected else None,
                    "meets_refine_rate_acceptance": report.meets_refine_rate_acceptance},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

