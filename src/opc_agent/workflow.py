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
    preflight: bool = False,
) -> None:
    """在 CUDA 上训练 PPO；支持单 clip 冒烟或按索引逐 clip、逐种子运行。"""
    environment = str(config.get("oracle", {}).get("environment", "legacy-candidate-point-v1"))
    if environment == "simpleopc-recipe-point-v1":
        if candidate_index is not None:
            raise ValueError("Recipe point PPO 直接读取 GLP，不接受旧 candidate-index")
        _train_recipe_point_stage(config, root, smoke=smoke, preflight=preflight)
        return
    if preflight:
        raise ValueError("--preflight 只支持 simpleopc-recipe-point-v1")
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


def create_recipe_point_solver(config: Dict[str, Any], clip_id: str):
    """按训练主线的同一配置为指定 ICCAD13 版图建立只读 Recipe solver。"""
    from .recipe_ppo import OpenILTRecipeAwareSolver

    data = config["data"]
    backend_config = config["openilt"]
    oracle = config["oracle"]
    simpleopc = config.get("simpleopc")
    if not isinstance(simpleopc, dict):
        raise ValueError("simpleopc-recipe-point-v1 需要 simpleopc 配置段")
    image_size = tuple(int(value) for value in simpleopc.get("image_size", [2048, 2048]))
    if len(image_size) != 2:
        raise ValueError("simpleopc.image_size 必须包含 width、height 两个整数")
    return OpenILTRecipeAwareSolver(
        openilt_dir=Path(data["openilt_dir"]),
        expected_commit=backend_config["commit"],
        layout_path=Path(data["iccad13_dir"]) / f"{clip_id}.glp",
        reward_weights=dict(oracle["reward_weights"]),
        lithography_config=Path(oracle["lithography_config"]),
        simulator=oracle["simulator"],
        image_size=(image_size[0], image_size[1]),
        openilt_scale=int(oracle.get("openilt_scale", 1)),
        nm_per_coordinate=float(simpleopc.get("nm_per_coordinate", 1.0)),
        base_fragment_length_nm=float(simpleopc["base_fragment_length_nm"]),
        min_fragment_length_nm=float(simpleopc["min_fragment_length_nm"]),
        recipe_displacement_limit_nm=float(oracle["displacement_nm_limit"]),
        epe_sample_distance_nm=float(simpleopc["epe_sample_distance_nm"]),
        inner_step_sizes_nm=[float(value) for value in simpleopc["inner_step_sizes_nm"]],
        mask_displacement_limit_nm=float(simpleopc["mask_displacement_limit_nm"]),
        threshold=float(simpleopc.get("threshold", 0.5)),
        cache_entries=int(simpleopc.get("solver_cache_entries", 8)),
    )


def _train_recipe_point_stage(
    config: Dict[str, Any], root: Path, smoke: bool, preflight: bool = False
) -> None:
    """训练共享的单步九分类 Recipe PPO，并逐 clip 导出 EPE/FRAG 最佳 Recipe。"""
    from .metrics import DISPLACEMENT_CLASSES_NM, RECIPE_OPC_LOSS_VERSION
    from .recipe_ppo import (
        RECIPE_ENV_VERSION,
        RECIPE_OBSERVATION_VERSION,
        RECIPE_POINT_VERSION,
        RecipePointPPOEnv,
    )
    from .recipe_ppo_runner import (
        PPO_RECIPE_LABEL_VERSION,
        build_recipe_payload,
        deterministic_recipe_rollout,
        run_default_recipe,
        train_shared_recipe_ppo,
    )

    data = config["data"]
    backend_config = config["openilt"]
    oracle = config["oracle"]
    simpleopc = config.get("simpleopc")
    if not isinstance(simpleopc, dict):
        raise ValueError("simpleopc-recipe-point-v1 需要 simpleopc 配置段")
    if str(oracle.get("reward_mode", "paper_raw")) != "paper_raw":
        raise ValueError("论文主线当前要求 oracle.reward_mode=paper_raw")
    displacement_limit = float(oracle["displacement_nm_limit"])
    if displacement_limit != 40.0:
        raise ValueError("论文公开的 Recipe 点位移范围要求 displacement_nm_limit=40")
    configured_classes = tuple(int(value) for value in simpleopc["displacement_classes_nm"])
    if configured_classes != DISPLACEMENT_CLASSES_NM:
        raise ValueError(
            "simpleopc.displacement_classes_nm 必须为 "
            f"{list(DISPLACEMENT_CLASSES_NM)}"
        )
    patch_size = int(simpleopc.get("local_patch_size", 64))
    if patch_size != 64:
        raise ValueError("论文主线当前固定使用 64×64 像素局部图像")
    if "step_sizes_nm" in simpleopc:
        raise ValueError("Recipe point PPO 禁止配置旧四步 step_sizes_nm")
    image_size = tuple(int(value) for value in simpleopc.get("image_size", [2048, 2048]))
    if len(image_size) != 2:
        raise ValueError("simpleopc.image_size 必须包含 width、height 两个整数")
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
    if smoke or preflight:
        jobs = jobs[:1]
    seeds = [int(oracle["seeds"][0])] if smoke else [int(value) for value in oracle["seeds"]]
    timesteps = int(oracle["smoke_timesteps"] if smoke else oracle["total_timesteps"])

    def make_solver(clip_id: str):
        """为一个 GLP 建立只读 OpenILT recipe-aware solver。"""
        return create_recipe_point_solver(config, clip_id)

    def make_env(solver, shuffle_points: bool) -> RecipePointPPOEnv:
        """为共享 solver 建立独立的点级 episode 状态。"""
        return RecipePointPPOEnv(
            solver=solver,
            reward_weights=dict(oracle["reward_weights"]),
            displacement_classes_nm=configured_classes,
            patch_size=patch_size,
            shuffle_points=shuffle_points,
            metric_epsilon=float(simpleopc.get("metric_epsilon", 1.0)),
        )

    train_clip_ids = [clip_id for clip_id, split in jobs if split == "train"]
    if not train_clip_ids:
        raise ValueError("Recipe PPO 运行没有 train clip")
    train_solvers = {clip_id: make_solver(clip_id) for clip_id in train_clip_ids}
    clip_results: Dict[str, Dict[str, Any]] = {}

    def ensure_clip_entry(clip_id: str, split: str, solver) -> Dict[str, Any]:
        """在训练前保存零位移默认 recipe 基线，并初始化 stage 条目。"""
        existing = clip_results.get(clip_id)
        if existing is not None:
            return existing
        clip_root = root / "clips" / clip_id
        clip_root.mkdir(parents=True, exist_ok=True)
        baseline = run_default_recipe(make_env(solver, shuffle_points=False), seed=0)
        baseline_path = clip_root / "default-recipe.json"
        baseline_path.write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        points = tuple(solver.recipe_points)
        entry = {
            "clip_id": clip_id,
            "split": split,
            "layout_path": str(Path(data["iccad13_dir"]) / f"{clip_id}.glp"),
            "layout_sha256": solver.layout_sha256,
            "point_count": len(points),
            "epe_point_count": sum(point.task_type == "EPE" for point in points),
            "frag_point_count": sum(point.task_type == "FRAG" for point in points),
            "episode_horizon": len(points),
            "default_recipe_path": str(baseline_path),
            "models": [],
            "ppo_recipes": [],
        }
        clip_results[clip_id] = entry
        return entry

    for clip_id in train_clip_ids:
        ensure_clip_entry(clip_id, "train", train_solvers[clip_id])

    shared_models = []

    def stage_payload() -> Dict[str, Any]:
        """构造当前可恢复进度和最终结果共用的数据协议。"""
        return {
            "mode": "preflight" if preflight else ("smoke" if smoke else "full"),
            "environment": RECIPE_ENV_VERSION,
            "observation_version": RECIPE_OBSERVATION_VERSION,
            "point_version": RECIPE_POINT_VERSION,
            "loss_version": RECIPE_OPC_LOSS_VERSION,
            "label_version": PPO_RECIPE_LABEL_VERSION,
            "reward_mode": "paper_raw",
            "openilt_mutation": "none",
            "policy_scope": "shared_train_clips",
            "action_space": "Discrete(9)",
            "action_semantics": "one_absolute_nine_class_decision_per_recipe_point",
            "patch_shape": [5, 64, 64],
            "vector_shape": [14],
            "seeds": seeds,
            "timesteps_per_shared_seed": timesteps,
            "shared_models": list(shared_models),
            "displacement_classes_nm": list(DISPLACEMENT_CLASSES_NM),
            "paper_displacement_range_nm": [-displacement_limit, displacement_limit],
            "clips": [
                clip_results[clip_id]
                for clip_id, _split in jobs
                if clip_id in clip_results
            ],
            "decision_tree_labels": "deferred_until_ppo_acceptance",
            "frag_status": "implemented_as_recipe_segmentation_points",
        }

    def write_progress(status: str, seed: int | None = None, clip_id: str | None = None) -> None:
        """在长任务关键节点原子替换小型进度文件，保留中断后的只读证据。"""
        payload = stage_payload()
        payload.update({"progress_status": status, "current_seed": seed, "current_clip": clip_id})
        progress_path = root / "stage-progress.json"
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(progress_path)

    write_progress("initialized")
    if preflight:
        clip_id = train_clip_ids[0]
        environment = make_env(train_solvers[clip_id], shuffle_points=False)
        observation, reset_info = environment.reset(seed=seeds[0])
        result = stage_payload()
        result["preflight"] = {
            "clip_id": clip_id,
            "image_shape": list(observation["image"].shape),
            "vector_shape": list(observation["vector"].shape),
            "reset": reset_info,
            "training_started": False,
        }
        (root / "stage-result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_progress("complete")
        return
    for seed in seeds:
        train_envs = [
            make_env(train_solvers[clip_id], shuffle_points=bool(simpleopc.get("shuffle_points", True)))
            for clip_id in train_clip_ids
        ]
        model, model_path = train_shared_recipe_ppo(
            train_envs=train_envs,
            output_path=root / "models" / f"shared-recipe-seed-{seed}",
            total_timesteps=timesteps,
            seed=seed,
            learning_rate=float(oracle["learning_rate"]),
            n_steps=int(oracle.get("ppo_n_steps", 256)),
            batch_size=int(oracle.get("ppo_batch_size", 64)),
        )
        shared_models.append(str(model_path))
        write_progress("model_saved", seed=seed)
        for clip_id, split in jobs:
            temporary_solver = clip_id not in train_solvers
            solver = train_solvers.get(clip_id) or make_solver(clip_id)
            entry = ensure_clip_entry(clip_id, split, solver)
            evaluation_env = make_env(solver, shuffle_points=False)
            rollout = deterministic_recipe_rollout(model, evaluation_env, seed)
            recipe_payload = build_recipe_payload(evaluation_env, rollout, model_path, seed)
            recipe_path = root / "recipes" / f"{clip_id}-seed-{seed}.recipe.json"
            recipe_path.parent.mkdir(parents=True, exist_ok=True)
            recipe_path.write_text(
                json.dumps(recipe_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            entry["models"].append(str(model_path))
            entry["ppo_recipes"].append(str(recipe_path))
            write_progress("recipe_saved", seed=seed, clip_id=clip_id)
            if temporary_solver:
                del evaluation_env
                del solver

    result = stage_payload()
    (root / "stage-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_progress("complete")


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

