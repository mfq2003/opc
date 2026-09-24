"""本模块实现 OPC Agent 的统一命令行入口和可复现实验工件初始化。

输入为 YAML 配置、子命令和可选运行编号；输出为 runs/<run_id> 中的快照、日志和 SQLite 记录。
关键依赖为 PyYAML、SQLite 与 OpenILT 适配层；数据或后端缺失时显式报错而不会伪造实验结果。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml

from .engine import OpenILTEngine
from .models import LayoutClip
from .official_simpleopc_results import archive_official_simpleopc_results
from .storage import ExperimentStore
from .workflow import build_recipe_stage, run_loop_stage, train_oracle_stage


def load_config(path: Path) -> Dict[str, Any]:
    """读取 YAML 配置，并拒绝非映射根节点。"""
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("配置根节点必须是映射")
    return content


def run_id(command: str) -> str:
    """生成包含 UTC 时间和随机后缀的不可冲突运行编号。"""
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{command}-{uuid.uuid4().hex[:8]}"


def _dependency_versions() -> Dict[str, str]:
    """读取已安装核心依赖版本；缺失依赖也被显式记录。"""
    result: Dict[str, str] = {"python": sys.version}
    for name in ("numpy", "pydantic", "sklearn", "gymnasium", "stable_baselines3", "torch"):
        try:
            module = __import__(name)
            result[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            result[name] = "not-installed"
    return result


def init_run(config: Dict[str, Any], command: str) -> tuple[str, Path, ExperimentStore]:
    """创建唯一运行目录、配置快照和 SQLite 运行记录。"""
    settings = config.get("run", {})
    identifier = run_id(command)
    root = Path(settings.get("output_root", "runs")) / identifier
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.snapshot.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=True), encoding="utf-8")
    metadata = {"platform": platform.platform(), "dependencies": _dependency_versions(), "command": command}
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    store = ExperimentStore(Path(settings.get("database", "runs/opc_agent.sqlite3")))
    store.upsert_run(identifier, config, metadata)
    return identifier, root, store


def prepare_data(config: Dict[str, Any], identifier: str, root: Path, store: ExperimentStore) -> None:
    """核验十个公开 GLP 文件并按父版图级固定分割写入 SQLite。"""
    data = config["data"]
    split_by_parent = {parent: "train" for parent in data["train_parents"]}
    split_by_parent.update({parent: "validation" for parent in data["validation_parents"]})
    split_by_parent.update({parent: "test" for parent in data["test_parents"]})
    if len(split_by_parent) != 10:
        raise ValueError("ICCAD13 必须恰好有十个不重叠父版图")
    base = Path(data["iccad13_dir"])
    missing = []
    for parent, split in split_by_parent.items():
        source = base / f"{parent}.glp"
        if not source.is_file():
            missing.append(str(source))
            continue
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        store.upsert_clip(LayoutClip(
            clip_id=parent, source="ICCAD13", parent_layout=parent, coordinates_nm=(0, 0, 0, 0),
            image_path=str(source), scale_nm_per_pixel=1.0, file_sha256=digest, split=split,
        ))
    if missing:
        raise FileNotFoundError("缺少 ICCAD13 文件：" + ", ".join(missing))
    (root / "prepare-data.json").write_text(json.dumps({"clips": 10, "splits": split_by_parent}, ensure_ascii=False, indent=2), encoding="utf-8")


def baseline(config: Dict[str, Any], identifier: str, root: Path, store: ExperimentStore) -> None:
    """在隔离目录原样运行固定提交的上游 SimpleOPC，并归档十图结果。"""
    backend = config["openilt"]
    engine = OpenILTEngine(Path(config["data"]["openilt_dir"]), backend["commit"], int(backend["timeout_seconds"]))
    execution_root = root / "official-simpleopc"
    started = time.monotonic()
    output = engine.optimize(Path(config["data"]["iccad13_dir"]), output_dir=execution_root)
    elapsed = time.monotonic() - started
    (root / "openilt-baseline.log").write_text(output, encoding="utf-8")
    revision = engine.revision()
    (root / "openilt-revision.txt").write_text(revision + "\n", encoding="utf-8")
    archive_official_simpleopc_results(
        root,
        execution_root,
        revision,
        dict(config["oracle"]["reward_weights"]),
        elapsed,
    )


def v2_preflight(config: Dict[str, Any], root: Path, layout_parent: str) -> None:
    """运行 v2 单图真实 OpenILT 灵敏度预检并保存 diagnostic-only 工件。"""
    from .recipe_v2_openilt import run_v2_openilt_preflight

    result = run_v2_openilt_preflight(config, layout_parent=layout_parent)
    artifact = root / "recipe-v2-preflight.json"
    artifact.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "artifact": str(artifact),
        "status": result["status"],
        "sensitivity_summary": result["sensitivity_summary"],
        "all_action_valid_probes_nm": [
            item["probe_distance_nm"]
            for item in result["geometry_scan"]
            if item["all_actions_valid_for_all_points"]
        ],
        "training_enabled": result["training_enabled"],
    }, ensure_ascii=False, indent=2))


def v2_episode_smoke(config: Dict[str, Any], root: Path, layout_parent: str) -> None:
    """运行 128 observation 完整 episode smoke，并保存 PPO 输入汇报样例。"""
    from .recipe_v2_openilt import run_v2_openilt_episode_smoke

    result = run_v2_openilt_episode_smoke(
        config, artifact_root=root, layout_parent=layout_parent
    )
    artifact = root / "recipe-v2-episode-smoke.json"
    artifact.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "artifact": str(artifact),
        "status": result["status"],
        "layout_parent": result["layout_parent"],
        "patch_size": result["patch_size"],
        "variant_count": len(result["variants"]),
        "solver_calls": result["solver_calls"],
        "cross_protocol_final_equal": result["cross_protocol_final_equal"],
        "repeat_baseline_equal": result["repeat_baseline_equal"],
        "ppo_input_examples": result["ppo_input_examples"]["manifest"],
        "ppo_input_example_count": result["ppo_input_examples"]["saved_example_count"],
        "pass": result["pass"],
        "training_enabled": result["training_enabled"],
    }, ensure_ascii=False, indent=2))


def v2_input_examples(config: Dict[str, Any], root: Path, layout_parent: str) -> None:
    """用一次冻结基线求解导出 128 observation 的 PPO 输入汇报样例。"""
    from .recipe_v2_openilt import run_v2_openilt_input_examples

    result = run_v2_openilt_input_examples(
        config, artifact_root=root, layout_parent=layout_parent
    )
    artifact = root / "recipe-v2-input-examples.json"
    artifact.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "artifact": str(artifact),
        "status": result["status"],
        "layout_parent": result["layout_parent"],
        "patch_size": result["patch_size"],
        "point_count": result["point_count"],
        "solver_calls": result["solver_calls"],
        "ppo_input_examples": result["ppo_input_examples"]["manifest"],
        "ppo_input_example_count": result["ppo_input_examples"]["saved_example_count"],
        "training_enabled": result["training_enabled"],
    }, ensure_ascii=False, indent=2))


def v2_ppo_smoke(
    config: Dict[str, Any], root: Path, layout_parent: str, protocol: str
) -> None:
    """运行一个独立 dense/terminal v2 PPO CUDA 数值 smoke。"""
    from .recipe_v2_runner import run_v2_ppo_smoke

    result = run_v2_ppo_smoke(
        config,
        artifact_root=root,
        protocol_alias=protocol,
        layout_parent=layout_parent,
    )
    artifact = root / "recipe-v2-ppo-smoke.json"
    artifact.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "artifact": str(artifact),
        "status": result["status"],
        "layout_parent": result["layout_parent"],
        "training_protocol": result["training_protocol"],
        "total_timesteps": result["training"]["total_timesteps_actual"],
        "ppo_updates": result["training"]["numerics"]["update_count"],
        "final_replay_equal": result["deterministic_final_replay"]["final_replay_equal"],
        "numeric_pass": result["numeric_pass"],
        "pass": result["pass"],
        "accepted": result["accepted"],
        "long_training_enabled": result["long_training_enabled"],
    }, ensure_ascii=False, indent=2))


def v2_ppo_pilot(
    config: Dict[str, Any], root: Path, layout_parent: str, protocol: str
) -> None:
    """运行一个独立 dense/terminal v2 PPO 三次更新稳定性 pilot。"""
    from .recipe_v2_runner import run_v2_ppo_pilot

    result = run_v2_ppo_pilot(
        config,
        artifact_root=root,
        protocol_alias=protocol,
        layout_parent=layout_parent,
    )
    artifact = root / "recipe-v2-ppo-pilot.json"
    artifact.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "artifact": str(artifact),
        "status": result["status"],
        "layout_parent": result["layout_parent"],
        "training_protocol": result["training_protocol"],
        "rollout_count": result["rollout_count"],
        "total_timesteps": result["training"]["total_timesteps_actual"],
        "ppo_updates": result["training"]["numerics"]["update_count"],
        "stability_pass": result["stability_pass"],
        "final_replay_equal": result["deterministic_final_replay"]["final_replay_equal"],
        "pass": result["pass"],
        "accepted": result["accepted"],
        "long_training_enabled": result["long_training_enabled"],
    }, ensure_ascii=False, indent=2))


def v2_ppo_small_train(
    config: Dict[str, Any], root: Path, layout_parents: list[str], protocol: str
) -> None:
    """运行双版图单共享模型的 terminal 受控小训练。"""
    from .recipe_v2_small_train import run_v2_ppo_small_train

    result = run_v2_ppo_small_train(
        config,
        artifact_root=root,
        protocol_alias=protocol,
        layout_parents=layout_parents,
    )
    artifact = root / "recipe-v2-ppo-small-train.json"
    artifact.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "artifact": str(artifact),
        "status": result["status"],
        "layout_parents": result["layout_parents"],
        "shared_model": result["shared_model"],
        "training_protocol": result["training_protocol"],
        "total_timesteps_requested": result["training"]["total_timesteps_requested"],
        "total_timesteps_actual": result["training"]["total_timesteps_actual"],
        "completed_update_count": result["training"]["completed_update_count"],
        "completed_episodes_per_layout": result["training"]["completed_episodes_per_layout"],
        "stopped_early": result["training"]["stopped_early"],
        "ppo_input_examples": {
            layout: details["manifest"]
            for layout, details in result["ppo_input_examples"].items()
        },
        "episode_contract_equal": result["episode_contract_equal"],
        "all_final_replays_equal": result["all_final_replays_equal"],
        "execution_pass": result["execution_pass"],
        "accepted": result["accepted"],
        "long_training_enabled": result["long_training_enabled"],
    }, ensure_ascii=False, indent=2))


def require_prior_data(config: Dict[str, Any], command: str) -> None:
    """对需要真实精算数据的后续阶段提供明确而安全的前置条件失败信息。"""
    database = Path(config.get("run", {}).get("database", "runs/opc_agent.sqlite3"))
    if not database.is_file():
        raise RuntimeError(f"{command} 需要先成功执行 prepare-data 和 baseline，未找到 {database}")
    raise NotImplementedError(f"{command} 的真实 GPU 实验必须在已完成 OpenILT 基线后以固定配置启动；当前适配层不会伪造结果")


def report(identifier: str) -> None:
    """为指定运行生成可人工补全的报告骨架，且不宣称未运行的实验结果。"""
    report_root = Path("reports")
    report_root.mkdir(exist_ok=True)
    reproduction = report_root / "reproduction.md"
    closed_loop = report_root / "closed_loop.md"
    reproduction.write_text(f"# 论文复现报告\n\n运行：`{identifier}`\n\n状态：待 GPU 基线和 PPO 真值完成后填写。不得将此骨架视为实验结果。\n", encoding="utf-8")
    closed_loop.write_text(f"# 双颗粒度闭环报告\n\n运行：`{identifier}`\n\n状态：待阈值扫描、精算调用率和消融实验完成后填写。\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """解析统一 CLI 并分派到不产生伪实验结果的阶段实现。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-data", "baseline", "build-recipe", "evaluate", "run-loop"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
    oracle_parser = subparsers.add_parser("train-oracle")
    oracle_parser.add_argument("--config", type=Path, required=True)
    oracle_mode = oracle_parser.add_mutually_exclusive_group()
    oracle_mode.add_argument("--smoke", action="store_true", help="只跑首个种子的少量 CUDA 步数")
    oracle_mode.add_argument(
        "--preflight", action="store_true",
        help="只构造首个 Recipe 环境、运行默认 solver 并检查 observation，不训练 PPO",
    )
    oracle_parser.add_argument(
        "--candidate-index", type=Path,
        help=(
            "仅供 legacy-candidate-point-v1 历史单步管线使用；"
            "当前 simpleopc-recipe-point-v1 主线会明确拒绝该参数"
        ),
    )
    v2_parser = subparsers.add_parser(
        "v2-preflight",
        help="运行单图真实 OpenILT v2 动作/probe/Golden 灵敏度预检，不训练 PPO",
    )
    v2_parser.add_argument("--config", type=Path, required=True)
    v2_parser.add_argument("--layout", default="M1_test1")
    v2_smoke_parser = subparsers.add_parser(
        "v2-episode-smoke",
        help="运行 128 observation 的 dense/terminal 完整 episode，不训练 PPO",
    )
    v2_smoke_parser.add_argument("--config", type=Path, required=True)
    v2_smoke_parser.add_argument("--layout")
    v2_examples_parser = subparsers.add_parser(
        "v2-input-examples",
        help="只求解一次基线并导出 128 observation 的五通道 PPO 输入样例",
    )
    v2_examples_parser.add_argument("--config", type=Path, required=True)
    v2_examples_parser.add_argument("--layout")
    v2_ppo_parser = subparsers.add_parser(
        "v2-ppo-smoke",
        help="在 M1_test4 上独立运行 dense 或 terminal 的单 rollout CUDA PPO smoke",
    )
    v2_ppo_parser.add_argument("--config", type=Path, required=True)
    v2_ppo_parser.add_argument("--layout")
    v2_ppo_parser.add_argument(
        "--protocol", choices=("dense", "terminal"), required=True
    )
    v2_pilot_parser = subparsers.add_parser(
        "v2-ppo-pilot",
        help="在 M1_test4 上独立运行 dense 或 terminal 的三次更新稳定性 pilot",
    )
    v2_pilot_parser.add_argument("--config", type=Path, required=True)
    v2_pilot_parser.add_argument("--layout")
    v2_pilot_parser.add_argument(
        "--protocol", choices=("dense", "terminal"), required=True
    )
    v2_small_train_parser = subparsers.add_parser(
        "v2-ppo-small-train",
        help="在 M1_test5/6 上运行一个共享 terminal PPO 模型的五次更新受控小训练",
    )
    v2_small_train_parser.add_argument("--config", type=Path, required=True)
    v2_small_train_parser.add_argument(
        "--layouts",
        nargs=2,
        required=True,
        metavar=("LAYOUT_A", "LAYOUT_B"),
        help="必须与 small_train.layout_parents 顺序一致，例如 M1_test5 M1_test6",
    )
    v2_small_train_parser.add_argument(
        "--protocol", choices=("terminal",), required=True
    )
    report_parser = subparsers.add_parser("report")
    search_parser = subparsers.add_parser("v2-search", help="全零基线与可续跑离散坐标搜索诊断")
    search_parser.add_argument("--config", type=Path, required=True)
    search_parser.add_argument(
        "--layouts",
        nargs="+",
        help="显式版图列表；默认使用配置中的训练六图",
    )
    search_parser.add_argument(
        "--allow-validation-test-diagnostic",
        action="store_true",
        help="明确授权在 M1_test7–10 上使用自身 solver 反馈做逐图启发式诊断",
    )
    report_parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "report":
        report(args.run_id)
        return 0
    config = load_config(args.config)
    if args.command == "v2-search":
        if args.allow_validation_test_diagnostic:
            expected = list(
                config["data"]["validation_parents"] + config["data"]["test_parents"]
            )
            requested = list(args.layouts or [])
            if requested != expected:
                raise ValueError(
                    "评估诊断必须显式按顺序提供全部 M1_test7–10，不能挑图"
                )
            config["search"]["scope"] = "validation_test_diagnostic"
            config["search"]["layout_parents"] = requested
            config["search"]["resume_from"] = None
        elif args.layouts is not None:
            if list(args.layouts) != list(config["data"]["train_parents"]):
                raise ValueError(
                    "未授权评估诊断时，--layouts 只能完整等于训练版图 M1_test1–6"
                )
            config["search"]["scope"] = "train_only"
            config["search"]["layout_parents"] = list(args.layouts)
    identifier, root, store = init_run(config, args.command)
    try:
        if args.command == "v2-search":
            from .recipe_v2_search import run_v2_search

            result = run_v2_search(config, root)
            print(json.dumps({"artifact": str(root / "recipe-v2-search.json"),
                              "status": result["status"], "solver_calls": result["solver_calls"],
                              "accepted": False}, ensure_ascii=False, indent=2))
        elif args.command == "prepare-data":
            prepare_data(config, identifier, root, store)
        elif args.command == "baseline":
            baseline(config, identifier, root, store)
        elif args.command == "train-oracle":
            train_oracle_stage(
                config,
                root,
                smoke=args.smoke,
                candidate_index=args.candidate_index,
                preflight=args.preflight,
            )
        elif args.command == "v2-preflight":
            v2_preflight(config, root, layout_parent=args.layout)
        elif args.command == "v2-episode-smoke":
            layout_parent = args.layout or config.get("episode_smoke", {}).get(
                "layout_parent", "M1_test4"
            )
            v2_episode_smoke(config, root, layout_parent=layout_parent)
        elif args.command == "v2-input-examples":
            layout_parent = args.layout or config.get("episode_smoke", {}).get(
                "layout_parent", "M1_test4"
            )
            v2_input_examples(config, root, layout_parent=layout_parent)
        elif args.command == "v2-ppo-smoke":
            layout_parent = args.layout or config.get("training", {}).get(
                "smoke", {}
            ).get("layout_parent", "M1_test4")
            v2_ppo_smoke(
                config,
                root,
                layout_parent=layout_parent,
                protocol=args.protocol,
            )
        elif args.command == "v2-ppo-pilot":
            layout_parent = args.layout or config.get("training", {}).get(
                "pilot", {}
            ).get("layout_parent", "M1_test4")
            v2_ppo_pilot(
                config,
                root,
                layout_parent=layout_parent,
                protocol=args.protocol,
            )
        elif args.command == "v2-ppo-small-train":
            v2_ppo_small_train(
                config,
                root,
                layout_parents=args.layouts,
                protocol=args.protocol,
            )
        elif args.command == "build-recipe":
            build_recipe_stage(config, root)
        elif args.command == "run-loop":
            run_loop_stage(config, root)
        else:
            require_prior_data(config, args.command)
        print(identifier)
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())


