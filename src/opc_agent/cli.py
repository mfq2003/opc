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
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml

from .engine import OpenILTEngine
from .models import LayoutClip
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
    """运行固定提交的上游 SimpleILT，并将未解析原始日志存入运行目录。"""
    backend = config["openilt"]
    engine = OpenILTEngine(Path(config["data"]["openilt_dir"]), backend["commit"], int(backend["timeout_seconds"]))
    output = engine.optimize(Path(config["data"]["iccad13_dir"]))
    (root / "openilt-baseline.log").write_text(output, encoding="utf-8")
    (root / "openilt-revision.txt").write_text(engine.revision() + "\n", encoding="utf-8")


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
    oracle_parser.add_argument("--smoke", action="store_true", help="只跑首个种子的少量 CUDA 步数")
    oracle_parser.add_argument(
        "--candidate-index", type=Path,
        help="可选多 clip 候选索引；与 --smoke 合用时执行所有索引 clip 的小步数试跑",
    )
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "report":
        report(args.run_id)
        return 0
    config = load_config(args.config)
    identifier, root, store = init_run(config, args.command)
    try:
        if args.command == "prepare-data":
            prepare_data(config, identifier, root, store)
        elif args.command == "baseline":
            baseline(config, identifier, root, store)
        elif args.command == "train-oracle":
            train_oracle_stage(config, root, smoke=args.smoke, candidate_index=args.candidate_index)
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


