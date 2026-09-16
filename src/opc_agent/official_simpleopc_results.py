"""本模块归档 OpenILT 官方 ``pyilt/simpleopc.py`` 的十图原样运行结果。

输入为上游未修改脚本的标准输出、隔离运行目录、固定提交号和统一评价权重；输出为逐图初始化、
八轮轨迹、官方最终轮、离线共同目标最佳轮以及 target/mask/resist 文件哈希。模块只解析和归档，
不会改变官方分段、EPE 检查、移动步长、最终图片选择或光刻计算。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List


INITIAL_PATTERN = re.compile(
    r"^\[Testcase (?P<index>\d+) Initialized\]: L2 (?P<l2>[\d.]+); "
    r"PVBand (?P<pvb>[\d.]+); EPE (?P<epe>[\d.]+); Shot: (?P<shot>-?[\d.]+)$",
    re.MULTILINE,
)
STEP_PATTERN = re.compile(
    r"^\[Testcase (?P<index>\d+) Step (?P<step>\d+)\]: L2 (?P<l2>[\d.]+); "
    r"PVBand (?P<pvb>[\d.]+); EPE (?P<epe>[\d.]+); Shot: (?P<shot>-?[\d.]+)$",
    re.MULTILINE,
)


def _metrics(match: re.Match, weights: Dict[str, float]) -> dict:
    """把一行官方指标转换为带共同加权损失的结构化记录。"""
    l2 = float(match.group("l2"))
    pvb = float(match.group("pvb"))
    epe = float(match.group("epe"))
    return {
        "l2": l2,
        "pvb": pvb,
        "epe_total": epe,
        "shot": float(match.group("shot")),
        "weighted_loss": weights["l2"] * l2 + weights["epe"] * epe + weights["pvb"] * pvb,
    }


def parse_official_simpleopc_log(content: str, reward_weights: Dict[str, float]) -> List[dict]:
    """严格解析十图各一条初始化记录和连续八轮官方 SimpleOPC 记录。"""
    weights = {name: float(value) for name, value in reward_weights.items()}
    if set(weights) != {"l2", "epe", "pvb"}:
        raise ValueError("reward_weights 必须恰好包含 l2、epe、pvb")
    initial_by_index = {}
    for match in INITIAL_PATTERN.finditer(content):
        index = int(match.group("index"))
        if index in initial_by_index:
            raise ValueError(f"Testcase {index} 出现重复 Initialized 记录")
        initial_by_index[index] = _metrics(match, weights)
    steps_by_index = {index: [] for index in range(1, 11)}
    for match in STEP_PATTERN.finditer(content):
        index = int(match.group("index"))
        if index not in steps_by_index:
            raise ValueError(f"出现范围外 Testcase {index}")
        steps_by_index[index].append({"step": int(match.group("step")), **_metrics(match, weights)})
    if sorted(initial_by_index) != list(range(1, 11)):
        raise ValueError("官方 SimpleOPC 日志必须包含 Testcase 1 到 10 的 Initialized 记录")
    cases = []
    for index in range(1, 11):
        steps = steps_by_index[index]
        if [item["step"] for item in steps] != list(range(8)):
            raise ValueError(f"Testcase {index} 必须按顺序包含 Step 0 到 7")
        cases.append({
            "parent_layout": f"M1_test{index}",
            "initial": initial_by_index[index],
            "steps": steps,
            "official_final": steps[-1],
            "common_objective_best": min(steps, key=lambda item: item["weighted_loss"]),
        })
    return cases


def _sha256(path: Path) -> str:
    """计算非空官方 PNG 的 SHA256，并对缺失产物显式失败。"""
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"官方 SimpleOPC 产物不存在或为空：{path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_official_simpleopc_results(
    run_root: Path,
    execution_root: Path,
    revision: str,
    reward_weights: Dict[str, float],
    runtime_seconds: float,
) -> Path:
    """将官方日志、十图轨迹、图片路径和来源哈希写入单一 JSON 证据文件。"""
    run_root = Path(run_root)
    log_path = run_root / "openilt-baseline.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"未找到官方 SimpleOPC 日志：{log_path}")
    cases = parse_official_simpleopc_log(log_path.read_text(encoding="utf-8"), reward_weights)
    tmp = Path(execution_root) / "tmp"
    for index, case in enumerate(cases, start=1):
        images = {}
        for kind in ("target", "mask", "resist"):
            path = tmp / f"SimpleOPC_{kind}{index}.png"
            images[kind] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        case["images"] = images
    payload = {
        "schema_version": "official-simpleopc-baseline-v1",
        "algorithm": "OpenILT pyilt/simpleopc.py unmodified",
        "selection_note": (
            "official_final 对应上游实际保存的 Step 7；common_objective_best 仅离线报告，"
            "没有改变上游输出或算法。"
        ),
        "openilt_revision": revision,
        "reward_weights": {name: float(value) for name, value in reward_weights.items()},
        "runtime_seconds_total": float(runtime_seconds),
        "case_count": len(cases),
        "cases": cases,
    }
    destination = run_root / "official-simpleopc.metrics.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination
