"""本模块离线解析 OpenILT SimpleILT 基线日志并将逐图结果写入 JSON 与 SQLite。

输入为已完成运行目录中的 openilt-baseline.log、运行编号和 SQLite 数据库路径；输出为结构化的逐图/汇总 JSON，
并向 metrics 表写入每个父版图的可追溯原始基线指标。关键依赖为标准库、Pydantic 和本项目 SQLite 模式；
模块不会调用 GPU、OpenILT、网络或 API。上游日志只提供总 EPE，因此本模块明确保存 epe_total，绝不伪造 EPE N/EPE D。
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import List, Tuple

from pydantic import BaseModel, Field


CASE_PATTERN = re.compile(
    r"^\[Testcase (?P<index>\d+)\]: L2 (?P<l2>\d+); PVBand (?P<pvb>\d+); "
    r"EPE (?P<epe>\d+); Shot: (?P<shots>\d+); SolveTime: (?P<runtime>[\d.]+)s$",
    re.MULTILINE,
)
SUMMARY_PATTERN = re.compile(
    r"^\[Result\]: L2 (?P<l2>[\d.]+); PVBand (?P<pvb>[\d.]+); EPE (?P<epe>[\d.]+); "
    r"Shot (?P<shots>[\d.]+); SolveTime (?P<runtime>[\d.]+)s$",
    re.MULTILINE,
)


class OpenILTBaselineCase(BaseModel):
    """保存上游日志中一个 ICCAD13 父版图的总 EPE 与其他公开基线指标。"""

    schema_version: str = "1.0"
    parent_layout: str
    l2: float = Field(ge=0)
    pvb: float = Field(ge=0)
    epe_total: float = Field(ge=0)
    shots: float = Field(ge=0)
    runtime_seconds: float = Field(ge=0)
    epe_n: None = None
    epe_d: None = None
    metric_note: str = "OpenILT 上游日志仅提供总 EPE；EPE N/EPE D 未测量，不能推断。"


class OpenILTBaselineSummary(BaseModel):
    """保存上游日志的十图平均指标及 EPE 粒度限制说明。"""

    schema_version: str = "1.0"
    l2: float = Field(ge=0)
    pvb: float = Field(ge=0)
    epe_total: float = Field(ge=0)
    shots: float = Field(ge=0)
    runtime_seconds: float = Field(ge=0)
    metric_note: str = "OpenILT 上游日志仅提供总 EPE；EPE N/EPE D 未测量，不能推断。"


def parse_simpleilt_log(content: str) -> Tuple[List[OpenILTBaselineCase], OpenILTBaselineSummary]:
    """解析严格的十图 SimpleILT 日志；缺图、重复图或缺少汇总行均显式失败。"""
    cases = [
        OpenILTBaselineCase(
            parent_layout=f"M1_test{match.group('index')}",
            l2=float(match.group("l2")),
            pvb=float(match.group("pvb")),
            epe_total=float(match.group("epe")),
            shots=float(match.group("shots")),
            runtime_seconds=float(match.group("runtime")),
        )
        for match in CASE_PATTERN.finditer(content)
    ]
    indices = [int(case.parent_layout[len("M1_test"): ]) for case in cases]
    if indices != list(range(1, 11)):
        raise ValueError("OpenILT 基线日志必须按顺序包含且仅包含 Testcase 1 到 10")
    summary_matches = list(SUMMARY_PATTERN.finditer(content))
    if len(summary_matches) != 1:
        raise ValueError("OpenILT 基线日志必须包含且仅包含一条 [Result] 汇总")
    summary = summary_matches[0]
    return cases, OpenILTBaselineSummary(
        l2=float(summary.group("l2")),
        pvb=float(summary.group("pvb")),
        epe_total=float(summary.group("epe")),
        shots=float(summary.group("shots")),
        runtime_seconds=float(summary.group("runtime")),
    )


def archive_baseline_results(run_id: str, run_root: Path, database: Path) -> Path:
    """解析已有基线日志，原子写入十条 SQLite 指标记录和一份 JSON 归档。"""
    log_path = run_root / "openilt-baseline.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"未找到基线日志：{log_path}")
    cases, summary = parse_simpleilt_log(log_path.read_text(encoding="utf-8"))
    if not database.is_file():
        raise FileNotFoundError(f"未找到 SQLite 数据库：{database}")
    stage = "openilt_simpleilt_baseline_v1"
    connection = sqlite3.connect(str(database))
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        for case in cases:
            payload = json.dumps(case.dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            connection.execute(
                "INSERT OR REPLACE INTO metrics(run_id, clip_id, stage, payload_json) VALUES (?, ?, ?, ?)",
                (run_id, case.parent_layout, stage, payload),
            )
        connection.commit()
    finally:
        connection.close()
    archive_path = run_root / "openilt-baseline.metrics.json"
    archive_path.write_text(
        json.dumps({"run_id": run_id, "stage": stage, "cases": [case.dict() for case in cases], "summary": summary.dict()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return archive_path


def main(argv: List[str] | None = None) -> int:
    """提供不重跑 GPU 基线的命令行归档入口。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.baseline_results")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--database", type=Path, default=Path("runs/opc_agent.sqlite3"))
    args = parser.parse_args(argv)
    archive_path = archive_baseline_results(args.run_id, args.runs_root / args.run_id, args.database)
    print(archive_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


