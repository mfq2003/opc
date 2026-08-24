"""本模块对验证集执行双颗粒度闭环阈值扫描并生成可审计的版本化结果。

输入为逐样本置信度、快速/全精算 EPE D、OOD 与 Recipe 校验状态，以及 closed_loop.yaml；输出为所有
候选阈值的精算率、闭环 EPE D、相对退化、最终阈值和验收状态。关键依赖为 Pydantic、PyYAML 与本项目
routing 模块；模块只分析已完成结果，不调用 GPU、OpenILT、PPO、Qwen 或网络，也不会伪造缺失精算值。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, validator

from .routing import ThresholdResult, ValidationOutcome, choose_threshold, evaluate_threshold


class ClosedLoopSample(BaseModel):
    """保存一个验证样本的快速预测和全精算对照。"""

    schema_version: str = "1.0"
    sample_id: str
    parent_layout: str
    split: str = "validation"
    confidence: float = Field(ge=0, le=1)
    fast_epe_d: float = Field(ge=0)
    oracle_epe_d: float = Field(ge=0)
    ood: bool = False
    recipe_valid: bool = True

    @validator("split")
    def _validation_only(cls, value: str) -> str:
        if value != "validation":
            raise ValueError("闭环阈值只能使用 validation 数据，禁止查看 test")
        return value


class ThresholdRecord(BaseModel):
    """保存单个候选阈值的质量与调用率。"""

    threshold: float = Field(ge=0, le=1)
    refine_rate: float = Field(ge=0, le=1)
    loop_epe_d: float = Field(ge=0)
    relative_degradation: float
    quality_feasible: bool


class ClosedLoopReport(BaseModel):
    """保存阈值扫描、最终选择和 v1 精算率验收结论。"""

    schema_version: str = "1.0"
    input_sha256: str = Field(min_length=64, max_length=64)
    sample_count: int = Field(ge=1)
    max_relative_epe_d_degradation: float = Field(ge=0)
    max_refine_rate: float = Field(ge=0, le=1)
    candidates: List[ThresholdRecord]
    selected: Optional[ThresholdRecord] = None
    status: str
    meets_refine_rate_acceptance: bool = False


def _as_outcomes(samples: List[ClosedLoopSample]) -> List[ValidationOutcome]:
    """转换为纯标准库路由层使用的数据结构。"""
    return [
        ValidationOutcome(
            confidence=sample.confidence,
            fast_epe_d=sample.fast_epe_d,
            oracle_epe_d=sample.oracle_epe_d,
            ood=sample.ood,
            recipe_valid=sample.recipe_valid,
        )
        for sample in samples
    ]


def build_closed_loop_report(
    samples: List[ClosedLoopSample],
    thresholds: List[float],
    max_relative_degradation: float,
    max_refine_rate: float,
) -> ClosedLoopReport:
    """扫描阈值并选择质量可行且精算调用率最低者。"""
    if not samples:
        raise ValueError("闭环验证样本不能为空")
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("闭环验证样本存在重复 sample_id")
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("thresholds 必须是 [0, 1] 内的非空数组")
    outcomes = _as_outcomes(samples)
    raw_results = [evaluate_threshold(outcomes, value) for value in thresholds]
    candidates = [
        ThresholdRecord(
            threshold=item.threshold,
            refine_rate=item.refine_rate,
            loop_epe_d=item.loop_epe_d,
            relative_degradation=item.relative_degradation,
            quality_feasible=item.relative_degradation <= max_relative_degradation,
        )
        for item in raw_results
    ]
    chosen: Optional[ThresholdResult] = choose_threshold(outcomes, thresholds, max_relative_degradation)
    selected = None
    if chosen is not None:
        selected = next(item for item in candidates if item.threshold == chosen.threshold)
    normalized = [sample.dict() for sample in sorted(samples, key=lambda item: item.sample_id)]
    digest = hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ClosedLoopReport(
        input_sha256=digest,
        sample_count=len(samples),
        max_relative_epe_d_degradation=max_relative_degradation,
        max_refine_rate=max_refine_rate,
        candidates=candidates,
        selected=selected,
        status="feasible" if selected is not None else "no_feasible_threshold",
        meets_refine_rate_acceptance=selected is not None and selected.refine_rate <= max_refine_rate,
    )


def _write_immutable(path: Path, report: ClosedLoopReport) -> None:
    """允许完全相同结果重复执行，拒绝覆盖不同的旧实验。"""
    encoded = json.dumps(report.dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"拒绝覆盖已有闭环结果：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(encoded, encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    """读取验证结果与配置，写出闭环阈值扫描 JSON。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.closed_loop_eval")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("闭环输入 JSON 根节点必须是数组")
    samples = [ClosedLoopSample.parse_obj(item) for item in raw]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["routing"]
    report = build_closed_loop_report(
        samples=samples,
        thresholds=[float(value) for value in config["thresholds"]],
        max_relative_degradation=float(config["max_relative_epe_d_degradation"]),
        max_refine_rate=float(config["max_refine_rate"]),
    )
    _write_immutable(args.output, report)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
