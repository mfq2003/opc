"""本模块实现快速模型到 PPO+OpenILT 精算层的确定性路由与阈值搜索。

输入为预测置信度、OOD/Recipe 校验状态及验证集快速或精算 EPE D；输出为路由决定和最省精算调用的可行阈值。
关键依赖仅为标准库，确保质量约束逻辑可以独立审计和测试。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class ValidationOutcome:
    """保存一个验证样本的快速结果、全精算结果与不可置信标志。"""

    confidence: float
    fast_epe_d: float
    oracle_epe_d: float
    ood: bool = False
    recipe_valid: bool = True


@dataclass(frozen=True)
class ThresholdResult:
    """保存一个阈值对应的精算率、闭环 EPE D 和相对退化。"""

    threshold: float
    refine_rate: float
    loop_epe_d: float
    relative_degradation: float


def should_refine(confidence: float, threshold: float, ood: bool, recipe_valid: bool) -> tuple[bool, str]:
    """按失效优先、OOD 次之、低置信度最后的顺序决定是否精算。"""
    if not recipe_valid:
        return True, "recipe_invalid"
    if ood:
        return True, "out_of_distribution"
    if confidence < threshold:
        return True, "low_confidence"
    return False, "fast_path"


def evaluate_threshold(outcomes: Sequence[ValidationOutcome], threshold: float) -> ThresholdResult:
    """用 oracle 替换被精算样本，计算闭环相对全精算的质量退化。"""
    if not outcomes:
        raise ValueError("验证集不能为空")
    decisions = [should_refine(item.confidence, threshold, item.ood, item.recipe_valid)[0] for item in outcomes]
    loop_total = sum(item.oracle_epe_d if refine else item.fast_epe_d for item, refine in zip(outcomes, decisions))
    oracle_total = sum(item.oracle_epe_d for item in outcomes)
    if oracle_total <= 0:
        degradation = 0.0 if loop_total <= 0 else float("inf")
    else:
        degradation = (loop_total - oracle_total) / oracle_total
    return ThresholdResult(threshold, sum(decisions) / len(decisions), loop_total, degradation)


def choose_threshold(
    outcomes: Sequence[ValidationOutcome], thresholds: Iterable[float], max_relative_degradation: float = 0.05
) -> Optional[ThresholdResult]:
    """返回满足质量约束且精算率最低的阈值；无可行解返回 None。"""
    candidates = [evaluate_threshold(outcomes, float(threshold)) for threshold in thresholds]
    feasible = [item for item in candidates if item.relative_degradation <= max_relative_degradation]
    return min(feasible, key=lambda item: (item.refine_rate, item.threshold)) if feasible else None

