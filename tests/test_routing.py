"""本模块验证置信度、OOD 和无效 Recipe 的精算路由以及质量受限阈值选择。

输入为可人工审查的验证集快速/精算 EPE D，输出为最低精算率的可行阈值或失败结果。
关键依赖仅为 pytest 和 routing 标准库实现，无需模型或 GPU。
"""
from opc_agent.routing import ValidationOutcome, choose_threshold, should_refine


def test_invalid_recipe_and_ood_always_refine():
    """即使置信度很高，无效 Recipe 与 OOD 都必须精算。"""
    assert should_refine(0.99, 0.5, False, False) == (True, "recipe_invalid")
    assert should_refine(0.99, 0.5, True, True) == (True, "out_of_distribution")


def test_choose_threshold_prefers_fewest_refinements_that_meets_quality():
    """阈值 0.70 修复一个低置信度坏样本，且比更高阈值调用更少。"""
    outcomes = [
        ValidationOutcome(0.95, 10, 10),
        ValidationOutcome(0.65, 20, 10),
        ValidationOutcome(0.40, 10, 10),
    ]
    selected = choose_threshold(outcomes, [0.50, 0.70, 0.90], max_relative_degradation=0.05)
    assert selected is not None
    assert selected.threshold == 0.70
    assert selected.refine_rate == 2 / 3


def test_choose_threshold_returns_none_when_no_quality_constraint_is_met():
    """若所有样本都无法精算且快速质量超标，应明确报告无解。"""
    outcome = ValidationOutcome(0.99, 20, 10, ood=False, recipe_valid=True)
    assert choose_threshold([outcome], [0.5], 0.05) is None

