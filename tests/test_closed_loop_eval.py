"""本模块验证闭环阈值选择、精算率验收和测试集隔离。

输入为小型验证样本集合；输出为最低精算率可行阈值、无可行解状态和错误 split 断言。
关键依赖为 Pydantic 与 pytest；测试不调用 GPU、OpenILT、PPO、Qwen、网络或数据库。
"""
import pytest

from opc_agent.closed_loop_eval import ClosedLoopSample, build_closed_loop_report


def _sample(sample_id: str, confidence: float, fast: float, oracle: float) -> ClosedLoopSample:
    """创建一个最小验证样本。"""
    return ClosedLoopSample(
        sample_id=sample_id,
        parent_layout="M1_test7",
        confidence=confidence,
        fast_epe_d=fast,
        oracle_epe_d=oracle,
    )


def test_closed_loop_selects_lowest_refine_rate_feasible_threshold():
    """多个质量可行阈值存在时应优先选择精算率最低者。"""
    samples = [_sample("a", 0.4, 20, 10), _sample("b", 0.8, 10, 10)]
    report = build_closed_loop_report(samples, [0.3, 0.5, 0.9], 0.05, 0.60)
    assert report.status == "feasible"
    assert report.selected is not None
    assert report.selected.threshold == 0.5
    assert report.selected.refine_rate == 0.5
    assert report.meets_refine_rate_acceptance is True


def test_closed_loop_reports_no_feasible_threshold():
    """所有阈值都不满足退化限制时不得人为选择结果。"""
    samples = [_sample("a", 0.9, 20, 10)]
    report = build_closed_loop_report(samples, [0.5, 0.8], 0.05, 0.20)
    assert report.status == "no_feasible_threshold"
    assert report.selected is None
    assert report.meets_refine_rate_acceptance is False


def test_closed_loop_rejects_test_split():
    """调阈值时禁止使用测试集。"""
    with pytest.raises(ValueError, match="validation"):
        ClosedLoopSample(
            sample_id="test-a", parent_layout="M1_test9", split="test",
            confidence=0.8, fast_epe_d=10, oracle_epe_d=9,
        )
