"""本模块验证 OPC Agent 版本化 Pydantic 模型的字段边界和 Recipe 约束。

输入为构造的模型参数，输出为 pytest 对合法数据和非法九分类动作的一致断言。
关键依赖为 pytest 与 Pydantic，测试无需 GPU、OpenILT 或网络。
"""
import pytest
from pydantic import ValidationError

from opc_agent.models import RecipeAction, TaskType


def test_recipe_action_rejects_out_of_range_class():
    """九分类动作不得接受第十个类别。"""
    with pytest.raises(ValidationError):
        RecipeAction(task_type=TaskType.EPE, condition="corner", displacement_class=9, displacement_nm=40)


def test_recipe_action_accepts_limit_displacement():
    """论文允许的 -40nm 边界应保持合法。"""
    action = RecipeAction(task_type=TaskType.FRAG, condition="line-end", displacement_class=0, displacement_nm=-40)
    assert action.displacement_nm == -40

