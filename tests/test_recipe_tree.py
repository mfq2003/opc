"""本模块验证 EPE/FRAG 双树训练、JSON 树推理、九分类指标和父版图防泄漏。

输入为覆盖九个位移类别的合成 geometry-v1 表格；输出为双树类型、预测类别/置信度和非法划分断言。
关键依赖为 scikit-learn、Pydantic 与 pytest；测试不调用 GPU、OpenILT、Qwen、网络或文件下载。
"""
import pytest

try:
    import sklearn  # noqa: F401
except Exception as exc:
    pytest.skip(f"本地 scikit-learn 二进制环境不可用：{exc}", allow_module_level=True)

from opc_agent.features import FEATURE_NAMES
from opc_agent.models import TaskType
from opc_agent.recipe_tree import PointTrainingDataset, PointTrainingRow, predict_tree, train_both_trees


def _features(category: int):
    """让第一特征唯一编码九分类，其余特征保持确定值。"""
    values = {name: 0.0 for name in FEATURE_NAMES}
    values[FEATURE_NAMES[0]] = category / 10.0
    return values


def _dataset() -> PointTrainingDataset:
    """为两类任务各构造九类训练与测试样本，且父版图不跨集合。"""
    rows = []
    for task in (TaskType.EPE, TaskType.FRAG):
        for split in ("train", "test"):
            for category in range(9):
                rows.append(PointTrainingRow(
                    sample_id=f"{task.value}-{split}-{category}",
                    clip_id=f"{task.value}-{split}",
                    parent_layout=f"parent-{task.value}-{split}",
                    split=split,
                    task_type=task,
                    features=_features(category),
                    displacement_class=category,
                ))
    return PointTrainingDataset(rows=rows)


def test_train_both_trees_and_predict_exported_json_tree():
    """两棵树必须独立训练，导出树应能恢复类别和有效置信度。"""
    result = train_both_trees(_dataset(), seed=42)
    assert result.epe_tree.task_type == TaskType.EPE
    assert result.frag_tree.task_type == TaskType.FRAG
    predicted, confidence = predict_tree(result.epe_tree, _features(7))
    assert predicted == 7
    assert confidence == 1.0
    assert result.epe_tree.macro_f1_all_nine_classes == 1.0
    assert {action.task_type for action in result.recipe.actions} == {TaskType.EPE, TaskType.FRAG}


def test_training_rejects_parent_layout_cross_split_leakage():
    """同一父版图出现在训练和测试集合时必须失败。"""
    dataset = _dataset()
    dataset.rows[-1].parent_layout = dataset.rows[0].parent_layout
    with pytest.raises(ValueError, match="跨数据集泄漏"):
        train_both_trees(dataset)

