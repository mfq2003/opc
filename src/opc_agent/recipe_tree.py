"""本模块把 PPO Oracle 点级真值训练成 EPE/FRAG 两棵独立决策树并导出可审计 JSON Recipe。

输入为包含父版图划分、geometry-v1 特征和九分类位移标签的 JSON 数据集；输出为两份版本化树结构、
测试集 macro-F1、训练数据哈希和确定性 Recipe 路径。关键依赖为 Pydantic、NumPy 与 scikit-learn；
模块不调用 Qwen、OpenILT 或 GPU，Qwen 后续只能解释这些已验证路径，不能修改类别和阈值。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field, validator



from .features import FEATURE_NAMES
from .metrics import DISPLACEMENT_CLASSES_NM
from .models import Recipe, RecipeAction, TaskType


class PointTrainingRow(BaseModel):
    """保存一个 EPE 或 FRAG 点的来源、固定划分、几何特征和 Oracle 类别。"""

    schema_version: str = "1.0"
    sample_id: str
    clip_id: str
    parent_layout: str
    split: str
    task_type: TaskType
    features: Dict[str, float]
    displacement_class: int = Field(ge=0, le=8)
    optimal_classes: List[int] = Field(default_factory=list)
    loss_margin: Optional[float] = Field(default=None, ge=0)
    ambiguous: bool = False
    candidate_collision: Optional[bool] = None

    @validator("split")
    def _valid_split(cls, value: str) -> str:
        if value not in {"train", "validation", "test"}:
            raise ValueError("split 必须是 train、validation 或 test")
        return value

    @validator("optimal_classes")
    def _valid_optimal_classes(cls, value: List[int]) -> List[int]:
        if any(item < 0 or item > 8 for item in value):
            raise ValueError("optimal_classes 只能包含 0 到 8")
        if len(value) != len(set(value)):
            raise ValueError("optimal_classes 不能重复")
        return sorted(value)

    @validator("features")
    def _valid_features(cls, value: Dict[str, float]) -> Dict[str, float]:
        if set(value) != set(FEATURE_NAMES):
            raise ValueError(f"features 必须恰好包含 {list(FEATURE_NAMES)}")
        if not np.isfinite([value[name] for name in FEATURE_NAMES]).all():
            raise ValueError("features 含 NaN 或无穷值")
        return value


class PointTrainingDataset(BaseModel):
    """保存无重复样本且父版图不跨集合的点级监督数据集。"""

    schema_version: str = "1.0"
    feature_version: str = "geometry-v1"
    label_version: str = "oracle-weighted-loss-v1"
    rows: List[PointTrainingRow]


class TreeNode(BaseModel):
    """保存一个可独立推理的决策树节点。"""

    node_id: int = Field(ge=0)
    left: int
    right: int
    feature_index: int
    threshold: float
    class_counts: List[float]


class TreeArtifact(BaseModel):
    """保存树结构、类别映射、数据哈希和测试质量。"""

    schema_version: str = "1.0"
    feature_version: str = "geometry-v1"
    task_type: TaskType
    feature_names: List[str]
    classes: List[int]
    nodes: List[TreeNode]
    train_rows: int = Field(ge=1)
    validation_rows: int = Field(ge=0)
    test_rows: int = Field(ge=1)
    macro_f1_all_nine_classes: float = Field(ge=0, le=1)
    data_sha256: str = Field(min_length=64, max_length=64)
    random_seed: int


class TreeTrainingResult(BaseModel):
    """聚合 EPE/FRAG 树和最终确定性 Recipe。"""

    schema_version: str = "1.0"
    epe_tree: TreeArtifact
    frag_tree: TreeArtifact
    recipe: Recipe


def validate_training_dataset(dataset: PointTrainingDataset) -> None:
    """拒绝重复 sample_id 和同一父版图跨集合的数据泄漏。"""
    if not dataset.rows:
        raise ValueError("点级训练数据集不能为空")
    sample_ids = [row.sample_id for row in dataset.rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("点级训练数据存在重复 sample_id")
    parent_splits: Dict[str, str] = {}
    for row in dataset.rows:
        previous = parent_splits.setdefault(row.parent_layout, row.split)
        if previous != row.split:
            raise ValueError(f"父版图 {row.parent_layout} 跨数据集泄漏")


def _rows_hash(rows: Sequence[PointTrainingRow]) -> str:
    """按 sample_id 排序后计算稳定数据版本哈希。"""
    payload = [row.dict() for row in sorted(rows, key=lambda item: item.sample_id)]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _matrix(rows: Sequence[PointTrainingRow]) -> Tuple[np.ndarray, np.ndarray]:
    """按固定特征顺序转换为 sklearn 输入矩阵和标签。"""
    x = np.asarray([[row.features[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float64)
    y = np.asarray([row.displacement_class for row in rows], dtype=np.int64)
    return x, y


def _export_tree(
    classifier: DecisionTreeClassifier,
    task_type: TaskType,
    rows: Sequence[PointTrainingRow],
    split_counts: Dict[str, int],
    macro_f1: float,
    seed: int,
) -> TreeArtifact:
    """把 sklearn 内部数组转成不依赖 pickle 的版本化节点列表。"""
    tree = classifier.tree_
    nodes = [
        TreeNode(
            node_id=index,
            left=int(tree.children_left[index]),
            right=int(tree.children_right[index]),
            feature_index=int(tree.feature[index]),
            threshold=float(tree.threshold[index]),
            class_counts=[float(value) for value in tree.value[index][0]],
        )
        for index in range(tree.node_count)
    ]
    return TreeArtifact(
        task_type=task_type,
        feature_names=list(FEATURE_NAMES),
        classes=[int(value) for value in classifier.classes_],
        nodes=nodes,
        train_rows=split_counts["train"],
        validation_rows=split_counts["validation"],
        test_rows=split_counts["test"],
        macro_f1_all_nine_classes=macro_f1,
        data_sha256=_rows_hash(rows),
        random_seed=seed,
    )


def train_task_tree(
    dataset: PointTrainingDataset,
    task_type: TaskType,
    seed: int = 42,
    max_depth: Optional[int] = None,
) -> TreeArtifact:
    """只用 train 拟合一类任务，并在独立 test 上按全部九类计算 macro-F1。"""
    
    

    from sklearn.metrics import f1_score
    from sklearn.tree import DecisionTreeClassifier

    validate_training_dataset(dataset)
    rows = [row for row in dataset.rows if row.task_type == task_type]
    split_rows = {split: [row for row in rows if row.split == split] for split in ("train", "validation", "test")}
    if not split_rows["train"] or not split_rows["test"]:
        raise ValueError(f"{task_type.value} 必须同时具有 train 和 test 数据")
    train_x, train_y = _matrix(split_rows["train"])
    test_x, test_y = _matrix(split_rows["test"])
    classifier = DecisionTreeClassifier(
        random_state=int(seed),
        max_depth=max_depth,
        class_weight="balanced",
    )
    classifier.fit(train_x, train_y)
    predictions = classifier.predict(test_x)
    macro_f1 = float(f1_score(test_y, predictions, labels=list(range(9)), average="macro", zero_division=0))
    counts = {split: len(values) for split, values in split_rows.items()}
    return _export_tree(classifier, task_type, rows, counts, macro_f1, int(seed))


def predict_tree(artifact: TreeArtifact, features: Dict[str, float]) -> Tuple[int, float]:
    """直接遍历 JSON 树，返回类别和叶节点概率，不依赖 sklearn 模型文件。"""
    if set(features) != set(artifact.feature_names):
        raise ValueError("预测特征字段与树版本不一致")
    node = artifact.nodes[0]
    while node.left != node.right:
        value = float(features[artifact.feature_names[node.feature_index]])
        node = artifact.nodes[node.left if value <= node.threshold else node.right]
    counts = np.asarray(node.class_counts, dtype=np.float64)
    if counts.sum() <= 0:
        raise RuntimeError("决策树叶节点没有有效样本计数")
    local_index = int(np.argmax(counts))
    return artifact.classes[local_index], float(counts[local_index] / counts.sum())


def _leaf_paths(artifact: TreeArtifact) -> List[RecipeAction]:
    """把根到叶的判断路径转换为确定性 RecipeAction。"""
    actions: List[RecipeAction] = []

    def visit(node_id: int, conditions: List[str]) -> None:
        node = artifact.nodes[node_id]
        if node.left == node.right:
            counts = np.asarray(node.class_counts, dtype=np.float64)
            local_index = int(np.argmax(counts))
            category = artifact.classes[local_index]
            actions.append(RecipeAction(
                task_type=artifact.task_type,
                condition=" and ".join(conditions) if conditions else "always",
                displacement_class=category,
                displacement_nm=float(DISPLACEMENT_CLASSES_NM[category]),
            ))
            return
        feature = artifact.feature_names[node.feature_index]
        visit(node.left, conditions + [f"{feature} <= {node.threshold:.12g}"])
        visit(node.right, conditions + [f"{feature} > {node.threshold:.12g}"])

    visit(0, [])
    return actions


def train_both_trees(
    dataset: PointTrainingDataset,
    seed: int = 42,
    max_depth: Optional[int] = None,
) -> TreeTrainingResult:
    """训练 EPE/FRAG 两棵独立树，并合并为通过 Pydantic 校验的 Recipe。"""
    epe_tree = train_task_tree(dataset, TaskType.EPE, seed=seed, max_depth=max_depth)
    frag_tree = train_task_tree(dataset, TaskType.FRAG, seed=seed, max_depth=max_depth)
    recipe = Recipe(actions=_leaf_paths(epe_tree) + _leaf_paths(frag_tree))
    return TreeTrainingResult(epe_tree=epe_tree, frag_tree=frag_tree, recipe=recipe)


def _write_versioned_json(path: Path, payload: dict) -> None:
    """禁止用不同内容覆盖旧模型版本；相同内容重复执行保持幂等。"""
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"拒绝覆盖已有不同版本：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(encoded, encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    """训练两棵树并把模型、指标和 Recipe 写入新的版本目录。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.recipe_tree")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int)
    args = parser.parse_args(argv)
    dataset = PointTrainingDataset.parse_obj(json.loads(args.dataset.read_text(encoding="utf-8")))
    result = train_both_trees(dataset, seed=args.seed, max_depth=args.max_depth)
    _write_versioned_json(args.output_dir / "epe.tree.json", result.epe_tree.dict())
    _write_versioned_json(args.output_dir / "frag.tree.json", result.frag_tree.dict())
    _write_versioned_json(args.output_dir / "recipe.raw.json", result.recipe.dict())
    metrics = {
        "epe_macro_f1_all_nine_classes": result.epe_tree.macro_f1_all_nine_classes,
        "frag_macro_f1_all_nine_classes": result.frag_tree.macro_f1_all_nine_classes,
        "epe_data_sha256": result.epe_tree.data_sha256,
        "frag_data_sha256": result.frag_tree.data_sha256,
    }
    _write_versioned_json(args.output_dir / "tree.metrics.json", metrics)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



