"""本模块把已 accepted 的 SimpleOPC PPO 最优 Recipe 转为决策树点级标签。

输入为完整 PPO stage-result、只用 validation 判定的质量报告及其中选中的模型 Recipe；输出为
simpleopc-segment-v1 的 EPE 点级数据集。关键依赖为 Pydantic 与 JSON/哈希；模块不会从九动作
Oracle 推导标签，也不会伪造尚未实现的 FRAG 标签，因此该 EPE-only 产物仍不能启动双树训练。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Optional

from .features import feature_names_for_version
from .metrics import DISPLACEMENT_CLASSES_NM, SIMPLEOPC_LOSS_VERSION
from .models import TaskType
from .recipe_tree import PointTrainingDataset, PointTrainingRow, validate_training_dataset


def _read_json(path: Path) -> dict:
    """读取 JSON 对象并拒绝不存在或错误根类型。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"PPO Recipe 标签输入不存在：{source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"PPO Recipe 标签输入不是 JSON 对象：{source}")
    return payload


def _verify_quality_report(report: dict) -> str:
    """复核 accepted 报告自身哈希，避免人工改状态后绕过门槛。"""
    if report.get("status") != "accepted" or report.get("tree_training_allowed") is not True:
        raise RuntimeError("PPO 质量报告未 accepted，禁止生成决策树标签")
    recorded = report.get("report_sha256")
    identity = dict(report)
    identity.pop("report_sha256", None)
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if recorded != actual:
        raise ValueError("PPO 质量报告哈希不一致")
    return actual


def build_epe_labels_from_ppo(stage_path: Path, quality_path: Path) -> PointTrainingDataset:
    """只读取质量报告为每个 clip 选定的最佳 PPO Recipe，生成 EPE 教师标签。"""
    stage = _read_json(stage_path)
    quality = _read_json(quality_path)
    quality_hash = _verify_quality_report(quality)
    if quality.get("loss_version") != SIMPLEOPC_LOSS_VERSION:
        raise ValueError("决策树标签要求与 PPO 一致的质量报告损失版本")
    if stage.get("environment") != "simpleopc-multistep-v3" or stage.get("mode") != "full":
        raise ValueError("决策树标签要求完整 simpleopc-multistep-v3 运行")
    if stage.get("loss_version") != SIMPLEOPC_LOSS_VERSION:
        raise ValueError("决策树标签要求训练/验收一致的损失版本")
    stage_clips = {item["clip_id"]: item for item in stage.get("clips", [])}
    rows = []
    expected_features = set(feature_names_for_version("simpleopc-segment-v1"))
    for selected in quality.get("clips", []):
        clip = stage_clips.get(selected["clip_id"])
        if clip is None:
            raise ValueError(f"质量报告 clip 不在 PPO stage 中：{selected['clip_id']}")
        recipe_path = Path(selected["best_recipe_path"])
        if str(recipe_path) not in set(clip.get("ppo_recipes", [])):
            raise ValueError(f"质量报告选择了 stage 外的 Recipe：{recipe_path}")
        recipe = _read_json(recipe_path)
        if recipe.get("label_version") != "ppo-simpleopc-multistep-v3":
            raise ValueError(f"不是 PPO SimpleOPC Recipe：{recipe_path}")
        if recipe.get("loss_version") != SIMPLEOPC_LOSS_VERSION:
            raise ValueError(f"PPO Recipe 损失版本不一致：{recipe_path}")
        if tuple(recipe.get("displacement_classes_nm", [])) != DISPLACEMENT_CLASSES_NM:
            raise ValueError(f"PPO Recipe 的九分类位移代表值不一致：{recipe_path}")
        if recipe.get("layout_sha256") != clip.get("layout_sha256"):
            raise ValueError(f"Recipe 与 stage 版图哈希不一致：{recipe_path}")
        for label in recipe.get("labels", []):
            if label.get("task_type") != "EPE":
                raise ValueError("当前转换器只接受 EPE PPO 标签")
            features = {name: float(value) for name, value in label.get("features", {}).items()}
            if set(features) != expected_features:
                raise ValueError("PPO Recipe 缺少 simpleopc-segment-v1 特征")
            segment = label["segment"]
            displacement_class = int(label["displacement_class"])
            exact_displacement = float(label["ppo_displacement_nm"])
            if not 0 <= displacement_class < len(DISPLACEMENT_CLASSES_NM):
                raise ValueError("PPO Recipe 位移类别越界")
            if exact_displacement != float(DISPLACEMENT_CLASSES_NM[displacement_class]):
                raise ValueError("PPO Recipe 精确位移与九分类代表值不一致")
            if float(label["quantization_error_nm"]) != 0.0:
                raise ValueError("PPO Recipe 不允许携带非零位移量化误差")
            rows.append(PointTrainingRow(
                schema_version="3.0",
                sample_id=f"{clip['clip_id']}:{segment['segment_id']}",
                clip_id=clip["clip_id"],
                parent_layout=clip["clip_id"],
                split=clip["split"],
                task_type=TaskType.EPE,
                features=features,
                displacement_class=displacement_class,
                optimal_classes=[displacement_class],
                ambiguous=False,
                candidate_collision=False,
                ppo_displacement_nm=exact_displacement,
                quantization_error_nm=float(label["quantization_error_nm"]),
                ppo_model_sha256=str(recipe["model_sha256"]),
            ))
    result = PointTrainingDataset(
        schema_version="3.0",
        feature_version="simpleopc-segment-v1",
        label_version="ppo-simpleopc-multistep-v3-epe-only",
        ppo_quality_status="accepted",
        ppo_quality_report_sha256=quality_hash,
        rows=rows,
    )
    validate_training_dataset(result)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    """从命令行生成不可静默覆盖的 EPE-only PPO 决策树标签。"""
    parser = argparse.ArgumentParser(prog="python -m opc_agent.ppo_recipe_labels")
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    dataset = build_epe_labels_from_ppo(args.stage, args.quality)
    encoded = json.dumps(dataset.dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"拒绝覆盖已有不同 PPO 标签：{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        args.output.write_text(encoded, encoding="utf-8")
    print(args.output)
    print(f"rows={len(dataset.rows)} task=EPE frag=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
