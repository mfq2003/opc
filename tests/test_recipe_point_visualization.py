"""本模块验证当前 Recipe 点级 PPO 的训练前十图和训练后逐 seed 静态可视化。

输入为小型正交 target、EPE/FRAG RecipePoint、临时 stage-result 与 Recipe JSON；输出为 outputs
等价临时目录中的 PNG 和摘要 JSON。测试核对点数、动作位移、路径、文件哈希及错误门槛，不调用
CUDA、OpenILT、PPO、网络或历史 point_visualization 语义。
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from opc_agent.recipe_contract import PPO_RECIPE_LABEL_VERSION, RECIPE_POINT_VERSION
from opc_agent.recipe_point_visualization import (
    render_recipe_point_overview,
    visualize_after_training,
    visualize_before_training,
)


@dataclass(frozen=True)
class _Point:
    """模拟真实 RecipePoint 的无 Gymnasium 纯数据结构。"""

    point_id: str
    task_type: str
    polygon_index: int
    edge_index: int
    anchor_index: int
    base_x: int
    base_y: int
    tangent_x: int
    tangent_y: int
    normal_x: int
    normal_y: int
    lower_delta_nm: float
    upper_delta_nm: float


POINTS = (
    _Point("p0:e0:epe0", "EPE", 0, 0, 0, 50, 30, 1, 0, 0, -1, -40, 40),
    _Point("p0:e1:frag1", "FRAG", 0, 1, 1, 90, 70, 0, 1, 1, 0, -40, 40),
)
POLYGONS = (((30, 30), (110, 30), (110, 90), (30, 90)),)


class _FakeSolver:
    """提供可视化所需的最小只读 solver 协议。"""

    target_polygons = POLYGONS
    recipe_points = POINTS
    nm_per_coordinate = 1.0
    layout_sha256 = "a" * 64


def _config() -> dict:
    """建立固定 6/2/2 十版图配置。"""
    return {
        "data": {
            "openilt_dir": "unused-openilt",
            "train_parents": [f"M1_test{index}" for index in range(1, 7)],
            "validation_parents": ["M1_test7", "M1_test8"],
            "test_parents": ["M1_test9", "M1_test10"],
        }
    }


def _factory(_config_value: dict, _clip_id: str) -> _FakeSolver:
    """返回不依赖 GPU 的固定 solver。"""
    return _FakeSolver()


def test_before_training_visualization_writes_ten_outputs_and_summary(tmp_path: Path, monkeypatch):
    """训练前命令必须写出十张默认打点图，且统一位于 outputs 风格目录。"""
    monkeypatch.setattr(
        "opc_agent.recipe_point_visualization._assert_openilt_clean",
        lambda _path: "dabb97c6ca3dfd159362e48273c436444c77353b",
    )
    output_root = tmp_path / "outputs" / "recipe_point_visualization"
    summary_path = visualize_before_training(_config(), output_root, solver_factory=_factory)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary_path == output_root / "before-training" / "visualization-summary.json"
    assert summary["mode"] == "before-training"
    assert summary["clip_count"] == 10
    assert all(item["point_count"] == 2 for item in summary["clips"])
    assert all(item["nonzero_displacement_count"] == 0 for item in summary["clips"])
    expected = [output_root / "before-training" / f"M1_test{index}-points.png" for index in range(1, 11)]
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)
    image = cv2.imdecode(np.frombuffer(expected[0].read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None and image.shape[0] >= 372 and image.shape[1] >= 360


def test_after_training_visualization_reads_recipe_and_draws_each_seed(tmp_path: Path, monkeypatch):
    """训练后图必须绑定当前 Recipe 几何、位移、运行编号和 seed 目录。"""
    monkeypatch.setattr(
        "opc_agent.recipe_point_visualization._assert_openilt_clean",
        lambda _path: "dabb97c6ca3dfd159362e48273c436444c77353b",
    )
    run_dir = tmp_path / "runs" / "full-run"
    recipe_path = run_dir / "recipes" / "M1_test1-seed-2.recipe.json"
    recipe_path.parent.mkdir(parents=True)
    recipe = {
        "label_version": PPO_RECIPE_LABEL_VERSION,
        "point_version": RECIPE_POINT_VERSION,
        "layout_sha256": "a" * 64,
        "seed": 2,
        "labels": [
            {"point": asdict(POINTS[0]), "ppo_displacement_nm": 40.0},
            {"point": asdict(POINTS[1]), "ppo_displacement_nm": -30.0},
        ],
    }
    recipe_path.write_text(json.dumps(recipe, ensure_ascii=False), encoding="utf-8")
    stage = {
        "environment": "simpleopc-recipe-point-v1",
        "clips": [{"clip_id": "M1_test1", "ppo_recipes": [str(recipe_path)]}],
    }
    (run_dir / "stage-result.json").write_text(json.dumps(stage), encoding="utf-8")
    output_root = tmp_path / "outputs" / "recipe_point_visualization"

    summary_path = visualize_after_training(
        _config(), run_dir, output_root=output_root, solver_factory=_factory,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    image_path = output_root / "after-training" / "full-run" / "seed-2" / "M1_test1-points.png"

    assert summary["mode"] == "after-training"
    assert summary["image_count"] == 1
    assert summary["clips"][0]["nonzero_displacement_count"] == 2
    assert summary["clips"][0]["displacement_class_counts"] == [0, 1, 0, 0, 0, 0, 0, 0, 1]
    assert image_path.is_file() and image_path.stat().st_size > 0


def test_visualization_rejects_non_nine_class_displacement(tmp_path: Path):
    """任意非九分类位移都必须在渲染前失败。"""
    with pytest.raises(ValueError, match="九分类"):
        render_recipe_point_overview(
            POLYGONS,
            POINTS,
            [12.0, 0.0],
            1.0,
            tmp_path / "invalid.png",
            "M1_test1",
            "train",
            "after-training",
            seed=0,
        )
