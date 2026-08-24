"""本模块定义 OPC Agent 的版本化边界数据模型。

输入为版图切片、指标、Recipe 与模型预测等 Python 数据；输出为经 Pydantic 校验的不可歧义对象。
关键依赖为 Pydantic 1.x，所有模型都携带 schema_version 以支持实验数据演进。
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, validator


SCHEMA_VERSION = "1.0"


class TaskType(str, Enum):
    """定义 Recipe 对应的点级优化任务类型。"""

    EPE = "EPE"
    FRAG = "FRAG"


class LayoutClip(BaseModel):
    """记录一个可追溯版图切片的来源、坐标、图像位置和内容哈希。"""

    schema_version: str = SCHEMA_VERSION
    clip_id: str
    source: str
    parent_layout: str
    coordinates_nm: Tuple[float, float, float, float]
    image_path: str
    scale_nm_per_pixel: float = Field(gt=0)
    file_sha256: str = Field(min_length=64, max_length=64)
    split: Optional[str] = None

    @validator("image_path")
    def _relative_or_absolute_path(cls, value: str) -> str:
        if not value:
            raise ValueError("image_path 不能为空")
        return str(Path(value))


class OpcMetrics(BaseModel):
    """保存一次 OPC 评估的 L2、PVB、EPE N、EPE D 与耗时。"""

    schema_version: str = SCHEMA_VERSION
    l2: float = Field(ge=0)
    pvb: float = Field(ge=0)
    epe_n: int = Field(ge=0)
    epe_d: float = Field(ge=0)
    runtime_seconds: float = Field(ge=0)


class RecipeAction(BaseModel):
    """表示一个 EPE 或 FRAG 点的条件化法线方向位移动作。"""

    schema_version: str = SCHEMA_VERSION
    task_type: TaskType
    condition: str = Field(min_length=1, max_length=500)
    displacement_class: int = Field(ge=0, le=8)
    displacement_nm: float = Field(ge=-40, le=40)


class Recipe(BaseModel):
    """表示由 Qwen 或确定性规则生成且可校验的版本化动作集合。"""

    schema_version: str = SCHEMA_VERSION
    feature_version: str = "geometry-v1"
    actions: List[RecipeAction] = Field(min_items=1)


class AgentPrediction(BaseModel):
    """保存快速模型的预测类别、置信度与是否已经路由到精算层。"""

    schema_version: str = SCHEMA_VERSION
    clip_id: str
    predicted_class: int = Field(ge=0, le=8)
    confidence: float = Field(ge=0, le=1)
    call_refinement: bool
    reason: Optional[str] = None

