"""本模块验证 SQLite 幂等性及 Qwen JSON 解析失败的显式错误语义。

输入为临时数据库、版本化模型和合法或非法 JSON，输出为不重复写入和可审计的异常断言。
关键依赖为 pytest、Pydantic 与标准库 SQLite，无需 API 密钥或网络。
"""
from pathlib import Path

import pytest

from opc_agent.models import LayoutClip, TaskType
from opc_agent.qwen import QwenResponseError, parse_recipe_response
from opc_agent.storage import ExperimentStore


def test_sqlite_upsert_is_idempotent(tmp_path: Path):
    """相同 run 和 clip 重复写入后只能保留一条记录。"""
    store = ExperimentStore(tmp_path / "runs.sqlite3")
    store.upsert_run("r1", {"seed": 42}, {"host": "test"})
    store.upsert_run("r1", {"seed": 42}, {"host": "test"})
    clip = LayoutClip(
        clip_id="c1", source="synthetic", parent_layout="p1", coordinates_nm=(0, 0, 1, 1),
        image_path="clip.png", scale_nm_per_pixel=1, file_sha256="a" * 64, split="train",
    )
    store.upsert_clip(clip)
    store.upsert_clip(clip)
    assert store.count("runs") == 1
    assert store.count("clips") == 1
    store.close()


def test_qwen_recipe_requires_valid_json_schema():
    """不完整 Recipe 不能静默进入训练。"""
    with pytest.raises(QwenResponseError) as error:
        parse_recipe_response('{"actions": [{"task_type": "EPE"}]}')
    assert len(error.value.response_sha256) == 64
    recipe = parse_recipe_response(
        '{"actions": [{"task_type": "EPE", "condition": "corner", "displacement_class": 4, "displacement_nm": 0}]}'
    )
    assert recipe.actions[0].task_type is TaskType.EPE

