"""本模块封装硅基流动 OpenAI 兼容 API 的图片能力探针和严格 Recipe 解析。

输入为环境变量中的 API 密钥、公开图像路径与模型响应；输出为经 Pydantic 校验的 Recipe 或明确失败。
关键依赖为 openai 客户端和 Pydantic；本模块不读取 .env、不在在线快速路径中调用 API。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI
from pydantic import ValidationError

from .models import Recipe


class QwenResponseError(ValueError):
    """表示 API 响应不是合法 Recipe，调用方必须记录并停止该样本训练。"""

    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response
        self.response_sha256 = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()


def parse_recipe_response(content: str) -> Recipe:
    """去除可选 Markdown 围栏后解析 JSON，任何结构错误都转换为 QwenResponseError。"""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    try:
        payload = json.loads(cleaned)
        return Recipe.parse_obj(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise QwenResponseError(f"Qwen Recipe 校验失败：{exc}", content) from exc


class SiliconFlowQwen:
    """仅供离线特征/Recipe 阶段使用的硅基流动 Qwen 客户端。"""

    def __init__(self, model: str, base_url: str = "https://api.siliconflow.cn/v1"):
        key = os.environ.get("SILICONFLOW_API_KEY")
        if not key:
            raise RuntimeError("未设置 SILICONFLOW_API_KEY；请仅通过进程环境变量注入")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model

    def probe_image_capability(self, image_path: Path) -> Dict[str, Any]:
        """发送一张公开图像并要求 JSON，供部署时显式确认视觉与结构化输出能力。"""
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "只返回 JSON。"},
                {"role": "user", "content": [
                    {"type": "text", "text": "读取此公开版图图像，并返回 {\\\"visual_ok\\\": true, \\\"shape\\\": \\\"...\\\"}。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                ]},
            ],
        )
        content = response.choices[0].message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise QwenResponseError("图片能力探针未返回 JSON", content) from exc

