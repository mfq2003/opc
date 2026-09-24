"""本模块封装硅基流动 OpenAI 兼容 API 的图片能力探针和严格 Recipe 解析。

输入为环境变量中的 API 密钥、公开图像路径与模型响应；输出为经 Pydantic 校验的 Recipe 或明确失败。
关键依赖为 openai 客户端和 Pydantic；本模块不读取 .env、不在在线快速路径中调用 API。
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI
from pydantic import ValidationError

from .models import Recipe


class QwenResponseError(ValueError):
    """表示 API 响应不是合法 Recipe，调用方必须记录并停止该样本训练。"""

    def __init__(self, message: str, raw_response: str, stream_diagnostics: Dict[str, Any] = None):
        super().__init__(message)
        self.raw_response = raw_response
        self.response_sha256 = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
        self.stream_diagnostics = stream_diagnostics


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

    def __init__(self, model: str, base_url: str = "https://api.siliconflow.cn/v1", timeout_seconds: float = 300.0):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须是有限正数")
        key = os.environ.get("SILICONFLOW_API_KEY")
        if not key:
            raise RuntimeError("未设置 SILICONFLOW_API_KEY；请仅通过进程环境变量注入")
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=timeout_seconds, max_retries=0)
        self.model = model

    def close(self):
        """关闭复用的 HTTP 客户端，包含中断和认证失败路径。"""
        self.client.close()

    def extract_point_features(self, image_paths, prompt: str) -> str:
        """单次发送同一点的两张图；返回原始文本，特征校验由数据集模块负责。"""
        if len(image_paths) != 2:
            raise ValueError("每个点必须提供局部图和上下文图")
        content = [{"type": "text", "text": prompt}]
        for path in image_paths:
            encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{encoded}", "detail": "high"}})
        self.last_stream_diagnostics = None
        response = self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=1024, stream=True,
            extra_body={"enable_thinking": False},
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": "你是版图特征标注员，只返回合法 JSON。"},
                      {"role": "user", "content": content}],
        )
        pieces = []
        diagnostics = {"chunk_count": 0, "choice_chunk_count": 0,
                       "content_chunk_count": 0, "content_char_count": 0,
                       "reasoning_chunk_count": 0, "reasoning_char_count": 0,
                       "finish_reasons": []}
        try:
            for chunk in response:
                diagnostics["chunk_count"] += 1
                if not chunk.choices:
                    continue
                diagnostics["choice_chunk_count"] += 1
                choice = chunk.choices[0]
                delta = choice.delta
                value = getattr(delta, "content", None)
                reasoning = getattr(delta, "reasoning_content", None)
                if value:
                    pieces.append(value)
                    diagnostics["content_chunk_count"] += 1
                    diagnostics["content_char_count"] += len(value)
                if reasoning:
                    diagnostics["reasoning_chunk_count"] += 1
                    diagnostics["reasoning_char_count"] += len(reasoning)
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason:
                    diagnostics["finish_reasons"].append(str(finish_reason))
        except Exception as exc:
            # 保留异常类型和安全计数，不保存可能包含请求头的异常全文。
            exc.stream_diagnostics = dict(diagnostics)
            exc.raw_response = "".join(pieces)
            raise
        finally:
            self.last_stream_diagnostics = dict(diagnostics)
            close = getattr(response,"close",None)
            if close is not None: close()
        text = "".join(pieces)
        if not text:
            raise QwenResponseError("流式图片请求未返回正文", text, diagnostics)
        return text

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
