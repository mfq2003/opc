"""本模块定义统一 OpcEngine 契约，并提供不修改上游源码的 OpenILT 进程适配层。

输入为 OpenILT 目录、版图文件及运行目录；输出为上游 SimpleILT 的原始日志和固定提交号。
关键依赖为标准库 subprocess；真实 GPU 计算由已固定版本的 OpenILT/PyTorch 在独立进程完成。
"""
from __future__ import annotations

import abc
import subprocess
import sys
from pathlib import Path
from typing import Optional


class OpcEngine(abc.ABC):
    """定义所有 OPC 后端必须实现的评估与优化接口。"""

    @abc.abstractmethod
    def evaluate(self, layout: Path, recipe: Optional[Path] = None) -> str:
        """评估给定版图或 Recipe，并返回可审计的原始输出。"""

    @abc.abstractmethod
    def optimize(self, layout: Path, recipe: Optional[Path] = None) -> str:
        """执行优化，并返回可审计的原始输出。"""


class OpenILTEngine(OpcEngine):
    """通过上游入口运行 SimpleILT，不向 OpenILT 工作树写入兼容补丁。"""

    def __init__(self, openilt_dir: Path, expected_commit: str, timeout_seconds: int = 7200):
        self.openilt_dir = Path(openilt_dir)
        self.expected_commit = expected_commit
        self.timeout_seconds = timeout_seconds

    def revision(self) -> str:
        """读取上游工作树 HEAD，用于运行元数据和版本核验。"""
        result = subprocess.run(
            ["git", "-C", str(self.openilt_dir), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    def _validate_installation(self) -> None:
        if not (self.openilt_dir / "pyilt" / "simpleilt.py").is_file():
            raise RuntimeError(f"未找到 OpenILT：{self.openilt_dir}")
        if self.revision() != self.expected_commit:
            raise RuntimeError("OpenILT 提交与配置不一致；拒绝生成不可复现实验")

    def optimize(self, layout: Path, recipe: Optional[Path] = None) -> str:
        """调用上游 SimpleILT 基线；上游入口处理其 ICCAD13 十个公开测试图形。"""
        self._validate_installation()
        if recipe is not None:
            raise NotImplementedError("v1 OpenILT 基线不接受外部 Recipe；Recipe 由 PPO 层应用后再评估")
        result = subprocess.run(
            [sys.executable, "pyilt/simpleilt.py"], cwd=self.openilt_dir, timeout=self.timeout_seconds,
            check=True, capture_output=True, text=True,
        )
        return result.stdout + result.stderr

    def evaluate(self, layout: Path, recipe: Optional[Path] = None) -> str:
        """v1 使用与优化同一上游基线入口，保留接口以接入逐 clip 评估适配器。"""
        return self.optimize(layout, recipe)

