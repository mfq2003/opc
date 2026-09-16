"""本模块定义统一 OpcEngine 契约，并提供不修改上游源码的 OpenILT 进程适配层。

输入为 OpenILT 目录、版图文件及运行目录；输出为上游 SimpleOPC 的原始日志和固定提交号。
关键依赖为标准库 subprocess；真实 GPU 计算由已固定版本的 OpenILT/PyTorch 在独立进程完成。
"""
from __future__ import annotations

import abc
import os
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
    def optimize(
        self,
        layout: Path,
        recipe: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> str:
        """执行优化，并返回可审计的原始输出。"""


class OpenILTEngine(OpcEngine):
    """通过上游入口运行 SimpleOPC，不向 OpenILT 工作树写入兼容补丁。"""

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
        if not (self.openilt_dir / "pyilt" / "simpleopc.py").is_file():
            raise RuntimeError(f"未找到 OpenILT SimpleOPC：{self.openilt_dir}")
        if self.revision() != self.expected_commit:
            raise RuntimeError("OpenILT 提交与配置不一致；拒绝生成不可复现实验")

    def _prepare_official_workspace(self, layout: Path, output_dir: Path) -> Path:
        """以只读符号链接组装官方脚本工作目录，避免写入 OpenILT clone。"""
        iccad13_dir = Path(layout).resolve()
        if not iccad13_dir.is_dir():
            raise FileNotFoundError(f"ICCAD2013 目录不存在：{iccad13_dir}")
        workspace = Path(output_dir).resolve()
        workspace.mkdir(parents=True, exist_ok=False)
        (workspace / "benchmark").mkdir()
        (workspace / "tmp").mkdir()
        links = {
            workspace / "benchmark" / "ICCAD2013": iccad13_dir,
            workspace / "config": (self.openilt_dir / "config").resolve(),
            workspace / "kernel": (self.openilt_dir / "kernel").resolve(),
        }
        for destination, source in links.items():
            if not source.is_dir():
                raise FileNotFoundError(f"OpenILT 官方运行依赖不存在：{source}")
            os.symlink(str(source), str(destination), target_is_directory=True)
        return workspace

    def optimize(
        self,
        layout: Path,
        recipe: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> str:
        """调用上游 SimpleOPC 基线；上游入口处理 ICCAD13 十个公开测试图形。"""
        self._validate_installation()
        if recipe is not None:
            raise NotImplementedError("上游 SimpleOPC 脚本不接受外部 Recipe；多步 Recipe 由本项目环境应用")
        cwd = self.openilt_dir
        command_entry = "pyilt/simpleopc.py"
        environment = None
        if output_dir is not None:
            cwd = self._prepare_official_workspace(layout, output_dir)
            command_entry = str((self.openilt_dir / "pyilt" / "simpleopc.py").resolve())
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            existing = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = str(self.openilt_dir.resolve()) + (os.pathsep + existing if existing else "")
        result = subprocess.run(
            [sys.executable, str(command_entry)], cwd=cwd, timeout=self.timeout_seconds,
            check=True, capture_output=True, text=True, env=environment,
        )
        return result.stdout + result.stderr

    def evaluate(self, layout: Path, recipe: Optional[Path] = None) -> str:
        """使用与优化同一上游基线入口，保留接口以接入逐 clip 评估适配器。"""
        return self.optimize(layout, recipe)
