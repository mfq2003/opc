"""本模块验证统一 OpenILT 基线入口已从 SimpleILT 切换到 SimpleOPC 且不接收伪 Recipe。

输入为临时上游目录和替身 subprocess；输出为固定提交校验、`pyilt/simpleopc.py` 命令及缺失入口失败
断言。关键依赖为 pytest 与标准库；测试不调用 GPU、OpenILT、网络或真实子进程。
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

import opc_agent.engine as engine_module
from opc_agent.engine import OpenILTEngine


def test_openilt_engine_launches_simpleopc_entrypoint(tmp_path: Path, monkeypatch):
    """基线适配器必须运行 SimpleOPC，不能再回退到 SimpleILT。"""
    openilt = tmp_path / "OpenILT"
    entry = openilt / "pyilt" / "simpleopc.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("# upstream", encoding="utf-8")
    commands = []

    def _run(command, **kwargs):
        commands.append((command, kwargs))
        if command[:3] == ["git", "-C", str(openilt)]:
            return SimpleNamespace(stdout="fixed\n")
        return SimpleNamespace(stdout="simpleopc-output", stderr="")

    monkeypatch.setattr(engine_module.subprocess, "run", _run)
    backend = OpenILTEngine(openilt, "fixed")
    assert backend.optimize(tmp_path / "ICCAD2013") == "simpleopc-output"
    assert commands[-1][0][1] == "pyilt/simpleopc.py"
    assert commands[-1][1]["cwd"] == openilt


def test_openilt_engine_rejects_missing_simpleopc(tmp_path: Path, monkeypatch):
    """只有 SimpleILT 而没有 SimpleOPC 时必须在启动前失败。"""
    openilt = tmp_path / "OpenILT"
    old = openilt / "pyilt" / "simpleilt.py"
    old.parent.mkdir(parents=True)
    old.write_text("# legacy", encoding="utf-8")
    monkeypatch.setattr(
        engine_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="fixed\n"),
    )
    with pytest.raises(RuntimeError, match="SimpleOPC"):
        OpenILTEngine(openilt, "fixed").optimize(tmp_path / "ICCAD2013")
