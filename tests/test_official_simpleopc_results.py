"""本模块验证 OpenILT 官方 SimpleOPC 十图日志和图片证据的严格归档。

输入为合成的官方 Initialized/Step 日志及临时 PNG；输出为十图八轮轨迹、官方最终轮、共同目标
最佳轮和 SHA256。测试不调用 CUDA、OpenILT、网络或真实光刻仿真。
"""
import json
from pathlib import Path

import pytest

from opc_agent.official_simpleopc_results import (
    archive_official_simpleopc_results,
    parse_official_simpleopc_log,
)


def _official_log() -> str:
    """构造与上游 pyilt/simpleopc.py 完全一致的十图日志形状。"""
    lines = []
    for index in range(1, 11):
        lines.append(
            f"[Testcase {index} Initialized]: L2 100; PVBand 200; EPE 3; Shot: -1"
        )
        for step in range(8):
            lines.append(
                f"[Testcase {index} Step {step}]: L2 {100-step}; "
                f"PVBand {200-step}; EPE {3 if step < 4 else 2}; Shot: -1"
            )
    return "\n".join(lines)


def test_parse_official_simpleopc_log_preserves_final_and_reports_common_best():
    """解析器不得用离线最佳轮冒充官方实际保存的最终轮。"""
    cases = parse_official_simpleopc_log(_official_log(), {"l2": 1, "epe": 100, "pvb": 1})
    assert len(cases) == 10
    assert cases[0]["official_final"]["step"] == 7
    assert cases[0]["common_objective_best"]["step"] == 7
    assert cases[0]["initial"]["weighted_loss"] == 600


def test_parse_official_simpleopc_log_rejects_incomplete_steps():
    """任一版图缺失一步时不得生成看似完整的十图基线。"""
    content = _official_log().replace(
        "[Testcase 6 Step 4]: L2 96; PVBand 196; EPE 2; Shot: -1\n", ""
    )
    with pytest.raises(ValueError, match="Step 0 到 7"):
        parse_official_simpleopc_log(content, {"l2": 1, "epe": 100, "pvb": 1})


def test_archive_official_simpleopc_results_hashes_all_thirty_images(tmp_path: Path):
    """归档必须绑定十图的 target、mask、resist 三类非空图片。"""
    run_root = tmp_path / "run"
    execution_root = run_root / "official-simpleopc"
    image_root = execution_root / "tmp"
    image_root.mkdir(parents=True)
    (run_root / "openilt-baseline.log").write_text(_official_log(), encoding="utf-8")
    for index in range(1, 11):
        for kind in ("target", "mask", "resist"):
            (image_root / f"SimpleOPC_{kind}{index}.png").write_bytes(
                f"{kind}-{index}".encode("ascii")
            )
    destination = archive_official_simpleopc_results(
        run_root,
        execution_root,
        "d" * 40,
        {"l2": 1, "epe": 100, "pvb": 1},
        12.5,
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["case_count"] == 10
    assert payload["algorithm"] == "OpenILT pyilt/simpleopc.py unmodified"
    assert len(payload["cases"][0]["images"]["mask"]["sha256"]) == 64
