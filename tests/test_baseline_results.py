"""本模块验证 OpenILT 基线日志的严格解析和离线 SQLite 指标归档。

输入为十图合成日志、临时运行目录和 SQLite 模式；输出为十条父版图指标、汇总 JSON 与非法日志的失败断言。
关键依赖为 pytest、Pydantic 和标准库 SQLite；测试不调用 GPU、OpenILT、网络或 API。
"""
from pathlib import Path

import pytest

from opc_agent.baseline_results import archive_baseline_results, parse_simpleilt_log
from opc_agent.models import LayoutClip
from opc_agent.storage import ExperimentStore


def _valid_log() -> str:
    """构造与固定 OpenILT 提交输出格式相同的十图最小日志。"""
    lines = [f"[Testcase {index}]: L2 {index}; PVBand {index + 1}; EPE {index + 2}; Shot: {index + 3}; SolveTime: 1.0s" for index in range(1, 11)]
    lines.append("[Result]: L2 5.5; PVBand 6.5; EPE 7.5; Shot 8.5; SolveTime 1.0s")
    return "\n".join(lines)


def test_parse_simpleilt_log_returns_ten_cases_and_total_epe_only():
    """上游总 EPE 必须保留，缺失的 EPE N/EPE D 必须为 None。"""
    cases, summary = parse_simpleilt_log(_valid_log())
    assert len(cases) == 10
    assert cases[0].parent_layout == "M1_test1"
    assert cases[0].epe_total == 3
    assert cases[0].epe_n is None and cases[0].epe_d is None
    assert summary.epe_total == 7.5


def test_parse_simpleilt_log_rejects_missing_case():
    """缺少任一测试图形时，归档器不得写入不完整的基线结果。"""
    with pytest.raises(ValueError, match="Testcase 1 到 10"):
        parse_simpleilt_log(_valid_log().replace("[Testcase 6]", "[Testcase 16]"))


def test_archive_baseline_results_writes_json_and_ten_sqlite_rows(tmp_path: Path):
    """离线归档不得重跑 GPU，并应向既有元数据数据库写入十条基线记录。"""
    database = tmp_path / "runs.sqlite3"
    store = ExperimentStore(database)
    store.upsert_run("baseline-run", {}, {})
    for index in range(1, 11):
        store.upsert_clip(LayoutClip(
            clip_id=f"M1_test{index}", source="ICCAD13", parent_layout=f"M1_test{index}", coordinates_nm=(0, 0, 0, 0),
            image_path=f"M1_test{index}.glp", scale_nm_per_pixel=1, file_sha256="a" * 64, split="train",
        ))
    store.close()
    run_root = tmp_path / "baseline-run"
    run_root.mkdir()
    (run_root / "openilt-baseline.log").write_text(_valid_log(), encoding="utf-8")
    archive_path = archive_baseline_results("baseline-run", run_root, database)
    assert archive_path.is_file()
    check_store = ExperimentStore(database)
    assert check_store.count("metrics") == 10
    check_store.close()

