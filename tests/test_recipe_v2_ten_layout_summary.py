"""本模块验证Recipe v2十图汇总的覆盖、协议一致性和宏观/微观统计。

输入为内存构造的十张逐图指标以及临时搜索目录；输出检查十图等权平均、全点汇总、重复版图、
缺图和核心协议漂移的拒绝行为。测试不调用GPU、OpenILT、Solver、网络或API。
"""
import json
import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from opc_agent.recipe_v2_search import _write_binary_png
from opc_agent.recipe_v2_ten_layout_summary import (
    EXPECTED_LAYOUTS,
    _baseline_printed,
    _discover_layouts,
    aggregate_rows,
    write_report,
)


def _row(index):
    """构造一张图的最小可聚合指标。"""
    return {
        "layout": f"M1_test{index}",
        "sample_count": index * 10,
        "baseline": {"pvb": 100 + index, "epe_n": 10 + index, "epe_d_nm": 20 + index},
        "final": {"pvb": 90 + index, "epe_n": 5 + index, "epe_d_nm": 10 + index},
    }


def test_ten_layout_aggregate_reports_macro_mean_and_micro_total():
    """十图宏平均必须每图等权，微观统计必须使用全部采样点作分母。"""
    rows = [_row(index) for index in range(1, 11)]
    result = aggregate_rows(rows)
    assert result["macro_average"]["baseline_mean"]["pvb"] == 105.5
    assert result["macro_average"]["final_mean"]["pvb"] == 95.5
    assert result["macro_average"]["baseline_mean"]["epe_n"] == 15.5
    assert result["micro_total"]["sample_count"] == 550
    assert result["micro_total"]["baseline_epe_n_total"] == 155
    assert result["micro_total"]["final_epe_d_nm_total"] == 155


def test_write_report_csv_contains_mean_and_total(tmp_path):
    """汇总CSV必须同时给出十图等权平均和按全部采样点累加的总量。"""
    rows = []
    for index in range(1, 11):
        row = _row(index)
        row["reduction"] = {
            name: {"relative_percent": 1.0}
            for name in ("pvb", "epe_n", "epe_d_nm")
        }
        row["baseline_artifact"] = {"source": "test"}
        row["final_replay_equal"] = True
        rows.append(row)
    report = {"layouts": rows, **aggregate_rows(rows)}
    _, csv_path = write_report(report, tmp_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["layout"] for row in csv_rows[-2:]] == ["ALL_MEAN", "ALL_TOTAL"]
    assert int(csv_rows[-1]["sample_count"]) == 550
    assert int(csv_rows[-1]["baseline_epe_n"]) == 155


def test_recorded_baseline_artifacts_must_be_complete_and_hash_matched(tmp_path):
    """新搜索已声明的基线图不得在缺失或损坏时被静默重放掩盖。"""
    coordinate = tmp_path / "coordinate"
    coordinate.mkdir()
    image = np.zeros((8, 8), dtype=np.uint8)
    image[2:6, 2:6] = 1
    printed_hash = _write_binary_png(coordinate / "baseline-printed.png", image)
    result = {
        "baseline_artifacts": {
            "mask_path": "baseline-mask.png",
            "mask_sha256": "0" * 64,
            "printed_path": "baseline-printed.png",
            "printed_sha256": printed_hash,
        }
    }
    with pytest.raises(FileNotFoundError, match="baseline mask"):
        _baseline_printed(
            {"coordinate_dir": coordinate}, "M1_test7", result, tmp_path / "output"
        )

    result["baseline_artifacts"]["mask_sha256"] = _write_binary_png(
        coordinate / "baseline-mask.png", image
    )
    (coordinate / "baseline-printed.png").write_bytes(b"damaged")
    with pytest.raises(RuntimeError, match="baseline printed"):
        _baseline_printed(
            {"coordinate_dir": coordinate}, "M1_test7", result, tmp_path / "output"
        )


def test_ten_layout_aggregate_rejects_partial_or_wrong_order():
    """缺图或顺序漂移不能静默产生所谓十图平均。"""
    with pytest.raises(ValueError, match="M1_test1–10"):
        aggregate_rows([_row(index) for index in range(1, 10)])
    rows = [_row(index) for index in range(1, 11)]
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="M1_test1–10"):
        aggregate_rows(rows)


def _write_run(root: Path, layouts, config):
    """写入只供目录发现测试使用的最小搜索运行。"""
    root.mkdir()
    (root / "config.snapshot.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
    )
    for layout in layouts:
        coordinate = root / layout / "seed-0" / "coordinate"
        coordinate.mkdir(parents=True)
        (coordinate / "result.json").write_text(json.dumps({}), encoding="utf-8")


def test_discover_layouts_combines_two_runs_without_overlap(tmp_path):
    """六图旧运行与四图新运行必须唯一合并成十图。"""
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/recipe_ppo_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_run(first, EXPECTED_LAYOUTS[:6], config)
    eval_config = yaml.safe_load(yaml.safe_dump(config))
    eval_config["search"]["scope"] = "validation_test_diagnostic"
    eval_config["search"]["layout_parents"] = list(EXPECTED_LAYOUTS[6:])
    _write_run(second, EXPECTED_LAYOUTS[6:], eval_config)
    discovered = _discover_layouts([first, second], seed=0)
    assert tuple(discovered) == EXPECTED_LAYOUTS


def test_discover_layouts_rejects_duplicate_layout(tmp_path):
    """同一版图存在两个候选来源时必须由用户处理，不能自动挑最新结果。"""
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/recipe_ppo_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_run(first, EXPECTED_LAYOUTS[:6], config)
    _write_run(second, EXPECTED_LAYOUTS[5:], config)
    with pytest.raises(RuntimeError, match="重复"):
        _discover_layouts([first, second], seed=0)


def test_discover_layouts_rejects_core_protocol_drift(tmp_path):
    """两个搜索运行的动作或solver协议漂移时不得合并十图平均。"""
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/recipe_ppo_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_run(first, EXPECTED_LAYOUTS[:6], config)
    drifted = yaml.safe_load(yaml.safe_dump(config))
    drifted["recipe_v2"]["epe_normal_offsets_nm"] = [-10, 0, 10]
    _write_run(second, EXPECTED_LAYOUTS[6:], drifted)
    with pytest.raises(RuntimeError, match="核心协议不一致"):
        _discover_layouts([first, second], seed=0)
