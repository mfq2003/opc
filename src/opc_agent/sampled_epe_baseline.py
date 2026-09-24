"""本模块计算全零基线 Recipe 在 1 nm 阈值下的采样 EPE N 与 EPE D。

计算过程复用正式结果的几何重建与采样逻辑，确保基线和优化后的 Recipe 使用同一组点，
便于逐版图核对基线指标。

This reuses the exact same reconstruction + sampling logic as
``sampled_epe_metrics.py`` so that baseline and final numbers are
directly comparable.

Prerequisite on the cloud runner:
    The script expects a ``baseline-printed.png`` next to ``target.png`` in
    each layout's output directory.  To produce it, run the solver once
    with the all-stay recipe (every point normal_offset=0, FRAG=(16,32))
    and save the printed image as ``baseline-printed.png``.

Usage (from repo root, after baseline-printed.png exists in each
``<run>/<layout>/seed-0/coordinate/`` dir):

    python -m opc_agent.sampled_epe_baseline \
        --run-root runs/20260915T014128Z-v2-search-5697a08c
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import xml.etree.ElementTree as ET


def _load_binary(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return (np.array(img) >= 128).astype(np.uint8)


def _rebuild_v2_points(result_json: Path, img_w: int, img_h: int):
    with open(result_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = data.get("config", {})
    frag = cfg.get("fragmentation", {"start": 16, "end": 32})
    edge_split_nm = cfg.get("edge_split_nm", 10.0)
    nm_per_coord = 1.0
    root = ET.parse(str(result_json.parent / "target.gds.xml")).getroot()
    layers = [(int(l.get("layer")), int(l.get("datatype"))) for l in root.iter("layer")]
    paths = []
    for layer, datatype in layers:
        for p in root.iter("path"):
            if int(p.get("layer")) != layer:
                continue
            d = p.find("data")
            if d is None or d.text is None:
                continue
            pts = [list(map(float, pt.split(":"))) for pt in d.text.strip().split()]
            if len(pts) < 2:
                continue
            paths.append((layer, datatype, pts))
    all_eps, all_frags = [], []
    for layer, datatype, pts in paths:
        ep_start, ep_end = frag["start"], frag["end"]
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            dx, dy = x2 - x1, y2 - y1
            length = (dx * dx + dy * dy) ** 0.5
            if length < 1e-9:
                continue
            nx, ny = -dy / length, dx / length
            n_sp = max(0, int(length // edge_split_nm))
            for k in range(n_sp):
                t = (k + 0.5) / n_sp if n_sp else 0.5
                px, py = x1 + t * dx, y1 + t * dy
                for off in (ep_start, ep_end):
                    all_eps.append((layer, datatype, px + off * nx, py + off * ny))
            for dist in range(int(ep_start), int(length) + 1, int(ep_end)):
                t = dist / length
                all_frags.append((layer, datatype, x1 + t * dx, y1 + t * dy))
    valid = [
        (x, y)
        for x, y in [(px, py) for _, _, px, py in all_eps + all_frags]
        if 0 <= x < img_w and 0 <= y < img_h
    ]
    return np.array(valid, dtype=np.float64)


def _foreground_boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = np.zeros_like(padded)
    for dy in range(3):
        for dx in range(3):
            eroded |= padded[dy:dy + padded.shape[0] - 2, dx:dx + padded.shape[1] - 2] & kernel[dy, dx]
    eroded = eroded[1:-1, 1:-1]
    return ((mask == 1) & (eroded == 0)).astype(np.uint8)


def _compute_layout(target_bin: np.ndarray, printed_bin: np.ndarray, sample_xy: np.ndarray,
                    threshold_nm: float, nm_per_pixel: float = 1.0):
    fg = _foreground_boundary(printed_bin)
    dist_to_bg = distance_transform_edt(1 - fg) * nm_per_pixel
    h, w = target_bin.shape
    out_of_bounds = np.zeros(len(sample_xy), dtype=bool)
    for i, (x, y) in enumerate(sample_xy):
        xi, yi = int(round(x)), int(round(y))
        out_of_bounds[i] = not (0 <= xi < w and 0 <= yi < h)
    safe = ~out_of_bounds
    xi = np.clip(np.round(sample_xy[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(sample_xy[:, 1]).astype(int), 0, h - 1)
    inside_target = target_bin[yi, xi].astype(bool)
    dist = dist_to_bg[yi, xi]
    dist[inside_target] = 0.0
    dist[out_of_bounds] = np.nan
    valid = ~np.isnan(dist)
    violations = valid & (dist > threshold_nm)
    n_violations = int(violations.sum())
    d_sum = float(dist[violations].sum())
    mean_viol = float(dist[violations].mean()) if n_violations else 0.0
    max_viol = float(dist[valid].max()) if valid.any() else 0.0
    return {
        "sample_count": int(valid.sum()),
        "out_of_bounds_count": int(out_of_bounds.sum()),
        "epe_n": n_violations,
        "epe_d_nm": round(d_sum, 2),
        "mean_violation_distance_nm": round(mean_viol, 2),
        "max_distance_nm": round(max_viol, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--threshold-nm", type=float, default=1.0)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    layouts = sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("M1_")])

    results = []
    agg_n, agg_d, agg_s = 0, 0.0, 0
    for layout_dir in layouts:
        coord_dir = layout_dir / "seed-0" / "coordinate"
        target_p = coord_dir / "target.png"
        baseline_printed = coord_dir / "baseline-printed.png"
        result_p = coord_dir / "result.json"
        if not baseline_printed.exists():
            print(f"SKIP {layout_dir.name}: baseline-printed.png not found")
            continue
        target = _load_binary(target_p)
        baseline = _load_binary(baseline_printed)
        pts = _rebuild_v2_points(result_p, target.shape[1], target.shape[0])
        m = _compute_layout(target, baseline, pts, args.threshold_nm)
        results.append({"layout": layout_dir.name, "metrics": m})
        agg_n += m["epe_n"]
        agg_d += m["epe_d_nm"]
        agg_s += m["sample_count"]
        print(f"{layout_dir.name}: EPE_N={m['epe_n']}  EPE_D={m['epe_d_nm']:.1f}nm  "
              f"points={m['sample_count']}")

    print()
    print(f"AGGREGATE ({len(results)} layouts): EPE_N={agg_n}  EPE_D={agg_d:.1f}nm  "
          f"EPE_N/图={agg_n/len(results):.1f}  EPE_D/图={agg_d/len(results):.1f}nm")


if __name__ == "__main__":
    main()
