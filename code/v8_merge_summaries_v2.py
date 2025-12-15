#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v8_merge_summaries_v2.py

Merge multiple v8_*_summary.csv (and optionally v8_*_summary.json) files into:
  - v8_theta_grid_merged.csv
  - v8_theta_grid_table.tex
  - v8_theta_grid_figs.tex (optional LaTeX figure include snippet)

Fixes common Windows pain:
  - you don't have to run from the exact directory
  - missing files won't crash; you'll get warnings instead
  - discovers files via glob

Usage examples (Windows):
  python v8_merge_summaries_v2.py --in-dir .
  python v8_merge_summaries_v2.py --in-dir C:\\rh\\reports
  python v8_merge_summaries_v2.py --in-dir . --glob "v8_*_summary.csv"

Outputs are written to --out-dir (default: current directory).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

DEFAULT_COLS = [
    "theta",
    "kappa",
    "n",
    "mean",
    "std",
    "skew",
    "kurt_excess",
    "ks_D",
    "ks_p",
    "ac1",
    "q_min",
    "q_max",
    "z_min",
    "z_max",
    "exp",
    "npz_path",
]

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None

def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return {}

def read_summary_csv(path: Path) -> Dict[str, Any]:
    """
    Reads a summary CSV with a header row and a single data row.
    Returns dict of that row.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return {}
        # If multiple rows exist, keep the last (most recent write).
        return rows[-1]

def infer_exp_from_name(name: str) -> Optional[str]:
    # Expected: v8_<exp>_summary.csv or v8_<exp>_summary.json
    m = re.search(r"v8_([0-9a-fA-F]{8,})_summary", name)
    return m.group(1) if m else None

def collect_one(prefix_path: Path) -> Tuple[Dict[str, Any], List[str]]:
    """
    prefix_path is like .../v8_<exp>
    We'll look for:
      v8_<exp>_summary.csv (required)
      v8_<exp>_summary.json (optional; used to fill theta/kappa if missing)
    """
    warnings: List[str] = []
    csv_path = prefix_path.with_name(prefix_path.name + "_summary.csv")
    json_path = prefix_path.with_name(prefix_path.name + "_summary.json")

    row: Dict[str, Any] = {}
    if csv_path.exists():
        row = read_summary_csv(csv_path)
    else:
        warnings.append(f"Missing CSV: {csv_path}")
        return {}, warnings

    meta: Dict[str, Any] = {}
    if json_path.exists():
        meta = read_json(json_path)
    else:
        warnings.append(f"Missing JSON (ok): {json_path}")

    # Normalize / coerce
    out: Dict[str, Any] = {}
    out["exp"] = row.get("exp") or meta.get("exp") or infer_exp_from_name(csv_path.name) or ""
    out["npz_path"] = row.get("npz_path") or meta.get("npz_path") or ""

    # theta/kappa sometimes are only in json; try both.
    out["theta"] = safe_float(row.get("theta") or meta.get("theta"))
    out["kappa"] = safe_float(row.get("kappa") or meta.get("kappa"))

    out["n"] = safe_int(row.get("n") or meta.get("n"))
    out["mean"] = safe_float(row.get("mean") or meta.get("mean"))
    out["std"] = safe_float(row.get("std") or meta.get("std"))
    out["skew"] = safe_float(row.get("skew") or meta.get("skew"))
    out["kurt_excess"] = safe_float(row.get("kurt_excess") or meta.get("kurt_excess"))
    out["ks_D"] = safe_float(row.get("ks_D") or meta.get("ks_D"))
    out["ks_p"] = safe_float(row.get("ks_p") or meta.get("ks_p"))
    out["ac1"] = safe_float(row.get("ac1") or meta.get("ac1"))
    out["q_min"] = safe_float(row.get("q_min") or meta.get("q_min"))
    out["q_max"] = safe_float(row.get("q_max") or meta.get("q_max"))
    out["z_min"] = safe_float(row.get("z_min") or meta.get("z_min"))
    out["z_max"] = safe_float(row.get("z_max") or meta.get("z_max"))

    # If theta is still missing, try to parse it from NPZ path or exp note (rare)
    if out["theta"] is None:
        warnings.append(f"theta missing in {csv_path.name}; please regenerate summaries with v8_report_v2.py")
    if out["kappa"] is None:
        # kappa is less critical for theta grid
        warnings.append(f"kappa missing in {csv_path.name}; continuing anyway")

    return out, warnings

def write_merged_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DEFAULT_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in DEFAULT_COLS})

def fmt_num(x: Any, nd: int = 3) -> str:
    if x is None or x == "":
        return "--"
    try:
        v = float(x)
    except Exception:
        return str(x)
    # Use scientific for very small/large numbers
    if abs(v) != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
        return f"{v:.2e}"
    return f"{v:.{nd}f}"

def write_latex_table(rows: List[Dict[str, Any]], out_tex: Path) -> None:
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    # Sort by theta then kappa
    def key(r: Dict[str, Any]):
        t = r.get("theta")
        k = r.get("kappa")
        return (1e9 if t is None else float(t), 1e9 if k is None else float(k))
    rows = sorted(rows, key=key)

    lines: List[str] = []
    lines.append(r"% Auto-generated by v8_merge_summaries_v2.py")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Summary diagnostics for standardized $Q_\theta(T)$ across a $\theta$ grid.}")
    lines.append(r"\label{tab:v8-theta-grid}")
    lines.append(r"\begin{tabular}{r r r r r r r}")
    lines.append(r"\hline")
    lines.append(r"$\theta$ & $n$ & skew & kurt & KS-$D$ & KS-$p$ & AC(1) \\")
    lines.append(r"\hline")
    for r in rows:
        theta = fmt_num(r.get("theta"), 2)
        n = r.get("n") if r.get("n") is not None else "--"
        skew = fmt_num(r.get("skew"), 3)
        kurt = fmt_num(r.get("kurt_excess"), 3)
        ksD = fmt_num(r.get("ks_D"), 3)
        ksp = fmt_num(r.get("ks_p"), 3)
        ac1 = fmt_num(r.get("ac1"), 3)
        lines.append(f"{theta} & {n} & {skew} & {kurt} & {ksD} & {ksp} & {ac1} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_fig_snippet(prefixes: List[Path], out_tex: Path) -> None:
    """
    Optional: generates LaTeX snippet to include series/hist/qq figures
    if they exist alongside summaries, using the same prefix.
    """
    lines: List[str] = []
    lines.append(r"% Auto-generated by v8_merge_summaries_v2.py")
    for p in prefixes:
        exp = infer_exp_from_name(p.name) or p.name.replace("v8_", "")
        series = p.with_name(p.name + "_series.png")
        hist = p.with_name(p.name + "_hist.png")
        qq = p.with_name(p.name + "_qq.png")
        have = [x for x in [series, hist, qq] if x.exists()]
        if not have:
            continue
        lines.append(r"\begin{figure}[t]")
        lines.append(r"\centering")
        # 3-up if we have all, else include what exists
        for img in have:
            lines.append(rf"\includegraphics[width=0.32\linewidth]{{{img.as_posix()}}}")
        lines.append(rf"\caption{{Diagnostics for experiment {exp}.}}")
        lines.append(rf"\label{{fig:v8-{exp}}}")
        lines.append(r"\end{figure}")
        lines.append("")
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=".", help="Directory where v8_*_summary.csv live.")
    ap.add_argument("--glob", default="v8_*_summary.csv", help="Glob to find summary CSV files.")
    ap.add_argument("--out-dir", default=".", help="Directory to write merged outputs.")
    ap.add_argument("--no-figs", action="store_true", help="Do not generate LaTeX figure snippet.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(in_dir.glob(args.glob))
    if not csv_files:
        print(f"[ERROR] No files matched: {in_dir / args.glob}")
        print("Tip: run this from the folder that contains v8_*_summary.csv, or pass --in-dir.")
        return

    rows: List[Dict[str, Any]] = []
    prefixes: List[Path] = []
    warnings_all: List[str] = []

    for csv_path in csv_files:
        # prefix: remove _summary.csv
        base = csv_path.name.replace("_summary.csv", "")
        prefix_path = csv_path.with_name(base)
        r, warns = collect_one(prefix_path)
        warnings_all.extend(warns)
        if r:
            rows.append(r)
            prefixes.append(prefix_path)

    # Filter out rows without theta (can't be placed on theta grid)
    rows_ok = [r for r in rows if r.get("theta") is not None]
    rows_bad = [r for r in rows if r.get("theta") is None]
    if rows_bad:
        warnings_all.append(f"{len(rows_bad)} rows missing theta were skipped in the theta-grid table.")

    merged_csv = out_dir / "v8_theta_grid_merged.csv"
    table_tex = out_dir / "v8_theta_grid_table.tex"
    figs_tex = out_dir / "v8_theta_grid_figs.tex"

    write_merged_csv(rows, merged_csv)
    write_latex_table(rows_ok, table_tex)
    if not args.no_figs:
        write_fig_snippet(prefixes, figs_tex)

    print("Done. Outputs:")
    print(f"  {merged_csv}")
    print(f"  {table_tex}")
    if not args.no_figs:
        print(f"  {figs_tex}")

    if warnings_all:
        print("\nWarnings:")
        for w in warnings_all:
            print("  - " + w)

if __name__ == "__main__":
    main()
