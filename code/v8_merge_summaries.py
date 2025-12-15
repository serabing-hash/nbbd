#!/usr/bin/env python3
"""
Merge v8_report_v2 summary CSVs into a single CSV + LaTeX table.

Typical usage (Windows):
  python v8_merge_summaries.py ^
    --csv v8_8d3d978be11ae250_summary.csv v8_019a8cb9269064ba_summary.csv v8_bc00b4127231e034_summary.csv ^
    --nnbd-out-dir ..\nnbd_v8_out ^
    --out-prefix v8_theta_grid

This will produce:
  v8_theta_grid_merged.csv
  v8_theta_grid_table.tex
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class Row:
    exp: str
    theta: Optional[float]
    kappa: Optional[float]
    T_min: Optional[float]
    T_max: Optional[float]
    N_T: Optional[int]
    center_J: Optional[int]
    X_need: Optional[int]
    n: int
    mean: float
    std: float
    skew: float
    kurt_excess: float
    ks_D: float
    ks_p: float
    ac1: float
    q_min: float
    q_max: float

def _safe_get(d: Dict[str, Any], k: str):
    return d.get(k, None)

def load_params(nnbd_out_dir: str, exp: str) -> Dict[str, Any]:
    # pipeline writes params_<exp>.json under output dir
    path = os.path.join(nnbd_out_dir, f"params_{exp}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def read_summary_csv(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")
    # report v2 writes one-row summary
    return rows[0]

def ffloat(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def fint(x):
    try:
        return int(float(x))
    except Exception:
        return None

def build_row(csv_row: Dict[str, Any], params: Dict[str, Any]) -> Row:
    exp = csv_row.get("exp", "").strip()
    # normalize various key spellings in params
    theta = _safe_get(params, "theta")
    kappa = _safe_get(params, "kappa")
    T_min = _safe_get(params, "T_min")
    T_max = _safe_get(params, "T_max")
    N_T = _safe_get(params, "N_T") or _safe_get(params, "N")
    center_J = _safe_get(params, "center_J")
    X_need = _safe_get(params, "X_need")
    return Row(
        exp=exp,
        theta=float(theta) if theta is not None else None,
        kappa=float(kappa) if kappa is not None else None,
        T_min=float(T_min) if T_min is not None else None,
        T_max=float(T_max) if T_max is not None else None,
        N_T=int(N_T) if N_T is not None else None,
        center_J=int(center_J) if center_J is not None else None,
        X_need=int(X_need) if X_need is not None else None,
        n=int(csv_row.get("n", 0)),
        mean=ffloat(csv_row.get("mean", "nan")),
        std=ffloat(csv_row.get("std", "nan")),
        skew=ffloat(csv_row.get("skew", "nan")),
        kurt_excess=ffloat(csv_row.get("kurt_excess", "nan")),
        ks_D=ffloat(csv_row.get("ks_D", "nan")),
        ks_p=ffloat(csv_row.get("ks_p", "nan")),
        ac1=ffloat(csv_row.get("ac1", "nan")),
        q_min=ffloat(csv_row.get("q_min", "nan")),
        q_max=ffloat(csv_row.get("q_max", "nan")),
    )

def write_merged_csv(rows: List[Row], out_csv: str) -> None:
    fieldnames = [
        "exp","theta","kappa","T_min","T_max","N_T","center_J","X_need",
        "n","mean","std","skew","kurt_excess","ks_D","ks_p","ac1","q_min","q_max"
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "exp": r.exp,
                "theta": "" if r.theta is None else f"{r.theta:.3g}",
                "kappa": "" if r.kappa is None else f"{r.kappa:.3g}",
                "T_min": "" if r.T_min is None else f"{r.T_min:.0f}",
                "T_max": "" if r.T_max is None else f"{r.T_max:.0f}",
                "N_T": "" if r.N_T is None else str(r.N_T),
                "center_J": "" if r.center_J is None else str(r.center_J),
                "X_need": "" if r.X_need is None else str(r.X_need),
                "n": r.n,
                "mean": f"{r.mean:.3g}",
                "std": f"{r.std:.3g}",
                "skew": f"{r.skew:.3g}",
                "kurt_excess": f"{r.kurt_excess:.3g}",
                "ks_D": f"{r.ks_D:.3g}",
                "ks_p": f"{r.ks_p:.3g}",
                "ac1": f"{r.ac1:.3g}",
                "q_min": f"{r.q_min:.3g}",
                "q_max": f"{r.q_max:.3g}",
            })

def latex_table(rows: List[Row], caption: str, label: str) -> str:
    # Sort by theta if available; otherwise by exp
    rows2 = sorted(rows, key=lambda r: (1e9 if r.theta is None else r.theta, r.exp))
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{rrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"$\theta$ & $n$ & mean & sd & skew & kurt & KS $D$ & KS $p$ & $\mathrm{AC}(1)$ \\")
    lines.append(r"\midrule")
    for r in rows2:
        th = "?" if r.theta is None else f"{r.theta:.2f}"
        lines.append(
            f"{th} & {r.n:d} & {r.mean:.3g} & {r.std:.3g} & {r.skew:.3g} & {r.kurt_excess:.3g} & {r.ks_D:.3g} & {r.ks_p:.3g} & {r.ac1:.3g} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="One-row summary CSVs produced by v8_report_v2.py")
    ap.add_argument("--nnbd-out-dir", default="nnbd_v8_out", help="Directory containing params_<exp>.json")
    ap.add_argument("--out-prefix", default="v8_theta_grid", help="Output prefix for merged CSV/TeX")
    ap.add_argument("--caption", default="Summary diagnostics for standardized $Q_{\\theta}(T)$ across a small $\\theta$ grid. Mean/sd are for standardized values; KS is versus $N(0,1)$; AC(1) is lag-1 autocorrelation.", help="LaTeX table caption")
    ap.add_argument("--label", default="tab:v8-theta-grid", help="LaTeX table label")
    args = ap.parse_args()

    rows: List[Row] = []
    for csv_path in args.csv:
        csv_row = read_summary_csv(csv_path)
        exp = csv_row.get("exp","").strip()
        params = load_params(args.nnbd_out_dir, exp) if exp else {}
        rows.append(build_row(csv_row, params))

    out_csv = args.out_prefix + "_merged.csv"
    out_tex = args.out_prefix + "_table.tex"
    write_merged_csv(rows, out_csv)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(latex_table(rows, args.caption, args.label))

    print("Wrote:", out_csv)
    print("Wrote:", out_tex)
    # Helpful hint
    print("\nTeX: \\input{" + out_tex.replace("\\","/") + "}")

if __name__ == "__main__":
    main()
