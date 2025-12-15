#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v8_report.py — batch analyzer for NNBD V8 coupling artifacts.

Inputs:
- --npz <path> (an explicit coupling_*.npz)
- or --exp <hex> with --in-dir (expects coupling_<exp>.npz in that dir)
- or just --in-dir to discover coupling_*.npz

Outputs (per experiment) into --out-dir:
- v8_<exp>_summary.json / .csv / .tex
- v8_<exp>_hist.png / _qq.png / _series.png

No SciPy required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _ks_test_normal(z: np.ndarray) -> Tuple[float, float]:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    n = z.size
    if n == 0:
        return float('nan'), float('nan')
    zs = np.sort(z)
    cdf = _norm_cdf(zs)
    ecdf = np.arange(1, n + 1) / n
    ecdf_prev = np.arange(0, n) / n
    D = float(max(np.max(np.abs(ecdf - cdf)), np.max(np.abs(cdf - ecdf_prev))))

    en = math.sqrt(n)
    lam = (en + 0.12 + 0.11 / en) * D
    s = 0.0
    for k in range(1, 200):
        term = 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * (k * k) * (lam * lam))
        s += term
        if abs(term) < 1e-10:
            break
    p = max(0.0, min(1.0, s))
    return D, float(p)


def _skew_kurt(z: np.ndarray) -> Tuple[float, float]:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    n = z.size
    if n < 3:
        return float('nan'), float('nan')
    m = float(np.mean(z))
    s = float(np.std(z, ddof=1))
    if s == 0:
        return 0.0, 0.0
    x = (z - m) / s
    return float(np.mean(x**3)), float(np.mean(x**4) - 3.0)


def _autocorr_lag1(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 3:
        return float('nan')
    a, b = z[:-1], z[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _pick_array(npz: np.lib.npyio.NpzFile, names: List[str]) -> Optional[np.ndarray]:
    for n in names:
        if n in npz.files:
            return npz[n]
    lower_map = {k.lower(): k for k in npz.files}
    for n in names:
        k = lower_map.get(n.lower())
        if k is not None:
            return npz[k]
    return None


def _extract_T_Q_z(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, List[str]]]:
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.files)
    meta = {'keys': keys}

    T = _pick_array(npz, ['T', 'Ts', 't', 't_values', 'T_values'])
    if T is None:
        idx = _pick_array(npz, ['idx', 'i', 'index'])
        if idx is not None:
            T = np.asarray(idx, dtype=float)
        else:
            raise KeyError(f'Could not find T array. Keys={keys}')

    Q = _pick_array(npz, ['Q_raw', 'Q', 'Qt', 'Q_theta', 'D', 'D_theta'])
    if Q is None:
        raise KeyError(f'Could not find Q array. Keys={keys}')

    z = _pick_array(npz, ['z', 'Z', 'Q_std', 'Q_standardized', 'z_standardized'])
    Qf = np.asarray(Q, dtype=float).reshape(-1)
    if z is None:
        mu = float(np.mean(Qf))
        sd = float(np.std(Qf, ddof=1)) if Qf.size > 1 else float('nan')
        z = (Qf - mu) / sd if sd and sd > 0 else np.full_like(Qf, np.nan, dtype=float)
    else:
        z = np.asarray(z, dtype=float).reshape(-1)

    T = np.asarray(T, dtype=float).reshape(-1)
    n = min(T.size, Qf.size, z.size)
    return T[:n], Qf[:n], z[:n], meta


@dataclass
class Summary:
    exp: str
    npz_path: str
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
    z_min: float
    z_max: float


def analyze_one(npz_path: str, exp: str) -> Tuple[Summary, Dict]:
    T, Q, z, meta = _extract_T_Q_z(npz_path)
    zf = z[np.isfinite(z)]
    n = int(zf.size)

    mean = float(np.mean(zf)) if n else float('nan')
    std = float(np.std(zf, ddof=1)) if n > 1 else float('nan')
    skew, kurt = _skew_kurt(zf)
    ks_D, ks_p = _ks_test_normal(zf)
    ac1 = _autocorr_lag1(zf)

    summary = Summary(
        exp=exp,
        npz_path=npz_path,
        n=n,
        mean=mean,
        std=std,
        skew=skew,
        kurt_excess=kurt,
        ks_D=ks_D,
        ks_p=ks_p,
        ac1=ac1,
        q_min=float(np.min(Q)) if Q.size else float('nan'),
        q_max=float(np.max(Q)) if Q.size else float('nan'),
        z_min=float(np.min(zf)) if n else float('nan'),
        z_max=float(np.max(zf)) if n else float('nan'),
    )
    extra = {'T': T.tolist(), 'Q': Q.tolist(), 'z': z.tolist(), 'meta': meta}
    return summary, extra


def write_outputs(summary: Summary, extra: Dict, out_dir: str, out_prefix: str, bins: int = 12) -> None:
    os.makedirs(out_dir, exist_ok=True)

    jpath = os.path.join(out_dir, f'{out_prefix}_summary.json')
    with open(jpath, 'w', encoding='utf-8') as f:
        json.dump(summary.__dict__, f, indent=2, ensure_ascii=False)

    cpath = os.path.join(out_dir, f'{out_prefix}_summary.csv')
    with open(cpath, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary.__dict__.keys()))
        w.writeheader()
        w.writerow(summary.__dict__)

    tpath = os.path.join(out_dir, f'{out_prefix}_summary.tex')
    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Summary diagnostics for standardized coupling statistic $z$ (experiment {summary.exp}).}}
\\label{{tab:summary:{summary.exp}}}
\\begin{{tabular}}{{lrrrrrrrr}}
\\hline
$N$ & mean & std & skew & kurt(ex) & KS $D$ & KS $p$ & acf$_1$ & $[z_{{\\min}},z_{{\\max}}]$ \\
\\hline
{summary.n:d} & {summary.mean:+.3f} & {summary.std:.3f} & {summary.skew:+.3f} & {summary.kurt_excess:+.3f} & {summary.ks_D:.3f} & {summary.ks_p:.3f} & {summary.ac1:+.3f} & [{summary.z_min:+.2f},{summary.z_max:+.2f}] \\
\\hline
\\end{{tabular}}
\\par\\smallskip
\\small{{\\emph{{Note.}} KS $p$ uses an asymptotic approximation; for small $N$ interpret cautiously.}}
\\end{{table}}
"""
    with open(tpath, 'w', encoding='utf-8') as f:
        f.write(tex)

    if plt is None:
        return

    T = np.asarray(extra['T'], dtype=float)
    Q = np.asarray(extra['Q'], dtype=float)
    z = np.asarray(extra['z'], dtype=float)
    zf = z[np.isfinite(z)]

    plt.figure()
    plt.hist(zf, bins=bins, density=True)
    xs = np.linspace(np.min(zf) - 0.5, np.max(zf) + 0.5, 300)
    ys = (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * xs * xs)
    plt.plot(xs, ys)
    plt.title('Histogram of standardized z')
    plt.xlabel('z')
    plt.ylabel('density')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{out_prefix}_hist.png'), dpi=200)
    plt.close()

    # QQ plot without SciPy: approximate inverse CDF using erfinv approx + Newton steps
    def erfinv(y: np.ndarray) -> np.ndarray:
        a = 0.147
        y = np.clip(y, -0.999999, 0.999999)
        ln = np.log(1.0 - y * y)
        first = (2.0 / (math.pi * a) + ln / 2.0)
        second = ln / a
        x = np.sign(y) * np.sqrt(np.sqrt(first * first - second) - first)
        for _ in range(2):
            err = np.vectorize(math.erf)(x) - y
            der = (2.0 / math.sqrt(math.pi)) * np.exp(-x * x)
            x = x - err / der
        return x

    zsort = np.sort(zf)
    n = zsort.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = math.sqrt(2.0) * erfinv(2.0 * probs - 1.0)

    plt.figure()
    plt.scatter(theo, zsort, s=18)
    lo = min(float(np.min(theo)), float(np.min(zsort)))
    hi = max(float(np.max(theo)), float(np.max(zsort)))
    plt.plot([lo, hi], [lo, hi])
    plt.title('QQ plot vs N(0,1)')
    plt.xlabel('theoretical')
    plt.ylabel('empirical')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{out_prefix}_qq.png'), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(T, Q, marker='o')
    plt.title('Q(T) (raw) vs T')
    plt.xlabel('T')
    plt.ylabel('Q')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{out_prefix}_series.png'), dpi=200)
    plt.close()


def discover_npz(in_dir: str) -> List[str]:
    if not os.path.isdir(in_dir):
        return []
    out = []
    for fn in os.listdir(in_dir):
        if fn.startswith('coupling_') and fn.endswith('.npz'):
            out.append(os.path.join(in_dir, fn))
    return sorted(out)


def exp_from_filename(path: str) -> str:
    m = re.search(r'coupling_([0-9a-fA-F]+)\.npz$', os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', default='.', help='Directory containing coupling_*.npz')
    ap.add_argument('--out-dir', default='.', help='Output directory for summaries/plots')
    ap.add_argument('--npz', default='', help='Path to a coupling_*.npz file')
    ap.add_argument('--exp', default='', help='Experiment id (hex) to locate coupling_<exp>.npz in --in-dir')
    ap.add_argument('--bins', type=int, default=12, help='Histogram bins')
    args = ap.parse_args()

    if args.npz:
        targets = [args.npz]
    elif args.exp:
        targets = [os.path.join(args.in_dir, f'coupling_{args.exp}.npz')]
    else:
        targets = discover_npz(args.in_dir)

    if not targets:
        print('No targets found. Provide --npz, or --exp, or point --in-dir to coupling_*.npz files.')
        return 2

    for p in targets:
        if not os.path.isfile(p):
            print(f'[ERR] Missing file: {p}')
            continue
        exp = args.exp if args.exp else exp_from_filename(p)
        out_prefix = f'v8_{exp}'
        try:
            summary, extra = analyze_one(p, exp=exp)
            write_outputs(summary, extra, out_dir=args.out_dir, out_prefix=out_prefix, bins=args.bins)
            print(f'[OK] {exp}: wrote outputs with prefix {out_prefix} into {args.out_dir}')
        except Exception as e:
            print(f'[FAIL] {p}: {e}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
