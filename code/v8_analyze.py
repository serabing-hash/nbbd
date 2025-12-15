#!/usr/bin/env python3
"""v8_analyze.py
Robust analyzer for NNBD V8 coupling artifacts.

- Finds newest coupling_*.npz in --in-dir if --npz not provided
- Prints available keys
- Computes summary stats
- Saves histogram and QQ plot
- Writes full traceback to analyze_error.log if something fails
"""
import argparse, math, traceback
from pathlib import Path

import numpy as np

def ndtri(p: float) -> float:
    # Acklam approximation for inverse normal CDF (adequate for QQ plots)
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00]
    plow=0.02425; phigh=1-plow
    if p <= 0.0: return float("-inf")
    if p >= 1.0: return float("inf")
    if p < plow:
        q=math.sqrt(-2*math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q=math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q=p-0.5; r=q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def pick_latest_npz(indir: Path) -> Path:
    cands = sorted(indir.glob('coupling_*.npz'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No coupling_*.npz found in: {indir.resolve()}")
    return cands[0]

def safe_get(d, names):
    for n in names:
        if n in d:
            return d[n]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', default='nnbd_v8_out')
    ap.add_argument('--npz', default=None)
    ap.add_argument('--out-prefix', default='v8')
    ap.add_argument('--bins', type=int, default=14)
    args = ap.parse_args()

    indir = Path(args.in_dir)
    npz_path = Path(args.npz) if args.npz else pick_latest_npz(indir)
    print(f"[info] Using artifact: {npz_path.resolve()}")
    d = np.load(npz_path, allow_pickle=True)
    keys = list(d.keys())
    print('[info] Keys in npz:', keys)

    Q = safe_get(d, ['Q','Q_theta','Qtheta'])
    T = safe_get(d, ['T','T_grid','Tvals'])
    if Q is None:
        raise KeyError(f"Could not find Q array. keys={keys}")
    Q = np.asarray(Q, float)
    Q = Q[np.isfinite(Q)]
    if Q.size < 5:
        raise ValueError(f"Q has too few finite values: {Q.size}")

    m=float(Q.mean()); s=float(Q.std(ddof=1))
    z=(Q-m)/s
    sk=float(np.mean(z**3))
    ku=float(np.mean(z**4)-3.0)
    print(f"[stats] N={Q.size} mean={m:.6g} std={s:.6g} skew={sk:.6g} excess_kurt={ku:.6g}")

    import matplotlib.pyplot as plt

    hist_png=f"{args.out_prefix}_hist.png"
    qq_png=f"{args.out_prefix}_qq.png"

    plt.figure()
    plt.hist(z, bins=args.bins, density=True)
    xs=np.linspace(z.min(), z.max(), 400)
    pdf=(1/np.sqrt(2*np.pi))*np.exp(-0.5*xs**2)
    plt.plot(xs, pdf)
    plt.title('Histogram of standardized Q')
    plt.xlabel('z'); plt.ylabel('density')
    plt.savefig(hist_png, dpi=180, bbox_inches='tight')
    plt.close()

    plt.figure()
    zs=np.sort(z)
    ps=(np.arange(1, zs.size+1)-0.5)/zs.size
    theo=np.array([ndtri(float(p)) for p in ps])
    plt.plot(theo, zs, marker='o', linestyle='none', markersize=3)
    mn=float(min(theo.min(), zs.min())); mx=float(max(theo.max(), zs.max()))
    plt.plot([mn,mx],[mn,mx])
    plt.title('QQ plot vs N(0,1)')
    plt.xlabel('theoretical'); plt.ylabel('empirical')
    plt.savefig(qq_png, dpi=180, bbox_inches='tight')
    plt.close()

    print(f"[ok] Saved: {hist_png}, {qq_png}")

    if T is not None:
        T=np.asarray(T,float)
        if T.size==Q.size:
            ts_png=f"{args.out_prefix}_Q_vs_T.png"
            plt.figure()
            plt.plot(T, Q, marker='o', linestyle='-')
            plt.title('Q(T) (raw) vs T')
            plt.xlabel('T'); plt.ylabel('Q')
            plt.savefig(ts_png, dpi=180, bbox_inches='tight')
            plt.close()
            print(f"[ok] Saved: {ts_png}")
        else:
            print(f"[warn] T exists but size mismatch: len(T)={T.size} vs len(Q)={Q.size}")

if __name__=='__main__':
    try:
        main()
    except Exception:
        Path('analyze_error.log').write_text(traceback.format_exc(), encoding='utf-8')
        print('[error] Failed. Full traceback written to analyze_error.log')
        raise
