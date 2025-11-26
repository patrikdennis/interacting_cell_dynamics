#!/usr/bin/env python3
# abc_plot_calibration.py
#
# Makes a scatter like the example:
#   x = real diffusion (log10 D, truth)
#   y = estimated diffusion (log10 D, posterior mode)
#   color = real proliferation (log10 alpha, truth)
#
# Inputs (from abc_batch.py runs):
#   <out-dir>/abc_points_all.csv
#
# Each row in that file looks like:
#   rep_id,param,space,estimate_mode,truth
#
# We use rows with:
#   - (param == "D",     space == "log10")  -> x (truth), y (estimate_mode)
#   - (param == "alpha", space == "log10")  -> color = truth
#
# Usage examples:
#   python abc_plot_calibration.py --in-dir /path/to/abc_runs
#   python abc_plot_calibration.py --in-dir /path/to/abc_runs --save /path/to/out/calibration.png
#   python abc_plot_calibration.py --in-dir /path/to/abc_runs --highlight-rep 17
#
# If SciPy is installed, we print R and p-value; otherwise we print R only.

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def load_points(in_dir: str) -> pd.DataFrame:
    path = os.path.join(in_dir, "abc_points_all.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")
    df = pd.read_csv(path)
    required = {"rep_id", "param", "space", "estimate_mode", "truth"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns in abc_points_all.csv: {missing}")
    return df

def compute_corr_and_p(x, y):
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Pearson R
    R = float(np.corrcoef(x, y)[0, 1])
    R2, p = pearsonr(x, y)
    R = float(R2)
    
    #p_text = f", p={p:.4f}"
    
    #return R, p_text
    return R

def plot_calibration(df_points: pd.DataFrame, save_path: str | None, highlight_rep: int | None):
    # Select needed rows
    d_rows = df_points[(df_points["param"] == "D") & (df_points["space"] == "log10")]
    a_rows = df_points[(df_points["param"] == "alpha") & (df_points["space"] == "log10")]

    if d_rows.empty:
        raise ValueError("No rows with param='D' and space='log10' found in abc_points_all.csv")
    if a_rows.empty:
        raise ValueError("No rows with param='alpha' and space='log10' found in abc_points_all.csv")

    # Merge on rep_id to get color (truth log10 alpha)
    d_rows = d_rows[["rep_id", "estimate_mode", "truth"]].rename(
        columns={"estimate_mode": "est_log10_D", "truth": "truth_log10_D"}
    )
    a_rows = a_rows[["rep_id", "truth"]].rename(columns={"truth": "truth_log10_alpha"})
    data = pd.merge(d_rows, a_rows, on="rep_id", how="inner")

    if data.empty:
        raise ValueError("Join produced no rows—check that rep_id values overlap between D and alpha entries.")

    x = data["truth_log10_D"].to_numpy(float)      # Real diffusion (log10)
    y = data["est_log10_D"].to_numpy(float)        # Estimated diffusion (log10)
    c = data["truth_log10_alpha"].to_numpy(float)  # Real proliferation (log10 alpha) for color

    #R, p_text = compute_corr_and_p(x, y)
    R = compute_corr_and_p(x, y)

    plt.figure()
    sc = plt.scatter(x, y, c=c, alpha=0.7)  # default colormap
    cb = plt.colorbar(sc)
    cb.set_label("Real proliferation (log10 divisions/s)")

    # Identity line
    lims = [
        np.nanmin([x.min(), y.min()]),
        np.nanmax([x.max(), y.max()])
    ]
    pad = 0.05 * (lims[1] - lims[0]) if np.isfinite(lims).all() else 0.5
    lo, hi = lims[0] - pad, lims[1] + pad
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="k", label="Identity")

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Real diffusion (log10 μm²/s)")
    plt.ylabel("Estimated diffusion (log10 μm²/s)")

    # Annotation with correlation
    #plt.text(0.02, 0.02, f"R={R:.3g}{p_text}", transform=plt.gca().transAxes)
    plt.text(0.02, 0.02, f"R={R:.3g}", transform = plt.gca().transAxes)

    # Optional highlight of one rep with a red square
    if highlight_rep is not None:
        row = data[data["rep_id"] == highlight_rep]
        if not row.empty:
            hx = float(row.iloc[0]["truth_log10_D"])
            hy = float(row.iloc[0]["est_log10_D"])
            plt.scatter([hx], [hy], facecolors="none", edgecolors="red", s=120, linewidths=2)
        else:
            print(f"Warning: rep_id {highlight_rep} not found; skipping highlight.")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved {save_path}")
    else:
        plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory containing abc_points_all.csv")
    ap.add_argument("--save", default=None, help="Path to save the PNG (if omitted, show interactively)")
    ap.add_argument("--highlight-rep", type=int, default=None, help="rep_id to highlight with a red square")
    args = ap.parse_args()

    df_points = load_points(args.in_dir)
    plot_calibration(df_points, args.save, args.highlight_rep)

if __name__ == "__main__":
    main()
