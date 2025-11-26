# Reads:
#   <out-dir>/abc_accepts_all.csv
#   <out-dir>/abc_points_all.csv
#
# Then, for each rep_id, plots histograms of the posterior samples for the chosen
# parameters with vertical lines for the truth (solid green) and the mode estimate (black dashed).
#
# Examples:
#   python abc_plot_posteriors.py --in-dir /Volumes/LaCie/Thesis_work/Simulations_base/abc_runs --rep 7 --space log10 --params alpha D
#   python abc_plot_posteriors.py --in-dir /Volumes/LaCie/Thesis_work/Simulations_base/abc_runs --rep all --space linear --params alpha D r --save-dir /tmp/abc_plots
#
# Note:
#   - In log10 space: plots alpha and D using columns log10_alpha/log10_D.
#   - In linear space: plots alpha, D, r using columns alpha, D, r.
#   - r has no log10 version in the batch outputs, so r is only available in linear space.

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ACCEPTS_FILE = "abc_accepts_all.csv"
POINTS_FILE  = "abc_points_all.csv"

def hist_with_truth_and_est(vals, truth_val, est_val, title):
    plt.figure()
    plt.hist(vals, bins="auto", density=True, label="posterior (accepted)", alpha=0.5)
    plt.axvline(truth_val, linewidth=2, label="truth", color="green")
    plt.axvline(est_val, linewidth=2, linestyle="--", label="estimate (mode)", color="black")
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

def load_inputs(in_dir):
    acc_path = os.path.join(in_dir, ACCEPTS_FILE)
    pts_path = os.path.join(in_dir, POINTS_FILE)
    if not os.path.exists(acc_path):
        raise FileNotFoundError(f"Could not find {acc_path}")
    if not os.path.exists(pts_path):
        raise FileNotFoundError(f"Could not find {pts_path}")
    accepts = pd.read_csv(acc_path)
    points  = pd.read_csv(pts_path)
    # Basic sanity
    if "rep_id" not in accepts.columns or "rep_id" not in points.columns:
        raise ValueError("Both CSVs must contain a 'rep_id' column.")
    return accepts, points

def get_truth_and_est(points_df, rep_id, param, space):
    """
    Returns (truth_val, est_val) for given rep/param/space from abc_points_all.csv
    where rows look like:
      rep_id,param,space,estimate_mode,truth
    """
    sel = points_df[(points_df.rep_id == rep_id) & (points_df.param == param) & (points_df.space == space)]
    if sel.empty:
        raise ValueError(f"No point-estimate row for rep_id={rep_id}, param={param}, space={space}")
    truth_val = float(sel.iloc[0]["truth"])
    est_val   = float(sel.iloc[0]["estimate_mode"])
    return truth_val, est_val

def get_posterior_samples(accepts_df, rep_id, param, space):
    """
    Returns the posterior samples array for a given rep/param/space from abc_accepts_all.csv.

    Columns available:
      - Linear: alpha, D, r
      - Log10:  log10_alpha, log10_D
    """
    sub = accepts_df[accepts_df.rep_id == rep_id]
    if sub.empty:
        raise ValueError(f"No accepted draws for rep_id={rep_id}")
    if space == "linear":
        col_map = {"alpha": "alpha", "D": "D", "r": "r"}
    elif space == "log10":
        col_map = {"alpha": "log10_alpha", "D": "log10_D"}
    else:
        raise ValueError("space must be 'linear' or 'log10'")

    if param not in col_map:
        raise ValueError(f"Parameter '{param}' not available in {space} space.")
    col = col_map[param]
    if col not in sub.columns:
        raise ValueError(f"Column '{col}' not found in accepts file for rep_id={rep_id}")
    return sub[col].to_numpy(dtype=float)


def plot_rep(accepts_df, points_df, rep_id, params, space, save_dir=None):
    """
    Make and optionally save plots for a single rep_id for the requested params/space.
    """
    for p in params:
        # r only in linear space
        if p == "r" and space != "linear":
            print(f"Skipping r in {space} space (not available).")
            continue
        vals = get_posterior_samples(accepts_df, rep_id, p, space)
        truth_val, est_val = get_truth_and_est(points_df, rep_id, p, space)
        title = f"Posterior of {p} ({space}) — rep {rep_id}"
        hist_with_truth_and_est(vals, truth_val, est_val, title)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            safe_p = p.replace("/", "_")
            out = os.path.join(save_dir, f"rep{rep_id}_{safe_p}_{space}.png")
            plt.savefig(out, dpi=150)
            plt.close()
            print(f"Saved {out}")
        else:
            plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory that contains abc_accepts_all.csv and abc_points_all.csv")
    ap.add_argument("--rep", default="all", help="rep_id to plot (int) or 'all'")
    ap.add_argument("--space", choices=["linear", "log10"], default="linear", help="Parameter space to plot")
    ap.add_argument("--params", nargs="+", default=["alpha", "D", "r"],
                    help="Which parameters to plot. In log10 space, r is skipped automatically.")
    ap.add_argument("--save-dir", default=None, help="Directory to save PNGs. If omitted, plots are shown interactively.")
    args = ap.parse_args()

    accepts, points = load_inputs(args.in_dir)

    # Determine which reps to plot
    if args.rep == "all":
        reps = sorted(accepts["rep_id"].unique().tolist())
    else:
        try:
            rep_id = int(args.rep)
        except ValueError:
            raise ValueError("--rep must be an integer or 'all'")
        reps = [rep_id]

    for rep_id in reps:
        plot_rep(accepts, points, rep_id, args.params, args.space, save_dir=args.save_dir)


if __name__ == "__main__":
    main()