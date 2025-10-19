# plot_sim_overview.py
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# IOstream helpers 

def _read_table(pq_path: str, csv_path: str) -> pd.DataFrame:
    """Read Parquet if available, else CSV; else raise."""
    if os.path.exists(pq_path):
        return pd.read_parquet(pq_path)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Neither {pq_path} nor {csv_path} found.")


def _split_contiguous_blocks(df_with_row: pd.DataFrame, row_col: str = "_row"):
    """Split filtered snapshot rows into contiguous blocks in file order."""
    if df_with_row.empty:
        return []
    idx = df_with_row[row_col].to_numpy()
    jumps = np.where(np.diff(idx) != 1)[0] + 1
    cuts = np.r_[0, jumps, len(idx)]
    return [df_with_row.iloc[cuts[i]:cuts[i+1]] for i in range(len(cuts)-1)]


#  selection by sim-id OR parameter filters 

def _maybe_val(x, name):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        raise SystemExit(f"Could not parse {name}={x} as float.")

def _select_by_params(sims: pd.DataFrame,
                      r_target=None, D_target=None, log10d_target=None,
                      a_target=None, log10a_target=None,
                      tol_r=1e-6, tol_log10d=1e-6, tol_log10a=1e-6,
                      nearest=True) -> pd.Series:
    """
    Select a single row from sims with optional filters on r, D/log10d, alpha/log10alpha.
    If multiple filters are given, all must satisfy the tolerance to be an in-tolerance match.
    If none in tolerance and nearest=True, pick the nearest overall (by normalized absolute deltas).
    """
    if sims.empty:
        raise SystemExit("No simulations to select from.")

    # Prepare columns & derived
    if "r" not in sims.columns:
        raise SystemExit("Column 'r' missing in simulations.")
    if "log10_d" not in sims.columns and (D_target is not None or log10d_target is not None):
        raise SystemExit("Column 'log10_d' missing in simulations (required for D/log10-d filters).")
    if "log10_alpha" not in sims.columns and (a_target is not None or log10a_target is not None):
        raise SystemExit("Column 'log10_alpha' missing in simulations (required for alpha/log10-alpha filters).")

    r_arr = sims["r"].to_numpy(dtype=float)
    log10d_arr = sims["log10_d"].to_numpy(dtype=float) if "log10_d" in sims.columns else None
    log10a_arr = sims["log10_alpha"].to_numpy(dtype=float) if "log10_alpha" in sims.columns else None

    # Convert linear targets to log10 if supplied
    if (D_target is not None) and (log10d_target is not None):
        print("WARNING: both --D and --log10-d provided; using --log10-d.", file=sys.stderr)
    if (a_target is not None) and (log10a_target is not None):
        print("WARNING: both --alpha and --log10-alpha provided; using --log10-alpha.", file=sys.stderr)

    if D_target is not None and log10d_target is None:
        if D_target <= 0:
            raise SystemExit("--D must be > 0.")
        log10d_target = np.log10(D_target)

    if a_target is not None and log10a_target is None:
        if a_target <= 0:
            raise SystemExit("--alpha must be > 0.")
        log10a_target = np.log10(a_target)

    # Build masks for in-tolerance matches (if corresponding target was provided)
    mask = np.ones(len(sims), dtype=bool)
    if r_target is not None:
        mask &= np.abs(r_arr - r_target) <= tol_r
    if log10d_target is not None:
        if log10d_arr is None:
            raise SystemExit("Missing log10_d in simulations.")
        mask &= np.abs(log10d_arr - log10d_target) <= tol_log10d
    if log10a_target is not None:
        if log10a_arr is None:
            raise SystemExit("Missing log10_alpha in simulations.")
        mask &= np.abs(log10a_arr - log10a_target) <= tol_log10a

    # If any exact (in-tolerance) matches, pick the first with the smallest total delta
    if np.any(mask):
        cand = sims.loc[mask].copy()
        score = np.zeros(len(cand), dtype=float)
        if r_target is not None:
            score += np.abs(cand["r"].to_numpy(dtype=float) - r_target) / max(tol_r, 1e-12)
        if log10d_target is not None:
            score += np.abs(cand["log10_d"].to_numpy(dtype=float) - log10d_target) / max(tol_log10d, 1e-12)
        if log10a_target is not None:
            score += np.abs(cand["log10_alpha"].to_numpy(dtype=float) - log10a_target) / max(tol_log10a, 1e-12)
        i = int(np.argmin(score))
        return cand.iloc[i]

    # Else, pick nearest overall (if allowed)
    if not nearest:
        raise SystemExit("No simulation matched the given tolerances.")

    # Build a comparable distance score even when some targets are None
    score = np.zeros(len(sims), dtype=float)
    if r_target is not None:
        score += np.abs(r_arr - r_target) / (tol_r if tol_r > 0 else 1.0)
    if log10d_target is not None and log10d_arr is not None:
        score += np.abs(log10d_arr - log10d_target) / (tol_log10d if tol_log10d > 0 else 1.0)
    if log10a_target is not None and log10a_arr is not None:
        score += np.abs(log10a_arr - log10a_target) / (tol_log10a if tol_log10a > 0 else 1.0)

    idx = int(np.argmin(score))
    sel = sims.iloc[idx]
    # warning
    parts = []
    if r_target is not None:
        parts.append(f"r→{sel['r']:.4g}")
    if log10d_target is not None:
        parts.append(f"log10D→{sel['log10_d']:.4g}")
    if log10a_target is not None:
        parts.append(f"log10α→{sel['log10_alpha']:.4g}")
    print("WARNING: no row within tolerance; using nearest match: " + ", ".join(parts))
    return sel


def main():
    ap = argparse.ArgumentParser(
        description="Plot endpoint snapshot (cells) + PCF for ONE simulation, selected by sim-id or parameter filters."
    )
    ap.add_argument("--output-dir", required=True,
                    help="Folder with simulations/snapshots/radii parquet (or csv fallbacks).")

    # Selection: either by sim-id OR by parameters
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--sim-id", type=int, help="Simulation ID to plot.")
    group.add_argument("--by-params", action="store_true",
                       help="Select by parameter filters instead of sim-id.")

    # Parameter filters (used only if --by-params)
    ap.add_argument("--radius", type=float, default=None, help="Target cell radius r.")
    ap.add_argument("--D", type=float, default=None, help="Target diffusion D (linear).")
    ap.add_argument("--log10-d", type=float, default=None, dest="log10d", help="Target diffusion log10(D).")
    ap.add_argument("--alpha", type=float, default=None, help="Target alpha (linear).")
    ap.add_argument("--log10-alpha", type=float, default=None, dest="log10a", help="Target log10(alpha).")
    ap.add_argument("--tol-r", type=float, default=1e-6, help="Tolerance on r (default 1e-6).")
    ap.add_argument("--tol-log10-d", type=float, default=1e-6, dest="tol_log10d", help="Tolerance on log10(D).")
    ap.add_argument("--tol-log10-alpha", type=float, default=1e-6, dest="tol_log10a", help="Tolerance on log10(alpha).")
    ap.add_argument("--nearest", action="store_true",
                    help="If no row within tolerances, choose the nearest match instead of failing.")

    # Plot cosmetics
    ap.add_argument("--size-scale", type=float, default=0.6,
                    help="Dot area = (size_scale * r)^2 in points^2 (default 0.6).")
    ap.add_argument("--point-alpha", type=float, default=0.7, help="Scatter transparency for cells.")
    ap.add_argument("--xmax", type=float, default=None, help="Max distance on PCF x-axis.")
    ap.add_argument("--ylim0", action="store_true", help="Force PCF y-axis to start at 0.")
    ap.add_argument("--save", action="store_true", help="Save PNG instead of showing.")
    ap.add_argument("--filename", default=None, help="Output filename (defaults to overview_<simid>.png).")

    args = ap.parse_args()
    outdir = args.output_dir

    # Load data
    sims = _read_table(os.path.join(outdir, "simulations.parquet"),
                       os.path.join(outdir, "simulations.csv"))
    if "exit_code" in sims.columns:
        sims = sims[sims["exit_code"].isin([0, 1])].reset_index(drop=True)
    if sims.empty:
        sys.exit("No valid simulations to plot.")

    # Choose the simulation row
    if args.sim_id is not None:
        if "sim_id" not in sims.columns:
            sys.exit("sim_id column missing in simulations file; cannot select by sim-id.")
        row = sims.loc[sims["sim_id"] == args.sim_id]
        if row.empty:
            sys.exit(f"sim_id={args.sim_id} not found.")
        row = row.iloc[0]
    else:
        # by parameters
        row = _select_by_params(
            sims,
            r_target=_maybe_val(args.radius, "radius"),
            D_target=_maybe_val(args.D, "D"),
            log10d_target=_maybe_val(args.log10d, "log10-d"),
            a_target=_maybe_val(args.alpha, "alpha"),
            log10a_target=_maybe_val(args.log10a, "log10-alpha"),
            tol_r=args.tol_r, tol_log10d=args.tol_log10d, tol_log10a=args.tol_log10a,
            nearest=args.nearest
        )

    # Extract identifiers & parameters
    sim_id = int(row["sim_id"]) if "sim_id" in row else None
    r_cell = float(row["r"]) if "r" in row else float("nan")
    D = float(10.0 ** float(row["log10_d"])) if "log10_d" in row else float("nan")
    alpha = float(10.0 ** float(row["log10_alpha"])) if "log10_alpha" in row else float("nan")

    # PCF data
    pcf_cols = [c for c in sims.columns if c.startswith("pc_")]
    if not pcf_cols:
        sys.exit("No PCF columns found (expected columns starting with 'pc_').")
    g = row[pcf_cols].to_numpy(dtype=float)
    radii_df = _read_table(os.path.join(outdir, "radii.parquet"),
                           os.path.join(outdir, "radii.csv"))
    r_bins = radii_df["radius"].to_numpy(dtype=float)
    if g.shape[0] != r_bins.shape[0]:
        sys.exit(f"PCF length ({g.shape[0]}) != radii length ({r_bins.shape[0]}).")

    # Snapshot data 
    snapshot_ok = False
    snap_df = None
    if sim_id is not None:
        snap_path_pq = os.path.join(outdir, "snapshots.parquet")
        snap_path_csv = os.path.join(outdir, "snapshots.csv")
        if os.path.exists(snap_path_pq) or os.path.exists(snap_path_csv):
            snaps = _read_table(snap_path_pq, snap_path_csv)
            if "sim_id" in snaps.columns:
                snaps = snaps.reset_index().rename(columns={"index": "_row"}).sort_values("_row")
                df_all = snaps[snaps["sim_id"] == sim_id]
                if not df_all.empty:
                    blocks = _split_contiguous_blocks(df_all, row_col="_row")
                    if len(blocks) > 1:
                        print(f"WARNING: Found {len(blocks)} snapshot blocks for sim_id={sim_id}. "
                              f"Plotting the LAST block only (newest).")
                    snap_df = blocks[-1]
                    snapshot_ok = True
                else:
                    print(f"WARNING: No snapshot rows found for sim_id={sim_id}.")
            else:
                print("WARNING: 'sim_id' column missing in snapshots file; cannot plot cells.")
        else:
            print("WARNING: snapshots file not found; cannot plot cells.")
    else:
        print("WARNING: sim_id unavailable (selected by parameters and simulations file has no sim_id). "
              "Cannot lookup snapshots; plotting PCF only.")

    # Prepare figure (two panels if snapshot OK, else just PCF)
    if snapshot_ok:
        fig = plt.figure(figsize=(11, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1])
        ax0 = fig.add_subplot(gs[0, 0])  # snapshot
        ax1 = fig.add_subplot(gs[0, 1])  # PCF

        # Snapshot scatter
        s_pts2 = (args.size_scale * r_cell) ** 2
        ax0.scatter(snap_df["x"].to_numpy(), snap_df["y"].to_numpy(),
                    s=s_pts2, alpha=args.point_alpha)
        ax0.set_aspect("equal", adjustable="box")
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        if sim_id is not None:
            ax0.set_title(f"Endpoint cells — sim_id={sim_id}\nr={r_cell:.2f}, D={D:.3g}, alpha={alpha:.3g}")
        else:
            ax0.set_title(f"Endpoint cells\nr={r_cell:.2f}, D={D:.3g}, alpha={alpha:.3g}")
    else:
        fig, ax1 = plt.subplots(figsize=(7, 5))

    # PCF line
    ax1.plot(r_bins, g, label="g(r)")
    ax1.set_xlabel("distance r")
    ax1.set_ylabel("g(r)")
    ttl = f"PCF"
    if sim_id is not None:
        ttl += f" — sim_id={sim_id}"
    ttl += f"\nr={r_cell:.2f}, D={D:.3g}, alpha={alpha:.3g}"
    ax1.set_title(ttl)
    if args.xmax is not None:
        ax1.set_xlim(0, args.xmax)
    else:
        ax1.set_xlim(0, r_bins[-1])
    if args.ylim0:
        ax1.set_ylim(0, None)

    fig.tight_layout()

    # Save/show
    if args.save:
        if args.filename:
            fname = args.filename
        else:
            base = f"overview_{sim_id}" if sim_id is not None else "overview_selected"
            fname = base + ".png"
        out_path = os.path.join(outdir, fname)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
