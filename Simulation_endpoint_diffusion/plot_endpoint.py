# # import argparse
# # import os
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # def main():
# #     ap = argparse.ArgumentParser(description="Plot cell positions for a saved simulation snapshot.")
# #     ap.add_argument("--output-dir", default="output", help="Directory with snapshots.parquet")
# #     ap.add_argument("--sim-id", type=int, required=True, help="Simulation ID to plot (as saved in snapshots)")
# #     ap.add_argument("--alpha", type=float, default=0.6, help="Point alpha")
# #     ap.add_argument("--s", type=float, default=10, help="Scatter size")
# #     ap.add_argument("--save", action="store_true", help="Save PNG instead of showing")
# #     args = ap.parse_args()

# #     snap_path = os.path.join(args.output_dir, "snapshots.parquet")
# #     if not os.path.exists(snap_path):
# #         raise FileNotFoundError(f"{snap_path} not found. Run with SAVE_SNAPSHOTS=1 first.")

# #     snaps = pd.read_parquet(snap_path)
# #     df = snaps[snaps["sim_id"] == args.sim_id]
# #     if df.empty:
# #         raise SystemExit(f"No snapshot found for sim_id={args.sim_id}. Try another id or increase SNAPSHOT_RATE.")

# #     # r is constant per sim; grab the first
# #     r = df["r"].iloc[0]

# #     fig, ax = plt.subplots(figsize=(6,6))
# #     ax.scatter(df["x"], df["y"], s=args.s, alpha=args.alpha)
# #     ax.set_aspect("equal", adjustable="box")
# #     ax.set_title(f"Last-timestep cells (sim_id={args.sim_id}, r≈{r:.2f})")
# #     ax.set_xlabel("x")
# #     ax.set_ylabel("y")

# #     if args.save:
# #         out = os.path.join(args.output_dir, f"snapshot_sim_{args.sim_id}.png")
# #         plt.savefig(out, bbox_inches="tight", dpi=150)
# #         print(f"Saved {out}")
# #     else:
# #         plt.show()

# # if __name__ == "__main__":
# #     main()


# import argparse
# import os
# import sys
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def read_snapshots(path: str) -> pd.DataFrame:
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"{path} not found. Run with SAVE_SNAPSHOTS=1 first.")
#     return pd.read_parquet(path)

# def split_into_contiguous_blocks(df_with_row: pd.DataFrame, row_col: str = "_row"):
#     """
#     Given df filtered to a single sim_id and containing a monotonic row order column,
#     split into contiguous blocks (where row indices are consecutive).
#     Returns a list of dataframes, in the same order they appear in the file.
#     """
#     if df_with_row.empty:
#         return []
#     idx = df_with_row[row_col].to_numpy()
#     # A new block starts when the row jumps by more than 1
#     # (first row is always start of a block)
#     separators = np.where(np.diff(idx) != 1)[0] + 1
#     cuts = np.r_[0, separators, len(idx)]
#     blocks = []
#     for i in range(len(cuts) - 1):
#         start, end = cuts[i], cuts[i + 1]
#         blocks.append(df_with_row.iloc[start:end])
#     return blocks

# def main():
#     ap = argparse.ArgumentParser(description="Plot final-timestep particles for a given sim_id from snapshots.parquet")
#     ap.add_argument("--output-dir", default="output", help="Directory containing snapshots.parquet")
#     ap.add_argument("--sim-id", type=int, required=True, help="Simulation ID to plot")
#     ap.add_argument("--alpha", type=float, default=0.6, help="Point alpha")
#     ap.add_argument("--s", type=float, default=10, help="Scatter marker size")
#     ap.add_argument("--save", action="store_true", help="Save PNG instead of showing")
#     ap.add_argument("--filename", default=None, help="Output PNG filename (optional)")
#     args = ap.parse_args()

#     snap_path = os.path.join(args.output_dir, "snapshots.parquet")
#     snaps = read_snapshots(snap_path)

#     # Preserve file order by creating a synthetic row index column
#     snaps = snaps.reset_index().rename(columns={"index": "_row"})

#     # Filter for the requested sim_id and keep file order
#     df_all = snaps[snaps["sim_id"] == args.sim_id].sort_values("_row")
#     if df_all.empty:
#         sys.exit(f"No snapshot rows found for sim_id={args.sim_id} in {snap_path}")

#     # Split by contiguous blocks → each block likely corresponds to one appended run
#     blocks = split_into_contiguous_blocks(df_all, row_col="_row")

#     # If multiple blocks, warn and pick the last (newest) one
#     if len(blocks) > 1:
#         print(
#             f"WARNING: Found {len(blocks)} distinct snapshot blocks for sim_id={args.sim_id}. "
#             f"Plotting the LAST block only (newest)."
#         )
#     df = blocks[-1]

#     # Grab radius (constant per sim within a block; use median)
#     rr = df["r"].median() if "r" in df.columns and not df["r"].isna().all() else np.nan

#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.scatter(df["x"].to_numpy(), df["y"].to_numpy(), s=args.s, alpha=args.alpha)
#     ax.set_aspect("equal", adjustable="box")
#     title = f"Final-timestep cells (sim_id={args.sim_id}"
#     if not np.isnan(rr):
#         title += f", r≈{rr:.2f}"
#     title += ")"
#     ax.set_title(title)
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.grid(False)

#     if args.save:
#         fn = args.filename or f"snapshot_sim_{args.sim_id}.png"
#         out_path = os.path.join(args.output_dir, fn)
#         plt.savefig(out_path, bbox_inches="tight", dpi=150)
#         print(f"Saved {out_path}")
#     else:
#         plt.show()

# if __name__ == "__main__":
#     main()


# /mnt/data/plot_snapshot.py
import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _read_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_parquet(path)

def _split_contiguous_blocks(df_with_row: pd.DataFrame, row_col: str = "_row"):
    if df_with_row.empty:
        return []
    idx = df_with_row[row_col].to_numpy()
    seps = np.where(np.diff(idx) != 1)[0] + 1
    cuts = np.r_[0, seps, len(idx)]
    blocks = [df_with_row.iloc[cuts[i]:cuts[i+1]] for i in range(len(cuts)-1)]
    return blocks

def main():
    ap = argparse.ArgumentParser(description="Plot final-timestep cells for a given sim_id.")
    ap.add_argument("--output-dir", default="output", help="Folder with snapshots.parquet & simulations.parquet")
    ap.add_argument("--sim-id", type=int, required=True, help="Simulation ID to plot")
    ap.add_argument("--save", action="store_true", help="Save PNG instead of showing")
    ap.add_argument("--filename", default=None, help="Output PNG name (optional)")
    ap.add_argument("--size-scale", type=float, default=0.6,
                    help="Scale factor converting radius (µm) to matplotlib scatter size (points^2 area). "
                         "Dot area = (size_scale * r)^2. Default 0.6.")
    ap.add_argument("--alpha", type=float, default=0.7, help="Point alpha (transparency)")
    args = ap.parse_args()

    outdir = args.output_dir
    snap_path = os.path.join(outdir, "snapshots.parquet")
    sim_path  = os.path.join(outdir, "simulations.parquet")

    # Load
    snaps = _read_parquet(snap_path).reset_index().rename(columns={"index": "_row"})
    sims  = _read_parquet(sim_path)

    # Filter snapshots for sim_id
    df_all = snaps[snaps["sim_id"] == args.sim_id].sort_values("_row")
    if df_all.empty:
        sys.exit(f"No snapshot rows found for sim_id={args.sim_id} in {snap_path}")

    # If duplicate blocks exist, warn & take the last (newest)
    blocks = _split_contiguous_blocks(df_all, row_col="_row")
    if len(blocks) > 1:
        print(f"WARNING: Found {len(blocks)} snapshot blocks for sim_id={args.sim_id}. Plotting the LAST block only.")
    df = blocks[-1]

    # Get parameters from simulations.parquet (one row per sim)
    sim_row = sims[sims["sim_id"] == args.sim_id]
    if sim_row.empty:
        print(f"WARNING: sim_id={args.sim_id} not found in {sim_path}. "
              f"Using radius from snapshots only.")
        r = float(np.median(df["r"])) if "r" in df.columns and not df["r"].isna().all() else 5.0
        D = np.nan
        alpha = np.nan
    else:
        sim_row = sim_row.iloc[0]
        r = float(sim_row["r"])
        D = float(10.0 ** float(sim_row["log10_d"]))
        alpha = float(10.0 ** float(sim_row["log10_alpha"]))

    # Matplotlib scatter size is in points^2 (area). Use (size_scale * r)^2.
    s_pts2 = (args.size_scale * r) ** 2

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["x"].to_numpy(), df["y"].to_numpy(), s=s_pts2, alpha=args.alpha)
    ax.set_aspect("equal", adjustable="box")
    # Title with parameters
    if np.isfinite(D) and np.isfinite(alpha):
        title = f"Final cells (sim_id={args.sim_id}) — r={r:.2f}, D={D:.3g}, alpha={alpha:.3g}"
    else:
        title = f"Final cells (sim_id={args.sim_id}) — r={r:.2f}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)

    if args.save:
        fn = args.filename or f"snapshot_sim_{args.sim_id}.png"
        out_path = os.path.join(outdir, fn)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
