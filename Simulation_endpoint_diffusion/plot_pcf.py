# # /mnt/data/plot_pcf.py
# import argparse
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt


# def load_data(output_dir):
#     radii = pd.read_csv(os.path.join(output_dir, "radii.csv"))
#     sims = pd.read_csv(os.path.join(output_dir, "simulations.csv"))
#     return radii, sims


# def extract_pcf_matrix(sims: pd.DataFrame):
#     pcf_cols = [c for c in sims.columns if c.startswith("pc_")]
#     return sims[pcf_cols].to_numpy(dtype=float), pcf_cols


# def main():
#     parser = argparse.ArgumentParser(description="Plot PCF from simulation outputs.")
#     parser.add_argument(
#         "--output-dir",
#         default="output",
#         help="Directory containing radii.csv and simulations.csv (default: output)",
#     )
#     parser.add_argument(
#         "--save",
#         action="store_true",
#         help="Save figures as PNGs in the output directory instead of showing them",
#     )
#     args = parser.parse_args()

#     radii_df, sims_df = load_data(args.output_dir)

#     # Use only valid runs if exit_code column exists
#     if "exit_code" in sims_df.columns:
#         sims_df = sims_df[sims_df["exit_code"].isin([0, 1])].reset_index(drop=True)

#     pcf_matrix, pcf_cols = extract_pcf_matrix(sims_df)
#     r = radii_df["radius"].to_numpy(dtype=float)

#     if pcf_matrix.size == 0:
#         print("No PCF data found to plot.")
#         return

#     # 1) Mean PCF with 5–95% percentile band
#     mean_pcf = np.nanmean(pcf_matrix, axis=0)
#     lo = np.nanpercentile(pcf_matrix, 5, axis=0)
#     hi = np.nanpercentile(pcf_matrix, 95, axis=0)

#     plt.figure()
#     plt.plot(r, mean_pcf, label="Mean PCF")
#     plt.fill_between(r, lo, hi, alpha=0.2, label="5–95%")
#     plt.xlabel("Radius")
#     plt.ylabel("g(r)")
#     plt.title("Pair Correlation Function (mean ± 5–95%)")
#     plt.legend()
#     if args.save:
#         out_path = os.path.join(args.output_dir, "pcf_mean_band.png")
#         plt.savefig(out_path, bbox_inches="tight", dpi=150)
#         print(f"Saved {out_path}")
#         plt.close()
#     else:
#         plt.show()

#     #  Example single-run PCF (first row)
#     plt.figure()
#     plt.plot(r, pcf_matrix[0, :], label="Example run")
#     plt.xlabel("Radius")
#     plt.ylabel("g(r)")
#     plt.title("Pair Correlation Function (single run)")
#     plt.legend()
#     if args.save:
#         out_path = os.path.join(args.output_dir, "pcf_single_example.png")
#         plt.savefig(out_path, bbox_inches="tight", dpi=150)
#         print(f"Saved {out_path}")
#         plt.close()
#     else:
#         plt.show()


# if __name__ == "__main__":
#     main()

# /mnt/data/plot_pcf.py
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _read_table(path_parquet: str, path_csv: str) -> pd.DataFrame:
    """Read Parquet if present, else CSV."""
    if os.path.exists(path_parquet):
        # Requires pyarrow (recommended)
        return pd.read_parquet(path_parquet)
    elif os.path.exists(path_csv):
        return pd.read_csv(path_csv)
    else:
        raise FileNotFoundError(f"Neither {path_parquet} nor {path_csv} found.")


def load_data(output_dir):
    radii = _read_table(
        os.path.join(output_dir, "radii.parquet"),
        os.path.join(output_dir, "radii.csv"),
    )
    sims = _read_table(
        os.path.join(output_dir, "simulations.parquet"),
        os.path.join(output_dir, "simulations.csv"),
    )
    return radii, sims


def extract_pcf_matrix(sims: pd.DataFrame):
    pcf_cols = [c for c in sims.columns if c.startswith("pc_")]
    return sims[pcf_cols].to_numpy(dtype=float), pcf_cols


def main():
    parser = argparse.ArgumentParser(description="Plot PCF from simulation outputs.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory containing radii.parquet/simulations.parquet (or CSV fallback).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figures as PNGs in the output directory instead of showing them",
    )
    args = parser.parse_args()

    radii_df, sims_df = load_data(args.output_dir)

    # Use only valid runs if exit_code column exists
    if "exit_code" in sims_df.columns:
        sims_df = sims_df[sims_df["exit_code"].isin([0, 1])].reset_index(drop=True)

    pcf_matrix, _pcf_cols = extract_pcf_matrix(sims_df)
    r = radii_df["radius"].to_numpy(dtype=float)

    if pcf_matrix.size == 0:
        print("No PCF data found to plot.")
        return

    # 1) Mean PCF with 5–95% percentile band
    mean_pcf = np.nanmean(pcf_matrix, axis=0)
    lo = np.nanpercentile(pcf_matrix, 5, axis=0)
    hi = np.nanpercentile(pcf_matrix, 95, axis=0)
    
    plt.figure()
    plt.plot(r, mean_pcf, label="Mean PCF")
    plt.fill_between(r, lo, hi, alpha=0.2, label="5-95%")
    plt.xlabel("Radius")
    plt.ylabel("g(r)")
    plt.title("Pair Correlation Function (mean ± 5-95%)")
    plt.legend()
    plt.xlim(0, r[-1])     # start x at 0 for the axis
    plt.ylim(0, None)      # start y at 0

    # plt.figure()
    # plt.plot(r, mean_pcf, label="Mean PCF")
    # plt.fill_between(r, lo, hi, alpha=0.2, label="5-95%")
    # plt.xlabel("Radius")
    # plt.ylabel("g(r)")
    # plt.title("Pair Correlation Function (mean ± 5-95%)")
    # plt.legend()
    if args.save:
        out_path = os.path.join(args.output_dir, "pcf_mean_band.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved {out_path}")
        plt.close()
    else:
        plt.show()
        
    #pcf_matrix[0, :]
    # 2) Example single-run PCF (first row)
    plt.figure()
    plt.plot(r, pcf_matrix[0, :], label="Example run")
    plt.xlabel("Radius")
    plt.ylabel("g(r)")
    plt.title("Pair Correlation Function (single run)")
    plt.legend()
    if args.save:
        out_path = os.path.join(args.output_dir, "pcf_single_example.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved {out_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
