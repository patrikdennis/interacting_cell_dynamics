"""
Create consistent, visualizations and tables from an segmentation evaluation (folder).

Assumes the evaluation script created in EVAL_DIR:
  - summary_global.csv
  - summary_per_image.csv

Outputs in EVAL_DIR:
  - global_metrics.tex         (LaTeX table of global metrics)
  - f1_per_image.png           (horizontal bar chart)
  - precision_recall.png       (scatter plot)
  - fp_fn_per_image.png        (stacked FP/FN bar chart)
  - confusion_matrix.png       ("confusion-matrix-style" figure)
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Color palette (Xcode-ish: purple, pink, light green)
# ---------------------------------------------------------------------

COLORS = {
    "purple": "#a277ff",
    "pink": "#ff6ac1",
    "green": "#7bd88f",
    "blue": "#82aaff",
    "background": "#f9f9fb",
    "grid": "#d0d3e0",
    "text": "#262738",
}


# ---------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------

def set_matplotlib_style():
    """
    Configure a clean, modern style with a unified color palette.
    """
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.facecolor": "white",
            "axes.facecolor": COLORS["background"],
            "axes.edgecolor": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.6,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
        }
    )


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

def load_results(eval_dir: Path):
    global_csv = eval_dir / "summary_global.csv"
    per_image_csv = eval_dir / "summary_per_image.csv"

    if not global_csv.exists():
        raise FileNotFoundError(f"Missing {global_csv}")
    if not per_image_csv.exists():
        raise FileNotFoundError(f"Missing {per_image_csv}")

    df_global = pd.read_csv(global_csv)
    df_per = pd.read_csv(per_image_csv)

    # Usually df_global has a single row; just in case, take the first.
    if len(df_global) > 1:
        df_global = df_global.iloc[[0]]

    return df_global, df_per


# ---------------------------------------------------------------------
# Global table
# ---------------------------------------------------------------------

def make_global_table(df_global: pd.DataFrame, eval_dir: Path, method_name: str):
    """
    Format and export global metrics as a clean LaTeX table.
    """
    df = df_global.copy()

    cols = ["TP", "FP", "FN", "precision", "recall", "f1"]
    df = df[cols]

    df_rounded = df.copy()
    df_rounded["precision"] = df["precision"].map(lambda x: f"{x:.3f}")
    df_rounded["recall"] = df["recall"].map(lambda x: f"{x:.3f}")
    df_rounded["f1"] = df["f1"].map(lambda x: f"{x:.3f}")

    df_rounded = df_rounded.rename(
        columns={
            "TP": "TP",
            "FP": "FP",
            "FN": "FN",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-score",
        }
    )

    print("\n=== Global segmentation metrics ===")
    print(df_rounded.to_string(index=False))

    latex_path = eval_dir / "global_metrics.tex"
    latex_table = df_rounded.to_latex(
        index=False,
        escape=False,
        caption=f"Global detection metrics for {method_name}.",
        label="tab:seg_global_metrics",
        column_format="rrrrrr",
    )
    latex_path.write_text(latex_table)
    print(f"\nSaved LaTeX table to {latex_path}")


# ---------------------------------------------------------------------
# Helpers for per-image plots
# ---------------------------------------------------------------------

def shorten_image_name(name: str) -> str:
    """
    Example shortening: "VID558_A1_1_00d02h00m.tif" -> "00d02h00m".
    Adjust if you want something else.
    """
    parts = name.split("_")
    if len(parts) >= 4:
        short = parts[-1]
    else:
        short = name
    return short.replace(".tif", "")


# ---------------------------------------------------------------------
# F1 per image (horizontal bars)
# ---------------------------------------------------------------------

def plot_f1_per_image(df_per: pd.DataFrame, eval_dir: Path, method_name: str):
    df = df_per.copy()
    df = df.sort_values("f1", ascending=True)

    df["short_name"] = df["image_name"].apply(shorten_image_name)

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.18)))

    bars = ax.barh(
        df["short_name"],
        df["f1"],
        alpha=0.9,
        color=COLORS["purple"],
    )

    ax.set_xlabel("F1-score")
    ax.set_ylabel("Image")
    ax.set_title(f"Per-image F1-score for {method_name}")
    ax.set_xlim(0, 1.0)

    for bar, val in zip(bars, df["f1"]):
        ax.text(
            val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=9,
            color=COLORS["text"],
        )

    fig.tight_layout()
    out_path = eval_dir / "f1_per_image.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved F1 per-image plot to {out_path}")


# ---------------------------------------------------------------------
# Precision–recall scatter
# ---------------------------------------------------------------------

def plot_precision_recall(df_per: pd.DataFrame, eval_dir: Path, method_name: str):
    df = df_per.copy()

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        df["recall"],
        df["precision"],
        s=45,
        alpha=0.9,
        color=COLORS["pink"],
        edgecolor="white",
        linewidth=0.6,
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall per image ({method_name})")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)

    # Reference lines
    for v in [0.2, 0.4, 0.6, 0.8]:
        ax.axhline(v, color=COLORS["grid"], linestyle="--", linewidth=0.7, alpha=0.7)
        ax.axvline(v, color=COLORS["grid"], linestyle="--", linewidth=0.7, alpha=0.7)

    fig.tight_layout()
    out_path = eval_dir / "precision_recall.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved precision–recall scatter to {out_path}")


# ---------------------------------------------------------------------
# FP / FN per image (stacked bars)
# ---------------------------------------------------------------------

def plot_fp_fn_per_image(df_per: pd.DataFrame, eval_dir: Path, method_name: str):
    df = df_per.copy()
    df = df.sort_values("num_gt", ascending=False)

    df["short_name"] = df["image_name"].apply(shorten_image_name)

    x = np.arange(len(df))
    width = 0.6

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.18)))

    fp_bars = ax.bar(
        x,
        df["FP"],
        width,
        label="False positives",
        color=COLORS["pink"],
        alpha=0.9,
    )
    fn_bars = ax.bar(
        x,
        df["FN"],
        width,
        bottom=df["FP"],
        label="False negatives",
        color=COLORS["purple"],
        alpha=0.9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["short_name"], rotation=90)
    ax.set_ylabel("Count")
    ax.set_title(f"False positives and false negatives per image ({method_name})")
    ax.legend()

    fig.tight_layout()
    out_path = eval_dir / "fp_fn_per_image.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved FP/FN per-image plot to {out_path}")


# ---------------------------------------------------------------------
# Confusion-matrix-style figure
# ---------------------------------------------------------------------

def plot_confusion_matrix_like(df_global: pd.DataFrame, eval_dir: Path, method_name: str):
    """
    For detection we don't have a meaningful TN, so we show:
        TP, FP, FN in a 2x2 layout and mark TN as "N/A".

    Layout (rows = GT, cols = prediction):

                  Predicted cell   Predicted none
        GT cell        TP              FN
        GT none        FP              N/A
    """
    row = df_global.iloc[0]
    tp = int(row["TP"])
    fp = int(row["FP"])
    fn = int(row["FN"])

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.set_axis_off()

    cell_text = [
        [f"TP\n{tp}", f"FN\n{fn}"],
        [f"FP\n{fp}", "N/A"],
    ]
    cell_colors = [
        [COLORS["green"], COLORS["purple"]],
        [COLORS["pink"], "#e0e1ee"],
    ]

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        colLabels=["Predicted cell", "Predicted none"],
        rowLabels=["GT cell", "GT none"],
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for (row_i, col_i), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(1.5)

    ax.set_title(f"Confusion-style summary for {method_name}", pad=20)

    fig.tight_layout()
    out_path = eval_dir / "confusion_matrix.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved confusion-matrix-style figure to {out_path}")


# ---------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate visual reports from an evaluation folder."
    )
    ap.add_argument(
        "--eval-dir",
        type=str,
        default="eval_results",
        help="Path to evaluation folder containing summary_global.csv and summary_per_image.csv "
             "(default: eval_results).",
    )
    ap.add_argument(
        "--method-name",
        type=str,
        default=None,
        help="Name of the segmentation method (used in titles/captions). "
             "If omitted, the eval folder name is used.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    method_name = args.method_name or eval_dir.name

    set_matplotlib_style()
    df_global, df_per = load_results(eval_dir)

    make_global_table(df_global, eval_dir, method_name)
    plot_f1_per_image(df_per, eval_dir, method_name)
    plot_precision_recall(df_per, eval_dir, method_name)
    plot_fp_fn_per_image(df_per, eval_dir, method_name)
    plot_confusion_matrix_like(df_global, eval_dir, method_name)


if __name__ == "__main__":
    main()
