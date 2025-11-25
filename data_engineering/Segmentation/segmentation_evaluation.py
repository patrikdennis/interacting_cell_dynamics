"""
Evaluate cell detection from label masks against click-based ground truth.

Ground truth:
    - CSV with columns: image_name, cell_id, x, y[, r]
      (your GUI currently writes at least: image_name,cell_id,x,y,r)

Predictions:
    - For each image, a label mask where:
        0 = background
        1,2,... = individual segments (cells or blobs)
    - The mask files live in a LABELS_DIR, either:
        LABELS_DIR / image_name
      or
        LABELS_DIR / (image_stem + label_suffix)

The script:
    - Only evaluates images that are present in the CSV AND have a mask file.
    - For each image, matches GT points to predicted segments using
      the **distance to the closest pixel of each segment**, not centroids.
    - Each segment label can be used at most once (one big blob over 3 cells
      → 1 TP + 2 FN).
    - Unmatched GT → FN; unused segment --> FP.

Outputs:
    - summary_per_image.csv    : TP/FP/FN/precision/recall/F1 per image
    - summary_global.csv       : aggregated metrics
    - matches.csv              : one row per GT click with assigned label id

You can later manually edit matches.csv (or use the GUI) and recompute metrics.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------
# Loading ground truth and label masks
# ---------------------------------------------------------------------

def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    """
    Load click-based ground truth.
    Returns a DataFrame with columns:
        image_name, cell_id, x, y
    'r' is kept if present but unused in the basic matching.
    """
    df = pd.read_csv(csv_path)
    required = {"image_name", "cell_id", "x", "y"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}, got {df.columns}")
    return df


def load_label_mask(labels_dir: Path, image_name: str, label_suffix: str = ""):
    """
    Try to load the label mask for a given image.

    We try, in order:
      1) labels_dir / image_name
      2) labels_dir / (stem + label_suffix)

    Returns:
        2D numpy array (int32) or None if not found.

    This version uses PIL instead of tifffile, so it does not require
    the 'imagecodecs' package for LZW-compressed TIFFs.
    """
    p1 = labels_dir / image_name
    stem = Path(image_name).stem
    p2 = labels_dir / f"{stem}{label_suffix}"

    for p in (p1, p2):
        if p.exists():
            img = Image.open(str(p))
            arr = np.array(img)

            # If there are multiple channels, take the first one
            if arr.ndim == 3:
                arr = arr[..., 0]

            if arr.ndim != 2:
                raise ValueError(f"Label mask {p} must be 2D, got shape {arr.shape}")

            return arr.astype(np.int32)

    return None


# ---------------------------------------------------------------------
# Matching: distance to **closest mask pixel**, not centroid
# ---------------------------------------------------------------------

def match_points_to_segments(
    points_xy,
    label_img: np.ndarray,
    distance_threshold: float = 10.0,
):
    """
    Match ground-truth points (x,y) to segments in label_img.

    IMPORTANT: This uses the distance to the **closest pixel** of each segment,
    NOT the segment centroid.

    - points_xy: array-like of shape (N, 2): (x, y) ground-truth clicks
    - label_img: 2D array of ints (0 = background, >0 = segment labels)
    - distance_threshold: maximum allowed distance (in pixels) between a point
      and the nearest pixel of a segment to consider a match.

    Behaviour:
      - Each segment label can be matched at most once (one-to-one matching).
      - If no segment is within distance_threshold, the point is unmatched (FN).
      - Segments that are never matched count as FP (handled later).

    Returns:
      matches: list of dicts, one per GT point:
          {
            "x": float,
            "y": float,
            "gt_index": int,
            "matched_label": int,          # -1 if no match
            "distance_to_segment": float,  # np.inf if no match
          }
      used_labels: set of segment labels that were matched at least once.
    """
    # All segment labels except background (0)
    labels = np.unique(label_img)
    labels = labels[labels != 0]

    # Precompute pixel coordinates (x,y) for each label
    label_to_coords = {}
    for lbl in labels:
        ys, xs = np.where(label_img == lbl)
        if xs.size == 0:
            coords = np.zeros((0, 2), dtype=np.float32)
        else:
            coords = np.stack([xs, ys], axis=1).astype(np.float32)  # (N,2) as (x,y)
        label_to_coords[lbl] = coords

    used_labels = set()
    matches = []

    for gt_idx, (x_gt, y_gt) in enumerate(points_xy):
        pt = np.array([x_gt, y_gt], dtype=np.float32)

        best_label = -1
        best_dist2 = float("inf")

        # search for the closest segment (by nearest pixel),
        # but only among labels that are not yet used
        for lbl, coords in label_to_coords.items():
            if lbl in used_labels:
                continue
            if coords.shape[0] == 0:
                continue

            # squared distances from point to all pixels of this segment
            diff = coords - pt  # (N,2)
            d2 = np.sum(diff * diff, axis=1)
            min_d2 = float(d2.min())

            if min_d2 < best_dist2:
                best_dist2 = min_d2
                best_label = lbl

        if best_label != -1:
            dist = float(np.sqrt(best_dist2))
            if dist <= distance_threshold:
                # accept match
                used_labels.add(best_label)
                matches.append(
                    {
                        "x": x_gt,
                        "y": y_gt,
                        "gt_index": gt_idx,
                        "matched_label": best_label,
                        "distance_to_segment": dist,
                    }
                )
            else:
                # nearest segment is too far away -> treat as FN
                matches.append(
                    {
                        "x": x_gt,
                        "y": y_gt,
                        "gt_index": gt_idx,
                        "matched_label": -1,
                        "distance_to_segment": float("inf"),
                    }
                )
        else:
            # no segments at all (or all already used)
            matches.append(
                {
                    "x": x_gt,
                    "y": y_gt,
                    "gt_index": gt_idx,
                    "matched_label": -1,
                    "distance_to_segment": float("inf"),
                }
            )

    return matches, used_labels


# ---------------------------------------------------------------------
# Metrics per image and globally
# ---------------------------------------------------------------------

def compute_metrics_for_image(
    gt_points: pd.DataFrame,
    label_img: np.ndarray,
    dist_thresh: float,
    image_name: str,
):
    """
    Compute TP, FP, FN for a single image and return detailed matches.

    gt_points: DataFrame subset for this image (columns x,y,cell_id,...)
    label_img: 2D label mask
    dist_thresh: distance threshold for matching
    image_name: string

    Returns:
      metrics: dict with TP, FP, FN, precision, recall, f1, image_name
      matches_df: DataFrame with one row per GT point (plus some info)
    """
    if len(gt_points) == 0:
        return None, None

    pts_xy = gt_points[["x", "y"]].to_numpy()

    matches, used_labels = match_points_to_segments(
        pts_xy, label_img, distance_threshold=dist_thresh
    )

    num_gt = len(matches)
    num_pred = int(label_img.max())  # labels 1..max_label

    tp = sum(1 for m in matches if m["matched_label"] != -1)
    fn = num_gt - tp
    fp = num_pred - len(used_labels)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    metrics = {
        "image_name": image_name,
        "num_gt": num_gt,
        "num_pred": num_pred,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    rows = []
    for gt_row, m in zip(gt_points.itertuples(index=False), matches):
        rows.append(
            {
                "image_name": image_name,
                "cell_id": gt_row.cell_id,
                "x": m["x"],
                "y": m["y"],
                "matched_label": m["matched_label"],
                "distance_to_segment": m["distance_to_segment"],
            }
        )

    matches_df = pd.DataFrame(rows)
    return metrics, matches_df


def aggregate_metrics(metrics_list):
    """
    Aggregate per-image metrics into a global summary.

    We re-count TP, FP, FN from all images, then recompute
    precision/recall/F1 globally.
    """
    tp = sum(m["TP"] for m in metrics_list)
    fp = sum(m["FP"] for m in metrics_list)
    fn = sum(m["FN"] for m in metrics_list)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate segmentation label masks against click-based cell ground truth."
    )
    ap.add_argument("--gt-csv", type=str, required=True,
                    help="Path to cell_labels.csv from the GUI.")
    ap.add_argument("--labels-dir", type=str, required=True,
                    help="Directory containing segmentation label masks.")
    ap.add_argument("--label-suffix", type=str, default="",
                    help="Optional suffix for label files, e.g. '_labels.tif'. "
                         "If empty, we first try labels_dir/image_name, "
                         "then labels_dir/(image_stem + label_suffix).")
    ap.add_argument("--distance-threshold", type=float, default=10.0,
                    help="Maximum distance (pixels) from click to nearest mask pixel "
                         "to consider a prediction matching a GT point.")
    ap.add_argument("--out-dir", type=str, default="eval_results",
                    help="Directory where summary and matches CSVs will be written.")
    args = ap.parse_args()

    gt_csv = Path(args.gt_csv)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_df = load_ground_truth(gt_csv)

    image_names = sorted(gt_df["image_name"].unique())

    all_metrics = []
    all_matches = []

    for img_name in image_names:
        mask = load_label_mask(labels_dir, img_name, label_suffix=args.label_suffix)
        if mask is None:
            print(f"[WARNING] No label mask found for {img_name}, skipping.")
            continue

        gt_points = gt_df[gt_df["image_name"] == img_name].reset_index(drop=True)
        metrics, matches_df = compute_metrics_for_image(
            gt_points,
            mask,
            dist_thresh=args.distance_threshold,
            image_name=img_name,
        )
        if metrics is None:
            continue

        all_metrics.append(metrics)
        all_matches.append(matches_df)

    if not all_metrics:
        print("No images with both ground truth and masks were found. Nothing to evaluate.")
        return

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / "summary_per_image.csv", index=False)

    global_metrics = aggregate_metrics(all_metrics)
    global_df = pd.DataFrame([global_metrics])
    global_df.to_csv(out_dir / "summary_global.csv", index=False)

    matches_df = pd.concat(all_matches, ignore_index=True)
    matches_df.to_csv(out_dir / "matches.csv", index=False)

    print("\nPer-image metrics:")
    print(metrics_df)

    print("\nGlobal metrics:")
    print(global_df)

    print(f"\nSaved per-image summary to {out_dir/'summary_per_image.csv'}")
    print(f"Saved global summary to {out_dir/'summary_global.csv'}")
    print(f"Saved detailed matches to {out_dir/'matches.csv'}")


if __name__ == "__main__":
    main()
