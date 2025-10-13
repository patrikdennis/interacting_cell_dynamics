#!/usr/bin/env python3
import os
import sys
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import cv2
import numpy as np

# helpers: label handling & robust centers

def as_label_image(img: np.ndarray) -> np.ndarray:
    """
    Ensure we have a single-channel integer label image.
    If img is 3-channel but channels are identical, take the first channel.
    Raises if channels differ (i.e., not a label mask).
    """
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if np.all(img[..., 0] == img[..., 1]) and np.all(img[..., 1] == img[..., 2]):
            return img[..., 0]
        raise ValueError("3-channel image with differing channels; need a single-channel labeled mask.")
    raise ValueError("Unsupported image ndim.")

def _robust_center_from_binary(mask2d: np.ndarray, method: str, alpha: float, erode_r: int):
    """
    Compute a robust center (cx, cy) for a single connected object mask (uint8 {0,1}).
    Methods:
      - edt_peak: farthest-from-boundary point (distance transform argmax)
      - edt_weighted: centroid weighted by distance^alpha
      - erode_centroid: erode by r and take ordinary centroid
    Returns floats (cx, cy) in mask coordinates; None if empty.
    """
    m = (mask2d > 0).astype(np.uint8)
    if m.sum() == 0:
        return None

    if method == "edt_peak":
        dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=3)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        return float(x), float(y)

    if method == "edt_weighted":
        dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=3)
        w = np.power(dist, max(1.0, float(alpha)))
        w[m == 0] = 0.0
        ys, xs = np.nonzero(m)
        wv = w[ys, xs]
        if wv.sum() <= 0:
            # fallback: ordinary centroid
            return float(xs.mean()), float(ys.mean())
        cx = np.average(xs, weights=wv)
        cy = np.average(ys, weights=wv)
        return float(cx), float(cy)

    if method == "erode_centroid":
        k = max(1, 2 * int(erode_r) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        eroded = cv2.erode(m, kernel, iterations=1)
        src = eroded if eroded.sum() > 0 else m
        ys, xs = np.nonzero(src)
        return float(xs.mean()), float(ys.mean())

    raise ValueError("method must be one of {'edt_peak','edt_weighted','erode_centroid'}")


def colorize_label_viridis(label_img: np.ndarray) -> np.ndarray:
    """
    Map a 2D integer label image to a viridis-colored BGR uint8 image.
    Background 0 -> low end of viridis. Per-image normalization.
    """
    if label_img.ndim != 2:
        raise ValueError("colorize_label_viridis expects a 2D label image.")

    lab = label_img.astype(np.float32)
    m = float(lab.max())
    if m <= 0:
        scaled = np.zeros_like(lab, dtype=np.uint8)
    else:
        scaled = np.clip(lab / m * 255.0, 0, 255).astype(np.uint8)

    # BGR image with viridis colormap
    return cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)


def robust_centroids_from_labels(
    label_img: np.ndarray,
    method: str = "edt_weighted",
    alpha: float = 2.0,
    erode_r: int = 5,
    pad: int = 3
):
    """
    Compute one robust center per unique label (>0). Returns list of (cx, cy) floats.
    Efficient: processes each label in a tight padded bounding box.
    """
    if label_img.ndim != 2:
        raise ValueError("robust_centroids_from_labels expects a 2D label image.")
    labels = np.unique(label_img)
    labels = labels[labels > 0]
    H, W = label_img.shape

    centers = []
    for L in labels:
        ys, xs = np.where(label_img == L)
        if xs.size == 0:
            continue
        x1 = max(0, xs.min() - pad); x2 = min(W - 1, xs.max() + pad)
        y1 = max(0, ys.min() - pad); y2 = min(H - 1, ys.max() + pad)
        local = (label_img[y1:y2+1, x1:x2+1] == L).astype(np.uint8)

        rc = _robust_center_from_binary(local, method=method, alpha=alpha, erode_r=erode_r)
        if rc is None:
            continue
        cx_loc, cy_loc = rc
        centers.append((cx_loc + x1, cy_loc + y1))
    return centers

# image generation (features & overlay)

def make_feature_image(centers, shape, square_size: int = 5):
    """
    Black background (uint8) with white squares centered at each (cx, cy).
    shape: (H, W)
    """
    H, W = shape
    out = np.zeros((H, W), dtype=np.uint8)
    half = square_size // 2
    for (cx, cy) in centers:
        x = int(round(cx)); y = int(round(cy))
        x1 = max(0, x - half); y1 = max(0, y - half)
        x2 = min(W - 1, x + half); y2 = min(H - 1, y + half)
        out[y1:y2+1, x1:x2+1] = 255
    return out

def make_debug_overlay(label_img: np.ndarray, centers, square_size: int = 5):
    """
    Colorize the label image with viridis and draw white squares on top.
    Returns BGR uint8.
    """
    # Build colorful background (viridis)
    canvas = colorize_label_viridis(label_img)  # BGR uint8

    # Draw white squares
    white = (255, 255, 255)
    H, W = canvas.shape[:2]
    half = square_size // 2
    for (cx, cy) in centers:
        x = int(round(cx)); y = int(round(cy))
        x1 = max(0, x - half); y1 = max(0, y - half)
        x2 = min(W - 1, x + half); y2 = min(H - 1, y + half)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), white, thickness=cv2.FILLED)
    return canvas

#  worker & CLI 

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

def process_one(
    image_path: str,
    out_dir: str,
    method: str,
    alpha: float,
    erode_r: int,
    square_size: int
):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return f"FAIL: {os.path.basename(image_path)} -> could not read"

        # label image for centers
        label_img = as_label_image(img)

        centers = robust_centroids_from_labels(
            label_img, method=method, alpha=alpha, erode_r=erode_r, pad=4
        )

        # outputs
        H, W = label_img.shape
        feat = make_feature_image(centers, (H, W), square_size=square_size)
        overlay = make_debug_overlay(label_img= label_img, centers = centers, square_size=square_size)

        # save
        base = os.path.basename(image_path)
        p_debug = os.path.join(out_dir, "debug_overlay")
        p_feat  = os.path.join(out_dir, "features")
        os.makedirs(p_debug, exist_ok=True)
        os.makedirs(p_feat,  exist_ok=True)

        # compression for TIFF, ignored by others
        comp = [cv2.IMWRITE_TIFF_COMPRESSION, 5]

        cv2.imwrite(os.path.join(p_debug, base), overlay, comp)
        cv2.imwrite(os.path.join(p_feat,  base), feat, comp)

        return f"OK: {base} (centers={len(centers)})"
    except Exception as e:
        return f"FAIL: {os.path.basename(image_path)} -> {e}"

def main():
    ap = argparse.ArgumentParser(
        description="Compute robust centroids from labeled images and write debug overlays + feature images."
    )
    ap.add_argument("--folder", required=True, help="Input folder containing labeled images.")
    ap.add_argument("--out-dir", required=True, help="Output base directory (will create debug_overlay/ and features/).")
    ap.add_argument("--method", choices=["edt_weighted", "edt_peak", "erode_centroid"], default="edt_weighted",
                    help="Robust centroid method.")
    ap.add_argument("--alpha", type=float, default=2.0, help="Alpha for edt_weighted (weight = distance^alpha).")
    ap.add_argument("--erode-r", type=int, default=6, help="Erosion radius (pixels) for erode_centroid.")
    ap.add_argument("--square-size", type=int, default=5, help="Square size in pixels for drawing at centroids.")
    ap.add_argument("--workers", type=int, default=0, help="Number of worker processes (0=auto).")
    args = ap.parse_args()

    folder = args.folder
    out_dir = args.out_dir

    # gather images
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    paths = sorted(paths)
    if not paths:
        print(f"No images found in {folder}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    print(f"Found {len(paths)} images. Writing to: {out_dir}")
    print(f"Subfolders: {os.path.join(out_dir, 'debug_overlay')} and {os.path.join(out_dir, 'features')}")

    max_workers = None if args.workers == 0 else args.workers
    worker = partial(
        process_one,
        out_dir=out_dir,
        method=args.method,
        alpha=args.alpha,
        erode_r=args.erode_r,
        square_size=args.square_size,
    )

    # parallel execution
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, p) for p in paths]
        for fut in as_completed(futures):
            print(fut.result())

if __name__ == "__main__":
    main()
