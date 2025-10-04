#!/usr/bin/env python3
"""
Contours with robust dark-object filtering + 1-pixel gray frame padding.

We pad the image with a constant gray border (default = median gray) to
stabilize edges. Processing is done on the padded image; contours are then
shifted back (un-padded) for overlay on the original image.

Usage (same as before, plus --pad/--pad-value):
  python plot_cell_contours_padded.py /path/to/image.tif \
    -o ./contours --pad 1 --pad-value auto
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from itertools import cycle

from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.measure import find_contours
from skimage.draw import polygon as draw_polygon
from scipy.ndimage import (
    gaussian_gradient_magnitude, gaussian_filter, map_coordinates
)


BRIGHT_COLORS = [
    "#ff3b30"
    # ,"#ff9500","#ffcc00","#34c759","#00c7be","#5ac8fa","#007aff",
    # "#5856d6","#af52de","#ff2d55","#ffd60a","#bf5af2","#64d2ff","#30d158",
    # "#ff9f0a","#ff375f","#0a84ff","#32d74b","#ffd426","#a2845e"
]

# ---------------- helpers ----------------

def load_grayscale(path: str) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3:
        arr = rgb2gray(arr)
    img = img_as_float(arr).astype(np.float32)
    if img.max() > 0:
        img = (img - img.min()) / max(1e-8, (img.max() - img.min()))
    return img

def shoelace_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x, y = poly[:, 1], poly[:, 0]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def sample_nearest(img: np.ndarray, coords: np.ndarray) -> np.ndarray:
    rr = np.clip(np.round(coords[:, 0]).astype(int), 0, img.shape[0]-1)
    cc = np.clip(np.round(coords[:, 1]).astype(int), 0, img.shape[1]-1)
    return img[rr, cc]

def sample_bilinear(img: np.ndarray, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
    return map_coordinates(img, np.vstack([rr, cc]), order=1, mode="nearest")

def choose_levels(img: np.ndarray, n: int, lo_q=5, hi_q=95) -> np.ndarray:
    lo = np.percentile(img, lo_q)
    hi = np.percentile(img, hi_q)
    if hi <= lo:
        hi, lo = img.max(), img.min()
        if hi == lo:
            hi = lo + 1e-3
    return np.linspace(lo, hi, n, endpoint=True)

def multiscale_grad_norm(img: np.ndarray, sigmas=(0.8, 1.2, 1.8, 2.5, 3.5)) -> np.ndarray:
    acc = None
    for s in sigmas:
        g = gaussian_gradient_magnitude(img, sigma=s)
        resp = s * g
        acc = resp if acc is None else np.maximum(acc, resp)
    m, M = np.percentile(acc, 1), np.percentile(acc, 99)
    return np.clip((acc - m) / max(1e-8, (M - m)), 0, 1)

def image_gradients(img: np.ndarray, sigma_for_deriv: float):
    gx = gaussian_filter(img, sigma=sigma_for_deriv, order=[0,1])  # d/dx
    gy = gaussian_filter(img, sigma=sigma_for_deriv, order=[1,0])  # d/dy
    return gx, gy

def is_near_closed(P: np.ndarray, eps=2.5) -> bool:
    return np.linalg.norm(P[0] - P[-1]) <= eps

def fill_poly_mask(shape, P):
    rr, cc = draw_polygon(P[:,0], P[:,1], shape=shape)
    m = np.zeros(shape, dtype=bool); m[rr, cc] = True
    return m

def prune_enclosing_contours(contours):
    keep = [True] * len(contours)
    paths = [MplPath(np.column_stack((c[:,1], c[:,0]))) for c in contours]
    samples = [c[len(c)//2, :] for c in contours]
    for i in range(len(contours)):
        if not keep[i]: continue
        Pi = paths[i]
        for j in range(len(contours)):
            if i == j or not keep[j]: continue
            pt = (samples[j][1], samples[j][0])
            if Pi.contains_point(pt, radius=0.0):
                keep[i] = False
                break
    return [c for c,k in zip(contours, keep) if k]

def pad_image(I, pad=1, pad_value="auto"):
    if pad <= 0:
        return I, 0, np.median(I)
    if pad_value == "auto":
        val = float(np.median(I))
    else:
        val = float(pad_value)
    Ip = np.pad(I, pad_width=((pad, pad), (pad, pad)), mode="constant", constant_values=val)
    return Ip, pad, val

def unpad_contour(P, pad, H, W):
    """Shift contour coords by -pad and clip to original image box [0,H-1]x[0,W-1]."""
    Q = P.copy()
    Q[:, 0] = Q[:, 0] - pad
    Q[:, 1] = Q[:, 1] - pad
    # keep only points that fall inside (or slightly crop the segments visually)
    mask = (Q[:,0] >= 0) & (Q[:,0] <= H-1) & (Q[:,1] >= 0) & (Q[:,1] <= W-1)
    if mask.sum() < 3:
        return None
    return Q[mask]


def draw_contours(
    image_path: str,
    out_dir: str = "./contours",
    out_filename = None,
    # padding
    pad: int = 1,
    pad_value: str = "auto",   # "auto" = image median; or numeric like "0.5"
    
    sigma: float = 1.4,
    n_levels: int = 18,
    grad_low: float = 0.018,
    grad_high: float = 0.045,
    grad_mean_min: float = 0.028,
    min_length: int = 30,
    min_area: float = 80.0,
    closed_min_area: float = 50.0,
    border_margin: int = 2,

    inside_max: float = 0.50,
    inside_percentile: float = 0.60,
    contrast_min: float = 0.020,
    dark_rel_min: float = 0.025,
    dark_abs: float = 0.015,
    min_dark_frac: float = 0.25,
    offset: float = 2.0,
    bg_sigma: float = 12.0,
    save_svg: bool = False,
    dark_rel_each: float = 0.020,
    min_dark_edge_cover: float = 0.60
):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    # load original (unpadded) for final overlay
    I0 = load_grayscale(image_path)
    H0, W0 = I0.shape

    # pad BEFORE everything else
    if pad_value != "auto":
        try:
            pv = float(pad_value)
        except Exception:
            pv = "auto"
    else:
        pv = "auto"
    I, pad_amt, used_val = pad_image(I0, pad=pad, pad_value=pv)

    # process the PADDED image
    S = gaussian(I, sigma=sigma, preserve_range=True)
    B = gaussian(S, sigma=bg_sigma, preserve_range=True)
    Gms = multiscale_grad_norm(S)
    gx, gy = image_gradients(S, sigma_for_deriv=max(0.8, sigma))

    levels = choose_levels(S, n_levels, lo_q=5, hi_q=95)

    candidates = []
    h, w = S.shape
    eps = 1e-8

    for lev in levels:
        for P in find_contours(S, level=lev):
            if P.shape[0] < min_length:
                if not (is_near_closed(P) and shoelace_area(P) >= closed_min_area):
                    continue

            if (
                (P[:,0] < border_margin).any() or (P[:,0] > h-1-border_margin).any() or
                (P[:,1] < border_margin).any() or (P[:,1] > w-1-border_margin).any()
            ):
                continue

            g_vals = sample_nearest(Gms, P)
            g_med = float(np.nanmedian(g_vals))
            g_max = float(np.nanmax(g_vals))
            keep_by_hyst = (g_med >= grad_low and g_max >= grad_high)
            keep_by_mean = (float(np.nanmean(g_vals)) >= grad_mean_min)
            if not (keep_by_hyst or keep_by_mean):
                continue

            rr, cc = P[:,0], P[:,1]
            gx_s = sample_bilinear(gx, rr, cc)
            gy_s = sample_bilinear(gy, rr, cc)
            mag  = np.hypot(gx_s, gy_s) + eps
            nx, ny = gx_s/mag, gy_s/mag

            rr_out = np.clip(rr + offset*ny, 0, h-1); cc_out = np.clip(cc + offset*nx, 0, w-1)
            rr_in  = np.clip(rr - offset*ny, 0, h-1); cc_in  = np.clip(cc - offset*nx, 0, w-1)
            Sout = sample_bilinear(S, rr_out, cc_out)
            Sin  = sample_bilinear(S, rr_in,  cc_in)

            signed_contrast = float(np.nanmedian(Sout - Sin))
            if signed_contrast < contrast_min:
                continue

            inside_stat = float(np.nanpercentile(Sin, inside_percentile*100.0))
            if inside_stat >= inside_max:
                continue

            Bout = sample_bilinear(B, rr_out, cc_out)
            dark_rel = float(np.nanmedian(Bout) - inside_stat)
            # if dark_rel < dark_rel_min:
            #     continue

            rel_each = Bout - Sin
            cover = float(np.nanmean(rel_each >= dark_rel_each))
            if cover < min_dark_edge_cover:
                continue

            if is_near_closed(P):
                area = shoelace_area(P)
                if area < min_area and area < closed_min_area:
                    continue
                m = fill_poly_mask((h, w), P.astype(np.float32))
                if m.sum() > 0:
                    dark_frac = float(np.mean(S[m] < (B[m] - dark_abs)))
                    if dark_frac < min_dark_frac:
                        continue

            candidates.append(P)

    kept_padded = prune_enclosing_contours(candidates)

    kept = []
    for P in kept_padded:
        Q = unpad_contour(P, pad_amt, H0, W0)
        if Q is not None and Q.shape[0] >= 3:
            kept.append(Q)

    S0 = gaussian(I0, sigma=sigma, preserve_range=True)

    H0, W0 = I0.shape
    dpi = 100.0
    fig_w = W0 / dpi
    fig_h = H0 / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(I0, cmap="gray")
    color_cycle = cycle(BRIGHT_COLORS)
    for P in kept:
        ax.plot(P[:, 1], P[:, 0], linewidth=0.9, color=next(color_cycle))
    ax.set_axis_off()

    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, out_filename)

    pil_kwargs = {}
    if png_path.lower().endswith((".tif", ".tiff")):
        pil_kwargs["compression"] = "tiff_lzw"   # lossless, widely supported

    plt.tight_layout(pad=0)
    fig.savefig(
        png_path,
        dpi=dpi,                  # ensures output pixels == (W0, H0)
        bbox_inches="tight",
        pad_inches=0,
        pil_kwargs=pil_kwargs     # no-op unless .tif/.tiff
    )
    plt.close(fig)

    if save_svg:
        svg_name = os.path.splitext(out_filename)[0] + ".svg"
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(I0, cmap="gray")
        for P in kept:
            ax.plot(P[:,1], P[:,0], linewidth=0.9)
        ax.set_axis_off()
        svg_path = os.path.join(out_dir, svg_name)
        plt.tight_layout(pad=0)
        fig.savefig(svg_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"SVG saved: {svg_path}")

    total_raw = sum(len(find_contours(S, l)) for l in levels)
    print(f"PNG saved: {png_path}")
    print(f"Contours kept (after unpad): {len(kept)} of {total_raw} raw.")
    print(f"Pad: {pad_amt}px @ value {used_val:.3f}")
    return png_path


IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

def process_folder(input_dir: str, out_parent: str, out_folder_name: str, **kwargs):
    """
    Iterate images in `input_dir`, run draw_contours on each, and save the overlay
    into `out_parent/out_folder_name` using the SAME filename (incl. extension)
    as the source image.

    Example:
      process_folder(
          "/Users/patrik/cell_diffusion_modelling/558/A1",
          "/Users/patrik/cell_diffusion_modelling/558",
          "A1_contoured",
          out_dir=None,  # ignored
          # plus any draw_contours kwargs you want to pass (sigma, levels, ...)
      )
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    out_dir = os.path.join(out_parent, out_folder_name)
    os.makedirs(out_dir, exist_ok=True)

    names = sorted(os.listdir(input_dir))
    count = 0
    for name in names:
        src = os.path.join(input_dir, name)
        if not os.path.isfile(src):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMAGE_EXTS:
            continue

        # Save overlay with the exact same filename in the new folder
        try:
            draw_contours(
                image_path=src,
                out_dir=out_dir,
                out_filename=name,   # same name as original (e.g., VID...47m.tif)
                **kwargs,
            )
            count += 1
        except Exception as e:
            print(f"Failed on {src}: {e}")

    print(f"Processed {count} images into: {out_dir}")
    return out_dir

def parse_args():
    ap = argparse.ArgumentParser(description="Contours with dark-object filters and 1px gray padding.")
    # make single-image arg optional so folder mode can be used
    ap.add_argument("image", nargs="?", help="Path to TIFF/PNG/JPG. Omit if using --folder.")
    ap.add_argument("-o", "--out-dir", default="./contours",
                    help="Output folder for single-image mode. Ignored when --folder is used.")

    ap.add_argument("--folder", help="Input folder containing images to process (non-recursive).")
    ap.add_argument("--out-parent", help="Parent directory where a new output folder will be created.")
    ap.add_argument("--out-folder-name", help="Name of the new output folder under --out-parent.")

    # padding
    ap.add_argument("--pad", type=int, default=8, help="Pixels of constant padding to add around the image.")
    ap.add_argument("--pad-value", default="auto",
                    help='"auto" (median gray) or a numeric value in [0,1] for the padding.')

    ap.add_argument("--sigma", type=float, default=1.4)
    ap.add_argument("--levels", type=int, default=18)
    ap.add_argument("--grad-low", type=float, default=0.018)
    ap.add_argument("--grad-high", type=float, default=0.045)
    ap.add_argument("--grad-mean-min", type=float, default=0.028)
    ap.add_argument("--min-length", type=int, default=30)
    ap.add_argument("--min-area", type=float, default=80.0)
    ap.add_argument("--closed-min-area", type=float, default=50.0)
    ap.add_argument("--border-margin", type=int, default=2)
    ap.add_argument("--inside-max", type=float, default=0.50)
    ap.add_argument("--inside-percentile", type=float, default=0.60)
    ap.add_argument("--contrast-min", type=float, default=0.020)
    ap.add_argument("--dark-rel-min", type=float, default=0.025)
    ap.add_argument("--dark-abs", type=float, default=0.015)
    ap.add_argument("--min-dark-frac", type=float, default=0.25)
    ap.add_argument("--offset", type=float, default=2.0)
    ap.add_argument("--bg-sigma", type=float, default=12.0)
    ap.add_argument("--save-svg", action="store_true")
    ap.add_argument("--dark-rel-each", type=float, default=0.020,
                help="Per-point local (B_out - inside) minimum gap.")
    ap.add_argument("--min-dark-edge-cover", type=float, default=0.60,
                help="Fraction of sampled edge points that must pass --dark-rel-each.")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # If folder mode is requested, run batch and save overlays with SAME filenames.
    if args.folder:
        # Validate required folder-mode args
        if not args.out_parent or not args.out_folder_name:
            raise ValueError("When using --folder, you must also provide --out-parent and --out-folder-name.")

        process_folder(
            input_dir=args.folder,
            out_parent=args.out_parent,
            out_folder_name=args.out_folder_name,
            pad=args.pad,
            pad_value=args.pad_value,
            sigma=args.sigma,
            n_levels=args.levels,
            grad_low=args.grad_low,
            grad_high=args.grad_high,
            grad_mean_min=args.grad_mean_min,
            min_length=args.min_length,
            min_area=args.min_area,
            closed_min_area=args.closed_min_area,
            border_margin=args.border_margin,
            inside_max=args.inside_max,
            inside_percentile=args.inside_percentile,
            contrast_min=args.contrast_min,
            dark_rel_min=args.dark_rel_min,
            dark_abs=args.dark_abs,
            min_dark_frac=args.min_dark_frac,
            offset=args.offset,
            bg_sigma=args.bg_sigma,
            save_svg=args.save_svg,
            dark_rel_each = args.dark_rel_each,
            min_dark_edge_cover = args.min_dark_edge_cover
            
        )

    # Otherwise fall back to single-image mode
    else:
        if not args.image:
            raise ValueError("Provide an IMAGE path, or use --folder with --out-parent and --out-folder-name.")
        draw_contours(
            image_path=args.image,
            out_dir=args.out_dir,
            pad=args.pad,
            pad_value=args.pad_value,
            sigma=args.sigma,
            n_levels=args.levels,
            grad_low=args.grad_low,
            grad_high=args.grad_high,
            grad_mean_min=args.grad_mean_min,
            min_length=args.min_length,
            min_area=args.min_area,
            closed_min_area=args.closed_min_area,
            border_margin=args.border_margin,
            inside_max=args.inside_max,
            inside_percentile=args.inside_percentile,
            contrast_min=args.contrast_min,
            dark_rel_min=args.dark_rel_min,
            dark_abs=args.dark_abs,
            min_dark_frac=args.min_dark_frac,
            offset=args.offset,
            bg_sigma=args.bg_sigma,
            save_svg=args.save_svg,
            dark_rel_each = args.dark_rel_each,
            min_dark_edge_cover = args.min_dark_edge_cover
        )
