#!/usr/bin/env python3
"""
Fully optimized contour detection.
Uses OpenCV for speed, along with multiprocessing and Numba.
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
import glob
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import traceback

import numba
import cv2

from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.measure import find_contours
from skimage.draw import polygon as draw_polygon
from scipy.ndimage import map_coordinates

BRIGHT_COLORS = [
    "#ff3b30"
]

# helpers

def load_grayscale(path: str) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3:
        arr = rgb2gray(arr)
    img = img_as_float(arr).astype(np.float32)
    if img.max() > 0:
        img = (img - img.min()) / max(1e-8, (img.max() - img.min()))
    return img

@numba.jit(nopython=True, cache=True)
def shoelace_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x = np.ascontiguousarray(poly[:, 1])
    y = np.ascontiguousarray(poly[:, 0])
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def sample_nearest(img: np.ndarray, coords: np.ndarray) -> np.ndarray:
    rr = np.clip(np.round(coords[:, 0]).astype(np.int32), 0, img.shape[0]-1)
    cc = np.clip(np.round(coords[:, 1]).astype(np.int32), 0, img.shape[1]-1)
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
        ksize = int(s * 4) // 2 * 2 + 1
        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=s)
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        g = np.hypot(gx, gy)
        resp = s * g
        acc = resp if acc is None else np.maximum(acc, resp)
    m, M = np.percentile(acc, 1), np.percentile(acc, 99)
    return np.clip((acc - m) / max(1e-8, (M - m)), 0, 1)

def image_gradients(img: np.ndarray, sigma_for_deriv: float):
    ksize = int(sigma_for_deriv * 4) // 2 * 2 + 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma_for_deriv)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy

@numba.jit(nopython=True, cache=True)
def is_near_closed(P: np.ndarray, eps=2.5) -> bool:
    dist_sq = (P[0,0] - P[-1,0])**2 + (P[0,1] - P[-1,1])**2
    return dist_sq <= eps**2

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
                keep[i] = False # remove the outer (enclosing) contour
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
    Q = P.copy()
    Q[:, 0] -= pad
    Q[:, 1] -= pad
    mask = (Q[:,0] >= 0) & (Q[:,0] < H) & (Q[:,1] >= 0) & (Q[:,1] < W)
    if mask.sum() < 3:
        return None
    return Q[mask]


def draw_contours(
    image_path: str,
    out_dir: str = "./contours",
    out_filename=None,
    pad: int = 1,
    pad_value: str = "auto",
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
    if out_filename is None:
        out_filename = os.path.basename(image_path)

    I0 = load_grayscale(image_path)
    H0, W0 = I0.shape

    if pad_value != "auto":
        try: pv = float(pad_value)
        except Exception: pv = "auto"
    else:
        pv = "auto"
    I, pad_amt, used_val = pad_image(I0, pad=pad, pad_value=pv)

    ksize_s = int(sigma * 4) // 2 * 2 + 1
    S = cv2.GaussianBlur(I, (ksize_s, ksize_s), sigmaX=sigma)
    ksize_b = int(bg_sigma * 4) // 2 * 2 + 1
    B = cv2.GaussianBlur(S, (ksize_b, ksize_b), sigmaX=bg_sigma)
    
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

    dpi = 100.0
    fig_w, fig_h = W0 / dpi, H0 / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(I0, cmap="gray")
    color_cycle = cycle(BRIGHT_COLORS)
    for P in kept:
        ax.plot(P[:, 1], P[:, 0], linewidth=0.9, color=next(color_cycle))
    ax.set_axis_off()
    
    png_path = os.path.join(out_dir, out_filename)

    pil_kwargs = {}
    if png_path.lower().endswith((".tif", ".tiff")):
        pil_kwargs["compression"] = "tiff_lzw"

    plt.tight_layout(pad=0)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0, pil_kwargs=pil_kwargs)
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

    total_raw = sum(len(find_contours(S, l)) for l in levels)
    print(f"PNG saved: {png_path} | Kept {len(kept)} of {len(candidates)} candidates ({total_raw} raw). Pad: {pad_amt}px @ {used_val:.3f}")
    return png_path


IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

def process_single_image_wrapper(image_path: str, all_kwargs: dict):
    kwargs_for_draw = all_kwargs.copy()
    out_dir = os.path.join(kwargs_for_draw['out_parent'], kwargs_for_draw['out_folder_name'])
    out_filename = os.path.basename(image_path)
    
    kwargs_for_draw.pop('out_dir', None)
    kwargs_for_draw.pop('out_parent', None)
    kwargs_for_draw.pop('out_folder_name', None)
    kwargs_for_draw.pop('folder', None)
    kwargs_for_draw.pop('image', None)
    
    try:
        draw_contours(
            image_path=image_path,
            out_dir=out_dir,
            out_filename=out_filename,
            **kwargs_for_draw,
        )
        return f"SUCCESS: {os.path.basename(image_path)}"
    except Exception as e:
        return f"FAILURE: {os.path.basename(image_path)} -> {e}\n{traceback.format_exc()}"

def parse_args():
    ap = argparse.ArgumentParser(description="Optimized contour detection with dark-object filters.")
    ap.add_argument("image", nargs="?", help="Path to TIFF/PNG/JPG. Omit if using --folder.")
    ap.add_argument("-o", "--out-dir", default="./contours", help="Output folder for single-image mode.")
    ap.add_argument("--folder", help="Input folder containing images to process.")
    ap.add_argument("--out-parent", help="Parent directory for the new output folder.")
    ap.add_argument("--out-folder-name", help="Name of the new output folder under --out-parent.")
    ap.add_argument("--pad", type=int, default=8, help="Pixels of constant padding.")
    ap.add_argument("--pad-value", default="auto", help='"auto" or numeric value in [0,1].')
    ap.add_argument("--sigma", type=float, default=1.4)
    ap.add_argument("--levels", dest="n_levels", type=int, default=18, help="Number of contour levels to check.")
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
    ap.add_argument("--dark-rel-each", type=float, default=0.020)
    ap.add_argument("--min-dark-edge-cover", type=float, default=0.60)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    kwargs = vars(args)

    if args.folder:
        if not args.out_parent or not args.out_folder_name:
            raise ValueError("Must provide --out-parent and --out-folder-name with --folder.")
        
        image_paths = []
        for ext in IMAGE_EXTS:
            image_paths.extend(glob.glob(os.path.join(args.folder, f"*{ext}")))
        
        if not image_paths:
            print(f"No images found in {args.folder}")
        else:
            print(f"Found {len(image_paths)} images to process in parallel...")
            out_dir = os.path.join(args.out_parent, args.out_folder_name)
            os.makedirs(out_dir, exist_ok=True)
            
            task_function = partial(process_single_image_wrapper, all_kwargs=kwargs)
            
            with ProcessPoolExecutor() as executor:
                results = executor.map(task_function, sorted(image_paths))
                for res in results:
                    print(res)
            print(f"Finished processing. Results are in: {out_dir}")

    else:
        if not args.image:
            raise ValueError("Provide an IMAGE path, or use --folder.")
        kwargs.pop('folder', None)
        kwargs.pop('out_parent', None)
        kwargs.pop('out_folder_name', None)
        draw_contours(**kwargs)