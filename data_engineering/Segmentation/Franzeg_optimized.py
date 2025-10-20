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
import io

import numba
import cv2

from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.measure import find_contours
from skimage.draw import polygon as draw_polygon
from scipy.ndimage import map_coordinates

BRIGHT_COLORS = ["#ff3b30"]

def hex_to_bgr(hex_color: str):
    """'#RRGGBB' -> (B, G, R) ints 0..255 for OpenCV."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

def load_grayscale(path: str) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3:
        arr = rgb2gray(arr)
    img = img_as_float(arr).astype(np.float32)
    if img.max() > 0:
        img = (img - img.min()) / max(1e-8, (img.max() - img.min()))
    return img

def to_cv_contour(P):
    # P is (N,2) in (row, col) = (y, x)
    xy = P[:, ::-1]  # -> (x, y)
    xy = np.round(xy).astype(np.int32)  # int32 for OpenCV
    xy = np.ascontiguousarray(xy.reshape(-1, 1, 2))  # (N,1,2), contiguous
    return xy

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
    Q = P.copy()
    Q[:, 0] -= pad
    Q[:, 1] -= pad
    mask = (Q[:,0] >= 0) & (Q[:,0] < H) & (Q[:,1] >= 0) & (Q[:,1] < W)
    if mask.sum() < 3:
        return None
    return Q[mask]

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
        y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        return float(x), float(y)
    if method == "edt_weighted":
        dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=3)
        w = np.power(dist, max(1.0, float(alpha)))
        w[m == 0] = 0.0
        ys, xs = np.nonzero(m)
        wv = w[ys, xs]
        if wv.sum() <= 0:
            return float(xs.mean()), float(ys.mean())
        cx = np.average(xs, weights=wv)
        cy = np.average(ys, weights=wv)
        return float(cx), float(cy)
    if method == "erode_centroid":
        if erode_r > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_r*2+1, erode_r*2+1))
            m = cv2.erode(m, k)
            if m.sum() == 0:
                m = (mask2d > 0).astype(np.uint8)
        ys, xs = np.nonzero(m)
        if xs.size == 0:
            return None
        return float(xs.mean()), float(ys.mean())
    # fallback
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())

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

def draw_contours(
    image_path: str,
    out_dir: str = "./contours",
    out_filename=None,
    mode: str = "segmentation",
    square_size: int = 7,
    approximation_level: float = 0.01,
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
            if ((P[:,0] < border_margin).any() or (P[:,0] > h-1-border_margin).any() or (P[:,1] < border_margin).any() or (P[:,1] > w-1-border_margin).any()):
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
            
    refined_contours = [to_cv_contour(P) for P in kept]

    base_u8 = (np.clip(I0, 0, 1) * 255).astype(np.uint8)            # HxW uint8
    segmented_image_cv = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2BGR)  # HxWx3
    contour_color = hex_to_bgr(BRIGHT_COLORS[0])

    if refined_contours:
        cv2.polylines(segmented_image_cv, refined_contours, isClosed=True,
                      color=contour_color, thickness=1, lineType=cv2.LINE_AA)

    # Label image one unique integer per object
    label_img = np.zeros((H0, W0), dtype=np.int32)
    for i, cnt in enumerate(refined_contours, start=1):
        pts = cnt.reshape(-1, 2)
        cv2.fillPoly(label_img, [pts.astype(np.int32)], int(i))

    centroid_method: str = "edt_weighted"
    centroid_alpha: float = 2.0
    centroid_erode_r: int = 5
    centroid_pad: int = 3
    
    # centroids directly from label image 
    centroids = robust_centroids_from_labels(
        label_img,
        method=centroid_method,
        alpha=centroid_alpha,
        erode_r=centroid_erode_r,
        pad=centroid_pad
    )

    # Debug overlay = contours + filled white squares at centroids
    overlay_image = segmented_image_cv.copy()
    half = int(square_size) // 2
    def _clip_int(v, lo, hi):
        return int(max(lo, min(hi, int(v))))
    for (cx, cy) in centroids:
        x = int(round(float(cx))); y = int(round(float(cy)))
        x1 = _clip_int(x - half, 0, W0 - 1); y1 = _clip_int(y - half, 0, H0 - 1)
        x2 = _clip_int(x + half, 0, W0 - 1); y2 = _clip_int(y + half, 0, H0 - 1)
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (255, 255, 255), -1)

    # features = white squares on black
    feature_image = make_feature_image(centroids, (H0, W0), square_size=square_size)

    # Save 
    if out_filename is None:
        out_filename = os.path.basename(image_path)
    os.makedirs(out_dir, exist_ok=True)

    tiff_compression = [cv2.IMWRITE_TIFF_COMPRESSION, 5]  # LZW

    if mode == 'all':
        path_segmented = os.path.join(out_dir, "segmented_contours")
        path_overlay = os.path.join(out_dir, "debug_overlay")
        path_features = os.path.join(out_dir, "trackpy_features")
        path_labels = os.path.join(out_dir, "labels")
        os.makedirs(path_segmented, exist_ok=True)
        os.makedirs(path_overlay, exist_ok=True)
        os.makedirs(path_features, exist_ok=True)
        os.makedirs(path_labels, exist_ok=True)
        

        # Save the segmented image
        cv2.imwrite(os.path.join(path_segmented, out_filename), segmented_image_cv, tiff_compression)
    
        # Save label image (int32 single-channel TIFF)
        cv2.imwrite(os.path.join(path_labels, out_filename), label_img, tiff_compression)

        # Save the feature image (squares on black background)
        feature_image = make_feature_image(centroids, (H0, W0), square_size=square_size)
        cv2.imwrite(os.path.join(path_features, out_filename), feature_image, tiff_compression)

        def _clip_int(v, lo, hi):
            return int(max(lo, min(hi, int(v))))

        overlay_image = segmented_image_cv.copy()
        half = square_size // 2

        for (cx, cy) in centroids:
            # force Python ints, clip to image, and ensure (x1<=x2, y1<=y2)
            x = int(round(float(cx))); y = int(round(float(cy)))
            x1 = _clip_int(x - half, 0, W0 - 1)
            y1 = _clip_int(y - half, 0, H0 - 1)
            x2 = _clip_int(x + half, 0, W0 - 1)
            y2 = _clip_int(y + half, 0, H0 - 1)
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            # Use positional -1 for filled thickness; color tuple for BGR image
            cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.imwrite(os.path.join(path_overlay, out_filename), overlay_image, tiff_compression)

    else: # mode == 'segmentation'
        cv2.imwrite(os.path.join(out_dir, out_filename), segmented_image_cv, tiff_compression)

    print(f"[{mode.upper()} MODE] Processed: {out_filename} | Found {len(kept)} objects.")
    return out_filename

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
    ap = argparse.ArgumentParser(description="Optimized contour detection and feature generation.")
    ap.add_argument("--mode", choices=['segmentation', 'all'], default='segmentation', 
                    help="'segmentation': saves only contoured images. 'all': saves segmented, overlay, and feature images in subfolders.")
    ap.add_argument("--square-size", type=int, default=7, help="Size of the white squares for feature generation in 'all' mode.")
    ap.add_argument("--approximation-level", type=float, default=0.01, 
                    help="Controls contour smoothing (0.005=less, 0.02=more). Default is 0.01 (1%%).")
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