from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import deque
import heapq
from typing import List, Tuple, Optional, Dict  # might not need dicts import for now 
import os
from pathlib import Path


def _to_gray_float32(I: np.ndarray) -> np.ndarray:
    if I.ndim == 3:
        if I.shape[2] == 3:
            I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        else:
            I = I[..., 0]
    return I.astype(np.float32, copy=False)

def _sobel_gradient_magnitude_cv(I: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    mag = cv2.magnitude(gx, gy)
    return mag

def _disk(radius: int = 1) -> np.ndarray:
    r = int(max(1, radius))
    k = 2 * r + 1
    se = np.zeros((k, k), dtype=np.uint8)
    cy, cx = r, r
    for y in range(k):
        for x in range(k):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                se[y, x] = 1
    return se

def _histogram_threshold_from_gradient(S: np.ndarray, manual_finetune: float = 0.0) -> float:
    S1 = S[S > 0]
    if S1.size == 0:
        return 0.0
    vmin = float(S1.min())
    vmax = float(S1.max())
    ratio = (vmax - vmin) / 1000.0 if vmax > vmin else 1.0
    if ratio <= 0 or not np.isfinite(ratio):
        ratio = 1.0
    bins = np.arange(vmin, vmax + ratio, ratio, dtype=np.float64)
    if bins.size < 2:
        q = np.clip(50.0 - manual_finetune, 1.0, 99.0) / 100.0
        return float(np.quantile(S1, q))

    hist_data, _ = np.histogram(S1, bins=bins)
    if hist_data.sum() == 0:
        q = np.clip(50.0 - manual_finetune, 1.0, 99.0) / 100.0
        return float(np.quantile(S1, q))

    top3 = np.argsort(hist_data)[-3:]
    hist_mode_loc = int(np.round(np.mean(top3)))
    temp_hist = hist_data.astype(np.float64)
    temp_hist = temp_hist / temp_hist.sum() * 100.0

    lower_bound = 3 * hist_mode_loc
    lower_bound = min(max(lower_bound, 0), temp_hist.size - 1)

    norm_hist = temp_hist / (temp_hist.max() if temp_hist.max() > 0 else 1.0)
    tail = norm_hist[hist_mode_loc:]
    idx_rel = np.argmax(tail < 0.05) if np.any(tail < 0.05) else (tail.size - 1)
    idx = hist_mode_loc + idx_rel
    upper_bound = max(idx, 18 * hist_mode_loc)
    upper_bound = min(max(upper_bound, 0), temp_hist.size - 1)

    if upper_bound < lower_bound:
        lower_bound, upper_bound = upper_bound, lower_bound
    density_metric = float(temp_hist[lower_bound : upper_bound + 1].sum())

    saturation1, saturation2 = 3.0, 42.0
    a = (95.0 - 40.0) / (saturation1 - saturation2)
    b = 95.0 - a * saturation1

    prct_value = round(a * density_metric + b)
    prct_value = int(np.clip(prct_value, 25, 98))
    prct_value = prct_value - float(manual_finetune)
    prct_value = float(np.clip(prct_value, 1.0, 100.0))

    q = prct_value / 100.0
    return float(np.quantile(S1, q))

def _remove_small_objects(mask: np.ndarray, min_size: int, connectivity: int = 8) -> np.ndarray:
    if min_size <= 1:
        return mask.astype(bool, copy=False)
    mask_u = (mask.astype(np.uint8) > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u, connectivity=connectivity)
    if num <= 1:
        return mask_u.astype(bool)
    keep = np.ones(num, dtype=bool)
    keep[0] = False
    areas = stats[:, cv2.CC_STAT_AREA]
    keep[areas < int(min_size)] = False
    out = np.zeros_like(mask_u, dtype=np.uint8)
    for k in range(1, num):
        if keep[k]:
            out[labels == k] = 1
    return out.astype(bool)

def _fill_holes_binary(S: np.ndarray) -> np.ndarray:
    S_u = (S.astype(np.uint8) > 0).astype(np.uint8)
    inv = 1 - S_u
    num, labels = cv2.connectedComponents(inv, connectivity=8)
    h, w = inv.shape
    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    holes = np.ones_like(inv, dtype=bool)
    for lbl in border_labels:
        holes &= (labels != lbl)
    holes &= (inv > 0)
    filled = S_u.copy().astype(bool)
    filled[holes] = True
    return filled

def fill_holes(
    S: np.ndarray,
    I: np.ndarray,
    min_hole_size: float = np.inf,
    max_hole_size: float = np.inf,
    hole_min_perct_intensity: float = 0.0,
    hole_max_perct_intensity: float = 100.0,
    fill_holes_bool_oper: str = "AND",
) -> np.ndarray:
    S_bool = S.astype(bool, copy=False)
    I_f = _to_gray_float32(I)

    filled = _fill_holes_binary(S_bool)
    holes = filled & ~S_bool
    if not holes.any():
        return S_bool

    holes_u = holes.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(holes_u, connectivity=8)
    if num <= 1:
        return S_bool

    flat = I_f[np.isfinite(I_f)]
    if flat.size == 0:
        flat = I_f.ravel()
    tmin = np.quantile(flat, np.clip(hole_min_perct_intensity, 0, 100) / 100.0)
    tmax = np.quantile(flat, np.clip(hole_max_perct_intensity, 0, 100) / 100.0)

    to_fill = np.zeros(num, dtype=bool)
    to_fill[0] = False
    for k in range(1, num):
        area = stats[k, cv2.CC_STAT_AREA]
        size_ok = True
        if np.isfinite(min_hole_size):
            size_ok &= (area >= min_hole_size)
        if np.isfinite(max_hole_size):
            size_ok &= (area <= max_hole_size)
        mask_k = (labels == k)
        mean_int = float(np.mean(I_f[mask_k])) if mask_k.any() else np.nan
        intensity_ok = (mean_int >= tmin) and (mean_int <= tmax)
        if fill_holes_bool_oper.upper() == "OR":
            to_fill[k] = bool(size_ok or intensity_ok)
        else:
            to_fill[k] = bool(size_ok and intensity_ok)

    out = S_bool.copy()
    for k in range(1, num):
        if to_fill[k]:
            out[labels == k] = True
    return out

def break_holes(I1: np.ndarray, S: np.ndarray, BW: np.ndarray) -> np.ndarray:
    """

    Parameters
    I1 : ndarray
        Grayscale weight image (will be squared internally). Can be any numeric dtype.
    S : ndarray of bool
        Foreground/object support mask. Pixels outside S are treated as background.
    BW : ndarray of bool
        Binary object mask whose interior holes you want to 'break' (carve a minimal-cost cut).

    Returns
    BW_out : ndarray of bool
        Modified BW where narrow 'break' lines are carved through selected holes.
    """

    def _to_float32(img):
        return img.astype(np.float32, copy=False)

    def _fill_holes_bin(mask: np.ndarray) -> np.ndarray:
        """imfill(mask, 'holes')"""
        mask_u = (mask.astype(np.uint8) > 0).astype(np.uint8)
        inv = (mask_u == 0).astype(np.uint8)
        h, w = inv.shape
        # flood fill from border on the inverted mask
        ff = inv.copy()
        cv2.floodFill(ff, None, (0, 0), 2)  # fill background label starting from corner
        # after flood, pixels == 1 are holes (not reached from border); pixels==2 are border-connected background
        holes = (ff == 1)
        filled = mask_u.copy()
        filled[holes] = 1
        return filled.astype(bool)

    def _bwareaopen(mask: np.ndarray, min_size: int, connectivity: int = 8) -> np.ndarray:
        if min_size <= 1:
            return mask.astype(bool)
        mask_u = (mask.astype(np.uint8) > 0).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u, connectivity=connectivity)
        if num <= 1:
            return mask_u.astype(bool)
        keep = np.zeros_like(mask_u, dtype=bool)
        for k in range(1, num):
            if stats[k, cv2.CC_STAT_AREA] >= int(min_size):
                keep[labels == k] = True
        return keep

    def _graydist(Iw: np.ndarray, seeds_mask: np.ndarray = None, seed_points=None, connectivity: int = 8) -> np.ndarray:
        """
        Weighted geodesic distance (Dijkstra) using per-pixel weights in Iw.
        Cost of a step is average of weights * step length (1 or sqrt(2)).
        Either seeds_mask (bool) or seed_points (list of (y,x)) must be provided.
        """
        h, w = Iw.shape
        dist = np.full((h, w), np.inf, dtype=np.float32)
        heap = []

        if seeds_mask is not None:
            seeds = np.argwhere(seeds_mask)
            for y, x in seeds:
                dist[y, x] = 0.0
                heap.append((0.0, int(y), int(x)))
        elif seed_points is not None and len(seed_points) > 0:
            for (y, x) in seed_points:
                if 0 <= y < h and 0 <= x < w:
                    dist[y, x] = 0.0
                    heap.append((0.0, int(y), int(x)))
        else:
            return dist  # all inf

        heapq.heapify(heap)
        if connectivity == 8:
            nbrs = [(-1, -1, 1.41421356), (-1, 0, 1.0), (-1, 1, 1.41421356),
                    (0, -1, 1.0),               (0, 1, 1.0),
                    (1, -1, 1.41421356),  (1, 0, 1.0),  (1, 1, 1.41421356)]
        else:
            nbrs = [(-1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (1, 0, 1.0)]

        while heap:
            d, y, x = heapq.heappop(heap)
            if d > dist[y, x]:
                continue
            w0 = Iw[y, x]
            for dy, dx, sl in nbrs:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                # average weight across the step
                step_cost = ((w0 + Iw[ny, nx]) * 0.5) * sl
                nd = d + step_cost
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(heap, (nd, ny, nx))
        return dist

    def _regional_minima(F: np.ndarray) -> np.ndarray:
        """
        Approximate imregionalmin for 8-neighborhood:
        mark pixels that are <= all neighbors (plateau minima included).
        """
        if F.dtype != np.float32 and F.dtype != np.float64:
            F = F.astype(np.float32, copy=False)
        k = np.ones((3, 3), np.uint8)
        F_min = cv2.erode(F, k, borderType=cv2.BORDER_REPLICATE)
        # pixels equal to local 3x3 minimum are candidate minima
        cand = (F == F_min)
        # remove NaN/Inf
        cand &= np.isfinite(F)
        return cand

    def _bwmorph_diag(mask: np.ndarray) -> np.ndarray:
        """
        add pixels to connect diagonal neighbors.
        """
        m = mask.astype(bool, copy=True)
        h, w = m.shape

        # UL-DR diagonal pairs --> set right and down as bridges
        pair1 = (m[:-1, :-1] & m[1:, 1:])
        m[:-1, 1:] |= pair1
        m[1:, :-1] |= pair1

        # UR-DL diagonal pairs --> set left-up and right-down as bridges
        pair2 = (m[:-1, 1:] & m[1:, :-1])
        m[:-1, :-1] |= pair2
        m[1:, 1:] |= pair2

        return m

    Iw = _to_float32(I1)
    BW = BW.astype(bool, copy=True)
    S = S.astype(bool, copy=False)

    # Fill holes in BW and restrict to S
    filled_BW = _fill_holes_bin(BW)
    filled_BW[~S] = False

    xor_img = np.logical_xor(BW, filled_BW)       # hole pixels
    xor_img = _bwareaopen(xor_img, 3, connectivity=8)

    # If no holes to break --> return early
    if not np.any(xor_img):
        return BW

    # weight the grayscale pixels
    Iw = Iw * Iw
    Iw[xor_img] = 0.0  # encourage cuts through candidate-hole regions

    # loop handles nested holes
    for _iter in range(1, 101):  # safety cap at 100
        # centroids of current holes (xor_img)
        num, labels, stats, cents = cv2.connectedComponentsWithStats(xor_img.astype(np.uint8), connectivity=8)
        if num <= 1:
            break

        centroids = np.rint(cents[1:, :]).astype(int)  # (x, y)
        if centroids.size == 0:
            break
        centroids = centroids[:, ::-1]  # -> (y, x)

        # grayscale distance from background (outside filled_BW)
        bg_seeds = (~filled_BW).astype(bool)
        gd2 = _graydist(Iw, seeds_mask=bg_seeds, seed_points=None, connectivity=8)

        # grayscale distance from hole centroids
        seeds_pts = [(int(cy), int(cx)) for cy, cx in centroids.tolist()]
        gd1 = _graydist(Iw, seeds_mask=None, seed_points=seeds_pts, connectivity=8)

        # minimal basin connecting hole(s) to background
        sumd = gd1 + gd2
        break_pixels = _regional_minima(sumd)
        break_pixels = _bwmorph_diag(break_pixels)

        # carve the break pixels out of BW
        BW[break_pixels] = False

        # recompute filled holes and xor for next iteration
        filled_BW = _fill_holes_bin(BW)
        filled_BW[~S] = False
        xor_img = np.logical_xor(BW, filled_BW)
        xor_img = _bwareaopen(xor_img, 3, connectivity=8)

        if not np.any(xor_img):
            break

    return BW


def assign_nearest_connected_label(marker_matrix: np.ndarray, mask_matrix: np.ndarray, connectivity: int = 4) -> np.ndarray:
    labels = np.array(marker_matrix, dtype=np.int32, copy=True)
    mask = mask_matrix.astype(bool, copy=False)
    h, w = labels.shape

    if connectivity == 8:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    q = deque()
    seeds = np.flatnonzero((labels > 0) & mask)
    for idx in seeds:
        y, x = divmod(int(idx), w)
        q.append((y, x))

    while q:
        y, x = q.popleft()
        lab = int(labels[y, x])
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if not mask[ny, nx]:
                continue
            if labels[ny, nx] > 0:
                continue
            labels[ny, nx] = lab
            q.append((ny, nx))
    return labels

def check_body_connectivity(image: np.ndarray) -> np.ndarray:
    img_out = image.copy()
    if not np.issubdtype(img_out.dtype, np.integer):
        img_out = img_out.astype(np.int32, copy=False)
    labels = np.unique(img_out)
    labels = labels[labels > 0]
    h, w = img_out.shape
    for obj in labels:
        mask = (img_out == obj).astype(np.uint8)
        num, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        if num <= 2:
            continue
        areas = stats[:, cv2.CC_STAT_AREA]
        areas[0] = 0
        winner = int(np.argmax(areas))
        for cid in range(1, num):
            if cid == winner:
                continue
            ys, xs = np.where(comp == cid)
            neighbor_counts = {}
            for y, x in zip(ys, xs):
                if x-1 >= 0:
                    lab = int(img_out[y, x-1])
                    if lab > 0 and lab != obj:
                        neighbor_counts[lab] = neighbor_counts.get(lab, 0) + 1
                if y-1 >= 0:
                    lab = int(img_out[y-1, x])
                    if lab > 0 and lab != obj:
                        neighbor_counts[lab] = neighbor_counts.get(lab, 0) + 1
                if x+1 < w:
                    lab = int(img_out[y, x+1])
                    if lab > 0 and lab != obj:
                        neighbor_counts[lab] = neighbor_counts.get(lab, 0) + 1
                if y+1 < h:
                    lab = int(img_out[y+1, x])
                    if lab > 0 and lab != obj:
                        neighbor_counts[lab] = neighbor_counts.get(lab, 0) + 1
            if neighbor_counts:
                max_ct = max(neighbor_counts.values())
                candidates = [lab for lab, ct in neighbor_counts.items() if ct == max_ct]
                target = int(min(candidates))
            else:
                target = 0
            img_out[comp == cid] = target
    return img_out

def egt_segmentation(
    I: np.ndarray,
    min_cell_size: int = 1,
    min_hole_size: float = np.inf,
    max_hole_size: float = np.inf,
    hole_min_perct_intensity: float = 0.0,
    hole_max_perct_intensity: float = 100.0,
    fill_holes_bool_oper: str = "AND",
    manual_finetune: float = 0.0,
) -> np.ndarray:
    I_f = _to_gray_float32(I)
    Sg = _sobel_gradient_magnitude_cv(I_f)
    threshold = _histogram_threshold_from_gradient(Sg, manual_finetune=manual_finetune)
    S = (Sg > threshold).astype(np.uint8)
    S_bool = fill_holes(
        S > 0,
        I_f,
        min_hole_size=min_hole_size,
        max_hole_size=max_hole_size,
        hole_min_perct_intensity=hole_min_perct_intensity,
        hole_max_perct_intensity=hole_max_perct_intensity,
        fill_holes_bool_oper=fill_holes_bool_oper,
    )
    se = _disk(1)
    S_er = cv2.erode(S_bool.astype(np.uint8), se, iterations=1).astype(bool)
    S_clean = _remove_small_objects(S_er, min_size=int(max(1, min_cell_size)), connectivity=8)
    return S_clean.astype(bool)


def _geodesic_distance(mask: np.ndarray, seed_coords: List[Tuple[int, int]], connectivity: int = 8) -> np.ndarray:
    """
    Compute geodesic distance (quasi-euclidean) inside 'mask' from multiple seeds.
    mask: bool array where True pixels are traversable.
    seed_coords: list of (y, x) integer coordinates.
    Returns float32 distances with np.inf for unreachable pixels.
    """
    h, w = mask.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    if len(seed_coords) == 0:
        return dist
    # Use Dijkstra with 8-neighbor costs (1 or sqrt(2))
    heap: List[Tuple[float, int, int]] = []
    for (y, x) in seed_coords:
        if 0 <= y < h and 0 <= x < w and mask[y, x]:
            dist[y, x] = 0.0
            heap.append((0.0, y, x))
    heapq.heapify(heap)
    if connectivity == 8:
        nbrs = [(-1, -1, np.sqrt(2)), (-1, 0, 1.0), (-1, 1, np.sqrt(2)),
                (0, -1, 1.0), (0, 1, 1.0),
                (1, -1, np.sqrt(2)), (1, 0, 1.0), (1, 1, np.sqrt(2))]
    else:
        nbrs = [(-1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (1, 0, 1.0)]
    while heap:
        d, y, x = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        for dy, dx, wcost in nbrs:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if not mask[ny, nx]:
                continue
            nd = d + wcost
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                heapq.heappush(heap, (nd, ny, nx))
    return dist

def cluster_objects(BW: np.ndarray, cluster_distance: float, valid_traversal_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Label objects in BW (8-connected).
    Compute centroids.
    Compute geodesic distance map inside valid_traversal_mask from all centroids.
    Threshold at cluster_distance/2, label connected regions -> clusters.
    Assign each object to the cluster id at its centroid.
    Return a relabeled image where each object's pixels are set to its cluster label.
    """
    BW_u = (BW.astype(np.uint8) > 0).astype(np.uint8)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(BW_u, connectivity=8)
    if num <= 1:
        return np.zeros_like(BW_u, dtype=np.int32)

    # Round centroids to nearest integer pixel
    centroids = np.rint(cents[1:, :]).astype(int)  # skip label 0
    # OpenCV returns (x, y) order; convert to (y, x)
    centroids[:, :] = centroids[:, ::-1]

    if valid_traversal_mask is None:
        valid_traversal_mask = np.ones_like(BW_u, dtype=bool)
    else:
        valid_traversal_mask = valid_traversal_mask.astype(bool, copy=False)

    # Compute geodesic distance inside mask from all centroids
    seeds = [(int(cy), int(cx)) for cy, cx in centroids.tolist()]
    D = _geodesic_distance(valid_traversal_mask, seeds, connectivity=8)

    # Threshold and label clusters
    thresh = float(cluster_distance) / 2.0
    in_radius = (D <= thresh) & valid_traversal_mask
    ncl, clabels = cv2.connectedComponents(in_radius.astype(np.uint8), connectivity=8)

    # Map each object to a cluster id via the centroid pixel
    cluster_labels = np.zeros(num, dtype=np.int32)  # include background entry
    for i in range(1, num):
        cy, cx = centroids[i-1]
        if not (0 <= cy < clabels.shape[0] and 0 <= cx < clabels.shape[1]):
            cid = 0
        else:
            cid = int(clabels[cy, cx])
        cluster_labels[i] = cid

    # replace each object id with its cluster id
    out = cluster_labels[labels]
    return out.astype(np.int32)

def fill_holes_between(S: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    """
    Fill holes whose area is in (lower_bound, upper_bound).
    If upper_bound is inf --> use (max background component size - 1)
    """
    S = S.astype(bool, copy=False)
    inv = (~S).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    if num <= 1:
        return S

    # Compute default upper bound if inf: max component size - 1
    if np.isinf(upper_bound):
        areas = stats[1:, cv2.CC_STAT_AREA]
        upper_bound = float(areas.max() - 1) if areas.size > 0 else 0.0

    out = S.copy()
    for k in range(1, num):
        area = float(stats[k, cv2.CC_STAT_AREA])
        if (area > float(lower_bound)) and (area < float(upper_bound)):
            out[labels == k] = True
    return out


def filter_by_circularity(BW: np.ndarray, circularity_threshold: float) -> np.ndarray:
    """
    Remove objects with circularity < threshold.
    Circularity = 4*pi*Area / Perimeter^2 (using external contour perimeter).
    """
    BW_u = (BW.astype(np.uint8) > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(BW_u, connectivity=8)
    if num <= 1:
        return BW_u.astype(bool)

    out = BW_u.copy().astype(bool)
    for k in range(1, num):
        mask_k = (labels == k).astype(np.uint8)
        contours, _ = cv2.findContours(mask_k, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            out[labels == k] = False
            continue
        # Area from stats
        # perimeter from contours = sum of external perimeters
        area = float(stats[k, cv2.CC_STAT_AREA])
        perim = sum(cv2.arcLength(cnt, True) for cnt in contours)
        if perim <= 0:
            out[labels == k] = False
            continue
        circ = (4.0 * np.pi * area) / (perim * perim)
        if circ < circularity_threshold:
            out[labels == k] = False
    return out.astype(bool)

def _edge_mask_from_labels(segmented_image: np.ndarray) -> np.ndarray:
    """
    Compute an edge mask where any 8-neighbor has a different label or pixel is on border.
    """
    L = segmented_image
    h, w = L.shape
    edge = np.zeros((h, w), dtype=bool)
    # Compare with shifted neighbors; pad with zeros
    pads = [
        ((0,1),(0,0)),  # up neighbor
        ((1,0),(0,0)),  # down
        ((0,0),(0,1)),  # left
        ((0,0),(1,0)),  # right
        ((0,1),(0,1)),  # up-left
        ((0,1),(1,0)),  # up-right
        ((1,0),(0,1)),  # down-left
        ((1,0),(1,0)),  # down-right
    ]
    for (pt, pl) in pads:
        shifted = np.pad(L, (pt, pl), mode='constant', constant_values=0)
        shifted = shifted[pt[0]:pt[0]+h, pl[0]:pl[0]+w]
        edge |= (L != shifted)
    edge &= (L > 0)
    return edge

def find_edges(segmented_image: np.ndarray):
    """
      edge_image: labels only on edge pixels (uint16)
      text_location: first encountered edge pixel (x,y) per label index (1..max)
      perimeter: count of edge pixels per label (length = max label)
    """
    L = segmented_image.astype(np.int32, copy=False)
    h, w = L.shape
    highest = int(L.max()) if L.size else 0
    edge_mask = _edge_mask_from_labels(L)
    edge_image = np.zeros_like(L, dtype=np.uint16)
    edge_image[edge_mask] = L[edge_mask].astype(np.uint16)

    perimeter = np.zeros(max(highest, 0), dtype=np.int64)
    text_location = np.zeros((max(highest, 0), 2), dtype=np.int64)  # (x, y)

    first_seen = np.ones(max(highest, 0), dtype=bool)

    ys, xs = np.nonzero(edge_mask)
    for y, x in zip(ys, xs):
        lab = int(L[y, x])
        if lab <= 0 or lab > highest:
            continue
        perimeter[lab - 1] += 1
        if first_seen[lab - 1]:
            text_location[lab - 1] = [x, y]
            first_seen[lab - 1] = False

    return edge_image, text_location, perimeter

def find_edges_labeled(segmented_image: np.ndarray, nb_cells: int):
    """
    Same as find_edges but text_location has shape (nb_cells, 2).
    """
    L = segmented_image.astype(np.int32, copy=False)
    h, w = L.shape
    highest = int(L.max()) if L.size else 0
    edge_mask = _edge_mask_from_labels(L)
    edge_image = np.zeros_like(L, dtype=np.float64)  # MATLAB returned double here
    edge_image[edge_mask] = L[edge_mask].astype(np.float64)

    perimeter = np.zeros(max(highest, 0), dtype=np.int64)
    text_location = np.zeros((int(nb_cells), 2), dtype=np.int64)

    filled = np.zeros(int(nb_cells), dtype=bool)
    ys, xs = np.nonzero(edge_mask)
    for y, x in zip(ys, xs):
        lab = int(L[y, x])
        if lab <= 0 or lab > highest:
            continue
        perimeter[lab - 1] += 1
        if lab - 1 < nb_cells and not filled[lab - 1]:
            text_location[lab - 1] = [x, y]
            filled[lab - 1] = True

    return edge_image, text_location, perimeter


def percentile(A, p):
    """
    percentile p in [0,1] (scalar or array). Ignores NaNs and uses
    index = round(p*len(B)+1) clamped to [1, len(B)] after sorting.
    """
    A = np.asarray(A)
    p = np.asarray(p, dtype=float)
    assert np.all((p >= 0) & (p <= 1)), "Percentiles must be between zero and one inclusive."
    B = A[np.isfinite(A)]
    if B.size == 0:
        return np.full(p.shape, np.nan)
    B = np.sort(B.ravel())
    idx = np.rint(p * len(B) + 1).astype(int)
    idx[idx < 1] = 1
    idx[idx > len(B)] = len(B)
    out = B[idx - 1]
    return out.reshape(p.shape)


def get_min_required_datatype(maxVal: int):
    """
    Returns a numpy dtype matching MATLAB get_min_required_datatype.
    """
    if maxVal <= np.iinfo(np.uint8).max:
        return np.uint8
    elif maxVal <= np.iinfo(np.uint16).max:
        return np.uint16
    elif maxVal <= np.iinfo(np.uint32).max:
        return np.uint32
    else:
        return np.float64


def check_cell_size(img: np.ndarray, Highest_cell_number: int, cell_size: np.ndarray, cell_size_threshold: int):
    """
    Renumber consecutively (1..highest) only cells with size > threshold.
    Unkept pixels are reassigned to nearest connected body.
    """
    renumber_cells = np.zeros(Highest_cell_number + 1, dtype=int)
    highest_cell_number = 0
    for i in range(1, Highest_cell_number + 1):
        if cell_size[i - 1] > cell_size_threshold:
            highest_cell_number += 1
            renumber_cells[i] = highest_cell_number
    BW = img > 0
    img = renumber_cells[img]
    img = assign_nearest_connected_label(img, BW.astype(bool))
    return img, highest_cell_number


def check_cell_size_renumber(img: np.ndarray, Highest_cell_number: int, cell_size: np.ndarray, cell_size_threshold: int):
    """
    Alternate version where kept labels are renumbered densely (1..highest) in order encountered.
    """
    renumber_cells = np.zeros(Highest_cell_number + 1, dtype=int)
    highest_cell_number = 0
    for i in range(1, Highest_cell_number + 1):
        if cell_size[i - 1] > cell_size_threshold:
            highest_cell_number += 1
            renumber_cells[i] = highest_cell_number
    BW = img > 0
    img = renumber_cells[img]
    img = assign_nearest_connected_label(img, BW.astype(bool))
    return img, highest_cell_number


def fog_bank_perctile_geodist(
    grayscale_image: np.ndarray,
    foreground_mask: np.ndarray,
    mask_matrix: np.ndarray,
    min_peak_size: int,
    min_object_size: int,
    fogbank_direction: int = 1,
    perc_binning: float = 5,
):
    # Input checks
    I = _to_gray_float32(grayscale_image)
    nb_rows, nb_cols = I.shape
    assert foreground_mask.dtype == bool and foreground_mask.shape == (nb_rows, nb_cols), "Invalid <foreground_mask>"
    assert mask_matrix.dtype == bool and mask_matrix.shape == (nb_rows, nb_cols), "Invalid <mask_matrix>"
    assert min_peak_size > 0 and min_object_size > 0, "Invalid sizes"
    assert (np.isnan(perc_binning)) or (0 <= perc_binning < 100), "Invalid <percentile_binning>"

    # Mask out background with NaN
    I = I.copy()
    I[~foreground_mask] = np.nan

    # Build percentile ladder
    if np.isnan(perc_binning) or perc_binning == 0:
        Y = np.unique(I[np.isfinite(I)]).astype(float)
    else:
        P_vec = np.arange(0, 100 + perc_binning, perc_binning, dtype=float) / 100.0
        Y = percentile(I, P_vec)
    Y = np.unique(np.concatenate(([0.0], Y[np.isfinite(Y)])))
    if fogbank_direction:
        Y = np.sort(Y)
    else:
        Y = np.sort(Y)[::-1]

    # Find first fog level that reveals >= min_peak_size components
    fog_level = 1
    nb_objects = 0
    CC_labels = None
    while nb_objects == 0 and fog_level <= Y.size:
        if fogbank_direction:
            img_b = (I <= Y[fog_level - 1]) & mask_matrix
        else:
            img_b = (I >= Y[fog_level - 1]) & mask_matrix
        num, labels, stats, _ = cv2.connectedComponentsWithStats(img_b.astype(np.uint8), connectivity=8)
        cnt = 0
        for k in range(1, num):
            if stats[k, cv2.CC_STAT_AREA] >= min_peak_size:
                cnt += 1
        nb_objects = cnt
        CC_labels = (labels, stats)
        fog_level += 1

    # Initialize seed image
    dtype = get_min_required_datatype(nb_objects)
    seed_image = np.zeros((nb_rows, nb_cols), dtype=dtype)
    nb_peaks = 0
    if CC_labels is not None:
        labels, stats = CC_labels
        for k in range(1, int(stats.shape[0])):
            if stats[k, cv2.CC_STAT_AREA] >= min_peak_size:
                nb_peaks += 1
                seed_image[labels == k] = nb_peaks

    # Drop the fog
    for n in range(fog_level - 1, Y.size):
        if fogbank_direction:
            image_b = (I <= Y[n]) & mask_matrix
        else:
            image_b = (I >= Y[n]) & mask_matrix

        # Assign labels to eligible pixels
        seed_image = assign_nearest_connected_label(seed_image, image_b)

        # Remove already-labeled pixels and find new peaks
        rem = image_b.copy()
        rem[seed_image > 0] = False
        num, labels, stats, _ = cv2.connectedComponentsWithStats(rem.astype(np.uint8), connectivity=8)

        # Type expansion if needed
        new_max = nb_peaks
        for k in range(1, num):
            if stats[k, cv2.CC_STAT_AREA] >= min_peak_size:
                new_max += 1
        new_dtype = get_min_required_datatype(new_max)
        if seed_image.dtype != new_dtype:
            seed_image = seed_image.astype(new_dtype, copy=False)

        for k in range(1, num):
            if stats[k, cv2.CC_STAT_AREA] >= min_peak_size:
                nb_peaks += 1
                seed_image[labels == k] = nb_peaks

    # Assign any remaining foreground pixels
    seed_image = assign_nearest_connected_label(seed_image, foreground_mask)

    # Size filtering
    if nb_peaks > 0:
        counts = np.bincount(seed_image.ravel(), minlength=nb_peaks + 1)
        sizes = counts[1:]  # ignore background
    else:
        sizes = np.array([], dtype=int)
    seed_image, nb_peaks = check_cell_size(seed_image, nb_peaks, sizes, min_object_size)

    # Connectivity enforcement
    seed_image = check_body_connectivity(seed_image)
    return seed_image, nb_peaks


def fog_bank_perctile_geodist_seed(
    grayscale_image: np.ndarray,
    foreground_mask: np.ndarray,
    mask_matrix: np.ndarray,
    seed_image: np.ndarray,
    min_object_size: int,
    fogbank_direction: int = 1,
    perc_binning: float = 5,
):
    I = _to_gray_float32(grayscale_image)
    nb_rows, nb_cols = I.shape
    assert foreground_mask.dtype == bool and foreground_mask.shape == (nb_rows, nb_cols), "Invalid <foreground_mask>"
    assert mask_matrix.dtype == bool and mask_matrix.shape == (nb_rows, nb_cols), "Invalid <mask_matrix>"
    assert seed_image.shape == (nb_rows, nb_cols), "Invalid <seed_image> wrong size"
    assert min_object_size > 0, "Invalid <min_object_size>"
    assert (np.isnan(perc_binning)) or (0 <= perc_binning < 100), "Invalid <percentile_binning>"

    I = I.copy()
    I[~foreground_mask] = np.nan

    if np.isnan(perc_binning) or perc_binning == 0:
        Y = np.unique(I[np.isfinite(I)]).astype(float)
    else:
        P_vec = np.arange(0, 100 + perc_binning, perc_binning, dtype=float) / 100.0
        Y = percentile(I, P_vec)

    if fogbank_direction:
        Y = np.sort(Y[np.isfinite(Y)])
        min_val = np.nanmin(I[seed_image > 0]) - 1.0
        I[seed_image > 0] = min_val
        Y = np.concatenate(([min_val], Y))
    else:
        Y = np.sort(Y[np.isfinite(Y)])[::-1]
        max_val = np.nanmax(I[seed_image > 0]) + 1.0
        I[seed_image > 0] = max_val
        Y = np.concatenate(([max_val], Y))

    seeds = seed_image.astype(np.int32, copy=True)

    for n in range(Y.size):
        if fogbank_direction:
            image_b = (I <= Y[n]) & mask_matrix
        else:
            image_b = (I >= Y[n]) & mask_matrix
        seeds = assign_nearest_connected_label(seeds, image_b)

    seeds = assign_nearest_connected_label(seeds, foreground_mask)
    nb_peaks = int(seeds.max())

    if nb_peaks > 0:
        counts = np.bincount(seeds.ravel(), minlength=nb_peaks + 1)
        sizes = counts[1:]
    else:
        sizes = np.array([], dtype=int)

    seeds, nb_peaks = check_cell_size_renumber(seeds, nb_peaks, sizes, min_object_size)
    seeds = check_body_connectivity(seeds)
    return seeds, nb_peaks


def generate_image_to_threshold(I: np.ndarray, method: str, fwr: int = 2):
    method = "".join(ch for ch in (method or "").lower() if ch.isalnum())
    if "gradient" in method:
        invalid = (I == 0)
        I32 = _to_gray_float32(I)
        grad = _sobel_gradient_magnitude_cv(I32)
        grad[invalid] = 0
        return grad
    if "entropy" in method:
        if fwr is None:
            fwr = 2
        rd = 2 * fwr + 1
        I8 = _to_gray_float32(I)
        I8 = cv2.normalize(I8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # 16-bin local entropy for speed
        bins = 16
        q = (I8 // (256 // bins)).astype(np.uint8)
        kernel = np.ones((rd, rd), np.uint8)
        area = float(rd * rd)
        H = np.zeros_like(I8, dtype=np.float32)
        for b in range(bins):
            mask = (q == b).astype(np.uint8)
            cnt = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_REFLECT)
            p = cnt.astype(np.float32) / area
            # avoid case: log(0)
            H -= np.where(p > 0, p * np.log2(p), 0.0)
        invalid = (I == 0)
        H[invalid] = 0
        return H
    if "std" in method:
        if fwr is None:
            fwr = 2
        rd = 2 * fwr + 1
        I32 = _to_gray_float32(I)
        mean = cv2.blur(I32, (rd, rd), borderType=cv2.BORDER_REFLECT)
        mean_sq = cv2.blur(I32 * I32, (rd, rd), borderType=cv2.BORDER_REFLECT)
        var = np.clip(mean_sq - mean * mean, 0, None)
        std = np.sqrt(var, dtype=np.float32)
        invalid = (I == 0)
        std[invalid] = 0
        return std
    return _to_gray_float32(I)


def _thinning_zhang_suen(BW: np.ndarray, max_iters: int | None = None) -> np.ndarray:
    """
    Binary thinning (skeletonization) using Zhang-Suen. BW is boolean/0-1.
    """
    img = (BW.astype(np.uint8) > 0).copy()
    changed = True
    iters = 0
    while changed and (max_iters is None or iters < max_iters):
        changed = False
        iters += 1
        for step in [0, 1]:
            to_remove = []
            for y in range(1, img.shape[0] - 1):
                row = img[y - 1 : y + 2]
                for x in range(1, img.shape[1] - 1):
                    P = row[:, x - 1 : x + 2]
                    if P[1, 1] == 0:
                        continue
                    p2, p3, p4 = P[0, 1], P[0, 2], P[1, 2]
                    p5, p6, p7 = P[2, 2], P[2, 1], P[2, 0]
                    p8, p9, p1 = P[1, 0], P[0, 0], P[0, 1]  # wrap
                    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                    C = sum((neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1) for i in range(8))
                    N = sum(neighbors)
                    if 2 <= N <= 6 and C == 1:
                        if step == 0:
                            if p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
                                to_remove.append((y, x))
                        else:
                            if p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0:
                                to_remove.append((y, x))
            if to_remove:
                changed = True
                for y, x in to_remove:
                    img[y, x] = 0
    return img.astype(bool)


def generate_border_mask(
    image: np.ndarray,
    img_filter: str = "none",
    percentile_threshold: float = 50.0,
    threshold_direction: str = ">=",
    border_break_holes_flag: bool = False,
    border_thin_mask_flag: bool = False,
    foreground_mask: np.ndarray | None = None,
):
    if foreground_mask is None:
        foreground_mask = np.ones(image.shape, dtype=bool)
    if img_filter is None:
        img_filter = "none"
    assert 0 <= percentile_threshold <= 100
    pt = percentile_threshold / 100.0
    assert image.shape == foreground_mask.shape, "border mask image must be the same size as the foreground mask"

    Iflt = generate_image_to_threshold(image, img_filter)

    P = percentile(Iflt.ravel(), pt)
    op = threshold_direction
    if op == ">":
        BW = Iflt > P
    elif op == "<":
        BW = Iflt < P
    elif op == ">=":
        BW = Iflt >= P
    elif op == "<=":
        BW = Iflt <= P
    else:
        raise ValueError("invalid threshold operator")

    BW = BW & foreground_mask

    # Morphological cleanup
    BW = cv2.dilate(BW.astype(np.uint8), _disk(1), iterations=1).astype(bool)
    # approximate 'bridge' and 'diag' by a small close
    BW = cv2.morphologyEx(BW.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    # one thinning iteration
    BW = _thinning_zhang_suen(BW, max_iters=1)
    BW = cv2.morphologyEx(BW.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1).astype(bool)

    BW = fill_holes(BW, None, min_hole_size=10)

    if border_thin_mask_flag:
        BW = _thinning_zhang_suen(BW, max_iters=None)
    BW = cv2.morphologyEx(BW.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1).astype(bool)

    if border_break_holes_flag and "break_holes" in globals():
        BW = break_holes(Iflt, foreground_mask, BW)

    return BW


def generate_seed_mask(
    image: np.ndarray,
    img_filter: str,
    percentile_thresholdL: float,
    threshold_operatorL: str,
    percentile_thresholdR: float,
    threshold_operatorR: str,
    min_obj_size: int,
    max_obj_size: int,
    circularity_threshold: float | None = None,
    cluster_distance: float | None = None,
    foreground_mask: np.ndarray | None = None,
    border_mask: np.ndarray | None = None,
):
    if foreground_mask is None:
        foreground_mask = np.ones(image.shape, dtype=bool)
    if img_filter is None:
        img_filter = "none"
    assert 0 <= percentile_thresholdL <= 100 and 0 <= percentile_thresholdR <= 100
    pL = percentile_thresholdL / 100.0
    pR = percentile_thresholdR / 100.0
    assert image.shape == foreground_mask.shape, "Seed mask image must be the same size as the foreground mask"

    I = _to_gray_float32(image)

    if border_mask is not None and border_mask.size:
        foreground_mask = foreground_mask & (~border_mask.astype(bool))

    Iflt = generate_image_to_threshold(I, img_filter)

    P1, P2 = percentile(Iflt[foreground_mask], np.array([pL, pR], dtype=float))
    def _apply(op, X, T):
        return {"<": X < T, "<=": X <= T, ">": X > T, ">=": X >= T}[op]
    BW_L = _apply(threshold_operatorL, Iflt, P1)
    BW_R = _apply(threshold_operatorR, Iflt, P2)
    BW = BW_L & BW_R
    BW = BW & foreground_mask

    # Fill holes smaller than twice min size and remove small objects
    BW = fill_holes(BW, None, min_hole_size=min_obj_size * 2)
    BW = _remove_small_objects(BW, min_size=int(min_obj_size), connectivity=8)

    # Remove large objects
    num, labels, stats, _ = cv2.connectedComponentsWithStats(BW.astype(np.uint8), connectivity=8)
    for k in range(1, num):
        if stats[k, cv2.CC_STAT_AREA] > max_obj_size:
            BW[labels == k] = False

    if circularity_threshold is not None:
        BW = filter_by_circularity(BW, float(circularity_threshold))

    if cluster_distance is not None:
        BW = cluster_objects(BW, float(cluster_distance), foreground_mask.astype(bool))

    return BW


def labeled_geodesic_dist(marker_matrix: np.ndarray, mask_matrix: np.ndarray):
    """
    Propagate labels (4-connected) from marker_matrix within mask_matrix and
    return (marker_matrix_filled, dist_mat) where dist_mat is the wave iteration count.
    """
    m, n = marker_matrix.shape
    assert mask_matrix.dtype == bool and mask_matrix.shape == (m, n)
    labels = marker_matrix.astype(np.int32, copy=True)
    dist_mat = np.full((m, n), np.inf, dtype=np.float32)
    dist_mat[labels > 0] = 0
    dist_mat[~mask_matrix] = np.nan

    # Initialize edge pixels queue (seeds with at least one 8-neighbor unlabeled or out-of-bounds)
    q = deque()
    seen = np.zeros((m, n), dtype=bool)
    for y, x in zip(*np.nonzero(labels > 0)):
        # edge if any 8-neighbor is unlabeled or outside
        edge = (y == 0 or x == 0 or y == m - 1 or x == n - 1)
        if not edge:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if labels[ny, nx] == 0:
                        edge = True
                        break
                if edge:
                    break
        if edge:
            q.append((y, x))
            seen[y, x] = True

    iter_count = 0
    while q:
        # Process current layer
        layer = list(q)
        q.clear()
        iter_count += 1
        for y, x in layer:
            lab = labels[y, x]
            # 4-neighbors only
            for dy, dx in ((0, -1), (-1, 0), (1, 0), (0, 1)):
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= m or nx < 0 or nx >= n:
                    continue
                if not mask_matrix[ny, nx]:
                    continue
                if labels[ny, nx] != 0:
                    continue
                labels[ny, nx] = lab
                dist_mat[ny, nx] = iter_count
                if not seen[ny, nx]:
                    q.append((ny, nx))
                    seen[ny, nx] = True
                    
    return labels, dist_mat


def geodesic_imdilate(BW: np.ndarray, allowed_mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Masked (geodesic) dilation: iteratively dilate BW but clamp to allowed_mask each iteration.
    """
    out = BW.astype(bool, copy=True)
    allowed = allowed_mask.astype(bool, copy=False)
    for _ in range(int(max(0, radius))):
        out = cv2.dilate(out.astype(np.uint8), _disk(1), iterations=1).astype(bool) & allowed
    return out


def geodesic_imopen(BW: np.ndarray, allowed_mask: np.ndarray, radius: int) -> np.ndarray:
    er = cv2.erode(BW.astype(np.uint8), _disk(int(max(1, radius))), iterations=1).astype(bool)
    return geodesic_imdilate(er, allowed_mask, radius)


def geodesic_imclose(BW: np.ndarray, allowed_mask: np.ndarray, radius: int) -> np.ndarray:
    dl = geodesic_imdilate(BW, allowed_mask, radius)
    er = cv2.erode(dl.astype(np.uint8), _disk(int(max(1, radius))), iterations=1).astype(bool)
    return er


def iterative_geodesic_gray_dilate(I: np.ndarray, BW: np.ndarray, allowed_mask: np.ndarray, radius: int, _step: float = 0.5) -> np.ndarray:
    """
    Simple placeholder: geodesic binary dilation limited by allowed_mask, repeated 'radius' times.
    """
    return geodesic_imdilate(BW, allowed_mask, radius)


def morphOp(I: np.ndarray, BW: np.ndarray, op_str: str, radius: int, border_mask: np.ndarray | None = None):
    use_border_flag = border_mask is not None
    if radius == 0:
        return BW
    allowed = None if not use_border_flag else (~border_mask.astype(bool))

    op = "".join(ch for ch in (op_str or "").lower() if ch.isalnum())
    if op == "dilate":
        if use_border_flag:
            return geodesic_imdilate(BW, allowed, radius)
        else:
            return cv2.dilate(BW.astype(np.uint8), _disk(radius), iterations=1).astype(bool)
    elif op == "erode":
        return cv2.erode(BW.astype(np.uint8), _disk(radius), iterations=1).astype(bool)
    elif op == "close":
        if use_border_flag:
            return geodesic_imclose(BW, allowed, radius)
        else:
            return cv2.morphologyEx(BW.astype(np.uint8), cv2.MORPH_CLOSE, _disk(radius), iterations=1).astype(bool)
    elif op == "open":
        if use_border_flag:
            return geodesic_imopen(BW, allowed, radius)
        else:
            return cv2.morphologyEx(BW.astype(np.uint8), cv2.MORPH_OPEN, _disk(radius), iterations=1).astype(bool)
    elif op == "iterativegraydilate":
        return iterative_geodesic_gray_dilate(I, BW, allowed if use_border_flag else np.ones_like(BW, dtype=bool), radius, 0.5)
    else:
        return BW


def imadjust(img, low_in=0.3, high_in=0.7, low_out=0.0, high_out=1.0):
    """
    Maps intensities from [low_in, high_in] -> [low_out, high_out] with saturation,
    matching MATLAB imadjust semantics. Works on grayscale. input can be int or float.
    """
    I = img.astype(np.float32)
    # Normalize to [0,1] using dtype range 
    # MATLAB thresholds are in [0,1] regardless of dtype
    if np.issubdtype(img.dtype, np.integer):
        maxv = float(np.iinfo(img.dtype).max)
        I = I / maxv
    else:
        # If float but not already in [0,1], rescale by dynamic range
        mn, mx = np.nanmin(I), np.nanmax(I)
        if mx > mn:
            I = (I - mn) / (mx - mn)
        else:
            I = np.zeros_like(I)
    # Piecewise linear mapping with saturation
    I = np.clip(I, low_in, high_in)
    I = (I - low_in) / (high_in - low_in + 1e-8)
    I = I * (high_out - low_out) + low_out
    return I.astype(np.float32)

def tophat_filter(img_float01, radius=4):
    """
    White tophat on grayscale float image in [0,1], disk structuring element.
    """
    r = int(max(1, radius))
    # Build a disk SE
    y, x = np.ogrid[-r:r+1, -r:r+1]
    se = (x*x + y*y) <= (r*r)
    kernel = se.astype(np.uint8)
    # OpenCV expects 32F
    # we use 32F
    return cv2.morphologyEx(img_float01, cv2.MORPH_TOPHAT, kernel)

def segplot(S, txt=True, colors_vector=None, shuffle=True, ax=None, title_prefix="Number of objects in image = "):
    """
    Display a labeled mask S with distinct colors and optional label numbers.
    S: 2D array of ints (0 = background).
    """
    L = S.astype(np.int32, copy=False)
    max_lab = int(L.max()) if L.size else 0
    if max_lab == 0:
        if ax is None:
            plt.figure(); ax = plt.gca()
        ax.imshow(np.zeros((*L.shape, 3), dtype=np.float32))
        ax.set_title(title_prefix + "0")
        ax.axis('off')
        return ax

    # Build a color table. Avoid color for background i.e. index 0
    if colors_vector is None:
        cmap = plt.get_cmap('jet', max_lab)
        table = (cmap(np.arange(1, max_lab+1))[:, :3])  # RGB
    else:
        table = np.array(colors_vector, dtype=np.float32)
        if table.shape[0] < max_lab:
            # tile if needed
            reps = int(np.ceil(max_lab / table.shape[0]))
            table = np.tile(table, (reps, 1))[:max_lab]
    if shuffle:
        rng = np.random.default_rng(0)  # deterministic shuffle
        perm = rng.permutation(max_lab)
        table = table[perm]

    # Map labels to colors
    rgb = np.zeros((*L.shape, 3), dtype=np.float32)
    # Build a map: label -> color
    # If shuffled, we need the inverse permutation so that text numbers match actual label IDs
    for lab in range(1, max_lab + 1):
        color = table[lab - 1]
        rgb[L == lab] = color

    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.imshow(rgb)
    # Count objects (non-zero labels present)
    present = np.unique(L[L > 0])
    nb_objects = present.size
    ax.set_title(f"{title_prefix}{nb_objects}")
    ax.axis('off')

    # Place text at first occurrence (scan order) of each label
    if txt:
        # Find first index per label
        flat = L.ravel()
        h, w = L.shape
        for lab in present:
            idx = np.argmax(flat == lab)  # first True
            if flat[idx] != lab:
                continue
            y, x = divmod(idx, w)
            ax.text(x, y, str(lab), fontsize=6, fontweight='bold',
                    color='k', bbox=dict(facecolor='w', edgecolor='none', pad=0.2))
    return ax


def _normalize_to_uint8(gray):
    g = gray.astype(np.float32)
    if np.issubdtype(gray.dtype, np.integer):
        maxv = float(np.iinfo(gray.dtype).max)
        if maxv > 0:
            g = g / maxv
    else:
        mn, mx = np.nanmin(g), np.nanmax(g)
        if mx > mn:
            g = (g - mn) / (mx - mn)
        else:
            g = np.zeros_like(g)
    g = np.clip(g, 0, 1)
    return (g * 255).astype(np.uint8)

def segplot_overlay(
    gray_image,
    labels,
    thickness: int = 1,
    show_ids: bool = True,
    id_font_scale: float = 0.35,
    id_thickness: int = 1,
    rng_seed: int = 0,
    return_image: bool = False,
    ax=None,
    title: str | None = None,
):
    """
    Draws colored contours of each label over the grayscale image (like your second screenshot).

    gray_image : 2D array (any dtype)
    labels     : 2D int array; 0 = background
    """
    # Prepare background
    bg = _normalize_to_uint8(gray_image)
    bg_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

    # Blank canvas for contours
    canvas = np.zeros_like(bg_rgb, dtype=np.uint8)

    max_label = int(labels.max()) if labels.size else 0
    if max_label == 0:
        if return_image:
            return bg_rgb
        if ax is None:
            plt.figure(); ax = plt.gca()
        ax.imshow(cv2.cvtColor(bg_rgb, cv2.COLOR_BGR2RGB))
        ax.set_title(title or "No objects")
        ax.axis('off')
        return ax

    # Generate distinct colors
    rng = np.random.default_rng(rng_seed)
    colors = (rng.uniform(0, 255, size=(max_label, 3))).astype(np.uint8)

    # Draw contours + optional IDs
    for lab in range(1, max_label + 1):
        mask = (labels == lab).astype(np.uint8)
        if mask.sum() == 0:
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue

        color = tuple(int(c) for c in colors[lab - 1])
        # Outline (on canvas, not on bg directly)
        cv2.drawContours(canvas, cnts, -1, color, thickness, lineType=cv2.LINE_AA)

        if show_ids:
            M = cv2.moments(mask, binaryImage=True)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # White halo
                cv2.putText(canvas, str(lab), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, id_font_scale, (255, 255, 255), id_thickness + 1, cv2.LINE_AA)
                # Black text on top
                cv2.putText(canvas, str(lab), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, id_font_scale, (0, 0, 0), id_thickness, cv2.LINE_AA)

    # add colored lines on top of background
    overlay_bgr = cv2.add(bg_rgb, canvas)

    if return_image:
        return overlay_bgr

    if ax is None:
        plt.figure(figsize=(9, 7))
        ax = plt.gca()
    ax.imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title or "Segmentation overlay")
    ax.axis('off')
    return ax

# TODO: put in processing.utils folder
def sort_files(self, path_to_folder: str) -> list:
    dict_of_files = {}
    for entry in os.scandir(path_to_folder):
        if entry.is_file() and entry.name.lower().endswith('.tif'):
            try:
                reduced_timestamp = reduce_timestamp(entry.name)
                dict_of_files[entry.name] = int(reduced_timestamp)
            except (AttributeError, IndexError):
                print(f"Skipping file with incorrect format: {entry.name}")
    sorted_items = sorted(dict_of_files.items(), key=lambda item: item[1])
    return [file_name for file_name, _ in sorted_items]

# TODO: put in processing.utils folder
def reduce_timestamp(self, file_name: str) -> str:
    parts = file_name.split("_")
    timestamp_part = parts[3].split(".")[0]
    match = re.match(r"(\d+)d(\d+)h(\d+)m", timestamp_part)
    days, hours, minutes = match.groups()
    return f"{int(days):d}{int(hours):02d}{int(minutes):02d}"

def get_segmentation_images(
    source_folder_path: str,
    min_cell_area = 100,
    morph_op = "erode", morph_radius = 0,
    min_seed_size = 250, 
    fogbank_direction = "max_to_min",
    min_object_area = 100, 
    perc_binning = 5
):
    """
    Creates a new folder with a '_segmentation' suffix in the same parent
    directory as the source folder, then scans the source folder for image
    files and copies them to the new folder.

    Args:
        source_folder_path (str): The full path to the source folder containing images.
    """
    # Define the paths using pathlib for robust handling
    source_dir = Path(source_folder_path)

    # Safety Check to Ensure the source directory actually exists
    if not source_dir.is_dir():
        print(f"The source folder was not found at '{source_dir}'")
        return

    target_dir = source_dir.parent / f"{source_dir.name}_segmentation"

    # create the new directory
    # The exist_ok=True argument prevents an error if the folder already exists.
    try:
        target_dir.mkdir(exist_ok=True)
        print(f"Successfully created or found target folder: '{target_dir}'")
    except OSError as e:
        print(f"Error creating directory '{target_dir}': {e}")
        return

    # os.scandir() to go through the source folder
    print(f"\nScanning '{source_dir}' for images...")
    image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
    
    with os.scandir(source_dir) as entries:
        for entry in entries:
            # Check if the entry is a file and has a valid image extension
            if entry.is_file() and entry.name.lower().endswith(image_extensions):
                
                # Define the full source and destination paths for the file
                source_file = Path(entry.path)
                destination_file = target_dir / entry.name

                # This is where you would normally do your segmentation processing.
                I = cv2.imread(source_file, cv2.IMREAD_UNCHANGED)
                if I is None:
                    raise FileNotFoundError(f"Could not read: {source_file}")
                if I.ndim == 3:
                    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
                    
                #  low_in=0.3, high_in=0.7, low_out=0, high_out=1 (specific settings)
                I_adj = imadjust(I, low_in=0.3, high_in=0.7, low_out=0.0, high_out=1.0)
                I_top = tophat_filter(I_adj, radius=4)

                # build foreground ebfore fogbank segmentation
                S_mask = egt_segmentation(
                    I_adj,                       # I_top works but I_adj works well with gradients
                    min_cell_size=int(min_cell_area),
                    min_hole_size=0,
                    max_hole_size=np.inf,
                    hole_min_perct_intensity=0.0,
                    hole_max_perct_intensity=100.0,
                    fill_holes_bool_oper="AND",
                    manual_finetune=0,
                ).astype(bool)

                # Optional Relax the mask slightly so growth isn’t too tight
                S_mask = cv2.dilate(S_mask.astype(np.uint8), _disk(1), 1).astype(bool)

                # FogBank segmentation
                fog_dir_flag = 0 if (str(fogbank_direction).lower() in ["max->min", "max_to_min", "max2min"]) else 1
                foreground_mask = S_mask
                mask_matrix     = S_mask

                # Apply requested morphological operation BEFORE/AROUND FogBank seed growth (radius 0 --> no-op)
                _ = morphOp(I_top, (I_top > -np.inf), morph_op, int(morph_radius), None)

                # Run FogBank percentile geodesic grower constrained to S_mask
                seed_image, nb_peaks = fog_bank_perctile_geodist(
                    I_top,
                    foreground_mask,
                    mask_matrix,
                    min_peak_size=int(min_seed_size),
                    min_object_size=int(min_object_area),
                    fogbank_direction=int(fog_dir_flag),
                    perc_binning=float(perc_binning),
                )

                # Overlay contours
                segplot_overlay(I_adj, seed_image, thickness=1, show_ids=True, title="FogBank contours")
                output_filename = source_file.stem + ".png"
                destination_path = target_dir / output_filename
                
                # Save the current plot to the specified file path
                # remove unnecessary white space around the image
                plt.savefig(destination_path, bbox_inches='tight', pad_inches=0, dpi=150)
                
                # close the plot to free up memory for the next loop iteration
                plt.close()
                
    print("Processing complete")

        

#  Pipeline for specifc settings
# this is our special case and can be changed depending
def run_segmentation_pipeline(
    tiff_path="/Users/patrik/cell_diffusion_modelling/558/A1/VID558_A1_1_04d23h07m.tif",
    min_cell_area=100,                   # FogBank "Min cell area"
    morph_op="erode", morph_radius=0,   # Morphological operation (erode) with radius 0 (no-op)
    min_seed_size=250,                  # Min Seed Size
    fogbank_direction="max_to_min",     # "max_to_min" or "min_to_max"
    min_object_area=100,                # Min object area (same as min_cell_area here)
    perc_binning=5,                     # default
    show_plots=True
):
    
    # already have grayscale
    I = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
    if I is None:
        raise FileNotFoundError(f"Could not read: {tiff_path}")
    if I.ndim == 3:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    #  low_in=0.3, high_in=0.7, low_out=0, high_out=1 (specific settings)
    I_adj = imadjust(I, low_in=0.3, high_in=0.7, low_out=0.0, high_out=1.0)
    I_top = tophat_filter(I_adj, radius=4)

    # build foreground ebfore fogbank segmentation
    S_mask = egt_segmentation(
        I_adj,                       # I_top works but I_adj works well with gradients
        min_cell_size=int(min_cell_area),
        min_hole_size=0,
        max_hole_size=np.inf,
        hole_min_perct_intensity=0.0,
        hole_max_perct_intensity=100.0,
        fill_holes_bool_oper="AND",
        manual_finetune=0,
    ).astype(bool)

    # Optional Relax the mask slightly so growth isn’t too tight
    S_mask = cv2.dilate(S_mask.astype(np.uint8), _disk(1), 1).astype(bool)

    # FogBank segmentation
    fog_dir_flag = 0 if (str(fogbank_direction).lower() in ["max->min", "max_to_min", "max2min"]) else 1
    foreground_mask = S_mask
    mask_matrix     = S_mask

    # Apply requested morphological operation BEFORE/AROUND FogBank seed growth (radius 0 --> no-op)
    _ = morphOp(I_top, (I_top > -np.inf), morph_op, int(morph_radius), None)

    # Run FogBank percentile geodesic grower constrained to S_mask
    seed_image, nb_peaks = fog_bank_perctile_geodist(
        I_top,
        foreground_mask,
        mask_matrix,
        min_peak_size=int(min_seed_size),
        min_object_size=int(min_object_area),
        fogbank_direction=int(fog_dir_flag),
        perc_binning=float(perc_binning),
    )

    # Overlay contours
    segplot_overlay(I_adj, seed_image, thickness=1, show_ids=True, title="FogBank contours")
    plt.show()

    # Convert labeled masks to binary 7x7 dot markers at centroids (not shown atm)
    dot_mask = np.zeros_like(seed_image, dtype=bool)
    max_label = int(seed_image.max())
    if max_label > 0:
        h, w = seed_image.shape
        half = 3  # 7x7 square
        for lab in range(1, max_label + 1):
            ys, xs = np.nonzero(seed_image == lab)
            if ys.size == 0:
                continue
            cy = int(np.round(ys.mean()))
            cx = int(np.round(xs.mean()))
            y0, y1 = max(0, cy - half), min(h, cy + half + 1)
            x0, x1 = max(0, cx - half), min(w, cx + half + 1)
            dot_mask[y0:y1, x0:x1] = True

    return {
        "adjusted": I_adj,
        "tophat": I_top,
        "labels": seed_image,
        "num_labels": nb_peaks,
        "dot_mask": dot_mask.astype(np.uint8)
    }

    # # 1) Read TIFF (grayscale)
    # I = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
    # if I is None:
    #     raise FileNotFoundError(f"Could not read: {tiff_path}")
    # if I.ndim == 3:
    #     I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    # # 2) MATLAB imadjust with low_in=0.3, high_in=0.7, low_out=0, high_out=1
    # I_adj = imadjust(I, low_in=0.3, high_in=0.7, low_out=0.0, high_out=1.0)

    # # 3) imtophat with disk radius 4
    # I_top = tophat_filter(I_adj, radius=4)

    # # 4) FogBank segmentation
    # fog_dir_flag = 0 if (str(fogbank_direction).lower() in ["max->min", "max_to_min", "max2min"]) else 1
    # foreground_mask = np.ones_like(I_top, dtype=bool)     # no explicit foreground given; use full image
    # mask_matrix = foreground_mask.copy()

    # # Apply requested morphological operation BEFORE/AROUND FogBank seed growth (radius 0 => no-op)
    # # Keeping this call to reflect your settings.
    # _ = morphOp(I_top, (I_top > -np.inf), morph_op, int(morph_radius), None)  # no-op with radius 0

    # # Run FogBank percentile geodesic grower
    # seed_image, nb_peaks = fog_bank_perctile_geodist(
    #     I_top,
    #     foreground_mask,
    #     mask_matrix,
    #     min_peak_size=int(min_seed_size),
    #     min_object_size=int(min_object_area),
    #     fogbank_direction=int(fog_dir_flag),
    #     perc_binning=float(perc_binning),
    # )
    
    # segplot_overlay(I_adj, seed_image, thickness=1, show_ids=True, title="FogBank contours")
    # plt.show()

    # # 5) Convert labeled masks to binary "dot masks" by replacing each label with a 7x7 square at its centroid
    # dot_mask = np.zeros_like(seed_image, dtype=bool)
    # max_label = int(seed_image.max())
    # if max_label > 0:
    #     h, w = seed_image.shape
    #     half = 3  # 7x7 square
    #     for lab in range(1, max_label + 1):
    #         ys, xs = np.nonzero(seed_image == lab)
    #         if ys.size == 0:
    #             continue
    #         # center of mass (unweighted centroid of pixels)
    #         cy = int(np.round(ys.mean()))
    #         cx = int(np.round(xs.mean()))
    #         y0, y1 = max(0, cy - half), min(h, cy + half + 1)
    #         x0, x1 = max(0, cx - half), min(w, cx + half + 1)
    #         dot_mask[y0:y1, x0:x1] = True
    # # # Optionally show plots
    # # if show_plots:
    # #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # #     axs[0].imshow(I_adj, cmap='gray'); axs[0].set_title("imadjust (0.3→0.7)"); axs[0].axis('off')
    # #     axs[1].imshow(I_top, cmap='gray'); axs[1].set_title("imtophat (disk r=4)"); axs[1].axis('off')
    # #     segplot(seed_image, txt=True, shuffle=True, ax=axs[2]); axs[2].set_title("FogBank labels")
    # #     plt.tight_layout()
    # #     plt.show()

    # #     plt.figure(figsize=(6,6))
    # #     plt.imshow(dot_mask, cmap='gray')
    # #     plt.title("Binary 7x7 dot mask")
    # #     plt.axis('off')
    # #     plt.show()

    # # Return key artifacts
    # return {
    #     "adjusted": I_adj,
    #     "tophat": I_top,
    #     "labels": seed_image,
    #     "num_labels": nb_peaks,
    #     "dot_mask": dot_mask.astype(np.uint8)
    # }
    
#run_segmentation_pipeline()




if __name__ == "__main__":

    path_from_user = input("Enter the full path to your TIFF folder and press Enter: ")
    
    # Strip whitespace and any quotes the user might have added (e.g., by dragging and dropping)
    path_to_your_tiffs = path_from_user.strip().strip("'\"")

    if path_to_your_tiffs:
        get_segmentation_images(path_to_your_tiffs)
    else:
        print("No path entered. Exiting program.")