import os
import json
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm

import pyarrow as pa
import pyarrow.parquet as pq

import sys
sys.path.append(os.path.dirname(__file__))
import simulation  # noqa: E402


###################
# Config & Inputs #
###################

OUTPUT_DIR    = os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "output"))
N_SIMULATIONS = int(os.environ.get("N_SIMULATIONS", "500"))
SEED          = int(os.environ.get("SEED", "42"))

# Parallelism and batching
N_JOBS     = int(os.environ.get("N_JOBS", max(1, (os.cpu_count() or 2) - 1)))
default_batch_size = N_SIMULATIONS / 4
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", default_batch_size))

# Always-on full snapshots toggle (all particles, last timestep)
SAVE_SNAPSHOTS = int(os.environ.get("SAVE_SNAPSHOTS", "0"))  # 0/1

# Measurement & arena
measure_rect_side = 702 // 2
measure_rect = [-measure_rect_side, -measure_rect_side, measure_rect_side, measure_rect_side]
arena_side = measure_rect_side * 1.3
arena_rect = [-arena_side, -arena_side, arena_side, arena_side]

simulation_parameters: Dict[str, Any] = {
    "num_particles": 100,
    "max_particles": 2000,
    # "simulation_length": 3600 * 24 * 4,
    "simulation_length": 428820,     # 4d23h07m
    "simulation_step": 50,
    "dr": 2.6,
    "rMax": 65,
    "arena_rect": arena_rect,
    "measure_rect": measure_rect,
    "edge_force": 0.04,
    "repulsion_force": 0.1,
    "beta_proportion": 0.8,
    "ceta": 1,
}

parameter_priors: Dict[str, Tuple[float, float]] = {
    "log10_alpha": (-6.0, -4.8),
    "log10_d":     (-3.0,  1.0),
    "r":           ( 2.0, 20.0),
}


##########################
# Summary statistic utils
##########################

def particles_in_rectangle(P: np.ndarray, rect: List[float]) -> int:
    if P.size == 0:
        return 0
    left, bottom, right, top = rect
    return int(np.sum((P[:, 0] >= left) & (P[:, 0] <= right) & (P[:, 1] >= bottom) & (P[:, 1] <= top)))

def pair_correlation(P: np.ndarray, rect: List[float], rMax: float, dr: float) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.arange(0, rMax, dr)
    radii = (edges[:-1] + edges[1:]) / 2.0
    if P.size == 0 or P.shape[0] < 2:
        return np.zeros_like(radii), radii

    left, bottom, right, top = rect
    interior_mask = (
        (P[:, 0] >= left + rMax) &
        (P[:, 0] <= right - rMax) &
        (P[:, 1] >= bottom + rMax) &
        (P[:, 1] <= top - rMax)
    )
    if not np.any(interior_mask):
        return np.zeros_like(radii), radii

    area = (right - left) * (top - bottom)
    density = particles_in_rectangle(P, rect) / max(area, 1e-12)

    D = cdist(P, P, metric="euclidean")
    np.fill_diagonal(D, np.nan)
    D = D[interior_mask, :]
    D = D[~np.isnan(D)]

    annuli_areas = np.pi * np.diff(edges**2)
    counts, _ = np.histogram(D.flatten(), bins=edges)

    g_r = counts / (np.sum(interior_mask) * annuli_areas * max(density, 1e-12))
    return g_r, radii


########################
# Simulation execution #
########################

def draw_parameters(rs: np.random.RandomState) -> Tuple[float, float, float]:
    log10_d = rs.uniform(*parameter_priors["log10_d"])
    log10_alpha = rs.uniform(*parameter_priors["log10_alpha"])
    r = rs.uniform(*parameter_priors["r"])
    return log10_d, log10_alpha, r

def run_one(sim_id: int, seed: int):
    """Return (sim_id, summary, drawn_params, exit_code, radii, P_last, r)."""
    params = simulation_parameters
    rs = np.random.RandomState(seed=seed)

    log10_d, log10_alpha, r = draw_parameters(rs)

    diffusion = 10.0 ** log10_d
    alpha = 10.0 ** log10_alpha
    cell_speed = np.sqrt(2.0 * diffusion)
    
    eps = 1e-4
    z_eps = norm.ppf(1 - eps)
    repulsion_edge_force = z_eps * np.sqrt(2* diffusion/ params['simulation_step'])
    repulsion_edge_force = z_eps * cell_speed / np.sqrt(params['simulation_step'])
    
    if repulsion_edge_force == 0:
        repulsion_edge_force = 0.01
        
    arena  = simulation.RectangularArena(params["arena_rect"], repulsion_edge_force , r)
        
    params["ceta"] = 1 / (2*3.07*r)
    #arena = simulation.RectangularArena(params["arena_rect"], params["edge_force"], r)
    events = [simulation.BirthEvent(alpha, params["beta_proportion"], params["ceta"], r)]
    collision_fn = simulation.createRigidPotential(params["repulsion_force"], 2.0 * r)

    sim = simulation.Simulation(
        minTimeStep=params["simulation_step"],
        initialParticles=params["num_particles"],
        maxParticles=params["max_particles"],
        arena=arena,
        particleSpeed=cell_speed,
        particleCollision=collision_fn,
        particleCollisionMaxDistance=2.0 * r,
        events=events,
        rs=rs,
    )

    P, ecode = sim.simulate(params["simulation_length"])

    pc, radii = pair_correlation(P, params["measure_rect"], params["rMax"], params["dr"])
    pc = np.nan_to_num(pc, nan=0.0)
    N = particles_in_rectangle(P, params["measure_rect"])

    summary = np.concatenate(([N], pc))
    drawn = {"log10_d": log10_d, "log10_alpha": log10_alpha, "r": r}

    return sim_id, summary, drawn, ecode, radii, P, r


#########################
# Partial correlations  #
#########################

def partial_corr(X: np.ndarray, y: np.ndarray, idx: int) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()

    ctrl = [j for j in range(Xc.shape[1]) if j != idx]
    if not ctrl:
        x = Xc[:, idx]
        denom = (np.linalg.norm(x) * np.linalg.norm(yc))
        return float(x @ yc / denom) if denom > 0 else np.nan

    X_ctrl = Xc[:, ctrl]
    XtX = X_ctrl.T @ X_ctrl
    ridge = 1e-8 * np.eye(XtX.shape[0])
    beta_y = np.linalg.solve(XtX + ridge, X_ctrl.T @ yc)
    resid_y = yc - X_ctrl @ beta_y

    xi = Xc[:, idx]
    beta_x = np.linalg.solve(XtX + ridge, X_ctrl.T @ xi)
    resid_xi = xi - X_ctrl @ beta_x

    denom = (np.linalg.norm(resid_xi) * np.linalg.norm(resid_y))
    return float(resid_xi @ resid_y / denom) if denom > 0 else np.nan

def compute_partial_correlations(params_df: pd.DataFrame, summaries_df: pd.DataFrame, radii: np.ndarray) -> pd.DataFrame:
    X = params_df[["log10_d", "log10_alpha", "r"]].to_numpy(dtype=float)
    stats = [col for col in summaries_df.columns if col != "N"]
    rows = []

    yN = summaries_df["N"].to_numpy(dtype=float)
    for i, pname in enumerate(["log10_d", "log10_alpha", "r"]):
        rows.append({"stat": "N", "radius": np.nan, "parameter": pname, "partial_corr": partial_corr(X, yN, i)})

    for j, col in enumerate(stats):
        y = summaries_df[col].to_numpy(dtype=float)
        for i, pname in enumerate(["log10_d", "log10_alpha", "r"]):
            rows.append({
                "stat": col,
                "radius": float(radii[j]) if j < len(radii) else np.nan,
                "parameter": pname,
                "partial_corr": partial_corr(X, y, i),
            })

    return pd.DataFrame(rows)


########################
# CLI overrides
########################

def _parse_cli_overrides(argv):
    pairs = {}
    toks = list(argv[1:])
    i = 0
    while i < len(toks):
        t = toks[i]
        if "=" in t and t != "=":
            key, val = t.split("=", 1)
            key = key.strip(); val = val.strip()
            if val == "" and i + 1 < len(toks) and toks[i+1] != "=":
                i += 1; val = toks[i].strip()
            if key: pairs[key] = val
        elif t == "=" and i - 1 >= 0 and i + 1 < len(toks):
            key = toks[i-1].strip(); val = toks[i+1].strip()
            if key: pairs[key] = val
            i += 1
        i += 1
    return pairs

def _coerce_int(s, default=None):
    try: return int(s)
    except Exception: return default

def _apply_cli_overrides():
    overrides = _parse_cli_overrides(sys.argv)
    global N_SIMULATIONS, SEED, OUTPUT_DIR, N_JOBS, BATCH_SIZE, SAVE_SNAPSHOTS
    if "N_SIMULATIONS" in overrides:
        v = _coerce_int(overrides["N_SIMULATIONS"], None)
        if v is not None and v > 0: N_SIMULATIONS = v
    if "SEED" in overrides:
        v = _coerce_int(overrides["SEED"], None)
        if v is not None: SEED = v
    if "OUTPUT_DIR" in overrides:
        out = overrides["OUTPUT_DIR"]
        if out: OUTPUT_DIR = out
    if "N_JOBS" in overrides:
        v = _coerce_int(overrides["N_JOBS"], None)
        if v is not None and v > 0: N_JOBS = v
    if "BATCH_SIZE" in overrides:
        v = _coerce_int(overrides["BATCH_SIZE"], None)
        if v is not None and v > 0: BATCH_SIZE = v
    if "SAVE_SNAPSHOTS" in overrides:
        v = _coerce_int(overrides["SAVE_SNAPSHOTS"], None)
        if v in (0,1): SAVE_SNAPSHOTS = v


########################
# Parquet single-file “append” helpers
########################

def _open_parquet_singlefile_appender(target_path: str, first_batch_df: pd.DataFrame):
    if os.path.exists(target_path):
        tmp_path = target_path + ".tmp"
        pf = pq.ParquetFile(target_path)
        existing_schema = pf.schema_arrow
        col_order = existing_schema.names
        writer = pq.ParquetWriter(tmp_path, existing_schema, compression="snappy")
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            writer.write_table(table)
        return writer, target_path, tmp_path, col_order
    else:
        table = pa.Table.from_pandas(first_batch_df, preserve_index=False)
        writer = pq.ParquetWriter(target_path, table.schema, compression="snappy")
        return writer, target_path, None, list(first_batch_df.columns)

def _finalize_parquet_singlefile_appender(writer: pq.ParquetWriter, final_path: str, tmp_path: str | None):
    writer.close()
    if tmp_path is not None:
        os.replace(tmp_path, final_path)


########################
# Main runner
########################

def main():
    _apply_cli_overrides()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Radii (small deterministic file → overwrite each run)
    edges = np.arange(0, simulation_parameters["rMax"], simulation_parameters["dr"])
    radii = (edges[:-1] + edges[1:]) / 2.0
    pc_cols = [f"pc_{i}" for i in range(len(radii))]

    radii_path = os.path.join(OUTPUT_DIR, "radii.parquet")
    pd.DataFrame({"pc_col": pc_cols, "radius": radii}).to_parquet(radii_path, index=False)

    sim_path = os.path.join(OUTPUT_DIR, "simulations.parquet")
    sim_cols = ["sim_id", "log10_d", "log10_alpha", "r", "N", *pc_cols, "exit_code"]

    snapshots_path = os.path.join(OUTPUT_DIR, "snapshots.parquet")

    rng = np.random.RandomState(SEED)
    all_seeds = rng.randint(0, 2**31 - 1, size=N_SIMULATIONS, dtype=np.int64)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    def pack_sim_row(sim_id, summary, drawn, ecode):
        row = {
            "sim_id": sim_id,
            "log10_d": drawn["log10_d"],
            "log10_alpha": drawn["log10_alpha"],
            "r": drawn["r"],
            "N": float(summary[0]),
            "exit_code": int(ecode),
        }
        for i, val in enumerate(summary[1:]):
            row[f"pc_{i}"] = float(val)
        return row

    # Bootstrap first batch to open writers
    first_end = min(BATCH_SIZE, N_SIMULATIONS)
    first_ids = np.arange(0, first_end, dtype=np.int64)
    first_seeds = all_seeds[0:first_end]

    first_rows = []
    first_snaps = []

    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        futs = [ex.submit(run_one, int(sim_id), int(seed)) for sim_id, seed in zip(first_ids, first_seeds)]
        for fut in as_completed(futs):
            sim_id, summary, drawn, ecode, _radii, P, r = fut.result()
            first_rows.append(pack_sim_row(sim_id, summary, drawn, ecode))
            if SAVE_SNAPSHOTS:
                # Write ALL cells/particles from last timestep
                if P.size:
                    for k in range(P.shape[0]):
                        first_snaps.append({"sim_id": int(sim_id), "x": float(P[k,0]), "y": float(P[k,1]), "r": float(r)})

    first_sim_df = pd.DataFrame(first_rows, columns=sim_cols)
    sim_writer, sim_final, sim_tmp, sim_col_order = _open_parquet_singlefile_appender(sim_path, first_sim_df)
    first_sim_df = first_sim_df.reindex(columns=sim_col_order)
    sim_writer.write_table(pa.Table.from_pandas(first_sim_df, preserve_index=False))

    snap_writer = None
    if SAVE_SNAPSHOTS:
        if first_snaps:
            first_snap_df = pd.DataFrame(first_snaps, columns=["sim_id","x","y","r"])
        else:
            first_snap_df = pd.DataFrame(columns=["sim_id","x","y","r"])
        snap_writer, snap_final, snap_tmp, snap_col_order = _open_parquet_singlefile_appender(
            snapshots_path, first_snap_df
        )
        if not first_snap_df.empty:
            first_snap_df = first_snap_df.reindex(columns=snap_col_order)
            snap_writer.write_table(pa.Table.from_pandas(first_snap_df, preserve_index=False))

    # Remaining batches
    written = len(first_rows)
    start = first_end
    try:
        while start < N_SIMULATIONS:
            end = min(start + BATCH_SIZE, N_SIMULATIONS)
            ids_batch = np.arange(start, end, dtype=np.int64)
            seeds_batch = all_seeds[start:end]

            sim_rows = []
            snap_rows = []

            with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
                futs = [ex.submit(run_one, int(sim_id), int(seed)) for sim_id, seed in zip(ids_batch, seeds_batch)]
                for fut in as_completed(futs):
                    sim_id, summary, drawn, ecode, _radii, P, r = fut.result()
                    sim_rows.append(pack_sim_row(sim_id, summary, drawn, ecode))
                    if SAVE_SNAPSHOTS and P.size:
                        for k in range(P.shape[0]):
                            snap_rows.append({"sim_id": int(sim_id), "x": float(P[k,0]), "y": float(P[k,1]), "r": float(r)})

            if sim_rows:
                batch_df = pd.DataFrame(sim_rows).reindex(columns=sim_col_order)
                sim_writer.write_table(pa.Table.from_pandas(batch_df, preserve_index=False))

            if SAVE_SNAPSHOTS and snap_rows:
                batch_snap = pd.DataFrame(snap_rows, columns=snap_col_order).reindex(columns=snap_col_order)
                snap_writer.write_table(pa.Table.from_pandas(batch_snap, preserve_index=False))

            written += len(sim_rows)
            start = end
            print(f"[batch] wrote {written}/{N_SIMULATIONS}")
    finally:
        _finalize_parquet_singlefile_appender(sim_writer, sim_final, sim_tmp)
        if SAVE_SNAPSHOTS and snap_writer is not None:
            _finalize_parquet_singlefile_appender(snap_writer, snap_final, snap_tmp)

    # Partial correlations (overwrite with current aggregate)
    sims_df = pd.read_parquet(sim_path, engine="pyarrow")
    pc_cols_present = [c for c in sims_df.columns if c.startswith("pc_")]
    valid_mask = sims_df["exit_code"].isin([0, 1]).to_numpy()
    summaries_df = sims_df.loc[valid_mask, ["N", *pc_cols_present]].reset_index(drop=True)
    params_df = sims_df.loc[valid_mask, ["log10_d", "log10_alpha", "r"]].reset_index(drop=True)

    # Radii again for safety
    edges = np.arange(0, simulation_parameters["rMax"], simulation_parameters["dr"])
    radii = (edges[:-1] + edges[1:]) / 2.0

    pcorr_df = compute_partial_correlations(params_df, summaries_df, radii)
    pcorr_path = os.path.join(OUTPUT_DIR, "partial_correlations.parquet")
    pcorr_df.to_parquet(pcorr_path, index=False)

    manifest = {
        "n_simulations_total": int(len(sims_df)),
        "n_simulations_valid": int(valid_mask.sum()),
        "pc_cols": pc_cols_present,
        "output_files": {
            "simulations": sim_path,
            "radii": radii_path,
            "partial_correlations": pcorr_path,
            "snapshots": os.path.join(OUTPUT_DIR, "snapshots.parquet") if SAVE_SNAPSHOTS else None,
        },
        "priors": parameter_priors,
        "simulation_parameters": simulation_parameters,
        "n_jobs": N_JOBS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "format": "parquet-snappy",
        "append_mode": "single-file rewrite (atomic replace)",
        "snapshots": {"enabled": bool(SAVE_SNAPSHOTS), "mode": "ALL particles, last timestep, every simulation"}
    }
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"Wrote: {sim_path}")
    print(f"Wrote: {pcorr_path}")
    print(f"Wrote: {radii_path}")
    if SAVE_SNAPSHOTS:
        print(f"Wrote: {os.path.join(OUTPUT_DIR, 'snapshots.parquet')}")


if __name__ == "__main__":
    main()
