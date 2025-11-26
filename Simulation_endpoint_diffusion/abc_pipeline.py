"""
Unified CLI to run Rejection, Regression-adjusted, or SMC ABC.

Examples:
1) Rejection ABC (bank reuse), 50 synthetic observed datasets:
   python abc_pipeline.py --method rejection --bank /path/simulations.parquet \
      --out-dir /path/abc_runs --reps 50 --n-accept 250 --parallel 6 --seed 123 --synthetic

2) Regression-adjusted ABC on real observed summaries from CSV:
   python abc_pipeline.py --method regression --bank /path/simulations.parquet \
      --observed-csv /path/observed.csv --out-dir /path/abc_runs --n-accept 250
   (observed.csv must have columns: N, pc_0..pc_{K-1}, and optionally truth_* columns)

3) SMC (bank reuse):
   python abc_pipeline.py --method smc --bank /path/simulations.parquet \
      --out-dir /path/abc_runs --reps 20 --T 4 --particles 2000 --parallel 6 --seed 7 --synthetic
"""
# abc_pipeline.py
# Unified CLI for Rejection, Regression-adjusted, and SMC (bank-reuse only)

from __future__ import annotations
import os, argparse, numpy as np, pandas as pd
from typing import Dict, Any, List
from abc_methods import load_bank_parquet, Bank, RejectionABC, RegressionABC, SMCABC
from simulate_once import simulate_once as sim_once

def build_observed_list(bank: Bank, reps: int, synthetic: bool, seed: int,
                        observed_csv: str | None) -> List[Dict[str,Any]]:
    if synthetic:
        rng = np.random.default_rng(seed)
        return [sim_once(rng, K=bank.K) for _ in range(reps)]

    if observed_csv is None:
        raise ValueError("Provide --observed-csv for non-synthetic runs.")
    df = pd.read_csv(observed_csv)
    pc_cols = [c for c in df.columns if c.lower().startswith("pc_")]
    if len(pc_cols) != bank.K:
        raise ValueError(f"Observed PCF columns mismatch: got {len(pc_cols)}, bank K={bank.K}.")
    out = []
    for _, row in df.iterrows():
        truth = None
        if {"log10_alpha","log10_D","r"}.issubset(row.index):
            truth = {"log10_alpha": float(row["log10_alpha"]),
                     "log10_D": float(row["log10_D"]),
                     "r": float(row["r"])}
        out.append({"N": float(row["N"]),
                    "PCF": row[pc_cols].to_numpy(dtype=float),
                    "truth": truth})
    return out

def append_frames(out_dir: str, res_list: List[Dict[str,Any]], rep_offset_start: int):
    accepts_path = os.path.join(out_dir, "abc_accepts_all.csv")
    points_path  = os.path.join(out_dir, "abc_points_all.csv")
    for j,res in enumerate(res_list):
        rid = rep_offset_start + j
        acc = res["accepts"].copy(); acc["rep_id"] = rid
        pts = res["points"].copy();  pts["rep_id"] = rid
        acc.to_csv(accepts_path, mode="a", index=False, header=not os.path.exists(accepts_path))
        pts.to_csv(points_path,  mode="a", index=False, header=not os.path.exists(points_path))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["rejection","regression","smc"])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true", help="Simulate observed data once per rep")
    ap.add_argument("--observed-csv", default=None, help="CSV with observed summaries (N and pc_* bins)")

    # Rejection / Regression
    ap.add_argument("--n-accept", type=int, default=250)
    ap.add_argument("--frac", type=float, default=None, help="Regression ABC: fraction of bank instead of n-accept")

    # SMC (bank-reuse only)
    ap.add_argument("--T", type=int, default=4, help="SMC populations")
    ap.add_argument("--particles", type=int, default=1000, help="Particles per population (cap posterior sample)")
    ap.add_argument("--kernel-scale", type=float, default=0.5, help="Diagonal kernel scale in parameter space")

    ap.add_argument("--no-exit-filter", action="store_true", help="Do not filter bank by exit_code==0")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    bank = load_bank_parquet(args.bank, require_exit_code_zero=(not args.no_exit_filter))

    # rep-id offset to append safely
    accepts_path = os.path.join(args.out_dir, "abc_accepts_all.csv")
    rep_offset = 0
    if os.path.exists(accepts_path):
        try:
            rep_offset = int(pd.read_csv(accepts_path, usecols=["rep_id"])["rep_id"].max()) + 1
        except Exception:
            rep_offset = 0

    observed_list = build_observed_list(bank, reps=args.reps, synthetic=args.synthetic,
                                        seed=args.seed, observed_csv=args.observed_csv)

    if args.method == "rejection":
        model = RejectionABC(bank)
        results = [model._fit_one(obs, n_accept=args.n_accept, rep_id=(rep_offset+i))
                   for i,obs in enumerate(observed_list)]

    elif args.method == "regression":
        model = RegressionABC(bank)
        results = [model._fit_one(obs, n_accept=args.n_accept, use_fraction=args.frac, rep_id=(rep_offset+i))
                   for i,obs in enumerate(observed_list)]

    elif args.method == "smc":
        model = SMCABC(bank)
        results = [model._fit_one(obs, rep_id=(rep_offset+i), T=args.T,
                                  particles=args.particles, kernel_scale=args.kernel_scale)
                   for i,obs in enumerate(observed_list)]
    else:
        raise ValueError("Unknown method")

    append_frames(args.out_dir, results, rep_offset)

    # concise terminal summary per rep
    for j,res in enumerate(results):
        pts = res["points"]
        def pick(df, prm, sp):
            row = df[(df.param==prm) & (df.space==sp)].iloc[0]
            return float(row["estimate_mode"]), (None if pd.isna(row["truth"]) else float(row["truth"]))
        la_hat, la_true = pick(pts, "alpha", "log10")
        lD_hat, lD_true = pick(pts, "D", "log10")
        r_hat,  r_true  = pick(pts, "r", "linear")
        rid = rep_offset + j
        msg = f"[{args.method} rep {rid}] log10(alpha)={la_hat:.4g}"
        if la_true is not None: msg += f" | truth {la_true:.4g}"
        msg += f"; log10(D)={lD_hat:.4g}"
        if lD_true is not None: msg += f" | truth {lD_true:.4g}"
        msg += f"; r={r_hat:.4g}"
        if r_true is not None: msg += f" | truth {r_true:.4g}"
        print(msg)

if __name__ == "__main__":
    main()
