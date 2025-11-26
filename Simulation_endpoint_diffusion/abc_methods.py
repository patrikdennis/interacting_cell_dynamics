"""
Core ABC utilities + three method classes:
- RejectionABC
- RegressionABCAjusted
- SMCABC (PMC-style SMC)

Works with bank schema:
  ['sim_id','log10_d','log10_alpha','r','N','pc_0'...'pc_{K-1}','exit_code']


Outputs mirror the current CSV structure.
"""
# abc_methods.py
# ABC methods: Rejection, Regression-adjusted (Beaumont), and SMC (bank-reuse only)

from __future__ import annotations
import math
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import dataclasses as dc

# py-abc package for abc-SMC

# ----------------------------
# Bank loading & distance
# ----------------------------

@dc.dataclass
class Bank:
    sim_id: np.ndarray
    log10_alpha: np.ndarray
    log10_D: np.ndarray
    r: np.ndarray
    N: np.ndarray
    PCF: np.ndarray        # shape (M, K)
    pc_cols: List[str]

    @property
    def M(self): return int(self.sim_id.shape[0])

    @property
    def K(self): return int(self.PCF.shape[1])

def load_bank_parquet(parquet_path: str, require_exit_code_zero: bool = True) -> Bank:
    df = pd.read_parquet(parquet_path)
    df.columns = [c.strip() for c in df.columns]
    if require_exit_code_zero and "exit_code" in df.columns:
        df = df[df["exit_code"] == 0].copy()

    required = ["sim_id","log10_d","log10_alpha","r","N"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in bank: {missing}")

    pc_cols = [c for c in df.columns if c.lower().startswith("pc_")]
    if not pc_cols:
        raise KeyError("No PCF bins (columns pc_*) found in bank.")
    def keyer(c):
        try: return int(c.split("_",1)[1])
        except: return c
    pc_cols = sorted(pc_cols, key=keyer)

    return Bank(
        sim_id=df["sim_id"].to_numpy(),
        log10_alpha=df["log10_alpha"].to_numpy(float),
        log10_D=df["log10_d"].to_numpy(float),
        r=df["r"].to_numpy(float),
        N=df["N"].to_numpy(float),
        PCF=df[pc_cols].to_numpy(float),
        pc_cols=pc_cols
    )

def abc_distance(Nw: float, PCFw: np.ndarray, Ns: np.ndarray, PCFs: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    E = sqrt( En + Epcf )
    En   = ((Nw - Ns) / (1 + Nw))^2
    Epcf = mean( ((PCFs - PCFw)/(1 + PCFw))^2, axis=1 )
    """
    En = ((Nw - Ns) / (1.0 + Nw))**2
    denom = (1.0 + PCFw)
    Epcf = np.mean(((PCFs - PCFw)/denom)**2, axis=1)
    E = np.sqrt(En + Epcf)
    return E, En, Epcf

def posterior_summary_and_mode(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    median = float(np.median(x))
    q025, q975 = [float(q) for q in np.quantile(x, [0.025,0.975])]
    counts, edges = np.histogram(x, bins="auto")
    mode_center = float(0.5*(edges[np.argmax(counts)]+edges[np.argmax(counts)+1]))
    return {"mean":mean, "median":median, "mode":mode_center, "q2.5%":q025, "q97.5%":q975, "n":int(x.size)}

# ----------------------------
# Base class
# ----------------------------

class BaseABC:
    def __init__(self, bank: Bank):
        self.bank = bank

    def _fit_one(self, observed: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    # helpers to shape outputs like your existing CSVs
    @staticmethod
    def pack_accepts(rep_id: int, idx: np.ndarray, E: np.ndarray, En: np.ndarray, Epcf: np.ndarray,
                     bank: Bank) -> pd.DataFrame:
        return pd.DataFrame({
            "rep_id": rep_id,
            "sim_id": bank.sim_id[idx],
            "E": E[idx],
            "En": En[idx],
            "Epcf": Epcf[idx],
            "log10_alpha": bank.log10_alpha[idx],
            "log10_D": bank.log10_D[idx],
            "alpha": 10.0**bank.log10_alpha[idx],
            "D":     10.0**bank.log10_D[idx],
            "r":     bank.r[idx],
            "N":     bank.N[idx],
        })

    @staticmethod
    def pack_points(rep_id: int, truth: Optional[Dict[str,float]],
                    est_log10_alpha: float, est_log10_D: float, est_r: float) -> pd.DataFrame:
        rows = []
        rows.append({"rep_id":rep_id, "param":"alpha", "space":"log10",
                     "estimate_mode": est_log10_alpha, "truth": (truth["log10_alpha"] if truth else np.nan)})
        rows.append({"rep_id":rep_id, "param":"D", "space":"log10",
                     "estimate_mode": est_log10_D, "truth": (truth["log10_D"] if truth else np.nan)})
        rows.append({"rep_id":rep_id, "param":"alpha", "space":"linear",
                     "estimate_mode": 10**est_log10_alpha, "truth": (10**truth["log10_alpha"] if truth else np.nan)})
        rows.append({"rep_id":rep_id, "param":"D", "space":"linear",
                     "estimate_mode": 10**est_log10_D, "truth": (10**truth["log10_D"] if truth else np.nan)})
        rows.append({"rep_id":rep_id, "param":"r", "space":"linear",
                     "estimate_mode": est_r, "truth": (truth["r"] if truth else np.nan)})
        return pd.DataFrame(rows)

# ----------------------------
# Rejection ABC
# ----------------------------

class RejectionABC(BaseABC):
    def _fit_one(self, observed: Dict[str,Any], n_accept: int = 250, rep_id: int = 0, **_) -> Dict[str,Any]:
        Nw  = float(observed["N"])
        PCF = np.asarray(observed["PCF"], dtype=float)
        E, En, Epcf = abc_distance(Nw, PCF, self.bank.N, self.bank.PCF)

        order = np.argsort(E)[:n_accept]
        accepts = self.pack_accepts(rep_id, order, E, En, Epcf, self.bank)

        la_mode = posterior_summary_and_mode(accepts["log10_alpha"].to_numpy())["mode"]
        lD_mode = posterior_summary_and_mode(accepts["log10_D"].to_numpy())["mode"]
        r_mode  = posterior_summary_and_mode(accepts["r"].to_numpy())["mode"]

        points = self.pack_points(rep_id, observed.get("truth"), la_mode, lD_mode, r_mode)
        return {"rep_id":rep_id, "accepts":accepts, "points":points}

# ----------------------------
# Regression-adjusted ABC (Beaumont 2002)
# ----------------------------

class RegressionABC(BaseABC):
    def _fit_one(self, observed: Dict[str,Any], n_accept: int = 250, rep_id: int = 0,
                 use_fraction: Optional[float] = None, kernel: str = "epanechnikov") -> Dict[str,Any]:

        Nw  = float(observed["N"])
        s_obs = np.asarray(observed["PCF"], dtype=float)
        E, En, Epcf = abc_distance(Nw, s_obs, self.bank.N, self.bank.PCF)

        # choose acceptance set
        k = max(5, int(round(use_fraction * self.bank.M))) if use_fraction is not None else n_accept
        idx = np.argsort(E)[:k]
        d  = E[idx]
        s  = self.bank.PCF[idx]                  # (k, K)
        s_full = np.column_stack([self.bank.N[idx][:,None], s])   # (k, K+1)
        s_obs_full = np.concatenate([[Nw], s_obs])                # (K+1,)

        # kernel weights in distance
        u = d / (d.max() + 1e-12)
        if kernel == "epanechnikov":
            w = np.maximum(0.0, 1.0 - u**2)
        elif kernel == "triangular":
            w = np.maximum(0.0, 1.0 - u)
        else:
            w = np.ones_like(u)

        def linreg_adjust(y: np.ndarray):
            X = np.column_stack([np.ones((k,1)), s_full])  # intercept + summaries
            W = np.diag(w + 1e-12)
            XtW = X.T @ W
            beta = np.linalg.pinv(XtW @ X) @ (XtW @ y)
            B = beta[1:]
            y_adj = y - (s_full - s_obs_full) @ B
            mode = posterior_summary_and_mode(y_adj)["mode"]
            return mode, y_adj

        y_la = self.bank.log10_alpha[idx]
        y_lD = self.bank.log10_D[idx]
        y_r  = self.bank.r[idx]

        la_mode, la_adj = linreg_adjust(y_la)
        lD_mode, lD_adj = linreg_adjust(y_lD)
        r_mode,  r_adj  = linreg_adjust(y_r)

        accepts = self.pack_accepts(rep_id, idx, E, En, Epcf, self.bank)
        accepts["log10_alpha_adj"] = la_adj
        accepts["log10_D_adj"]     = lD_adj
        accepts["r_adj"]           = r_adj
        accepts["alpha_adj"]       = 10.0**accepts["log10_alpha_adj"]
        accepts["D_adj"]           = 10.0**accepts["log10_D_adj"]

        points = self.pack_points(rep_id, observed.get("truth"), la_mode, lD_mode, r_mode)
        return {"rep_id":rep_id, "accepts":accepts, "points":points}

# ----------------------------
# SMC ABC (bank-reuse only; no on-the-fly)
# ----------------------------

class SMCABC(BaseABC):
    """
    Bank-reuse PMC-ABC:
      - Build epsilon schedule from distance quantiles
      - At each population, keep bank samples within epsilon, update weights via kernel in parameter space
      - Final population -> take top-weight entries as 'accepted'
    """
    def _fit_one(self, observed: Dict[str,Any],
                 rep_id: int = 0,
                 T: int = 4,
                 particles: int = 1000,
                 eps_schedule: Optional[List[float]] = None,
                 kernel_scale: float = 0.5) -> Dict[str,Any]:

        Nw  = float(observed["N"])
        s_obs = np.asarray(observed["PCF"], dtype=float)
        E_all, En_all, Epcf_all = abc_distance(Nw, s_obs, self.bank.N, self.bank.PCF)

        # epsilon schedule: quantiles over distances
        if eps_schedule is None:
            qs = [0.6, 0.35, 0.15, 0.07]
            eps_schedule = [float(np.quantile(E_all, q)) for q in qs[:T]]
            while len(eps_schedule) < T:
                eps_schedule.append(max(1e-12, eps_schedule[-1] * 0.7))

        # population 0
        idx_t = np.where(E_all <= eps_schedule[0])[0]
        if idx_t.size == 0:
            raise RuntimeError("No bank samples within initial epsilon; relax eps or check summaries.")
        w_t = np.ones(idx_t.size, dtype=float) / idx_t.size

        params_bank = np.column_stack([self.bank.log10_alpha, self.bank.log10_D, self.bank.r])

        # shrink eps and reweight
        for t in range(1, T):
            eps_t = eps_schedule[t]
            idx_next = np.where(E_all <= eps_t)[0]
            if idx_next.size == 0:
                idx_next = idx_t
            prev_theta = params_bank[idx_t]
            prev_w     = w_t
            cov = np.cov(prev_theta.T, aweights=prev_w)
            if np.ndim(cov)==0:
                cov = np.eye(3) * 1e-3
            Sigma = np.diag(np.clip(np.diag(cov), 1e-8, None)) * (kernel_scale**2)

            cur_theta = params_bank[idx_next]
            inv_d = 1.0 / np.diag(Sigma)
            norm_c = (2*math.pi)**(-1.5) * (np.prod(np.diag(Sigma))**-0.5)
            q = np.zeros(cur_theta.shape[0], dtype=float)

            CHUNK = 2000
            for start in range(0, cur_theta.shape[0], CHUNK):
                end = min(cur_theta.shape[0], start+CHUNK)
                th = cur_theta[start:end]
                diff = th[:,None,:] - prev_theta[None,:,:]
                ex = -0.5 * np.sum((diff**2) * inv_d[None,None,:], axis=2)
                q[start:end] = norm_c * np.exp(ex) @ prev_w

            w_next = 1.0 / (q + 1e-300)
            w_next /= w_next.sum()

            # keep exactly 'particles' by weighted resampling (or keep all if small)
            if idx_next.size > particles:
                draw = np.random.choice(idx_next, size=particles, replace=True, p=w_next)
                idx_t = draw
                w_t = np.ones(particles) / particles
            else:
                idx_t = idx_next
                w_t = w_next

        # final selection: top by weight (cap at 250 to match earlier summaries)
        order = np.argsort(-w_t)[:min(len(w_t), 250)]
        sel = idx_t[order]
        accepts = self.pack_accepts(rep_id, sel, E_all, En_all, Epcf_all, self.bank)

        la_mode = posterior_summary_and_mode(accepts["log10_alpha"].to_numpy())["mode"]
        lD_mode = posterior_summary_and_mode(accepts["log10_D"].to_numpy())["mode"]
        r_mode  = posterior_summary_and_mode(accepts["r"].to_numpy())["mode"]
        points = self.pack_points(rep_id, observed.get("truth"), la_mode, lD_mode, r_mode)

        return {"rep_id":rep_id, "accepts":accepts, "points":points, "extra":{"eps_schedule":eps_schedule}}
