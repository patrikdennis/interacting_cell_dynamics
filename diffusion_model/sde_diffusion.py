# interacting_diffusion_sde_birth_death.py
"""
Overdamped interacting diffusion SDE with pair interactions, repulsive walls,
and birth-death dynamics. Births create a new cell tangent to the parent:
the child center is offset by 2*cell_radius along a random direction (angle).

Model Euler-Maruyama method:
    dX_i = [ kappa * F_int(X) + F_wall(X) ] dt + sqrt(2D) dW_i

- Pair potential:
      U(r) = De * (1 - phi_d(r)^a)^2 - De,
      phi_d(r) = phi(r)/phi(d), phi(r) = exp(-r/l) ⇒ phi_d(r) = exp(-(r-d)/l)
  Force magnitude = -dU/dr, direction along displacement.

- Mobility (kappa): scales the interaction drift (kappa = 1 recovers the base SDE).
- Walls: soft repulsion from each side of a square domain [0, L]^2, acting on centers,
  plus reflecting boundary to keep centers in [cell_radius, L-cell_radius]^2.
- Birth-death process:
    - Each alive cell dies with probability p_death = death_rate * dt per step.
    - Each surviving cell attempts one birth with probability p_birth = birth_rate * dt.
      New child center = parent_center + 2*cell_radius * û, where hat(u) is a random unit vector.
      If this would place the child center outside the legal interior (i.e., closer than
      cell_radius to a wall), we retry a few random directions; if all fail, we skip that birth.
    * A maximum population cap (max_cells) prevents runaway growth.

How to use:
EXAMPLE:
    from interacting_diffusion_sde_birth_death import (
        InteractingDiffusionSDE, PotentialParams, WallParams, SDEParams, BirthDeathParams
    )

    sim = InteractingDiffusionSDE(
        N=100, L=40.0, cell_radius=0.5,
        pot=PotentialParams(De=2.1e-4, a=3.5, d=1.0, ell=1.0, cutoff=6.0),
        wall=WallParams(A=4e-3, lamb=0.7),
        sde=SDEParams(D=0.06, kappa=1.0, dt=0.01, steps=2000, seed=1),
        bd=BirthDeathParams(birth_rate=0.02, death_rate=0.01, max_cells=2000),
    )
    X_final, traj = sim.simulate(record_every=20)


Output of demo: 
    Plots and data of final cell distribution over chosen domain (in our case [0,L]^2), together with PACF (under construction see development branch) 
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


# -------------------- Parameter dataclasses --------------------


@dataclass
class PotentialParams:
    """
    Parameters of the pairwise potential U(r).

    De     : depth parameter of the potential.
    a      : shape parameter (exponent on phi_d).
    d      : preferred/interaction scale.
             phi_d(d) = 1 by construction and default.
    ell    : kernel length l in phi(r) = exp(-r/l).
    cutoff : optional interaction cutoff radius (None = infinite range).
    """
    
    De: float = 2.1e-4
    a: float = 3.5
    d: float = 1.0
    ell: float = 1.0
    cutoff: Optional[float] = None


@dataclass
class WallParams:
    """
    Soft repulsive wall potential acting inside the domain.

    A    : wall strength in U_wall(d) = A exp(-d/lambda)
    lamb : decay length lambda
    """
    A: float = 5e-3
    lamb: float = 0.5


@dataclass
class SDEParams:
    """
    SDE parameters for dX_i(t) = F_i(X_i(t))dt + sqrt(2D)dW(t) 

    D     : diffusivity in dX = ... + sqrt(2D) dW.
    kappa : mobility (inverse friction) scaling interaction drift.
    dt    : time step.
    steps : number of steps to simulate.
    seed  : seed for reproducibility (None = no seeding).
    """
    D: float = 0.05
    kappa: float = 1.0
    dt: float = 0.01
    steps: int = 800
    seed: Optional[int] = 123


@dataclass
class BirthDeathParams:
    """
    Birth-death process parameters.

    birth_rate : per-cell birth rate lambda_b (1/time).
    death_rate : per-cell death rate lamdba_d (1/time).
    max_cells  : hard cap on population to avoid runaway memory usage.
    max_birth_attempts_direction : retries to find an in-bounds direction for a newborn.
    """
    birth_rate: float = 0.02
    death_rate: float = 0.01
    max_cells: int = 5000
    max_birth_attempts_direction: int = 16

# ------------------
# --- SIMULATION ---
# ------------------ 

class InteractingDiffusionSDE:
    """
    Overdamped interacting diffusion with birth-death in a square (see readme).

    Update rule for  Euler-Maruyama:
        X_(n+1) = X_n + [kappa * F_int(X_n) + F_wall(X_n)] dt + sqrt(2D dt) * N(0, I)
    
    In code we have N(0,I) is the brownian motion W(t) from the model.

    - F_int:        sum of pairwise forces derived from U(r).
    - F_wall:       repulsion from each of the four walls.
    - Birth-death:  occurs every step with probabilities lambda_b dt and lambda_d dt, respectively.
                    After each step, positions are reflected back into the region of study (i.e. the sqaure domain).

    Complexity: O(N^2) per time step.
    """

    def __init__(
        self,
        N: int = 60,
        L: float = 20.0,
        cell_radius: float = 0.5,
        pot: PotentialParams = PotentialParams(),
        wall: WallParams = WallParams(),
        sde: SDEParams = SDEParams(),
        bd: BirthDeathParams = BirthDeathParams(),
    ) -> None:
        
        self.L = float(L)
        self.cell_radius = float(cell_radius)
        self.pot = pot
        self.wall = wall
        self.sde = sde
        self.bd = bd
        
        # check for random seed
        if sde.seed is not None:
            np.random.seed(sde.seed)

        # initialize positions uniformly at random and ensure centers are in [R, L-R]^2.
        R = self.cell_radius
        self.X = np.column_stack([
            np.random.uniform(low=R, high=self.L - R, size=N),
            np.random.uniform(low=R, high=self.L - R, size=N),
        ])

        # population history
        self.pop_history: List[int] = [N]

    # ----------------------------
    # --- Kernel and potential ---
    # ----------------------------
    
    def phi_d(self, r: np.ndarray) -> np.ndarray:
        """
        phi_d(r) = exp(-(r - d)/l), so that phi_d(d) = 1.
        """
        return np.exp(-(r - self.pot.d) / self.pot.ell)

    def U(self, r: np.ndarray) -> np.ndarray:
        """
        U(r) = De * (1 - phi_d(r)^a)^2 - De
        """
        p = self.phi_d(r) ** self.pot.a
        return self.pot.De * (1.0 - p) ** 2 - self.pot.De

    def dU_dr(self, r: np.ndarray) -> np.ndarray:
        """
        dU/dr = 2 * De * (1 - p) * (a/l) * p, where p = phi_d(r)^a and phi_d'(r) = -phi_d(r)/l.
        """
        p = self.phi_d(r) ** self.pot.a
        return 2.0 * self.pot.De * (1.0 - p) * (self.pot.a / self.pot.ell) * p

    # --------------
    # --- Forces ---
    # --------------

    def pair_forces(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise interaction forces F_int on all cells.

        X : (N, 2) positions
        returns F_int : (N, 2)
        """
        # Pairwise displacements and distances
        dx = X[:, None, :] - X[None, :, :]  # (N, N, 2)
        r2 = np.sum(dx * dx, axis=-1)       # (N, N)
        np.fill_diagonal(r2, np.inf)        # avoid self-interaction
        r = np.sqrt(r2)

        # cutoff mask
        if self.pot.cutoff is not None:
            mask = (r < self.pot.cutoff)
        else:
            mask = np.isfinite(r)

        # dU/dr on valid pairs
        dU = np.zeros_like(r)
        valid = mask & (r > 0.0) & np.isfinite(r)
        if np.any(valid):
            dU[valid] = self.dU_dr(r[valid])

        # Force magnitude is -dU/dr direction is along displacement
        fmag = -dU  # (N, N)
        F = np.zeros_like(dx)

        with np.errstate(invalid="ignore", divide="ignore"):
            unit = np.zeros_like(dx)
            unit[valid, :] = dx[valid, :] / r[valid, None]

        F[valid, :] = (fmag[valid, None] * unit[valid, :])
        F_int = np.nansum(F, axis=1)  # sum over j
        return F_int

    def wall_forces(self, X: np.ndarray) -> np.ndarray:
        """
        Soft repulsion from each wall of the square [0,L]^2.

        The force depends on the distance center-to-wall minus radius:
            dL = x - R, dR = (L - x) - R, dB = y - R, dT = (L - y) - R

        U_wall(d) = A * exp(-d/lambda)  ⇒  F = (A/lambda) * exp(-d/lambda) * n_inward
        """
        A, lam, R = self.wall.A, self.wall.lamb, self.cell_radius
        x = X[:, 0]
        y = X[:, 1]

        # Edge distances: >= 0 inside the legal region clamp below at 0 for the force formula
        dL = np.maximum(x - R, 0.0)
        dR = np.maximum((self.L - x) - R, 0.0)
        dB = np.maximum(y - R, 0.0)
        dT = np.maximum((self.L - y) - R, 0.0)

        # Magnitudes
        mL = (A / lam) * np.exp(-dL / lam)  # push right (+x)
        mR = (A / lam) * np.exp(-dR / lam)  # push left  (-x)
        mB = (A / lam) * np.exp(-dB / lam)  # push up    (+y)
        mT = (A / lam) * np.exp(-dT / lam)  # push down  (-y)

        FL = np.column_stack([ mL, 0.0 * mL])   # +x
        FR = np.column_stack([-mR, 0.0 * mR])   # -x
        FB = np.column_stack([0.0 * mB,  mB])   # +y
        FT = np.column_stack([0.0 * mT, -mT])   # -y

        return FL + FR + FB + FT

    # ---------------------------
    # --- Birth–death helpers ---
    # ---------------------------

    def _reflect_into_domain(self, X: np.ndarray) -> np.ndarray:
        """
        Reflect centers into the legal region [R, L - R]^2.
        """
        R = self.cell_radius
        
        # Reflect across the walls at x=R and x=L-R
        X[:, 0] = np.where(X[:, 0] < R, 2 * R - X[:, 0], X[:, 0])
        X[:, 0] = np.where(X[:, 0] > self.L - R, 2 * (self.L - R) - X[:, 0], X[:, 0])
        
        # Reflect across y=R and y=L-R
        X[:, 1] = np.where(X[:, 1] < R, 2 * R - X[:, 1], X[:, 1])
        X[:, 1] = np.where(X[:, 1] > self.L - R, 2 * (self.L - R) - X[:, 1], X[:, 1])
        
        return X

    def _in_bounds_center(self, pos: np.ndarray) -> bool:
        """
        Check if a proposed center is in the legal interior: [R, L - R]^2.
        """
        R = self.cell_radius
        return (R <= pos[0] <= self.L - R) and (R <= pos[1] <= self.L - R)

    def _attempt_birth_position(self, parent: np.ndarray) -> Optional[np.ndarray]:
        """
        Try multiple random directions to place a child tangent to the parent
        at distance 2*cell_radius from the parent center, while staying in bounds.
        
        We use random angle placement.
        """
        R = self.cell_radius
        for _ in range(self.bd.max_birth_attempts_direction):
            theta = 2.0 * np.pi * np.random.rand()
            disp = (2.0 * R) * np.array([np.cos(theta), np.sin(theta)])
            child = parent + disp
            if self._in_bounds_center(child):
                return child
        return None  # give up if no in-bounds direction found  --> no birth

    def _apply_deaths(self) -> None:
        """
        Remove cells that die in this step according to Beroulli(lambda_d dt).
        """
        if self.X.size == 0:
            return
        N = self.X.shape[0]
        p = self.bd.death_rate * self.sde.dt
        if p <= 0:
            return
        alive_mask = np.random.rand(N) >= p
        self.X = self.X[alive_mask, :]

    def _apply_births(self) -> None:
        """
        For each surviving parent, attempt a birth with Bernoulli(lambda_b dt),
        placing the newborn tangent to the parent with offset 2R.
        Check if max cells is reached.
        """
        if self.X.size == 0:
            return
        N_current = self.X.shape[0]
        capacity = max(self.bd.max_cells - N_current, 0)
        if capacity <= 0:
            return
        p = self.bd.birth_rate * self.sde.dt
        if p <= 0:
            return

        parents_mask = np.random.rand(N_current) < p
        parents = self.X[parents_mask, :]
        if parents.shape[0] == 0:
            return

        # Attempt birth stop if capacity is reached capacitey
        newborns: List[np.ndarray] = []
        for parent in parents:
            if len(newborns) >= capacity:
                break
            child = self._attempt_birth_position(parent)
            if child is not None:
                newborns.append(child)

        if newborns:
            self.X = np.vstack([self.X, np.vstack(newborns)])

    # ---------------------------------------------
    # --- One Euler–Maruyama step + birth–death ---
    # ---------------------------------------------

    def step(self) -> None:
        """
        Advance one time step with Euler–Maruyama, reflect, then apply birth–death.
        """
        X = self.X
        if X.size > 0:
            # diffusion + drift
            F_int = self.pair_forces(X)
            F_wall = self.wall_forces(X)
            drift = self.sde.kappa * F_int + F_wall
            noise = np.sqrt(2.0 * self.sde.D * self.sde.dt) * np.random.normal(size=X.shape)
            X = X + drift * self.sde.dt + noise

            # reflect centers into legal region
            X = self._reflect_into_domain(X)
            self.X = X

        # Apply deaths then births only survivors can give birth this step
        self._apply_deaths()
        self._apply_births()

        # Record population data
        self.pop_history.append(self.X.shape[0])

    # -------------------------
    # --- Sim funcitonality ---
    # -------------------------
    
    def simulate(self, record_every: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Method: 
                Run the simulation for sde.steps steps.
                record_every : store a distribution plot (snapshot) for specified steps.

        Returns:
            X_final : (N_T, 2) final positions where N_T = final population
            traj   : (K, N_k, 2) plots are packed as object array of shape (K,)
                      where each element is an (N_k, 2) float array.
            pop     : (steps+1,) population over time including initial
        """
        
        snapshots: List[np.ndarray] = []
        for t in range(self.sde.steps):
            self.step()
            if (t % record_every) == 0:
                snapshots.append(self.X.copy())

        traj = np.empty(len(snapshots), dtype=object)
        for i, snap in enumerate(snapshots):
            traj[i] = snap
        return self.X, traj, np.array(self.pop_history, dtype=int)


# ----------------------------------------
# --- Demo (will be moved to a main.py) ---
# ----------------------------------------

if __name__ == "__main__":

    # sim = InteractingDiffusionSDE(
    #     N=100, L=40.0, cell_radius=0.5,
    #     pot=PotentialParams(De=2.1e-4, a=3.5, d=1.0, ell=1.0, cutoff=6.0),
    #     wall=WallParams(A=4e-3, lamb=0.7),
    #     sde=SDEParams(D=0.00, kappa=1.0, dt=0.01, steps=15000, seed=7),
    #     bd=BirthDeathParams(birth_rate=0.02, death_rate=0.01, max_cells=4000),
    # )

    # SANITY CHECK: NO DIFFUSION --> expect clustering (see E.Rosén)
    sim = InteractingDiffusionSDE(
    N=100, L=40.0, cell_radius=0.5,
    pot=PotentialParams(De=0.03, a=3.5, d=0.5, ell=1.0, cutoff=3.0),
    wall=WallParams(A=4e-3, lamb=0.7),
    sde=SDEParams(D=0.0, kappa=25.0, dt=0.02, steps=1500, seed=7),
    bd=BirthDeathParams(birth_rate=0.03, death_rate=0.005, max_cells=4000),
)
    
    X_final, traj, pop = sim.simulate(record_every=20)

    # final configuration/distribution
    plt.figure(figsize=(5.8, 5.8))
    if X_final.size:
        plt.scatter(X_final[:, 0], X_final[:, 1], s=10)
    plt.xlim(sim.cell_radius, sim.L - sim.cell_radius)
    plt.ylim(sim.cell_radius, sim.L - sim.cell_radius)
    plt.gca().set_aspect("equal", "box")
    plt.title(f"Final positions (N={X_final.shape[0]})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, ls="--", alpha=0.35)
    plt.tight_layout()

    # population over time
    plt.figure(figsize=(6.6, 3.6))
    plt.plot(pop, lw=1.5)
    plt.xlabel("step")
    plt.ylabel("population")
    plt.title("Population (birth-death)")
    plt.grid(True, ls="--", alpha=0.35)
    plt.tight_layout()

    plt.show()
