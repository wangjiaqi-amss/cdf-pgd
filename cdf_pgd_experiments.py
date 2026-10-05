"""
Numerical experiments for the exact smooth penalty method (CDF-PGD).

Author: Jiaqi Wang (LSEC, AMSS), Nachuan Xiao (SDS, CUHK-SZ), Xin Liu (LSEC, AMSS)

This is a reproduction script for the numerical experiments section. It implements

  * LMSVD, used for every large-scale singular value decomposition;
  * the four application problems (spherical data fitting, graph similarity,
    international migration stock estimation, image denoising) together with
    their analytic gradients and constraint-dissolving maps;
  * the convergence metrics (stationarity residual and constraint violation);
  * CDF-PGD and the comparison solvers SD-RGD, SD-RTR, ALM-PGD, and the
    factorised equality formulations EQ1/EQ2 solved by SLSQP and trust-constr;
  * the figure generation for the convergence triptychs and the denoising
    panel.

All tunable parameters live in the CONFIGURATION section below.  Every solve
runs single threaded; timed solves run serially in separate worker processes.
A watchdog enforces the stated wall-clock budgets.

Usage
-----
    python cdf_pgd_experiments.py --exp sphere
    python cdf_pgd_experiments.py --exp graph
    python cdf_pgd_experiments.py --exp eq-sphere
    python cdf_pgd_experiments.py --exp eq-graph
    python cdf_pgd_experiments.py --exp migration
    python cdf_pgd_experiments.py --exp denoise
    python cdf_pgd_experiments.py --exp figures
    python cdf_pgd_experiments.py --exp all

Results are accumulated in ``results/results.json``; figures are written to
``figures/``.  Re-running a single experiment only overwrites the entries it
owns, so the remaining entries are preserved.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Single-threaded BLAS.  This must happen before numpy is imported, in the
# parent process and in every worker process spawned by the task pool.
# --------------------------------------------------------------------------
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "OMP_DYNAMIC",
    "MKL_DYNAMIC",
):
    os.environ[_var] = "1"

import argparse
import json
import time
import multiprocessing as mp
import platform
import sys
import traceback
import hashlib
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# ==========================================================================
# CONFIGURATION
# ==========================================================================
# --- paths ----------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
MIGRATION_XLSX = os.path.join(ROOT, "data", "undesa_2024_dest_origin.xlsx")
RESULTS_JSON = os.path.join(ROOT, "results", "results.json")
FIGURE_DIR = os.path.join(ROOT, "figures")

# --- task scheduling ------------------------------------------------------
NPROC = 1  # timed solves run serially

# --- common stopping test (spherical fitting and graph similarity) ---------
RES_TOL = 1e-4  # stationarity residual tolerance
VIO_TOL = 1e-6  # equality violation tolerance
MAXITER = 300  # iteration budget per solve
MAXTIME = 120.0  # wall-clock budget per solve, seconds
MAXTIME_LARGE = 600.0  # wall-clock budget for the appended large instance

# --- CDF-PGD --------------------------------------------------------------
BETA = 1.0  # penalty parameter for spherical fitting / graphs
ALPHA0 = 0.1  # initial step size
ALPHA_LO = 1e-8  # BB safeguard, lower end
ALPHA_HI = 1e2  # BB safeguard, upper end
DELTA = 0.8  # backtracking factor
ARMIJO = 1e-4  # Armijo constant
MEMORY = 10  # nonmonotone memory
ETA0 = 1e9  # initial acceleration threshold
ETA_DECAY = 0.9  # acceleration threshold decay factor

# --- space-decoupling baselines -------------------------------------------
OMEGA_RGD = 0.5  # metric parameter, SD-RGD
OMEGA_RTR = 10.0  # metric parameter, SD-RTR
TRUST_RADIUS0 = 0.1  # initial trust-region radius
TRUST_THETA = float(np.sqrt(2.0) - 1.2)  # truncated-CG stopping parameter
TRUST_KAPPA = 0.1
TRUST_RHO_PRIME = 0.1
TRUST_MAXINNER = 100

# --- augmented Lagrangian -------------------------------------------------
ALM_RHO = 1.0
ALM_EPS0 = 1e-1
ALM_EPS_MIN = 1e-8
ALM_EPS_DECAY = 0.1
ALM_INNER_MAX = 30

# --- factorised equality formulations -------------------------------------
EQ_MAXITER = 1000
EQ_MAXTIME = 60.0
EQ_TOL = 1e-10  # native SciPy tolerance

# --- LMSVD ----------------------------------------------------------------
LMSVD_M = 5  # memory parameter (number of extra Krylov blocks)
LMSVD_TOL = 1e-10  # relative residual tolerance
LMSVD_MAXIT = 1000

# --- synthetic instances --------------------------------------------------
SPHERE_SIZES = [(500, 500, 10), (1000, 1000, 25), (1500, 1500, 40), (2000, 2000, 50)]
SPHERE_LARGE = (5000, 5000, 50)
EQ_SIZES = [(50, 50, 5), (100, 100, 10), (200, 200, 20)]
REPETITIONS = 1  # one fixed-seed run per instance/method

SEED_DATA = 42  # seed of the generated data
SEED_INIT = 1000  # fixed shared initialization seed
NOISE_NU = 0.01  # correlated noise level, spherical fitting
NOISE_D = 3  # geometric decay exponent of the signal
GRAPH_AVG_DEG = 6.0  # expected out-degree of the directed Erdos-Renyi graphs
GRAPH_VAL_DEGREES = (3.0, 6.0)
GRAPH_VAL_SIZES = (500, 1000)
GRAPH_VAL_RANKS = (5, 10)

# --- migration stock estimation -------------------------------------------
MIGRATION_RANKS = (3, 5, 10)
MIGRATION_LAMBDA = 10.0
MIGRATION_BETA = 1.0
MIGRATION_STEPS = 5000  # maximum accepted PGD steps
MIGRATION_CDF_TOL = RES_TOL  # Algorithm 2 penalty-stationarity tolerance
MIGRATION_HOLDOUT = 0.10
MIGRATION_SEED = 2024
ITERATIVE_SVD_TOL = 1e-5
SOFT_IMPUTE_TOL = 1e-3
SOFT_IMPUTE_SHRINK = 50.0  # threshold is sigma_max / this value

# --- image denoising ------------------------------------------------------
DENOISE_RANK = 50
DENOISE_NU = 50.0  # noise standard deviation on the [0, 255] scale
DENOISE_SEED = 7
DENOISE_SIGMA = 0.1  # regularisation of the dissolving map
DENOISE_BETA = 1e4
DENOISE_STEPS = 2000  # the iteration is not converged at 500 steps
NLM_PATCH = 7
NLM_PATCH_DISTANCE = 11
NLM_FILTER = 0.8  # filtering parameter, as a multiple of nu/255

# Corrupted regions, as [row, column] of the top-left corner of a square.
# The reference work does not state the placement; these are the centred
# positions, which reproduce the reported NLM/BM3D values (those two baselines
# depend only on the region and the noise, not on the solver).
DENOISE_REGIONS = {
    "small": [(108, 108)],
    "large": [(88, 88)],
    "multiple": [(44, 44), (44, 172), (108, 108), (172, 44), (172, 172)],
}
DENOISE_SIDE = {"small": 40, "large": 80, "multiple": 40}

# --- figure styling -------------------------------------------------------
FIGURE_LINEWIDTH = 3.5
FIGURE_DPI = 170
METHOD_STYLE = {
    "CDF-PGD": dict(color="#1f77b4", ls="-"),
    "SD-RGD": dict(color="#d62728", ls="--"),
    "SD-RTR": dict(color="#2ca02c", ls="-."),
    "ALM-PGD": dict(color="#9467bd", ls=":"),
}
MAIN_METHODS = ["CDF-PGD", "SD-RGD", "SD-RTR", "ALM-PGD"]


class LMSVDConvergenceError(np.linalg.LinAlgError):
    """The requested SVD accuracy was not reached within its internal cap."""


LMSVD_DIAGNOSTICS = {
    "calls": 0,
    "nonconverged": 0,
    "max_relative_residual": 0.0,
    "iterations": 0,
    "max_iterations": 0,
}


def lmsvd(A, k, M=LMSVD_M, tol=LMSVD_TOL, maxit=LMSVD_MAXIT, Y0=None, seed=0):
    """Return verified leading singular triplets ``(U, s, V)`` using LMSVD.

    The subspace contains the current simultaneous-iteration block and up to
    M previous blocks. Pivoted QR identifies its numerical rank, and a small
    projected SVD solves its Rayleigh--Ritz problem without forming a Gram
    matrix. The basis image is evaluated afresh each iteration. The total
    internal iteration cap is maxit; the enclosing solve's deadline also
    applies. Unconverged or nonfinite factors are never returned.
    """
    from scipy.linalg import qr

    _budget_check()
    m, n = A.shape
    if (
        int(k) != k
        or int(maxit) != maxit
        or int(M) != M
        or k <= 0
        or maxit <= 0
        or M < 0
        or not np.isfinite(tol)
        or tol <= 0
    ):
        raise ValueError(
            "k and maxit must be positive integers, M nonnegative, tol positive"
        )
    r = min(int(k), m, n)
    if r <= 0:
        raise ValueError("A must have nonzero dimensions")
    block = min(2 * r, r + 10, m, n)
    rng = np.random.default_rng(seed)
    if Y0 is None:
        start = rng.standard_normal((n, block))
    else:
        warm = np.asarray(Y0, dtype=float)
        if warm.ndim != 2 or warm.shape[0] != n or not np.isfinite(warm).all():
            raise ValueError("Y0 must be a finite matrix with A.shape[1] rows")
        # Independent guard vectors supplement the requested warm subspace.
        warm = warm[:, : min(r, warm.shape[1])]
        start = np.column_stack((warm, rng.standard_normal((n, block - warm.shape[1]))))
    start = np.linalg.qr(start, mode="reduced")[0]
    X = np.linalg.qr(np.asarray(A @ start), mode="reduced")[0]
    history = []
    eps = np.finfo(float).eps
    relative = np.inf
    LMSVD_DIAGNOSTICS["calls"] += 1

    for iteration in range(1, int(maxit) + 1):
        _budget_check()
        Q = X
        if history and X.shape[1] < m:
            H = np.column_stack(history)
            P = H - X @ (X.T @ H)
            P -= X @ (X.T @ P)
            Z, T, _ = qr(P, mode="economic", pivoting=True, check_finite=True)
            diagonal = np.abs(np.diag(T))
            # Every retained block has orthonormal columns, so sqrt(len(history))
            # bounds the history matrix's operator norm. This cutoff discards
            # directions below its floating-point backward-error scale.
            cutoff = eps * max(P.shape) * max(1.0, np.sqrt(len(history)))
            keep = min(int(np.count_nonzero(diagonal > cutoff)), m - X.shape[1])
            if keep:
                Z = Z[:, :keep]
                Z -= X @ (X.T @ Z)
                Z = np.linalg.qr(Z, mode="reduced")[0]
                Q = np.column_stack((X, Z))
        _budget_check()
        image = np.asarray(A.T @ Q)
        if not np.isfinite(image).all():
            LMSVD_DIAGNOSTICS["nonconverged"] += 1
            raise LMSVDConvergenceError("Nonfinite LMSVD image; no projection returned")
        P, T = np.linalg.qr(image, mode="reduced")
        left_small, values, right_small_t = np.linalg.svd(T, full_matrices=False)
        count = min(block, values.size)
        Xhat = Q @ right_small_t[:count].T
        right = P @ left_small[:, :count]
        values = values[:count]
        U, s, V = Xhat[:, :r], values[:r], right[:, :r]
        _budget_check()
        next_image = np.asarray(A @ right)
        residual_left = next_image[:, :r] - U * s
        residual_right = np.asarray(A.T @ U) - V * s
        left_norm = float(np.linalg.norm(residual_left, axis=0).max(initial=0.0))
        right_norm = float(np.linalg.norm(residual_right, axis=0).max(initial=0.0))
        finite = (
            np.isfinite(left_norm)
            and np.isfinite(right_norm)
            and np.isfinite(U).all()
            and np.isfinite(s).all()
            and np.isfinite(V).all()
        )
        residual = max(left_norm, right_norm) if finite else np.inf
        relative = residual / float(s[0]) if s.size and s[0] > 0 else residual
        lmsvd.last_info = dict(
            iterations=iteration,
            converged=False,
            relative_residual=relative,
            requested_rank=r,
            working_rank=block,
        )
        LMSVD_DIAGNOSTICS["iterations"] += 1
        LMSVD_DIAGNOSTICS["max_iterations"] = max(
            LMSVD_DIAGNOSTICS["max_iterations"], iteration
        )
        if not np.isfinite(relative):
            LMSVD_DIAGNOSTICS["nonconverged"] += 1
            raise LMSVDConvergenceError(
                "Nonfinite LMSVD residual; no projection returned"
            )
        if relative <= tol:
            lmsvd.last_info["converged"] = True
            LMSVD_DIAGNOSTICS["max_relative_residual"] = max(
                LMSVD_DIAGNOSTICS["max_relative_residual"], relative
            )
            _budget_check()
            return U, s, V
        history.insert(0, X.copy())
        del history[int(M) :]
        # Scaling each nonzero image column before simultaneous iteration
        # preserves its exact span and avoids squaring its singular scale.
        X = np.linalg.qr(next_image, mode="reduced")[0]

    LMSVD_DIAGNOSTICS["nonconverged"] += 1
    raise LMSVDConvergenceError(
        f"LMSVD reached {maxit} iterations: relative residual "
        f"{relative:.3e} > {tol:.3e}; no projection was returned"
    )


lmsvd.last_info = None


def rank_projection(X, r, Y0=None, tol=LMSVD_TOL, maxit=LMSVD_MAXIT):
    """Numerical rank-r projection and its thin singular factors."""
    U, s, V = lmsvd(X, r, tol=tol, maxit=maxit, Y0=Y0)
    return (U * s) @ V.T, U, s, V


def spectral_norm(A, tol=1e-12, maxit=1000):
    """Leading singular value, evaluated by LMSVD."""
    return float(lmsvd(A, 1, tol=tol, maxit=maxit)[1][0])


def polar(Z):
    """Thin orthonormal polar factor, evaluated with LMSVD."""
    U, _, V = lmsvd(Z, min(Z.shape))
    return U @ V.T


def numerical_rank(X):
    """Numerical matrix rank with NumPy's default relative cutoff, using LMSVD."""
    _, s, _ = lmsvd(X, min(X.shape))
    threshold = max(X.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    return int(np.count_nonzero(s > threshold))


class BaseProblem:
    """Interface shared by the four applications.

    ``dA`` is the directional derivative, while ``adjoint_dA`` is its
    Frobenius adjoint. The two coincide only for the sphere and oblique maps.
    ``mH_proj``/``mH_projH`` describe the space-decoupling geometry.
    """

    name = "base"

    def __init__(self, n, p, r):
        self.n, self.p, self.r = int(n), int(p), int(r)

    def f(self, X):
        raise NotImplementedError

    def gradf(self, X):
        raise NotImplementedError

    def hessf(self, X, D):
        raise NotImplementedError

    def hvec(self, X):
        raise NotImplementedError

    def gradh(self, X, lam):
        raise NotImplementedError

    def hnorm(self, X):
        return float(np.linalg.norm(self.hvec(X)))

    def A(self, X):
        raise NotImplementedError

    def dA(self, X, V):
        raise NotImplementedError

    def adjoint_dA(self, X, G):
        raise NotImplementedError

    def projC(self, X):
        raise NotImplementedError

    def proj_TC(self, X, D):
        raise NotImplementedError

    def mH_proj(self, H, U):
        raise NotImplementedError

    def mH_projH(self, Y):
        raise NotImplementedError

    def mH_curv(self, X, E, H, Z, K):
        raise NotImplementedError

    def accept_dissolving(self, XA):
        return True


class SphericalFitting(BaseProblem):
    """min ||X - Adata||_F^2, s.t. rank(X) <= r, diag(X X^T) = 1_n."""

    name = "spherical"

    def __init__(self, Adata, r):
        super().__init__(*Adata.shape, r)
        self.Adata = np.asarray(Adata, dtype=float)

    def f(self, X):
        return float(np.sum((X - self.Adata) ** 2))

    def gradf(self, X):
        return 2.0 * (X - self.Adata)

    def hessf(self, X, D):
        return 2.0 * D

    def hvec(self, X):
        return np.sum(X * X, axis=1) - 1.0

    def gradh(self, X, lam):
        return 2.0 * lam[:, None] * X

    def A(self, X):
        d = np.sum(X * X, axis=1) - 1.0
        return X - 0.5 * d[:, None] * X

    def dA(self, X, V):
        d = np.sum(X * X, axis=1) - 1.0
        dd = 2.0 * np.sum(X * V, axis=1)
        return V - 0.5 * (dd[:, None] * X + d[:, None] * V)

    def adjoint_dA(self, X, G):
        return self.dA(X, G)

    def projC(self, X):
        rn = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(rn <= 1e-300, 1.0, rn)

    def proj_TC(self, X, D):
        d = np.sum(X * X, axis=1)
        d = np.where(d <= 1e-300, 1.0, d)
        return D - (np.sum(D * X, axis=1) / d)[:, None] * X

    def mH_proj(self, H, U):
        return U - np.sum(H * U, axis=1)[:, None] * H

    def mH_projH(self, Y):
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        return Y / np.where(rn <= 1e-300, 1.0, rn)

    def mH_curv(self, X, E, H, Z, K):
        return np.sum(H * Z, axis=1)[:, None] * K


class GraphSimilarity(BaseProblem):
    """min -tr(X^T L(L(X)))  s.t.  rank(X) <= r, ||X||_F^2 = 1."""

    name = "graph"

    def __init__(self, AdjA, AdjB, r, nA=None, nB=None):
        super().__init__(AdjA.shape[0], AdjB.shape[0], r)
        self.Aop, self.Bop = AdjA, AdjB
        self.nA = spectral_norm(AdjA) if nA is None else float(nA)
        self.nB = spectral_norm(AdjB) if nB is None else float(nB)
        self.scale = 2.0 * self.nA * self.nB

    def L(self, X):
        return (self.Aop @ X @ self.Bop.T + self.Aop.T @ X @ self.Bop) / self.scale

    def L2(self, X):
        return self.L(self.L(X))

    def f(self, X):
        return -float(np.sum(X * self.L2(X)))

    def gradf(self, X):
        return -2.0 * self.L2(X)

    def hessf(self, X, D):
        return -2.0 * self.L2(D)

    def hvec(self, X):
        return np.array([float(np.sum(X * X)) - 1.0])

    def gradh(self, X, lam):
        return 2.0 * lam[0] * X

    def A(self, X):
        return (1.5 - 0.5 * float(np.sum(X * X))) * X

    def dA(self, X, V):
        return (1.5 - 0.5 * float(np.sum(X * X))) * V - float(np.sum(X * V)) * X

    def adjoint_dA(self, X, G):
        return self.dA(X, G)

    def projC(self, X):
        return X / max(float(np.linalg.norm(X)), 1e-300)

    def proj_TC(self, X, D):
        nx2 = float(np.sum(X * X))
        return D - (float(np.sum(X * D)) / max(nx2, 1e-300)) * X

    def mH_proj(self, H, U):
        nh2 = float(np.sum(H * H))
        return U - (float(np.sum(H * U)) / max(nh2, 1e-300)) * H

    def mH_projH(self, Y):
        return Y / max(float(np.linalg.norm(Y)), 1e-300)

    def mH_curv(self, X, E, H, Z, K):
        return float(np.sum(E * X)) * K

    def accept_dissolving(self, XA):
        nrm2 = float(np.sum(XA * XA))
        return 0.5 <= nrm2 <= 1.5


class Migration(BaseProblem):
    """min 1/2||M*(X-R)||_F^2 + lam/2||(1-M)*(X-F)||_F^2, s.t. rank<=r, X 1_p = b."""

    name = "migration"

    def __init__(self, R, mask, F, b, r, lam=MIGRATION_LAMBDA):
        super().__init__(R.shape[0], R.shape[1], r)
        self.R = np.asarray(R, dtype=float)
        self.mask = np.asarray(mask, dtype=float)
        self.F = np.asarray(F, dtype=float)
        self.b = np.asarray(b, dtype=float).reshape(-1)
        self.lam = float(lam)
        self.ones_p = np.ones((self.p, 1))
        self.b2 = float(self.b @ self.b)
        if self.b2 == 0.0:
            raise ValueError("The row-sum dissolving map requires nonzero b.")

    def f(self, X):
        res = self.mask * (X - self.R)
        pri = (1.0 - self.mask) * (X - self.F)
        return 0.5 * float(np.sum(res * res)) + 0.5 * self.lam * float(
            np.sum(pri * pri)
        )

    def gradf(self, X):
        return self.mask * (X - self.R) + self.lam * (1.0 - self.mask) * (X - self.F)

    def hessf(self, X, D):
        return (self.mask + self.lam * (1.0 - self.mask)) * D

    def hvec(self, X):
        return (X @ self.ones_p).ravel() - self.b

    def gradh(self, X, lam):
        return lam[:, None] @ self.ones_p.T

    def A(self, X):
        s = X @ self.ones_p
        return X - ((s - self.b[:, None]) @ (self.b[None, :] @ X)) / self.b2

    def dA(self, X, V):
        s = X @ self.ones_p
        return (
            V
            - (
                (V @ self.ones_p) @ (self.b[None, :] @ X)
                + (s - self.b[:, None]) @ (self.b[None, :] @ V)
            )
            / self.b2
        )

    def adjoint_dA(self, X, G):
        h = self.hvec(X)
        bX = self.b @ X
        return (
            G
            - ((G @ bX)[:, None] @ self.ones_p.T + self.b[:, None] @ (h @ G)[None, :])
            / self.b2
        )

    def projC(self, X):
        return X - ((X @ self.ones_p - self.b[:, None]) @ self.ones_p.T) / self.p

    def proj_TC(self, X, D):
        return D - ((D @ self.ones_p) @ self.ones_p.T) / self.p


class ImageDenoising(BaseProblem):
    """min 1/2||X - Icor||_F^2  s.t.  rank(X) <= r, X_ij = Iclean_ij on Omega.

    The constraint-dissolving map is the regularised one,

        Q(X)[lam] = unvec_M(lam) (X^T X + sigma ||h(X)|| I_p)^{-1} X^T,

    A thin spectral evaluation avoids forming a nearly singular inverse.
    All calls are at matrices of rank at most r. The numerical floor also
    enters the derivative. The paper's unfloored map is undefined at a
    rank-deficient feasible matrix and is not globally smooth there.
    """

    name = "denoise"

    def __init__(self, Icor, Iclean, mask, r, sigma=DENOISE_SIGMA):
        super().__init__(*Icor.shape, r)
        self.Icor = np.asarray(Icor, dtype=float)
        self.Iclean = np.asarray(Iclean, dtype=float)
        self.mask = np.asarray(mask, dtype=float)
        self.known = self.mask > 0.5
        self.sigma = float(sigma)
        self._spectral_cache = None

    def f(self, X):
        return 0.5 * float(np.sum((X - self.Icor) ** 2))

    def gradf(self, X):
        return X - self.Icor

    def hessf(self, X, D):
        return D

    def _residual(self, X):
        return self.mask * (X - self.Iclean)

    def hvec(self, X):
        return (X - self.Iclean)[self.known]

    def gradh(self, X, lam):
        G = np.zeros_like(X)
        G[self.known] = lam
        return G

    def _tau_floor(self, X):
        return 1e-18 * max(float(np.sum(X * X)), 1.0)

    def _spectral_state(self, X):
        cache = self._spectral_cache
        if cache is not None and np.array_equal(cache[0], X):
            return cache[1:]
        Rm = self._residual(X)
        hn = float(np.linalg.norm(Rm))
        floor = self._tau_floor(X)
        if self.sigma * hn > floor:
            tau = self.sigma * hn
            grad_tau = (self.sigma / hn) * Rm
        else:
            tau = floor
            grad_tau = 2e-18 * X if float(np.sum(X * X)) > 1.0 else np.zeros_like(X)
        U, s, V = lmsvd(X, min(self.r, self.n, self.p), tol=LMSVD_TOL)
        denom = s * s + tau
        q = s * s / denom
        W = (U * (s / denom)) @ V.T  # X (X^T X + tau I)^{-1}
        b = q / denom
        self._spectral_cache = (X.copy(), Rm, V, q, W, b, grad_tau)
        return self._spectral_cache[1:]

    def A(self, X):
        Rm, V, q, _, _, _ = self._spectral_state(X)
        return X - ((Rm @ V) * q) @ V.T

    def dA(self, X, D):
        Rm, V, q, W, b, grad_tau = self._spectral_state(X)
        DT = D - ((D @ V) * q) @ V.T
        dS = W.T @ DT + DT.T @ W
        dS -= float(np.sum(grad_tau * D)) * ((V * b) @ V.T)
        return D - (((self.mask * D) @ V) * q) @ V.T - Rm @ dS

    def adjoint_dA(self, X, G):
        Rm, V, q, W, b, grad_tau = self._spectral_state(X)
        GS = ((G @ V) * q) @ V.T
        Z = W @ (Rm.T @ G + G.T @ Rm)
        ZT = Z - ((Z @ V) * q) @ V.T
        tau_weight = float(np.sum((G @ V) * (Rm @ V) * b))
        return G - self.mask * GS - ZT + tau_weight * grad_tau

    def projC(self, X):
        return np.where(self.mask > 0, self.Iclean, X)

    def proj_TC(self, X, D):
        return (1.0 - self.mask) * D


# ==========================================================================
# 3. Convergence metrics
# ==========================================================================
def vio(prob, X):
    """Equality violation at the actual iterate, without normalization."""
    return prob.hnorm(X)


def _tangent_rank_norm(U, V, G):
    """Norm of the rank-r tangent projection, using orthogonal thin blocks."""
    UtG = U.T @ G
    GV = G @ V
    GV -= U @ (UtG @ V)
    return float(np.sqrt(np.sum(UtG * UtG) + np.sum(GV * GV)))


def res(prob, X, rank_tol=1e-9):
    """dist(-grad f(X), Nhat_rank(X) + range(grad h(X))) at raw X.

    This residual is used only for sphere and graph comparisons. Their
    equality normals are orthogonal to the rank normal space, even away
    from equality feasibility. At rank below r, the regular rank normal
    cone is {0}. No equality projection or rank truncation is applied to X.
    """
    if prob.name not in ("spherical", "graph"):
        raise ValueError(
            "The common stationarity test is defined for sphere and graph."
        )
    U, s, V = lmsvd(X, min(prob.r + 1, prob.n, prob.p), tol=LMSVD_TOL)
    cutoff = rank_tol * max(float(s[0]) if s.size else 0.0, 1e-300)
    rank = int(np.count_nonzero(s > cutoff))
    prob._metric_rank = rank
    E = prob.gradf(X)
    prob._metric_fun = (
        0.5 * float(np.sum(E * X))
        if prob.name == "graph"
        else 0.25 * float(np.sum(E * E))
    )
    if rank > prob.r:
        return float("inf")
    E = prob.proj_TC(X, E)
    if rank < prob.r:
        return float(np.linalg.norm(E))
    return _tangent_rank_norm(U[:, : prob.r], V[:, : prob.r], E)


def converged(prob, X):
    return vio(prob, X) <= VIO_TOL and res(prob, X) <= RES_TOL


RESUME = False
_WORKER_PIPE = None
_BUDGET_START = None
_BUDGET_SECONDS = None


class _BudgetExpired(Exception):
    """The prescribed solve budget has expired."""


def _budget_check():
    if (
        _BUDGET_START is not None
        and _BUDGET_SECONDS is not None
        and time.perf_counter() - _BUDGET_START >= _BUDGET_SECONDS
    ):
        raise _BudgetExpired


def _send_progress(message):
    if _WORKER_PIPE is not None:
        _WORKER_PIPE.send(message)


def _start_budget(prob, X0, seconds, eq=False):
    global _BUDGET_START, _BUDGET_SECONDS
    if eq:
        _send_progress(("eq_setup", prob, np.asarray(X0)))
    _BUDGET_SECONDS = float(seconds)
    _BUDGET_START = time.perf_counter()
    _send_progress(("start", _BUDGET_START, _BUDGET_SECONDS))


class _Run:
    """Records only completed iterates, including their common diagnostics."""

    def __init__(self, prob):
        self.prob = prob
        self.t0 = _BUDGET_START if _BUDGET_START is not None else time.perf_counter()
        self.k, self.hist, self.last = 0, [], None

    @property
    def elapsed(self):
        return time.perf_counter() - self.t0

    def record(self, X):
        _budget_check()
        rr, vv = res(self.prob, X), vio(self.prob, X)
        _budget_check()
        elapsed = self.elapsed
        self.hist.append((elapsed, rr, vv, self.k))
        self.last = dict(
            iters=self.k,
            time=elapsed,
            endpoint_time=elapsed,
            res=rr,
            vio=vv,
            hist=list(self.hist),
            fun=self.prob._metric_fun,
            rank=self.prob._metric_rank,
        )
        _send_progress(("snapshot", self.last))
        return rr, vv

    def stop_status(self, rr, vv, maxiter, maxtime):
        _budget_check()
        if rr <= RES_TOL and vv <= VIO_TOL:
            return "C"
        if self.elapsed >= maxtime:
            return "T"
        if self.k >= maxiter:
            return "I"
        return None

    def finish(self, X, status):
        _budget_check()
        out = dict(self.last)
        out.update(X=X, status=status, time=self.elapsed)
        return out


def _task_key(task, fn):
    if fn in (_sphere_task, _graph_task):
        kind, n, p, r, rep, method = task[:6]
        return f"{kind}|{n}x{p}x{r}|rep{rep}|{method}"
    if fn is _eq_cdf_task:
        kind, n, p, r = task
        return f"{kind}|{n}x{p}x{r}|CDF-PGD"
    kind, n, p, r, eq2, solver = task[:6]
    name = f"EQ{2 if eq2 else 1}-{'TC' if solver == 'trust-constr' else 'SLSQP'}"
    return f"{kind}|{n}x{p}x{r}|{name}"


def _timed_worker(conn, task, fn):
    global _WORKER_PIPE
    _WORKER_PIPE = conn
    try:
        conn.send(("done", fn(task)))
    except _BudgetExpired:
        conn.send(("expired",))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


def _hard_limited_task(task, fn):
    """Enforce wall time outside the worker, even during a native LAPACK call.

    Data construction and worker startup precede the budget. At a cutoff the
    last completed, diagnosed iterate is retained; incomplete work is discarded.
    EQ diagnostics follow the native solve and are excluded from solve time.
    """
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_timed_worker, args=(child, task, fn))
    proc.start()
    child.close()
    deadline = limit = started = snapshot = result = error = None
    eq_prob = eq_X = None
    eq_iters, endpoint_time = 0, 0.0
    while True:
        remaining = (
            1.0 if deadline is None else max(0.0, deadline - time.perf_counter())
        )
        if deadline is not None and remaining <= 0.0:
            break
        if parent.poll(min(remaining, 1.0)):
            try:
                msg = parent.recv()
            except EOFError:
                break
            if msg[0] == "start":
                started, limit = msg[1:]
                deadline = started + limit
            elif msg[0] == "eq_setup":
                eq_prob, eq_X = msg[1:]
            elif msg[0] == "eq_snapshot":
                eq_X, eq_iters, endpoint_time = msg[1:]
            elif msg[0] == "native_finished":
                deadline = None
            elif msg[0] == "snapshot":
                snapshot = msg[1]
            elif msg[0] == "done":
                result = msg[1]
                break
            elif msg[0] == "error":
                error = msg[1]
                break
            elif msg[0] == "expired":
                break
        elif not proc.is_alive():
            break
    if proc.is_alive():
        proc.terminate()
    proc.join()
    parent.close()
    if result is not None:
        return result
    if eq_prob is not None:
        try:
            rr, vv, ff = res(eq_prob, eq_X), vio(eq_prob, eq_X), eq_prob.f(eq_X)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            rr, vv, ff = float("inf"), float("inf"), float("nan")
        snapshot = dict(
            iters=eq_iters, res=rr, vio=vv, fun=ff, endpoint_time=endpoint_time
        )
    if snapshot is None:
        snapshot = dict(
            iters=0, res=float("inf"), vio=float("inf"), endpoint_time=0.0, hist=[]
        )
    snapshot = dict(snapshot)
    used = time.perf_counter() - started if error and started else (limit or 0.0)
    snapshot.update(status="F" if error else "T", time=min(used, limit or used))
    if error:
        snapshot["error"] = error
    return dict(key=_task_key(task, fn), res=snapshot)


def write_environment():
    from threadpoolctl import threadpool_info
    from importlib.metadata import version, PackageNotFoundError

    packages = {}
    for name in (
        "numpy",
        "scipy",
        "pandas",
        "openpyxl",
        "matplotlib",
        "scikit-image",
        "bm3d",
        "bm4d",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    configuration = {}
    for name, value in globals().items():
        if (
            not name.isupper()
            or name.startswith("_")
            or name
            in {
                "ROOT",
                "MIGRATION_XLSX",
                "RESULTS_JSON",
                "FIGURE_DIR",
                "RESUME",
                "EXPERIMENTS",
            }
        ):
            continue
        try:
            json.dumps(value)
            configuration[name] = value
        except TypeError:
            continue
    path = Path(ROOT, "results", "environment.json")
    if RESUME and Path(RESULTS_JSON).exists() and path.exists():
        if json.loads(path.read_text()).get("source_sha256") != source_hash:
            raise RuntimeError(
                "Source differs from saved run; resuming would mix settings."
            )
    record = dict(
        source_sha256=source_hash,
        configuration=configuration,
        python=sys.version,
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        packages=packages,
        threadpools=threadpool_info(),
        timed_concurrency=1,
        repetition_count=1,
    )
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(record, indent=2))


def cdf_value(prob, X, beta):
    return prob.f(prob.A(X)) + 0.5 * beta * prob.hnorm(X) ** 2


def cdf_grad(prob, X, beta):
    """Chain rule with the adjoint derivative of the dissolving map."""
    G = prob.adjoint_dA(X, prob.gradf(prob.A(X)))
    hv = prob.hvec(X)
    return G + beta * prob.gradh(X, hv)


def _rank_cone_grad_norm(U, s, V, G):
    """dist(-G, regular rank normal) from a cached rank-r decomposition."""
    if s.size == 0 or s[-1] <= 1e-9 * max(s[0], 1e-300):
        return float(np.linalg.norm(G))
    return _tangent_rank_norm(U, V, G)


def _bb_step(S, Z, even, alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI):
    SS, SZ, ZZ = float(np.sum(S * S)), float(np.sum(S * Z)), float(np.sum(Z * Z))
    a = (
        (SS / abs(SZ) if abs(SZ) > 0 else alpha_hi)
        if even
        else (abs(SZ) / ZZ if ZZ > 0 else alpha_hi)
    )
    return min(alpha_hi, max(alpha_lo, a))


def cdf_pgd(prob, X0, beta=BETA, maxiter=MAXITER, maxtime=MAXTIME):
    """Algorithm 2, with the manuscript's common endpoint stopping test."""
    run = _Run(prob)
    X, U, s, V = rank_projection(np.array(X0, dtype=float, copy=True), prob.r)
    phi = cdf_value(prob, X, beta)
    phis, eta, alpha = [phi], ETA0, ALPHA0
    X_prev = G_prev = None
    while True:
        rr, vv = run.record(X)
        status = run.stop_status(rr, vv, maxiter, maxtime)
        if status:
            return run.finish(X, status)
        G = cdf_grad(prob, X, beta)
        if _rank_cone_grad_norm(U, s, V, G) < eta:
            eta *= ETA_DECAY
            XA = prob.A(X)
            phiA = cdf_value(prob, XA, beta)
            if phiA < phi and prob.accept_dissolving(XA):
                X, phi = XA, phiA
                G = cdf_grad(prob, X, beta)
                phis[-1] = phi
        if X_prev is not None:
            alpha = _bb_step(X - X_prev, G - G_prev, run.k % 2 == 0)
        mu = max(phis[max(0, run.k - MEMORY) : run.k + 1])
        alpha_t = alpha
        while True:
            _budget_check()
            Y, Uc, sc, Vc = rank_projection(X - alpha_t * G, prob.r, Y0=V)
            phiY = cdf_value(prob, Y, beta)
            if (
                prob.accept_dissolving(Y)
                and np.isfinite(phiY)
                and phiY <= mu + ARMIJO * float(np.sum(G * (Y - X)))
            ):
                break
            alpha_t *= DELTA
            if alpha_t < np.finfo(float).tiny or np.array_equal(X - alpha_t * G, X):
                run.record(X)
                return run.finish(X, "L")
        _budget_check()
        X_prev, G_prev = X, G
        X, U, s, V, phi = Y, Uc, sc, Vc, phiY
        phis.append(phi)
        alpha = alpha_t
        run.k += 1


def cdf_pgd_image(prob, X0):
    """Image Algorithm 2 core with the prescribed fixed 2000-step budget."""
    beta, nsteps = DENOISE_BETA, DENOISE_STEPS
    t0 = time.perf_counter()
    X, U, s, V = rank_projection(np.array(X0, dtype=float, copy=True), prob.r)
    alpha, eta, X_prev, G_prev = ALPHA0, ETA0, None, None
    phi = cdf_value(prob, X, beta)
    phis, hist = [phi], []
    for k in range(nsteps):
        G = cdf_grad(prob, X, beta)
        _, U, s, V = rank_projection(X, prob.r, Y0=V)
        if _rank_cone_grad_norm(U, s, V, G) < eta:
            eta *= ETA_DECAY
            XA = prob.A(X)
            phiA = cdf_value(prob, XA, beta)
            if phiA < phi:
                X, phi = XA, phiA
                G = cdf_grad(prob, X, beta)
                phis[-1] = phi
        if X_prev is not None:
            alpha = _bb_step(X - X_prev, G - G_prev, k % 2 == 0)
        mu = max(phis[max(0, k - MEMORY) : k + 1])
        alpha_t = alpha
        while True:
            Y, Uc, sc, Vc = rank_projection(X - alpha_t * G, prob.r, Y0=V)
            phiY = cdf_value(prob, Y, beta)
            if np.isfinite(phiY) and phiY <= mu + ARMIJO * float(np.sum(G * (Y - X))):
                break
            alpha_t *= DELTA
            if alpha_t < np.finfo(float).tiny or np.array_equal(X - alpha_t * G, X):
                raise RuntimeError(f"Line search stagnated at step {k}")
        X_prev, G_prev = X, G
        X, U, s, V, phi = Y, Uc, sc, Vc, phiY
        phis.append(phi)
        alpha = alpha_t
        if (k + 1) % 100 == 0 or k + 1 == nsteps:
            hist.append((time.perf_counter() - t0, k + 1, phi, prob.hnorm(X)))
            print(
                f"{prob.name}: {k+1}/{nsteps}, phi={phi:.6e}, vio={prob.hnorm(X):.3e}",
                flush=True,
            )
    return dict(
        X=X,
        iters=nsteps,
        time=time.perf_counter() - t0,
        hist=hist,
        refinement_counts=[0, 0],
    )


def cdf_pgd_migration(prob, X0):
    """Algorithm 2, with penalty stationarity and a 1000-step upper bound.

    The only dissolving update is the threshold-triggered, strictly descending
    update inside the loop. There are no initial or final refinements.
    """
    t0 = time.perf_counter()
    beta = MIGRATION_BETA
    X, U, s, V = rank_projection(np.array(X0, dtype=float, copy=True), prob.r)
    phi = cdf_value(prob, X, beta)
    phis, eta, alpha = [phi], ETA0, ALPHA0
    X_prev = G_prev = None
    hist, trace, acceleration_log = [], [], []
    acceleration_count = 0
    for k in range(MIGRATION_STEPS + 1):
        G = cdf_grad(prob, X, beta)
        rho = _rank_cone_grad_norm(U, s, V, G)
        if not np.isfinite(rho) or not np.isfinite(phi):
            raise FloatingPointError("Nonfinite Migration Algorithm 2 state")
        converged_cdf = rho < MIGRATION_CDF_TOL
        terminal = converged_cdf or k == MIGRATION_STEPS
        step = dict(k=k, cdf_residual=rho, eta=eta, phi=phi, terminal=terminal)
        trace.append(step)
        if k % 100 == 0 or terminal:
            hist.append((time.perf_counter() - t0, k, phi, prob.hnorm(X)))
            print(
                f"migration: {k}/{MIGRATION_STEPS}, phi={phi:.6e}, "
                f"cdf_res={rho:.3e}, vio={prob.hnorm(X):.3e}",
                flush=True,
            )
        if terminal:
            status = "S" if converged_cdf else "I"
            break

        triggered = rho < eta
        accepted = False
        if triggered:
            XA = prob.A(X)
            phiA = cdf_value(prob, XA, beta)
            accepted = bool(phiA < phi)
            acceleration_log.append(
                dict(k=k, cdf_residual=rho, eta_before=eta,
                     phi_before=phi, phi_after=phiA, accepted=accepted)
            )
            if accepted:
                X, phi = XA, phiA
                G = cdf_grad(prob, X, beta)
                phis[-1] = phi
                acceleration_count += 1
            eta *= ETA_DECAY

        mu = max(phis[max(0, k - MEMORY):k + 1])
        if X_prev is not None:
            alpha = _bb_step(X - X_prev, G - G_prev, k % 2 == 0)
        alpha_t = alpha
        backtracking_steps = 0
        while True:
            Y, Uc, sc, Vc = rank_projection(X - alpha_t * G, prob.r, Y0=V)
            phiY = cdf_value(prob, Y, beta)
            armijo_rhs = mu + ARMIJO * float(np.sum(G * (Y - X)))
            if np.isfinite(phiY) and phiY <= armijo_rhs:
                break
            alpha_t *= DELTA
            backtracking_steps += 1
            if alpha_t < np.finfo(float).tiny or np.array_equal(X - alpha_t * G, X):
                raise RuntimeError(f"Migration line search stagnated at step {k}")

        step.update(
            threshold_triggered=triggered,
            acceleration_accepted=accepted,
            phi_after_acceleration=phi,
            alpha_initial=alpha,
            alpha_accepted=alpha_t,
            backtracking_steps=backtracking_steps,
            line_search_reference=mu,
            phi_next=phiY,
            armijo_rhs=armijo_rhs,
        )
        X_prev, G_prev = X, G
        X, U, s, V, phi = Y, Uc, sc, Vc, phiY
        phis.append(phi)
        alpha = alpha_t

    return dict(
        X=X, iters=k, time=time.perf_counter() - t0, hist=hist, status=status,
        phi=phi, cdf_residual=rho, cdf_tolerance=MIGRATION_CDF_TOL,
        refinement_counts=[0, 0], threshold_trigger_count=len(acceleration_log),
        acceleration_count=acceleration_count, acceleration_log=acceleration_log,
        algorithm2_trace=trace,
    )


# --- space-decoupling geometry on M_h -------------------------------------
def _sd_factorize(prob, X):
    """Return X = H V.T with r orthonormal columns in V, including zero modes."""
    r = prob.r
    U, s, V = rank_projection(np.asarray(X, dtype=float), r)[1:]
    if V.shape[1] < r:
        pad = r - V.shape[1]
        Z = np.random.default_rng(0).standard_normal((prob.p, pad))
        Z -= V @ (V.T @ Z)
        Z -= V @ (V.T @ Z)
        Q, _ = np.linalg.qr(Z)
        V = np.column_stack((V, Q[:, :pad]))
        H = np.column_stack((U * s[None, :], np.zeros((prob.n, pad))))
    else:
        H = U * s[None, :]
    return H, V


def _sd_M(H, omega):
    return 2.0 * omega * np.eye(H.shape[1]) + H.T @ H


def _sd_inner(H, K1, Vp1, K2, Vp2, omega):
    return float(np.sum(K1 * K2) + np.sum(Vp1 * (Vp2 @ _sd_M(H, omega))))


def _sd_norm(H, K, Vp, omega):
    return float(np.sqrt(max(_sd_inner(H, K, Vp, K, Vp, omega), 0.0)))


def _sd_grad(prob, H, V, E, omega):
    Minv = np.linalg.solve(_sd_M(H, omega), np.eye(H.shape[1]))
    K = prob.mH_proj(H, E @ V)
    Vp = (E.T @ H) @ Minv
    return K, Vp - V @ (V.T @ Vp), Minv


def _sd_retr_first(prob, H, V, tK, tVp):
    return prob.mH_projH(H + tK), polar(V + tVp)


def _sd_retr_second(prob, H, V, K, Vp, Minv, t=1.0):
    KtH = (K.T @ H) @ Minv
    Z = V + t * (Vp @ (np.eye(H.shape[1]) - t * KtH))
    Vn = polar(Z)
    Hn = prob.mH_projH((H + t * K) @ (V.T @ Vn) + t * (H @ (Vp.T @ Vn)))
    return Hn, Vn


def _sd_transport(prob, H2, V2, K1, Vp1):
    return prob.mH_proj(H2, K1), Vp1 - V2 @ (V2.T @ Vp1)


def _sd_hess(prob, H, V, E, Minv, K, Vp):
    """Analytic Riemannian Hessian in the space-decoupling metric."""
    X = H @ V.T
    D = K @ V.T + H @ Vp.T
    HD = prob.hessf(X, D)
    ZV = E @ V
    XdV = prob.mH_proj(H, HD @ V) - prob.mH_curv(X, E, H, ZV, K)
    HK = prob.mH_proj(H, XdV + E @ Vp - (H @ Minv) @ (H.T @ (E @ Vp)))
    MK = K - (H @ Minv) @ (H.T @ K)
    modification = -Vp @ ((ZV - prob.mH_proj(H, ZV)).T @ H)
    W = ((H.T @ HD).T + (MK.T @ E).T + modification) @ Minv
    return HK, W - V @ (V.T @ W)


def _sd_tcg(prob, H, V, E, Minv, grad, Delta, omega):
    """Steihaug CG with the metric norm in every trust-region boundary solve."""
    gK, gVp = grad
    grad_norm = _sd_norm(H, gK, gVp, omega)
    if grad_norm == 0.0:
        return np.zeros_like(gK), np.zeros_like(gVp), 0, "tolerance"
    target = min(TRUST_KAPPA, grad_norm**TRUST_THETA) * grad_norm
    pK, pVp = -gK, -gVp
    rK, rVp = -gK, -gVp
    etaK, etaVp = np.zeros_like(gK), np.zeros_like(gVp)
    r_r = _sd_inner(H, rK, rVp, rK, rVp, omega)

    for j in range(int(TRUST_MAXINNER)):
        _budget_check()
        HK, HVp = _sd_hess(prob, H, V, E, Minv, pK, pVp)
        p_Hp = _sd_inner(H, pK, pVp, HK, HVp, omega)
        p_p = _sd_inner(H, pK, pVp, pK, pVp, omega)
        p_eta = _sd_inner(H, pK, pVp, etaK, etaVp, omega)
        eta_eta = _sd_inner(H, etaK, etaVp, etaK, etaVp, omega)
        if p_p <= 0.0:
            return etaK, etaVp, j, "tolerance"

        def boundary_step():
            radicand = max(p_eta * p_eta + p_p * (Delta * Delta - eta_eta), 0.0)
            tau = (-p_eta + np.sqrt(radicand)) / p_p
            return etaK + tau * pK, etaVp + tau * pVp

        if p_Hp <= 0.0:
            bK, bVp = boundary_step()
            return bK, bVp, j + 1, "negative curvature"
        alpha = r_r / p_Hp
        etaK_new, etaVp_new = etaK + alpha * pK, etaVp + alpha * pVp
        eta_eta_new = _sd_inner(H, etaK_new, etaVp_new, etaK_new, etaVp_new, omega)
        if eta_eta_new >= Delta * Delta:
            bK, bVp = boundary_step()
            return bK, bVp, j + 1, "boundary"
        etaK, etaVp = etaK_new, etaVp_new
        rK, rVp = rK - alpha * HK, rVp - alpha * HVp
        r_r_new = _sd_inner(H, rK, rVp, rK, rVp, omega)
        if np.sqrt(max(r_r_new, 0.0)) <= target:
            return etaK, etaVp, j + 1, "tolerance"
        if r_r <= 0.0:
            return etaK, etaVp, j + 1, "tolerance"
        beta = r_r_new / r_r
        pK, pVp = rK + beta * pK, rVp + beta * pVp
        r_r = r_r_new
    return etaK, etaVp, int(TRUST_MAXINNER), "maxinner"


def sd_rgd(prob, X0, maxiter=MAXITER, maxtime=MAXTIME):
    """Nonmonotone Riemannian gradient descent with alternating BB steps."""
    run = _Run(prob)
    H, V = _sd_factorize(prob, X0)
    alpha, prev_step, prev_grad, fvals = ALPHA0, None, None, []
    while True:
        _budget_check()
        X = H @ V.T
        rr, vv = run.record(X)
        status = run.stop_status(rr, vv, maxiter, maxtime)
        if status:
            return run.finish(X, status)
        E = prob.gradf(X)
        K, Vp, _ = _sd_grad(prob, H, V, E, OMEGA_RGD)
        fvals.append(prob.f(X))
        if prev_step is not None:
            sK, sVp = _sd_transport(prob, H, V, *prev_step)
            tK, tVp = _sd_transport(prob, H, V, *prev_grad)
            SS = _sd_inner(H, sK, sVp, sK, sVp, OMEGA_RGD)
            SZ = _sd_inner(H, sK, sVp, K - tK, Vp - tVp, OMEGA_RGD)
            ZZ = _sd_inner(H, K - tK, Vp - tVp, K - tK, Vp - tVp, OMEGA_RGD)
            if run.k % 2 == 0:
                a = SS / abs(SZ) if abs(SZ) > 0.0 else ALPHA_HI
            else:
                a = abs(SZ) / ZZ if ZZ > 0.0 else ALPHA_HI
            alpha = min(ALPHA_HI, max(ALPHA_LO, a))
        mu = max(fvals[max(0, run.k - MEMORY) : run.k + 1])
        alpha_t = alpha
        grad_sq = _sd_inner(H, K, Vp, K, Vp, OMEGA_RGD)
        while True:
            _budget_check()
            Hn, Vn = _sd_retr_first(prob, H, V, -alpha_t * K, -alpha_t * Vp)
            if prob.f(Hn @ Vn.T) <= mu - ARMIJO * alpha_t * grad_sq:
                break
            next_alpha = alpha_t * DELTA
            if next_alpha == 0.0 or (np.array_equal(Hn, H) and np.array_equal(Vn, V)):
                return run.finish(X, "L")
            alpha_t = next_alpha
        alpha = alpha_t
        prev_step, prev_grad = (-alpha_t * K, -alpha_t * Vp), (K, Vp)
        H, V = Hn, Vn
        run.k += 1


def sd_rtr(prob, X0, maxiter=MAXITER, maxtime=MAXTIME):
    """Space-decoupling trust regions with analytic Hessian and second-order retraction."""
    run = _Run(prob)
    H, V = _sd_factorize(prob, X0)
    Delta_bar = float(np.sqrt((prob.n + prob.p - prob.r) * prob.r))
    Delta = TRUST_RADIUS0
    while True:
        _budget_check()
        X = H @ V.T
        rr, vv = run.record(X)
        status = run.stop_status(rr, vv, maxiter, maxtime)
        if status:
            return run.finish(X, status)
        E, fX = prob.gradf(X), prob.f(X)
        K, Vp, Minv = _sd_grad(prob, H, V, E, OMEGA_RTR)
        if _sd_norm(H, K, Vp, OMEGA_RTR) == 0.0:
            return run.finish(X, "A")
        etaK, etaVp, _, stop_inner = _sd_tcg(
            prob, H, V, E, Minv, (K, Vp), Delta, OMEGA_RTR
        )
        _budget_check()
        Hn, Vn = _sd_retr_second(prob, H, V, etaK, etaVp, Minv)
        num = fX - prob.f(Hn @ Vn.T)
        HetaK, HetaVp = _sd_hess(prob, H, V, E, Minv, etaK, etaVp)
        model_dec = -(
            _sd_inner(H, K, Vp, etaK, etaVp, OMEGA_RTR)
            + 0.5 * _sd_inner(H, etaK, etaVp, HetaK, HetaVp, OMEGA_RTR)
        )
        rho = num / model_dec if model_dec > 0.0 else -1.0
        if rho < 0.25:
            Delta = max(Delta / 4.0, 1e-12)
        elif rho > 0.75 and stop_inner in ("boundary", "negative curvature"):
            Delta = min(2.0 * Delta, Delta_bar)
        if rho > TRUST_RHO_PRIME and num > 0.0:
            H, V = Hn, Vn
        run.k += 1


def _alm_value(prob, X, lam, rho):
    hv = prob.hvec(X)
    return prob.f(X) + float(lam @ hv) + 0.5 * rho * float(hv @ hv)


def alm_pgd(prob, X0, maxiter=MAXITER, maxtime=MAXTIME):
    """ALM with monotone PGD subproblems; iterations count accepted inner steps."""
    run = _Run(prob)
    X, U, s, V = rank_projection(np.array(X0, dtype=float, copy=True), prob.r)
    lam = np.zeros(prob.hvec(X).size)
    eps, alpha = ALM_EPS0, ALPHA0
    rr, vv = run.record(X)
    while True:
        _budget_check()
        status = run.stop_status(rr, vv, maxiter, maxtime)
        if status:
            return run.finish(X, status)
        for _ in range(ALM_INNER_MAX):
            _budget_check()
            G = prob.gradf(X) + prob.gradh(X, lam + ALM_RHO * prob.hvec(X))
            if _rank_cone_grad_norm(U, s, V, G) <= eps:
                break
            fX = _alm_value(prob, X, lam, ALM_RHO)
            alpha_t = alpha
            while True:
                _budget_check()
                Y, Uc, sc, Vc = rank_projection(X - alpha_t * G, prob.r, Y0=V)
                if _alm_value(prob, Y, lam, ALM_RHO) <= fX + ARMIJO * float(
                    np.sum(G * (Y - X))
                ):
                    break
                next_alpha = alpha_t * DELTA
                if next_alpha == 0.0 or np.array_equal(Y, X):
                    return run.finish(X, "L")
                alpha_t = next_alpha
            alpha = alpha_t
            X, U, s, V = Y, Uc, sc, Vc
            run.k += 1
            rr, vv = run.record(X)
            status = run.stop_status(rr, vv, maxiter, maxtime)
            if status:
                return run.finish(X, status)
        lam += ALM_RHO * prob.hvec(X)
        eps = max(ALM_EPS_MIN, eps * ALM_EPS_DECAY)


def _initial_factors(prob, X0):
    """A rank-``r`` factorisation ``X0 = U0 V0^T`` with ``V0`` orthonormal."""
    r = prob.r
    U, s, V = lmsvd(X0, r)
    Ur, Vr, Sr = U[:, :r], V[:, :r], s[:r]
    if Vr.shape[1] < r:
        pad = r - Vr.shape[1]
        Z = np.random.default_rng(0).standard_normal((prob.p, pad))
        Z = Z - Vr @ (Vr.T @ Z)
        Q, _ = np.linalg.qr(Z)
        Vr = np.hstack([Vr, Q])
        Ur = np.hstack([Ur, np.zeros((prob.n, pad))])
        Sr = np.concatenate([Sr, np.zeros(pad)])
    return Ur * Sr[None, :], Vr


def _pack(U, V):
    return np.concatenate([U.ravel(), V.ravel()])


def _unpack(z, n, p, r):
    return z[: n * r].reshape(n, r), z[n * r :].reshape(p, r)


def make_eq_functions(prob, eq2):
    """Objective, gradient, constraints and Jacobian for the EQ formulations."""
    n, p, r = prob.n, prob.p, prob.r
    tri = np.triu_indices(r)

    def reduced_h(U):
        if prob.name == "spherical":
            return np.sum(U * U, axis=1) - 1.0
        return np.array([float(np.sum(U * U)) - 1.0])

    def fun(z):
        U, V = _unpack(z, n, p, r)
        return prob.f(U @ V.T)

    def jac(z):
        U, V = _unpack(z, n, p, r)
        G = prob.gradf(U @ V.T)
        return np.concatenate([(G @ V).ravel(), (G.T @ U).ravel()])

    def con(z):
        U, V = _unpack(z, n, p, r)
        if eq2:
            return np.concatenate([reduced_h(U), (V.T @ V)[tri] - np.eye(r)[tri]])
        return prob.hvec(U @ V.T)

    def conjac(z):
        U, V = _unpack(z, n, p, r)
        rows = []
        if eq2:
            if prob.name == "spherical":
                for i in range(n):
                    gU = np.zeros((n, r))
                    gU[i, :] = 2.0 * U[i, :]
                    rows.append(np.concatenate([gU.ravel(), np.zeros(p * r)]))
            else:
                rows.append(np.concatenate([(2.0 * U).ravel(), np.zeros(p * r)]))
            for k in range(tri[0].size):
                a, b = int(tri[0][k]), int(tri[1][k])
                gV = np.zeros((p, r))
                gV[:, a] += V[:, b]
                gV[:, b] += V[:, a]
                rows.append(np.concatenate([np.zeros(n * r), gV.ravel()]))
            return np.array(rows)
        lam = np.zeros(prob.hvec(U @ V.T).size)
        for i in range(lam.size):
            lam[:] = 0.0
            lam[i] = 1.0
            Gh = prob.gradh(U @ V.T, lam)
            rows.append(np.concatenate([(Gh @ V).ravel(), (Gh.T @ U).ravel()]))
        return np.array(rows)

    return fun, jac, con, conjac


def solve_eq(prob, X0, eq2, solver, maxtime=EQ_MAXTIME, maxiter=EQ_MAXITER):
    """Native SciPy stopping followed by the common endpoint diagnostic."""
    global _BUDGET_SECONDS
    from scipy.optimize import minimize

    U0, V0 = _initial_factors(prob, X0)
    z0 = _pack(U0, V0)
    fun, jac, con, conjac = make_eq_functions(prob, eq2)
    t0 = _BUDGET_START if _BUDGET_START is not None else time.perf_counter()
    state = {"X": U0 @ V0.T, "iters": 0}

    def record(z, st=None):
        _budget_check()
        U, V = _unpack(z, prob.n, prob.p, prob.r)
        X = U @ V.T
        if not np.all(np.isfinite(X)):
            raise FloatingPointError("Nonfinite factorized iterate")
        state["X"], state["iters"] = X, state["iters"] + 1
        _send_progress(("eq_snapshot", X, state["iters"], time.perf_counter() - t0))
        _budget_check()
        return False

    cons = [{"type": "eq", "fun": con, "jac": conjac}]
    try:
        if solver == "SLSQP":
            opt = minimize(
                fun,
                z0,
                jac=jac,
                method="SLSQP",
                constraints=cons,
                options={"maxiter": maxiter, "ftol": EQ_TOL},
                callback=record,
            )
        else:
            opt = minimize(
                fun,
                z0,
                jac=jac,
                method="trust-constr",
                constraints=cons,
                options={
                    "maxiter": maxiter,
                    "gtol": EQ_TOL,
                    "xtol": EQ_TOL,
                    "barrier_tol": EQ_TOL,
                    "verbose": 0,
                },
                callback=record,
            )
        _budget_check()
        U, V = _unpack(opt.x, prob.n, prob.p, prob.r)
        if np.all(np.isfinite(U @ V.T)):
            state["X"] = U @ V.T
        state["iters"] = min(int(opt.nit), maxiter)
        status = "A" if opt.success else ("I" if opt.nit >= maxiter else "F")
        message = str(opt.message)
    except _BudgetExpired:
        raise
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        status, message = "F", str(exc)
    elapsed = time.perf_counter() - t0
    _BUDGET_SECONDS = None
    _send_progress(("native_finished", elapsed))
    X = state["X"]
    rr, vv, ff = res(prob, X), vio(prob, X), prob.f(X)
    if rr <= RES_TOL and vv <= VIO_TOL:
        status = "C"
    return dict(
        X=X,
        fun=ff,
        res=rr,
        vio=vv,
        iters=state["iters"],
        time=elapsed,
        status=status,
        native_message=message,
    )


def make_spherical(n, p, r, nu=NOISE_NU, d=NOISE_D, seed=SEED_DATA, seed_init=None):
    """Rank-``r`` signal on the oblique manifold plus correlated rank-``r`` noise."""
    rng = np.random.default_rng(seed)
    Vstar, _ = np.linalg.qr(rng.standard_normal((p, r)))
    scales = 10.0 ** (-d * np.arange(r) / (r - 1)) if r > 1 else np.ones(1)
    A0 = (rng.standard_normal((n, r)) * scales[None, :]) @ Vstar.T
    A0 = A0 / np.linalg.norm(A0, axis=1, keepdims=True)

    if nu > 0:
        E = rng.standard_normal((n, r)) @ rng.standard_normal((r, p))
        E = E / np.linalg.norm(E) * (nu * np.sqrt(n))
        Adata = A0 + E
        Adata = Adata / np.linalg.norm(Adata, axis=1, keepdims=True)
    else:
        Adata = A0

    rng0 = np.random.default_rng(seed if seed_init is None else seed_init)
    V0, _ = np.linalg.qr(rng0.standard_normal((p, r)))
    H0 = rng0.standard_normal((n, r))
    H0 = H0 / np.linalg.norm(H0, axis=1, keepdims=True)
    return Adata, H0 @ V0.T


def _directed_er(N, avg_deg, seed):
    """Directed Erdos-Renyi adjacency matrix without self loops."""
    rng = np.random.default_rng(seed)
    A = (rng.random((N, N)) < avg_deg / (N - 1)).astype(float)
    np.fill_diagonal(A, 0.0)
    return sp.csr_matrix(A)


def make_graph(n, p, avg_deg=GRAPH_AVG_DEG, seed=SEED_DATA):
    """Independent directed Erdos-Renyi graphs and the shared initial matrix."""
    return (
        _directed_er(n, avg_deg, seed),
        _directed_er(p, avg_deg, seed + 1),
        np.ones((n, p)) / np.sqrt(n * p),
    )


def make_graph_cycle_pair(n, avg_deg, seed):
    """Directed cycle paired with an independent directed random graph."""
    idx = np.arange(n)
    A = sp.csr_matrix((np.ones(n), (idx, (idx + 1) % n)), shape=(n, n))
    return A, _directed_er(n, avg_deg, seed)


def load_migration(
    r,
    holdout=MIGRATION_HOLDOUT,
    seed=MIGRATION_SEED,
    lam=MIGRATION_LAMBDA,
    path=MIGRATION_XLSX,
):
    """Build the 233 x 233 destination-origin matrix from the UN DESA 2024 data.

    The 233 entities are the 232 countries/areas plus the official ``Others``
    origin category, used on both axes; destination rows absent from the source
    table are zero.  This reproduces exactly 9042 positive bilateral entries, of
    which a random 10 % (seed 2024) are held out for testing.
    """
    if os.path.exists(path):
        import pandas as pd

        df = pd.read_excel(path, sheet_name="Table 1", header=10)
        df = df[df["Index"].notna()].rename(
            columns={
                "Region, development group, country or area of destination": "dest",
                "Location code of destination": "dc",
                "Region, development group, country or area of origin": "orig",
                "Location code of origin": "oc",
            }
        )
        df["dc"] = pd.to_numeric(df["dc"], errors="coerce")
        df["oc"] = pd.to_numeric(df["oc"], errors="coerce")
        ctry = df[
            (df["dc"].notna())
            & (df["oc"].notna())
            & (df["dc"] < 900)
            & (df["oc"] < 900)
        ]
        ycol = [c for c in ctry.columns if str(c).startswith("2024")][0]
        val = pd.to_numeric(ctry[ycol], errors="coerce").fillna(0.0).to_numpy(float)

        codes = np.sort(ctry["oc"].unique())
        idx = {c: i for i, c in enumerate(codes)}
        N = len(codes)
        R = np.zeros((N, N))
        for dc, oc, v in zip(ctry["dc"].to_numpy(), ctry["oc"].to_numpy(), val):
            if v > 0:
                R[idx[dc], idx[oc]] = v
    else:
        archive = Path(ROOT, "data", "migration_un2024.npz")
        if not archive.exists():
            raise FileNotFoundError(
                "Provide the UN workbook or bundled migration_un2024.npz"
            )
        with np.load(archive) as source:
            R = np.array(source["R"], dtype=float)
        N = R.shape[0]

    b = R.sum(axis=1)
    F = np.outer(b, np.ones(N)) / N
    pos = np.argwhere(R > 0)
    perm = np.random.default_rng(seed).permutation(pos.shape[0])
    n_test = int(round(holdout * pos.shape[0]))
    test_idx, train_idx = pos[perm[:n_test]], pos[perm[n_test:]]

    mask = np.zeros((N, N))
    mask[train_idx[:, 0], train_idx[:, 1]] = 1.0
    meta = {
        "n_pos": int(pos.shape[0]),
        "n_train": int(train_idx.shape[0]),
        "n_test": int(test_idx.shape[0]),
        "test_idx": test_idx,
    }
    return R, mask, F, b, r, lam, meta


def _cameraman256():
    from skimage import data, transform

    img = data.camera().astype(float) / 255.0
    return transform.resize(
        img, (256, 256), order=1, anti_aliasing=True, preserve_range=True
    )


def make_denoising(kind, r=DENOISE_RANK, nu=DENOISE_NU, seed=DENOISE_SEED):
    """Rank-``r`` Cameraman reference with Gaussian noise on a region.

    Returns ``(Icor, Iclean, mask, X0)`` on the ``[0, 1]`` scale; ``mask`` is 1
    on the known (uncorrupted) pixels.
    """
    clean = rank_projection(_cameraman256(), r)[0]
    region = np.zeros(clean.shape, dtype=bool)
    for i, j in DENOISE_REGIONS[kind]:
        side = DENOISE_SIDE[kind]
        region[i : i + side, j : j + side] = True

    noise = np.random.default_rng(seed).standard_normal(clean.shape) * (nu / 255.0)
    Icor = clean.copy()
    Icor[region] = clean[region] + noise[region]
    return Icor, clean, (~region).astype(float), rank_projection(Icor, r)[0]


# ==========================================================================
# 7. Task pool and experiment drivers
# ==========================================================================


def _summarise(out):
    record = {k: v for k, v in out.items() if k != "X"}
    record["svd_diagnostics"] = dict(LMSVD_DIAGNOSTICS)
    return record


def _run_tasks(tasks, fn, label):
    """One fixed-seed run per task, with immediate persistent checkpoints."""
    results, previous = {}, _load_results()
    for task in tasks:
        key = _task_key(task, fn)
        if RESUME and key in previous:
            results[key] = previous[key]
            print(f"[{label}] resume {key}", flush=True)
            continue
        out = _hard_limited_task(task, fn)
        results[out["key"]] = out["res"]
        previous.update(results)
        _save_results(previous)
        rec = out["res"]
        print(
            f"[{label}] {out['key']} -> {rec['status']} "
            f"it={rec['iters']} t={rec['time']:.3f}s res={rec['res']:.3e} "
            f"vio={rec['vio']:.3e}",
            flush=True,
        )
        # Cooperative interruption only, after a complete result has been saved.
        if Path(ROOT, "STOP_AFTER_CURRENT").exists():
            raise SystemExit(
                "Stopped after checkpoint; remove STOP_AFTER_CURRENT to resume"
            )
    return results


def _load_results():
    if Path(RESULTS_JSON).exists():
        return json.loads(Path(RESULTS_JSON).read_text())
    return {}


def _finite_json(value):
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _save_results(results):
    path = Path(RESULTS_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_finite_json(results), indent=1, allow_nan=False))
    temporary.replace(path)


def _sphere_task(args):
    kind, n, p, r, rep, method, maxtime = args
    Adata, X0 = make_spherical(n, p, r, seed=SEED_DATA, seed_init=SEED_INIT + rep)
    prob = SphericalFitting(Adata, r)
    fn = {
        "CDF-PGD": lambda: cdf_pgd(prob, X0, BETA, MAXITER, maxtime),
        "SD-RGD": lambda: sd_rgd(prob, X0, MAXITER, maxtime),
        "SD-RTR": lambda: sd_rtr(prob, X0, MAXITER, maxtime),
        "ALM-PGD": lambda: alm_pgd(prob, X0, MAXITER, maxtime),
    }[method]
    _start_budget(prob, X0, maxtime)
    return dict(key=f"{kind}|{n}x{p}x{r}|rep{rep}|{method}", res=_summarise(fn()))


def _graph_task(args):
    kind, n, p, r, rep, method, maxtime, cycle, deg = args
    if cycle:
        AdjA, AdjB = make_graph_cycle_pair(n, deg, SEED_DATA)
    else:
        AdjA, AdjB, _ = make_graph(n, p, deg, SEED_DATA)
    prob = GraphSimilarity(AdjA, AdjB, r)
    X0 = np.ones((n, p)) / np.sqrt(n * p)
    fn = {
        "CDF-PGD": lambda: cdf_pgd(prob, X0, BETA, MAXITER, maxtime),
        "SD-RGD": lambda: sd_rgd(prob, X0, MAXITER, maxtime),
        "SD-RTR": lambda: sd_rtr(prob, X0, MAXITER, maxtime),
        "ALM-PGD": lambda: alm_pgd(prob, X0, MAXITER, maxtime),
    }[method]
    _start_budget(prob, X0, maxtime)
    return dict(key=f"{kind}|{n}x{p}x{r}|rep{rep}|{method}", res=_summarise(fn()))


def _eq_task(args):
    which, n, p, r, eq2, solver, maxtime = args
    if which == "sphere":
        Adata, X0 = make_spherical(n, p, r, seed=SEED_DATA, seed_init=SEED_INIT)
        prob = SphericalFitting(Adata, r)
    else:
        AdjA, AdjB, _ = make_graph(n, p, GRAPH_AVG_DEG, SEED_DATA)
        prob = GraphSimilarity(AdjA, AdjB, r)
        X0 = np.ones((n, p)) / np.sqrt(n * p)
    _start_budget(prob, X0, maxtime, eq=True)
    out = solve_eq(prob, X0, eq2, solver, maxtime=maxtime)
    name = f"EQ{2 if eq2 else 1}-{'TC' if solver == 'trust-constr' else 'SLSQP'}"
    rec = _summarise(out)
    rec["fun"] = float(out["fun"])
    return dict(key=f"{which}|{n}x{p}x{r}|{name}", res=rec)


def _eq_cdf_task(args):
    which, n, p, r = args
    if which == "sphere":
        Adata, X0 = make_spherical(n, p, r, seed=SEED_DATA, seed_init=SEED_INIT)
        prob = SphericalFitting(Adata, r)
    else:
        AdjA, AdjB, _ = make_graph(n, p, GRAPH_AVG_DEG, SEED_DATA)
        prob = GraphSimilarity(AdjA, AdjB, r)
        X0 = np.ones((n, p)) / np.sqrt(n * p)
    _start_budget(prob, X0, EQ_MAXTIME)
    out = cdf_pgd(prob, X0, BETA, EQ_MAXITER, EQ_MAXTIME)
    rec = _summarise(out)
    rec["fun"] = float(prob.f(out["X"]))
    return dict(key=f"{which}|{n}x{p}x{r}|CDF-PGD", res=rec)


def exp_scale(kind):
    """Main comparison for spherical fitting or graph similarity."""
    taskfn = _sphere_task if kind == "sphere" else _graph_task
    small_tasks, large_tasks = [], []
    for n, p, r in SPHERE_SIZES:
        for rep in range(REPETITIONS):
            for m in MAIN_METHODS:
                if kind == "sphere":
                    small_tasks.append((kind, n, p, r, rep, m, MAXTIME))
                else:
                    small_tasks.append(
                        (kind, n, p, r, rep, m, MAXTIME, False, GRAPH_AVG_DEG)
                    )
    for m in MAIN_METHODS:
        if kind == "sphere":
            large_tasks.append((kind, *SPHERE_LARGE, 0, m, MAXTIME_LARGE))
        else:
            large_tasks.append(
                (kind, *SPHERE_LARGE, 0, m, MAXTIME_LARGE, False, GRAPH_AVG_DEG)
            )

    results = _load_results()
    results.update(_run_tasks(small_tasks, taskfn, kind))
    results.update(_run_tasks(large_tasks, taskfn, kind + "-large"))
    _save_results(results)


def exp_graph_validation():
    """Additional validation instances for graph similarity (16 instances).

    Each instance pairs a directed random graph with either an independent
    random graph or a directed cycle, over the expected degrees, sizes and
    ranks listed in the configuration section.  Every instance is solved once.
    """
    tasks, idx = [], 0
    for deg in GRAPH_VAL_DEGREES:
        for n in GRAPH_VAL_SIZES:
            for r in GRAPH_VAL_RANKS:
                for cycle in (False, True):
                    for m in MAIN_METHODS:
                        tasks.append(
                            ("graphval", n, n, r, idx, m, EQ_MAXTIME, cycle, deg)
                        )
                    idx += 1
    results = _load_results()
    results.update(_run_tasks(tasks, _graph_task, "graphval"))
    _save_results(results)


def exp_eq(which):
    """Factorised equality formulations, plus CDF-PGD on the same instances."""
    eq_tasks = [
        (which, n, p, r, eq2, solver, EQ_MAXTIME)
        for (n, p, r) in EQ_SIZES
        for eq2 in (False, True)
        for solver in ("SLSQP", "trust-constr")
    ]
    cdf_tasks = [(which, n, p, r) for (n, p, r) in EQ_SIZES]
    results = _load_results()
    results.update(_run_tasks(eq_tasks, _eq_task, f"eq-{which}"))
    results.update(_run_tasks(cdf_tasks, _eq_cdf_task, f"eqcdf-{which}"))
    _save_results(results)


# --- migration stock estimation -------------------------------------------
def iterative_svd(Xin, rank, threshold=ITERATIVE_SVD_TOL, max_iters=1000):
    """IterativeSVD of fancyimpute 0.7.0, with LMSVD as the truncated SVD."""
    missing = np.isnan(Xin)
    X = np.nan_to_num(Xin, nan=0.0)
    X_filled = X.copy()
    it = 0
    for i in range(max_iters):
        it = i + 1
        Xrec = rank_projection(X_filled, min(2**i, rank))[0]
        diff = X_filled[missing] - Xrec[missing]
        ssd, ons = float(np.sum(diff**2)), float(np.sum(X_filled[missing] ** 2))
        conv = (
            not (
                ons == 0.0
                or (ons < np.finfo(np.float32).eps and ssd > np.finfo(np.float32).eps)
            )
            and (ssd / ons) < threshold
        )
        X_filled[missing] = Xrec[missing]
        if conv:
            break
    X_filled[~missing] = X[~missing]
    return X_filled, it


def soft_impute(Xin, max_rank, threshold=SOFT_IMPUTE_TOL, max_iters=1000):
    """SoftImpute of fancyimpute 0.7.0, with LMSVD as the decomposition."""
    missing = np.isnan(Xin)
    X = np.nan_to_num(Xin, nan=0.0)
    X_filled = X.copy()
    shrinkage = lmsvd(X_filled, 1, tol=1e-10, maxit=200)[1][0] / SOFT_IMPUTE_SHRINK
    it, rk = 0, 0
    for i in range(max_iters):
        it = i + 1
        U, s, V = lmsvd(X_filled, max_rank)
        s_thresh = np.maximum(s - shrinkage, 0.0)
        rk = int((s_thresh > 0).sum())
        Xrec = (
            (U[:, :rk] * s_thresh[:rk]) @ V[:, :rk].T if rk else np.zeros_like(X_filled)
        )
        diff = X_filled[missing] - Xrec[missing]
        ssd, ons = float(np.sum(diff**2)), float(np.sum(X_filled[missing] ** 2))
        conv = (
            not (
                ons == 0.0
                or (ons < np.finfo(np.float32).eps and ssd > np.finfo(np.float32).eps)
            )
            and np.sqrt(ssd / ons) < threshold
        )
        X_filled[missing] = Xrec[missing]
        if conv:
            break
    X_filled[~missing] = X[~missing]
    return X_filled, it


def exp_migration():
    """Migration comparison with a checkpoint after each completed method."""
    results = _load_results()
    array_dir = Path(ROOT, "results", "arrays")
    array_dir.mkdir(parents=True, exist_ok=True)
    for r in MIGRATION_RANKS:
        R, mask, F, b, _, _, meta = load_migration(r)
        tr, tc = meta["test_idx"][:, 0], meta["test_idx"][:, 1]

        key = f"migration|r{r}|CDF-PGD"
        array_path = array_dir / f"migration_r{r}_CDF-PGD.npz"
        if RESUME and key in results and array_path.is_file():
            print(f"[migration] resume {key}", flush=True)
        else:
            prob = Migration(R, mask, F, b, r)
            X0 = rank_projection(mask * R, r)[0]
            o = cdf_pgd_migration(prob, X0)
            X = o["X"]
            np.savez_compressed(array_path, X=X, X0=X0)
            err = X[tr, tc] - R[tr, tc]
            results[key] = dict(
                rmse=float(np.sqrt(np.mean(err**2))),
                mae=float(np.mean(np.abs(err))),
                iters=int(o["iters"]),
                time=float(o["time"]),
                rank=int(numerical_rank(X)),
                refinement_counts=o["refinement_counts"],
                hist=o["hist"],
                status=o["status"],
                phi=o["phi"],
                cdf_residual=o["cdf_residual"],
                cdf_tolerance=o["cdf_tolerance"],
                threshold_trigger_count=o["threshold_trigger_count"],
                acceleration_count=o["acceleration_count"],
                acceleration_log=o["acceleration_log"],
                algorithm2_trace=o["algorithm2_trace"],
                training_input_sha256=hashlib.sha256(
                    np.ascontiguousarray(mask * R).tobytes()
                ).hexdigest(),
                training_mask_sha256=hashlib.sha256(
                    np.ascontiguousarray(mask).tobytes()
                ).hexdigest(),
                test_indices_sha256=hashlib.sha256(
                    np.ascontiguousarray(meta["test_idx"]).tobytes()
                ).hexdigest(),
                re=float(100.0 * np.linalg.norm(X.sum(axis=1) - b) / np.linalg.norm(b)),
            )
            _save_results(results)
            print(f"r={r} CDF-PGD done t={o['time']:.2f}s", flush=True)
            if Path(ROOT, "STOP_AFTER_CURRENT").exists():
                raise SystemExit(
                    "Stopped after checkpoint; remove STOP_AFTER_CURRENT to resume"
                )

        Xin = np.full(R.shape, np.nan)
        Xin[mask > 0] = R[mask > 0]
        for label, fn in (
            ("IterativeSVD", lambda: iterative_svd(Xin, r)),
            ("SoftImpute", lambda: soft_impute(Xin, r)),
        ):
            key = f"migration|r{r}|{label}"
            array_path = array_dir / f"migration_r{r}_{label}.npz"
            if RESUME and key in results and array_path.is_file():
                print(f"[migration] resume {key}", flush=True)
                continue
            t0 = time.perf_counter()
            Xc, it = fn()
            elapsed = time.perf_counter() - t0
            np.savez_compressed(array_path, X=Xc)
            e = Xc[tr, tc] - R[tr, tc]
            results[key] = dict(
                rmse=float(np.sqrt(np.mean(e**2))),
                mae=float(np.mean(np.abs(e))),
                iters=int(it),
                time=float(elapsed),
                rank=int(numerical_rank(Xc)),
                re=float(
                    100.0 * np.linalg.norm(Xc.sum(axis=1) - b) / np.linalg.norm(b)
                ),
            )
            _save_results(results)
            print(f"r={r} {label} done t={elapsed:.2f}s", flush=True)
            if Path(ROOT, "STOP_AFTER_CURRENT").exists():
                raise SystemExit(
                    "Stopped after checkpoint; remove STOP_AFTER_CURRENT to resume"
                )


# --- image denoising ------------------------------------------------------
def _psnr(a, b, mask):
    return 10.0 * np.log10(1.0 / max(float(np.mean((a[mask] - b[mask]) ** 2)), 1e-300))


def _rmse(a, b, mask):
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def exp_denoise():
    """Image comparison with one checkpoint per completed corrupted region."""
    from skimage.restoration import denoise_nl_means
    import bm3d

    results = _load_results()
    panels = {}
    array_dir = Path(ROOT, "results", "arrays")
    array_dir.mkdir(parents=True, exist_ok=True)
    profile = bm3d.BM3DProfile()
    profile.num_threads = 1
    for kind in DENOISE_REGIONS:
        key = f"denoise|{kind}"
        array_path = array_dir / f"denoise_{kind}.npz"
        if RESUME and key in results and array_path.is_file():
            with np.load(array_path) as saved:
                panels[kind] = (
                    saved["noisy"],
                    saved["cdf"],
                    saved["nlm"],
                    saved["bm3d"],
                    saved["corrupted"],
                )
            print(f"[denoise] resume {key}", flush=True)
            continue
        Icor, Iclean, known, X0 = make_denoising(kind)
        corrupted = known < 0.5
        prob = ImageDenoising(Icor, Iclean, known, DENOISE_RANK)
        o = cdf_pgd_image(prob, X0)
        Xcdf = o["X"]

        sigma = DENOISE_NU / 255.0
        Xnlm = np.where(
            corrupted,
            denoise_nl_means(
                Icor,
                patch_size=NLM_PATCH,
                patch_distance=NLM_PATCH_DISTANCE,
                h=NLM_FILTER * sigma,
                sigma=sigma,
                fast_mode=True,
            ),
            Iclean,
        )
        Xbm3d = np.where(corrupted, bm3d.bm3d(Icor, sigma, profile=profile), Iclean)

        results[key] = dict(
            noisy=_psnr(Icor, Iclean, corrupted),
            cdf=_psnr(Xcdf, Iclean, corrupted),
            nlm=_psnr(Xnlm, Iclean, corrupted),
            bm3d=_psnr(Xbm3d, Iclean, corrupted),
            cdf_clean_rmse=_rmse(Xcdf, Iclean, ~corrupted),
            cdf_rank=int(numerical_rank(Xcdf)),
            cdf_iters=int(o["iters"]),
            cdf_time=float(o["time"]),
            hist=o["hist"],
            refinement_counts=o["refinement_counts"],
            bm3d_threads=1,
        )
        panels[kind] = (Icor, Xcdf, Xnlm, Xbm3d, corrupted)
        np.savez_compressed(
            array_path,
            noisy=Icor,
            clean=Iclean,
            cdf=Xcdf,
            nlm=Xnlm,
            bm3d=Xbm3d,
            corrupted=corrupted,
        )
        _save_results(results)
        print(
            f"{kind}: "
            + " ".join(
                f"{field}={results[key][field]:.2f}"
                for field in ("noisy", "cdf", "nlm", "bm3d")
            ),
            flush=True,
        )
        if Path(ROOT, "STOP_AFTER_CURRENT").exists():
            raise SystemExit(
                "Stopped after checkpoint; remove STOP_AFTER_CURRENT to resume"
            )

    make_denoise_figure(panels)


# ==========================================================================
# 8. Figures
# ==========================================================================
def _single_runs(results, kind, dims):
    return {
        m: results[f"{kind}|{dims}|rep0|{m}"]
        for m in MAIN_METHODS
        if f"{kind}|{dims}|rep0|{m}" in results
    }


def make_triptych(kind, n, p, r, fname):
    """Residual vs iteration, residual vs time, violation vs iteration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = _single_runs(_load_results(), kind, f"{n}x{p}x{r}")
    if not runs:
        print(f"no {kind} results for ({n},{p},{r}); run --exp {kind} first")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))
    for m, run in runs.items():
        h = np.array(run["hist"], dtype=float)
        style = dict(METHOD_STYLE[m], lw=FIGURE_LINEWIDTH)
        axes[0].semilogy(h[:, 3], np.maximum(h[:, 1], 1e-16), label=m, **style)
        axes[1].semilogy(h[:, 0], np.maximum(h[:, 1], 1e-16), **style)
        axes[2].semilogy(h[:, 3], np.maximum(h[:, 2], 1e-18), **style)

    for ax, tol in zip(axes, (RES_TOL, RES_TOL, VIO_TOL)):
        ax.axhline(tol, color="gray", lw=0.8, ls=":")
        ax.grid(True, which="both", alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=9)
    axes[0].set_xlabel("iteration", fontsize=11)
    axes[1].set_xlabel("time (s)", fontsize=11)
    axes[2].set_xlabel("iteration", fontsize=11)
    # All four methods are plotted with the identical stationarity measure:
    # the distance of grad f to the normal cone of the feasible set.
    axes[0].set_ylabel("Stationarity", fontsize=11)
    axes[1].set_ylabel("Stationarity", fontsize=11)
    axes[2].set_ylabel("Constraint Violation", fontsize=11)
    axes[0].legend(fontsize=10, frameon=False)

    fig.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, fname)
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print("wrote", path)


def make_denoise_figure(panels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy import ndimage

    kinds = list(DENOISE_REGIONS)
    fig, axes = plt.subplots(len(kinds), 4, figsize=(11, 8.6))
    for row, kind in enumerate(kinds):
        Icor, Xcdf, Xnlm, Xbm3d, corrupted = panels[kind]
        for col, img in enumerate([Icor, Xcdf, Xnlm, Xbm3d]):
            ax = axes[row, col]
            ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(["Noisy", "CDF-PGD", "NLM", "BM3D"][col], fontsize=11)
        lab, nlab = ndimage.label(corrupted)
        for i in range(1, nlab + 1):
            ys, xs = np.where(lab == i)
            if ys.size == 0:
                continue
            for col in range(4):
                axes[row, col].add_patch(
                    Rectangle(
                        (xs.min() - 1.5, ys.min() - 1.5),
                        xs.max() - xs.min() + 3,
                        ys.max() - ys.min() + 3,
                        fill=False,
                        edgecolor="red",
                        lw=1.2,
                    )
                )
    fig.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, "denoising_effect_rank50.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", path)


def exp_figures():
    make_triptych("sphere", 1000, 1000, 25, "sphere_1000_triptych.png")
    make_triptych("sphere", 2000, 2000, 50, "sphere_2000_triptych.png")
    make_triptych("graph", 1000, 1000, 25, "graph_1000_triptych.png")
    make_triptych("graph", 2000, 2000, 50, "graph_2000_triptych.png")


# ==========================================================================
# 9. Entry point
# ==========================================================================
EXPERIMENTS = {
    "sphere": lambda: exp_scale("sphere"),
    "graph": lambda: (exp_scale("graph"), exp_graph_validation()),
    "eq-sphere": lambda: exp_eq("sphere"),
    "eq-graph": lambda: exp_eq("graph"),
    "migration": exp_migration,
    "denoise": exp_denoise,
    "figures": exp_figures,
}
# ``figures`` consumes the sphere/graph histories, so it runs last under --exp all.
ALL_ORDER = [
    "sphere",
    "graph",
    "eq-sphere",
    "eq-graph",
    "migration",
    "denoise",
    "figures",
]


def main():
    global NPROC, RESUME
    ap = argparse.ArgumentParser(
        description="Reproduce the numerical experiments of the CDF-PGD paper."
    )
    ap.add_argument(
        "--exp",
        required=True,
        choices=sorted(EXPERIMENTS) + ["all"],
        help="which experiment to run",
    )
    ap.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of single-threaded worker processes",
    )
    ap.add_argument("--resume", action="store_true", help="skip complete saved solves")
    args = ap.parse_args()
    RESUME = args.resume
    NPROC = args.nproc
    if NPROC != 1:
        ap.error("Timed solves run serially; use --nproc 1")
    write_environment()

    names = ALL_ORDER if args.exp == "all" else [args.exp]
    for name in names:
        print(f"=== {name} ===", flush=True)
        EXPERIMENTS[name]()


if __name__ == "__main__":
    main()
