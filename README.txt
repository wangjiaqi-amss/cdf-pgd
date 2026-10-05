# CDF-PGD: Exact Smooth Penalty Method — Numerical Experiments

Numerical experiments for the exact smooth penalty method (CDF-PGD)
on bounded-rank optimization problems coupled with equality constraints.

The script reproduces the experiments of the CDF-PGD paper across four
applications — spherical data fitting, graph similarity, international
migration stock estimation, and image denoising — and compares CDF-PGD
against the space-decoupling Riemannian baselines SD-RGD and SD-RTR, the
augmented-Lagrangian baseline ALM-PGD, and the factorised equality
formulations EQ1/EQ2 solved by SLSQP and trust-constr.

It implements

- LMSVD, the memory-limited truncated SVD used for every large-scale
  singular value decomposition in the script;
- the four application problems together with their analytic gradients,
  Hessian actions, and constraint-dissolving maps;
- the convergence metrics (stationarity residual and constraint
  violation);
- CDF-PGD (Algorithm 2) and the comparison solvers;
- the figure generation for the convergence triptychs and the denoising
  panel.

Author: Jiaqi Wang (LSEC, AMSS), Nachuan Xiao (SDS, CUHK-SZ), Xin Liu (LSEC, AMSS)

## Requirements

| Package       | Notes                                                          |
| ------------- | ------------------------------------------------------------- |
| Python        | 3.12+                                                         |
| NumPy         | core arrays and linear algebra                                |
| SciPy         | sparse matrices, optimize (SLSQP / trust-constr), linalg      |
| pandas        | reads the UN DESA 2024 migration workbook                     |
| openpyxl      | Excel backend for pandas                                      |
| Matplotlib    | convergence plots and the denoising panel only                |
| scikit-image  | Cameraman test image and the NLM baseline                     |
| bm3d          | BM3D baseline (single-threaded profile)                       |
| threadpoolctl | environment fingerprint (results/environment.json)            |

Exact package versions are recorded in `results/environment.json` at
every run.  BLAS backend: single-threaded; the threading environment
variables are set before NumPy is imported, in the parent process and in
every worker process.

## Setup

From the extracted directory, with Python 3.12:

    python -m pip install -r requirements.txt

## Quick Start

Run a single experiment (spherical data fitting, all four methods):

    python cdf_pgd_experiments.py --exp sphere

Run the full suite (all experiments in order, then the figures):

    python cdf_pgd_experiments.py --exp all

Results are accumulated in `results/results.json`; figures are written
to `figures/`.  Re-running a single experiment only overwrites the
entries it owns, so the remaining entries are preserved.  Use `--resume`
to skip already-completed solves.

## Command-Line Interface

    python cdf_pgd_experiments.py --exp <name> [--resume] [--nproc 1]

| Argument | Type | Description                                                  |
| -------- | ---- | ------------------------------------------------------------ |
| `--exp`  | str  | Required. Experiment name: sphere, graph, eq-sphere, eq-graph, migration, denoise, figures, or all |
| `--resume` | flag | Skip solves that already have a saved result; a source change since the saved run is rejected |
| `--nproc` | int | Number of worker processes. Timed solves run serially, so only 1 is accepted (any other value is a usage error) |

## Experiments

### 1. Spherical data fitting (sphere)

    min ||X - Adata||_F^2   s.t.  rank(X) <= r,  diag(X X^T) = 1_n

Rank-r signal on the oblique manifold with correlated rank-r noise
(nu = 0.01, geometric decay exponent d = 3).  Instances (n, p, r):
(500, 500, 10), (1000, 1000, 25), (1500, 1500, 40), (2000, 2000, 50),
plus a large appended instance (5000, 5000, 50).

### 2. Graph similarity (graph)

    min -tr(X^T L(L(X)))   s.t.  rank(X) <= r,  ||X||_F^2 = 1

where L is the scaled two-sided adjacency operator between two directed
Erdos-Renyi graphs (expected out-degree 6.0).  Same size ladder as the
sphere problem, plus 16 validation instances (2 degrees x 2 sizes x 2
ranks x {independent random pair, directed-cycle pair}), each solved by
all four methods.

### 3. International migration stock estimation (migration)

    min 1/2 ||M * (X - R)||_F^2 + lam/2 ||(1 - M) * (X - F)||_F^2
        s.t.  rank(X) <= r,  X 1_p = b

233 x 233 destination-origin matrix built from the UN DESA 2024 workbook
(9042 positive bilateral entries; 10% held out for testing with seed
2024), ranks r in {3, 5, 10}, lam = 10.  Baselines: IterativeSVD and
SoftImpute (fancyimpute 0.7.0 reproductions, with LMSVD as the truncated
SVD).

### 4. Image denoising (denoise)

    min 1/2 ||X - Icor||_F^2   s.t.  rank(X) <= r,  X_ij = Iclean_ij on Omega

256 x 256 Cameraman with a rank-50 clean reference, Gaussian noise
(nu = 50 on the [0, 255] scale), and three corruption scenarios: one
40 x 40 patch, one 80 x 80 patch, and five 40 x 40 patches.  Solved with
a fixed 2000-step budget.  Baselines: NLM (scikit-image) and BM3D
(single-threaded profile).

### 5. Factorised equality formulations (eq-sphere, eq-graph)

For the sphere and graph instances (n, p, r) = (50, 50, 5), (100, 100, 10),
(200, 200, 20), the rank factorisation X = U V^T is used:

- EQ1: h(U V^T) = 0
- EQ2: h(U V^T) = 0  and  V^T V = I_r

each solved by SLSQP and trust-constr, with CDF-PGD run on the same
instances as the reference.

## Methods

All methods are compared on the same problem instance, under the identical
stopping measure (the distance of -grad f to the normal cone of the
feasible set) and the identical per-solve budget.

### 1. CDF-PGD (proposed, Algorithm 2)

Exact smooth penalty method.  Minimizes the constraint-dissolving penalty

    phi(X) = f(A(X)) + (beta/2) ||h(X)||_F^2

where A is the problem's constraint-dissolving map (h(A(X)) = 0 by
construction) and beta is the penalty parameter.  Iterates stay on the
rank-r manifold through LMSVD rank projections; steps are nonmonotone
Barzilai-Borwein with Armijo backtracking.  When the normal-cone gradient
norm drops below an accelerating threshold eta (initial 1e9, decay factor
0.9), the dissolving map is applied whenever it strictly decreases phi —
driving the iterates toward feasibility without second-order information.

### 2. SD-RGD — Space-Decoupling Riemannian Gradient Descent

Nonmonotone Riemannian gradient descent on the rank manifold, with
alternating BB step sizes and a first-order retraction; metric parameter
omega = 0.5.

### 3. SD-RTR — Space-Decoupling Riemannian Trust Region

Riemannian trust-region method with the analytic Riemannian Hessian,
Steihaug (truncated) CG inner solves, and a second-order retraction;
metric parameter omega = 10.0.

### 4. ALM-PGD — Augmented Lagrangian with Projected Gradient Descent

Augmented Lagrangian outer loop (rho = 1.0) with monotone projected
gradient subproblems (at most 30 inner steps); the subproblem
stationarity tolerance eps decays by a factor 0.1 per outer loop, from
1e-1 down to 1e-8.

### 5. EQ1 / EQ2 — Factorised Equality Formulations

Native SLSQP and trust-constr on the factorised formulations of the
previous section (1000 iterations / 60 s, native tolerance 1e-10),
followed by the common endpoint diagnostic.

Additional baselines for the data problems: IterativeSVD and SoftImpute
for migration; NLM and BM3D for image denoising.

## Running the Full Suite

    python cdf_pgd_experiments.py --exp all

runs, in order:

    sphere -> graph -> eq-sphere -> eq-graph -> migration -> denoise -> figures

The full suite produces 146 records (sphere 20, graph 20, graph validation
64, eq-sphere 15, eq-graph 15, migration 9, denoise 3), the data behind
the paper's 8 standard tables, and 5 PNG figures (4 convergence triptychs
plus the denoising panel).

### Features

- Subprocess isolation: each timed solve runs in its own spawn
  subprocess, and solves run serially (`--nproc 1`).
- Watchdog deadlines: an external watchdog enforces the wall-clock
  budget even during native LAPACK calls; the last completed iterate is
  retained and reported.
- Persistent checkpoints: `results/results.json` is saved after every
  completed solve (atomic rename); `--resume` skips saved keys.
- Cooperative stop: creating the file `STOP_AFTER_CURRENT` stops the run
  after the current checkpoint; remove the file and resume with
  `--resume`.
- Environment fingerprint: `results/environment.json` records the source
  SHA-256, the full configuration, package versions, Python and
  platform; resuming against a different source is rejected.

## Output Format

Every solve is stored under a per-instance key in `results/results.json`,
e.g. `sphere|1000x1000x25|rep0|CDF-PGD`, `migration|r5|CDF-PGD`,
`denoise|small`.

### Common fields

| Field | Description                                                  |
| ----- | ------------------------------------------------------------ |
| status | C converged, T time limit, I iteration limit, L line-search stagnation, F failure, S penalty-stationarity reached (migration), A accepted by the native solver / stationary |
| iters | Number of completed, recorded iterations                     |
| time  | Wall-clock seconds of the solve                              |
| res   | Stationarity residual: distance of -grad f to the normal cone of the feasible set (identical measure for all methods) |
| vio   | Equality violation \|\|h(X)\|\|                              |
| fun   | Objective value at the endpoint                              |
| rank  | Numerical rank of the endpoint (LMSVD)                       |
| hist  | Per-iterate (time, res, vio, iteration) snapshots            |
| svd_diagnostics | LMSVD call count, nonconvergences, maximal relative residual, iteration statistics |

### Problem-specific fields

- migration: rmse, mae, re (relative row-sum error), phi, cdf_residual,
  threshold_trigger_count, acceleration_count and its log,
  algorithm2_trace, input/test SHA-256 digests
- denoise: PSNR of the noisy image and of the CDF-PGD / NLM / BM3D
  outputs on the corrupted pixels, clean-pixel RMSE of CDF-PGD, its
  rank, iterations, and time

Data arrays (migration estimates, denoising panels) are written to
`results/arrays/` as .npz files.

### Convergence plots

`--exp figures` writes four triptychs — stationarity vs. iteration,
stationarity vs. time, and violation vs. iteration for the sphere and
graph instances at sizes 1000 and 2000 — to `figures/`.  The denoising
panel `figures/denoising_effect_rank50.png` is written at the end of
`--exp denoise`.

## Reproducibility

Each configuration is run once with a fixed seed; no averaging is
performed (REPETITIONS = 1).  Timed solves run sequentially with one
BLAS/OpenMP thread.  Times are wall-clock seconds.

| Experiment                            | Iteration limit | Time limit |
| ------------------------------------- | --------------- | ---------- |
| Sphere/Graph, sizes 500-2000          | 300             | 120 s      |
| Sphere/Graph, size 5000               | 300             | 600 s      |
| Factorised equality formulations      | 1000            | 60 s       |
| (incl. CDF-PGD on the same instances) |                 |            |
| Graph validation                      | 300             | 60 s       |
| Migration — CDF-PGD                   | 5000            | none       |
| Migration — IterativeSVD / SoftImpute | 1000            | none       |
| Image CDF-PGD                         | 2000 (fixed)    | none       |

## Key Design Decisions

### Deterministic RNG

Data, initialization, holdout splits and noise are all seeded:
SEED_DATA = 42 (synthetic data), SEED_INIT = 1000 (shared initialization),
MIGRATION_SEED = 2024 (holdout split), DENOISE_SEED = 7 (image noise).
Every instance/method is solved exactly once.

Why: one fixed-seed run per configuration, as prescribed by the
paper's protocol, keeps the suite reproducible and the total wall-clock
cost bounded.

### Single-threaded BLAS

OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS,
VECLIB_MAXIMUM_THREADS, NUMEXPR_NUM_THREADS and BLIS_NUM_THREADS are all
set to 1 before NumPy is imported, in the parent process and in every
worker process spawned by the task pool.

Why: deterministic single-threaded execution; timings are wall-clock
seconds of serial solves and do not depend on the machine's core count.

### Hard wall-clock limits

Timed solves run serially, each in its own spawn subprocess.  A watchdog
outside the worker enforces the deadline even while the worker is inside a
native LAPACK call; at the cutoff the last completed, diagnosed iterate is
retained (status T) and incomplete work is discarded.

Why: a Python worker cannot interrupt a native call, so an in-process
time check alone cannot guarantee the stated budgets.

### LMSVD for every large SVD

lmsvd() supplies all truncated SVDs: the rank projections of the
iterates, spectral norms, polar factors, numerical ranks, and the
truncated SVDs inside the migration baselines.  It keeps a
simultaneous-iteration block plus up to M = 5 history blocks, detects
numerical rank by pivoted QR, and solves the Rayleigh-Ritz problem through
a small projected SVD without forming a Gram matrix.  Unconverged or
nonfinite factors are never returned; a global diagnostics counter tracks
calls, nonconvergences and the maximal relative residual.

Why: keeps memory and per-iteration cost low on the 5000 x 5000
instances while returning verified accuracy (relative residual
tolerance 1e-10).

### Analytic gradients and adjoints

Every problem implements its analytic gradient, Hessian action, and the
Frobenius adjoint of its constraint-dissolving map; no finite differences
are used anywhere.

Why: gradient accuracy at machine precision, required by the
stationarity residual used as the common stopping test.

### Regularised dissolving map for denoising

The image-dissolving map carries a numerical floor sigma ||h(X)||; the
paper's unfloored map is undefined at a rank-deficient feasible matrix
and is not globally smooth there.

### Checkpointing and resume

`results/results.json` is rewritten atomically (temp file + replace) after
every completed solve; `--resume` skips already-saved keys; the file
`STOP_AFTER_CURRENT` stops the suite cooperatively after the current
checkpoint.  `results/environment.json` fingerprints the source (SHA-256),
configuration, package versions, Python and platform; resuming with a
different source is refused.

Why: the full suite takes hours; checkpoints make it interruptible and
resumable without losing completed work or mixing settings.

## Code Structure

    cdf_pgd_experiments.py
    |
    |-- Global configuration           # paths, tolerances, budgets, RNG seeds,
    |                                  #   figure styling
    |-- LMSVD                          # lmsvd, rank_projection, spectral_norm,
    |                                  #   polar, numerical_rank
    |-- Problem classes                # BaseProblem; SphericalFitting,
    |                                  #   GraphSimilarity, Migration,
    |                                  #   ImageDenoising
    |-- Convergence metrics            # res, vio, converged, _tangent_rank_norm
    |-- Budget & checkpointing         # _budget_check, _Run, _hard_limited_task,
    |                                  #   _timed_worker, _load/_save_results
    |-- Environment fingerprint        # write_environment
    |-- CDF-PGD core                   # cdf_value, cdf_grad, _bb_step, cdf_pgd,
    |                                  #   cdf_pgd_image, cdf_pgd_migration
    |-- Space-decoupling geometry      # _sd_factorize, _sd_grad, _sd_retr_*,
    |                                  #   _sd_hess, _sd_tcg
    |-- Comparison solvers             # sd_rgd, sd_rtr, alm_pgd
    |-- Factorised formulations        # make_eq_functions, solve_eq
    |-- Problem generators             # make_spherical, make_graph,
    |                                  #   make_graph_cycle_pair, load_migration,
    |                                  #   make_denoising
    |-- Task pool & experiment drivers # _run_tasks, exp_scale,
    |                                  #   exp_graph_validation, exp_eq,
    |                                  #   exp_migration, exp_denoise
    |-- Figures                        # make_triptych, make_denoise_figure,
    |                                  #   exp_figures
    `-- Entry point                    # EXPERIMENTS registry, ALL_ORDER, main

### Key functions

| Function          | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| lmsvd             | Verified leading singular triplets by simultaneous iteration with up to M history blocks |
| rank_projection   | Numerical rank-r projection and thin factors                 |
| cdf_pgd           | Algorithm 2: exact smooth penalty PGD with BB steps, nonmonotone Armijo line search, and threshold-triggered dissolving updates |
| cdf_pgd_migration | Migration variant with penalty-stationarity stopping and a per-step threshold/acceleration trace |
| cdf_pgd_image     | Image variant with a fixed 2000-step budget                  |
| sd_rgd / sd_rtr   | Space-decoupling Riemannian gradient / trust-region baselines |
| alm_pgd           | Augmented Lagrangian with monotone projected-gradient subproblems |
| solve_eq          | Native SLSQP / trust-constr on EQ1/EQ2                       |
| res / vio         | Stationarity residual and equality violation                 |
| _hard_limited_task | Watchdog-enforced serial execution with checkpointing        |

## Files and data

- `cdf_pgd_experiments.py`   solvers, LMSVD, problem classes, experiment drivers
- `requirements.txt`         Python dependencies
- `data/`                    Migration input data and provenance; see `data/README.md`
- `results/`                 `results.json`, `environment.json`, `arrays/` (generated)
- `figures/`                 PNG figures (generated)
