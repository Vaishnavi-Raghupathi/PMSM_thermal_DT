"""pso.py
=========
Vectorised PSO with parallel particle evaluation.
Optimised for Apple Silicon M-series (joblib threads, numpy vectorisation).
"""
from __future__ import annotations

import time
import numpy as np
from joblib import Parallel, delayed

np.random.seed(42)


def pso(
    objective_func,
    bounds,
    n_particles: int = 30,
    n_iterations: int = 100,
    w_max: float = 0.9,
    w_min: float = 0.4,
    c1: float = 1.5,
    c2: float = 1.5,
    init_positions=None,
    map_func=None,
    patience: int = 10,
    tol: float = 1e-6,
    n_jobs: int = -1,
):
    n_dims = len(bounds)
    lb = np.array([b[0] for b in bounds], dtype=np.float64)
    ub = np.array([b[1] for b in bounds], dtype=np.float64)

    # ── initialise positions ──────────────────────────────────────────────────
    if init_positions is not None:
        positions = np.clip(np.array(init_positions, dtype=np.float64), lb, ub)
    else:
        positions = lb + np.random.rand(n_particles, n_dims) * (ub - lb)

    velocities        = np.zeros((n_particles, n_dims), dtype=np.float64)
    personal_best_pos = positions.copy()

    # ── initial scoring — parallel ────────────────────────────────────────────
    if map_func is not None:
        personal_best_scores = np.array(list(map_func(objective_func, positions)))
    else:
        personal_best_scores = np.array(
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(objective_func)(positions[i]) for i in range(n_particles)
            )
        )

    global_best_idx   = int(np.argmin(personal_best_scores))
    global_best_pos   = personal_best_pos[global_best_idx].copy()
    global_best_score = float(personal_best_scores[global_best_idx])

    history    = []
    no_improve = 0
    prev_best  = global_best_score

    for iteration in range(n_iterations):
        iter_start = time.perf_counter()

        # ── adaptive inertia ──────────────────────────────────────────────────
        w = w_max - (w_max - w_min) * iteration / max(n_iterations - 1, 1)

        # ── vectorised velocity + position update ─────────────────────────────
        r1 = np.random.rand(n_particles, n_dims)
        r2 = np.random.rand(n_particles, n_dims)

        velocities = (
            w * velocities
            + c1 * r1 * (personal_best_pos - positions)
            + c2 * r2 * (global_best_pos   - positions)
        )

        positions = np.clip(positions + velocities, lb, ub)

        # ── parallel scoring ──────────────────────────────────────────────────
        if map_func is not None:
            scores = list(map_func(objective_func, positions))
        else:
            scores = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(objective_func)(positions[i]) for i in range(n_particles)
            )

        scores = np.array(scores, dtype=np.float64)

        # ── update personal and global bests ──────────────────────────────────
        improved = scores < personal_best_scores
        personal_best_scores = np.where(improved, scores, personal_best_scores)
        personal_best_pos[improved] = positions[improved]

        best_idx = int(np.argmin(personal_best_scores))
        if personal_best_scores[best_idx] < global_best_score:
            global_best_score = float(personal_best_scores[best_idx])
            global_best_pos   = personal_best_pos[best_idx].copy()

        history.append(global_best_score)

        if global_best_score < prev_best - tol:
            no_improve = 0
            prev_best  = global_best_score
        else:
            no_improve += 1

        iter_elapsed = time.perf_counter() - iter_start
        print(
            f"Iteration {iteration+1:4d} | Best Score: {global_best_score:.6f}"
            f" | Time: {iter_elapsed:.2f}s"
        )

        if no_improve >= patience:
            print(f"Early stop: no improvement for {patience} iterations")
            break

    return global_best_pos, global_best_score, history