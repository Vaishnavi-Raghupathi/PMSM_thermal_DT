"""calibration.py
=================
PSO-based LPTN parameter calibration.
Optimised for Apple Silicon: numba simulation, numpy-only objective,
parallel particle evaluation via joblib threads.

PSO fits on stator_winding + stator_tooth only.
PM is held out as the virtual sensing target.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from pso import pso
from thermal_model import (
    PARAM_KEYS, DEFAULT_PARAMS, DEFAULT_R_S,
    vector_to_params, params_to_vector,
    simulate_numba, df_to_data_array, warmup_numba,
)

TARGET_SCORE_FACTOR = 0.98


def _bounds_list() -> List[Tuple[float, float]]:
    return [
        (50,    5000),   # C_w
        (50,    5000),   # C_s
        (20,    5000),   # C_pm
        (0.02,  1.0),    # R_ws
        (0.02,  5.0),    # R_sa
        (0.02,  5.0),    # R_ps
        (0.5,  10.0),    # R_pa
        (1e-10, 1e-4),   # k_fe
        (1e-10, 1e-4),   # k_pm
    ]


def simulate_profile(
    df_profile: pd.DataFrame,
    params: Dict[str, float],
    dt: float = 0.5,
) -> np.ndarray:
    """Simulate profile. Returns (N,3) array [Tw, Ts, Tpm]."""
    data = df_to_data_array(df_profile.reset_index(drop=True))
    p    = params_to_vector(params)
    return simulate_numba(data, p, dt)


def _make_objective(data_arrays: List[np.ndarray]):
    """
    Return a closure over pre-extracted numpy arrays.
    No pandas inside the hot path — pure numpy + numba.
    Fits winding (col 7) and tooth (col 8) only.
    PM (col 9) is never seen.
    """
    def objective(param_vector: np.ndarray) -> float:
        assert len(param_vector) == len(PARAM_KEYS)
        p = np.asarray(param_vector, dtype=np.float64)

        rmse_list = []
        for data in data_arrays:
            preds   = simulate_numba(data, p, 0.5)
            # winding residual
            rw = preds[:, 0] - data[:, 7]
            # tooth residual
            rs = preds[:, 1] - data[:, 8]
            rmse_w = float(np.sqrt(np.mean(rw**2)))
            rmse_s = float(np.sqrt(np.mean(rs**2)))
            rmse_list.append((rmse_w, rmse_s))

        arr       = np.array(rmse_list)          # (n_profiles, 2)
        mean_rmse = arr.mean(axis=0)             # (2,)
        return float(0.7 * np.max(mean_rmse) + 0.3 * np.mean(mean_rmse))

    return objective


def objective_function(
    param_vector: np.ndarray,
    df_profiles: List[pd.DataFrame],
) -> float:
    """Convenience wrapper used when dataframes are passed directly."""
    data_arrays = [df_to_data_array(df.reset_index(drop=True)) for df in df_profiles]
    return _make_objective(data_arrays)(param_vector)


def run_pso(
    df_profiles: Iterable[pd.DataFrame],
    n_particles: int = 30,
    n_iterations: int = 100,
    dt: float = 0.5,
    init_params: Dict[str, float] | None = None,
) -> Tuple[Dict[str, float], float, List[float]]:

    profiles = list(df_profiles)
    if not profiles:
        raise ValueError("No calibration profiles provided.")

    # warm up numba on first call
    warmup_numba()

    # pre-extract numpy arrays once — no pandas in hot path
    data_arrays = [df_to_data_array(df.reset_index(drop=True)) for df in profiles]
    obj         = _make_objective(data_arrays)

    bounds = _bounds_list()
    lb     = np.array([b[0] for b in bounds], dtype=np.float64)
    ub     = np.array([b[1] for b in bounds], dtype=np.float64)

    print("Objective: 0.7*max(RMSE) + 0.3*mean(RMSE)  [winding + tooth only]")
    print("PM is held out as virtual sensing target.")

    # ── analytical warm-start ─────────────────────────────────────────────────
    analytical_vec   = None
    analytical_score = None

    if init_params is not None:
        analytical_vec   = np.clip(params_to_vector(init_params), lb, ub)
        analytical_score = obj(analytical_vec)
        print(f"Analytical init score: {analytical_score:.6f}")
        base = analytical_vec.copy()
    else:
        base = np.clip(np.array([
            500.0, 1000.0, 500.0,
            0.15,  0.5,    0.2,   3.0,
            1e-7,  1e-7,
        ], dtype=np.float64), lb, ub)

    WARM_SIGMA = np.array([
        100.0, 200.0, 100.0,
        0.05,  0.10,  0.10,  0.50,
        1e-7,  1e-7,
    ], dtype=np.float64)

    init_positions    = base + np.random.randn(n_particles, len(base)) * WARM_SIGMA
    init_positions    = np.clip(init_positions, lb, ub)
    init_positions[0] = base.copy()

    # ── run PSO ───────────────────────────────────────────────────────────────
    pso_vec, pso_score, history = pso(
        obj,
        bounds,
        n_particles=n_particles,
        n_iterations=n_iterations,
        init_positions=init_positions,
    )

    pso_vec = np.clip(pso_vec, lb, ub)

    # ── select best ───────────────────────────────────────────────────────────
    if analytical_vec is not None and analytical_score is not None:
        if pso_score < analytical_score * TARGET_SCORE_FACTOR:
            best_vec, best_score, best_label = pso_vec, pso_score, "PSO result"
        else:
            best_vec   = analytical_vec
            best_score = analytical_score
            best_label = "analytical init (PSO did not improve sufficiently)"
    else:
        best_vec, best_score, best_label = pso_vec, pso_score, "PSO result"

    print(f"Selected: {best_label}  (score={best_score:.6f})")
    return vector_to_params(best_vec), float(best_score), history


def evaluate_params(
    params: Dict[str, float],
    df_profile: pd.DataFrame,
    dt: float = 0.5,
) -> Dict[str, float]:
    """Evaluate on all three nodes. PM RMSE is the virtual sensing result."""
    preds     = simulate_profile(df_profile, params, dt=dt)
    targets   = df_profile[["stator_winding", "stator_tooth", "pm"]].values
    residuals = preds - targets
    rmse      = np.sqrt(np.mean(residuals**2, axis=0))
    mae       = np.mean(np.abs(residuals),    axis=0)

    labels  = ["winding", "stator", "pm"]
    metrics: Dict[str, float] = {}
    for i, label in enumerate(labels):
        metrics[f"rmse_{label}"] = float(rmse[i])
        metrics[f"mae_{label}"]  = float(mae[i])
    metrics["rmse_mean"] = float(rmse.mean())
    metrics["mae_mean"]  = float(mae.mean())
    return metrics


def save_params(params: Dict[str, float], path: str) -> None:
    with open(path, "w") as f:
        json.dump(params, f, indent=2)


def load_params(path: str) -> Dict[str, float]:
    with open(path) as f:
        return json.load(f)


def load_analytical_init(path: str, profile_id: int | str) -> Dict[str, float] | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get(str(profile_id))


if __name__ == "__main__":
    print("calibration.py loaded. Use run_pso(...) to calibrate parameters.")