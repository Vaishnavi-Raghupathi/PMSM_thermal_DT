"""
feature_engineering.py
=======================
Build the residual dataset for the PMSM thermal hybrid model.

This script replaces build_hybrid_dataset.py entirely.
Output format (.npz with X, Y, profile_ids) matches train_hybrid.py exactly.

For each profile:
  1. Load calibrated physics parameters (or fall back to DEFAULT_PARAMS)
  2. Run LPTN simulation  →  T_physics  (winding, tooth, PM)
  3. Compute residuals    →  Y = T_measured − T_physics
  4. Build feature matrix →  X  (14 columns)

Feature vector (22 features)
─────────────────────────────
  0  motor_speed          [rpm]
  1  i_d                  [A]
  2  i_q                  [A]
  3  u_d                  [V]
  4  u_q                  [V]
  5  ambient              [°C]
  6  coolant              [°C]
  7  T_winding_physics    [°C]
  8  T_tooth_physics      [°C]
  9  T_pm_physics         [°C]
 10  d(speed)/dt          [rpm/s]
 11  d(iq)/dt             [A/s]
 12  dynamic_score        [normalised]
 13  ewma_speed_20        [rpm]
 14  ewma_speed_100       [rpm]
 15  ewma_iq_20           [A]
 16  ewma_iq_100          [A]
 17  ewma_coolant_20      [°C]
 18  ewma_coolant_100     [°C]
 19  ewma_tpm_phys_20     [°C]
 20  ewma_tpm_phys_100    [°C]
 21  rolling_residual_mag [°C]   ← gate input (zero at inference time)

Target vector (3 outputs)
──────────────────────────
  0  r_winding = T_winding_meas − T_winding_physics
  1  r_tooth   = T_tooth_meas   − T_tooth_physics
  2  r_pm      = T_pm_meas      − T_pm_physics

Usage
-----
    python feature_engineering.py
    python feature_engineering.py --profiles 7 17 27 46 65
    python feature_engineering.py --params-dir results/params/
"""

from __future__ import annotations

import json

import argparse
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
try:
    from thermal_model import DEFAULT_PARAMS
    from calibration import simulate_profile, load_params
except ImportError as exc:
    print(f"[ERROR] Cannot import thermal modules: {exc}")
    sys.exit(1)

# ── constants ─────────────────────────────────────────────────────────────────
SAMPLE_DT       = 0.5           # seconds between measurements
MIN_SPEED       = 100           # rpm — filter near-zero speed rows
ROLLING_WINDOW  = 20            # samples for rolling residual magnitude (~10 s)
TEMP_NODES      = ["stator_winding", "stator_tooth", "pm"]
INPUT_COLS      = ["motor_speed", "i_d", "i_q", "u_d", "u_q", "ambient", "coolant"]


DEFAULT_PROFILES = [
    # hard tier (ranks 1-10)
    65, 57, 68,
    # medium-hard (ranks 11-25)
    60, 20, 41, 44, 66, 58,
    # medium (ranks 26-45)
    27, 29, 24, 31, 6,
    # easy (ranks 46-67) — keep some but not dominant
    17, 14, 7,
]


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_dynamic_score(
    speed: np.ndarray,
    iq: np.ndarray,
    ambient: np.ndarray,
    dt: float = SAMPLE_DT,
) -> np.ndarray:
    """
    Normalised dynamic-activity score (3-signal sum, each capped at 99th pctile).

    score = |d(speed)/dt|_n  +  |d(iq)/dt|_n  +  |d(ambient)/dt|_n
    """
    def _nd(arr):
        d   = np.abs(np.gradient(arr)) / dt
        p99 = np.percentile(d, 99) + 1e-12
        return (d / p99).astype(np.float32)

    return _nd(speed) + _nd(iq) + _nd(ambient)


def _rolling_residual_magnitude(
    r_winding: np.ndarray,
    r_tooth: np.ndarray,
    r_pm: np.ndarray,
    window: int = ROLLING_WINDOW,
) -> np.ndarray:
    """
    Causal rolling mean of total residual magnitude over the last `window` samples.

    Note: causal (looks only at past samples) — no future-lookahead.
    At inference time (no measurements) callers pass zeros for all three residuals,
    so this feature is 0 and the gate falls back to a default alpha.
    """
    raw = np.abs(r_winding) + np.abs(r_tooth) + np.abs(r_pm)
    out = np.array([
        raw[max(0, i - window):i + 1].mean()
        for i in range(len(raw))
    ], dtype=np.float32)
    return out


def ewma(x: np.ndarray, span: int) -> np.ndarray:
    """Causal exponentially-weighted moving average (no future lookahead)."""
    alpha = 2 / (span + 1)
    out = np.zeros_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out.astype(np.float32)


def build_features(
    df: pd.DataFrame,
    T_physics: np.ndarray,
    r_winding: np.ndarray | None = None,
    r_tooth: np.ndarray | None   = None,
    r_pm: np.ndarray | None      = None,
    window: int = ROLLING_WINDOW,
) -> np.ndarray:
    """
    Build (N, 22) feature matrix.

    Parameters
    ----------
    df        : profile DataFrame (must contain INPUT_COLS columns)
    T_physics : (N, 3) physics predictions  [winding, tooth, pm]
    r_winding, r_tooth, r_pm : residuals (None at inference time → zeros used)
    window    : rolling window size for residual magnitude feature

    Returns
    -------
    X : (N, 22) float32 array
    """
    speed   = df["motor_speed"].values.astype(np.float32)
    i_d     = df["i_d"].values.astype(np.float32)
    i_q     = df["i_q"].values.astype(np.float32)
    u_d     = df["u_d"].values.astype(np.float32)
    u_q     = df["u_q"].values.astype(np.float32)
    ambient = df["ambient"].values.astype(np.float32)
    coolant = df["coolant"].values.astype(np.float32)

    d_speed = (np.gradient(speed)  / SAMPLE_DT).astype(np.float32)
    d_iq    = (np.gradient(i_q)    / SAMPLE_DT).astype(np.float32)

    dyn_score = _compute_dynamic_score(speed, i_q, ambient)

    # EWMA features (spans: ~10 s and ~50 s at 0.5 s sampling)
    ewma_speed20   = ewma(speed,   20)
    ewma_speed100  = ewma(speed,  100)
    ewma_iq20      = ewma(i_q,    20)
    ewma_iq100     = ewma(i_q,   100)
    ewma_cool20    = ewma(coolant, 20)
    ewma_cool100   = ewma(coolant,100)
    ewma_phys20    = ewma(T_physics[:, 2], 20)
    ewma_phys100   = ewma(T_physics[:, 2],100)

    if r_winding is not None and r_tooth is not None and r_pm is not None:
        def _roll(arr):
            out = np.array([
                np.abs(arr[max(0, i - window):i + 1]).mean()
                for i in range(len(arr))
            ], dtype=np.float32)
            return out
        r_bar_winding = _roll(r_winding)
        r_bar_tooth   = _roll(r_tooth)
        r_bar_pm      = _roll(r_pm)
    else:
        r_bar_winding = np.zeros(len(speed), dtype=np.float32)
        r_bar_tooth   = np.zeros(len(speed), dtype=np.float32)
        r_bar_pm      = np.zeros(len(speed), dtype=np.float32)

    X = np.stack([
        speed,                                     #  0
        i_d,                                       #  1
        i_q,                                       #  2
        u_d,                                       #  3
        u_q,                                       #  4
        ambient,                                   #  5
        coolant,                                   #  6
        T_physics[:, 0].astype(np.float32),        #  7
        T_physics[:, 1].astype(np.float32),        #  8
        T_physics[:, 2].astype(np.float32),        #  9
        d_speed,                                   # 10
        d_iq,                                      # 11
        dyn_score,                                 # 12
        ewma_speed20,                              # 13
        ewma_speed100,                             # 14
        ewma_iq20,                                 # 15
        ewma_iq100,                                # 16
        ewma_cool20,                               # 17
        ewma_cool100,                              # 18
        ewma_phys20,                               # 19
        ewma_phys100,                              # 20
        r_bar_winding,                             # 21  ← gate input winding
        r_bar_tooth,                               # 22  ← gate input tooth
        r_bar_pm,                                  # 23  ← gate input pm
    ], axis=1)



    return X.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# PROFILE PROCESSOR  (runs in worker process)
# ══════════════════════════════════════════════════════════════════════════════

def _process_profile(args_tuple):
    """
    Worker function — one profile at a time.
    Returns (X, Y, pid_arr) or None on failure.
    """
    profile_id, data_path, params_dir = args_tuple

    try:
        df_all = pd.read_csv(data_path)
        df_all["profile_id"] = df_all["profile_id"].astype("Int64")
        raw = df_all[df_all["profile_id"] == profile_id].reset_index(drop=True)
        if len(raw) == 0:
            print(f"  Profile {profile_id}: not found, skipping.")
            return None

        data = raw[abs(raw["motor_speed"]) > MIN_SPEED].reset_index(drop=True)
        if len(data) < 20:
            print(f"  Profile {profile_id}: too few samples ({len(data)}), skipping.")
            return None

        # load calibrated params if available, else default
        # ALWAYS use default params — MLP must learn to correct raw physics, not PSO

        if params_dir is not None:
            with open("data/calibrated_params.json") as f:
                all_params = json.load(f)

            params = all_params[str(profile_id)]


        else:
            raise ValueError("Need --params-dir for PSO residual training.")

        T_physics = simulate_profile(data, params).astype(np.float32)



        
        print(f"  Profile {profile_id}: using DEFAULT_PARAMS (always)")
        T_meas    = data[TEMP_NODES].values.astype(np.float32)          # (N, 3)

        r_winding = T_meas[:, 0] - T_physics[:, 0]
        r_tooth   = T_meas[:, 1] - T_physics[:, 1]
        r_pm      = T_meas[:, 2] - T_physics[:, 2]

        X = build_features(data, T_physics,
                           r_winding=r_winding,
                           r_tooth=r_tooth,
                           r_pm=r_pm)                                   # (N, 22)
        Y = np.stack([r_winding, r_tooth, r_pm], axis=1).astype(np.float32)  # (N, 3)

        valid     = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        n_dropped = int((~valid).sum())
        X, Y      = X[valid], Y[valid]

        if len(X) == 0:
            print(f"  Profile {profile_id}: all rows invalid after NaN filter, skipping.")
            return None

        pid_arr = np.full(len(X), profile_id, dtype=np.int32)
        print(f"  Profile {profile_id}: {len(X)} samples  (dropped {n_dropped})"
              f"  r_w={Y[:,0].mean():+.2f}±{Y[:,0].std():.2f}°C"
              f"  r_t={Y[:,1].mean():+.2f}±{Y[:,1].std():.2f}°C"
              f"  r_pm={Y[:,2].mean():+.2f}±{Y[:,2].std():.2f}°C")
        return X, Y, pid_arr

    except Exception as exc:
        print(f"  Profile {profile_id}: ERROR — {exc}")
        import traceback; traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Build thermal residual dataset for hybrid model"
    )
    parser.add_argument("--data",       default="data/measures_v2.csv")
    parser.add_argument("--out",        default="results/thermal_hybrid_dataset.npz")
    parser.add_argument("--profiles",   type=int, nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--params-dir", type=str, default=None,
                        help="Directory with params_<id>.json calibrated parameter files. "
                             "Run PSO calibration first and save outputs here.")
    parser.add_argument("--workers",    type=int, default=min(8, cpu_count()))
    args = parser.parse_args()

    print(f"Loading {args.data} ...")
    print(f"Profiles  : {args.profiles}")
    print(f"Workers   : {args.workers}")
    print(f"Params dir: {args.params_dir}\n")

    worker_args = [(pid, args.data, args.params_dir) for pid in args.profiles]

    print(f"Running {len(worker_args)} profiles across {args.workers} workers ...\n")
    with Pool(processes=args.workers) as pool:
        results = pool.map(_process_profile, worker_args)

    all_X, all_Y, all_pids = [], [], []
    for r in results:
        if r is not None:
            X, Y, pid_arr = r
            all_X.append(X)
            all_Y.append(Y)
            all_pids.append(pid_arr)

    if not all_X:
        print("No valid profiles processed.  Exiting.")
        return

    X_all   = np.concatenate(all_X,    axis=0)
    Y_all   = np.concatenate(all_Y,    axis=0)
    pid_all = np.concatenate(all_pids, axis=0)

    print(f"\nDataset   : {len(X_all)} samples across {len(all_X)} profiles")
    print(f"X shape   : {X_all.shape}   (22 features)")
    print(f"Y shape   : {Y_all.shape}   (3 targets: winding, tooth, pm)")
    print(f"\nResidual stats (mean ± std):")
    labels = ["winding", "tooth", "pm"]
    for i, lbl in enumerate(labels):
        print(f"  r_{lbl:<8}: {Y_all[:,i].mean():+.3f} ± {Y_all[:,i].std():.3f} °C")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=X_all, Y=Y_all, profile_ids=pid_all)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()