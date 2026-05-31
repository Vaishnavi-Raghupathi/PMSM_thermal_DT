"""build_hybrid_dataset.py
==========================
Run calibrated LPTN on training profiles and save residuals for MLP training.

Inputs per sample:
    i_d, i_q, u_d, u_q, motor_speed, coolant, ambient,
    Tw_phys, Ts_phys, Tpm_phys, dynamic_score

Targets:
    delta_Tpm = T_pm_measured - T_pm_phys   (PM residual only)

The MLP learns to correct PM prediction only.
Winding and tooth are calibration signals, not MLP targets.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from calibration import simulate_profile, load_analytical_init, run_pso
from thermal_model import DEFAULT_PARAMS


TRAINING_PROFILES = [7, 13, 17, 14, 27, 29, 46, 10, 65, 57]
EXCLUDED_IDS      = {2, 3}
WINDOW_W          = 10


def dynamic_score(df: pd.DataFrame) -> np.ndarray:
    dw = np.gradient(df["motor_speed"].values)
    du = np.gradient(df["u_d"].values)
    dv = np.gradient(df["u_q"].values)
    p99 = lambda x: np.percentile(np.abs(x), 99) + 1e-6
    s = np.abs(dw) / p99(dw) + np.abs(du) / p99(du) + np.abs(dv) / p99(dv)
    return s


def rolling_residual(Tw_phys, Ts_phys, Tpm_phys,
                     Tw_meas, Ts_meas, Tpm_meas, W=10):
    raw = (np.abs(Tw_meas  - Tw_phys) +
           np.abs(Ts_meas  - Ts_phys) +
           np.abs(Tpm_meas - Tpm_phys))
    r_bar = np.convolve(raw, np.ones(W)/W, mode='same')
    return r_bar


def build_profile_features(
    df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    preds = simulate_profile(df, params)
    Tw_phys  = preds[:, 0]
    Ts_phys  = preds[:, 1]
    Tpm_phys = preds[:, 2]

    Tw_meas  = df["stator_winding"].values
    Ts_meas  = df["stator_tooth"].values
    Tpm_meas = df["pm"].values

    s    = dynamic_score(df)
    rbar = rolling_residual(Tw_phys, Ts_phys, Tpm_phys,
                            Tw_meas, Ts_meas, Tpm_meas, WINDOW_W)

    delta_Tpm = Tpm_meas - Tpm_phys

    out = pd.DataFrame({
        "i_d":          df["i_d"].values,
        "i_q":          df["i_q"].values,
        "u_d":          df["u_d"].values,
        "u_q":          df["u_q"].values,
        "motor_speed":  df["motor_speed"].values,
        "coolant":      df["coolant"].values,
        "ambient":      df["ambient"].values,
        "Tw_phys":      Tw_phys,
        "Ts_phys":      Ts_phys,
        "Tpm_phys":     Tpm_phys,
        "dynamic_score": s,
        "r_bar":        rbar,
        "delta_Tpm":    delta_Tpm,
        "profile_id":   df["profile_id"].values,
    })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    default="data/measures_v2.csv")
    parser.add_argument("--params", default="data/calibrated_params.json",
                        help="JSON with calibrated params per profile_id")
    parser.add_argument("--output", default="data/hybrid_dataset.csv")
    parser.add_argument("--n_particles",  type=int, default=30)
    parser.add_argument("--n_iterations", type=int, default=100)
    args = parser.parse_args()

    df_full = pd.read_csv(args.csv)
    df_full = df_full[~df_full["profile_id"].isin(EXCLUDED_IDS)]

    # load pre-calibrated params if available, else run PSO per profile
    if os.path.exists(args.params):
        with open(args.params) as f:
            all_params = json.load(f)
        print(f"Loaded calibrated params from {args.params}")
    else:
        all_params = {}

    frames = []
    for pid in TRAINING_PROFILES:
        if pid not in df_full["profile_id"].unique():
            print(f"Profile {pid} not found, skipping.")
            continue

        profile_df = df_full[df_full["profile_id"] == pid].reset_index(drop=True)
        print(f"\nProcessing profile {pid} ({len(profile_df)} samples)")

        if str(pid) in all_params:
            params = all_params[str(pid)]
            print(f"  Using pre-calibrated params.")
        else:
            init = load_analytical_init("data/analytical_thermal_init.json", pid)
            params, score, _ = run_pso(
                [profile_df],
                n_particles=args.n_particles,
                n_iterations=args.n_iterations,
                init_params=init,
            )
            print(f"  PSO score: {score:.4f}")
            all_params[str(pid)] = params

        feat_df = build_profile_features(profile_df, params)
        frames.append(feat_df)

    dataset = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    dataset.to_csv(args.output, index=False)

    # save params for reuse
    with open(args.params, "w") as f:
        json.dump(all_params, f, indent=2)

    print(f"\nDataset saved: {args.output}  ({len(dataset)} samples)")
    print(f"Params saved:  {args.params}")
    print(f"\nFeature columns: {[c for c in dataset.columns if c != 'delta_Tpm']}")
    print(f"Target column  : delta_Tpm")
    print(f"delta_Tpm stats:\n{dataset['delta_Tpm'].describe().round(3)}")


if __name__ == "__main__":
    main()