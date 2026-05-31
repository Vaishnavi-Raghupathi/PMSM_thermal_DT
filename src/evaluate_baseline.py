"""evaluate_baseline.py
=======================
Baseline evaluation for the PMSM thermal digital twin.

PSO calibrates on winding + tooth only.
PM is reported separately as the virtual sensing output.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Tuple
from calibration import evaluate_params, run_pso, load_analytical_init, simulate_profile, save_params

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from thermal_model import DEFAULT_PARAMS


REQUIRED_COLUMNS = [
    "profile_id",
    "i_d", "i_q", "u_d", "u_q",
    "motor_speed", "coolant", "ambient",
    "stator_winding", "stator_tooth", "pm",
]


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df      = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    profile_ids = sorted(df["profile_id"].unique().tolist())
    print("Dataset summary")
    print(f"  Rows       : {len(df)}")
    print(f"  Profiles   : {len(profile_ids)}")
    print(f"  Profile IDs: {profile_ids}")
    return df


def plot_cost_history(cost_history: List[float], outputs_dir: str, profile_id: int) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cost_history, color="tab:blue")
    ax.set_title(f"PSO Convergence (Profile {profile_id})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective score")
    ax.grid(True, alpha=0.3)
    out_path = os.path.join(outputs_dir, f"pso_profile_{profile_id}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def plot_profile_comparison(
    df_profile: pd.DataFrame,
    default_preds: np.ndarray,
    calibrated_preds: np.ndarray,
    outputs_dir: str,
    dt: float,
) -> str:
    t          = np.arange(len(df_profile)) * dt / 60.0
    meas       = df_profile[["stator_winding", "stator_tooth", "pm"]].values
    labels     = ["Winding (calibration)", "Tooth (calibration)", "PM (virtual sensing — not used in PSO)"]
    colors_cal = ["tab:orange", "tab:orange", "tab:red"]
    profile_id = int(df_profile["profile_id"].iloc[0])

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(f"Profile {profile_id} — Physics Baseline", fontsize=12)

    for i, ax in enumerate(axes):
        meas_i = meas[:, i]
        ax.plot(t, meas_i,                    label="Measured",          color="black",      linewidth=1.5)
        ax.plot(t, default_preds[:, i],       label="Physics (Default)", color="tab:blue",   alpha=0.8)
        ax.plot(t, calibrated_preds[:, i],    label="Physics (PSO-cal)", color=colors_cal[i], alpha=0.85)
        ax.set_ylabel(f"{labels[i]} (°C)")
        ax.set_ylim(float(meas_i.min() - 5), float(meas_i.max() + 10))
        ax.grid(True, alpha=0.3)

        rmse_def = float(np.sqrt(np.mean((default_preds[:, i]    - meas_i) ** 2)))
        rmse_cal = float(np.sqrt(np.mean((calibrated_preds[:, i] - meas_i) ** 2)))
        ax.text(
            0.01, 0.95,
            f"RMSE def: {rmse_def:.2f}°C\nRMSE cal: {rmse_cal:.2f}°C",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
        if i == 2:
            ax.text(
                0.5, 0.05,
                "PM was NOT used during PSO calibration — this is a blind prediction",
                transform=ax.transAxes, va="bottom", ha="center", fontsize=8,
                color="darkred",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            )

    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Time (minutes)")
    out_path = os.path.join(outputs_dir, f"profile_{profile_id}_comparison.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",          default="data/measures_v2.csv")
    parser.add_argument("--dt",           type=float, default=0.5)
    parser.add_argument("--profile",      type=int,   default=None)
    parser.add_argument("--samples",      type=int,   default=None)
    parser.add_argument("--n_particles",  type=int,   default=30)
    parser.add_argument("--n_iterations", type=int,   default=100)
    args = parser.parse_args()

    outputs_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    if args.profile is None:
        raise ValueError("Please provide --profile.")

    df = load_dataset(args.csv)
    if args.profile not in df["profile_id"].unique():
        raise ValueError(f"Profile {args.profile} not found.")

    profile_df = df[df["profile_id"] == args.profile].reset_index(drop=True)
    if args.samples is not None:
        profile_df = profile_df.head(args.samples).reset_index(drop=True)
        print(f"Using first {len(profile_df)} samples from profile {args.profile}")

    init_params = load_analytical_init("data/analytical_thermal_init.json", args.profile)
    if init_params is None:
        print("No analytical init found; using default warm-start.")
    else:
        print("Loaded analytical init for warm-start PSO.")

    print(f"\nRunning PSO calibration — profile {args.profile}")
    best_params, best_score, history = run_pso(
        [profile_df],
        n_particles=args.n_particles,
        n_iterations=args.n_iterations,
        dt=args.dt,
        init_params=init_params,
    )

    print(f"\nPSO score: {best_score:.4f}")
    print("\nOptimized parameters")
    for k, v in best_params.items():
        print(f"  {k:<6s}: {v:.6g}")

    default_metrics = evaluate_params(DEFAULT_PARAMS, profile_df, dt=args.dt)
    cal_metrics     = evaluate_params(best_params,    profile_df, dt=args.dt)

    print("\n── Calibration targets (used in PSO) ──────────────────")
    print(f"  winding RMSE : {cal_metrics['rmse_winding']:.4f} °C")
    print(f"  tooth   RMSE : {cal_metrics['rmse_stator']:.4f} °C")
    print("\n── Virtual sensing output (PM — blind prediction) ─────")
    print(f"  PM RMSE default    : {default_metrics['rmse_pm']:.4f} °C")
    print(f"  PM RMSE calibrated : {cal_metrics['rmse_pm']:.4f} °C")

    print("\n── Full comparison table ───────────────────────────────")
    eval_df = pd.DataFrame([
        {"profile_id": args.profile, "baseline": "default",    **default_metrics},
        {"profile_id": args.profile, "baseline": "calibrated", **cal_metrics},
    ])
    print(eval_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    save_params_path = os.path.join("results", f"params_{args.profile}.json")
    os.makedirs("results", exist_ok=True)
    save_params(best_params, save_params_path)
    print(f"\nParams saved: {save_params_path}")


    artifacts = [plot_cost_history(history, outputs_dir, args.profile)]
    default_preds    = simulate_profile(profile_df, DEFAULT_PARAMS, dt=args.dt)
    calibrated_preds = simulate_profile(profile_df, best_params,    dt=args.dt)
    artifacts.append(
        plot_profile_comparison(profile_df, default_preds, calibrated_preds, outputs_dir, args.dt)
    )

    print("\nArtifacts saved")
    for p in artifacts:
        print(f"  {p}")


if __name__ == "__main__":
    main()