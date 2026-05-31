"""
evaluate_hybrid.py
==================
Evaluate the alpha-gated thermal hybrid on any profile — including unseen ones
(e.g. profile 75) that were never in the training dataset.

For each profile the script reports per-node RMSE for:
    ① LPTN physics alone   (T_phys)
    ② PSO-calibrated LPTN  (T_pso  — same physics, best-fit parameters)
    ③ Alpha-gated hybrid   (T_hybrid = T_pso + alpha * delta_T_hat)

Plot shows all four curves on one figure:
    • Measured (ground truth)
    • Physics (default params)
    • PSO (calibrated params)
    • Hybrid (PSO + MLP correction)

Usage:
    # Evaluate a single unseen profile
    python evaluate_hybrid.py --profiles 75

    # Evaluate several profiles
    python evaluate_hybrid.py --profiles 75,80,81

    # Use pre-built hybrid features for training profiles (fast path)
    python evaluate_hybrid.py --profiles 7,13,17
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    print("PyTorch not installed.  pip install torch")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# local imports
sys.path.insert(0, str(Path(__file__).parent))
from train_hybrid import (
    ResidualGRU, GateNetwork, HybridNormaliser,
    make_sequences, SEQ_LEN,
    N_MLP_FEAT, GATE_FEAT_IDX, N_TARGETS, TARGET_NAMES, N_GATE_FEAT
)
from feature_engineering import build_features, TEMP_NODES, INPUT_COLS

try:
    from thermal_model import DEFAULT_PARAMS
    from calibration import simulate_profile, load_params, load_analytical_init, run_pso
    HAS_CALIB = True
except ImportError:
    HAS_CALIB = False
    print("[WARNING] calibration / thermal_model not found.  "
          "PSO calibration for unseen profiles will be skipped.")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL FROM UNIFIED CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str):
    """Load ResidualMLP + GateNetwork + HybridNormaliser from a single .pt file."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    mlp  = ResidualGRU()
    gate = GateNetwork()
    mlp.load_state_dict(ckpt["mlp_state"])
    gate.load_state_dict(ckpt["gate_state"])
    norm = HybridNormaliser.from_dict(ckpt["normaliser"])
    mlp.eval(); gate.eval()
    return mlp, gate, norm


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-PROFILE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_profile(
    df: pd.DataFrame,
    mlp: ResidualGRU,
    gate: GateNetwork,
    norm: HybridNormaliser,
    pso_params: dict | None = None,
    default_params: dict | None = None,
) -> dict:
    """
    Run physics simulation(s) + MLP correction for one profile.

    Parameters
    ----------
    df             : raw profile DataFrame (must contain INPUT_COLS + TEMP_NODES)
    mlp / gate     : trained models (eval mode)
    norm           : fitted normaliser
    pso_params     : calibrated (PSO) physics parameters  — required for PSO curve
    default_params : default (uncalibrated) physics parameters — for baseline

    Returns
    -------
    dict with keys:
        T_phys   (N,3), T_pso (N,3), T_hybrid (N,3),
        T_meas   (N,3),
        alpha    (N,),
        rmse_phys  {node: float}, rmse_pso {node: float}, rmse_hybrid {node: float}
    """
    assert HAS_CALIB, "calibration module required for evaluate_profile"

    T_meas = df[TEMP_NODES].values.astype(np.float32)   # (N, 3)

    # ── physics with default params ───────────────────────────────────────────
    params_default = default_params if default_params is not None else DEFAULT_PARAMS
    # default physics — kept only for the comparison plot
    T_phys_default = simulate_profile(df, params_default).astype(np.float32)

    # PSO physics — this is what the GRU was trained to correct
    params_pso = pso_params if pso_params is not None else params_default
    T_phys = simulate_profile(df, params_pso).astype(np.float32)
    T_pso  = T_phys  # same array, alias for plot clarity

    r_winding = T_meas[:, 0] - T_phys[:, 0]
    r_tooth   = T_meas[:, 1] - T_phys[:, 1]
    r_pm      = T_meas[:, 2] - T_phys[:, 2]

    X = build_features(df, T_phys,
                    r_winding=r_winding,
                    r_tooth=r_tooth,
                    r_pm=r_pm).astype(np.float32)



    X_norm = norm.transform_X(X)

    X_seq, _ = make_sequences(
        X_norm,
        np.zeros((len(X_norm), 3), dtype=np.float32),
        seq_len=SEQ_LEN,
    )

    Xt = torch.from_numpy(X_seq[:, :, :N_MLP_FEAT])
    rt = torch.from_numpy(X_seq[:, -1, GATE_FEAT_IDX:GATE_FEAT_IDX+N_GATE_FEAT])

    with torch.no_grad():
        delta_hat_n = mlp(Xt).numpy()
        alpha       = gate(rt).numpy()

        print("\nALPHA STATS")
        print("mean :", alpha.mean(axis=0))
        print("max  :", alpha.max(axis=0))
        print("min  :", alpha.min(axis=0))

    delta_hat = norm.inverse_Y(delta_hat_n)

    T_phys_default = T_phys_default[SEQ_LEN:]
    T_phys = T_phys[SEQ_LEN:]
    T_pso  = T_pso[SEQ_LEN:]
    T_meas = T_meas[SEQ_LEN:]
    T_hybrid = T_phys + alpha * delta_hat





    # ── RMSE per node ─────────────────────────────────────────────────────────
    def rmse_dict(T_pred):
        return {name: float(np.sqrt(np.mean((T_pred[:, i] - T_meas[:, i]) ** 2)))
                for i, name in enumerate(TARGET_NAMES)}

    return {
    "T_phys_default": T_phys_default,
    "T_phys":         T_phys,
    "T_pso":          T_pso,
    "T_hybrid":       T_hybrid,
    "T_meas":         T_meas,
    "alpha":          alpha.squeeze(),
    "rmse_phys":      rmse_dict(T_phys),
    "rmse_pso":       rmse_dict(T_pso),
    "rmse_hybrid":    rmse_dict(T_hybrid),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_profile(result: dict, pid: int, dt: float, output_dir: str) -> str:
    """
    4-row figure:
        rows 0-2 : winding / tooth / PM temperature  (measured, physics, PSO, hybrid)
        row 3    : alpha gate signal
    """
    if not HAS_PLT:
        return ""

    t = np.arange(len(result["T_meas"])) * dt / 60.0   # minutes
    node_labels = ["Winding", "Tooth", "PM"]

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    fig.suptitle(f"Profile {pid} — Thermal Hybrid Virtual Sensing",
                 fontsize=13, fontweight="bold")

    for i, (ax, label) in enumerate(zip(axes[:3], node_labels)):
        rmse_phys   = result["rmse_phys"][TARGET_NAMES[i]]
        rmse_pso    = result["rmse_pso"][TARGET_NAMES[i]]
        rmse_hybrid = result["rmse_hybrid"][TARGET_NAMES[i]]

        ax.plot(t, result["T_meas"][:, i],
                color="black",      linewidth=1.8, label="Measured (ground truth)")
        ax.plot(t, result["T_phys_default"][:, i],
                color="#2196F3",    linewidth=1.2, alpha=0.75,
                label=f"Physics (default)  RMSE={rmse_phys:.2f}°C")
        ax.plot(t, result["T_pso"][:, i],
                color="#FF9800",    linewidth=1.2, alpha=0.85,
                label=f"PSO calibrated     RMSE={rmse_pso:.2f}°C")
        ax.plot(t, result["T_hybrid"][:, i],
                color="#E91E63",    linewidth=1.4,
                label=f"Hybrid (Physics+MLP)   RMSE={rmse_hybrid:.2f}°C")

        ax.set_ylabel(f"{label} Temp (°C)")
        ax.set_title(f"{label} Temperature")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    # alpha gate
    ax = axes[3]
    ax.plot(t, result["alpha"], color="#4CAF50", linewidth=1.2)
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Gate α")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylim(0, 1)
    ax.set_title("Alpha gate  (0 = physics only  ·  1 = full MLP correction)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"thermal_hybrid_profile_{pid}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate thermal hybrid on any profile (seen or unseen)"
    )
    parser.add_argument("--model",      default="results/thermal_hybrid_model.pt",
                        help="Unified .pt checkpoint from train_hybrid.py")
    parser.add_argument("--raw-csv",    default="data/measures_v2.csv")
    parser.add_argument("--params-dir", default=None,
                        help="Directory with params_<id>.json files (PSO results). "
                             "If a profile is missing, PSO is run on-the-fly.")
    parser.add_argument("--output-dir", default="results/eval/")
    parser.add_argument("--profiles",   default="75",
                        help="Comma-separated profile IDs to evaluate (default: 75)")
    parser.add_argument("--dt",         type=float, default=0.5,
                        help="Sampling interval in seconds")
    parser.add_argument("--min-speed",  type=float, default=100.0)
    # PSO settings for unseen profiles
    parser.add_argument("--pso-particles",  type=int, default=30)
    parser.add_argument("--pso-iterations", type=int, default=100)
    parser.add_argument("--analytical-init",
                        default="data/analytical_thermal_init.json")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {args.model} ...")
    mlp, gate, norm = load_model(args.model)

    # ── load raw data ─────────────────────────────────────────────────────────
    print(f"Loading {args.raw_csv} ...")
    raw_df = pd.read_csv(args.raw_csv)
    raw_df["profile_id"] = raw_df["profile_id"].astype("Int64")

    pids = [int(p.strip()) for p in args.profiles.split(",")]

    # ── evaluate each profile ─────────────────────────────────────────────────
    rows = []
    for pid in pids:
        print(f"\n{'='*60}")
        print(f"Profile {pid}")
        print(f"{'='*60}")

        profile_raw = raw_df[raw_df["profile_id"] == pid].reset_index(drop=True)
        if len(profile_raw) == 0:
            print(f"  Profile {pid} not found in {args.raw_csv}, skipping.")
            continue

        # filter low-speed rows
        df = profile_raw[abs(profile_raw["motor_speed"]) > args.min_speed
                         ].reset_index(drop=True)
        if len(df) < 20:
            print(f"  Too few samples after speed filter ({len(df)}), skipping.")
            continue

        # ── get PSO params ────────────────────────────────────────────────────
        pso_params = None
        if args.params_dir is not None:
            p = Path(args.params_dir) / f"params_{pid}.json"
            if p.exists() and HAS_CALIB:
                pso_params = load_params(str(p))
                print(f"  Loaded PSO params from {p}")

        if pso_params is None and HAS_CALIB:
            print(f"  No cached PSO params — running PSO calibration ...")
            init = None
            if os.path.exists(args.analytical_init):
                try:
                    init = load_analytical_init(args.analytical_init, pid)
                except Exception as e:
                    print(f"  Warning: analytical init failed: {e}")

            pso_params, score, _ = run_pso(
                [df],
                n_particles=args.pso_particles,
                n_iterations=args.pso_iterations,
                init_params=init,
            )
            print(f"  PSO score: {score:.4f}")

            # save for next time
            if args.params_dir is not None:
                Path(args.params_dir).mkdir(parents=True, exist_ok=True)
                import json
                out_p = Path(args.params_dir) / f"params_{pid}.json"
                with open(out_p, "w") as f:
                    json.dump(pso_params, f, indent=2)
                print(f"  Saved PSO params → {out_p}")

        if not HAS_CALIB:
            print("  [SKIP] calibration module missing — cannot evaluate.")
            continue

        # ── run evaluation ────────────────────────────────────────────────────
        result  = evaluate_profile(df, mlp, gate, norm,
                                   pso_params=pso_params,
                                   default_params=DEFAULT_PARAMS)
        plot_path = plot_profile(result, pid, args.dt, args.output_dir)

        # ── print results ─────────────────────────────────────────────────────
        print(f"\n  {'Node':<10}  {'Physics':>10}  {'PSO':>10}  {'Hybrid':>10}")
        print(f"  {'-'*44}")
        for name in TARGET_NAMES:
            rp = result["rmse_phys"][name]
            rs = result["rmse_pso"][name]
            rh = result["rmse_hybrid"][name]
            imp = (rs - rh) / rs * 100 if rs > 0 else 0.0
            print(f"  {name:<10}  {rp:9.3f}°C  {rs:9.3f}°C  "
                  f"{rh:9.3f}°C  ({imp:+.1f}% vs PSO)")

        if plot_path:
            print(f"\n  Plot saved → {plot_path}")

        rows.append({
            "profile_id": pid,
            **{f"rmse_phys_{n}":   result["rmse_phys"][n]   for n in TARGET_NAMES},
            **{f"rmse_pso_{n}":    result["rmse_pso"][n]    for n in TARGET_NAMES},
            **{f"rmse_hybrid_{n}": result["rmse_hybrid"][n] for n in TARGET_NAMES},
        })

    # ── summary ───────────────────────────────────────────────────────────────
    if not rows:
        print("\nNo profiles evaluated.")
        return

    print(f"\n{'='*60}")
    print("SUMMARY — Mean RMSE across evaluated profiles")
    print(f"{'='*60}")
    summary = pd.DataFrame(rows)
    print(f"\n  {'Node':<10}  {'Physics':>10}  {'PSO':>10}  {'Hybrid':>10}")
    print(f"  {'-'*44}")
    for name in TARGET_NAMES:
        mp = summary[f"rmse_phys_{name}"].mean()
        ms = summary[f"rmse_pso_{name}"].mean()
        mh = summary[f"rmse_hybrid_{name}"].mean()
        print(f"  {name:<10}  {mp:9.3f}°C  {ms:9.3f}°C  {mh:9.3f}°C")

    csv_path = os.path.join(args.output_dir, "thermal_hybrid_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nFull summary saved → {csv_path}")


if __name__ == "__main__":
    main()