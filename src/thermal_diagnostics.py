"""
thermal_diagnostics.py
======================
Numerical diagnostics for the PMSM Thermal Digital Twin.

Mirrors the structure of diagnostics_profile.py (electrical) but
adapted for the 3-node LPTN thermal model.

Codebase context
----------------
- thermal_model.py  : DEFAULT_PARAMS, thermal_ode(), params_to_vector()
- calibration.py    : simulate_profile(), evaluate_params(), run_pso()
- data/measures_v2.csv : Paderborn dataset, 0.5 s sampling

Thermal states
--------------
    T = [T_winding, T_tooth, T_pm]

Goal
----
Determine whether failures on a given profile come from:
  1. Calibration quality         → Diagnostic 1 (default vs calibrated RMSE)
  2. Residual structure          → Diagnostic 2 (residual analysis)
  3. Operating-condition bias    → Diagnostic 3 (residual vs speed / load)
  4. Dynamic segmentation        → Diagnostic 4 (transient vs steady-state)

Usage
-----
    python thermal_diagnostics.py --profile 65
    python thermal_diagnostics.py --profile 65 --params results/params_65.json
    python thermal_diagnostics.py --profile 65 --skip 1 2
"""

import argparse
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import argrelmax

sys.path.insert(0, str(Path(__file__).parent))
try:
    from thermal_model import DEFAULT_PARAMS, thermal_ode, params_to_vector, vector_to_params
    from calibration import simulate_profile, evaluate_params, load_params
except ImportError as exc:
    print(f"[ERROR] Cannot import thermal modules: {exc}")
    print("        Run this script from the project root directory.")
    sys.exit(1)

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("results/thermal_diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
SAMPLE_DT   = 0.5      # seconds per dataset row (2 Hz)
MIN_SPEED   = 100      # rpm — filter near-zero speed rows
TEMP_NODES  = ["stator_winding", "stator_tooth", "pm"]
NODE_LABELS = ["Winding", "Tooth", "PM"]
NODE_COLORS = ["steelblue", "darkorange", "crimson"]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_profile(
    profile_id: int,
    data_path: str = "data/measures_v2.csv",
    n_samples: int = None,
    params: dict = None,
) -> dict:
    """
    Load one profile and run the default + calibrated physics simulation.

    Returns
    -------
    dict with keys:
        t             : (N,) time axis [s]
        df            : raw DataFrame for the profile
        T_meas        : (N, 3) measured temperatures [winding, tooth, pm]
        speed_rpm     : (N,) motor speed [rpm]
        i_sq          : (N,) |i_d|^2 + |i_q|^2  (proxy for copper loss)
        params_default: DEFAULT_PARAMS
        params_cal    : calibrated params (or DEFAULT_PARAMS if not provided)
        T_default     : (N, 3) physics predictions with DEFAULT_PARAMS
        T_cal         : (N, 3) physics predictions with calibrated params
        ss_mask       : (N,) boolean — steady-state region (low dynamic score)
    """
    print(f"  Loading {data_path} ...", end=" ", flush=True)
    df_all = pd.read_csv(data_path)
    df_all["profile_id"] = df_all["profile_id"].astype("Int64")
    raw = df_all[df_all["profile_id"] == profile_id].reset_index(drop=True)
    if len(raw) == 0:
        raise ValueError(f"Profile {profile_id} not found in {data_path}.")

    data = raw[abs(raw["motor_speed"]) > MIN_SPEED].reset_index(drop=True)
    if n_samples is not None:
        data = data.head(n_samples).reset_index(drop=True)
    if len(data) == 0:
        raise ValueError(f"Profile {profile_id}: zero samples above MIN_SPEED={MIN_SPEED}.")
    print(f"{len(data)} samples loaded.")

    t         = np.arange(len(data)) * SAMPLE_DT
    T_meas    = data[TEMP_NODES].values.astype(np.float64)
    speed_rpm = data["motor_speed"].values.astype(np.float64)
    i_sq      = (data["i_d"].values ** 2 + data["i_q"].values ** 2).astype(np.float64)

    params_cal = params if params is not None else DEFAULT_PARAMS

    print("  Running default physics simulation ...", end=" ", flush=True)
    T_default = simulate_profile(data, DEFAULT_PARAMS, dt=SAMPLE_DT)
    print("OK")

    print("  Running calibrated physics simulation ...", end=" ", flush=True)
    T_cal = simulate_profile(data, params_cal, dt=SAMPLE_DT)
    print("OK")

    # steady-state mask: low dynamic score (bottom 25%)
    dyn = _dynamic_score_raw(speed_rpm, data["i_q"].values, data["ambient"].values, t)
    ss_mask = dyn < np.percentile(dyn, 25)

    return dict(
        t=t, df=data,
        T_meas=T_meas, speed_rpm=speed_rpm, i_sq=i_sq,
        params_default=DEFAULT_PARAMS, params_cal=params_cal,
        T_default=T_default, T_cal=T_cal,
        ss_mask=ss_mask,
    )


# ══════════════════════════════════════════════════════════════════════════════
# METRIC / UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """NaN-safe RMSE."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mask = np.isfinite(diff)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean(diff[mask] ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    mask = np.isfinite(diff)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(diff[mask]))


def _dynamic_score_raw(
    speed_rpm: np.ndarray,
    iq: np.ndarray,
    ambient: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Thermal dynamic-activity score per sample.

    score(t) = |d(speed)/dt|_n  +  |d(iq)/dt|_n  +  |d(ambient)/dt|_n

    Each derivative normalised to its 99th-percentile so that fast
    load changes and slow ambient drifts contribute equally.
    """
    def _nd(arr):
        d   = np.abs(np.gradient(arr, t))
        p99 = np.percentile(d, 99) + 1e-12
        return d / p99

    return _nd(speed_rpm) + _nd(iq) + _nd(ambient)


def _dynamic_score(profile: dict) -> np.ndarray:
    return _dynamic_score_raw(
        profile["speed_rpm"],
        profile["df"]["i_q"].values,
        profile["df"]["ambient"].values,
        profile["t"],
    )


def _osc_score(r: np.ndarray) -> float:
    """Oscillation score via normalised autocorrelation. Sinusoid → ~1, noise → ~0."""
    r_valid = r[np.isfinite(r)]
    if len(r_valid) < 20:
        return 0.0
    r_norm = r_valid - r_valid.mean()
    acf    = np.correlate(r_norm, r_norm, mode="full")
    acf    = acf[len(acf) // 2:]
    acf   /= (acf[0] + 1e-12)
    peaks  = argrelmax(np.abs(acf[1:50]), order=2)[0]
    return float(np.abs(acf[1 + peaks[0]])) if len(peaks) > 0 else 0.0


def _time_axis(t: np.ndarray):
    """Return (time_array, xlabel) — use minutes for long profiles."""
    if t[-1] > 120:
        return t / 60, "Time (minutes)"
    return t, "Time (seconds)"


def _print_sep(title: str):
    print("\n" + "═" * 70)
    print(title)
    print("═" * 70)


def _save(fig: plt.Figure, filename: str) -> None:
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 1 — Default vs Calibrated RMSE
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic_1_calibration_quality(profile: dict, profile_id: int) -> None:
    """
    Scientific rationale
    --------------------
    Compares DEFAULT_PARAMS vs calibrated params on every thermal node.

    If calibrated RMSE >> default RMSE → PSO diverged or wrong bounds.
    If both are large → model structure wrong (missing loss term, wrong topology).
    If calibrated is much better on winding but not PM → inter-node coupling
    parameters (R_ps, R_pa) need wider bounds.

    Also computes steady-state vs full-profile RMSE to check whether errors
    concentrate at transients or persist at steady state.
    """
    _print_sep("DIAGNOSTIC 1 — CALIBRATION QUALITY")

    T_meas    = profile["T_meas"]
    T_default = profile["T_default"]
    T_cal     = profile["T_cal"]
    ss_mask   = profile["ss_mask"]

    print(f"\n  {'Node':>10} | {'Default RMSE':>14} {'Cal RMSE':>12} "
          f"{'Default SS':>12} {'Cal SS':>10} {'Improvement':>12}")
    print("  " + "-" * 76)

    for i, (node, label) in enumerate(zip(TEMP_NODES, NODE_LABELS)):
        r_def  = rmse(T_meas[:, i], T_default[:, i])
        r_cal  = rmse(T_meas[:, i], T_cal[:, i])
        r_def_ss = rmse(T_meas[ss_mask, i], T_default[ss_mask, i])
        r_cal_ss = rmse(T_meas[ss_mask, i], T_cal[ss_mask, i])
        improv = (1.0 - r_cal / (r_def + 1e-12)) * 100.0

        def _f(v): return f"{v:12.4f}" if np.isfinite(v) else f"{'NaN':>12}"
        print(f"  {label:>10} | {_f(r_def)} {_f(r_cal)} {_f(r_def_ss)} "
              f"{_f(r_cal_ss)} {improv:>11.1f}%")

    print("\n[INTERPRETATION]")
    for i, label in enumerate(NODE_LABELS):
        r_def = rmse(T_meas[:, i], T_default[:, i])
        r_cal = rmse(T_meas[:, i], T_cal[:, i])
        improv = (1.0 - r_cal / (r_def + 1e-12)) * 100.0
        if r_cal > 5.0:
            print(f"  ⚠ {label}: calibrated RMSE={r_cal:.2f}°C — large residual remains.")
            print(f"     → Check if loss model (Q_fe, Q_pm) includes correct terms.")
        elif improv < 10.0:
            print(f"  △ {label}: calibration improved only {improv:.1f}% — PSO may not have converged.")
        else:
            print(f"  ✓ {label}: calibration improved {improv:.1f}%  (RMSE {r_def:.2f}→{r_cal:.2f}°C)")

    # ── plot ─────────────────────────────────────────────────────────────────
    t     = profile["t"]
    time, xlabel = _time_axis(t)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(f"D1 - Calibration Quality  (Profile {profile_id})",
                 fontsize=13, fontweight="bold")

    for i, (label, color) in enumerate(zip(NODE_LABELS, NODE_COLORS)):
        ax = axes[i]
        ax.plot(time, T_meas[:, i],    color="black",    lw=2.0, label="Measured")
        ax.plot(time, T_default[:, i], color="tab:blue", lw=1.5, ls="--",
                label=f"Default (RMSE={rmse(T_meas[:,i], T_default[:,i]):.2f}°C)")
        ax.plot(time, T_cal[:, i],     color="tab:orange", lw=1.5, ls="-.",
                label=f"Calibrated (RMSE={rmse(T_meas[:,i], T_cal[:,i]):.2f}°C)")
        ax.set_ylabel(f"T_{label}  [°C]", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.35)

    axes[-1].set_xlabel(xlabel, fontsize=11)
    fig.tight_layout()
    _save(fig, f"d1_calibration_quality_profile_{profile_id}.png")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 2 — Residual analysis
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic_2_residuals(profile: dict, profile_id: int) -> None:
    """
    Scientific rationale
    --------------------
    Residuals r(t) = T_measured(t) − T_physics(t) carry a diagnostic signature:

    Failure mode                    Residual pattern
    ──────────────────────────────────────────────────────────────────────
    Parameter bias (wrong R_s, C_w) Constant non-zero mean; flat offset.
                                    |mean| >> std.
    Missing loss term (e.g. iron    Residual tracks speed² — correlated
    loss underestimated)            with operating condition.
    Dynamic model mismatch          Residual small at steady state but
    (wrong thermal capacitance)     spikes during load transients.
    Sensor / dataset issue          Oscillatory residual with fixed freq.

    Uses calibrated params so residuals reflect model structure, not just
    parameter error.
    """
    _print_sep("DIAGNOSTIC 2 — RESIDUAL ANALYSIS")

    T_meas    = profile["T_meas"]
    T_cal     = profile["T_cal"]
    t         = profile["t"]
    time, xlabel = _time_axis(t)

    dyn_score   = _dynamic_score(profile)
    trans_mask  = dyn_score >= np.percentile(dyn_score, 75)
    ss_mask_dyn = ~trans_mask

    print(f"\n  {'Node':>10} | {'mean':>9} {'std':>9} {'max|r|':>9} "
          f"{'osc':>7} {'bias?':>8} {'transient?':>11}")
    print("  " + "-" * 70)

    residuals = {}
    for i, (node, label) in enumerate(zip(TEMP_NODES, NODE_LABELS)):
        r      = T_meas[:, i] - T_cal[:, i]
        residuals[label] = r
        mn     = np.nanmean(r)
        sd     = np.nanstd(r) + 1e-12
        mx     = np.nanmax(np.abs(r))
        osc    = _osc_score(r)

        e_trans = np.nanmean(r[trans_mask]  ** 2) if trans_mask.any()   else 0.0
        e_ss    = np.nanmean(r[ss_mask_dyn] ** 2) if ss_mask_dyn.any()  else 0.0
        trans_ratio  = e_trans / (e_ss + 1e-12)
        bias_score   = abs(mn) / sd
        bias_flag    = "YES" if bias_score > 2.0 else "no"
        trans_flag   = "YES" if trans_ratio > 3.0 else "no"

        print(f"  {label:>10} | {mn:+9.3f} {sd:9.3f} {mx:9.3f} "
              f"{osc:7.2f} {bias_flag:>8} {trans_flag:>11}")

    print("\n[RESIDUAL CLASSIFICATION]")
    for label, r in residuals.items():
        mn   = np.nanmean(r)
        sd   = np.nanstd(r) + 1e-12
        bias_score = abs(mn) / sd

        e_trans = np.nanmean(r[trans_mask]  ** 2) if trans_mask.any()  else 0.0
        e_ss    = np.nanmean(r[ss_mask_dyn] ** 2) if ss_mask_dyn.any() else 0.0
        trans_ratio = e_trans / (e_ss + 1e-12)
        osc = _osc_score(r)

        print(f"\n  {label.upper()}:  bias={bias_score:.2f}  "
              f"transient_ratio={trans_ratio:.2f}  osc={osc:.2f}")
        if bias_score > 2.0:
            print("    → Likely parameter bias (constant offset).")
            print("      Check R_s, C_w, C_s, C_pm — re-run PSO or widen bounds.")
        elif trans_ratio > 3.0:
            print("    → Likely dynamic model mismatch.")
            print("      Residuals peak during load transients.")
            print("      Consider cross-coupling terms or non-linear capacitance.")
        elif osc > 0.5:
            print("    → Possible numerical artefact (oscillatory residual).")
            print("      Check RK4 step size or thermal ODE coupling.")
        else:
            print("    → Broadband residual — inconclusive from heuristics alone.")
            print("      Check D3 (operating condition) and D4 (dynamic segmentation).")

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    def _pr(ax, r, label, color):
        rmin = np.nanmin(r) * 1.2 if np.nanmin(r) < 0 else np.nanmin(r) * 0.8
        rmax = np.nanmax(r) * 1.2
        ax.fill_between(time, rmin, rmax,
                        where=trans_mask, alpha=0.10, color="red",
                        label="High-dynamic region")
        ax.plot(time, r, color=color, lw=1.2, label=f"Residual T_{label}")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.fill_between(time, r, alpha=0.12, color=color)
        ax.set_ylabel(f"Residual T_{label}  [°C]", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.35)

    for i, (label, color) in enumerate(zip(NODE_LABELS, NODE_COLORS)):
        _pr(axes[i], residuals[label], label, color)

    axes[-1].set_xlabel(xlabel, fontsize=11)
    fig.suptitle(f"D2 - Residual Analysis  (Profile {profile_id})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"d2_residuals_profile_{profile_id}.png")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 3 — Residual vs operating conditions
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic_3_operating_condition_bias(profile: dict, profile_id: int) -> None:
    """
    Scientific rationale
    --------------------
    If physics residuals correlate with operating conditions, it means
    the thermal loss model is missing a term or has the wrong functional form.

    Key correlations to check:
      residual vs speed²     → iron/PM loss coefficient (k_fe, k_pm) wrong
      residual vs i_sq       → stator copper loss coefficient (R_s) wrong
      residual vs ambient    → ambient coupling (R_sa, R_pa) wrong
      residual vs T_measured → thermal runaway / non-linear capacitance

    A strong correlation (|ρ| > 0.5) with a specific condition tells you
    exactly which parameter or loss term to fix before training the hybrid.
    """
    _print_sep("DIAGNOSTIC 3 — RESIDUAL vs OPERATING CONDITIONS")

    T_meas = profile["T_meas"]
    T_cal  = profile["T_cal"]
    df     = profile["df"]

    speed_rpm = profile["speed_rpm"]
    speed_sq  = speed_rpm ** 2
    i_sq      = profile["i_sq"]
    ambient   = df["ambient"].values.astype(np.float64)

    cond_names  = ["speed²", "i_d²+i_q²", "ambient", "T_winding_meas"]
    conditions  = [speed_sq, i_sq, ambient, T_meas[:, 0]]

    print(f"\n  Pearson correlation  |r(residual, condition)|")
    print(f"\n  {'Node':>10} | " + "  ".join(f"{c:>14}" for c in cond_names))
    print("  " + "-" * (14 + 16 * len(cond_names)))

    for i, (node, label) in enumerate(zip(TEMP_NODES, NODE_LABELS)):
        r = T_meas[:, i] - T_cal[:, i]
        corrs = []
        for cond in conditions:
            valid = np.isfinite(r) & np.isfinite(cond)
            if valid.sum() > 10:
                rho = np.corrcoef(r[valid], cond[valid])[0, 1]
            else:
                rho = np.nan
            corrs.append(rho)
        row = "  ".join(f"{c:+14.3f}" if np.isfinite(c) else f"{'NaN':>14}"
                        for c in corrs)
        print(f"  {label:>10} | {row}")

    print("\n[INTERPRETATION]")
    print("  |ρ| > 0.5  → strong correlation; that loss term / parameter is likely wrong.")
    print("  |ρ| < 0.2  → condition not driving residual.")

    # ── scatter plots — winding residual vs each condition ────────────────────
    r_w = T_meas[:, 0] - T_cal[:, 0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"D3 - Residual vs Operating Conditions  (Profile {profile_id}, Winding)",
                 fontsize=13, fontweight="bold")

    scatter_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for ax, name, cond, color in zip(
        axes.flat, cond_names, conditions, scatter_colors
    ):
        valid = np.isfinite(r_w) & np.isfinite(cond)
        ax.scatter(cond[valid], r_w[valid], s=5, alpha=0.4, color=color)
        # regression line
        if valid.sum() > 5:
            z = np.polyfit(cond[valid], r_w[valid], 1)
            x_line = np.linspace(cond[valid].min(), cond[valid].max(), 200)
            ax.plot(x_line, np.polyval(z, x_line), color="black", lw=1.5,
                    label=f"slope={z[0]:.3g}")
            rho = np.corrcoef(r_w[valid], cond[valid])[0, 1]
            ax.set_title(f"{name}   ρ={rho:+.3f}", fontsize=11)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel("Residual T_Winding [°C]", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.35)

    fig.tight_layout()
    _save(fig, f"d3_operating_cond_bias_profile_{profile_id}.png")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 4 — Dynamic segmentation
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic_4_dynamic_segmentation(
    profile: dict,
    profile_id: int,
    dynamic_percentile: float = 75,
) -> None:
    """
    Scientific rationale
    --------------------
    The LPTN thermal model assumes time-invariant parameters.  Under rapid
    load changes or speed transients, the effective thermal resistances and
    capacitances may change (e.g. forced convection changes R_sa dynamically).

    By splitting into HIGH_DYNAMIC and LOW_DYNAMIC windows and computing RMSE
    separately, we test whether the fixed-parameter model breaks specifically
    in transient regions:

      LOW_DYNAMIC RMSE  ≈  HIGH_DYNAMIC RMSE  → parameter bias or broadband error.
      HIGH_DYNAMIC RMSE >> LOW_DYNAMIC RMSE   → fixed params inadequate under
                                                transients; hybrid correction needed.

    Dynamic score = |d(speed)/dt|_n + |d(iq)/dt|_n + |d(ambient)/dt|_n
    """
    _print_sep("DIAGNOSTIC 4 — DYNAMIC vs QUASI-STEADY SEGMENTATION")

    T_meas    = profile["T_meas"]
    T_cal     = profile["T_cal"]
    speed_rpm = profile["speed_rpm"]
    t         = profile["t"]
    time, xlabel = _time_axis(t)

    dyn_score = _dynamic_score(profile)
    threshold = np.percentile(dyn_score, dynamic_percentile)
    high_dyn  = dyn_score >= threshold
    low_dyn   = ~high_dyn

    N = len(t)
    print(f"  Dynamic threshold ({dynamic_percentile:.0f}th pct): {threshold:.4f}")
    print(f"  HIGH_DYNAMIC: {high_dyn.sum()}/{N}  ({100*high_dyn.sum()/N:.1f}%)")
    print(f"  LOW_DYNAMIC : {low_dyn.sum()}/{N}   ({100*low_dyn.sum()/N:.1f}%)")

    def seg(mask, meas_col, pred_col):
        if mask.sum() == 0:
            return np.nan
        return rmse(meas_col[mask], pred_col[mask])

    print(f"\n  {'Node':>10} | {'LOW_DYN RMSE':>14} {'HIGH_DYN RMSE':>14} "
          f"{'Ratio H/L':>10}")
    print("  " + "-" * 54)

    ratios = {}
    for i, label in enumerate(NODE_LABELS):
        r_low  = seg(low_dyn,  T_meas[:, i], T_cal[:, i])
        r_high = seg(high_dyn, T_meas[:, i], T_cal[:, i])
        ratio  = r_high / (r_low + 1e-12) if np.isfinite(r_low) else np.nan
        ratios[label] = ratio
        flag = "  ⚠" if (np.isfinite(ratio) and ratio > 3) else ""
        def _f(v): return f"{v:14.4f}" if np.isfinite(v) else f"{'NaN':>14}"
        print(f"  {label:>10} | {_f(r_low)} {_f(r_high)} {ratio:>10.2f}{flag}")

    print("\n[INTERPRETATION]")
    for label, ratio in ratios.items():
        if not np.isfinite(ratio):
            print(f"  {label.upper()}: ratio unavailable.")
            continue
        if ratio > 5.0:
            print(f"  ⚠ {label}: HIGH_DYNAMIC error >> LOW_DYNAMIC  (ratio={ratio:.1f})")
            print(f"     → Fixed-parameter LPTN collapses under load transients.")
            print(f"       Hybrid correction most needed in transient regions.")
        elif ratio > 2.0:
            print(f"  △ {label}: Moderate dynamic degradation  (ratio={ratio:.1f})")
            print(f"     → Consider adaptive R_sa or wider PSO bounds.")
        else:
            print(f"  ✓ {label}: Error uniformly distributed  (ratio={ratio:.1f})")
            print(f"     → Error not dynamics-specific; likely parameter bias.")

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    # panel 1: speed + region overlay
    ax = axes[0]
    spmin, spmax = speed_rpm.min(), speed_rpm.max()
    ax.fill_between(time, spmin, spmax, where=high_dyn,
                    alpha=0.18, color="red",      label="HIGH_DYNAMIC")
    ax.fill_between(time, spmin, spmax, where=low_dyn,
                    alpha=0.10, color="steelblue", label="LOW_DYNAMIC")
    ax.plot(time, speed_rpm, color="black", lw=1.5, label="Speed (rpm)")
    ax.set_ylabel("Speed  [rpm]", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.35)

    # panel 2: dynamic score
    ax = axes[1]
    ax.plot(time, dyn_score, color="darkorange", lw=1.2, label="Dynamic score")
    ax.axhline(threshold, color="red", ls="--", lw=1.5,
               label=f"Threshold ({dynamic_percentile:.0f}th pct = {threshold:.3f})")
    ax.set_ylabel("Dynamic score\n[normalised]", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.35)

    # panel 3: winding residual coloured by region
    ax = axes[2]
    r_w = T_meas[:, 0] - T_cal[:, 0]
    ax.plot(time, r_w, color="lightgray", lw=0.8, zorder=1)
    ax.scatter(time[high_dyn], r_w[high_dyn],
               s=6, color="red",      alpha=0.7, label="HIGH_DYNAMIC residuals", zorder=3)
    ax.scatter(time[low_dyn],  r_w[low_dyn],
               s=4, color="steelblue",alpha=0.5, label="LOW_DYNAMIC residuals",  zorder=2)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("T_Winding residual  [°C]", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.35)

    fig.suptitle(f"D4 - Dynamic Segmentation  (Profile {profile_id})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"d4_dynamic_segmentation_profile_{profile_id}.png")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(profile_id: int) -> None:
    _print_sep(f"THERMAL DIAGNOSTICS COMPLETE - Profile {profile_id}")
    print(f"  Outputs: {OUT_DIR.resolve()}\n")
    print(
        "  Quick-read guide\n"
        "  ────────────────────────────────────────────────────────────────\n"
        "  D1 Calibration quality\n"
        "      Large improvement       → PSO working; residual is model structure.\n"
        "      No improvement          → PSO diverged; check bounds or init.\n"
        "      Node-specific failure   → That node's R or C param needs attention.\n\n"
        "  D2 Residual analysis\n"
        "      Flat offset             → Parameter bias (R_s, R_sa, C_w, etc.).\n"
        "      Spikes at transients    → Dynamic model mismatch; hybrid needed.\n"
        "      Oscillatory             → Numerical RK4 artefact.\n\n"
        "  D3 Operating condition bias\n"
        "      |ρ| > 0.5 vs speed²    → k_fe or k_pm coefficient wrong.\n"
        "      |ρ| > 0.5 vs i_sq      → R_s wrong or temperature-dependent.\n"
        "      |ρ| > 0.5 vs ambient   → R_sa or R_pa wrong.\n\n"
        "  D4 Dynamic segmentation\n"
        "      H/L ratio ≈ 1          → Uniform error; likely parameter bias.\n"
        "      H/L ratio >> 1         → Model breaks under transients;\n"
        "                               hybrid correction most valuable here.\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PMSM Thermal Digital Twin - Diagnostics Module",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile",   type=int, required=True,
                        help="Profile ID to diagnose (e.g. 65)")
    parser.add_argument("--data-path", type=str, default="data/measures_v2.csv")
    parser.add_argument("--params",    type=str, default=None,
                        help="Path to calibrated params JSON (optional)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Max samples to load")
    parser.add_argument("--dyn-pct",   type=float, default=75,
                        help="Percentile threshold for HIGH_DYNAMIC in D4")
    parser.add_argument("--skip",      type=int, nargs="*", default=[],
                        choices=[1, 2, 3, 4],
                        help="Diagnostic numbers to skip (e.g. --skip 1 2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("═" * 70)
    print("PMSM THERMAL DIGITAL TWIN - DIAGNOSTICS MODULE")
    print(f"  Profile: {args.profile}   data: {args.data_path}")
    print("═" * 70)

    params = None
    if args.params is not None:
        print(f"  Loading calibrated params from {args.params} ...")
        params = load_params(args.params)

    try:
        profile = load_profile(
            args.profile,
            data_path=args.data_path,
            n_samples=args.n_samples,
            params=params,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    N = len(profile["t"])
    print(f"  Profile loaded: {N} samples   "
          f"t=[0, {profile['t'][-1]:.1f}] s\n")

    def _run(n, fn, *a, **kw):
        if n in args.skip:
            print(f"\n[Skipping Diagnostic {n}]")
            return
        try:
            fn(*a, **kw)
        except Exception:
            print(f"\n[ERROR] Diagnostic {n} failed:")
            traceback.print_exc()

    _run(1, diagnostic_1_calibration_quality,       profile, args.profile)
    _run(2, diagnostic_2_residuals,                 profile, args.profile)
    _run(3, diagnostic_3_operating_condition_bias,  profile, args.profile)
    _run(4, diagnostic_4_dynamic_segmentation,
         profile, args.profile, args.dyn_pct)

    print_summary(args.profile)


if __name__ == "__main__":
    main()