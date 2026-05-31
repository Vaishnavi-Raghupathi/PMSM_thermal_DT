"""thermal_model.py
==================
3-node LPTN for PMSM thermal modelling.
Numba-JIT compiled ODE and RK4 for maximum throughput on Apple Silicon.
"""
from __future__ import annotations

from typing import Dict, List
import numpy as np
from numba import njit


DEFAULT_PARAMS: Dict[str, float] = {
    "C_w":   500.0,
    "C_s":  4000.0,
    "C_pm":  300.0,
    "R_ws":    0.15,
    "R_sa":    0.10,
    "R_ps":    0.80,
    "R_pa":    1.50,
    "R_s":    0.0034,
    "k_fe":   1e-7,
    "k_pm":   5e-8,
}
DEFAULT_R_S: float = DEFAULT_PARAMS["R_s"]

PARAM_KEYS: List[str] = [
    "C_w", "C_s", "C_pm",
    "R_ws", "R_sa", "R_ps", "R_pa",
    "k_fe", "k_pm",
]


# ── numba kernel ──────────────────────────────────────────────────────────────

@njit(cache=True)
def _ode_numba(
    Tw: float, Ts: float, Tpm: float,
    i_d: float, i_q: float, u_d: float, u_q: float,
    motor_speed: float, coolant: float, ambient: float,
    C_w: float, C_s: float, C_pm: float,
    R_ws: float, R_sa: float, R_ps: float, R_pa: float,
    R_s: float, k_fe: float, k_pm: float,
) -> tuple:
    omega = motor_speed * 2.0 * 3.141592653589793 / 60.0
    Q_cu  = R_s  * (i_d**2 + i_q**2)
    Q_fe  = k_fe * omega**2 * (u_d**2 + u_q**2)
    Q_pm  = k_pm * omega**2

    dTw  = (Q_cu - (Tw - Ts)  / R_ws) / C_w
    dTs  = ((Tw - Ts) / R_ws + Q_fe - (Ts - coolant) / R_sa - (Ts - Tpm) / R_ps) / C_s
    dTpm = (Q_pm + (Ts - Tpm) / R_ps - (Tpm - ambient) / R_pa) / C_pm
    return dTw, dTs, dTpm


@njit(cache=True)
def simulate_numba(
    data: np.ndarray,
    p: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    data : (N, 10) float64
           cols 0-6  : i_d, i_q, u_d, u_q, motor_speed, coolant, ambient
           cols 7-9  : stator_winding, stator_tooth, pm  (for t=0 init)
    p    : (9,) float64
           C_w, C_s, C_pm, R_ws, R_sa, R_ps, R_pa, k_fe, k_pm
    """
    N = data.shape[0]
    C_w, C_s, C_pm        = p[0], p[1], p[2]
    R_ws, R_sa, R_ps, R_pa = p[3], p[4], p[5], p[6]
    k_fe, k_pm             = p[7], p[8]
    R_s = 0.0034

    out = np.empty((N, 3), dtype=np.float64)
    Tw  = data[0, 7]
    Ts  = data[0, 8]
    Tpm = data[0, 9]
    out[0, 0] = Tw
    out[0, 1] = Ts
    out[0, 2] = Tpm

    for i in range(1, N):
        i_d         = data[i-1, 0]
        i_q         = data[i-1, 1]
        u_d         = data[i-1, 2]
        u_q         = data[i-1, 3]
        motor_speed = data[i-1, 4]
        coolant     = data[i-1, 5]
        ambient     = data[i-1, 6]

        # RK4
        k1w, k1s, k1p = _ode_numba(
            Tw, Ts, Tpm, i_d, i_q, u_d, u_q, motor_speed, coolant, ambient,
            C_w, C_s, C_pm, R_ws, R_sa, R_ps, R_pa, R_s, k_fe, k_pm)

        k2w, k2s, k2p = _ode_numba(
            Tw + 0.5*dt*k1w, Ts + 0.5*dt*k1s, Tpm + 0.5*dt*k1p,
            i_d, i_q, u_d, u_q, motor_speed, coolant, ambient,
            C_w, C_s, C_pm, R_ws, R_sa, R_ps, R_pa, R_s, k_fe, k_pm)

        k3w, k3s, k3p = _ode_numba(
            Tw + 0.5*dt*k2w, Ts + 0.5*dt*k2s, Tpm + 0.5*dt*k2p,
            i_d, i_q, u_d, u_q, motor_speed, coolant, ambient,
            C_w, C_s, C_pm, R_ws, R_sa, R_ps, R_pa, R_s, k_fe, k_pm)

        k4w, k4s, k4p = _ode_numba(
            Tw + dt*k3w, Ts + dt*k3s, Tpm + dt*k3p,
            i_d, i_q, u_d, u_q, motor_speed, coolant, ambient,
            C_w, C_s, C_pm, R_ws, R_sa, R_ps, R_pa, R_s, k_fe, k_pm)

        Tw  += (dt / 6.0) * (k1w + 2.0*k2w + 2.0*k3w + k4w)
        Ts  += (dt / 6.0) * (k1s + 2.0*k2s + 2.0*k3s + k4s)
        Tpm += (dt / 6.0) * (k1p + 2.0*k2p + 2.0*k3p + k4p)

        out[i, 0] = Tw
        out[i, 1] = Ts
        out[i, 2] = Tpm

    return out


# ── Python-facing helpers ─────────────────────────────────────────────────────

def thermal_ode(
    t: float,
    states: np.ndarray,
    inputs: Dict[str, float],
    params: Dict[str, float],
) -> np.ndarray:
    """Python-facing ODE for diagnostics / single-step use."""
    Tw, Ts, Tpm = states
    dTw, dTs, dTpm = _ode_numba(
        Tw, Ts, Tpm,
        inputs["i_d"], inputs["i_q"], inputs["u_d"], inputs["u_q"],
        inputs["motor_speed"], inputs["coolant"], inputs["ambient"],
        params["C_w"], params["C_s"], params["C_pm"],
        params["R_ws"], params["R_sa"], params["R_ps"], params["R_pa"],
        params.get("R_s", DEFAULT_R_S), params["k_fe"], params["k_pm"],
    )
    return np.array([dTw, dTs, dTpm], dtype=np.float64)


def params_to_vector(params: Dict[str, float]) -> np.ndarray:
    return np.array([params[k] for k in PARAM_KEYS], dtype=np.float64)


def vector_to_params(x: np.ndarray) -> Dict[str, float]:
    p = {k: float(x[i]) for i, k in enumerate(PARAM_KEYS)}
    p["R_s"] = DEFAULT_R_S
    return p


def df_to_data_array(df) -> np.ndarray:
    """Convert profile dataframe to (N,10) numpy array for simulate_numba."""
    import pandas as pd
    cols = ["i_d", "i_q", "u_d", "u_q", "motor_speed", "coolant", "ambient",
            "stator_winding", "stator_tooth", "pm"]
    return df[cols].values.astype(np.float64)


def simulate_single_step(
    states: np.ndarray,
    inputs: Dict[str, float],
    params: Dict[str, float],
    dt: float,
) -> np.ndarray:
    dT = thermal_ode(0.0, states, inputs, params)
    return states + dT * dt


def warmup_numba():
    """Call once at startup to trigger JIT compilation."""
    dummy_data = np.zeros((5, 10), dtype=np.float64)
    dummy_data[:, 7:] = 25.0
    dummy_p = np.array([500., 4000., 300., 0.15, 0.10, 0.80, 1.50, 1e-7, 5e-8])
    simulate_numba(dummy_data, dummy_p, 0.5)


if __name__ == "__main__":
    print("Warming up numba kernels...")
    warmup_numba()
    print("Done.")