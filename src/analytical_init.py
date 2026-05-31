"""analytical_init.py
====================
Compute analytical steady-state parameter estimates for the LPTN model.

This uses quasi-steady-state samples (low motor_speed variability) to estimate
thermal resistances via steady-state ODE assumptions. Capacitances are set to
DEFAULT_PARAMS because they are not identifiable from steady-state data.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd

try:
	from thermal_model import DEFAULT_PARAMS
except ModuleNotFoundError:
	from src.thermal_model import DEFAULT_PARAMS


R_S_FIXED = 0.0034
K_FE_FIXED = 1e-7
K_PM_FIXED = 5e-8

EXCLUDED_IDS = {2, 3}


def _rpm_to_rad_per_sec(rpm: float) -> float:
	return rpm * 2.0 * np.pi / 60.0


def _clip(value: float, lower: float, upper: float) -> float:
	return float(np.clip(value, lower, upper))


def filter_quasi_steady_samples(df_profile: pd.DataFrame, window: int = 60) -> pd.DataFrame:
	"""Select quasi-steady-state samples using rolling std of motor_speed.

	Keeps samples where rolling std is below the 20th percentile.
	"""
	speed_std = df_profile["motor_speed"].rolling(window=window, min_periods=window).std()
	threshold = np.nanpercentile(speed_std.to_numpy(), 20)
	mask = speed_std <= threshold
	return df_profile.loc[mask].copy()


def estimate_profile_params(df_profile: pd.DataFrame) -> Dict[str, float]:
	"""Estimate steady-state parameters for a single profile."""
	steady = filter_quasi_steady_samples(df_profile)
	if steady.empty:
		steady = df_profile.copy()

	id_med = float(steady["i_d"].median())
	iq_med = float(steady["i_q"].median())
	ud_med = float(steady["u_d"].median())
	uq_med = float(steady["u_q"].median())
	speed_med = float(steady["motor_speed"].median())
	omega_med = _rpm_to_rad_per_sec(speed_med)

	Tw = float(steady["stator_winding"].median())
	Ts = float(steady["stator_tooth"].median())
	Tpm = float(steady["pm"].median())
	Tamb = float(steady["ambient"].median())

	Q_cu = R_S_FIXED * (id_med ** 2 + iq_med ** 2)
	Q_fe = K_FE_FIXED * omega_med ** 2 * (ud_med ** 2 + uq_med ** 2)
	Q_pm = K_PM_FIXED * omega_med ** 2

	R_ws = _clip((Tw - Ts) / max(Q_cu, 1e-6), 0.02, 0.80)
	R_sa = _clip((Ts - Tamb) / max(Q_cu + Q_fe, 1e-6), 0.02, 0.50)
	R_pa = _clip((Tpm - Tamb) / max(Q_pm, 1e-6), 0.20, 4.00)

	params = {
		"C_w": DEFAULT_PARAMS["C_w"],
		"C_s": DEFAULT_PARAMS["C_s"],
		"C_pm": DEFAULT_PARAMS["C_pm"],
		"R_ws": R_ws,
		"R_sa": R_sa,
		"R_ps": 0.50,
		"R_pa": R_pa,
		"R_s": R_S_FIXED,
		"k_fe": K_FE_FIXED,
		"k_pm": K_PM_FIXED,
	}
	return params


def compute_all_profiles(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
	"""Compute analytical steady-state parameters for all profiles (excluding 2,3)."""
	results: Dict[str, Dict[str, float]] = {}
	for pid in sorted(df["profile_id"].unique()):
		if int(pid) in EXCLUDED_IDS:
			continue
		profile_df = df[df["profile_id"] == pid].reset_index(drop=True)
		results[str(int(pid))] = estimate_profile_params(profile_df)
	return results


def save_params(path: str, data: Dict[str, Dict[str, float]]) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w") as f:
		json.dump(data, f, indent=2)


def main() -> None:
	parser = argparse.ArgumentParser(description="Compute analytical steady-state thermal parameters")
	parser.add_argument("--csv", default="data/measures_v2.csv", help="Path to measures_v2.csv")
	parser.add_argument(
		"--output",
		default="data/analytical_thermal_init.json",
		help="Output JSON path",
	)
	args = parser.parse_args()

	if not os.path.exists(args.csv):
		raise FileNotFoundError(args.csv)

	df = pd.read_csv(args.csv)
	results = compute_all_profiles(df)
	save_params(args.output, results)
	print(f"Saved analytical initial parameters -> {args.output}")


if __name__ == "__main__":
	main()
