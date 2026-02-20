"""
deid.swe.calibrators.energy_linear
----------------------------------

Baseline linear calibrator.

This is intentionally simple and versioned.
"""

from __future__ import annotations

import pandas as pd

from deid.swe.calibrators.base import MassCalibrator


class EnergyLinearCalibrator(MassCalibrator):
    calibration_version = "uncalibrated_v1"

    def __init__(self, a: float = 1.0, b: float = 0.0) -> None:
        self.a = float(a)
        self.b = float(b)

    def predict_mass(self, event_df: pd.DataFrame):
        E = event_df["energy_proxy_E"].fillna(0.0).astype(float)
        mass = self.a * E + self.b
        uncert = 0.25 * mass.abs()  # placeholder uncertainty model
        return mass, uncert