"""
deid.swe.calibrators.base
-------------------------

MassCalibrator interface.

Maps event features -> estimated mass (mg).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import pandas as pd


class MassCalibrator(ABC):
    """
    Interface for converting event features into mass estimates.
    """

    calibration_version: str = "unknown"

    @abstractmethod
    def predict_mass(self, event_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Returns:
            (mass_mg, uncertainty_mg)
        """
        raise NotImplementedError