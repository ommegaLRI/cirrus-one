"""
deid.core.units
---------------

Unit conversions and validators.

Important:
- These are dimensional conversions only.
- No calibration or density assumptions beyond water equivalence.
"""

from __future__ import annotations

import math
from typing import Optional

from deid.core.errors import SchemaError


# -------------------------------------------------------------------
# Validators
# -------------------------------------------------------------------


def _validate_non_negative(value: Optional[float], name: str) -> None:
    if value is None:
        return
    if math.isnan(value):
        raise SchemaError(f"{name} cannot be NaN")
    if value < 0:
        raise SchemaError(f"{name} must be non-negative", details={"value": value})


# -------------------------------------------------------------------
# SWE / mass conversions
# -------------------------------------------------------------------


def mass_mg_to_swe_mm(mass_mg: float, sensing_area_mm2: float) -> float:
    """
    Convert water mass (mg) to SWE (mm).

    Assumes:
        1 mg water ≈ 1 mm^3 volume
        SWE(mm) = volume(mm^3) / sensing_area(mm^2)
    """

    _validate_non_negative(mass_mg, "mass_mg")
    _validate_non_negative(sensing_area_mm2, "sensing_area_mm2")

    if sensing_area_mm2 <= 0:
        raise SchemaError("sensing_area_mm2 must be > 0")

    return mass_mg / sensing_area_mm2


def swe_mm_to_mass_mg(swe_mm: float, sensing_area_mm2: float) -> float:
    """
    Convert SWE (mm) back to water mass (mg).
    """

    _validate_non_negative(swe_mm, "swe_mm")
    _validate_non_negative(sensing_area_mm2, "sensing_area_mm2")

    return swe_mm * sensing_area_mm2