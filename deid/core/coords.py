"""
deid.core.coords
----------------

Coordinate convention helpers.

Thermal cube indexing:
    I[frame, y, x]

Particle table:
    (x, y)

All conversions must pass through this module.
"""

from __future__ import annotations

from typing import Tuple

from deid.core.errors import SchemaError


def xy_to_yx(x: float, y: float) -> Tuple[float, float]:
    """
    Convert particle (x, y) to cube (y, x) order.
    """
    return float(y), float(x)


def validate_pixel(y: float, x: float, H: int, W: int) -> Tuple[int, int]:
    """
    Validate pixel indices are inside cube bounds.

    Returns integer indices.
    """

    yi = int(round(y))
    xi = int(round(x))

    if yi < 0 or yi >= H or xi < 0 or xi >= W:
        raise SchemaError(
            "Pixel coordinate out of bounds",
            details={"y": yi, "x": xi, "H": H, "W": W},
        )

    return yi, xi