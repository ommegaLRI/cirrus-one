"""
deid.events.extractors.base
---------------------------

Pluggable EventExtractor interface.

Event extraction must be deterministic under a seed/config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

import pandas as pd

from deid.core.types import ThermalCubeRef


class MaskStore(Protocol):
    def add_event_masks(self, event_id: str, frame_indices: list[int], masks_yx: list["object"]) -> str: ...
    def finalize(self) -> None: ...


class EventExtractor(ABC):
    """
    Interface: extract events from thermal cube given plate state and alignment.

    Returns:
        (event_catalog_df, event_qc_dict, mask_store)
    """

    @abstractmethod
    def extract(
        self,
        *,
        thermal_ref: ThermalCubeRef,
        plate_state_npz_path: Path,
        plate_state_meta: Dict[str, Any],
        alignment_payload: Dict[str, Any],
        config: Dict[str, Any],
        mask_store: MaskStore,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        raise NotImplementedError