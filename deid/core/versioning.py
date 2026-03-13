"""
deid.core.versioning
--------------------

Global pipeline and schema version constants.

This module must remain dependency-light because it is imported
throughout the codebase, including early-stage bootstrap logic.

Rules:
- NEVER import heavy libraries here.
- Version constants must be immutable.
"""

from __future__ import annotations

from typing import Dict

# -------------------------------------------------------------------
# Pipeline version
# -------------------------------------------------------------------

PIPELINE_VERSION: str = "v2"

# -------------------------------------------------------------------
# Schema versions (normative identifiers)
# -------------------------------------------------------------------

SCHEMA_PARTICLE: str = "particle_v1"
SCHEMA_PROCESSED: str = "processed_v1"
SCHEMA_EVENT_CATALOG: str = "event_catalog_v1"
SCHEMA_MATCHED_EVENTS: str = "matched_events_v1"
SCHEMA_SWE_PRODUCTS: str = "swe_products_v1"
SCHEMA_ALIGNMENT: str = "alignment_v2"
SCHEMA_PLATE_STATE: str = "plate_state_v1"
SCHEMA_FINDINGS: str = "findings_v1"

# Central registry so downstream artifacts can embed schema info
SCHEMA_REGISTRY: Dict[str, str] = {
    "particle": SCHEMA_PARTICLE,
    "processed": SCHEMA_PROCESSED,
    "event_catalog": SCHEMA_EVENT_CATALOG,
    "matched_events": SCHEMA_MATCHED_EVENTS,
    "swe_products": SCHEMA_SWE_PRODUCTS,
    "alignment": SCHEMA_ALIGNMENT,
    "plate_state": SCHEMA_PLATE_STATE,
    "findings": SCHEMA_FINDINGS,
}


# -------------------------------------------------------------------
# Public helpers
# -------------------------------------------------------------------

def get_versions_dict() -> Dict[str, object]:
    """
    Return a version dictionary suitable for embedding into artifacts.

    This function intentionally returns only JSON-safe primitives.
    """

    return {
        "pipeline_version": PIPELINE_VERSION,
        "schemas": dict(SCHEMA_REGISTRY),
    }