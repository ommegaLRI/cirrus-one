"""
deid.core.errors
----------------

Typed exception hierarchy for DEID service.

All pipeline stages should raise subclasses of DEIDError so that
errors can be serialized into structured reports and provenance logs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class DEIDError(Exception):
    """
    Base class for all DEID exceptions.

    Attributes
    ----------
    code : str
        Stable machine-readable error code.
    message : str
        Human-readable description.
    details : dict
        Optional structured metadata.
    """

    DEFAULT_CODE = "DEID_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.code: str = code or self.DEFAULT_CODE
        self.message: str = message
        self.details: Dict[str, Any] = details or {}

        super().__init__(self._format())

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _format(self) -> str:
        if self.details:
            return f"[{self.code}] {self.message} | details={self.details}"
        return f"[{self.code}] {self.message}"

    # ---------------------------------------------------------------
    # Public serialization
    # ---------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to JSON-serializable structure.
        """
        return {
            "type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


# -------------------------------------------------------------------
# Ingest / schema layer
# -------------------------------------------------------------------


class IngestError(DEIDError):
    DEFAULT_CODE = "INGEST_ERROR"


class SchemaError(DEIDError):
    DEFAULT_CODE = "SCHEMA_ERROR"


# -------------------------------------------------------------------
# Alignment / timebase
# -------------------------------------------------------------------


class AlignmentError(DEIDError):
    DEFAULT_CODE = "ALIGNMENT_ERROR"


# -------------------------------------------------------------------
# Storage / IO
# -------------------------------------------------------------------


class ArtifactIOError(DEIDError):
    DEFAULT_CODE = "ARTIFACT_IO_ERROR"


# -------------------------------------------------------------------
# Configuration / validation
# -------------------------------------------------------------------


class ConfigError(DEIDError):
    DEFAULT_CODE = "CONFIG_ERROR"