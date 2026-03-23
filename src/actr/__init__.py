"""Adaptive Calibration Triggered Reasoning (ACTR).

A framework for dynamically selecting reasoning strategies based on
confidence calibration signals from language models.
"""

__version__ = "0.1.0"
__author__ = "Corbin Zhang"

from actr.data import (
    CalibratedReasoningState,
    ReasoningMode,
    ConfidenceTag,
    CalibrationRecord,
    VerificationResult,
)
from actr.config import ACTRConfig
from actr.ssu import (
    ThreeSampleSSU,
    SSUSample,
    SSUConfig,
    SSUResult,
    EmbeddingSimilarity,
    VerbalizedConfidenceExtractor,
)

__all__ = [
    "__version__",
    "CalibratedReasoningState",
    "ReasoningMode",
    "ConfidenceTag",
    "CalibrationRecord",
    "VerificationResult",
    "ACTRConfig",
    "ThreeSampleSSU",
    "SSUSample",
    "SSUConfig",
    "SSUResult",
    "EmbeddingSimilarity",
    "VerbalizedConfidenceExtractor",
]
