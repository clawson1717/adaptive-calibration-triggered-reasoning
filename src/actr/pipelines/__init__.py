"""Pipelines for ACTR reasoning modes.

Each module in this package implements a mode-specific reasoning pipeline:
- ``fast_pipeline`` — Fast Mode (high confidence, single-pass, minimal verification)
- ``moderate_pipeline`` — Moderate Mode (medium confidence, two-pass, standard verification)
- ``slow_pipeline`` — Slow Mode (low confidence, three-pass, deep verification + KG)
"""

from __future__ import annotations

from actr.pipelines.fast_pipeline import (
    FastModePipeline,
    FastPipelineConfig,
)

__all__ = [
    "FastModePipeline",
    "FastPipelineConfig",
]
