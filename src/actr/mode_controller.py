"""Reasoning Mode Controller for ACTR — threshold-based mode routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from actr.data import CalibratedReasoningState, ReasoningMode
from actr.config import ACTRConfig, ConfidenceThresholds

__all__ = ["ReasoningModeController", "ReasoningModeEnum", "ModeSelectionResult"]


# Architecture-specified thresholds (distinct from ConfidenceThresholds config)
_ARCH_HIGH_THRESHOLD = 0.85
_ARCH_MEDIUM_THRESHOLD = 0.5


class ReasoningModeEnum(str, Enum):
    """Output modes of the Reasoning Mode Controller.

    These represent the computational intensity tiers applied after
    a calibrated confidence score is produced by the SSU estimator.
    """

    FAST = "fast"
    MODERATE = "moderate"
    SLOW = "slow"

    def description(self) -> str:
        """Human-readable description of the reasoning mode."""
        descriptions = {
            ReasoningModeEnum.FAST: (
                "High confidence (p > 0.85) — single-pass generation with minimal verification."
            ),
            ReasoningModeEnum.MODERATE: (
                "Medium confidence (0.5 < p <= 0.85) — two-pass reasoning with standard verification."
            ),
            ReasoningModeEnum.SLOW: (
                "Low or unknown confidence (p <= 0.5 or out-of-range) — "
                "three-pass reasoning with deep verification and knowledge grounding."
            ),
        }
        return descriptions[self]

    @property
    def n_candidates(self) -> int:
        """Number of candidate reasoning paths explored in this mode."""
        return {
            ReasoningModeEnum.FAST: 1,
            ReasoningModeEnum.MODERATE: 2,
            ReasoningModeEnum.SLOW: 3,
        }[self]

    @property
    def verification_depth(self) -> str:
        """Depth of verification applied in this mode."""
        return {
            ReasoningModeEnum.FAST: "minimal",
            ReasoningModeEnum.MODERATE: "standard",
            ReasoningModeEnum.SLOW: "deep",
        }[self]

    @property
    def uses_knowledge_grounding(self) -> bool:
        """Whether this mode employs knowledge-grounded verification."""
        return self == ReasoningModeEnum.SLOW


@dataclass
class ModeSelectionResult:
    """Result of a single mode selection call.

    Attributes
    ----------
    selected_mode : ReasoningModeEnum
        The routing decision — one of FAST, MODERATE, SLOW.
    confidence : float
        The calibrated probability that triggered this selection.
    confidence_tag : str
        The discrete confidence tag (``"high"``, ``"medium"``, ``"low"``, ``"unknown"``).
    transition_reason : str
        Human-readable explanation of why this mode was selected.
    timestamp : datetime
        When the selection was made.
    previous_mode : ReasoningModeEnum | None
        The mode selected in the previous call, if any.
    is_transition : bool
        Whether the selected mode differs from ``previous_mode``.
    """

    selected_mode: ReasoningModeEnum
    confidence: float
    confidence_tag: str
    transition_reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    previous_mode: Optional[ReasoningModeEnum] = None
    is_transition: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_mode": self.selected_mode.value,
            "confidence": self.confidence,
            "confidence_tag": self.confidence_tag,
            "transition_reason": self.transition_reason,
            "timestamp": self.timestamp.isoformat(),
            "previous_mode": self.previous_mode.value if self.previous_mode else None,
            "is_transition": self.is_transition,
        }


class ReasoningModeController:
    """Threshold-based reasoning mode selector.

    Takes a calibrated probability from the SSU/calibration pipeline and
    routes it to one of three computational intensity tiers:

    - **FAST** — high confidence (p > 0.85), single-pass, minimal verification
    - **MODERATE** — medium confidence (0.5 < p <= 0.85), two-pass, standard verification
    - **SLOW** — low or unknown confidence (p <= 0.5 or out-of-range),
      three-pass, deep verification + knowledge grounding

    Parameters
    ----------
    config : ACTRConfig | None
        ACTR configuration object. If None, the default ``ACTRConfig()`` is used.
    """

    def __init__(self, config: Optional[ACTRConfig] = None) -> None:
        self.config = config if config is not None else ACTRConfig()
        self._thresholds = self.config.thresholds

    def select_mode(
        self,
        confidence: float,
        previous_mode: Optional[ReasoningModeEnum] = None,
    ) -> ModeSelectionResult:
        """Select the reasoning mode for a given calibrated confidence.

        Parameters
        ----------
        confidence : float
            Calibrated probability in [0.0, 1.0] (values outside this range
            are treated as ``"unknown"``).
        previous_mode : ReasoningModeEnum | None
            The mode selected in the previous call, used to detect transitions.

        Returns
        -------
        ModeSelectionResult
            The routing decision with metadata.
        """
        # Determine the discrete confidence tag using ConfidenceThresholds
        confidence_tag = self._thresholds.tag_for_confidence(confidence)

        # Unknown tag (out-of-range confidence) always routes to SLOW.
        if confidence_tag == "unknown":
            selected_mode = ReasoningModeEnum.SLOW
        # Map confidence value to mode using architecture-specified thresholds.
        # The tag is used only for the output field; the threshold values
        # (p > 0.85 for FAST, p > 0.5 for MODERATE, else SLOW) drive the routing.
        elif confidence > _ARCH_HIGH_THRESHOLD:
            selected_mode = ReasoningModeEnum.FAST
        elif confidence > _ARCH_MEDIUM_THRESHOLD:
            selected_mode = ReasoningModeEnum.MODERATE
        else:
            selected_mode = ReasoningModeEnum.SLOW

        is_transition = previous_mode is not None and previous_mode != selected_mode

        # Build a human-readable reason string
        reason_parts = [
            f"confidence={confidence:.3f}",
            f"tag={confidence_tag!r}",
            f"mode={selected_mode.value}",
        ]
        if previous_mode is not None:
            reason_parts.append(f"prev={previous_mode.value}")
        if is_transition:
            reason_parts.append("transition=True")
        transition_reason = ", ".join(reason_parts)

        return ModeSelectionResult(
            selected_mode=selected_mode,
            confidence=confidence,
            confidence_tag=confidence_tag,
            transition_reason=transition_reason,
            timestamp=datetime.now(timezone.utc),
            previous_mode=previous_mode,
            is_transition=is_transition,
        )

    def select_mode_from_state(
        self,
        state: CalibratedReasoningState,
    ) -> ModeSelectionResult:
        """Convenience method that extracts confidence from a ``CalibratedReasoningState``.

        Parameters
        ----------
        state : CalibratedReasoningState
            The reasoning state containing a ``calibrated_confidence`` field.

        Returns
        -------
        ModeSelectionResult
            The routing decision.
        """
        return self.select_mode(state.calibrated_confidence)

    def get_mode_for_confidence_tag(self, tag: str) -> ReasoningModeEnum:
        """Map a confidence tag string to the corresponding mode.

        Parameters
        ----------
        tag : str
            One of ``"high"``, ``"medium"``, ``"low"``, ``"unknown"``.

        Returns
        -------
        ReasoningModeEnum
            FAST for ``"high"``, MODERATE for ``"medium"``, SLOW otherwise.
        """
        mapping = {
            "high": ReasoningModeEnum.FAST,
            "medium": ReasoningModeEnum.MODERATE,
        }
        return mapping.get(tag, ReasoningModeEnum.SLOW)

    def batch_select(
        self,
        confidence_scores: list[float],
    ) -> list[ModeSelectionResult]:
        """Process a list of confidence scores into mode selections.

        Parameters
        ----------
        confidence_scores : list[float]
            List of calibrated probabilities.

        Returns
        -------
        list[ModeSelectionResult]
            One result per input score, in the same order.
        """
        return [self.select_mode(score) for score in confidence_scores]

    def get_transition_summary(
        self,
        results: list[ModeSelectionResult],
    ) -> dict[str, Any]:
        """Summarise mode transitions across a batch of results.

        Parameters
        ----------
        results : list[ModeSelectionResult]
            Results from consecutive ``select_mode`` calls.

        Returns
        -------
        dict[str, Any]
            Summary with keys: ``total_transitions``,
            ``mode_counts`` (dict of mode → count), and
            ``avg_confidence_per_mode`` (dict of mode → mean confidence).
        """
        total_transitions = sum(1 for r in results if r.is_transition)

        mode_counts: dict[str, int] = {}
        mode_confidences: dict[str, list[float]] = {}
        for r in results:
            key = r.selected_mode.value
            mode_counts[key] = mode_counts.get(key, 0) + 1
            mode_confidences.setdefault(key, []).append(r.confidence)

        avg_confidence_per_mode = {
            mode: sum(vals) / len(vals) for mode, vals in mode_confidences.items()
        }

        return {
            "total_transitions": total_transitions,
            "mode_counts": mode_counts,
            "avg_confidence_per_mode": avg_confidence_per_mode,
        }
