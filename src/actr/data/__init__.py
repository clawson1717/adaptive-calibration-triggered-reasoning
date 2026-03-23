"""Core data structures for ACTR.

Contains CalibratedReasoningState, ReasoningMode, ConfidenceTag,
and supporting types for the adaptive calibration triggered reasoning framework.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

__all__ = [
    "CalibratedReasoningState",
    "ReasoningMode",
    "ConfidenceTag",
    "CalibrationRecord",
    "VerificationResult",
]


class ReasoningMode(str, Enum):
    """Enumeration of available reasoning strategies.

    Each mode represents a different approach to generating and validating
    reasoning chains, with escalating computational cost and typically
    higher accuracy on complex problems.
    """

    DIRECT = "direct"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    SELF_VERIFICATION = "self_verification"
    RECURSIVE_CALIBRATION = "recursive_calibration"

    def __str__(self) -> str:
        return self.value

    @property
    def description(self) -> str:
        """Human-readable description of the reasoning mode."""
        descriptions = {
            ReasoningMode.DIRECT: "Single-pass generation without explicit reasoning steps.",
            ReasoningMode.CHAIN_OF_THOUGHT: "Generates intermediate reasoning steps before the final answer.",
            ReasoningMode.TREE_OF_THOUGHT: "Explores multiple reasoning paths and synthesizes a final answer.",
            ReasoningMode.SELF_VERIFICATION: "Verifies the generated answer by checking consistency with premises.",
            ReasoningMode.RECURSIVE_CALIBRATION: "Recursively applies calibration signals to refine reasoning quality.",
        }
        return descriptions[self]

    @property
    def supports_calibration(self) -> bool:
        """Whether this mode supports confidence calibration feedback."""
        return self in {
            ReasoningMode.CHAIN_OF_THOUGHT,
            ReasoningMode.SELF_VERIFICATION,
            ReasoningMode.RECURSIVE_CALIBRATION,
        }


ConfidenceTag = Literal["high", "medium", "low", "unknown"]
"""A discrete confidence level derived from a calibrated probability.

- ``high``: Calibrated confidence >= high_threshold (typically 0.8+)
- ``medium``: Calibrated confidence between medium and high thresholds
- ``low``: Calibrated confidence below medium threshold
- ``unknown``: Insufficient data to compute a reliable confidence
"""


@dataclass
class CalibrationRecord:
    """A single calibration event in the history of a reasoning session.

    Records the state before and after a calibration adjustment, along
    with the rationale for the adjustment.
    """

    step: int
    input_confidence: float
    output_confidence: float
    adjustment_reason: str
    reasoning_mode: ReasoningMode = ReasoningMode.DIRECT
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def delta(self) -> float:
        """The change in confidence caused by this calibration step."""
        return self.output_confidence - self.input_confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "step": self.step,
            "input_confidence": self.input_confidence,
            "output_confidence": self.output_confidence,
            "adjustment_reason": self.adjustment_reason,
            "reasoning_mode": str(self.reasoning_mode),
            "delta": self.delta,
        }


@dataclass
class VerificationResult:
    """Result of a self-verification check on a reasoning step.

    Verifies that a generated answer is consistent with the premises
    and constraints provided in the original prompt.
    """

    is_verified: bool
    verification_method: str
    consistency_score: float  # 0.0 to 1.0
    failed_checks: list[str] = field(default_factory=list)
    passed_checks: list[str] = field(default_factory=list)
    verification_details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Alias for is_verified for fluent readability."""
        return self.is_verified

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_verified": self.is_verified,
            "verification_method": self.verification_method,
            "consistency_score": self.consistency_score,
            "failed_checks": self.failed_checks,
            "passed_checks": self.passed_checks,
            "verification_details": self.verification_details,
        }


@dataclass
class CalibratedReasoningState:
    """Encapsulates the complete state of an ACTR reasoning session.

    This is the primary data structure returned by the ACTR pipeline.
    It contains the original prompt, generated reasoning, confidence
    scores at various stages of calibration, and metadata about the
    reasoning process.

    Attributes
    ----------
    prompt : str
        The original user prompt that initiated the reasoning.
    reasoning_content : str
        The final generated reasoning / answer text.
    raw_confidence : float
        The model's raw self-reported confidence (0.0 to 1.0), as
        extracted from logits, token probabilities, or an initial
        calibration prompt.
    calibrated_confidence : float
        The confidence score after applying ACTR calibration
        adjustments based on the reasoning mode and verification
        signals. This is the primary output confidence.
    reasoning_mode : ReasoningMode
        The reasoning strategy that was selected and applied.
    confidence_tag : ConfidenceTag
        A discrete confidence category derived from
        ``calibrated_confidence``.
    calibration_history : list[CalibrationRecord]
        Chronological log of every calibration adjustment applied.
    model_name : str | None
        Identifier of the language model used.
    timestamp : datetime
        When the reasoning session began.
    error_flags : list[str]
        Any errors or warnings encountered during reasoning or
        calibration (e.g. "verification_timeout",
        "model_rate_limit").
    metadata : dict[str, Any]
        Arbitrary additional context (e.g. token counts, latency,
        prompt tokens).
    verification_result : VerificationResult | None
        Result of the self-verification step, if applicable.
    reasoning_steps : list[str]
        Intermediate reasoning steps when using chain-of-thought or
        tree-of-thought modes.
    """

    prompt: str
    reasoning_content: str = ""
    raw_confidence: float = 0.0
    calibrated_confidence: float = 0.0
    reasoning_mode: ReasoningMode = ReasoningMode.DIRECT
    confidence_tag: ConfidenceTag = "unknown"
    calibration_history: list[CalibrationRecord] = field(default_factory=list)
    model_name: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    verification_result: Optional[VerificationResult] = None
    reasoning_steps: list[str] = field(default_factory=list)

    def add_calibration_record(
        self,
        step: int,
        input_confidence: float,
        output_confidence: float,
        adjustment_reason: str,
    ) -> None:
        """Append a calibration record to the history.

        Parameters
        ----------
        step : int
            The calibration step number (1-indexed).
        input_confidence : float
            Confidence before this calibration adjustment.
        output_confidence : float
            Confidence after this calibration adjustment.
        adjustment_reason : str
            Human-readable rationale for the adjustment.
        """
        record = CalibrationRecord(
            step=step,
            input_confidence=input_confidence,
            output_confidence=output_confidence,
            adjustment_reason=adjustment_reason,
            reasoning_mode=self.reasoning_mode,
        )
        self.calibration_history.append(record)
        # Keep the state object in sync
        self.calibrated_confidence = output_confidence

    def add_error(self, error_message: str) -> None:
        """Register an error or warning with this reasoning session."""
        if error_message not in self.error_flags:
            self.error_flags.append(error_message)

    @property
    def total_calibration_steps(self) -> int:
        """Number of calibration adjustments applied."""
        return len(self.calibration_history)

    @property
    def is_verified(self) -> bool:
        """Whether the reasoning has passed verification (if attempted)."""
        if self.verification_result is None:
            return False
        return self.verification_result.is_verified

    def to_dict(self) -> dict[str, Any]:
        """Serialize the reasoning state to a dictionary."""
        return {
            "prompt": self.prompt,
            "reasoning_content": self.reasoning_content,
            "raw_confidence": self.raw_confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "reasoning_mode": str(self.reasoning_mode),
            "confidence_tag": self.confidence_tag,
            "calibration_history": [r.to_dict() for r in self.calibration_history],
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "error_flags": self.error_flags,
            "metadata": self.metadata,
            "verification_result": (
                self.verification_result.to_dict()
                if self.verification_result is not None
                else None
            ),
            "reasoning_steps": self.reasoning_steps,
        }
