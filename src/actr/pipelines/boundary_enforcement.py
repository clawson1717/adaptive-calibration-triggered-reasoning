"""Box Maze Boundary Enforcement Layer for ACTR.

This module implements the boundary enforcement layer that wraps the existing
mode-specific pipelines (Fast, Moderate, Slow). It applies:
- Inference safety bounds (reject low-confidence outputs)
- Memory grounding gates (trigger grounding when confidence is borderline)
- Envelope propagation (soft constraint decay through reasoning steps)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from actr.data import CalibratedReasoningState, ReasoningMode

__all__ = ["BoundaryEnforcementLayer", "BoundaryEnforcementConfig"]


# Architecture-specified mode boundaries for mode mismatch detection
_MODE_HIGH_THRESHOLD = 0.85
_MODE_MEDIUM_THRESHOLD = 0.5

# Hard limit for reasoning content length
_MAX_REASONING_LENGTH = 50000


@dataclass
class BoundaryEnforcementConfig:
    """Configuration for the Boundary Enforcement Layer.

    Attributes
    ----------
    memory_grounding_threshold : float
        Only trigger memory grounding if calibrated confidence is below this
        threshold. Default: 0.7.
    safety_reject_threshold : float
        Reject (return error state) if calibrated confidence is below this
        threshold. Default: 0.3.
    envelope_decay_factor : float
        Decay factor applied per step during envelope propagation.
        Default: 0.95.
    max_envelope_propagation_steps : int
        Maximum number of steps for envelope spread/propagation.
        Default: 5.
    enable_memory_grounding : bool
        Whether to enable the memory grounding gate. Default: True.
    enable_safety_bounds : bool
        Whether to enable safety bounds checking. Default: True.
    enable_envelope_propagation : bool
        Whether to enable envelope propagation. Default: True.
    """

    memory_grounding_threshold: float = 0.7
    safety_reject_threshold: float = 0.3
    envelope_decay_factor: float = 0.95
    max_envelope_propagation_steps: int = 5
    enable_memory_grounding: bool = True
    enable_safety_bounds: bool = True
    enable_envelope_propagation: bool = True


class BoundaryEnforcementLayer:
    """Boundary enforcement layer wrapping mode-specific reasoning pipelines.

    This layer applies safety bounds, memory grounding gates, and envelope
    propagation to the output of a mode-specific pipeline (Fast, Moderate,
    or Slow). It operates on a pre-populated ``CalibratedReasoningState``
    and modifies it in place.

    Parameters
    ----------
    config : BoundaryEnforcementConfig | None
        Layer configuration. If None, defaults are used.
    """

    def __init__(
        self,
        config: BoundaryEnforcementConfig | None = None,
    ) -> None:
        self.config = config if config is not None else BoundaryEnforcementConfig()

    def run(
        self,
        state: CalibratedReasoningState,
        mode_result: Any = None,
    ) -> CalibratedReasoningState:
        """Apply boundary enforcement to a reasoning state.

        Applies the following checks/gates in order:
        1. Safety bounds — reject if confidence below ``safety_reject_threshold``
        2. Memory grounding gate — activate grounding if confidence below
           ``memory_grounding_threshold``
        3. Envelope propagation — decay soft constraints through reasoning steps

        The state is modified in place and also returned.

        Parameters
        ----------
        state : CalibratedReasoningState
            The reasoning state produced by a mode-specific pipeline.
            Must have ``calibrated_confidence`` already set.
        mode_result : Any, optional
            Unused. Present for interface compatibility with mode pipelines.

        Returns
        -------
        CalibratedReasoningState
            The same (modified) state object.
        """
        # Step 1: Safety bounds check (highest priority — reject early)
        if self.config.enable_safety_bounds:
            self._apply_safety_bounds(state)

        # If already rejected by safety bounds, stop early
        if "safety_rejected" in state.error_flags:
            return state

        # Step 2: Memory grounding gate
        if self.config.enable_memory_grounding:
            self._apply_memory_grounding_gate(state)

        # Step 3: Envelope propagation
        if self.config.enable_envelope_propagation:
            self._apply_envelope_propagation(state)

        # Step 4: Check and record boundary violations
        violations = self._check_boundaries(state)
        if violations:
            for violation in violations:
                state.add_error(violation)

        return state

    def _apply_safety_bounds(self, state: CalibratedReasoningState) -> None:
        """Apply safety bounds: reject if confidence is too low.

        If ``calibrated_confidence < safety_reject_threshold``, the state
        is cleared and ``"safety_rejected"`` is added to error flags.

        Parameters
        ----------
        state : CalibratedReasoningState
            The state to check and (possibly) reject.
        """
        if state.calibrated_confidence < self.config.safety_reject_threshold:
            state.reasoning_content = ""
            state.add_error("safety_rejected")
            state.calibration_history.append(
                _make_calibration_record(
                    step=len(state.calibration_history) + 1,
                    input_confidence=state.calibrated_confidence,
                    output_confidence=0.0,
                    adjustment_reason="safety_rejected: confidence below safety threshold",
                    reasoning_mode=state.reasoning_mode,
                )
            )

    def _apply_memory_grounding_gate(self, state: CalibratedReasoningState) -> None:
        """Apply memory grounding gate when confidence is borderline.

        If ``calibrated_confidence < memory_grounding_threshold``, adds a
        note to metadata indicating that memory grounding was activated.
        Also appends a calibration record.

        Parameters
        ----------
        state : CalibratedReasoningState
            The state to annotate with grounding activation.
        """
        if state.calibrated_confidence < self.config.memory_grounding_threshold:
            state.metadata["memory_grounding_activated"] = True
            state.calibration_history.append(
                _make_calibration_record(
                    step=len(state.calibration_history) + 1,
                    input_confidence=state.calibrated_confidence,
                    output_confidence=state.calibrated_confidence,
                    adjustment_reason="memory_grounding_activated: confidence below grounding threshold",
                    reasoning_mode=state.reasoning_mode,
                )
            )

    def _apply_envelope_propagation(self, state: CalibratedReasoningState) -> None:
        """Apply envelope propagation (soft constraint decay).

        Propagates soft constraints through ``max_envelope_propagation_steps``.
        Each step reduces envelope strength by ``envelope_decay_factor``.
        Results are stored in state metadata.

        Parameters
        ----------
        state : CalibratedReasoningState
            The state to update with envelope propagation data.
        """
        result = self._propagate_envelope(state)
        state.metadata["envelope_steps"] = result["steps"]
        state.metadata["envelope_strength"] = result["final_strength"]

    def _check_boundaries(self, state: CalibratedReasoningState) -> list[str]:
        """Check for boundary violations in the reasoning state.

        Checks performed:
        - ``confidence_out_of_range``: raw_confidence or calibrated_confidence
          outside [0.0, 1.0]
        - ``mode_mismatch``: reasoning_mode does not match expected mode
          derived from calibrated_confidence
        - ``reasoning_too_long``: reasoning_content exceeds 50000 characters

        Parameters
        ----------
        state : CalibratedReasoningState
            The state to check.

        Returns
        -------
        list[str]
            List of violation names (empty if no violations).
        """
        violations = []

        # Check confidence range
        if not (0.0 <= state.raw_confidence <= 1.0):
            violations.append("confidence_out_of_range")
        if not (0.0 <= state.calibrated_confidence <= 1.0):
            violations.append("confidence_out_of_range")

        # Check mode mismatch
        expected_mode = _expected_reasoning_mode(state.calibrated_confidence)
        if state.reasoning_mode != expected_mode:
            violations.append("mode_mismatch")

        # Check reasoning length
        if len(state.reasoning_content) > _MAX_REASONING_LENGTH:
            violations.append("reasoning_too_long")

        return violations

    def _propagate_envelope(self, state: CalibratedReasoningState) -> dict:
        """Propagate soft constraints through reasoning steps.

        Starting from strength 1.0, each step decays the envelope by
        ``envelope_decay_factor``. Propagation stops after
        ``max_envelope_propagation_steps`` steps or when strength
        becomes negligible (< 1e-6).

        Parameters
        ----------
        state : CalibratedReasoningState
            Unused. Present for interface completeness.

        Returns
        -------
        dict
            A dict with keys ``"steps"`` (int) and ``"final_strength"`` (float).
        """
        strength = 1.0
        steps = 0
        decay = self.config.envelope_decay_factor

        for step in range(1, self.config.max_envelope_propagation_steps + 1):
            strength *= decay
            steps = step
            if strength < 1e-6:
                break

        return {"steps": steps, "final_strength": strength}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expected_reasoning_mode(confidence: float) -> ReasoningMode:
    """Derive the expected reasoning mode from a confidence value.

    Uses architecture-specified thresholds:
    - > 0.85 → ReasoningMode.DIRECT (fast/high-confidence path)
    - 0.5–0.85 → ReasoningMode.CHAIN_OF_THOUGHT (moderate)
    - ≤ 0.5 → ReasoningMode.TREE_OF_THOUGHT (slow/low-confidence)

    Parameters
    ----------
    confidence : float
        Calibrated confidence in [0.0, 1.0].

    Returns
    -------
    ReasoningMode
        The expected reasoning mode for the given confidence.
    """
    if confidence > _MODE_HIGH_THRESHOLD:
        return ReasoningMode.DIRECT
    elif confidence > _MODE_MEDIUM_THRESHOLD:
        return ReasoningMode.CHAIN_OF_THOUGHT
    else:
        return ReasoningMode.TREE_OF_THOUGHT


def _make_calibration_record(
    step: int,
    input_confidence: float,
    output_confidence: float,
    adjustment_reason: str,
    reasoning_mode: ReasoningMode,
):
    """Create a CalibrationRecord for boundary enforcement events.

    Parameters
    ----------
    step : int
        Calibration step number.
    input_confidence : float
        Confidence before adjustment.
    output_confidence : float
        Confidence after adjustment.
    adjustment_reason : str
        Human-readable reason for the calibration event.
    reasoning_mode : ReasoningMode
        The reasoning mode active during this event.

    Returns
    -------
    CalibrationRecord
        A new calibration record instance.
    """
    # Local import to avoid circular issues at module level
    from actr.data import CalibrationRecord
    from datetime import datetime, timezone

    return CalibrationRecord(
        step=step,
        input_confidence=input_confidence,
        output_confidence=output_confidence,
        adjustment_reason=adjustment_reason,
        reasoning_mode=reasoning_mode,
        timestamp=datetime.now(timezone.utc),
    )
