"""Fast Mode reasoning pipeline for ACTR.

This module implements the FAST reasoning pipeline — single-pass generation
with minimal verification for high-confidence queries.

Architecture
-----------
FAST mode is selected when calibrated confidence > 0.85. It uses:
- n_candidates=1 (single-pass generation)
- verification_depth="minimal"
- uses_knowledge_grounding=False

The pipeline takes a pre-computed calibrated probability and produces
a response with minimal overhead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from actr.data import CalibratedReasoningState, ReasoningMode, ConfidenceTag
from actr.mode_controller import ModeSelectionResult, ReasoningModeEnum
from actr.config import ACTRConfig

__all__ = ["FastModePipeline", "FastPipelineConfig"]

# Architecture-specified thresholds for confidence tag derivation
_ARCH_HIGH_THRESHOLD = 0.85
_ARCH_MEDIUM_THRESHOLD = 0.5


@dataclass
class FastPipelineConfig:
    """Configuration for the Fast Mode pipeline.

    Attributes
    ----------
    max_response_length : int
        Maximum allowed response length in characters. Responses exceeding
        this are considered invalid and trigger an error flag.
    min_response_length : int
        Minimum required response length in characters. Responses shorter
        than this are considered invalid (too short to be meaningful).
    stub_response_template : str
        Template string used for mock/stub responses when no real LLM
        is available. The placeholder ``{prompt}`` is replaced with the
        actual prompt.
    """

    max_response_length: int = 10000
    min_response_length: int = 10
    stub_response_template: str = "[FAST response for: {prompt}]"


class FastModePipeline:
    """Fast Mode reasoning pipeline.

    Takes a calibrated probability (already computed by the SSU) and a prompt,
    produces a response using single-pass generation with minimal verification.

    This pipeline is intentionally lightweight — it does not use knowledge
    grounding, multi-candidate ranking, or deep verification. Its purpose
    is to respond quickly when the system has high confidence in the answer.

    Parameters
    ----------
    config : FastPipelineConfig | None
        Pipeline configuration. If None, defaults are used.
    actr_config : ACTRConfig | None
        Global ACTR configuration. If None, the default ``ACTRConfig()`` is used.
    """

    def __init__(
        self,
        config: Optional[FastPipelineConfig] = None,
        actr_config: Optional[ACTRConfig] = None,
    ) -> None:
        self.config = config if config is not None else FastPipelineConfig()
        self.actr_config = actr_config if actr_config is not None else ACTRConfig()

    def run(
        self,
        prompt: str,
        calibrated_confidence: float,
        mode_result: ModeSelectionResult,
    ) -> CalibratedReasoningState:
        """Execute the Fast Mode pipeline for a given prompt.

        This is the main entry point. It:
        1. Validates the mode result is FAST
        2. Generates a response using single-pass generation (stubbed)
        3. Applies shallow heuristic checks on the response
        4. Returns a fully-populated CalibratedReasoningState

        Parameters
        ----------
        prompt : str
            The original user prompt.
        calibrated_confidence : float
            The pre-computed calibrated probability from the SSU/calibration
            pipeline (already in [0.0, 1.0]).
        mode_result : ModeSelectionResult
            The result of the mode selection step, which must have
            ``selected_mode == ReasoningModeEnum.FAST``.

        Returns
        -------
        CalibratedReasoningState
            A fully-populated reasoning state including the generated response,
            confidence scores, and metadata. The ``reasoning_mode`` field is
            set to ``ReasoningMode.DIRECT``.

        Raises
        ------
        ValueError
            If ``mode_result.selected_mode`` is not ``ReasoningModeEnum.FAST``.
        """
        if mode_result.selected_mode != ReasoningModeEnum.FAST:
            raise ValueError(
                f"FastModePipeline requires FAST mode, got {mode_result.selected_mode.value}"
            )

        # Single-pass generation (stubbed — real LLM integration comes later)
        response = self._generate(prompt)

        # Shallow heuristic validation
        passes_heuristic, heuristic_reason = self._shallow_heuristic_check(response)

        # Determine confidence tag using architecture thresholds
        confidence_tag = self._confidence_tag_for_confidence(calibrated_confidence)

        # Build and return the reasoning state
        state = self._build_state(
            prompt=prompt,
            response=response,
            calibrated_confidence=calibrated_confidence,
            mode_result=mode_result,
            confidence_tag=confidence_tag,
        )

        # Attach error flag if heuristic check failed
        if not passes_heuristic:
            state.add_error(f"heuristic_check_failed:{heuristic_reason}")

        return state

    def _generate(self, prompt: str) -> str:
        """Generate a response for the given prompt.

        This is a **stub** implementation. The actual LLM integration will be
        added in a future step. Currently returns a formatted mock response.

        Parameters
        ----------
        prompt : str
            The user prompt to generate a response for.

        Returns
        -------
        str
            A mock/generated response string.
        """
        return self.config.stub_response_template.format(prompt=prompt)

    def _shallow_heuristic_check(self, text: str) -> tuple[bool, str]:
        """Apply shallow heuristic validation to generated text.

        Checks performed:
        - Non-empty (not blank/whitespace only)
        - Length within bounds [min_response_length, max_response_length]

        Parameters
        ----------
        text : str
            The generated text to validate.

        Returns
        -------
        tuple[bool, str]
            A two-element tuple where the first element is True if the text
            passes all checks, and the second is a reason string:
            - ``"passed"`` — all checks passed
            - ``"empty_response"`` — text is empty or whitespace-only
            - ``"excessively_long"`` — text exceeds max_response_length
            - ``"too_short"`` — text is shorter than min_response_length
        """
        # Strip whitespace and check for emptiness
        stripped = text.strip()
        if not stripped:
            return (False, "empty_response")

        # Check length bounds
        if len(text) > self.config.max_response_length:
            return (False, "excessively_long")

        if len(text) < self.config.min_response_length:
            return (False, "too_short")

        return (True, "passed")

    def _confidence_tag_for_confidence(self, confidence: float) -> ConfidenceTag:
        """Derive the discrete confidence tag from a continuous confidence score.

        Uses the architecture-specified thresholds:
        - ``"high"``: confidence > 0.85
        - ``"medium"``: 0.5 < confidence <= 0.85
        - ``"low"``: confidence <= 0.5

        Parameters
        ----------
        confidence : float
            Calibrated confidence in [0.0, 1.0].

        Returns
        -------
        ConfidenceTag
            One of ``"high"``, ``"medium"``, ``"low"``.
        """
        if confidence > _ARCH_HIGH_THRESHOLD:
            return "high"
        elif confidence > _ARCH_MEDIUM_THRESHOLD:
            return "medium"
        else:
            return "low"

    def _build_state(
        self,
        prompt: str,
        response: str,
        calibrated_confidence: float,
        mode_result: ModeSelectionResult,
        confidence_tag: ConfidenceTag,
    ) -> CalibratedReasoningState:
        """Build a fully-populated CalibratedReasoningState.

        Parameters
        ----------
        prompt : str
            Original user prompt.
        response : str
            Generated response text.
        calibrated_confidence : float
            Calibrated confidence score.
        mode_result : ModeSelectionResult
            Mode selection result from the controller.
        confidence_tag : ConfidenceTag
            Discrete confidence tag.

        Returns
        -------
        CalibratedReasoningState
            Fully initialized reasoning state with all fields populated.
        """
        state = CalibratedReasoningState(
            prompt=prompt,
            reasoning_content=response,
            raw_confidence=calibrated_confidence,  # In fast mode, raw == calibrated
            calibrated_confidence=calibrated_confidence,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag=confidence_tag,
            model_name=self.actr_config.model.model_name,
            timestamp=datetime.now(timezone.utc),
            verification_result=None,  # No KG in fast mode
            reasoning_steps=[],  # Single-pass, no intermediate steps
        )

        return state
