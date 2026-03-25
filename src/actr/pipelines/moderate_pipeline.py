"""Moderate Mode reasoning pipeline for ACTR.

This module implements the MODERATE reasoning pipeline — two-pass generation
with standard verification for medium-confidence queries.

Architecture
-----------
MODERATE mode is selected when calibrated confidence is in (0.5, 0.85].
It uses:
- n_candidates=2 (two-pass generation)
- verification_depth="standard"
- uses_knowledge_grounding=False by default, BUT KG is triggered on failure
- ReasoningMode.CHAIN_OF_THOUGHT (explicit intermediate steps)

The pipeline performs two passes and triggers knowledge grounding only
when both passes fail verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from actr.data import (
    CalibratedReasoningState,
    ReasoningMode,
    ConfidenceTag,
    VerificationResult,
)
from actr.mode_controller import ModeSelectionResult, ReasoningModeEnum
from actr.config import ACTRConfig

__all__ = ["ModerateModePipeline", "ModeratePipelineConfig"]

# Architecture-specified thresholds for confidence tag derivation
_ARCH_HIGH_THRESHOLD = 0.85
_ARCH_MEDIUM_THRESHOLD = 0.5


@dataclass
class ModeratePipelineConfig:
    """Configuration for the Moderate Mode pipeline.

    Attributes
    ----------
    max_response_length : int
        Maximum allowed response length in characters.
    min_response_length : int
        Minimum required response length in characters.
    stub_response_template : str
        Template string used for mock/stub responses.
    verification_threshold : float
        Consistency score threshold for verification pass.
        Scores >= threshold pass verification.
    kg_trigger_on_failure : bool
        Whether to trigger knowledge grounding when both passes fail.
    """

    max_response_length: int = 15000
    min_response_length: int = 20
    stub_response_template: str = "[MODERATE response for: {prompt}]"
    verification_threshold: float = 0.7
    kg_trigger_on_failure: bool = True


class ModerateModePipeline:
    """Moderate Mode reasoning pipeline.

    Takes a calibrated probability and prompt, produces a response using
    two-pass generation with standard verification. Knowledge grounding
    is triggered only when both passes fail verification.

    Parameters
    ----------
    config : ModeratePipelineConfig | None
        Pipeline configuration. If None, defaults are used.
    actr_config : ACTRConfig | None
        Global ACTR configuration. If None, the default ``ACTRConfig()`` is used.
    """

    def __init__(
        self,
        config: Optional[ModeratePipelineConfig] = None,
        actr_config: Optional[ACTRConfig] = None,
    ) -> None:
        self.config = config if config is not None else ModeratePipelineConfig()
        self.actr_config = actr_config if actr_config is not None else ACTRConfig()

    def run(
        self,
        prompt: str,
        calibrated_confidence: float,
        mode_result: ModeSelectionResult,
    ) -> CalibratedReasoningState:
        """Execute the Moderate Mode pipeline for a given prompt.

        This is the main entry point. It:
        1. Validates the mode result is MODERATE
        2. Generates Pass 1 with chain-of-thought reasoning
        3. Verifies Pass 1 — if it passes, return
        4. If verification fails, generates Pass 2
        5. Consistency check between passes — if consistent, pick best
        6. If inconsistent, trigger KG (if kg_trigger_on_failure is True)
        7. Returns a fully-populated CalibratedReasoningState

        Parameters
        ----------
        prompt : str
            The original user prompt.
        calibrated_confidence : float
            The pre-computed calibrated probability from the SSU/calibration
            pipeline (already in [0.0, 1.0]).
        mode_result : ModeSelectionResult
            The result of the mode selection step, which must have
            ``selected_mode == ReasoningModeEnum.MODERATE``.

        Returns
        -------
        CalibratedReasoningState
            A fully-populated reasoning state.

        Raises
        ------
        ValueError
            If ``mode_result.selected_mode`` is not ``ReasoningModeEnum.MODERATE``.
        """
        if mode_result.selected_mode != ReasoningModeEnum.MODERATE:
            raise ValueError(
                f"ModerateModePipeline requires MODERATE mode, "
                f"got {mode_result.selected_mode.value}"
            )

        reasoning_steps: list[str] = []

        # ---- Pass 1: Generate with chain-of-thought ----
        pass_1 = self._generate_candidates(prompt, 1)[0]
        reasoning_steps.append(f"Pass 1: {pass_1}")

        # ---- Verify Pass 1 ----
        pass_1_passes, pass_1_reason = self._verify_response(pass_1)
        reasoning_steps.append(f"Pass 1 verification: {pass_1_reason}")

        # If Pass 1 passes verification, we're done
        if pass_1_passes:
            confidence_tag = self._confidence_tag_for_confidence(calibrated_confidence)
            verification_result = VerificationResult(
                is_verified=True,
                verification_method="standard",
                consistency_score=1.0,
                passed_checks=["pass_1_verification"],
                verification_details={"passes": ["pass_1"]},
            )
            return self._build_state(
                prompt=prompt,
                response=pass_1,
                calibrated_confidence=calibrated_confidence,
                mode_result=mode_result,
                confidence_tag=confidence_tag,
                reasoning_steps=reasoning_steps,
                verification_result=verification_result,
            )

        # ---- Pass 2: Generate second response ----
        pass_2 = self._generate_candidates(prompt, 1)[0]
        reasoning_steps.append(f"Pass 2: {pass_2}")

        # ---- Verify Pass 2 ----
        pass_2_passes, pass_2_reason = self._verify_response(pass_2)
        reasoning_steps.append(f"Pass 2 verification: {pass_2_reason}")

        pass_2_verification = VerificationResult(
            is_verified=pass_2_passes,
            verification_method="standard",
            consistency_score=1.0 if pass_2_passes else 0.0,
            passed_checks=["pass_2"] if pass_2_passes else [],
            failed_checks=["pass_2"] if not pass_2_passes else [],
        )

        # ---- Consistency check between passes ----
        consistency_score = self._consistency_check(pass_1, pass_2)
        reasoning_steps.append(f"Consistency check: score={consistency_score}")

        # ---- KG trigger: both fail or inconsistent ----
        both_fail_verification = not pass_1_passes and not pass_2_passes
        inconsistent = consistency_score < self.config.verification_threshold

        if self.config.kg_trigger_on_failure and (both_fail_verification or inconsistent):
            reasoning_steps.append("Triggering knowledge grounding...")
            kg_response, kg_verification = self._trigger_knowledge_grounding(
                prompt, pass_1
            )
            reasoning_steps.append(f"KG result: {kg_response[:50]}...")

            confidence_tag = self._confidence_tag_for_confidence(calibrated_confidence)
            return self._build_state(
                prompt=prompt,
                response=kg_response,
                calibrated_confidence=calibrated_confidence,
                mode_result=mode_result,
                confidence_tag=confidence_tag,
                reasoning_steps=reasoning_steps,
                verification_result=kg_verification,
            )

        # ---- Select best response ----
        candidates = [pass_1, pass_2]
        verifications = [
            VerificationResult(
                is_verified=pass_1_passes,
                verification_method="standard",
                consistency_score=1.0 if pass_1_passes else 0.0,
                passed_checks=["pass_1"] if pass_1_passes else [],
                failed_checks=["pass_1"] if not pass_1_passes else [],
            ),
            pass_2_verification,
        ]

        best_response, best_verification = self._select_best_response(
            candidates, verifications
        )
        reasoning_steps.append(f"Selected response (verification={best_verification.is_verified})")

        confidence_tag = self._confidence_tag_for_confidence(calibrated_confidence)

        return self._build_state(
            prompt=prompt,
            response=best_response,
            calibrated_confidence=calibrated_confidence,
            mode_result=mode_result,
            confidence_tag=confidence_tag,
            reasoning_steps=reasoning_steps,
            verification_result=best_verification,
        )

    def _generate_candidates(self, prompt: str, n: int) -> list[str]:
        """Generate n candidate responses for the given prompt.

        This is a **stub** implementation. The actual LLM integration
        will be added in a future step.

        Parameters
        ----------
        prompt : str
            The user prompt.
        n : int
            Number of candidates to generate.

        Returns
        -------
        list[str]
            A list of n mock/generated response strings.
        """
        return [
            self.config.stub_response_template.format(prompt=prompt)
            for _ in range(n)
        ]

    def _verify_response(self, response: str) -> tuple[bool, str]:
        """Verify a response for consistency and self-verification.

        This is a **stub** implementation. In a real implementation,
        this would check the response for internal consistency.

        Parameters
        ----------
        response : str
            The generated response to verify.

        Returns
        -------
        tuple[bool, str]
            A two-element tuple where the first element is True if the
            response passes verification, and the second is a reason string.
        """
        # Stub: check length bounds
        stripped = response.strip()
        if not stripped:
            return (False, "empty_response")

        if len(response) > self.config.max_response_length:
            return (False, "excessively_long")

        if len(response) < self.config.min_response_length:
            return (False, "too_short")

        # Stub: pass if stub template is used (length is correct)
        return (True, "passed")

    def _consistency_check(self, r1: str, r2: str) -> float:
        """Check semantic consistency between two responses.

        This is a **stub** implementation. In a real implementation,
        this would compute semantic similarity between the two responses.

        Parameters
        ----------
        r1 : str
            First response.
        r2 : str
            Second response.

        Returns
        -------
        float
            A consistency score in [0.0, 1.0]. Higher means more consistent.
        """
        if r1 == r2:
            return 1.0
        return 0.5

    def _trigger_knowledge_grounding(
        self, prompt: str, response: str
    ) -> tuple[str, VerificationResult]:
        """Trigger knowledge grounding to revise a failing response.

        This is a **stub** implementation. In a real implementation,
        this would search a knowledge base and produce a grounded response.

        Parameters
        ----------
        prompt : str
            The original user prompt.
        response : str
            The current response that failed verification.

        Returns
        -------
        tuple[str, VerificationResult]
            A tuple of (revised_response, verification_result).
        """
        revised_response = self.config.stub_response_template.format(
            prompt=f"[KG-grounded] {prompt}"
        )
        verification_result = VerificationResult(
            is_verified=True,
            verification_method="knowledge_grounding",
            consistency_score=1.0,
            passed_checks=["kg_grounded"],
            verification_details={"grounded": True},
        )
        return revised_response, verification_result

    def _select_best_response(
        self,
        candidates: list[str],
        verifications: list[VerificationResult],
    ) -> tuple[str, VerificationResult]:
        """Select the best response from candidates based on verification.

        Parameters
        ----------
        candidates : list[str]
            List of candidate responses.
        verifications : list[VerificationResult]
            List of verification results, one per candidate.

        Returns
        -------
        tuple[str, VerificationResult]
            The best response and its verification result.

        Raises
        ------
        ValueError
            If candidates and verifications have different lengths.
        """
        if len(candidates) != len(verifications):
            raise ValueError(
                f"candidates ({len(candidates)}) and verifications "
                f"({len(verifications)}) must have the same length"
            )

        if not candidates:
            raise ValueError("At least one candidate is required")

        # Find the candidate with the highest consistency score
        best_idx = 0
        best_score = verifications[0].consistency_score

        for i, verification in enumerate(verifications):
            if verification.consistency_score > best_score:
                best_score = verification.consistency_score
                best_idx = i

        return candidates[best_idx], verifications[best_idx]

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
        reasoning_steps: list[str],
        verification_result: VerificationResult | None,
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
        reasoning_steps : list[str]
            Intermediate reasoning steps.
        verification_result : VerificationResult | None
            Verification result, if verification was performed.

        Returns
        -------
        CalibratedReasoningState
            Fully initialized reasoning state.
        """
        state = CalibratedReasoningState(
            prompt=prompt,
            reasoning_content=response,
            raw_confidence=calibrated_confidence,
            calibrated_confidence=calibrated_confidence,
            reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
            confidence_tag=confidence_tag,
            model_name=self.actr_config.model.model_name,
            timestamp=datetime.now(timezone.utc),
            verification_result=verification_result,
            reasoning_steps=reasoning_steps,
        )

        return state
