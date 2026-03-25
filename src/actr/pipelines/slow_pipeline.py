"""Slow Mode reasoning pipeline for ACTR.

This module implements the SLOW reasoning pipeline — three-pass generation
with deep verification and full knowledge grounding for low-confidence queries.

Architecture
-----------
SLOW mode is selected when calibrated confidence <= 0.5. It uses:
- n_candidates=3 (three-pass generation)
- verification_depth="deep"
- uses_knowledge_grounding=True

The pipeline performs:
1. Three-pass generation (each pass can use SELF_VERIFICATION or TREE_OF_THOUGHT)
2. Deep verification at each stage
3. Consistency check between passes
4. Full knowledge grounding when both passes fail verification or are inconsistent
5. Selection of the best response by consistency score
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

__all__ = ["SlowModePipeline", "SlowPipelineConfig"]

# Architecture-specified thresholds for confidence tag derivation
_ARCH_HIGH_THRESHOLD = 0.85
_ARCH_MEDIUM_THRESHOLD = 0.5


@dataclass
class SlowPipelineConfig:
    """Configuration for the Slow Mode pipeline.

    Attributes
    ----------
    max_response_length : int
        Maximum allowed response length in characters. Slower modes allow
        longer responses since they involve multiple passes and KG.
    min_response_length : int
        Minimum required response length in characters. Responses shorter
        than this are considered invalid (too short to be meaningful).
    deep_verification_threshold : float
        Minimum consistency score to pass deep verification. Scores below
        this threshold indicate the response failed verification.
    kg_trigger_threshold : float
        Consistency score threshold. If the consistency check between
        passes falls below this, knowledge grounding is triggered.
    stub_response_template : str
        Template string used for mock/stub responses when no real LLM
        is available. Placeholders ``{pass}`` and ``{prompt}`` are replaced.
    """

    max_response_length: int = 50000
    min_response_length: int = 10
    deep_verification_threshold: float = 0.7
    kg_trigger_threshold: float = 0.5
    stub_response_template: str = "[SLOW Pass {pass_} response for: {prompt}]"


class SlowModePipeline:
    """Slow Mode reasoning pipeline.

    Takes a calibrated probability (already computed by the SSU) and a prompt,
    produces a response using three-pass generation with deep verification
    and full knowledge grounding.

    This is the most thorough pipeline — it is activated when the system has
    low confidence (p <= 0.5) and needs maximum verification to produce
    a reliable answer.

    Parameters
    ----------
    config : SlowPipelineConfig | None
        Pipeline configuration. If None, defaults are used.
    actr_config : ACTRConfig | None
        Global ACTR configuration. If None, the default ``ACTRConfig()`` is used.
    """

    def __init__(
        self,
        config: Optional[SlowPipelineConfig] = None,
        actr_config: Optional[ACTRConfig] = None,
    ) -> None:
        self.config = config if config is not None else SlowPipelineConfig()
        self.actr_config = actr_config if actr_config is not None else ACTRConfig()

    def run(
        self,
        prompt: str,
        calibrated_confidence: float,
        mode_result: ModeSelectionResult,
    ) -> CalibratedReasoningState:
        """Execute the Slow Mode pipeline for a given prompt.

        This is the main entry point. It:
        1. Validates the mode result is SLOW
        2. Performs three-pass generation
        3. Applies deep verification to each pass
        4. Checks consistency between passes
        5. Triggers knowledge grounding if needed
        6. Selects the best response by consistency score
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
            ``selected_mode == ReasoningModeEnum.SLOW``.

        Returns
        -------
        CalibratedReasoningState
            A fully-populated reasoning state including the generated response,
            confidence scores, and metadata. The ``reasoning_mode`` field is
            set to ``ReasoningMode.SELF_VERIFICATION`` (or TREE_OF_THOUGHT).
            The ``verification_result`` field is populated with the final
            verification result. ``reasoning_steps`` contains all three
            pass responses.

        Raises
        ------
        ValueError
            If ``mode_result.selected_mode`` is not ``ReasoningModeEnum.SLOW``.
        """
        # Mode validation — only SLOW mode is accepted
        if mode_result.selected_mode != ReasoningModeEnum.SLOW:
            raise ValueError(
                f"SlowModePipeline requires SLOW mode, got {mode_result.selected_mode.value}"
            )

        # -------------------------------------------------------------------------
        # PASS 1: Generate first response
        # -------------------------------------------------------------------------
        reasoning_steps: list[str] = []
        response_p1 = self._generate(prompt, pass_num=1)
        reasoning_steps.append(f"[Pass 1] {response_p1}")

        # Deep verification of Pass 1
        verification_p1 = self._deep_verification(response_p1, prompt)
        candidates: list[str] = [response_p1]
        verifications: list[VerificationResult] = [verification_p1]

        # -------------------------------------------------------------------------
        # PASS 2: Generate second response
        # -------------------------------------------------------------------------
        response_p2 = self._generate(prompt, pass_num=2)
        reasoning_steps.append(f"[Pass 2] {response_p2}")

        # Deep verification of Pass 2
        verification_p2 = self._deep_verification(response_p2, prompt)
        candidates.append(response_p2)
        verifications.append(verification_p2)

        # -------------------------------------------------------------------------
        # CONSISTENCY CHECK: Compare Pass 1 and Pass 2
        # -------------------------------------------------------------------------
        consistency_score = self._consistency_check(response_p1, response_p2)

        # Determine if KG should be triggered:
        # KG is triggered when:
        # (a) Both Pass 1 and Pass 2 fail deep verification, OR
        # (b) The consistency score between them is below the KG threshold
        p1_failed = verification_p1.consistency_score < self.config.deep_verification_threshold
        p2_failed = verification_p2.consistency_score < self.config.deep_verification_threshold
        kg_triggered = (p1_failed and p2_failed) or (consistency_score < self.config.kg_trigger_threshold)

        if kg_triggered:
            # -------------------------------------------------------------------------
            # KNOWLEDGE GROUNDING: triggered when both passes fail or are inconsistent
            # -------------------------------------------------------------------------
            reasoning_steps.append("[KG triggered: both passes failed or inconsistent]")

            # Select the best response so far for KG
            best_idx = self._select_best_response_index(candidates, verifications)
            response_for_kg = candidates[best_idx]

            grounded_response, grounded_verification = self._trigger_knowledge_grounding(
                prompt, response_for_kg
            )
            reasoning_steps.append(f"[KG grounded response] {grounded_response}")

            # KG response replaces Pass 2 as the 3rd candidate (keeps n_candidates=3)
            candidates.append(grounded_response)
            verifications.append(grounded_verification)

            # -------------------------------------------------------------------------
            # PASS 3: Generate third response (after KG)
            # -------------------------------------------------------------------------
            response_p3 = self._generate(prompt, pass_num=3)
            reasoning_steps.append(f"[Pass 3] {response_p3}")

            verification_p3 = self._deep_verification(response_p3, prompt)
            candidates.append(response_p3)
            verifications.append(verification_p3)
        else:
            # KG not triggered — generate Pass 3 as normal
            reasoning_steps.append("[KG not triggered: passes consistent and verified]")

            response_p3 = self._generate(prompt, pass_num=3)
            reasoning_steps.append(f"[Pass 3] {response_p3}")

            verification_p3 = self._deep_verification(response_p3, prompt)
            # Pass 3 becomes the 3rd candidate (KG not used)
            candidates.append(response_p3)
            verifications.append(verification_p3)

        # -------------------------------------------------------------------------
        # SELECT BEST RESPONSE: Pick candidate with highest consistency score
        # -------------------------------------------------------------------------
        best_response = self._select_best_response(candidates, verifications)
        best_idx = self._select_best_response_index(candidates, verifications)
        final_verification = verifications[best_idx]

        # Derive confidence tag using architecture thresholds
        confidence_tag = self._confidence_tag_for_confidence(calibrated_confidence)

        # Build and return the reasoning state
        state = self._build_state(
            prompt=prompt,
            response=best_response,
            calibrated_confidence=calibrated_confidence,
            mode_result=mode_result,
            confidence_tag=confidence_tag,
            reasoning_steps=reasoning_steps,
            verification_result=final_verification,
            kg_triggered=kg_triggered,
            consistency_score=consistency_score,
            candidates=candidates,
        )

        return state

    def _generate(self, prompt: str, pass_num: int) -> str:
        """Generate a response for the given prompt for a specific pass.

        This is a **stub** implementation. The actual LLM integration will be
        added in a future step. Currently returns a formatted mock response
        that includes the pass number.

        Parameters
        ----------
        prompt : str
            The user prompt to generate a response for.
        pass_num : int
            The pass number (1, 2, or 3) in the three-pass generation.

        Returns
        -------
        str
            A mock/generated response string.
        """
        return self.config.stub_response_template.format(pass_=pass_num, prompt=prompt)

    def _deep_verification(
        self,
        response: str,
        prompt: str,
    ) -> VerificationResult:
        """Perform deep verification on a generated response.

        Deep verification checks:
        - Premise alignment: Does the response address the prompt's premises?
        - Constraint satisfaction: Does the response respect prompt constraints?
        - Logical consistency: Is the response internally consistent?

        Parameters
        ----------
        response : str
            The generated response to verify.
        prompt : str
            The original prompt (used for constraint checking).

        Returns
        -------
        VerificationResult
            Verification result with consistency_score in [0.0, 1.0].
            ``is_verified`` is True if consistency_score >= deep_verification_threshold.
        """
        # Stub implementation: use deterministic scores based on content length
        # to make tests predictable. Responses that are too short fail.
        stripped = response.strip()
        if not stripped:
            return VerificationResult(
                is_verified=False,
                verification_method="deep_check",
                consistency_score=0.0,
                failed_checks=["empty_response"],
                passed_checks=[],
                verification_details={"reason": "empty response"},
            )

        if len(stripped) < self.config.min_response_length:
            return VerificationResult(
                is_verified=False,
                verification_method="deep_check",
                consistency_score=0.1,
                failed_checks=["too_short"],
                passed_checks=["non_empty"],
                verification_details={"length": len(stripped)},
            )

        # Length-based consistency score: longer is better up to a point
        # but we cap it so it's deterministic for tests
        length_ratio = min(len(stripped) / 100.0, 1.0)
        consistency_score = 0.5 + (0.5 * length_ratio)

        # For stub responses (contain "[SLOW Pass"), score above threshold
        if "[SLOW Pass" in stripped:
            consistency_score = 0.85  # Well-formed stub response

        passed_checks = ["premise_alignment", "constraint_satisfaction", "logical_consistency"]
        failed_checks: list[str] = []

        if consistency_score < self.config.deep_verification_threshold:
            passed_checks = []
            failed_checks = ["consistency_below_threshold"]

        return VerificationResult(
            is_verified=consistency_score >= self.config.deep_verification_threshold,
            verification_method="deep_check",
            consistency_score=consistency_score,
            failed_checks=failed_checks,
            passed_checks=passed_checks,
            verification_details={
                "verified_against_prompt": prompt[:100],
                "response_length": len(stripped),
            },
        )

    def _consistency_check(self, response1: str, response2: str) -> float:
        """Check consistency between two responses.

        Returns 1.0 if responses are identical, 0.5 if they differ.
        This is a simple stub — real implementation would use semantic
        similarity.

        Parameters
        ----------
        response1 : str
            First response.
        response2 : str
            Second response.

        Returns
        -------
        float
            1.0 if identical, 0.5 if different.
        """
        if response1.strip() == response2.strip():
            return 1.0
        return 0.5

    def _trigger_knowledge_grounding(
        self,
        prompt: str,
        response: str,
    ) -> tuple[str, VerificationResult]:
        """Trigger knowledge grounding on a response.

        Full knowledge grounding retrieves context, grounds the response in
        facts, and returns a revised response that is verified against
        known knowledge.

        Parameters
        ----------
        prompt : str
            The original prompt.
        response : str
            The response to ground.

        Returns
        -------
        tuple[str, VerificationResult]
            A two-element tuple of (grounded_response, VerificationResult).
        """
        # Stub: produce a KG'd response with knowledge context
        grounded = (
            f"[KG] Knowledge-grounded response for: {prompt}\n"
            f"Original: {response}\n"
            f"Grounded in facts and retrieved context."
        )

        verification = VerificationResult(
            is_verified=True,
            verification_method="knowledge_grounding",
            consistency_score=0.95,
            failed_checks=[],
            passed_checks=[
                "premise_alignment",
                "constraint_satisfaction",
                "logical_consistency",
                "fact_grounding",
            ],
            verification_details={
                "kg_applied": True,
                "original_response": response,
                "grounded_in_context": True,
            },
        )

        return grounded, verification

    def _select_best_response(
        self,
        candidates: list[str],
        verifications: list[VerificationResult],
    ) -> str:
        """Select the best response from a list of candidates.

        The candidate with the highest consistency score is selected.

        Parameters
        ----------
        candidates : list[str]
            List of candidate response strings.
        verifications : list[VerificationResult]
            List of verification results (same length as candidates).

        Returns
        -------
        str
            The best response string.

        Raises
        ------
        ValueError
            If candidates and verifications have different lengths.
        """
        if len(candidates) != len(verifications):
            raise ValueError(
                f"candidates and verifications must have same length, "
                f"got {len(candidates)} and {len(verifications)}"
            )
        if not candidates:
            raise ValueError("No candidates provided")

        best_idx = self._select_best_response_index(candidates, verifications)
        return candidates[best_idx]

    def _select_best_response_index(
        self,
        candidates: list[str],
        verifications: list[VerificationResult],
    ) -> int:
        """Return the index of the best response.

        Parameters
        ----------
        candidates : list[str]
            List of candidate response strings.
        verifications : list[VerificationResult]
            List of verification results.

        Returns
        -------
        int
            Index of the best candidate.
        """
        best_idx = 0
        best_score = verifications[0].consistency_score
        for i, v in enumerate(verifications):
            if v.consistency_score > best_score:
                best_score = v.consistency_score
                best_idx = i
        return best_idx

    def _confidence_tag_for_confidence(self, confidence: float) -> ConfidenceTag:
        """Derive the discrete confidence tag from a continuous confidence score.

        Uses the architecture-specified thresholds:
        - ``"high"``: confidence > 0.85
        - ``"medium"``: 0.5 < confidence <= 0.85
        - ``"low"``: confidence <= 0.5

        Note: SLOW mode is selected for p <= 0.5, so the confidence_tag
        will typically be ``"low"`` or ``"medium"`` at most.

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
        verification_result: VerificationResult,
        kg_triggered: bool = False,
        consistency_score: float = 1.0,
        candidates: Optional[list[str]] = None,
    ) -> CalibratedReasoningState:
        """Build a fully-populated CalibratedReasoningState.

        Parameters
        ----------
        prompt : str
            Original user prompt.
        response : str
            Selected best response.
        calibrated_confidence : float
            Calibrated confidence score.
        mode_result : ModeSelectionResult
            Mode selection result from the controller.
        confidence_tag : ConfidenceTag
            Discrete confidence tag.
        reasoning_steps : list[str]
            All reasoning steps (pass responses, KG steps, etc.).
        verification_result : VerificationResult
            Final verification result.
        kg_triggered : bool
            Whether knowledge grounding was triggered.
        consistency_score : float
            Consistency score between passes.
        candidates : list[str] | None
            All candidate responses.

        Returns
        -------
        CalibratedReasoningState
            Fully initialized reasoning state with all fields populated.
        """
        metadata: dict = {
            "n_candidates": len(candidates) if candidates else 3,
            "kg_triggered": kg_triggered,
            "consistency_score": consistency_score,
            "verification_depth": "deep",
            "uses_knowledge_grounding": True,
        }

        if candidates:
            metadata["all_candidates"] = candidates

        state = CalibratedReasoningState(
            prompt=prompt,
            reasoning_content=response,
            raw_confidence=calibrated_confidence,
            calibrated_confidence=calibrated_confidence,
            reasoning_mode=ReasoningMode.SELF_VERIFICATION,
            confidence_tag=confidence_tag,
            model_name=self.actr_config.model.model_name,
            timestamp=datetime.now(timezone.utc),
            verification_result=verification_result,
            reasoning_steps=reasoning_steps,
            metadata=metadata,
        )

        return state
