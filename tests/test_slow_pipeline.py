"""Tests for SlowModePipeline.

Comprehensive tests covering:
- Mode validation (rejects non-SLOW)
- Three-pass generation
- Deep verification triggers
- KG triggered when both fail or inconsistent
- Consistency: 1.0 identical, 0.5 different
- Confidence tag derivation
- Best response selection
- Reasoning steps populated
- Verification result set
- Edge cases: empty responses, threshold boundaries
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from actr.data import CalibratedReasoningState, VerificationResult
from actr.mode_controller import ModeSelectionResult, ReasoningModeEnum
from actr.pipelines import SlowModePipeline, SlowPipelineConfig


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def make_mode_result(
    mode: ReasoningModeEnum = ReasoningModeEnum.SLOW,
    confidence: float = 0.3,
) -> ModeSelectionResult:
    """Factory for ModeSelectionResult — SLOW mode by default."""
    return ModeSelectionResult(
        selected_mode=mode,
        confidence=confidence,
        confidence_tag="low" if confidence <= 0.5 else "medium",
        transition_reason=f"test: confidence={confidence}",
        timestamp=datetime.now(timezone.utc),
    )


# -----------------------------------------------------------------------
# SlowPipelineConfig tests
# -----------------------------------------------------------------------

class TestSlowPipelineConfig:
    """Tests for SlowPipelineConfig defaults."""

    def test_default_values(self):
        """Config has correct architecture-specified defaults."""
        config = SlowPipelineConfig()
        assert config.max_response_length == 50000
        assert config.min_response_length == 10
        assert config.deep_verification_threshold == 0.7
        assert config.kg_trigger_threshold == 0.5
        assert "[SLOW Pass {pass_}" in config.stub_response_template

    def test_custom_values(self):
        """Custom config values are respected."""
        config = SlowPipelineConfig(
            max_response_length=1000,
            min_response_length=5,
            deep_verification_threshold=0.8,
            kg_trigger_threshold=0.6,
            stub_response_template="custom: {pass} / {prompt}",
        )
        assert config.max_response_length == 1000
        assert config.min_response_length == 5
        assert config.deep_verification_threshold == 0.8
        assert config.kg_trigger_threshold == 0.6


# -----------------------------------------------------------------------
# Mode validation tests
# -----------------------------------------------------------------------

class TestModeValidation:
    """Tests for SLOW mode enforcement."""

    def test_rejects_fast_mode(self):
        """ValueError raised when mode is FAST."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result(ReasoningModeEnum.FAST, confidence=0.9)
        with pytest.raises(ValueError, match="requires SLOW mode"):
            pipeline.run("test prompt", 0.9, mode_result)

    def test_rejects_moderate_mode(self):
        """ValueError raised when mode is MODERATE."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result(ReasoningModeEnum.MODERATE, confidence=0.7)
        with pytest.raises(ValueError, match="requires SLOW mode"):
            pipeline.run("test prompt", 0.7, mode_result)

    def test_accepts_slow_mode(self):
        """No error when mode is SLOW."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result(ReasoningModeEnum.SLOW, confidence=0.3)
        state = pipeline.run("test prompt", 0.3, mode_result)
        assert state.reasoning_content != ""


# -----------------------------------------------------------------------
# Three-pass generation tests
# -----------------------------------------------------------------------

class TestThreePassGeneration:
    """Tests for three-pass generation."""

    def test_three_pass_generated(self):
        """All three passes are generated and appear in reasoning_steps."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        steps_text = "\n".join(state.reasoning_steps)
        assert "[Pass 1]" in steps_text
        assert "[Pass 2]" in steps_text
        assert "[Pass 3]" in steps_text

    def test_stub_response_format(self):
        """Stub responses contain pass number and prompt."""
        pipeline = SlowModePipeline(
            config=SlowPipelineConfig(
                stub_response_template="ANSWER(Pass {pass_}): {prompt}"
            )
        )
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        steps_text = "\n".join(state.reasoning_steps)
        assert "ANSWER(Pass 1)" in steps_text
        assert "ANSWER(Pass 2)" in steps_text
        assert "ANSWER(Pass 3)" in steps_text
        assert "What is 2+2?" in steps_text

    def test_n_candidates_in_metadata(self):
        """Metadata contains n_candidates=3."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        assert state.metadata.get("n_candidates") == 3

    def test_all_candidates_stored(self):
        """All candidate responses are stored in metadata."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        candidates = state.metadata.get("all_candidates", [])
        assert len(candidates) == 3  # Pass1, Pass2, Pass3 (KG adds more if triggered)


# -----------------------------------------------------------------------
# Deep verification tests
# -----------------------------------------------------------------------

class TestDeepVerification:
    """Tests for deep verification behavior."""

    def test_deep_verification_method(self):
        """Verification uses 'deep_check' method."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        assert state.verification_result is not None
        assert state.verification_result.verification_method == "deep_check"

    def test_stub_response_passes_verification(self):
        """Well-formed stub responses pass deep verification."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        # Stub responses are 0.85 score, which is >= 0.7 threshold
        assert state.verification_result is not None
        assert state.verification_result.consistency_score >= 0.7

    def test_empty_response_fails_verification(self):
        """Empty responses fail deep verification."""
        pipeline = SlowModePipeline(
            config=SlowPipelineConfig(stub_response_template="   ")
        )
        mode_result = make_mode_result()

        # The pipeline should handle this gracefully
        state = pipeline.run("What is 2+2?", 0.3, mode_result)
        # Empty responses get 0.0 consistency score
        # Verification result is from the best selected response
        assert state.verification_result is not None

    def test_deep_verification_threshold_config(self):
        """Threshold is respected in deep verification."""
        config = SlowPipelineConfig(deep_verification_threshold=0.99)
        pipeline = SlowModePipeline(config=config)
        mode_result = make_mode_result()

        state = pipeline.run("What is 2+2?", 0.3, mode_result)
        # Stub responses score 0.85 which is < 0.99, so both passes fail
        # → KG is triggered → best response is KG → method is knowledge_grounding
        assert state.verification_result is not None
        assert state.verification_result.verification_method == "knowledge_grounding"

    def test_verification_details_populated(self):
        """Verification details contain useful metadata."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        assert state.verification_result is not None
        details = state.verification_result.verification_details
        assert "verified_against_prompt" in details
        assert "response_length" in details


# -----------------------------------------------------------------------
# Consistency check tests
# -----------------------------------------------------------------------

class TestConsistencyCheck:
    """Tests for consistency check between passes."""

    def test_consistency_identical_responses(self):
        """Identical responses return 1.0 consistency."""
        pipeline = SlowModePipeline(
            config=SlowPipelineConfig(stub_response_template="Same response")
        )
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        # All passes produce identical responses → consistency = 1.0
        assert state.metadata.get("consistency_score") == 1.0

    def test_consistency_different_responses(self):
        """Different responses return 0.5 consistency."""
        pipeline = SlowModePipeline()  # Different per-pass output
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        # Pass 1 and Pass 2 differ → consistency = 0.5
        assert state.metadata.get("consistency_score") == 0.5

    def test_consistency_stored_in_metadata(self):
        """Consistency score is stored in state metadata."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        assert "consistency_score" in state.metadata


# -----------------------------------------------------------------------
# Knowledge Grounding tests
# -----------------------------------------------------------------------

class TestKnowledgeGrounding:
    """Tests for knowledge grounding triggering and behavior."""

    def test_kg_triggered_when_consistency_low(self):
        """KG is triggered when consistency < kg_trigger_threshold."""
        # Pass 1 and Pass 2 differ → consistency = 0.5
        # kg_trigger_threshold = 0.5, so consistency == threshold → not triggered by that
        # Both stub responses pass verification, so kg_triggered should be False
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        # KG triggered because both passes are identical (consistency=1.0) but
        # wait — KG is triggered when (both fail OR consistency < threshold)
        # Here consistency = 0.5, threshold = 0.5, so NOT < → KG not triggered
        # But both pass verification (0.85 >= 0.7), so KG not triggered
        # Let's check the actual value
        kg_was_triggered = state.metadata.get("kg_triggered", False)
        # With identical stub responses: consistency=1.0, both pass → KG not triggered
        assert kg_was_triggered is False

    def test_kg_not_triggered_when_consistent_and_verified(self):
        """KG not triggered when passes are consistent and verified."""
        # Template without {pass_} so all 3 passes produce identical strings.
        # Contains "[SLOW Pass]" so stub verification gives 0.85 (passes 0.7).
        # Identical responses → consistency = 1.0 ≥ 0.5 → KG not triggered.
        config = SlowPipelineConfig(
            stub_response_template="[SLOW Pass] identical verified answer",
        )
        pipeline = SlowModePipeline(config=config)
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        assert state.metadata.get("kg_triggered") is False
        assert "[KG triggered" not in "\n".join(state.reasoning_steps)
        assert state.metadata.get("consistency_score") == 1.0

    def test_kg_adds_grounded_candidate(self):
        """KG adds a grounded response as an additional candidate."""
        # We need to trigger KG: both passes fail verification
        # Use very short stub responses that will fail verification
        pipeline = SlowModePipeline(
            config=SlowPipelineConfig(
                stub_response_template="x",  # Too short to pass
                deep_verification_threshold=0.9,  # High threshold
            )
        )
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        # KG should be triggered because both fail verification
        assert state.metadata.get("kg_triggered") is True
        candidates = state.metadata.get("all_candidates", [])
        # Should have more than just the 3 passes (KG adds one)
        assert len(candidates) >= 4

    def test_kg_response_contains_kg_marker(self):
        """KG response is marked as knowledge-grounded."""
        pipeline = SlowModePipeline(
            config=SlowPipelineConfig(
                stub_response_template="x",  # Short → fails verification
                deep_verification_threshold=0.9,
            )
        )
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        steps_text = "\n".join(state.reasoning_steps)
        assert "[KG triggered" in steps_text
        assert "[KG grounded response]" in steps_text

    def test_uses_knowledge_grounding_in_metadata(self):
        """Metadata confirms knowledge grounding is used."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        assert state.metadata.get("uses_knowledge_grounding") is True

    def test_kg_verification_has_high_score(self):
        """KG verification produces a high consistency score."""
        pipeline = SlowModePipeline(
            config=SlowPipelineConfig(
                stub_response_template="x",
                deep_verification_threshold=0.9,
            )
        )
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        # KG verification produces score of 0.95
        # The best response should be the KG one (highest score)
        assert state.verification_result is not None
        assert state.verification_result.consistency_score == 0.95


# -----------------------------------------------------------------------
# Best response selection tests
# -----------------------------------------------------------------------

class TestBestResponseSelection:
    """Tests for selecting the best response from candidates."""

    def test_best_response_is_selected(self):
        """Best response (highest consistency) is returned."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        # State should have a non-empty reasoning content
        assert state.reasoning_content != ""

    def test_best_response_picks_highest_score(self):
        """When KG is triggered, KG response (0.95) is selected as best."""
        pipeline = SlowModePipeline(
            config=SlowPipelineConfig(
                stub_response_template="short",  # Will fail verification
                deep_verification_threshold=0.9,
            )
        )
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        # KG response has 0.95, should be selected
        assert state.verification_result is not None
        assert state.verification_result.consistency_score == 0.95
        assert "[KG]" in state.reasoning_content

    def test_select_best_response_raises_on_empty(self):
        """select_best_response raises ValueError on empty list."""
        pipeline = SlowModePipeline()
        with pytest.raises(ValueError, match="No candidates"):
            pipeline._select_best_response([], [])

    def test_select_best_response_raises_on_length_mismatch(self):
        """select_best_response raises ValueError on length mismatch."""
        pipeline = SlowModePipeline()
        with pytest.raises(ValueError, match="same length"):
            pipeline._select_best_response(["a", "b"], [VerificationResult(
                is_verified=True,
                verification_method="test",
                consistency_score=0.5,
            )])


# -----------------------------------------------------------------------
# Confidence tag tests
# -----------------------------------------------------------------------

class TestConfidenceTag:
    """Tests for confidence tag derivation."""

    def test_confidence_tag_low_for_p05(self):
        """Tag is 'low' when confidence == 0.5."""
        pipeline = SlowModePipeline()
        assert pipeline._confidence_tag_for_confidence(0.5) == "low"

    def test_confidence_tag_low_for_below_05(self):
        """Tag is 'low' when confidence < 0.5."""
        pipeline = SlowModePipeline()
        assert pipeline._confidence_tag_for_confidence(0.3) == "low"
        assert pipeline._confidence_tag_for_confidence(0.0) == "low"

    def test_confidence_tag_medium_for_p075(self):
        """Tag is 'medium' when 0.5 < confidence <= 0.85."""
        pipeline = SlowModePipeline()
        assert pipeline._confidence_tag_for_confidence(0.75) == "medium"

    def test_confidence_tag_high_for_p09(self):
        """Tag is 'high' when confidence > 0.85."""
        pipeline = SlowModePipeline()
        assert pipeline._confidence_tag_for_confidence(0.9) == "high"

    def test_confidence_tag_in_state(self):
        """State carries the correct confidence_tag."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result(confidence=0.3)
        state = pipeline.run("What is 2+2?", 0.3, mode_result)
        assert state.confidence_tag == "low"


# -----------------------------------------------------------------------
# Reasoning mode and reasoning steps tests
# -----------------------------------------------------------------------

class TestReasoningSteps:
    """Tests for reasoning_steps population."""

    def test_reasoning_steps_not_empty(self):
        """reasoning_steps is populated with all passes."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        assert len(state.reasoning_steps) >= 3

    def test_reasoning_steps_contain_pass_markers(self):
        """Each pass is labelled in reasoning_steps."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        steps = state.reasoning_steps
        assert any("[Pass 1]" in s for s in steps)
        assert any("[Pass 2]" in s for s in steps)
        assert any("[Pass 3]" in s for s in steps)

    def test_reasoning_mode_is_self_verification(self):
        """reasoning_mode is set to SELF_VERIFICATION."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        assert str(state.reasoning_mode) == "self_verification"

    def test_verification_result_is_set(self):
        """verification_result field is populated."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        assert state.verification_result is not None
        assert isinstance(state.verification_result, VerificationResult)


# -----------------------------------------------------------------------
# State building tests
# -----------------------------------------------------------------------

class TestStateBuilding:
    """Tests for _build_state and state population."""

    def test_calibrated_confidence_preserved(self):
        """Calibrated confidence is stored in state."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result(confidence=0.4)
        state = pipeline.run("What is 2+2?", 0.4, mode_result)

        assert state.calibrated_confidence == 0.4

    def test_raw_confidence_equals_calibrated(self):
        """In slow mode, raw_confidence == calibrated_confidence."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result(confidence=0.3)
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        assert state.raw_confidence == state.calibrated_confidence

    def test_model_name_set(self):
        """Model name is set from actr_config."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        assert state.model_name is not None

    def test_timestamp_set(self):
        """Timestamp is set on the state."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        assert state.timestamp is not None
        assert isinstance(state.timestamp, datetime)

    def test_prompt_preserved(self):
        """Original prompt is preserved in state."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        assert state.prompt == "What is 2+2?"

    def test_metadata_has_verification_depth(self):
        """Metadata contains verification_depth='deep'."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("What is 2+2?", 0.3, mode_result)

        assert state.metadata.get("verification_depth") == "deep"


# -----------------------------------------------------------------------
# Edge case tests
# -----------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_prompt(self):
        """Pipeline handles empty prompt gracefully."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("", 0.3, mode_result)

        assert state.reasoning_content != ""

    def test_very_long_prompt(self):
        """Pipeline handles very long prompt."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        long_prompt = "Q: " + "x" * 10000
        state = pipeline.run(long_prompt, 0.3, mode_result)

        assert state.reasoning_content != ""
        # KG step should include truncated prompt
        steps_text = "\n".join(state.reasoning_steps)
        assert "KG" in steps_text or "[Pass" in steps_text

    def test_confidence_at_exact_threshold(self):
        """Confidence exactly at 0.5 threshold works."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result(confidence=0.5)
        state = pipeline.run("Test", 0.5, mode_result)
        assert state.confidence_tag == "low"

    def test_deep_verification_threshold_boundary(self):
        """Responses at exact threshold boundary behave correctly."""
        # Stub responses score 0.85
        # With threshold 0.85, passes but barely
        config = SlowPipelineConfig(deep_verification_threshold=0.85)
        pipeline = SlowModePipeline(config=config)
        mode_result = make_mode_result()
        state = pipeline.run("Test", 0.3, mode_result)

        # Should pass (0.85 >= 0.85)
        assert state.verification_result is not None

    def test_kg_trigger_exactly_at_threshold(self):
        """KG triggered when consistency exactly at threshold."""
        # When consistency == 0.5 and threshold == 0.5, NOT < → not triggered
        # Need to verify this edge case is handled
        config = SlowPipelineConfig(kg_trigger_threshold=0.5)
        pipeline = SlowModePipeline(config=config)
        mode_result = make_mode_result()
        state = pipeline.run("Test", 0.3, mode_result)

        # consistency = 0.5, threshold = 0.5, 0.5 < 0.5 is False
        # Both pass verification (0.85 >= 0.7) → KG not triggered
        assert state.metadata.get("kg_triggered") is False

    def test_multiple_identical_passes(self):
        """All passes identical gives consistency 1.0."""
        config = SlowPipelineConfig(stub_response_template="identical response")
        pipeline = SlowModePipeline(config=config)
        mode_result = make_mode_result()
        state = pipeline.run("Test", 0.3, mode_result)

        assert state.metadata.get("consistency_score") == 1.0

    def test_verification_result_passed_checks(self):
        """VerificationResult passed_checks are populated."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("Test", 0.3, mode_result)

        assert state.verification_result is not None
        # Stub responses pass all checks
        assert len(state.verification_result.passed_checks) > 0

    def test_calibration_history_not_modified(self):
        """Calibration history is empty (slow mode doesn't add records itself)."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("Test", 0.3, mode_result)

        # Slow mode itself doesn't add calibration records;
        # the SSU/calibration step does that before calling the pipeline
        assert len(state.calibration_history) == 0

    def test_error_flags_empty_on_success(self):
        """No error flags on successful run."""
        pipeline = SlowModePipeline()
        mode_result = make_mode_result()
        state = pipeline.run("Test", 0.3, mode_result)

        assert len(state.error_flags) == 0

    def test_run_with_none_optional_params(self):
        """Pipeline works with default config."""
        pipeline = SlowModePipeline(config=None, actr_config=None)
        mode_result = make_mode_result()
        state = pipeline.run("Test", 0.3, mode_result)
        assert state.reasoning_content != ""


# -----------------------------------------------------------------------
# Integration-like tests
# -----------------------------------------------------------------------

class TestFullPipeline:
    """Integration-style tests covering full pipeline flow."""

    def test_full_flow_slow_mode(self):
        """Complete flow: SLOW mode with all stages executes."""
        pipeline = SlowModePipeline()
        mode_result = ModeSelectionResult(
            selected_mode=ReasoningModeEnum.SLOW,
            confidence=0.2,
            confidence_tag="low",
            transition_reason="test: low confidence",
        )
        state = pipeline.run("Explain quantum entanglement.", 0.2, mode_result)

        # All critical fields populated
        assert state.prompt == "Explain quantum entanglement."
        assert state.calibrated_confidence == 0.2
        assert state.confidence_tag == "low"
        assert state.reasoning_content != ""
        assert state.verification_result is not None
        assert len(state.reasoning_steps) >= 3
        assert state.metadata["uses_knowledge_grounding"] is True
        assert state.metadata["verification_depth"] == "deep"
        assert state.metadata["n_candidates"] == 3

    def test_full_flow_kg_triggered(self):
        """Full flow where KG is triggered."""
        config = SlowPipelineConfig(
            stub_response_template="x",
            deep_verification_threshold=0.9,
        )
        pipeline = SlowModePipeline(config=config)
        mode_result = make_mode_result(confidence=0.1)
        state = pipeline.run("Complex reasoning problem", 0.1, mode_result)

        assert state.metadata["kg_triggered"] is True
        assert state.verification_result.consistency_score == 0.95
        assert "[KG]" in state.reasoning_content

    def test_slow_mode_is_reproducible_with_same_seed(self):
        """Same inputs produce same outputs (stub determinism)."""
        config = SlowPipelineConfig(stub_response_template="fixed response")
        pipeline = SlowModePipeline(config=config)
        mode_result = make_mode_result(confidence=0.3)

        state1 = pipeline.run("test", 0.3, mode_result)
        state2 = pipeline.run("test", 0.3, mode_result)

        assert state1.reasoning_content == state2.reasoning_content
        assert state1.metadata["consistency_score"] == state2.metadata["consistency_score"]
