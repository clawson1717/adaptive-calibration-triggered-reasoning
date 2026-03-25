"""Tests for BoundaryEnforcementLayer and BoundaryEnforcementConfig."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from actr.data import CalibratedReasoningState, ReasoningMode, CalibrationRecord
from actr.pipelines.boundary_enforcement import (
    BoundaryEnforcementLayer,
    BoundaryEnforcementConfig,
    _expected_reasoning_mode,
    _MAX_REASONING_LENGTH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config() -> BoundaryEnforcementConfig:
    return BoundaryEnforcementConfig()


@pytest.fixture
def layer(default_config: BoundaryEnforcementConfig) -> BoundaryEnforcementLayer:
    return BoundaryEnforcementLayer(config=default_config)


@pytest.fixture
def base_state() -> CalibratedReasoningState:
    """A basic valid reasoning state for testing."""
    return CalibratedReasoningState(
        prompt="What is 2+2?",
        reasoning_content="The answer is 4.",
        raw_confidence=0.86,
        calibrated_confidence=0.86,
        reasoning_mode=ReasoningMode.DIRECT,
        confidence_tag="high",
    )


@pytest.fixture
def empty_state() -> CalibratedReasoningState:
    """An empty reasoning state."""
    return CalibratedReasoningState(
        prompt="Test prompt",
        reasoning_content="",
        raw_confidence=0.0,
        calibrated_confidence=0.0,
        reasoning_mode=ReasoningMode.DIRECT,
        confidence_tag="unknown",
    )


# ---------------------------------------------------------------------------
# BoundaryEnforcementConfig tests — defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_memory_grounding_threshold_default(self):
        config = BoundaryEnforcementConfig()
        assert config.memory_grounding_threshold == 0.7

    def test_safety_reject_threshold_default(self):
        config = BoundaryEnforcementConfig()
        assert config.safety_reject_threshold == 0.3

    def test_envelope_decay_factor_default(self):
        config = BoundaryEnforcementConfig()
        assert config.envelope_decay_factor == 0.95

    def test_max_envelope_propagation_steps_default(self):
        config = BoundaryEnforcementConfig()
        assert config.max_envelope_propagation_steps == 5

    def test_enable_memory_grounding_default(self):
        config = BoundaryEnforcementConfig()
        assert config.enable_memory_grounding is True

    def test_enable_safety_bounds_default(self):
        config = BoundaryEnforcementConfig()
        assert config.enable_safety_bounds is True

    def test_enable_envelope_propagation_default(self):
        config = BoundaryEnforcementConfig()
        assert config.enable_envelope_propagation is True

    def test_config_custom_values(self):
        config = BoundaryEnforcementConfig(
            memory_grounding_threshold=0.6,
            safety_reject_threshold=0.2,
            envelope_decay_factor=0.9,
            max_envelope_propagation_steps=10,
            enable_memory_grounding=False,
            enable_safety_bounds=False,
            enable_envelope_propagation=False,
        )
        assert config.memory_grounding_threshold == 0.6
        assert config.safety_reject_threshold == 0.2
        assert config.envelope_decay_factor == 0.9
        assert config.max_envelope_propagation_steps == 10
        assert config.enable_memory_grounding is False
        assert config.enable_safety_bounds is False
        assert config.enable_envelope_propagation is False


# ---------------------------------------------------------------------------
# Safety bounds tests
# ---------------------------------------------------------------------------

class TestSafetyBounds:
    def test_safety_reject_below_threshold_sets_error_flag(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Confidence < 0.3 → 'safety_rejected' in error_flags."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.2,
            calibrated_confidence=0.2,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="low",
        )
        result = layer.run(state)
        assert "safety_rejected" in result.error_flags

    def test_safety_reject_below_threshold_clears_content(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Confidence < 0.3 → reasoning_content is cleared."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.2,
            calibrated_confidence=0.2,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="low",
        )
        result = layer.run(state)
        assert result.reasoning_content == ""

    def test_safety_reject_adds_calibration_record(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Confidence < 0.3 → calibration record is added."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.2,
            calibrated_confidence=0.2,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="low",
        )
        layer.run(state)
        reasons = [r.adjustment_reason for r in state.calibration_history]
        assert any("safety_rejected" in r for r in reasons)

    def test_safety_reject_exactly_at_threshold_passes(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Confidence == 0.3 → no safety rejection."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.3,
            calibrated_confidence=0.3,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="low",
        )
        result = layer.run(state)
        assert "safety_rejected" not in result.error_flags
        assert result.reasoning_content == "Answer"

    def test_safety_reject_above_threshold_passes(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Confidence > 0.3 → no safety rejection."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.5,
            calibrated_confidence=0.5,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="medium",
        )
        result = layer.run(state)
        assert "safety_rejected" not in result.error_flags
        assert result.reasoning_content == "Answer"

    def test_safety_bounds_disabled_allows_low_confidence(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """With safety bounds disabled, low confidence is NOT rejected."""
        config = BoundaryEnforcementConfig(enable_safety_bounds=False)
        layer = BoundaryEnforcementLayer(config=config)
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.1,
            calibrated_confidence=0.1,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="low",
        )
        result = layer.run(state)
        assert "safety_rejected" not in result.error_flags
        assert result.reasoning_content == "Answer"

    def test_safety_bounds_boundary_at_zero(self):
        """Confidence == 0.0 is below threshold and triggers rejection."""
        config = BoundaryEnforcementConfig()
        layer = BoundaryEnforcementLayer(config=config)
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.0,
            calibrated_confidence=0.0,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="low",
        )
        result = layer.run(state)
        assert "safety_rejected" in result.error_flags
        assert result.reasoning_content == ""

    def test_safety_bounds_boundary_at_one(self):
        """Confidence == 1.0 is above threshold and passes."""
        config = BoundaryEnforcementConfig()
        layer = BoundaryEnforcementLayer(config=config)
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=1.0,
            calibrated_confidence=1.0,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="high",
        )
        result = layer.run(state)
        assert "safety_rejected" not in result.error_flags


# ---------------------------------------------------------------------------
# Memory grounding gate tests
# ---------------------------------------------------------------------------

class TestMemoryGroundingGate:
    def test_memory_grounding_activates_below_threshold(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Confidence < 0.7 → memory grounding activated."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.5,
            calibrated_confidence=0.5,
            reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
            confidence_tag="medium",
        )
        result = layer.run(state)
        assert result.metadata.get("memory_grounding_activated") is True

    def test_memory_grounding_adds_calibration_record(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """When grounding activates, calibration record is added."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.5,
            calibrated_confidence=0.5,
            reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
            confidence_tag="medium",
        )
        layer.run(state)
        reasons = [r.adjustment_reason for r in state.calibration_history]
        assert any("memory_grounding_activated" in r for r in reasons)

    def test_memory_grounding_not_activated_at_threshold(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Confidence == 0.7 → grounding NOT activated (threshold is exclusive)."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.7,
            calibrated_confidence=0.7,
            reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
            confidence_tag="medium",
        )
        result = layer.run(state)
        assert result.metadata.get("memory_grounding_activated") is not True

    def test_memory_grounding_not_activated_above_threshold(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Confidence > 0.7 → grounding NOT activated."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.9,
            calibrated_confidence=0.9,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="high",
        )
        result = layer.run(state)
        assert result.metadata.get("memory_grounding_activated") is not True

    def test_memory_grounding_disabled_no_activation(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """With grounding disabled, no activation metadata even below threshold."""
        config = BoundaryEnforcementConfig(enable_memory_grounding=False)
        layer = BoundaryEnforcementLayer(config=config)
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.5,
            calibrated_confidence=0.5,
            reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
            confidence_tag="medium",
        )
        result = layer.run(state)
        assert result.metadata.get("memory_grounding_activated") is not True
        assert not any(
            "memory_grounding_activated" in r.adjustment_reason
            for r in state.calibration_history
        )

    def test_memory_grounding_below_safety_threshold_still_rejects_first(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """When confidence < 0.3, safety rejection happens before grounding."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.2,
            calibrated_confidence=0.2,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="low",
        )
        result = layer.run(state)
        # Safety rejection should happen first
        assert "safety_rejected" in result.error_flags
        # Grounding should NOT be activated since we already returned early
        assert result.metadata.get("memory_grounding_activated") is not True


# ---------------------------------------------------------------------------
# Envelope propagation tests
# ---------------------------------------------------------------------------

class TestEnvelopePropagation:
    def test_envelope_propagation_steps_correct(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Default config: 5 steps."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.85,
            calibrated_confidence=0.85,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="high",
        )
        layer.run(state)
        assert state.metadata.get("envelope_steps") == 5

    def test_envelope_propagation_decay(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Strength decays by 0.95 each step: 1.0 * 0.95^5."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.85,
            calibrated_confidence=0.85,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="high",
        )
        layer.run(state)
        strength = state.metadata.get("envelope_strength")
        expected = 0.95**5
        assert abs(strength - expected) < 1e-6

    def test_envelope_propagation_custom_decay(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Custom decay factor: 1.0 * 0.8^5."""
        config = BoundaryEnforcementConfig(envelope_decay_factor=0.8)
        layer = BoundaryEnforcementLayer(config=config)
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.85,
            calibrated_confidence=0.85,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="high",
        )
        layer.run(state)
        strength = state.metadata.get("envelope_strength")
        expected = 0.8**5
        assert abs(strength - expected) < 1e-6
        assert state.metadata.get("envelope_steps") == 5

    def test_envelope_propagation_custom_steps(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Custom steps: 3 steps."""
        config = BoundaryEnforcementConfig(max_envelope_propagation_steps=3)
        layer = BoundaryEnforcementLayer(config=config)
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.85,
            calibrated_confidence=0.85,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="high",
        )
        layer.run(state)
        assert state.metadata.get("envelope_steps") == 3
        strength = state.metadata.get("envelope_strength")
        expected = 0.95**3
        assert abs(strength - expected) < 1e-6

    def test_envelope_propagation_disabled_no_metadata(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """When disabled, no envelope metadata set."""
        config = BoundaryEnforcementConfig(enable_envelope_propagation=False)
        layer = BoundaryEnforcementLayer(config=config)
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.85,
            calibrated_confidence=0.85,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="high",
        )
        layer.run(state)
        assert "envelope_steps" not in state.metadata
        assert "envelope_strength" not in state.metadata

    def test_envelope_propagation_disabled_zero_steps(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """max_envelope_propagation_steps=0 → steps=0, strength=1.0 (no propagation)."""
        config = BoundaryEnforcementConfig(max_envelope_propagation_steps=0)
        layer = BoundaryEnforcementLayer(config=config)
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.85,
            calibrated_confidence=0.85,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="high",
        )
        layer.run(state)
        assert state.metadata.get("envelope_steps") == 0
        assert state.metadata.get("envelope_strength") == 1.0


# ---------------------------------------------------------------------------
# Boundary violation detection tests
# ---------------------------------------------------------------------------

class TestBoundaryViolations:
    def test_confidence_out_of_range_negative(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Negative calibrated confidence → confidence_out_of_range."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=-0.1,
            calibrated_confidence=-0.1,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="unknown",
        )
        # Safety bounds would reject first, so disable for this test
        config = BoundaryEnforcementConfig(enable_safety_bounds=False)
        layer = BoundaryEnforcementLayer(config=config)
        result = layer.run(state)
        assert "confidence_out_of_range" in result.error_flags

    def test_confidence_out_of_range_above_one(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Confidence > 1.0 → confidence_out_of_range."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=1.5,
            calibrated_confidence=1.5,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="unknown",
        )
        config = BoundaryEnforcementConfig(enable_safety_bounds=False)
        layer = BoundaryEnforcementLayer(config=config)
        result = layer.run(state)
        assert "confidence_out_of_range" in result.error_flags

    def test_reasoning_too_long(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Reasoning content > 50000 chars → reasoning_too_long."""
        long_content = "x" * (_MAX_REASONING_LENGTH + 1)
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content=long_content,
            raw_confidence=0.85,
            calibrated_confidence=0.85,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="high",
        )
        result = layer.run(state)
        assert "reasoning_too_long" in result.error_flags

    def test_reasoning_exactly_at_limit_not_flagged(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Reasoning content == 50000 chars → no violation."""
        content = "x" * _MAX_REASONING_LENGTH
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content=content,
            raw_confidence=0.85,
            calibrated_confidence=0.85,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="high",
        )
        result = layer.run(state)
        assert "reasoning_too_long" not in result.error_flags

    def test_mode_mismatch_high_confidence_wrong_mode(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """High confidence (>0.85) but TREE_OF_THOUGHT mode → mode_mismatch."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.9,
            calibrated_confidence=0.9,
            reasoning_mode=ReasoningMode.TREE_OF_THOUGHT,  # Wrong for high conf
            confidence_tag="high",
        )
        result = layer.run(state)
        assert "mode_mismatch" in result.error_flags

    def test_mode_mismatch_low_confidence_wrong_mode(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Low confidence (<=0.5) but DIRECT mode → mode_mismatch."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.3,
            calibrated_confidence=0.3,
            reasoning_mode=ReasoningMode.DIRECT,  # Wrong for low conf
            confidence_tag="low",
        )
        # Disable safety so we don't get rejected first
        config = BoundaryEnforcementConfig(enable_safety_bounds=False)
        layer = BoundaryEnforcementLayer(config=config)
        result = layer.run(state)
        assert "mode_mismatch" in result.error_flags

    def test_mode_correct_for_confidence_no_mismatch(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Correct mode for confidence → no mode_mismatch."""
        # 0.9 → DIRECT is correct
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.9,
            calibrated_confidence=0.9,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="high",
        )
        result = layer.run(state)
        assert "mode_mismatch" not in result.error_flags

    def test_no_violations_valid_state(
        self,
        layer: BoundaryEnforcementLayer,
        base_state: CalibratedReasoningState,
    ):
        """Valid state with no boundary issues → no violations."""
        result = layer.run(base_state)
        assert "confidence_out_of_range" not in result.error_flags
        assert "mode_mismatch" not in result.error_flags
        assert "reasoning_too_long" not in result.error_flags


# ---------------------------------------------------------------------------
# Independent feature disable tests
# ---------------------------------------------------------------------------

class TestIndependentDisables:
    def test_all_three_features_disabled(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """All three features disabled → state passes through mostly unchanged."""
        config = BoundaryEnforcementConfig(
            enable_safety_bounds=False,
            enable_memory_grounding=False,
            enable_envelope_propagation=False,
        )
        layer = BoundaryEnforcementLayer(config=config)
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.2,
            calibrated_confidence=0.2,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="low",
        )
        result = layer.run(state)
        # No safety rejection
        assert "safety_rejected" not in result.error_flags
        assert result.reasoning_content == "Answer"
        # No grounding
        assert result.metadata.get("memory_grounding_activated") is not True
        # No envelope
        assert "envelope_steps" not in result.metadata

    def test_safety_and_memory_disabled_envelope_enabled(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Safety and memory disabled, envelope enabled."""
        config = BoundaryEnforcementConfig(
            enable_safety_bounds=False,
            enable_memory_grounding=False,
            enable_envelope_propagation=True,
        )
        layer = BoundaryEnforcementLayer(config=config)
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.2,
            calibrated_confidence=0.2,
            reasoning_mode=ReasoningMode.TREE_OF_THOUGHT,
            confidence_tag="low",
        )
        result = layer.run(state)
        assert "safety_rejected" not in result.error_flags
        assert "envelope_steps" in result.metadata
        assert "envelope_strength" in result.metadata

    def test_safety_and_envelope_disabled_memory_enabled(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Safety and envelope disabled, memory grounding enabled."""
        config = BoundaryEnforcementConfig(
            enable_safety_bounds=False,
            enable_memory_grounding=True,
            enable_envelope_propagation=False,
        )
        layer = BoundaryEnforcementLayer(config=config)
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.5,
            calibrated_confidence=0.5,
            reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
            confidence_tag="medium",
        )
        result = layer.run(state)
        assert result.metadata.get("memory_grounding_activated") is True
        assert "envelope_steps" not in result.metadata

    def test_memory_and_envelope_disabled_safety_enabled(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Memory and envelope disabled, safety enabled."""
        config = BoundaryEnforcementConfig(
            enable_safety_bounds=True,
            enable_memory_grounding=False,
            enable_envelope_propagation=False,
        )
        layer = BoundaryEnforcementLayer(config=config)
        # Low confidence should still be rejected
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.1,
            calibrated_confidence=0.1,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="low",
        )
        result = layer.run(state)
        assert "safety_rejected" in result.error_flags


# ---------------------------------------------------------------------------
# _expected_reasoning_mode helper tests
# ---------------------------------------------------------------------------

class TestExpectedReasoningMode:
    def test_direct_for_high_confidence(self):
        assert _expected_reasoning_mode(0.86) == ReasoningMode.DIRECT
        assert _expected_reasoning_mode(0.9) == ReasoningMode.DIRECT
        assert _expected_reasoning_mode(1.0) == ReasoningMode.DIRECT

    def test_cot_for_medium_confidence(self):
        assert _expected_reasoning_mode(0.51) == ReasoningMode.CHAIN_OF_THOUGHT
        assert _expected_reasoning_mode(0.6) == ReasoningMode.CHAIN_OF_THOUGHT
        assert _expected_reasoning_mode(0.85) == ReasoningMode.CHAIN_OF_THOUGHT

    def test_tree_for_low_confidence(self):
        assert _expected_reasoning_mode(0.0) == ReasoningMode.TREE_OF_THOUGHT
        assert _expected_reasoning_mode(0.3) == ReasoningMode.TREE_OF_THOUGHT
        assert _expected_reasoning_mode(0.5) == ReasoningMode.TREE_OF_THOUGHT
        # At exactly 0.5, it's the boundary — cot > 0.5, so 0.5 goes to tree


# ---------------------------------------------------------------------------
# Integration / run method tests
# ---------------------------------------------------------------------------

class TestRunMethod:
    def test_run_returns_same_state_object(
        self,
        layer: BoundaryEnforcementLayer,
        base_state: CalibratedReasoningState,
    ):
        """run() returns the same (modified) state object."""
        result = layer.run(base_state)
        assert result is base_state

    def test_run_with_mode_result_param(
        self,
        layer: BoundaryEnforcementLayer,
        base_state: CalibratedReasoningState,
    ):
        """run() accepts mode_result parameter for compatibility."""
        result = layer.run(base_state, mode_result="some_mode_result")
        assert result is base_state
        assert "safety_rejected" not in result.error_flags

    def test_run_adds_envelope_metadata_high_confidence(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """High confidence with all features enabled gets envelope metadata."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.9,
            calibrated_confidence=0.9,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="high",
        )
        layer.run(state)
        assert state.metadata.get("envelope_steps") == 5
        assert abs(state.metadata.get("envelope_strength") - 0.95**5) < 1e-6

    def test_run_high_confidence_no_memory_grounding(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """High confidence → memory grounding NOT activated."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.9,
            calibrated_confidence=0.9,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="high",
        )
        layer.run(state)
        assert state.metadata.get("memory_grounding_activated") is not True

    def test_run_medium_confidence_triggers_memory_grounding(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Medium confidence (0.5) triggers memory grounding."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.5,
            calibrated_confidence=0.5,
            reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
            confidence_tag="medium",
        )
        layer.run(state)
        assert state.metadata.get("memory_grounding_activated") is True

    def test_run_idempotent_multiple_calls(
        self,
        layer: BoundaryEnforcementLayer,
    ):
        """Calling run() multiple times is safe (records accumulate)."""
        state = CalibratedReasoningState(
            prompt="Test",
            reasoning_content="Answer",
            raw_confidence=0.5,
            calibrated_confidence=0.5,
            reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
            confidence_tag="medium",
        )
        layer.run(state)
        layer.run(state)
        # Should have grounding activation recorded twice
        grounding_records = [
            r for r in state.calibration_history
            if "memory_grounding_activated" in r.adjustment_reason
        ]
        assert len(grounding_records) == 2
