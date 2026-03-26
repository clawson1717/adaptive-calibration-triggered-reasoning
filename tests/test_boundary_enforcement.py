"""Tests for the boundary enforcement module (Step 8)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is on the path
SRC_ROOT = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from actr.boundary_enforcement import (
    BoundaryEnforcementConfig,
    BoundaryEnforcementLayer,
    BoundaryViolation,
    BoundaryViolationError,
    EnforcedReasoningState,
    GateAction,
    GateResult,
    ReasoningContext,
    ReasoningEnvelopeState,
    ReasoningStep,
    ViolationSeverity,
)
from actr.data import CalibratedReasoningState, ReasoningMode


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
def sample_reasoning_step() -> ReasoningStep:
    return ReasoningStep(
        step_id="step_1",
        content="The answer is 42 because of reasoning.",
        confidence=0.5,
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_context() -> ReasoningContext:
    return ReasoningContext(
        prompt="What is the meaning of life?",
        prior_steps=[],
        constraints={"accuracy": 0.9},
    )


@pytest.fixture
def sample_calibrated_state() -> CalibratedReasoningState:
    return CalibratedReasoningState(
        prompt="What is 2+2?",
        reasoning_content="2+2 equals 4 because adding two to two gives four.",
        raw_confidence=0.8,
        calibrated_confidence=0.85,
        reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
        confidence_tag="high",
        model_name="test-model",
        reasoning_steps=["First, consider what 2+2 means", "Adding two units to two units gives four"],
    )


# ---------------------------------------------------------------------------
# BoundaryEnforcementConfig Tests
# ---------------------------------------------------------------------------

class TestBoundaryEnforcementConfig:
    def test_default_values(self) -> None:
        config = BoundaryEnforcementConfig()
        assert config.memory_grounding_threshold == 0.7
        assert config.inference_safety_bound == 0.3
        assert config.envelope_softness == 0.15
        assert config.max_envelope_depth == 5

    def test_custom_values(self) -> None:
        config = BoundaryEnforcementConfig(
            memory_grounding_threshold=0.8,
            inference_safety_bound=0.4,
            envelope_softness=0.2,
            max_envelope_depth=10,
        )
        assert config.memory_grounding_threshold == 0.8
        assert config.inference_safety_bound == 0.4
        assert config.envelope_softness == 0.2
        assert config.max_envelope_depth == 10


# ---------------------------------------------------------------------------
# BoundaryEnforcementLayer.gate Tests
# ---------------------------------------------------------------------------

class TestGate:
    def test_gate_pass_high_confidence(
        self,
        layer: BoundaryEnforcementLayer,
        sample_reasoning_step: ReasoningStep,
        sample_context: ReasoningContext,
    ) -> None:
        """p=0.9 should result in PASS action."""
        result = layer.gate(sample_reasoning_step, confidence=0.9, context=sample_context)
        assert result.action == GateAction.PASS
        assert "Passed" in result.reason
        assert result.adjusted_confidence == 0.9

    def test_gate_reject_below_safety_bound(
        self,
        layer: BoundaryEnforcementLayer,
        sample_reasoning_step: ReasoningStep,
        sample_context: ReasoningContext,
    ) -> None:
        """p=0.2 (below 0.3 safety bound) should result in REJECT action."""
        result = layer.gate(sample_reasoning_step, confidence=0.2, context=sample_context)
        assert result.action == GateAction.REJECT
        assert "Rejected" in result.reason
        assert len(result.violations) >= 1
        assert any(v.severity == ViolationSeverity.CRITICAL for v in result.violations)

    def test_gate_ground_memory_below_threshold(
        self,
        layer: BoundaryEnforcementLayer,
        sample_reasoning_step: ReasoningStep,
        sample_context: ReasoningContext,
    ) -> None:
        """p=0.5 (below 0.7 grounding threshold but above 0.3 bound) should GROUND."""
        result = layer.gate(sample_reasoning_step, confidence=0.5, context=sample_context)
        assert result.action == GateAction.GROUND
        assert "Ground" in result.reason
        assert result.adjusted_confidence == 0.5

    def test_gate_escalate_low_confidence(
        self,
        layer: BoundaryEnforcementLayer,
        sample_reasoning_step: ReasoningStep,
        sample_context: ReasoningContext,
    ) -> None:
        """p=0.15 (critically low) should result in ESCALATE action."""
        result = layer.gate(sample_reasoning_step, confidence=0.15, context=sample_context)
        assert result.action == GateAction.ESCALATE
        assert "Escalated" in result.reason
        assert result.adjusted_confidence == 0.15

    def test_gate_boundary_70_returns_pass(
        self,
        layer: BoundaryEnforcementLayer,
        sample_reasoning_step: ReasoningStep,
        sample_context: ReasoningContext,
    ) -> None:
        """p=0.7 at threshold should PASS (threshold is exclusive)."""
        result = layer.gate(sample_reasoning_step, confidence=0.7, context=sample_context)
        assert result.action == GateAction.PASS

    def test_gate_at_safety_bound_grounds(
        self,
        layer: BoundaryEnforcementLayer,
        sample_reasoning_step: ReasoningStep,
        sample_context: ReasoningContext,
    ) -> None:
        """p=0.3 at safety bound is safe (>=0.3) but below memory threshold (0.7) -> GROUND."""
        result = layer.gate(sample_reasoning_step, confidence=0.3, context=sample_context)
        # p=0.3 is safe (>= 0.3) but below memory threshold (0.7), so GROUND
        assert result.action == GateAction.GROUND


# ---------------------------------------------------------------------------
# BoundaryEnforcementLayer.check_safety_bound Tests
# ---------------------------------------------------------------------------

class TestCheckSafetyBound:
    def test_safety_bound_above_threshold(self, layer: BoundaryEnforcementLayer) -> None:
        """p=0.4 >= 0.3 should return True (safe)."""
        assert layer.check_safety_bound(0.4) is True

    def test_safety_bound_below_threshold(self, layer: BoundaryEnforcementLayer) -> None:
        """p=0.25 < 0.3 should return False (unsafe)."""
        assert layer.check_safety_bound(0.25) is False

    def test_safety_bound_at_exact_boundary(self, layer: BoundaryEnforcementLayer) -> None:
        """p=0.3 == 0.3 should return True (inclusive boundary)."""
        assert layer.check_safety_bound(0.3) is True

    def test_safety_bound_high_confidence(self, layer: BoundaryEnforcementLayer) -> None:
        """p=0.9 should return True."""
        assert layer.check_safety_bound(0.9) is True

    def test_safety_bound_custom_config(self) -> None:
        """Custom safety bound should be respected."""
        config = BoundaryEnforcementConfig(inference_safety_bound=0.5)
        layer = BoundaryEnforcementLayer(config)
        assert layer.check_safety_bound(0.4) is False
        assert layer.check_safety_bound(0.5) is True
        assert layer.check_safety_bound(0.6) is True


# ---------------------------------------------------------------------------
# BoundaryEnforcementLayer.should_ground_memory Tests
# ---------------------------------------------------------------------------

class TestShouldGroundMemory:
    def test_should_ground_memory_below(self, layer: BoundaryEnforcementLayer) -> None:
        """p=0.6 < 0.7 should return True (should ground)."""
        assert layer.should_ground_memory(0.6) is True

    def test_should_ground_memory_above(self, layer: BoundaryEnforcementLayer) -> None:
        """p=0.8 > 0.7 should return False (no grounding needed)."""
        assert layer.should_ground_memory(0.8) is False

    def test_should_ground_memory_at_threshold(self, layer: BoundaryEnforcementLayer) -> None:
        """p=0.7 == 0.7 should return False (exclusive threshold)."""
        assert layer.should_ground_memory(0.7) is False

    def test_should_ground_memory_very_low(self, layer: BoundaryEnforcementLayer) -> None:
        """p=0.1 should return True."""
        assert layer.should_ground_memory(0.1) is True

    def test_should_ground_memory_custom_threshold(self) -> None:
        """Custom threshold should be respected."""
        config = BoundaryEnforcementConfig(memory_grounding_threshold=0.5)
        layer = BoundaryEnforcementLayer(config)
        assert layer.should_ground_memory(0.4) is True
        assert layer.should_ground_memory(0.5) is False
        assert layer.should_ground_memory(0.6) is False


# ---------------------------------------------------------------------------
# BoundaryEnforcementLayer.propagate_envelope Tests
# ---------------------------------------------------------------------------

class TestEnvelopePropagation:
    def test_envelope_propagation_basic(self, layer: BoundaryEnforcementLayer) -> None:
        """Constraints should propagate forward with decay."""
        state = ReasoningEnvelopeState(
            active_constraints={"accuracy": 1.0},
            propagated_constraints={},
            depth=0,
        )
        result = layer.propagate_envelope(state)
        # After one step with softness=0.15: 1.0 * (1-0.15)^1 = 0.85
        assert "accuracy" in result.propagated_constraints
        assert result.depth == 1

    def test_envelope_propagation_with_existing_constraints(
        self, layer: BoundaryEnforcementLayer
    ) -> None:
        """Existing propagated constraints should be preserved and decayed."""
        state = ReasoningEnvelopeState(
            active_constraints={"accuracy": 1.0},
            propagated_constraints={"precision": 0.9},
            depth=1,
        )
        result = layer.propagate_envelope(state)
        # precision should be decayed by 0.85: 0.9 * 0.85 = 0.765
        assert "precision" in result.propagated_constraints
        assert result.propagated_constraints["precision"] < 0.9
        assert result.propagated_constraints["precision"] > 0.7  # sanity check
        assert result.depth == 2

    def test_envelope_propagation_respects_max_depth(self, layer: BoundaryEnforcementLayer) -> None:
        """Propagation should stop at max_envelope_depth."""
        config = BoundaryEnforcementConfig(max_envelope_depth=3)
        layer = BoundaryEnforcementLayer(config)
        state = ReasoningEnvelopeState(
            active_constraints={"accuracy": 1.0},
            propagated_constraints={},
            depth=3,  # Already at max
        )
        result = layer.propagate_envelope(state)
        # Depth should not increase past max
        assert result.depth == 3
        assert result.propagated_constraints == {}

    def test_envelope_propagation_decays_with_depth(self, layer: BoundaryEnforcementLayer) -> None:
        """Multiple propagations should decay constraints progressively."""
        state = ReasoningEnvelopeState(
            active_constraints={"accuracy": 1.0},
            propagated_constraints={},
            depth=0,
        )
        # Propagate multiple times
        for expected_depth in range(1, 4):
            state = layer.propagate_envelope(state)
            assert state.depth == expected_depth
            if "accuracy" in state.propagated_constraints:
                assert state.propagated_constraints["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# BoundaryEnforcementLayer.enforce Tests
# ---------------------------------------------------------------------------

class TestEnforce:
    def test_enforce_with_violations(
        self,
        layer: BoundaryEnforcementLayer,
        sample_calibrated_state: CalibratedReasoningState,
    ) -> None:
        """Full enforcement run should detect violations and return EnforcedReasoningState."""
        # State with low confidence should trigger grounding
        state = CalibratedReasoningState(
            prompt="Is this true?",
            reasoning_content="Based on limited data, probably yes.",
            raw_confidence=0.5,
            calibrated_confidence=0.5,
            reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
            confidence_tag="medium",
            model_name="test-model",
            reasoning_steps=["Consider the evidence", "Evidence is limited"],
        )
        enforced = layer.enforce(state)
        assert isinstance(enforced, EnforcedReasoningState)
        assert enforced.enforcement_applied is True
        assert len(enforced.gate_decisions) >= 1
        assert len(enforced.boundary_violations) >= 1
        # Low confidence should ground
        assert any(g.action == GateAction.GROUND for g in enforced.gate_decisions)

    def test_enforce_high_confidence_passes(
        self,
        layer: BoundaryEnforcementLayer,
        sample_calibrated_state: CalibratedReasoningState,
    ) -> None:
        """High confidence state should pass enforcement without violations."""
        state = CalibratedReasoningState(
            prompt="Simple math question",
            reasoning_content="2+2=4",
            raw_confidence=0.95,
            calibrated_confidence=0.95,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="high",
            model_name="test-model",
            reasoning_steps=["2+2 is 4"],
        )
        enforced = layer.enforce(state)
        assert all(g.action == GateAction.PASS for g in enforced.gate_decisions)
        assert len(enforced.boundary_violations) == 0

    def test_enforce_empty_reasoning_steps(
        self,
        layer: BoundaryEnforcementLayer,
        sample_calibrated_state: CalibratedReasoningState,
    ) -> None:
        """State with no reasoning steps should still create a gate decision from content."""
        state = CalibratedReasoningState(
            prompt="Quick question",
            reasoning_content="The answer is yes.",
            raw_confidence=0.6,
            calibrated_confidence=0.6,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="medium",
            model_name="test-model",
            reasoning_steps=[],  # No explicit steps
        )
        enforced = layer.enforce(state)
        # Should still have at least one gate decision from the content
        assert len(enforced.gate_decisions) >= 1

    def test_enforce_returns_enforced_state(
        self,
        layer: BoundaryEnforcementLayer,
        sample_calibrated_state: CalibratedReasoningState,
    ) -> None:
        """enforce should return EnforcedReasoningState with correct structure."""
        enforced = layer.enforce(sample_calibrated_state)
        assert hasattr(enforced, "base_state")
        assert hasattr(enforced, "boundary_violations")
        assert hasattr(enforced, "envelope_state")
        assert hasattr(enforced, "gate_decisions")
        assert hasattr(enforced, "enforcement_applied")


# ---------------------------------------------------------------------------
# BoundaryViolationError Tests
# ---------------------------------------------------------------------------

class TestBoundaryViolationError:
    def test_boundary_violation_error_with_message(self) -> None:
        """Error should store and display the message."""
        error = BoundaryViolationError("Safety bound violated")
        assert str(error) == "Safety bound violated"
        assert error.violations == []

    def test_boundary_violation_error_with_violations(self) -> None:
        """Error should store violations list."""
        violations = [
            BoundaryViolation(
                reason="Below safety bound",
                severity=ViolationSeverity.CRITICAL,
                step="step_1",
            )
        ]
        error = BoundaryViolationError("Rejected", violations=violations)
        assert error.message == "Rejected"
        assert len(error.violations) == 1
        assert error.violations[0].severity == ViolationSeverity.CRITICAL

    def test_boundary_violation_error_rounds_through_exception_chain(self) -> None:
        """Error should be raiseable and catchable as Exception."""
        error = BoundaryViolationError("test")
        with pytest.raises(BoundaryViolationError):
            raise error


# ---------------------------------------------------------------------------
# Supporting Types Tests
# ---------------------------------------------------------------------------

class TestBoundaryViolation:
    def test_to_dict(self) -> None:
        violation = BoundaryViolation(
            reason="test reason",
            severity=ViolationSeverity.WARN,
            step="step_1",
        )
        d = violation.to_dict()
        assert d["reason"] == "test reason"
        assert d["severity"] == "warn"
        assert d["step"] == "step_1"


class TestReasoningStep:
    def test_defaults(self) -> None:
        step = ReasoningStep(step_id="1", content="test")
        assert step.confidence == 0.5
        assert step.metadata == {}


class TestReasoningEnvelopeState:
    def test_defaults(self) -> None:
        state = ReasoningEnvelopeState()
        assert state.active_constraints == {}
        assert state.propagated_constraints == {}
        assert state.depth == 0


class TestGateResult:
    def test_defaults(self) -> None:
        result = GateResult(action=GateAction.PASS, reason="ok")
        assert result.adjusted_confidence == 0.0
        assert result.violations == []

    def test_to_dict(self) -> None:
        result = GateResult(
            action=GateAction.PASS,
            reason="ok",
            adjusted_confidence=0.9,
        )
        d = result.to_dict()
        assert d["action"] == "pass"
        assert d["adjusted_confidence"] == 0.9


class TestEnforcedReasoningState:
    def test_has_critical_violations_false(self) -> None:
        state = CalibratedReasoningState(prompt="test", reasoning_content="test")
        enforced = EnforcedReasoningState(
            base_state=state,
            boundary_violations=[
                BoundaryViolation(
                    reason="info",
                    severity=ViolationSeverity.INFO,
                    step="step_1",
                )
            ],
        )
        assert enforced.has_critical_violations is False

    def test_has_critical_violations_true(self) -> None:
        state = CalibratedReasoningState(prompt="test", reasoning_content="test")
        enforced = EnforcedReasoningState(
            base_state=state,
            boundary_violations=[
                BoundaryViolation(
                    reason="critical",
                    severity=ViolationSeverity.CRITICAL,
                    step="step_1",
                )
            ],
        )
        assert enforced.has_critical_violations is True


# ---------------------------------------------------------------------------
# Integration / Edge Cases
# ---------------------------------------------------------------------------

class TestBoundaryEnforcementEdgeCases:
    def test_very_low_confidence_escalated(self, layer: BoundaryEnforcementLayer) -> None:
        """Critically low confidence (near 0) should be escalated (p <= 0.15)."""
        step = ReasoningStep(step_id="low", content="maybe")
        result = layer.gate(step, confidence=0.01, context=ReasoningContext())
        # p=0.01 is critically low (<= 0.15) → ESCALATE
        assert result.action == GateAction.ESCALATE

    def test_confidence_at_85_boundary(
        self,
        layer: BoundaryEnforcementLayer,
    ) -> None:
        """p=0.85 should PASS (above all thresholds)."""
        step = ReasoningStep(step_id="boundary", content="test")
        result = layer.gate(step, confidence=0.85, context=ReasoningContext())
        assert result.action == GateAction.PASS

    def test_enforce_with_error_flags(
        self,
        layer: BoundaryEnforcementLayer,
        sample_calibrated_state: CalibratedReasoningState,
    ) -> None:
        """Enforce should preserve base state error flags."""
        state = CalibratedReasoningState(
            prompt="test",
            reasoning_content="test",
            calibrated_confidence=0.9,
            reasoning_mode=ReasoningMode.DIRECT,
            confidence_tag="high",
        )
        state.add_error("test_error")
        enforced = layer.enforce(state)
        assert "test_error" in enforced.base_state.error_flags

    def test_multiple_reasoning_steps_all_grounded(
        self,
        layer: BoundaryEnforcementLayer,
    ) -> None:
        """Multiple low-confidence steps should all be grounded."""
        state = CalibratedReasoningState(
            prompt="Complex reasoning",
            reasoning_content="Multi-step reasoning",
            calibrated_confidence=0.5,
            reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
            confidence_tag="medium",
            reasoning_steps=["Step 1", "Step 2", "Step 3"],
        )
        enforced = layer.enforce(state)
        # All steps should be grounded since confidence is below threshold
        assert all(g.action == GateAction.GROUND for g in enforced.gate_decisions)
