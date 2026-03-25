"""Tests for the Reasoning Mode Controller (Step 4)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Ensure src/ is on the path
SRC_ROOT = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from actr.mode_controller import (
    ReasoningModeController,
    ReasoningModeEnum,
    ModeSelectionResult,
)
from actr.data import CalibratedReasoningState, ReasoningMode
from actr.config import ACTRConfig, ConfidenceThresholds


# ---------------------------------------------------------------------------
# ReasoningModeEnum Tests
# ---------------------------------------------------------------------------

class TestReasoningModeEnum:
    @pytest.mark.parametrize("mode", [ReasoningModeEnum.FAST, ReasoningModeEnum.MODERATE, ReasoningModeEnum.SLOW])
    def test_mode_is_string_enum(self, mode: ReasoningModeEnum) -> None:
        assert isinstance(mode.value, str)

    @pytest.mark.parametrize("mode", [ReasoningModeEnum.FAST, ReasoningModeEnum.MODERATE, ReasoningModeEnum.SLOW])
    def test_description_non_empty(self, mode: ReasoningModeEnum) -> None:
        desc = mode.description()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_fast_n_candidates(self) -> None:
        assert ReasoningModeEnum.FAST.n_candidates == 1

    def test_moderate_n_candidates(self) -> None:
        assert ReasoningModeEnum.MODERATE.n_candidates == 2

    def test_slow_n_candidates(self) -> None:
        assert ReasoningModeEnum.SLOW.n_candidates == 3

    def test_fast_uses_knowledge_grounding(self) -> None:
        assert ReasoningModeEnum.FAST.uses_knowledge_grounding is False

    def test_slow_uses_knowledge_grounding(self) -> None:
        assert ReasoningModeEnum.SLOW.uses_knowledge_grounding is True

    def test_verification_depth_fast(self) -> None:
        assert ReasoningModeEnum.FAST.verification_depth == "minimal"

    def test_verification_depth_moderate(self) -> None:
        assert ReasoningModeEnum.MODERATE.verification_depth == "standard"

    def test_verification_depth_slow(self) -> None:
        assert ReasoningModeEnum.SLOW.verification_depth == "deep"


# ---------------------------------------------------------------------------
# ModeSelectionResult Tests
# ---------------------------------------------------------------------------

class TestModeSelectionResult:
    def test_basic_creation(self) -> None:
        result = ModeSelectionResult(
            selected_mode=ReasoningModeEnum.FAST,
            confidence=0.9,
            confidence_tag="high",
            transition_reason="confidence=0.900, tag='high', mode=fast",
        )
        assert result.selected_mode == ReasoningModeEnum.FAST
        assert result.confidence == 0.9
        assert result.confidence_tag == "high"
        assert result.previous_mode is None
        assert result.is_transition is False

    def test_is_transition_flag_true(self) -> None:
        result = ModeSelectionResult(
            selected_mode=ReasoningModeEnum.SLOW,
            confidence=0.3,
            confidence_tag="low",
            transition_reason="...",
            previous_mode=ReasoningModeEnum.FAST,
            is_transition=True,
        )
        assert result.is_transition is True
        assert result.previous_mode == ReasoningModeEnum.FAST

    def test_to_dict(self) -> None:
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = ModeSelectionResult(
            selected_mode=ReasoningModeEnum.MODERATE,
            confidence=0.7,
            confidence_tag="medium",
            transition_reason="test",
            timestamp=ts,
        )
        d = result.to_dict()
        assert d["selected_mode"] == "moderate"
        assert d["confidence"] == 0.7
        assert d["confidence_tag"] == "medium"
        assert d["is_transition"] is False
        assert d["previous_mode"] is None


# ---------------------------------------------------------------------------
# ReasoningModeController Tests
# ---------------------------------------------------------------------------

class TestReasoningModeController:
    def setup_method(self) -> None:
        self.controller = ReasoningModeController()

    # --- Core select_mode tests ---

    def test_select_mode_fast_high_confidence(self) -> None:
        result = self.controller.select_mode(0.9)
        assert result.selected_mode == ReasoningModeEnum.FAST
        assert result.confidence_tag == "high"
        assert isinstance(result.transition_reason, str)
        assert "0.9" in result.transition_reason or "0.900" in result.transition_reason

    def test_select_mode_fast_with_previous_non_fast(self) -> None:
        result = self.controller.select_mode(0.9, previous_mode=ReasoningModeEnum.SLOW)
        assert result.selected_mode == ReasoningModeEnum.FAST
        assert result.is_transition is True
        assert result.previous_mode == ReasoningModeEnum.SLOW

    def test_select_mode_moderate(self) -> None:
        result = self.controller.select_mode(0.7)
        assert result.selected_mode == ReasoningModeEnum.MODERATE
        assert result.confidence_tag == "medium"

    def test_select_mode_slow_low_confidence(self) -> None:
        result = self.controller.select_mode(0.3)
        assert result.selected_mode == ReasoningModeEnum.SLOW
        assert result.confidence_tag == "low"

    def test_select_mode_boundary_high(self) -> None:
        """p=0.85 exactly is NOT fast because fast requires p > 0.85."""
        result = self.controller.select_mode(0.85)
        assert result.selected_mode != ReasoningModeEnum.FAST
        # p=0.85 is "high" tag per ConfidenceThresholds (high=0.8)
        # but 0.85 > 0.85 is False, so it's MODERATE
        assert result.selected_mode == ReasoningModeEnum.MODERATE

    def test_select_mode_boundary_medium_low(self) -> None:
        """p=0.5 exactly is SLOW because moderate requires p > 0.5."""
        result = self.controller.select_mode(0.5)
        assert result.selected_mode == ReasoningModeEnum.SLOW

    def test_select_mode_unknown_confidence(self) -> None:
        """p=1.5 is out of range -> 'unknown' tag -> SLOW."""
        result = self.controller.select_mode(1.5)
        assert result.selected_mode == ReasoningModeEnum.SLOW
        assert result.confidence_tag == "unknown"

    def test_select_mode_unknown_negative(self) -> None:
        """p=-0.1 is out of range -> 'unknown' tag -> SLOW."""
        result = self.controller.select_mode(-0.1)
        assert result.selected_mode == ReasoningModeEnum.SLOW
        assert result.confidence_tag == "unknown"

    # --- select_mode_from_state ---

    def test_select_mode_from_state(self) -> None:
        state = CalibratedReasoningState(
            prompt="What is 2+2?",
            reasoning_content="4",
            calibrated_confidence=0.75,
            confidence_tag="medium",
        )
        result = self.controller.select_mode_from_state(state)
        assert result.selected_mode == ReasoningModeEnum.MODERATE
        assert result.confidence == 0.75

    # --- batch_select ---

    def test_batch_select(self) -> None:
        scores = [0.9, 0.7, 0.3, 1.5, -0.1]
        results = self.controller.batch_select(scores)
        assert len(results) == 5
        assert results[0].selected_mode == ReasoningModeEnum.FAST
        assert results[1].selected_mode == ReasoningModeEnum.MODERATE
        assert results[2].selected_mode == ReasoningModeEnum.SLOW
        assert results[3].selected_mode == ReasoningModeEnum.SLOW
        assert results[4].selected_mode == ReasoningModeEnum.SLOW

    # --- Transition detection ---

    def test_transition_detection(self) -> None:
        """Consecutive calls with different modes detect is_transition=True on the switch."""
        r1 = self.controller.select_mode(0.9)
        assert r1.is_transition is False  # No previous mode

        r2 = self.controller.select_mode(0.3, previous_mode=r1.selected_mode)
        assert r2.is_transition is True
        assert r2.previous_mode == ReasoningModeEnum.FAST

        r3 = self.controller.select_mode(0.9, previous_mode=r2.selected_mode)
        assert r3.is_transition is True
        assert r3.previous_mode == ReasoningModeEnum.SLOW

    def test_no_transition_same_mode(self) -> None:
        """Same mode consecutively -> is_transition=False."""
        r1 = self.controller.select_mode(0.7)
        r2 = self.controller.select_mode(0.75, previous_mode=r1.selected_mode)
        assert r2.is_transition is False
        assert r2.previous_mode == ReasoningModeEnum.MODERATE

    # --- get_transition_summary ---

    def test_get_transition_summary(self) -> None:
        results = [
            self.controller.select_mode(0.9),          # FAST
            self.controller.select_mode(0.3, previous_mode=ReasoningModeEnum.FAST),   # SLOW (transition)
            self.controller.select_mode(0.7, previous_mode=ReasoningModeEnum.SLOW),   # MODERATE (transition)
            self.controller.select_mode(0.6, previous_mode=ReasoningModeEnum.MODERATE),  # MODERATE (no transition)
            self.controller.select_mode(0.2, previous_mode=ReasoningModeEnum.MODERATE),  # SLOW (transition)
        ]
        summary = self.controller.get_transition_summary(results)
        assert summary["total_transitions"] == 3
        assert summary["mode_counts"]["fast"] == 1
        assert summary["mode_counts"]["moderate"] == 2
        assert summary["mode_counts"]["slow"] == 2
        assert "fast" in summary["avg_confidence_per_mode"]
        assert "moderate" in summary["avg_confidence_per_mode"]
        assert "slow" in summary["avg_confidence_per_mode"]
        # Check average for slow mode: (0.3 + 0.2) / 2 = 0.25
        assert abs(summary["avg_confidence_per_mode"]["slow"] - 0.25) < 1e-9

    # --- Custom thresholds ---

    def test_custom_thresholds(self) -> None:
        custom_config = ACTRConfig(
            thresholds=ConfidenceThresholds(low=0.3, medium=0.6, high=0.9),
        )
        controller = ReasoningModeController(config=custom_config)
        # p=0.5 with low=0.3, medium=0.6, high=0.9 -> "medium" tag (0.3 <= 0.5 < 0.9)
        result = controller.select_mode(0.5)
        assert result.confidence_tag == "medium"
        # But 0.5 > 0.5 is False -> not MODERATE per architecture -> SLOW
        assert result.selected_mode == ReasoningModeEnum.SLOW

    # --- get_mode_for_confidence_tag ---

    def test_get_mode_for_confidence_tag_high(self) -> None:
        assert self.controller.get_mode_for_confidence_tag("high") == ReasoningModeEnum.FAST

    def test_get_mode_for_confidence_tag_medium(self) -> None:
        assert self.controller.get_mode_for_confidence_tag("medium") == ReasoningModeEnum.MODERATE

    def test_get_mode_for_confidence_tag_low(self) -> None:
        assert self.controller.get_mode_for_confidence_tag("low") == ReasoningModeEnum.SLOW

    def test_get_mode_for_confidence_tag_unknown(self) -> None:
        assert self.controller.get_mode_for_confidence_tag("unknown") == ReasoningModeEnum.SLOW
