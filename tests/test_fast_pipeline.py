"""Tests for the Fast Mode pipeline (Step 5)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Ensure src/ is on the path
SRC_ROOT = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from actr.pipelines.fast_pipeline import (
    FastModePipeline,
    FastPipelineConfig,
)
from actr.data import CalibratedReasoningState, ReasoningMode, ConfidenceTag
from actr.mode_controller import ModeSelectionResult, ReasoningModeEnum
from actr.config import ACTRConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config() -> FastPipelineConfig:
    return FastPipelineConfig()


@pytest.fixture
def pipeline(default_config: FastPipelineConfig) -> FastModePipeline:
    return FastModePipeline(config=default_config)


@pytest.fixture
def fast_mode_result() -> ModeSelectionResult:
    """A ModeSelectionResult in FAST mode with high confidence."""
    return ModeSelectionResult(
        selected_mode=ReasoningModeEnum.FAST,
        confidence=0.9,
        confidence_tag="high",
        transition_reason="confidence=0.900, tag='high', mode=fast",
        timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# FastModePipeline Init Tests
# ---------------------------------------------------------------------------

class TestFastPipelineInit:
    def test_fast_pipeline_init(self) -> None:
        """FastModePipeline initializes without error."""
        pipeline = FastModePipeline()
        assert pipeline is not None
        assert isinstance(pipeline, FastModePipeline)

    def test_fast_pipeline_init_with_config(self, default_config: FastPipelineConfig) -> None:
        """FastModePipeline initializes with custom config."""
        pipeline = FastModePipeline(config=default_config)
        assert pipeline.config is default_config

    def test_fast_pipeline_init_with_actr_config(self) -> None:
        """FastModePipeline initializes with custom ACTR config."""
        actr_config = ACTRConfig()
        pipeline = FastModePipeline(actr_config=actr_config)
        assert pipeline.actr_config is actr_config


# ---------------------------------------------------------------------------
# FastPipelineConfig Tests
# ---------------------------------------------------------------------------

class TestFastPipelineConfig:
    def test_config_defaults(self) -> None:
        """Default config values are correct."""
        config = FastPipelineConfig()
        assert config.max_response_length == 10000
        assert config.min_response_length == 10
        assert "[FAST response for: {prompt}]" in config.stub_response_template

    def test_config_custom_values(self) -> None:
        """Custom config values are set correctly."""
        config = FastPipelineConfig(
            max_response_length=5000,
            min_response_length=5,
            stub_response_template="Custom: {prompt}",
        )
        assert config.max_response_length == 5000
        assert config.min_response_length == 5
        assert config.stub_response_template == "Custom: {prompt}"


# ---------------------------------------------------------------------------
# Shallow Heuristic Check Tests
# ---------------------------------------------------------------------------

class TestShallowHeuristicCheck:
    def test_shallow_check_passes_normal_text(self, pipeline: FastModePipeline) -> None:
        """Normal text passes heuristic."""
        text = "This is a normal response with enough content."
        passes, reason = pipeline._shallow_heuristic_check(text)
        assert passes is True
        assert reason == "passed"

    def test_shallow_check_fails_empty(self, pipeline: FastModePipeline) -> None:
        """Empty string fails."""
        passes, reason = pipeline._shallow_heuristic_check("")
        assert passes is False
        assert reason == "empty_response"

    def test_shallow_check_fails_whitespace(self, pipeline: FastModePipeline) -> None:
        """Whitespace-only fails."""
        passes, reason = pipeline._shallow_heuristic_check("   \t\n   ")
        assert passes is False
        assert reason == "empty_response"

    def test_shallow_check_fails_too_long(self, pipeline: FastModePipeline) -> None:
        """Text exceeding max_response_length fails."""
        long_text = "x" * 10001  # 10001 chars > 10000 limit
        passes, reason = pipeline._shallow_heuristic_check(long_text)
        assert passes is False
        assert reason == "excessively_long"

    def test_shallow_check_fails_newlines_only(self, pipeline: FastModePipeline) -> None:
        """Newlines-only fails (whitespace-only check)."""
        passes, reason = pipeline._shallow_heuristic_check("\n\n\n")
        assert passes is False
        assert reason == "empty_response"

    def test_shallow_check_single_char_fails(self, pipeline: FastModePipeline) -> None:
        """Single character fails (too short)."""
        passes, reason = pipeline._shallow_heuristic_check("x")
        assert passes is False
        assert reason == "too_short"

    def test_shallow_check_exactly_min_length_passes(self, pipeline: FastModePipeline) -> None:
        """Exactly min_response_length chars passes."""
        text = "x" * 10  # exactly 10 chars
        passes, reason = pipeline._shallow_heuristic_check(text)
        assert passes is True
        assert reason == "passed"

    def test_shallow_check_exactly_max_length_passes(self, pipeline: FastModePipeline) -> None:
        """Exactly max_response_length chars passes."""
        text = "x" * 10000  # exactly 10000 chars
        passes, reason = pipeline._shallow_heuristic_check(text)
        assert passes is True
        assert reason == "passed"


# ---------------------------------------------------------------------------
# Confidence Tag Tests
# ---------------------------------------------------------------------------

class TestConfidenceTag:
    def test_confidence_tag_high(self, pipeline: FastModePipeline) -> None:
        """Confidence > 0.85 maps to 'high'."""
        assert pipeline._confidence_tag_for_confidence(0.9) == "high"
        assert pipeline._confidence_tag_for_confidence(0.86) == "high"
        assert pipeline._confidence_tag_for_confidence(0.851) == "high"

    def test_confidence_tag_medium(self, pipeline: FastModePipeline) -> None:
        """0.5 < confidence <= 0.85 maps to 'medium'."""
        assert pipeline._confidence_tag_for_confidence(0.7) == "medium"
        assert pipeline._confidence_tag_for_confidence(0.51) == "medium"
        assert pipeline._confidence_tag_for_confidence(0.85) == "medium"

    def test_confidence_tag_low(self, pipeline: FastModePipeline) -> None:
        """confidence <= 0.5 maps to 'low'."""
        assert pipeline._confidence_tag_for_confidence(0.3) == "low"
        assert pipeline._confidence_tag_for_confidence(0.5) == "low"  # boundary: 0.5 NOT > 0.5
        assert pipeline._confidence_tag_for_confidence(0.0) == "low"
        assert pipeline._confidence_tag_for_confidence(0.49) == "low"


# ---------------------------------------------------------------------------
# Run Method Tests
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_returns_calibrated_reasoning_state(
        self,
        pipeline: FastModePipeline,
        fast_mode_result: ModeSelectionResult,
    ) -> None:
        """run() returns a CalibratedReasoningState."""
        state = pipeline.run("What is 2+2?", 0.9, fast_mode_result)
        assert isinstance(state, CalibratedReasoningState)

    def test_run_prompt_preserved(
        self,
        pipeline: FastModePipeline,
        fast_mode_result: ModeSelectionResult,
    ) -> None:
        """Returned state has the correct prompt."""
        prompt = "What is the capital of France?"
        state = pipeline.run(prompt, 0.9, fast_mode_result)
        assert state.prompt == prompt

    def test_run_uses_direct_mode(
        self,
        pipeline: FastModePipeline,
        fast_mode_result: ModeSelectionResult,
    ) -> None:
        """reasoning_mode is ReasoningMode.DIRECT."""
        state = pipeline.run("Test prompt", 0.9, fast_mode_result)
        assert state.reasoning_mode == ReasoningMode.DIRECT

    def test_run_sets_confidence_tag_high(
        self,
        pipeline: FastModePipeline,
        fast_mode_result: ModeSelectionResult,
    ) -> None:
        """When confidence > 0.85, tag is 'high'."""
        state = pipeline.run("Test prompt", 0.9, fast_mode_result)
        assert state.confidence_tag == "high"

    def test_run_sets_confidence_tag_medium(
        self,
        pipeline: FastModePipeline,
        fast_mode_result: ModeSelectionResult,
    ) -> None:
        """When 0.5 < confidence <= 0.85, tag is 'medium'."""
        medium_result = ModeSelectionResult(
            selected_mode=ReasoningModeEnum.FAST,
            confidence=0.7,
            confidence_tag="medium",
            transition_reason="confidence=0.700, tag='medium', mode=fast",
        )
        state = pipeline.run("Test prompt", 0.7, medium_result)
        assert state.confidence_tag == "medium"

    def test_run_sets_confidence_tag_low(
        self,
        pipeline: FastModePipeline,
        fast_mode_result: ModeSelectionResult,
    ) -> None:
        """When confidence <= 0.5, tag is 'low'."""
        # Note: in practice, FAST mode should not be selected for low confidence,
        # but the pipeline still handles it gracefully
        low_result = ModeSelectionResult(
            selected_mode=ReasoningModeEnum.FAST,
            confidence=0.3,
            confidence_tag="low",
            transition_reason="confidence=0.300, tag='low', mode=fast",
        )
        state = pipeline.run("Test prompt", 0.3, low_result)
        assert state.confidence_tag == "low"

    def test_run_stores_calibrated_confidence(
        self,
        pipeline: FastModePipeline,
        fast_mode_result: ModeSelectionResult,
    ) -> None:
        """state.calibrated_confidence matches input."""
        state = pipeline.run("Test prompt", 0.87, fast_mode_result)
        assert state.calibrated_confidence == 0.87

    def test_run_stores_raw_confidence(
        self,
        pipeline: FastModePipeline,
        fast_mode_result: ModeSelectionResult,
    ) -> None:
        """state.raw_confidence is set."""
        state = pipeline.run("Test prompt", 0.75, fast_mode_result)
        assert state.raw_confidence == 0.75

    def test_run_generates_response(
        self,
        pipeline: FastModePipeline,
        fast_mode_result: ModeSelectionResult,
    ) -> None:
        """Response is a non-empty string."""
        state = pipeline.run("Test prompt", 0.9, fast_mode_result)
        assert isinstance(state.reasoning_content, str)
        assert len(state.reasoning_content) > 0

    def test_run_with_high_confidence(
        self,
        pipeline: FastModePipeline,
        fast_mode_result: ModeSelectionResult,
    ) -> None:
        """Full run with high confidence (p=0.9)."""
        state = pipeline.run("Explain quantum entanglement.", 0.9, fast_mode_result)
        assert state.calibrated_confidence == 0.9
        assert state.confidence_tag == "high"
        assert state.reasoning_mode == ReasoningMode.DIRECT
        assert len(state.reasoning_content) > 0

    def test_run_with_medium_confidence(
        self,
        pipeline: FastModePipeline,
    ) -> None:
        """Full run with medium confidence (p=0.7)."""
        medium_result = ModeSelectionResult(
            selected_mode=ReasoningModeEnum.FAST,
            confidence=0.7,
            confidence_tag="medium",
            transition_reason="confidence=0.700, tag='medium', mode=fast",
        )
        state = pipeline.run("Explain gravity.", 0.7, medium_result)
        assert state.calibrated_confidence == 0.7
        assert state.confidence_tag == "medium"
        assert state.reasoning_mode == ReasoningMode.DIRECT

    def test_run_with_low_confidence(
        self,
        pipeline: FastModePipeline,
    ) -> None:
        """Full run with low confidence (p=0.3)."""
        low_result = ModeSelectionResult(
            selected_mode=ReasoningModeEnum.FAST,
            confidence=0.3,
            confidence_tag="low",
            transition_reason="confidence=0.300, tag='low', mode=fast",
        )
        state = pipeline.run("Explain consciousness.", 0.3, low_result)
        assert state.calibrated_confidence == 0.3
        assert state.confidence_tag == "low"

    def test_run_response_in_state(
        self,
        pipeline: FastModePipeline,
        fast_mode_result: ModeSelectionResult,
    ) -> None:
        """Response text appears in reasoning_content."""
        prompt = "What is 2+2?"
        state = pipeline.run(prompt, 0.9, fast_mode_result)
        assert prompt in state.reasoning_content or len(state.reasoning_content) > 0

    def test_response_includes_prompt_hint(
        self,
        pipeline: FastModePipeline,
        fast_mode_result: ModeSelectionResult,
    ) -> None:
        """The stub response contains a hint of the prompt."""
        prompt = "What is 2+2?"
        state = pipeline.run(prompt, 0.9, fast_mode_result)
        # The stub response template includes the prompt
        assert prompt in state.reasoning_content

    def test_run_no_verification_result(
        self,
        pipeline: FastModePipeline,
        fast_mode_result: ModeSelectionResult,
    ) -> None:
        """verification_result is None (no KG in fast mode)."""
        state = pipeline.run("Test prompt", 0.9, fast_mode_result)
        assert state.verification_result is None

    def test_run_empty_reasoning_steps(
        self,
        pipeline: FastModePipeline,
        fast_mode_result: ModeSelectionResult,
    ) -> None:
        """reasoning_steps is empty list (single pass)."""
        state = pipeline.run("Test prompt", 0.9, fast_mode_result)
        assert state.reasoning_steps == []

    def test_run_raises_on_wrong_mode(self, pipeline: FastModePipeline) -> None:
        """ValueError is raised if mode is not FAST."""
        wrong_mode_result = ModeSelectionResult(
            selected_mode=ReasoningModeEnum.SLOW,
            confidence=0.3,
            confidence_tag="low",
            transition_reason="confidence=0.300, tag='low', mode=slow",
        )
        with pytest.raises(ValueError, match="FastModePipeline requires FAST mode"):
            pipeline.run("Test prompt", 0.3, wrong_mode_result)

    def test_run_error_flag_on_heuristic_failure(
        self,
        pipeline: FastModePipeline,
        fast_mode_result: ModeSelectionResult,
    ) -> None:
        """Heuristic failure adds an error flag to the state."""
        # The default stub passes heuristic; override config to force failure
        bad_config = FastPipelineConfig(stub_response_template="x")  # too short
        bad_pipeline = FastModePipeline(config=bad_config)
        state = bad_pipeline.run("Test prompt", 0.9, fast_mode_result)
        assert len(state.error_flags) > 0
        assert any("heuristic_check_failed" in flag for flag in state.error_flags)


# ---------------------------------------------------------------------------
# Generate Stub Tests
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_generate_returns_string(self, pipeline: FastModePipeline) -> None:
        """_generate returns a string."""
        result = pipeline._generate("Test prompt")
        assert isinstance(result, str)

    def test_generate_uses_template(self, pipeline: FastModePipeline) -> None:
        """_generate uses the stub_response_template."""
        prompt = "Test prompt"
        result = pipeline._generate(prompt)
        assert prompt in result


# ---------------------------------------------------------------------------
# Integration / Regression Tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_pipeline_with_real_mode_selection_result(self) -> None:
        """End-to-end with a real ModeSelectionResult from controller."""
        from actr.mode_controller import ReasoningModeController

        controller = ReasoningModeController()
        mode_result = controller.select_mode(0.92)

        assert mode_result.selected_mode == ReasoningModeEnum.FAST

        pipeline = FastModePipeline()
        state = pipeline.run("What is the speed of light?", 0.92, mode_result)

        assert state.calibrated_confidence == 0.92
        assert state.confidence_tag == "high"
        assert state.reasoning_mode == ReasoningMode.DIRECT
        assert state.verification_result is None
        assert len(state.reasoning_content) > 0
        assert state.error_flags == []

    def test_state_timestamp_is_recent(self) -> None:
        """The state's timestamp is close to the current time."""
        pipeline = FastModePipeline()
        fast_result = ModeSelectionResult(
            selected_mode=ReasoningModeEnum.FAST,
            confidence=0.9,
            confidence_tag="high",
            transition_reason="test",
        )
        before = datetime.now(timezone.utc)
        state = pipeline.run("Test", 0.9, fast_result)
        after = datetime.now(timezone.utc)

        assert before <= state.timestamp <= after

    def test_custom_stub_template(self) -> None:
        """Custom stub template is used in the response."""
        custom_template = "CUSTOM_PREFIX:{prompt}:CUSTOM_SUFFIX"
        config = FastPipelineConfig(stub_response_template=custom_template)
        pipeline = FastModePipeline(config=config)
        fast_result = ModeSelectionResult(
            selected_mode=ReasoningModeEnum.FAST,
            confidence=0.9,
            confidence_tag="high",
            transition_reason="test",
        )
        state = pipeline.run("my question", 0.9, fast_result)
        assert "CUSTOM_PREFIX:" in state.reasoning_content
        assert "my question" in state.reasoning_content
        assert ":CUSTOM_SUFFIX" in state.reasoning_content
