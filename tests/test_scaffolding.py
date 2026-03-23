"""First-pass smoke tests for the ACTR scaffolding.

Tests that:
1. All public classes and functions are importable.
2. The CLI --help flag works.
3. Data structures instantiate correctly.
4. Config loading / threshold tagging works.
5. The 'reason', 'benchmark', and 'calibrate' subcommands are recognised.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure src/ is on the path for direct invocation
SRC_ROOT = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

class TestImports:
    """Verify that all public API symbols are importable."""

    def test_import_version(self):
        from actr import __version__

        assert isinstance(__version__, str)
        assert __version__

    def test_import_data_module(self):
        from actr.data import (
            CalibratedReasoningState,
            ReasoningMode,
            ConfidenceTag,
            CalibrationRecord,
            VerificationResult,
        )

        assert CalibratedReasoningState is not None
        assert ReasoningMode is not None
        assert CalibrationRecord is not None
        assert VerificationResult is not None

    def test_import_config_module(self):
        from actr.config import ACTRConfig, ConfidenceThresholds, ModelSettings, CalibrationSettings

        assert ACTRConfig is not None
        assert ConfidenceThresholds is not None
        assert ModelSettings is not None
        assert CalibrationSettings is not None

    def test_import_cli_module(self):
        from actr.cli import main as cli_main

        assert callable(cli_main)


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------

class TestReasoningMode:
    def test_reasoning_mode_values(self):
        from actr.data import ReasoningMode

        assert ReasoningMode.DIRECT.value == "direct"
        assert ReasoningMode.CHAIN_OF_THOUGHT.value == "chain_of_thought"
        assert ReasoningMode.TREE_OF_THOUGHT.value == "tree_of_thought"
        assert ReasoningMode.SELF_VERIFICATION.value == "self_verification"
        assert ReasoningMode.RECURSIVE_CALIBRATION.value == "recursive_calibration"

    def test_reasoning_mode_str(self):
        from actr.data import ReasoningMode

        assert str(ReasoningMode.DIRECT) == "direct"

    def test_reasoning_mode_supports_calibration(self):
        from actr.data import ReasoningMode

        assert ReasoningMode.DIRECT.supports_calibration is False
        assert ReasoningMode.CHAIN_OF_THOUGHT.supports_calibration is True
        assert ReasoningMode.SELF_VERIFICATION.supports_calibration is True

    def test_reasoning_mode_description(self):
        from actr.data import ReasoningMode

        desc = ReasoningMode.SELF_VERIFICATION.description
        assert isinstance(desc, str)
        assert len(desc) > 10


class TestCalibrationRecord:
    def test_calibration_record_delta(self):
        from actr.data import CalibrationRecord, ReasoningMode

        rec = CalibrationRecord(
            step=1,
            input_confidence=0.6,
            output_confidence=0.75,
            adjustment_reason="verification passed",
            reasoning_mode=ReasoningMode.SELF_VERIFICATION,
        )
        assert rec.delta == pytest.approx(0.15)

    def test_calibration_record_to_dict(self):
        from actr.data import CalibrationRecord

        rec = CalibrationRecord(
            step=2,
            input_confidence=0.5,
            output_confidence=0.65,
            adjustment_reason="threshold adjustment",
        )
        d = rec.to_dict()
        assert d["step"] == 2
        assert d["input_confidence"] == 0.5
        assert d["output_confidence"] == 0.65
        assert d["delta"] == pytest.approx(0.15)
        assert "timestamp" in d


class TestVerificationResult:
    def test_verification_result_passed_alias(self):
        from actr.data import VerificationResult

        vr = VerificationResult(
            is_verified=True,
            verification_method="consistency_check",
            consistency_score=0.92,
        )
        assert vr.passed is True

    def test_verification_result_to_dict(self):
        from actr.data import VerificationResult

        vr = VerificationResult(
            is_verified=False,
            verification_method="regression_test",
            consistency_score=0.55,
            failed_checks=["math_step_3", "unit_conversion"],
            passed_checks=["syntax"],
        )
        d = vr.to_dict()
        assert d["is_verified"] is False
        assert d["consistency_score"] == 0.55
        assert "math_step_3" in d["failed_checks"]


class TestCalibratedReasoningState:
    def test_state_defaults(self):
        from actr.data import CalibratedReasoningState

        state = CalibratedReasoningState(prompt="What is 2+2?")
        assert state.prompt == "What is 2+2?"
        assert state.reasoning_content == ""
        assert state.raw_confidence == 0.0
        assert state.calibration_history == []
        assert state.error_flags == []

    def test_state_add_calibration_record(self):
        from actr.data import CalibratedReasoningState

        state = CalibratedReasoningState(prompt="Test")
        state.add_calibration_record(
            step=1,
            input_confidence=0.5,
            output_confidence=0.7,
            adjustment_reason="verified",
        )
        assert len(state.calibration_history) == 1
        assert state.calibrated_confidence == 0.7

    def test_state_add_error(self):
        from actr.data import CalibratedReasoningState

        state = CalibratedReasoningState(prompt="Test")
        state.add_error("timeout")
        assert "timeout" in state.error_flags
        # Adding the same error twice should not duplicate
        state.add_error("timeout")
        assert len(state.error_flags) == 1

    def test_state_total_calibration_steps(self):
        from actr.data import CalibratedReasoningState

        state = CalibratedReasoningState(prompt="Test")
        for i in range(3):
            state.add_calibration_record(
                step=i + 1,
                input_confidence=0.5,
                output_confidence=0.6 + i * 0.1,
                adjustment_reason=f"step {i + 1}",
            )
        assert state.total_calibration_steps == 3

    def test_state_is_verified(self):
        from actr.data import CalibratedReasoningState, VerificationResult

        state = CalibratedReasoningState(prompt="Test")
        assert state.is_verified is False
        state.verification_result = VerificationResult(
            is_verified=True,
            verification_method="test",
            consistency_score=0.9,
        )
        assert state.is_verified is True

    def test_state_to_dict(self):
        from actr.data import CalibratedReasoningState

        state = CalibratedReasoningState(
            prompt="Test prompt",
            reasoning_content="Test answer",
            raw_confidence=0.5,
            calibrated_confidence=0.7,
            confidence_tag="medium",
        )
        d = state.to_dict()
        assert d["prompt"] == "Test prompt"
        assert d["raw_confidence"] == 0.5
        assert d["calibrated_confidence"] == 0.7
        assert d["confidence_tag"] == "medium"
        assert "calibration_history" in d


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfidenceThresholds:
    def test_tag_for_confidence_high(self):
        from actr.config import ConfidenceThresholds

        t = ConfidenceThresholds(high=0.8, medium=0.65, low=0.4)
        assert t.tag_for_confidence(0.9) == "high"
        assert t.tag_for_confidence(0.85) == "high"

    def test_tag_for_confidence_medium(self):
        from actr.config import ConfidenceThresholds

        t = ConfidenceThresholds(high=0.8, medium=0.65, low=0.4)
        assert t.tag_for_confidence(0.7) == "medium"
        assert t.tag_for_confidence(0.65) == "medium"

    def test_tag_for_confidence_low(self):
        from actr.config import ConfidenceThresholds

        t = ConfidenceThresholds(high=0.8, medium=0.65, low=0.4)
        assert t.tag_for_confidence(0.3) == "low"
        assert t.tag_for_confidence(0.39) == "low"

    def test_tag_for_confidence_out_of_range(self):
        from actr.config import ConfidenceThresholds

        t = ConfidenceThresholds()
        assert t.tag_for_confidence(-0.1) == "unknown"
        assert t.tag_for_confidence(1.5) == "unknown"

    def test_thresholds_to_dict_from_dict(self):
        from actr.config import ConfidenceThresholds

        t = ConfidenceThresholds(low=0.3, medium=0.6, high=0.85, unknown_epsilon=0.03)
        d = t.to_dict()
        assert d["low"] == 0.3
        assert d["high"] == 0.85
        t2 = ConfidenceThresholds.from_dict(d)
        assert t2.low == 0.3
        assert t2.high == 0.85


class TestModelSettings:
    def test_resolved_api_key_from_env(self, monkeypatch):
        import os

        monkeypatch.setenv("OPENAI_API_KEY", "env-api-key-123")
        from actr.config import ModelSettings

        ms = ModelSettings()
        assert ms.resolved_api_key() == "env-api-key-123"

    def test_resolved_api_key_explicit(self):
        from actr.config import ModelSettings

        ms = ModelSettings(api_key="explicit-key-456")
        assert ms.resolved_api_key() == "explicit-key-456"

    def test_model_settings_to_dict_redacts_key(self):
        from actr.config import ModelSettings

        ms = ModelSettings(api_key="secret")
        d = ms.to_dict()
        assert d["api_key"] == "***"


class TestCalibrationSettings:
    def test_calibration_settings_defaults(self):
        from actr.config import CalibrationSettings

        cs = CalibrationSettings()
        assert cs.learning_rate == 0.3
        assert cs.window_size == 100
        assert cs.max_recursion_depth == 3
        assert cs.calibration_model_type == "linear"


class TestACTRConfig:
    def test_config_tag_for_confidence(self):
        from actr.config import ACTRConfig

        cfg = ACTRConfig()
        assert cfg.tag_for_confidence(0.9) == "high"
        assert cfg.tag_for_confidence(0.5) == "medium"
        assert cfg.tag_for_confidence(0.2) == "low"

    def test_config_to_dict(self):
        from actr.config import ACTRConfig

        cfg = ACTRConfig()
        d = cfg.to_dict()
        assert "thresholds" in d
        assert "model" in d
        assert "calibration" in d
        assert d["default_reasoning_mode"] == "chain_of_thought"

    def test_config_from_dict(self):
        from actr.config import ACTRConfig

        data = {
            "thresholds": {"low": 0.3, "medium": 0.6, "high": 0.85},
            "model": {"model_name": "claude-3-opus", "temperature": 0.5},
            "calibration": {"learning_rate": 0.2},
            "default_reasoning_mode": "self_verification",
        }
        cfg = ACTRConfig.from_dict(data)
        assert cfg.thresholds.high == 0.85
        assert cfg.model.model_name == "claude-3-opus"
        assert cfg.calibration.learning_rate == 0.2
        assert cfg.default_reasoning_mode == "self_verification"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestCLIHelp:
    """Smoke tests for CLI argument parsing via subprocess."""

    def _run_actr(self, args: list[str]) -> subprocess.CompletedProcess:
        src_cli = Path(__file__).parent.parent / "src" / "actr" / "cli.py"
        return subprocess.run(
            [sys.executable, str(src_cli)] + args,
            capture_output=True,
            text=True,
        )

    def test_cli_help_flag(self):
        result = self._run_actr(["--help"])
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "ACTR" in result.stdout
        assert "reason" in result.stdout
        assert "benchmark" in result.stdout
        assert "calibrate" in result.stdout

    def test_cli_reason_help(self):
        result = self._run_actr(["reason", "--help"])
        assert result.returncode == 0
        assert "prompt" in result.stdout

    def test_cli_benchmark_help(self):
        result = self._run_actr(["benchmark", "--help"])
        assert result.returncode == 0
        assert "suite" in result.stdout

    def test_cli_calibrate_help(self):
        result = self._run_actr(["calibrate", "--help"])
        assert result.returncode == 0
        assert "report" in result.stdout

    def test_cli_reason_command_accepts_prompt(self):
        src_cli = Path(__file__).parent.parent / "src" / "actr" / "cli.py"
        result = subprocess.run(
            [sys.executable, str(src_cli), "reason", "What is 2+2?"],
            capture_output=True,
            text=True,
        )
        # With no real API key / model, we accept any exit as valid scaffolding;
        # the point is argument parsing does not reject the prompt.
        # We also check that the JSON output contains expected keys.
        output = json.loads(result.stdout)
        assert "prompt" in output
        assert output["prompt"] == "What is 2+2?"

    def test_cli_reason_json_format(self):
        src_cli = Path(__file__).parent.parent / "src" / "actr" / "cli.py"
        result = subprocess.run(
            [sys.executable, str(src_cli), "reason", "--format", "json", "Test?"],
            capture_output=True,
            text=True,
        )
        data = json.loads(result.stdout)
        assert "raw_confidence" in data
        assert "calibrated_confidence" in data
        assert "confidence_tag" in data

    def test_cli_reason_invalid_mode(self):
        src_cli = Path(__file__).parent.parent / "src" / "actr" / "cli.py"
        result = subprocess.run(
            [sys.executable, str(src_cli), "reason", "--mode", "not_a_mode", "Test?"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_cli_missing_subcommand(self):
        src_cli = Path(__file__).parent.parent / "src" / "actr" / "cli.py"
        result = subprocess.run(
            [sys.executable, str(src_cli)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
