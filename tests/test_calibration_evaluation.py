"""Tests for the calibration evaluation module (src/actr/evaluation.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is on the path
SRC_ROOT = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from actr import (
    CalibrationEvaluator,
    CalibrationEvaluationMetrics,
    EVALUATION_BENCHMARK_QUERIES,
    run_calibration_evaluation,
)
from actr.benchmark import BenchmarkQuery, BenchmarkResult
from actr.calibration import CalibrationDataset, CalibrationEngine, PlattCalibrator
from actr.config import ACTRConfig
from actr.data import CalibratedReasoningState, ReasoningMode
from actr.mode_controller import ModeSelectionResult, ReasoningModeController, ReasoningModeEnum
from actr.ssu import SSUConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_result(
    difficulty: str,
    selected_mode: ReasoningModeEnum,
    is_correct: bool,
    calibrated_confidence: float = 0.5,
    boundary_violations: list[str] | None = None,
    reasoning_steps: list | None = None,
) -> BenchmarkResult:
    """Factory to build a mock BenchmarkResult for testing."""
    query = BenchmarkQuery(
        prompt=f"Test prompt ({difficulty})",
        expected_answer="test answer",
        difficulty=difficulty,
    )
    # Build a minimal CalibratedReasoningState just to satisfy BenchmarkResult fields
    state = CalibratedReasoningState(
        prompt=query.prompt,
        reasoning_content="mock response",
        calibrated_confidence=calibrated_confidence,
        reasoning_mode=ReasoningMode.CHAIN_OF_THOUGHT,
        confidence_tag="medium",
        reasoning_steps=reasoning_steps or [],
    )
    mode_result = ModeSelectionResult(
        selected_mode=selected_mode,
        confidence=calibrated_confidence,
        confidence_tag="medium",
        transition_reason="test",
    )
    return BenchmarkResult(
        query=query,
        predicted_response="mock response",
        calibrated_confidence=calibrated_confidence,
        confidence_tag="medium",
        selected_mode=selected_mode,
        is_correct=is_correct,
        mode_selection_result=mode_result.selected_mode,
        boundary_violations=boundary_violations or [],
        reasoning_steps=reasoning_steps or [],
    )


# ---------------------------------------------------------------------------
# CalibrationEvaluationMetrics tests
# ---------------------------------------------------------------------------

class TestCalibrationEvaluationMetrics:
    def test_metrics_can_be_instantiated(self):
        m = CalibrationEvaluationMetrics(
            ece=0.03,
            mode_switch_accuracy=0.95,
            accuracy_at_fast=0.92,
            accuracy_at_moderate=0.80,
            accuracy_at_slow=0.88,
            compute_speedup_vs_slow=2.1,
            boundary_violation_rate=0.0,
            kg_precision=0.85,
            auroc=0.91,
            n_queries=20,
            timestamp="2026-03-26T00:00:00Z",
        )
        assert m.ece == 0.03
        assert m.mode_switch_accuracy == 0.95
        assert m.accuracy_at_fast == 0.92
        assert m.accuracy_at_slow == 0.88
        assert m.compute_speedup_vs_slow == 2.1
        assert m.boundary_violation_rate == 0.0
        assert m.kg_precision == 0.85
        assert m.auroc == 0.91
        assert m.n_queries == 20

    def test_to_dict_rounds_floats(self):
        m = CalibrationEvaluationMetrics(
            ece=0.033333,
            mode_switch_accuracy=0.955555,
            accuracy_at_fast=0.922222,
            accuracy_at_moderate=0.800000,
            accuracy_at_slow=0.888888,
            compute_speedup_vs_slow=2.144444,
            boundary_violation_rate=0.005555,
            kg_precision=0.855555,
            auroc=0.911111,
            n_queries=20,
            timestamp="2026-03-26T00:00:00Z",
        )
        d = m.to_dict()
        assert d["ece"] == 0.0333
        assert d["mode_switch_accuracy"] == 0.9556
        assert d["accuracy_at_fast"] == 0.9222
        assert d["compute_speedup_vs_slow"] == 2.1444
        assert d["boundary_violation_rate"] == 0.0056
        assert d["kg_precision"] == 0.8556
        assert d["auroc"] == 0.9111
        assert d["n_queries"] == 20


# ---------------------------------------------------------------------------
# Evaluation benchmark queries
# ---------------------------------------------------------------------------

class TestEvaluationBenchmarkQueries:
    def test_evaluation_queries_exist(self):
        assert len(EVALUATION_BENCHMARK_QUERIES) == 20

    def test_evaluation_queries_difficulty_distribution(self):
        difficulties = [q.difficulty for q in EVALUATION_BENCHMARK_QUERIES]
        assert difficulties.count("factual") == 7
        assert difficulties.count("mathematical") == 7
        assert difficulties.count("adversarial") == 6

    def test_evaluation_queries_have_prompts_and_answers(self):
        for q in EVALUATION_BENCHMARK_QUERIES:
            assert q.prompt, "Query must have a non-empty prompt"
            assert q.expected_answer, "Query must have a non-empty expected_answer"
            assert q.difficulty in ("factual", "mathematical", "adversarial")


# ---------------------------------------------------------------------------
# CalibrationEvaluator initialization
# ---------------------------------------------------------------------------

class TestCalibrationEvaluatorInit:
    def test_init_with_engine(self):
        ssu_cfg = SSUConfig()
        cfg = ACTRConfig()
        engine = CalibrationEngine(ssu_config=ssu_cfg)
        evaluator = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=engine,
        )
        assert evaluator.benchmark_queries == EVALUATION_BENCHMARK_QUERIES
        assert evaluator.calibration_engine is engine

    def test_init_with_custom_configs(self):
        ssu_cfg = SSUConfig()
        cfg = ACTRConfig()
        engine = CalibrationEngine(ssu_config=ssu_cfg)
        evaluator = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=engine,
            config=cfg,
            ssu_config=ssu_cfg,
        )
        assert evaluator.config is cfg
        assert evaluator.ssu_config is ssu_cfg


# ---------------------------------------------------------------------------
# _difficulty_to_expected_mode
# ---------------------------------------------------------------------------

class TestDifficultyToExpectedMode:
    def _make_evaluator(self) -> CalibrationEvaluator:
        engine = CalibrationEngine(ssu_config=SSUConfig())
        return CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=engine,
        )

    def test_factual_maps_to_fast(self):
        ev = self._make_evaluator()
        assert ev._difficulty_to_expected_mode("factual") == ReasoningModeEnum.FAST

    def test_mathematical_maps_to_moderate(self):
        ev = self._make_evaluator()
        assert ev._difficulty_to_expected_mode("mathematical") == ReasoningModeEnum.MODERATE

    def test_adversarial_maps_to_slow(self):
        ev = self._make_evaluator()
        assert ev._difficulty_to_expected_mode("adversarial") == ReasoningModeEnum.SLOW

    def test_unknown_difficulty_raises(self):
        ev = self._make_evaluator()
        with pytest.raises(ValueError, match="Unknown difficulty"):
            ev._difficulty_to_expected_mode("unknown")


# ---------------------------------------------------------------------------
# compute_mode_switch_accuracy
# ---------------------------------------------------------------------------

class TestComputeModeSwitchAccuracy:
    def test_perfect_accuracy(self):
        engine = CalibrationEngine(ssu_config=SSUConfig())
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=engine,
        )
        # factual → FAST, mathematical → MODERATE, adversarial → SLOW
        results = [
            _make_mock_result("factual", ReasoningModeEnum.FAST, True),
            _make_mock_result("factual", ReasoningModeEnum.FAST, True),
            _make_mock_result("mathematical", ReasoningModeEnum.MODERATE, True),
            _make_mock_result("adversarial", ReasoningModeEnum.SLOW, True),
        ]
        acc = ev.compute_mode_switch_accuracy(results)
        assert acc == 1.0

    def test_zero_accuracy(self):
        engine = CalibrationEngine(ssu_config=SSUConfig())
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=engine,
        )
        # All wrong mode selections
        results = [
            _make_mock_result("factual", ReasoningModeEnum.SLOW, False),
            _make_mock_result("mathematical", ReasoningModeEnum.FAST, False),
            _make_mock_result("adversarial", ReasoningModeEnum.MODERATE, False),
        ]
        acc = ev.compute_mode_switch_accuracy(results)
        assert acc == 0.0

    def test_partial_accuracy(self):
        engine = CalibrationEngine(ssu_config=SSUConfig())
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=engine,
        )
        results = [
            _make_mock_result("factual", ReasoningModeEnum.FAST, True),   # correct
            _make_mock_result("factual", ReasoningModeEnum.SLOW, True),   # wrong
            _make_mock_result("mathematical", ReasoningModeEnum.MODERATE, True),  # correct
            _make_mock_result("mathematical", ReasoningModeEnum.FAST, True),  # wrong
        ]
        acc = ev.compute_mode_switch_accuracy(results)
        assert acc == 0.5

    def test_empty_results_returns_zero(self):
        engine = CalibrationEngine(ssu_config=SSUConfig())
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=engine,
        )
        assert ev.compute_mode_switch_accuracy([]) == 0.0


# ---------------------------------------------------------------------------
# compute_boundary_violation_rate
# ---------------------------------------------------------------------------

class TestComputeBoundaryViolationRate:
    def _engine(self):
        return CalibrationEngine(ssu_config=SSUConfig())

    def test_no_violations(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        results = [
            _make_mock_result("factual", ReasoningModeEnum.FAST, True, boundary_violations=[]),
            _make_mock_result("adversarial", ReasoningModeEnum.SLOW, True, boundary_violations=[]),
        ]
        assert ev.compute_boundary_violation_rate(results) == 0.0

    def test_all_have_violations(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        results = [
            _make_mock_result("factual", ReasoningModeEnum.FAST, True, boundary_violations=["length_exceeded"]),
            _make_mock_result("adversarial", ReasoningModeEnum.SLOW, True, boundary_violations=["content_policy"]),
        ]
        assert ev.compute_boundary_violation_rate(results) == 1.0

    def test_partial_violations(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        results = [
            _make_mock_result("factual", ReasoningModeEnum.FAST, True, boundary_violations=["length_exceeded"]),
            _make_mock_result("factual", ReasoningModeEnum.FAST, True, boundary_violations=[]),
            _make_mock_result("adversarial", ReasoningModeEnum.SLOW, True, boundary_violations=["content_policy"]),
            _make_mock_result("adversarial", ReasoningModeEnum.SLOW, True, boundary_violations=[]),
        ]
        # 2 out of 4 have violations
        assert ev.compute_boundary_violation_rate(results) == 0.5

    def test_empty_results_returns_zero(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        assert ev.compute_boundary_violation_rate([]) == 0.0


# ---------------------------------------------------------------------------
# compute_kg_precision
# ---------------------------------------------------------------------------

class TestComputeKgPrecision:
    def _engine(self):
        return CalibrationEngine(ssu_config=SSUConfig())

    def test_no_kg_activations_returns_zero(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        results = [
            _make_mock_result("adversarial", ReasoningModeEnum.SLOW, True, reasoning_steps=["[Pass 1] answer"]),
        ]
        assert ev.compute_kg_precision(results) == 0.0

    def test_kg_triggered_correct_answer(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        results = [
            _make_mock_result(
                "adversarial",
                ReasoningModeEnum.SLOW,
                True,
                reasoning_steps=["[KG triggered: both passes failed]"],
            ),
        ]
        assert ev.compute_kg_precision(results) == 1.0

    def test_kg_triggered_incorrect_answer(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        results = [
            _make_mock_result(
                "adversarial",
                ReasoningModeEnum.SLOW,
                False,
                reasoning_steps=["[KG triggered: both passes failed]"],
            ),
        ]
        assert ev.compute_kg_precision(results) == 0.0

    def test_kg_precision_multiple_queries(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        results = [
            _make_mock_result(
                "adversarial", ReasoningModeEnum.SLOW, True,
                reasoning_steps=["[KG triggered: both passes failed]"],
            ),
            _make_mock_result(
                "adversarial", ReasoningModeEnum.SLOW, False,
                reasoning_steps=["[KG triggered: both passes failed]"],
            ),
            _make_mock_result(
                "adversarial", ReasoningModeEnum.SLOW, True,
                reasoning_steps=["[KG triggered: both passes failed]"],
            ),
            _make_mock_result(
                "adversarial", ReasoningModeEnum.SLOW, True,
                reasoning_steps=["[KG not triggered]"],
            ),
        ]
        # 3 KG activations, 2 useful (correct)
        assert ev.compute_kg_precision(results) == 2.0 / 3.0

    def test_kg_detected_from_structured_dict_step(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        results = [
            _make_mock_result(
                "adversarial",
                ReasoningModeEnum.SLOW,
                True,
                reasoning_steps=[{"type": "knowledge_grounding", "content": "grounded"}],
            ),
        ]
        assert ev.compute_kg_precision(results) == 1.0

    def test_kg_detected_from_kg_grounded_marker(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        results = [
            _make_mock_result(
                "adversarial",
                ReasoningModeEnum.SLOW,
                True,
                reasoning_steps=["[KG grounded response] improved answer"],
            ),
        ]
        assert ev.compute_kg_precision(results) == 1.0


# ---------------------------------------------------------------------------
# compute_accuracy_per_mode
# ---------------------------------------------------------------------------

class TestComputeAccuracyPerMode:
    def _engine(self):
        return CalibrationEngine(ssu_config=SSUConfig())

    def test_all_fast_correct(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        results = [
            _make_mock_result("factual", ReasoningModeEnum.FAST, True),
            _make_mock_result("factual", ReasoningModeEnum.FAST, True),
        ]
        acc = ev.compute_accuracy_per_mode(results)
        assert acc["fast"] == 1.0
        assert acc["moderate"] == 0.0
        assert acc["slow"] == 0.0

    def test_mixed_accuracy(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        results = [
            _make_mock_result("factual", ReasoningModeEnum.FAST, True),
            _make_mock_result("factual", ReasoningModeEnum.FAST, False),
            _make_mock_result("mathematical", ReasoningModeEnum.MODERATE, True),
            _make_mock_result("mathematical", ReasoningModeEnum.MODERATE, False),
            _make_mock_result("mathematical", ReasoningModeEnum.MODERATE, True),
            _make_mock_result("adversarial", ReasoningModeEnum.SLOW, False),
            _make_mock_result("adversarial", ReasoningModeEnum.SLOW, True),
        ]
        acc = ev.compute_accuracy_per_mode(results)
        assert acc["fast"] == 0.5
        assert acc["moderate"] == 2.0 / 3.0
        assert acc["slow"] == 0.5

    def test_empty_results_returns_zeros(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        acc = ev.compute_accuracy_per_mode([])
        assert acc["fast"] == 0.0
        assert acc["moderate"] == 0.0
        assert acc["slow"] == 0.0


# ---------------------------------------------------------------------------
# compute_speedup_vs_slow
# ---------------------------------------------------------------------------

class TestComputeSpeedupVsSlow:
    def _engine(self):
        return CalibrationEngine(ssu_config=SSUConfig())

    def test_all_fast_max_speedup(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        results = [
            _make_mock_result("factual", ReasoningModeEnum.FAST, True),
            _make_mock_result("factual", ReasoningModeEnum.FAST, True),
            _make_mock_result("factual", ReasoningModeEnum.FAST, True),
        ]
        # 3 slow passes / (1+1+1) actual passes = 3.0
        assert ev.compute_speedup_vs_slow(results) == 3.0

    def test_all_slow_no_speedup(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        results = [
            _make_mock_result("adversarial", ReasoningModeEnum.SLOW, True),
            _make_mock_result("adversarial", ReasoningModeEnum.SLOW, True),
        ]
        # 2 slow passes / (3+3) actual = 6/6 = 1.0
        assert ev.compute_speedup_vs_slow(results) == 1.0

    def test_mixed_modes(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        # 4 queries: FAST(1) + MODERATE(2) + SLOW(3) + SLOW(3) = 9 actual passes
        # 4 * 3 = 12 slow passes → 12/9 = 1.333...
        results = [
            _make_mock_result("factual", ReasoningModeEnum.FAST, True),
            _make_mock_result("mathematical", ReasoningModeEnum.MODERATE, True),
            _make_mock_result("adversarial", ReasoningModeEnum.SLOW, True),
            _make_mock_result("adversarial", ReasoningModeEnum.SLOW, True),
        ]
        speedup = ev.compute_speedup_vs_slow(results)
        assert speedup == 12.0 / 9.0


# ---------------------------------------------------------------------------
# compute_ece (from dataset)
# ---------------------------------------------------------------------------

class TestComputeECE:
    def _engine(self):
        return CalibrationEngine(ssu_config=SSUConfig())

    def test_ece_computed_from_calibration_dataset(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        dataset = CalibrationDataset()
        # Add varied calibration pairs
        for i in range(100):
            dataset.add(raw_score=0.9, is_correct=True)
            dataset.add(raw_score=0.2, is_correct=False)
        # Calibrate
        ev.calibration_engine.calibrate_full(dataset)
        ece = ev.compute_ece(dataset)
        # ECE should be in [0, 1]
        assert 0.0 <= ece <= 1.0

    def test_ece_computed_from_results(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        results = [
            _make_mock_result("factual", ReasoningModeEnum.FAST, True, calibrated_confidence=0.9),
            _make_mock_result("factual", ReasoningModeEnum.FAST, True, calibrated_confidence=0.9),
            _make_mock_result("factual", ReasoningModeEnum.FAST, False, calibrated_confidence=0.3),
            _make_mock_result("factual", ReasoningModeEnum.FAST, True, calibrated_confidence=0.85),
        ]
        ece = ev.compute_ece_from_results(results)
        assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------

class TestAUROC:
    def _engine(self):
        return CalibrationEngine(ssu_config=SSUConfig())

    def test_auroc_perfect(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        y_true = np.array([1, 1, 0, 0])
        y_cal = np.array([0.9, 0.85, 0.15, 0.1])
        auroc = CalibrationEvaluator._compute_auroc(y_true, y_cal)
        assert auroc == pytest.approx(1.0)

    def test_auroc_random(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        y_true = np.array([1, 0, 1, 0])
        y_cal = np.array([0.5, 0.5, 0.5, 0.5])
        auroc = CalibrationEvaluator._compute_auroc(y_true, y_cal)
        assert auroc == pytest.approx(0.5)

    def test_auroc_single_class(self):
        ev = CalibrationEvaluator(
            benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
            calibration_engine=self._engine(),
        )
        y_true = np.array([1, 1, 1])
        y_cal = np.array([0.9, 0.7, 0.5])
        auroc = CalibrationEvaluator._compute_auroc(y_true, y_cal)
        assert auroc == 0.5


# ---------------------------------------------------------------------------
# run_calibration_evaluation
# ---------------------------------------------------------------------------

class TestRunCalibrationEvaluation:
    def test_returns_valid_metrics_object(self):
        metrics = run_calibration_evaluation(n_calibration_samples=50)
        assert isinstance(metrics, CalibrationEvaluationMetrics)
        assert metrics.n_queries == 20
        assert metrics.timestamp != ""

    def test_all_metrics_are_valid_floats(self):
        metrics = run_calibration_evaluation(n_calibration_samples=50)
        for field_name in [
            "ece", "mode_switch_accuracy", "accuracy_at_fast",
            "accuracy_at_moderate", "accuracy_at_slow",
            "compute_speedup_vs_slow", "boundary_violation_rate",
            "kg_precision", "auroc",
        ]:
            val = getattr(metrics, field_name)
            assert isinstance(val, float), f"{field_name} should be float, got {type(val)}"
            assert 0.0 <= val <= 5.0, f"{field_name}={val} is out of reasonable range"

    def test_metrics_in_expected_ranges(self):
        metrics = run_calibration_evaluation(n_calibration_samples=50)
        # All should be in [0, 1] except speedup which can be > 1
        assert 0.0 <= metrics.ece <= 1.0
        assert 0.0 <= metrics.mode_switch_accuracy <= 1.0
        assert 0.0 <= metrics.accuracy_at_fast <= 1.0
        assert 0.0 <= metrics.accuracy_at_moderate <= 1.0
        assert 0.0 <= metrics.accuracy_at_slow <= 1.0
        assert 0.0 <= metrics.boundary_violation_rate <= 1.0
        assert 0.0 <= metrics.kg_precision <= 1.0
        assert 0.0 <= metrics.auroc <= 1.0
        assert 1.0 <= metrics.compute_speedup_vs_slow <= 3.0

    def test_to_dict_contains_all_fields(self):
        metrics = run_calibration_evaluation(n_calibration_samples=50)
        d = metrics.to_dict()
        expected = {
            "ece", "mode_switch_accuracy", "accuracy_at_fast",
            "accuracy_at_moderate", "accuracy_at_slow",
            "compute_speedup_vs_slow", "boundary_violation_rate",
            "kg_precision", "auroc", "n_queries", "timestamp",
        }
        assert expected == set(d.keys())
