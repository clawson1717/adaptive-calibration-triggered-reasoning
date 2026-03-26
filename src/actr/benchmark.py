"""Benchmark runner for ACTR — evaluates the full pipeline on test queries."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from actr.config import ACTRConfig
from actr.ssu import SSUConfig, ThreeSampleSSU
from actr.calibration import CalibrationEngine, CalibrationDataset
from actr.mode_controller import ReasoningModeController, ReasoningModeEnum
from actr.pipelines import (
    FastModePipeline,
    ModerateModePipeline,
    SlowModePipeline,
    BoundaryEnforcementLayer,
)

__all__ = ["BenchmarkRunner", "BenchmarkQuery", "BenchmarkResult"]


# ---------------------------------------------------------------------------
# Benchmark query / result types
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkQuery:
    """A single benchmark query with ground-truth answer.

    Attributes
    ----------
    prompt : str
        The question or problem to reason about.
    expected_answer : str
        The correct answer (used for accuracy scoring).
    difficulty : str
        One of ``"factual"`` (low uncertainty), ``"mathematical"``
        (medium uncertainty), or ``"adversarial"`` (high uncertainty).
    category : str
        Optional sub-category label.
    """

    prompt: str
    expected_answer: str
    difficulty: str = "factual"
    category: str = "general"


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark query through the ACTR pipeline.

    Attributes
    ----------
    query : BenchmarkQuery
        The original benchmark query.
    predicted_response : str
        The generated response from the ACTR pipeline.
    calibrated_confidence : float
        Calibrated confidence score from the SSU engine.
    confidence_tag : str
        Discrete confidence tag (``"high"``, ``"medium"``, ``"low"``, ``"unknown"``).
    selected_mode : ReasoningModeEnum
        The reasoning mode that was selected.
    is_correct : bool
        Whether the predicted response matches the expected answer
        (simple substring check).
    mode_selection_result : ReasoningModeEnum
        The mode that was selected by the controller.
    boundary_violations : list[str]
        Any boundary violations detected by the enforcement layer.
    reasoning_steps : list[str]
        Intermediate reasoning steps recorded by the pipeline.
    """

    query: BenchmarkQuery
    predicted_response: str
    calibrated_confidence: float
    confidence_tag: str
    selected_mode: ReasoningModeEnum
    is_correct: bool
    mode_selection_result: ReasoningModeEnum
    boundary_violations: list[str] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)


@dataclass
class BenchmarkSummary:
    """Aggregate metrics across all benchmark queries.

    Attributes
    ----------
    total_queries : int
        Total number of queries run.
    accuracy_per_mode : dict[str, float]
        Accuracy (fraction correct) for each reasoning mode.
    mode_distribution : dict[str, int]
        Count of queries routed to each reasoning mode.
    average_calibrated_confidence : float
        Mean calibrated confidence across all queries.
    boundary_violation_rate : float
        Fraction of queries with at least one boundary violation.
    ece : float
        Expected Calibration Error across all queries (requires correctness labels).
    timestamp : datetime
        When the benchmark was run.
    """

    total_queries: int
    accuracy_per_mode: dict[str, float]
    mode_distribution: dict[str, int]
    average_calibrated_confidence: float
    boundary_violation_rate: float
    ece: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "accuracy_per_mode": self.accuracy_per_mode,
            "mode_distribution": self.mode_distribution,
            "average_calibrated_confidence": self.average_calibrated_confidence,
            "boundary_violation_rate": self.boundary_violation_rate,
            "ece": self.ece,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Built-in benchmark queries
# ---------------------------------------------------------------------------

BUILTIN_BENCHMARK_QUERIES: list[BenchmarkQuery] = [
    # ---- Factual (low uncertainty) ----
    BenchmarkQuery(
        prompt="What is the capital of France?",
        expected_answer="Paris",
        difficulty="factual",
        category="geography",
    ),
    BenchmarkQuery(
        prompt="What is the chemical symbol for gold?",
        expected_answer="Au",
        difficulty="factual",
        category="chemistry",
    ),
    BenchmarkQuery(
        prompt="Who wrote 'Romeo and Juliet'?",
        expected_answer="Shakespeare",
        difficulty="factual",
        category="literature",
    ),
    BenchmarkQuery(
        prompt="What year did World War II end?",
        expected_answer="1945",
        difficulty="factual",
        category="history",
    ),
    # ---- Mathematical (medium uncertainty) ----
    BenchmarkQuery(
        prompt="What is 17 times 23?",
        expected_answer="391",
        difficulty="mathematical",
        category="arithmetic",
    ),
    BenchmarkQuery(
        prompt="Solve for x: 3x + 7 = 22",
        expected_answer="5",
        difficulty="mathematical",
        category="algebra",
    ),
    BenchmarkQuery(
        prompt="What is the square root of 144?",
        expected_answer="12",
        difficulty="mathematical",
        category="arithmetic",
    ),
    # ---- Adversarial / reasoning (high uncertainty) ----
    BenchmarkQuery(
        prompt="If all Zorks are Morks, and some Morks are Borks, can we conclude that some Zorks are Borks?",
        expected_answer="no",
        difficulty="adversarial",
        category="logic",
    ),
    BenchmarkQuery(
        prompt="A bat and ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
        expected_answer="5 cents",
        difficulty="adversarial",
        category="reasoning",
    ),
    BenchmarkQuery(
        prompt="What is the next number in the sequence: 2, 6, 12, 20, 30, ?",
        expected_answer="42",
        difficulty="adversarial",
        category="pattern",
    ),
    BenchmarkQuery(
        prompt="If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        expected_answer="5 minutes",
        difficulty="adversarial",
        category="reasoning",
    ),
    BenchmarkQuery(
        prompt="In a race, you pass the person in second place. What place are you now in?",
        expected_answer="second",
        difficulty="adversarial",
        category="reasoning",
    ),
]


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Runs the full ACTR pipeline on a suite of benchmark queries.

    The runner:
    1. Optionally calibrates the SSU engine using synthetic data
    2. Runs each query through the full pipeline
       (SSU → mode selection → mode pipeline → boundary enforcement)
    3. Computes aggregate metrics

    Parameters
    ----------
    config : ACTRConfig | None
        Global ACTR configuration. Uses defaults if None.
    ssu_config : SSUConfig | None
        SSU engine configuration. Uses defaults if None.
    calibrate : bool
        Whether to run calibration before benchmarking. Default True.
    seed : int
        Random seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        config: Optional[ACTRConfig] = None,
        ssu_config: Optional[SSUConfig] = None,
        calibrate: bool = True,
        seed: int = 42,
    ) -> None:
        self.config = config if config is not None else ACTRConfig()
        self.ssu_config = ssu_config if ssu_config is not None else SSUConfig()
        self.calibrate = calibrate
        self.seed = seed

        # Components
        self._ssu = ThreeSampleSSU(config=self.ssu_config)
        self._calibration_engine = CalibrationEngine(ssu_config=self.ssu_config)
        self._platt = None
        self._mode_controller = ReasoningModeController(config=self.config)
        self._boundary_layer = BoundaryEnforcementLayer()

        # Pipelines
        self._fast_pipeline = FastModePipeline(actr_config=self.config)
        self._moderate_pipeline = ModerateModePipeline(actr_config=self.config)
        self._slow_pipeline = SlowModePipeline(actr_config=self.config)

    def _ensure_calibration(self) -> None:
        """Run calibration if not already done."""
        if self._platt is not None:
            return
        if self.calibrate:
            rng = random.Random(self.seed)
            n = 200
            difficulty_dist = [rng.uniform(0.3, 0.95) for _ in range(n)]
            dataset = self._calibration_engine.build_calibration_dataset(
                n_samples=n,
                difficulty_dist=difficulty_dist,
            )
            self._platt, _ = self._calibration_engine.calibrate_full(dataset)
        else:
            # No-op calibrator that just returns the raw score
            from actr.calibration import PlattCalibrator
            self._platt = PlattCalibrator()

    def _run_pipeline(
        self,
        prompt: str,
        calibrated_confidence: float,
    ):
        """Run the full ACTR pipeline for a single prompt.

        1. Mode selection
        2. Mode-specific pipeline
        3. Boundary enforcement

        Returns
        -------
        CalibratedReasoningState
        """
        # Step 1: Mode selection
        mode_result = self._mode_controller.select_mode(calibrated_confidence)

        # Step 2: Run mode-specific pipeline
        if mode_result.selected_mode == ReasoningModeEnum.FAST:
            state = self._fast_pipeline.run(
                prompt=prompt,
                calibrated_confidence=calibrated_confidence,
                mode_result=mode_result,
            )
        elif mode_result.selected_mode == ReasoningModeEnum.MODERATE:
            state = self._moderate_pipeline.run(
                prompt=prompt,
                calibrated_confidence=calibrated_confidence,
                mode_result=mode_result,
            )
        else:
            state = self._slow_pipeline.run(
                prompt=prompt,
                calibrated_confidence=calibrated_confidence,
                mode_result=mode_result,
            )

        # Step 3: Boundary enforcement
        state = self._boundary_layer.run(state, mode_result=mode_result)

        return state

    def run_query(
        self,
        query: BenchmarkQuery,
        run_ssu: bool = True,
        override_confidence: Optional[float] = None,
    ) -> BenchmarkResult:
        """Run a single benchmark query through the ACTR pipeline.

        Parameters
        ----------
        query : BenchmarkQuery
            The benchmark query to run.
        run_ssu : bool
            Whether to run the SSU engine to get a real calibrated confidence.
            If False, uses ``override_confidence`` directly.
        override_confidence : float | None
            If provided and ``run_ssu=False``, use this confidence value directly.

        Returns
        -------
        BenchmarkResult
            The result of running the query through the ACTR pipeline.
        """
        self._ensure_calibration()

        # Get calibrated confidence
        if override_confidence is not None:
            calibrated_confidence = override_confidence
        else:
            ssu_result = self._ssu.run(query.prompt)
            raw_score = ssu_result.consistency_score
            # Apply Platt calibration
            if self._platt is not None and hasattr(self._platt, "calibrate"):
                calibrated_confidence = self._platt.calibrate(raw_score)
            else:
                calibrated_confidence = raw_score

        # Run full pipeline
        state = self._run_pipeline(query.prompt, calibrated_confidence)

        # Check correctness (simple substring check)
        response_lower = state.reasoning_content.lower()
        answer_lower = query.expected_answer.lower()
        is_correct = answer_lower in response_lower or response_lower in answer_lower

        return BenchmarkResult(
            query=query,
            predicted_response=state.reasoning_content,
            calibrated_confidence=state.calibrated_confidence,
            confidence_tag=state.confidence_tag,
            selected_mode=state.reasoning_mode,
            is_correct=is_correct,
            mode_selection_result=self._mode_controller.select_mode(calibrated_confidence).selected_mode,
            boundary_violations=list(state.error_flags),
            reasoning_steps=state.reasoning_steps,
        )

    def run_suite(
        self,
        queries: Optional[list[BenchmarkQuery]] = None,
        run_ssu: bool = True,
    ) -> tuple[list[BenchmarkResult], BenchmarkSummary]:
        """Run a full benchmark suite.

        Parameters
        ----------
        queries : list[BenchmarkQuery] | None
            List of queries to run. If None, uses the built-in suite.
        run_ssu : bool
            Whether to run the SSU engine for each query.

        Returns
        -------
        results, summary
            List of per-query results and aggregate summary metrics.
        """
        if queries is None:
            queries = BUILTIN_BENCHMARK_QUERIES

        self._ensure_calibration()

        results: list[BenchmarkResult] = []
        for query in queries:
            # Derive a deterministic confidence per difficulty level for consistency
            override_confidence: Optional[float] = None
            if not run_ssu:
                override_confidence = {
                    "factual": 0.9,
                    "mathematical": 0.65,
                    "adversarial": 0.35,
                }.get(query.difficulty, 0.5)

            result = self.run_query(query, run_ssu=run_ssu, override_confidence=override_confidence)
            results.append(result)

        summary = self._compute_summary(results)
        return results, summary

    def _compute_summary(self, results: list[BenchmarkResult]) -> BenchmarkSummary:
        """Compute aggregate metrics from per-query results."""
        total = len(results)

        # Mode distribution
        mode_counts: dict[str, int] = {}
        mode_correct: dict[str, int] = {}
        mode_total: dict[str, int] = {}

        for r in results:
            mode_key = r.selected_mode.value
            mode_counts[mode_key] = mode_counts.get(mode_key, 0) + 1
            mode_total[mode_key] = mode_total.get(mode_key, 0) + 1
            if r.is_correct:
                mode_correct[mode_key] = mode_correct.get(mode_key, 0) + 1

        # Accuracy per mode
        accuracy_per_mode = {}
        for mode, total_count in mode_total.items():
            correct_count = mode_correct.get(mode, 0)
            accuracy_per_mode[mode] = correct_count / total_count if total_count > 0 else 0.0

        # Average calibrated confidence
        avg_conf = (
            sum(r.calibrated_confidence for r in results) / total if total else 0.0
        )

        # Boundary violation rate
        n_violations = sum(1 for r in results if r.boundary_violations)
        violation_rate = n_violations / total if total else 0.0

        # ECE (Expected Calibration Error)
        y_true = [1 if r.is_correct else 0 for r in results]
        y_cal = [r.calibrated_confidence for r in results]
        ece = self._compute_ece(y_true, y_cal)

        return BenchmarkSummary(
            total_queries=total,
            accuracy_per_mode=accuracy_per_mode,
            mode_distribution=mode_counts,
            average_calibrated_confidence=avg_conf,
            boundary_violation_rate=violation_rate,
            ece=ece,
        )

    @staticmethod
    def _compute_ece(y_true: list[int], y_cal: list[float], n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        import numpy as np
        y_true_arr = np.array(y_true, dtype=np.float64)
        y_cal_arr = np.array(y_cal, dtype=np.float64)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        total_weight = len(y_true_arr)
        for i in range(n_bins):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            if i < n_bins - 1:
                mask = (y_cal_arr >= lo) & (y_cal_arr < hi)
            else:
                mask = (y_cal_arr >= lo) & (y_cal_arr <= hi)
            bin_count = int(np.sum(mask))
            if bin_count == 0:
                continue
            bin_accuracy = np.mean(y_true_arr[mask])
            bin_confidence = np.mean(y_cal_arr[mask])
            ece += (bin_count / total_weight) * abs(bin_confidence - bin_accuracy)
        return float(ece)
