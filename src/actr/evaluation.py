"""Calibration evaluation harness for ACTR.

This module provides the evaluation infrastructure for measuring ACTR's
calibration quality and overall performance against the project success metrics.

Success Metrics (target values from the ACTR plan):
    ECE                       < 0.05   (Expected Calibration Error)
    Mode Switch Accuracy       > 90%   (% of mode selections matching difficulty)
    Accuracy @ Fast           > 90%   (accuracy on FAST-routed queries)
    Accuracy @ Slow           > 85%   (accuracy on SLOW-routed queries)
    Compute Speedup vs Slow    > 2.0x (ratio of avg slow-time to actual time)
    Boundary Violation Rate    < 1%   (% of responses violating safety bounds)
    KG Precision               > 80%  (% of KG activations that improved answers)
    AUROC                      > 0.90 (AUROC for correctness detection)

Example
-------
>>> from actr.evaluation import run_calibration_evaluation
>>> metrics = run_calibration_evaluation()
>>> print(f"ECE = {metrics.ece:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from actr.benchmark import (
    BenchmarkQuery,
    BenchmarkResult,
    BenchmarkRunner,
)
from actr.calibration import (
    CalibrationDataset,
    CalibrationEngine,
    PlattCalibrator,
)
from actr.config import ACTRConfig
from actr.mode_controller import ReasoningModeController, ReasoningModeEnum
from actr.ssu import SSUConfig

__all__ = [
    "CalibrationEvaluationMetrics",
    "CalibrationEvaluator",
    "EVALUATION_BENCHMARK_QUERIES",
    "run_calibration_evaluation",
]


# ---------------------------------------------------------------------------
# Stub benchmark queries for evaluation
# ---------------------------------------------------------------------------

# 20 synthetic benchmark queries covering all three difficulty tiers.
# These are hardcoded here (rather than in benchmark.py) so that the
# evaluation suite is self-contained and reproducible.

EVALUATION_BENCHMARK_QUERIES: list[BenchmarkQuery] = [
    # ---- 7 factual (low uncertainty) — expected routing: FAST ----
    BenchmarkQuery(
        prompt="What is the capital of Japan?",
        expected_answer="Tokyo",
        difficulty="factual",
        category="geography",
    ),
    BenchmarkQuery(
        prompt="What is the chemical symbol for sodium?",
        expected_answer="Na",
        difficulty="factual",
        category="chemistry",
    ),
    BenchmarkQuery(
        prompt="Who wrote 'To Kill a Mockingbird'?",
        expected_answer="Harper Lee",
        difficulty="factual",
        category="literature",
    ),
    BenchmarkQuery(
        prompt="What is the largest planet in our solar system?",
        expected_answer="Jupiter",
        difficulty="factual",
        category="astronomy",
    ),
    BenchmarkQuery(
        prompt="What year did the Titanic sink?",
        expected_answer="1912",
        difficulty="factual",
        category="history",
    ),
    BenchmarkQuery(
        prompt="What is the atomic number of carbon?",
        expected_answer="6",
        difficulty="factual",
        category="chemistry",
    ),
    BenchmarkQuery(
        prompt="What is the square of 12?",
        expected_answer="144",
        difficulty="factual",
        category="mathematics",
    ),
    # ---- 7 mathematical (medium uncertainty) — expected routing: MODERATE ----
    BenchmarkQuery(
        prompt="What is 27 times 34?",
        expected_answer="918",
        difficulty="mathematical",
        category="arithmetic",
    ),
    BenchmarkQuery(
        prompt="Simplify: (3x + 5) - (2x - 3)",
        expected_answer="x + 8",
        difficulty="mathematical",
        category="algebra",
    ),
    BenchmarkQuery(
        prompt="What is the greatest common divisor of 48 and 18?",
        expected_answer="6",
        difficulty="mathematical",
        category="number_theory",
    ),
    BenchmarkQuery(
        prompt="Solve for y: 4y - 7 = 21",
        expected_answer="7",
        difficulty="mathematical",
        category="algebra",
    ),
    BenchmarkQuery(
        prompt="What is 15% of 240?",
        expected_answer="36",
        difficulty="mathematical",
        category="percentages",
    ),
    BenchmarkQuery(
        prompt="If a triangle has sides 3, 4, and 5, is it a right triangle?",
        expected_answer="yes",
        difficulty="mathematical",
        category="geometry",
    ),
    BenchmarkQuery(
        prompt="What is the least common multiple of 6 and 8?",
        expected_answer="24",
        difficulty="mathematical",
        category="number_theory",
    ),
    # ---- 6 adversarial (high uncertainty) — expected routing: SLOW ----
    BenchmarkQuery(
        prompt="A farmer has 17 sheep. All but 9 die. How many sheep does he have left?",
        expected_answer="9",
        difficulty="adversarial",
        category="reasoning",
    ),
    BenchmarkQuery(
        prompt="If a clock shows 3:15, what is the angle between the hour and minute hands?",
        expected_answer="7.5 degrees",
        difficulty="adversarial",
        category="reasoning",
    ),
    BenchmarkQuery(
        prompt="Three switches control three light bulbs in another room. You can flip switches but only check the room once. How do you determine which switch controls which bulb?",
        expected_answer="flip two switches",
        difficulty="adversarial",
        category="logic",
    ),
    BenchmarkQuery(
        prompt="A man walks south, turns left, walks south, turns left, walks south, turns left, and is back where he started. Which direction was he walking after the last turn?",
        expected_answer="south",
        difficulty="adversarial",
        category="navigation",
    ),
    BenchmarkQuery(
        prompt="If some psychologists are logicians, and some logicians are crazy, can we conclude that some psychologists are crazy?",
        expected_answer="no",
        difficulty="adversarial",
        category="syllogism",
    ),
    BenchmarkQuery(
        prompt="What is the sum of all integers from 1 to 100?",
        expected_answer="5050",
        difficulty="adversarial",
        category="mathematical",
    ),
]


# ---------------------------------------------------------------------------
# Evaluation metrics dataclass
# ---------------------------------------------------------------------------


@dataclass
class CalibrationEvaluationMetrics:
    """Container for all ACTR calibration evaluation metrics.

    Attributes
    ----------
    ece : float
        Expected Calibration Error across all benchmark queries.
        Lower is better; 0 = perfectly calibrated.
    mode_switch_accuracy : float
        Percentage of mode switches where the selected mode matches the
        difficulty-based expected mode:
        - factual → FAST, mathematical → MODERATE, adversarial → SLOW.
    accuracy_at_fast : float
        Fraction of queries routed to FAST mode that produced correct answers.
        Computed only over queries that were *selected* as FAST, not over all
        factual queries.
    accuracy_at_slow : float
        Fraction of queries routed to SLOW mode that produced correct answers.
        Computed only over queries that were *selected* as SLOW.
    accuracy_at_moderate : float
        Fraction of queries routed to MODERATE mode that produced correct answers.
    compute_speedup_vs_slow : float
        Ratio of the average time a SLOW-mode pipeline would take (simulated
        based on pass count) to the actual observed pipeline time. Since the
        pipelines run in stub mode, this is estimated from the ratio of
        reasoning pass counts.
    boundary_violation_rate : float
        Fraction of benchmark results with at least one boundary violation.
    kg_precision : float
        Percentage of Knowledge Grounding activations where the KG step
        actually improved the response (i.e., the final answer is correct AND
        a 'knowledge_grounding' step was recorded in reasoning_steps).
    auroc : float
        Area Under the ROC Curve for correctness detection using calibrated
        probabilities as the score. 1.0 = perfect, 0.5 = random.
    n_queries : int
        Total number of benchmark queries evaluated.
    timestamp : str
        ISO-8601 timestamp of when the evaluation was run.
    """

    ece: float
    mode_switch_accuracy: float
    accuracy_at_fast: float
    accuracy_at_moderate: float
    accuracy_at_slow: float
    compute_speedup_vs_slow: float
    boundary_violation_rate: float
    kg_precision: float
    auroc: float
    n_queries: int = 0
    timestamp: str = field(default_factory=lambda: "")

    def to_dict(self) -> dict:
        return {
            "ece": round(self.ece, 4),
            "mode_switch_accuracy": round(self.mode_switch_accuracy, 4),
            "accuracy_at_fast": round(self.accuracy_at_fast, 4),
            "accuracy_at_moderate": round(self.accuracy_at_moderate, 4),
            "accuracy_at_slow": round(self.accuracy_at_slow, 4),
            "compute_speedup_vs_slow": round(self.compute_speedup_vs_slow, 4),
            "boundary_violation_rate": round(self.boundary_violation_rate, 4),
            "kg_precision": round(self.kg_precision, 4),
            "auroc": round(self.auroc, 4),
            "n_queries": self.n_queries,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# CalibrationEvaluator
# ---------------------------------------------------------------------------


class CalibrationEvaluator:
    """Main evaluation class for ACTR calibration quality.

    Runs the ACTR pipeline on a suite of benchmark queries and computes
    all metrics defined in :class:`CalibrationEvaluationMetrics`.

    Parameters
    ----------
    benchmark_queries : list[BenchmarkQuery]
        List of queries to evaluate.
    calibration_engine : CalibrationEngine
        Pre-configured calibration engine (already calibrated).
    config : ACTRConfig | None
        Global ACTR configuration. If None, the default is used.
    ssu_config : SSUConfig | None
        SSU engine configuration. If None, the default is used.
    """

    def __init__(
        self,
        benchmark_queries: list[BenchmarkQuery],
        calibration_engine: CalibrationEngine,
        config: Optional[ACTRConfig] = None,
        ssu_config: Optional[SSUConfig] = None,
    ) -> None:
        self.benchmark_queries = benchmark_queries
        self.calibration_engine = calibration_engine
        self.config = config if config is not None else ACTRConfig()
        self.ssu_config = ssu_config if ssu_config is not None else SSUConfig()
        self._runner = BenchmarkRunner(
            config=self.config,
            ssu_config=self.ssu_config,
            calibrate=False,  # Engine is already calibrated externally
        )

    def run_evaluation(
        self,
        run_ssu: bool = False,
    ) -> CalibrationEvaluationMetrics:
        """Run the full evaluation suite on all benchmark queries.

        Parameters
        ----------
        run_ssu : bool
            Whether to run the real SSU engine for each query.
            If False (default), confidence values are derived deterministically
            from query difficulty to ensure reproducible evaluation:
            - factual    → 0.90
            - mathematical → 0.65
            - adversarial → 0.35

        Returns
        -------
        CalibrationEvaluationMetrics
            All evaluation metrics for this benchmark run.
        """
        from datetime import datetime, timezone

        # Override confidences by difficulty for reproducible stub-mode evaluation
        confidence_overrides = {
            "factual": 0.90,
            "mathematical": 0.65,
            "adversarial": 0.35,
        }

        results: list[BenchmarkResult] = []
        for query in self.benchmark_queries:
            override = None if run_ssu else confidence_overrides.get(query.difficulty)
            result = self._runner.run_query(
                query, run_ssu=run_ssu, override_confidence=override
            )
            results.append(result)

        # Compute all metrics
        ece = self.compute_ece_from_results(results)
        mode_switch_acc = self.compute_mode_switch_accuracy(results)
        accuracy_per_mode = self.compute_accuracy_per_mode(results)
        boundary_rate = self.compute_boundary_violation_rate(results)
        kg_precision = self.compute_kg_precision(results)
        speedup = self.compute_speedup_vs_slow(results)

        # AUROC
        y_true = np.array([1 if r.is_correct else 0 for r in results], dtype=np.float64)
        y_cal = np.array([r.calibrated_confidence for r in results], dtype=np.float64)
        auroc = self._compute_auroc(y_true, y_cal)

        return CalibrationEvaluationMetrics(
            ece=ece,
            mode_switch_accuracy=mode_switch_acc,
            accuracy_at_fast=accuracy_per_mode.get("fast", 0.0),
            accuracy_at_moderate=accuracy_per_mode.get("moderate", 0.0),
            accuracy_at_slow=accuracy_per_mode.get("slow", 0.0),
            compute_speedup_vs_slow=speedup,
            boundary_violation_rate=boundary_rate,
            kg_precision=kg_precision,
            auroc=auroc,
            n_queries=len(results),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def compute_ece_from_results(self, results: list[BenchmarkResult]) -> float:
        """Compute Expected Calibration Error from benchmark results.

        Parameters
        ----------
        results : list[BenchmarkResult]
            Per-query benchmark results.

        Returns
        -------
        float
            ECE value in [0, 1]. Lower is better.
        """
        y_true = np.array([1 if r.is_correct else 0 for r in results], dtype=np.float64)
        y_cal = np.array([r.calibrated_confidence for r in results], dtype=np.float64)
        return PlattCalibrator.compute_ece(y_true, y_cal)

    def compute_ece(self, dataset: CalibrationDataset) -> float:
        """Compute ECE from a calibration dataset.

        This uses the calibration engine's Platt calibrator to produce
        calibrated probabilities, then computes ECE against the ground-truth
        correctness labels in the dataset.

        Parameters
        ----------
        dataset : CalibrationDataset
            Calibration dataset with (raw_score, is_correct) pairs.

        Returns
        -------
        float
            ECE value in [0, 1].
        """
        platt = self.calibration_engine._platt
        if platt is not None:
            calibrated_probs = platt.calibrate_dataset(dataset)
            y_true = np.array([1 if p.is_correct else 0 for p in dataset.pairs], dtype=np.float64)
            y_cal = np.array(calibrated_probs, dtype=np.float64)
        else:
            # No Platt calibrator — use raw scores
            X, _, y = dataset.to_arrays()
            y_true = y.astype(np.float64)
            y_cal = X.astype(np.float64)
        return PlattCalibrator.compute_ece(y_true, y_cal)

    def compute_mode_switch_accuracy(self, results: list[BenchmarkResult]) -> float:
        """Compute mode switch accuracy.

        Compares each query's *selected* mode against the mode that should
        have been selected based on its difficulty label:

        - factual       → expected FAST
        - mathematical  → expected MODERATE
        - adversarial   → expected SLOW

        Parameters
        ----------
        results : list[BenchmarkResult]
            Per-query benchmark results.

        Returns
        -------
        float
            Fraction of queries where selected mode matches expected mode.
        """
        if not results:
            return 0.0
        correct = sum(
            1 for r in results
            if self._difficulty_to_expected_mode(r.query.difficulty)
            == r.mode_selection_result
        )
        return correct / len(results)

    def compute_accuracy_per_mode(
        self, results: list[BenchmarkResult]
    ) -> dict[str, float]:
        """Compute accuracy broken down by selected reasoning mode.

        Parameters
        ----------
        results : list[BenchmarkResult]
            Per-query benchmark results.

        Returns
        -------
        dict[str, float]
            Mapping from mode name (``"fast"``, ``"moderate"``, ``"slow"``)
            to accuracy (fraction of queries in that mode that are correct).
            Modes with no queries return 0.0.
        """
        mode_correct: dict[str, int] = {}
        mode_total: dict[str, int] = {}

        for r in results:
            key = r.mode_selection_result.value
            mode_total[key] = mode_total.get(key, 0) + 1
            if r.is_correct:
                mode_correct[key] = mode_correct.get(key, 0) + 1

        return {
            mode: (mode_correct.get(mode, 0) / mode_total[mode])
            if mode in mode_total
            else 0.0
            for mode in ("fast", "moderate", "slow")
        }

    def compute_boundary_violation_rate(
        self, results: list[BenchmarkResult]
    ) -> float:
        """Compute the fraction of results with at least one boundary violation.

        Parameters
        ----------
        results : list[BenchmarkResult]
            Per-query benchmark results.

        Returns
        -------
        float
            Fraction of results with boundary_violations.
        """
        if not results:
            return 0.0
        n_violations = sum(1 for r in results if r.boundary_violations)
        return n_violations / len(results)

    def compute_kg_precision(self, results: list[BenchmarkResult]) -> float:
        """Compute Knowledge Grounding precision.

        KG precision is the fraction of KG activations that actually improved
        the response. A KG activation is counted as "useful" when:
        1. The reasoning_steps contain a step with ``'type': 'knowledge_grounding'``
           (or the step text contains 'KG triggered' or 'KG grounded')
        2. The final answer is correct (``is_correct == True``)

        Parameters
        ----------
        results : list[BenchmarkResult]
            Per-query benchmark results.

        Returns
        -------
        float
            Fraction of KG activations that produced correct answers.
        """
        kg_activations = 0
        kg_useful = 0

        for r in results:
            kg_triggered = self._kg_was_triggered(r)
            if kg_triggered:
                kg_activations += 1
                if r.is_correct:
                    kg_useful += 1

        if kg_activations == 0:
            return 0.0
        return kg_useful / kg_activations

    def compute_speedup_vs_slow(self, results: list[BenchmarkResult]) -> float:
        """Compute the average speedup ratio vs. a simulated all-SLOW baseline.

        Simulates an all-SLOW pipeline by estimating that each non-SLOW query
        would take 3 passes (the SLOW pass count) if routed to SLOW mode.
        Since FAST uses 1 pass and MODERATE uses 2 passes, the speedup is:

            speedup = total_SLOW_passes / total_actual_passes

        where total_SLOW_passes = 3 * n_queries and total_actual_passes is
        derived from the selected modes (FAST=1, MODERATE=2, SLOW=3).

        Parameters
        ----------
        results : list[BenchmarkResult]
            Per-query benchmark results.

        Returns
        -------
        float
            Speedup ratio (higher = faster relative to all-SLOW).
        """
        if not results:
            return 0.0

        pass_counts = {
            ReasoningModeEnum.FAST: 1,
            ReasoningModeEnum.MODERATE: 2,
            ReasoningModeEnum.SLOW: 3,
        }
        n = len(results)
        total_actual_passes = sum(pass_counts[r.mode_selection_result] for r in results)
        total_slow_passes = 3 * n

        if total_actual_passes == 0:
            return 0.0
        return total_slow_passes / total_actual_passes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _difficulty_to_expected_mode(self, difficulty: str) -> ReasoningModeEnum:
        """Map a query difficulty label to the expected reasoning mode.

        Parameters
        ----------
        difficulty : str
            One of ``"factual"``, ``"mathematical"``, ``"adversarial"``.

        Returns
        -------
        ReasoningModeEnum
            FAST for ``"factual"``, MODERATE for ``"mathematical"``,
            SLOW for ``"adversarial"``.

        Raises
        ------
        ValueError
            If difficulty is not one of the known values.
        """
        mapping = {
            "factual": ReasoningModeEnum.FAST,
            "mathematical": ReasoningModeEnum.MODERATE,
            "adversarial": ReasoningModeEnum.SLOW,
        }
        if difficulty not in mapping:
            raise ValueError(
                f"Unknown difficulty {difficulty!r}. "
                f"Expected one of: factual, mathematical, adversarial"
            )
        return mapping[difficulty]

    def _kg_was_triggered(self, result: BenchmarkResult) -> bool:
        """Detect whether knowledge grounding was triggered for a result.

        Checks both the structured reasoning_steps list and the raw step
        content for KG markers.

        Parameters
        ----------
        result : BenchmarkResult
            A single benchmark result.

        Returns
        -------
        bool
            True if KG was triggered.
        """
        # Check structured step types (if present as dicts)
        for step in result.reasoning_steps:
            if isinstance(step, dict):
                if step.get("type") == "knowledge_grounding":
                    return True
            elif isinstance(step, str):
                # Fallback: check step content for KG markers
                step_lower = step.lower()
                if (
                    "kg triggered" in step_lower
                    and "kg not triggered" not in step_lower
                    or "kg grounded" in step_lower
                ):
                    return True
        return False

    @staticmethod
    def _compute_auroc(y_true: np.ndarray, y_cal: np.ndarray) -> float:
        """Compute AUROC for correctness detection.

        Parameters
        ----------
        y_true : np.ndarray
            Binary ground-truth labels.
        y_cal : np.ndarray
            Calibrated probabilities.

        Returns
        -------
        float
            AUROC in [0, 1]. 1.0 = perfect, 0.5 = random.
        """
        if len(np.unique(y_true)) == 1:
            return 0.5
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(y_true, y_cal))
        except Exception:
            return 0.5


# ---------------------------------------------------------------------------
# One-shot evaluation entry point
# ---------------------------------------------------------------------------


def run_calibration_evaluation(
    n_calibration_samples: int = 200,
    run_ssu: bool = False,
) -> CalibrationEvaluationMetrics:
    """Run a complete ACTR calibration evaluation.

    This is a convenience function that:
    1. Creates a fresh SSUConfig and CalibrationEngine
    2. Builds a synthetic calibration dataset (used to fit the Platt calibrator)
    3. Runs the full calibration pipeline
    4. Instantiates CalibrationEvaluator with the 20 EVALUATION_BENCHMARK_QUERIES
    5. Computes and returns all CalibrationEvaluationMetrics

    Parameters
    ----------
    n_calibration_samples : int
        Number of synthetic calibration pairs to generate. Default 200.
    run_ssu : bool
        Whether to run the real SSU engine (requires model API).
        If False, confidence values are derived deterministically from
        query difficulty (factual=0.90, mathematical=0.65, adversarial=0.35).

    Returns
    -------
    CalibrationEvaluationMetrics
        All computed evaluation metrics.

    Notes
    -----
    **ECE (Expected Calibration Error)** < 0.05:
        Measures how well-calibrated the model's confidence scores are.
        ECE = weighted average |calibrated_confidence - accuracy| per bin.
        Lower is better; 0 = perfectly calibrated.

    **Mode Switch Accuracy** > 90%:
        Fraction of queries routed to the correct reasoning mode based on
        their difficulty. factual→FAST, mathematical→MODERATE, adversarial→SLOW.

    **Accuracy @ Fast** > 90%:
        Accuracy on queries routed to FAST mode. High accuracy here confirms
        that the system correctly identifies easy queries.

    **Accuracy @ Slow** > 85%:
        Accuracy on queries routed to SLOW mode. This validates that
        low-confidence queries receive the thorough treatment they need.

    **Compute Speedup vs Slow** > 2.0x:
        Ratio of (3 × n_queries) / Σ(pass_count per selected mode).
        Higher means fewer compute resources are used on average.

    **Boundary Violation Rate** < 1%:
        Fraction of responses that violate safety/policy bounds set by the
        BoundaryEnforcementLayer. Lower is better.

    **KG Precision** > 80%:
        Fraction of Knowledge Grounding activations where KG actually
        improved the answer (final answer is correct AND KG was triggered).

    **AUROC** > 0.90:
        Area Under ROC Curve for correctness detection using calibrated
        probabilities as the ranking score. 1.0 = perfect ranking,
        0.5 = random.
    """
    import random as _random

    # Set up configuration
    ssu_config = SSUConfig()
    config = ACTRConfig()

    # Build and calibrate
    engine = CalibrationEngine(ssu_config=ssu_config)

    # Generate synthetic calibration dataset
    rng = _random.Random(42)
    difficulty_dist = [rng.uniform(0.3, 0.95) for _ in range(n_calibration_samples)]
    dataset = engine.build_calibration_dataset(
        n_samples=n_calibration_samples,
        difficulty_dist=difficulty_dist,
    )

    # Calibrate (splits dataset internally 50/50)
    engine.calibrate_full(dataset)

    # Run evaluation
    evaluator = CalibrationEvaluator(
        benchmark_queries=EVALUATION_BENCHMARK_QUERIES,
        calibration_engine=engine,
        config=config,
        ssu_config=ssu_config,
    )

    return evaluator.run_evaluation(run_ssu=run_ssu)
