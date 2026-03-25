"""Temperature-scaled calibration for ACTR — Platt scaling + temperature tuning.

Provides calibration infrastructure for the adaptive calibration triggered
reasoning framework, including:
- CalibrationPair / CalibrationDataset for managing calibration data
- PlattCalibrator for sigmoid (logistic regression) calibration
- TemperatureCalibrator for temperature-scaled probability estimation
- CalibrationEngine for building datasets and orchestrating the full pipeline
"""

from __future__ import annotations

import json
import random
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

from actr.ssu import SSUConfig, ThreeSampleSSU

__all__ = [
    "CalibrationPair",
    "CalibrationDataset",
    "PlattCalibrator",
    "TemperatureCalibrator",
    "CalibrationEngine",
]


# ---------------------------------------------------------------------------
# CalibrationPair & CalibrationDataset
# ---------------------------------------------------------------------------


@dataclass
class CalibrationPair:
    """A single (raw_score, correctness) pair used for calibration.

    Parameters
    ----------
    raw_score : float
        Raw consistency / uncertainty score from the SSU engine (0–1).
    is_correct : bool
        Ground-truth correctness of the reasoning (True = correct).
    verbalized : float, optional
        Optional verbalized confidence value (0–1).
    """

    raw_score: float
    is_correct: bool
    verbalized: Optional[float] = None


class CalibrationDataset:
    """A collection of calibration pairs.

    Supports adding pairs, splitting into train/val sets, and converting
    to numpy arrays for sklearn consumption.
    """

    def __init__(self) -> None:
        self.pairs: list[CalibrationPair] = []

    def add(
        self, raw_score: float, is_correct: bool, verbalized: Optional[float] = None
    ) -> None:
        """Append a new calibration pair.

        Parameters
        ----------
        raw_score : float
            Raw consistency/uncertainty score (0–1).
        is_correct : bool
            Ground-truth correctness.
        verbalized : float, optional
            Verbalized confidence (0–1).
        """
        self.pairs.append(
            CalibrationPair(raw_score=raw_score, is_correct=is_correct, verbalized=verbalized)
        )

    def split(
        self, ratio: float = 0.5
    ) -> tuple["CalibrationDataset", "CalibrationDataset"]:
        """Split the dataset sequentially into two parts.

        Parameters
        ----------
        ratio : float
            Fraction of pairs to include in the first (train) set.
            Must be in (0, 1). Default 0.5.

        Returns
        -------
        train, val
            Two CalibrationDataset objects. ``train`` contains the first
            ``ratio * len(self)`` pairs; ``val`` contains the remainder.
        """
        if not (0.0 < ratio < 1.0):
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")
        if len(self.pairs) == 0:
            raise ValueError("Cannot split an empty dataset")
        n = len(self.pairs)
        split_idx = int(n * ratio)
        train = CalibrationDataset()
        val = CalibrationDataset()
        train.pairs = self.pairs[:split_idx]
        val.pairs = self.pairs[split_idx:]
        return train, val

    def to_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert all pairs to numpy arrays.

        Returns
        -------
        X_consistency : np.ndarray
            1-D array of raw consistency scores.
        X_verbalized : np.ndarray
            1-D array of verbalized confidences (uses 0.5 when None).
        y : np.ndarray
            1-D binary array (1 = correct, 0 = incorrect).
        """
        X_consistency = np.array([p.raw_score for p in self.pairs], dtype=np.float64)
        X_verbalized = np.array(
            [p.verbalized if p.verbalized is not None else 0.5 for p in self.pairs],
            dtype=np.float64,
        )
        y = np.array([1 if p.is_correct else 0 for p in self.pairs], dtype=np.int32)
        return X_consistency, X_verbalized, y


# ---------------------------------------------------------------------------
# PlattCalibrator
# ---------------------------------------------------------------------------


class PlattCalibrator:
    """Sigmoid / Platt calibration via sklearn LogisticRegression.

    Fits a logistic regression on consistency scores to produce calibrated
    probabilities. The logistic curve maps the raw score to a well-calibrated
    probability that can be directly interpreted as a confidence.
    """

    def __init__(self) -> None:
        self.model: Optional[LogisticRegression] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        """Fit a logistic regression model.

        Parameters
        ----------
        X : np.ndarray
            1-D array of raw consistency scores.
        y : np.ndarray
            1-D binary array (1 = correct, 0 = incorrect).

        Returns
        -------
        self
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
        self.model.fit(X, y)
        return self

    def calibrate(self, raw_prob: float) -> float:
        """Map a raw consistency score to a calibrated probability.

        Parameters
        ----------
        raw_prob : float
            Raw score in [0, 1].

        Returns
        -------
        float
            Calibrated probability in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("PlattCalibrator must be fit before calibrate()")
        return float(self.model.predict_proba([[raw_prob]])[0, 1])

    def calibrate_dataset(
        self, dataset: CalibrationDataset
    ) -> list[float]:
        """Calibrate all pairs in a dataset.

        Parameters
        ----------
        dataset : CalibrationDataset
            Dataset whose raw scores will be calibrated.

        Returns
        -------
        list[float]
            List of calibrated probabilities in the same order as dataset.pairs.
        """
        if self.model is None:
            raise RuntimeError("PlattCalibrator must be fit before calibrate_dataset()")
        X, _, _ = dataset.to_arrays()
        probs = self.model.predict_proba(X.reshape(-1, 1))
        return [float(p[1]) for p in probs]

    @staticmethod
    def compute_ece(
        y_true: np.ndarray, y_calibrated: np.ndarray, n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        Bins the calibrated probabilities into ``n_bins`` equal-width bins,
        then computes the weighted average absolute deviation of each bin's
        average calibrated probability from the bin's fraction of positives.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary labels.
        y_calibrated : np.ndarray
            Calibrated probabilities in [0, 1].
        n_bins : int
            Number of equal-width bins. Default 10.

        Returns
        -------
        float
            ECE value in [0, 1]. Lower is better; 0 = perfectly calibrated.
        """
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        total_weight = len(y_true)
        for i in range(n_bins):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            # Mask for samples in this bin
            if i < n_bins - 1:
                mask = (y_calibrated >= lo) & (y_calibrated < hi)
            else:
                # Last bin is inclusive on both ends
                mask = (y_calibrated >= lo) & (y_calibrated <= hi)
            bin_count = int(np.sum(mask))
            if bin_count == 0:
                continue
            bin_accuracy = np.mean(y_true[mask])
            bin_confidence = np.mean(y_calibrated[mask])
            ece += (bin_count / total_weight) * abs(bin_confidence - bin_accuracy)
        return float(ece)

    @staticmethod
    def compute_auroc(
        y_true: np.ndarray, y_calibrated: np.ndarray
    ) -> float:
        """Compute AUROC for correctness detection.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary labels.
        y_calibrated : np.ndarray
            Calibrated probabilities in [0, 1].

        Returns
        -------
        float
            AUROC in [0, 1]. 1.0 = perfect, 0.5 = random.
        """
        # Handle single-class edge case
        if len(np.unique(y_true)) == 1:
            return 0.5
        return float(roc_auc_score(y_true, y_calibrated))


# ---------------------------------------------------------------------------
# TemperatureCalibrator
# ---------------------------------------------------------------------------


class TemperatureCalibrator:
    """Temperature-scaled sigmoid calibration.

    Scales raw consistency scores through a temperature-parameterized sigmoid:
        p = exp(-raw / T) / (1 + exp(-raw / T))

    The optimal temperature is found via grid search to maximize F1 on the
    correctness prediction task.
    """

    @staticmethod
    def _sigmoid_with_temp(raw: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature-scaled sigmoid to an array of raw scores."""
        # exp(-raw / T) / (1 + exp(-raw / T))
        # This is sigmoid(x) where x = -raw / T
        x = -raw / temperature
        # Numerically stable sigmoid
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )

    def find_optimal_temperature(
        self,
        raw_scores: np.ndarray,
        y_true: np.ndarray,
        temp_range: tuple[float, float] = (0.1, 5.0),
        n_steps: int = 50,
    ) -> float:
        """Find the temperature that maximizes F1 for correctness detection.

        Performs a grid search over ``n_steps`` temperature values in
        ``temp_range``, applies temperature scaling to ``raw_scores``, and
        selects the temperature that yields the highest F1 score.

        Parameters
        ----------
        raw_scores : np.ndarray
            1-D array of raw consistency scores.
        y_true : np.ndarray
            1-D binary array (1 = correct, 0 = incorrect).
        temp_range : tuple[float, float]
            (min_temperature, max_temperature). Default (0.1, 5.0).
        n_steps : int
            Number of grid points to evaluate. Default 50.

        Returns
        -------
        float
            Optimal temperature value.
        """
        temps = np.linspace(temp_range[0], temp_range[1], n_steps)
        best_temp = temp_range[0]
        best_f1 = -1.0

        for T in temps:
            scaled = self._sigmoid_with_temp(raw_scores, T)
            # Threshold at 0.5 to get binary predictions
            preds = (scaled >= 0.5).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_temp = T

        return float(best_temp)

    @staticmethod
    def apply(raw_score: float, temperature: float) -> float:
        """Apply temperature scaling to a single raw score.

        Parameters
        ----------
        raw_score : float
            Raw consistency score.
        temperature : float
            Temperature parameter (> 0). Higher = softer/more uncertain.

        Returns
        -------
        float
            Scaled probability in [0, 1].
        """
        x = -raw_score / temperature
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            return math.exp(x) / (1.0 + math.exp(x))


# ---------------------------------------------------------------------------
# CalibrationEngine
# ---------------------------------------------------------------------------


class CalibrationEngine:
    """Main calibration orchestration class.

    Coordinates synthetic dataset generation, temperature tuning, Platt
    calibration, and persistence of calibration state. Provides a unified
    interface for calibrating the SSU pipeline.
    """

    def __init__(self, ssu_config: SSUConfig) -> None:
        """Initialize the calibration engine.

        Parameters
        ----------
        ssu_config : SSUConfig
            SSU configuration that will be updated with calibration params.
        """
        self.ssu_config = ssu_config
        self._ssu = ThreeSampleSSU(config=ssu_config)
        self._platt: Optional[PlattCalibrator] = None
        self._optimal_temperature: Optional[float] = None

    # ------------------------------------------------------------------
    # Synthetic data generation
    # ------------------------------------------------------------------

    def build_calibration_dataset(
        self,
        n_samples: int = 200,
        difficulty_dist: Optional[list[float]] = None,
    ) -> CalibrationDataset:
        """Build a synthetic calibration dataset using the SSU engine.

        Generates ``n_samples`` synthetic prompts at varying difficulty levels,
        runs each through the SSU engine (in mock mode), and assigns correctness
        labels based on whether the resulting consistency score exceeds a
        difficulty-dependent threshold.

        Parameters
        ----------
        n_samples : int
            Number of calibration pairs to generate. Default 200.
        difficulty_dist : list[float], optional
            Difficulty parameters for each sample. If None, samples are drawn
            uniformly from [0.3, 0.95]. Each difficulty value is used as the
            threshold: samples with consistency >= threshold are labeled correct.

        Returns
        -------
        CalibrationDataset
            Dataset populated with synthetic calibration pairs.
        """
        rng = random.Random(42)  # Deterministic for reproducibility
        dataset = CalibrationDataset()

        # Pre-generate difficulty values
        if difficulty_dist is None:
            difficulty_dist = [rng.uniform(0.3, 0.95) for _ in range(n_samples)]

        for i in range(n_samples):
            difficulty = difficulty_dist[i] if i < len(difficulty_dist) else rng.uniform(0.3, 0.95)
            # Create a synthetic prompt with embedded difficulty signal
            # The mock embedding uses hash(text) → same text → same embedding
            # so we craft unique prompts that will produce varying consistency
            prompt = f"Calibration prompt {i}: solve problem at difficulty {difficulty:.4f}"

            # Run SSU
            result = self._ssu.run(prompt)

            # Use the consistency score as raw_score
            raw_score = result.consistency_score

            # Correctness label: higher consistency → correct if above threshold
            # We inject some noise so it's not perfectly deterministic
            noise = rng.uniform(-0.05, 0.05)
            threshold = difficulty + noise
            is_correct = raw_score >= threshold

            # Verbalized confidence from SSU result (may be None)
            verbalized = result.verbalized_confidence

            dataset.add(raw_score=raw_score, is_correct=is_correct, verbalized=verbalized)

        return dataset

    # ------------------------------------------------------------------
    # Calibration methods
    # ------------------------------------------------------------------

    def calibrate_temperature(
        self,
        dataset: CalibrationDataset,
        temp_range: tuple[float, float] = (0.1, 5.0),
        n_steps: int = 50,
    ) -> float:
        """Find the optimal temperature for this dataset.

        Parameters
        ----------
        dataset : CalibrationDataset
            Calibration dataset.
        temp_range : tuple[float, float]
            Temperature search range.
        n_steps : int
            Number of grid search steps.

        Returns
        -------
        float
            Optimal temperature value.
        """
        X, _, y = dataset.to_arrays()
        calibrator = TemperatureCalibrator()
        self._optimal_temperature = calibrator.find_optimal_temperature(
            X, y, temp_range=temp_range, n_steps=n_steps
        )
        return self._optimal_temperature

    def calibrate_platt(self, dataset: CalibrationDataset) -> PlattCalibrator:
        """Fit a Platt calibrator on the dataset.

        Parameters
        ----------
        dataset : CalibrationDataset
            Calibration dataset.

        Returns
        -------
        PlattCalibrator
            Fitted Platt calibrator instance.
        """
        X, _, y = dataset.to_arrays()
        self._platt = PlattCalibrator()
        self._platt.fit(X, y)
        return self._platt

    def calibrate_full(
        self,
        dataset: CalibrationDataset,
        temp_range: tuple[float, float] = (0.1, 5.0),
        n_steps: int = 50,
    ) -> tuple[PlattCalibrator, float]:
        """Run the full calibration pipeline.

        Splits the dataset 50/50 into train/val. Trains both the temperature
        calibrator and Platt calibrator on the train split. Stores the results
        and updates the internal SSUConfig.

        Parameters
        ----------
        dataset : CalibrationDataset
            Calibration dataset.
        temp_range : tuple[float, float]
            Temperature search range.
        n_steps : int
            Number of temperature grid steps.

        Returns
        -------
        platt, optimal_temp
            Fitted PlattCalibrator and the optimal temperature.
        """
        train, _ = dataset.split(ratio=0.5)

        # Temperature calibration on training split
        optimal_temp = self.calibrate_temperature(train, temp_range=temp_range, n_steps=n_steps)

        # Platt calibration on training split
        platt = self.calibrate_platt(train)

        # Update SSUConfig with calibration results
        self.ssu_config.alpha_consistency = optimal_temp  # type: ignore[assignment]

        return platt, optimal_temp

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_calibration(self, path: Path) -> None:
        """Save calibration state to a JSON file.

        Parameters
        ----------
        path : Path
            Output file path.
        """
        data: dict[str, object] = {
            "optimal_temperature": (
                self._optimal_temperature if self._optimal_temperature is not None else 1.0
            ),
        }

        # Serialize Platt coefficients if available
        if self._platt is not None and self._platt.model is not None:
            data["platt_coef"] = self._platt.model.coef_[0].tolist()  # type: ignore[union-attr]
            data["platt_intercept"] = self._platt.model.intercept_[0].tolist()  # type: ignore[union-attr]
        else:
            data["platt_coef"] = [1.0]
            data["platt_intercept"] = 0.0

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)

    def load_calibration(self, path: Path) -> None:
        """Load calibration state from a JSON file.

        Parameters
        ----------
        path : Path
            Input file path.
        """
        with open(path) as fh:
            data = json.load(fh)

        self._optimal_temperature = float(data.get("optimal_temperature", 1.0))

        platt = PlattCalibrator()
        coefs = data.get("platt_coef", [1.0])
        intercept = float(data.get("platt_intercept", 0.0))
        platt.model = LogisticRegression(solver="lbfgs", max_iter=1000)
        # Fake-fit by manually setting coefficients (needed for deserialization)
        # We'll just store a pre-fitted model with known params
        # Since sklearn doesn't expose direct coefficient setting easily post-init,
        # we use a workaround: fit a dummy dataset that produces the same mapping
        platt.model.coef_ = np.array([coefs])
        platt.model.intercept_ = np.array([intercept])
        platt.model.classes_ = np.array([0, 1])
        self._platt = platt
