"""Tests for the calibration module (Step 3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is on the path
SRC_ROOT = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from actr.calibration import (
    CalibrationPair,
    CalibrationDataset,
    PlattCalibrator,
    TemperatureCalibrator,
    CalibrationEngine,
)
from actr.ssu import SSUConfig


# ---------------------------------------------------------------------------
# CalibrationPair Tests
# ---------------------------------------------------------------------------

class TestCalibrationPair:
    def test_basic_fields(self) -> None:
        pair = CalibrationPair(raw_score=0.8, is_correct=True, verbalized=0.75)
        assert pair.raw_score == 0.8
        assert pair.is_correct is True
        assert pair.verbalized == 0.75

    def test_verbalized_optional(self) -> None:
        pair = CalibrationPair(raw_score=0.5, is_correct=False)
        assert pair.verbalized is None


# ---------------------------------------------------------------------------
# CalibrationDataset Tests
# ---------------------------------------------------------------------------

class TestCalibrationDataset:
    def setup_method(self) -> None:
        self.dataset = CalibrationDataset()

    def test_add_single_pair(self) -> None:
        self.dataset.add(raw_score=0.7, is_correct=True)
        assert len(self.dataset.pairs) == 1
        assert self.dataset.pairs[0].raw_score == 0.7
        assert self.dataset.pairs[0].is_correct is True

    def test_add_with_verbalized(self) -> None:
        self.dataset.add(raw_score=0.6, is_correct=False, verbalized=0.55)
        pair = self.dataset.pairs[0]
        assert pair.verbalized == 0.55

    def test_add_multiple(self) -> None:
        for i in range(10):
            self.dataset.add(raw_score=float(i) / 10.0, is_correct=i % 2 == 0)
        assert len(self.dataset.pairs) == 10

    def test_split_half(self) -> None:
        for i in range(10):
            self.dataset.add(raw_score=float(i) * 0.1, is_correct=True)
        train, val = self.dataset.split(ratio=0.5)
        assert len(train.pairs) == 5
        assert len(val.pairs) == 5
        # Sequential check
        assert train.pairs[0].raw_score == pytest.approx(0.0)
        assert val.pairs[0].raw_score == pytest.approx(0.5)

    def test_split_70_30(self) -> None:
        for i in range(10):
            self.dataset.add(raw_score=float(i) * 0.1, is_correct=True)
        train, val = self.dataset.split(ratio=0.7)
        assert len(train.pairs) == 7
        assert len(val.pairs) == 3

    def test_split_ratio_validation(self) -> None:
        self.dataset.add(raw_score=0.5, is_correct=True)
        with pytest.raises(ValueError):
            self.dataset.split(ratio=0.0)
        with pytest.raises(ValueError):
            self.dataset.split(ratio=1.0)

    def test_to_arrays(self) -> None:
        self.dataset.add(raw_score=0.8, is_correct=True, verbalized=0.7)
        self.dataset.add(raw_score=0.4, is_correct=False, verbalized=None)
        self.dataset.add(raw_score=0.6, is_correct=True, verbalized=0.5)
        X_con, X_ver, y = self.dataset.to_arrays()
        assert X_con == pytest.approx(np.array([0.8, 0.4, 0.6]))
        assert X_ver == pytest.approx(np.array([0.7, 0.5, 0.5]))
        assert y == pytest.approx(np.array([1, 0, 1]))

    def test_to_arrays_all_verbalized_none_uses_05(self) -> None:
        self.dataset.add(raw_score=0.5, is_correct=True, verbalized=None)
        self.dataset.add(raw_score=0.3, is_correct=False, verbalized=None)
        _, X_ver, _ = self.dataset.to_arrays()
        assert X_ver == pytest.approx(np.array([0.5, 0.5]))


# ---------------------------------------------------------------------------
# PlattCalibrator Tests
# ---------------------------------------------------------------------------

class TestPlattCalibrator:
    def setup_method(self) -> None:
        self.calibrator = PlattCalibrator()

    def test_fit_requires_1d_X(self) -> None:
        X = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y = np.array([0, 0, 1, 1, 1])
        result = self.calibrator.fit(X, y)
        assert result is self.calibrator
        assert self.calibrator.model is not None

    def test_fit_stores_model(self) -> None:
        X = np.linspace(0.1, 0.9, 20)
        y = (X > 0.5).astype(int)
        self.calibrator.fit(X, y)
        assert self.calibrator.model is not None

    def test_calibrate_single_value(self) -> None:
        # Train: higher raw_score → correct
        X = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        y = np.array([0, 0, 0, 1, 1, 1])
        self.calibrator.fit(X, y)
        calibrated = self.calibrator.calibrate(0.85)
        assert 0.0 <= calibrated <= 1.0

    def test_calibrate_not_fitted_raises(self) -> None:
        with pytest.raises(RuntimeError):
            self.calibrator.calibrate(0.5)

    def test_calibrate_dataset(self) -> None:
        dataset = CalibrationDataset()
        for score, correct in [(0.2, False), (0.4, False), (0.6, True), (0.8, True)]:
            dataset.add(raw_score=score, is_correct=correct)
        X, _, y = dataset.to_arrays()
        self.calibrator.fit(X, y)
        probs = self.calibrator.calibrate_dataset(dataset)
        assert len(probs) == len(dataset.pairs)
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_calibrate_preserves_order(self) -> None:
        dataset = CalibrationDataset()
        for score, correct in [(0.1, False), (0.5, True), (0.9, True)]:
            dataset.add(raw_score=score, is_correct=correct)
        X, _, y = dataset.to_arrays()
        self.calibrator.fit(X, y)
        probs = self.calibrator.calibrate_dataset(dataset)
        assert probs[0] <= probs[1] <= probs[2]  # monotonic since correct labels are too


class TestPlattCalibratorECE:
    def test_ece_perfect_calibration(self) -> None:
        # When calibrated_prob == true_prob for every sample, ECE = 0
        n = 100
        y_true = np.array([1 if i < 50 else 0 for i in range(n)])
        # Perfectly calibrated: prob == fraction_positive in each bin
        y_calibrated = y_true.astype(float)
        ece = PlattCalibrator.compute_ece(y_true, y_calibrated, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-6)

    def test_ece_worst_case(self) -> None:
        # Worst case: all calibrated to 0.0, all are positive → ECE close to 1
        y_true = np.ones(100, dtype=int)
        y_calibrated = np.zeros(100, dtype=float)
        ece = PlattCalibrator.compute_ece(y_true, y_calibrated, n_bins=10)
        # All samples in bin 0, accuracy=1.0, confidence=0.0 → ECE = |0-1| = 1
        assert ece == pytest.approx(1.0, abs=1e-6)

    def test_ece_random_predictions(self) -> None:
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 2, 200)
        y_calibrated = rng.rand(200)
        ece = PlattCalibrator.compute_ece(y_true, y_calibrated, n_bins=10)
        assert 0.0 <= ece <= 1.0

    def test_ece_n_bins_respected(self) -> None:
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        y_calibrated = np.array([0.1, 0.15, 0.55, 0.6, 0.3, 0.7, 0.25, 0.65, 0.35, 0.75])
        ece_5 = PlattCalibrator.compute_ece(y_true, y_calibrated, n_bins=5)
        ece_10 = PlattCalibrator.compute_ece(y_true, y_calibrated, n_bins=10)
        # Both should be valid ECE values
        assert 0.0 <= ece_5 <= 1.0
        assert 0.0 <= ece_10 <= 1.0

    def test_ece_single_bin(self) -> None:
        y_true = np.array([0, 1, 1, 0])
        y_calibrated = np.array([0.4, 0.6, 0.5, 0.45])
        ece = PlattCalibrator.compute_ece(y_true, y_calibrated, n_bins=1)
        # Single bin: mean_calibrated ≈ 0.4875, accuracy = 0.5 → ECE ≈ 0.0125
        assert 0.0 <= ece <= 0.1


class TestPlattCalibratorAUROC:
    def test_auroc_perfect_separation(self) -> None:
        # Perfect separation: all positives have high score, all negatives low
        y_true = np.array([0, 0, 1, 1])
        y_calibrated = np.array([0.1, 0.2, 0.9, 0.8])
        auroc = PlattCalibrator.compute_auroc(y_true, y_calibrated)
        assert auroc == pytest.approx(1.0, abs=1e-6)

    def test_auroc_random(self) -> None:
        rng = np.random.RandomState(123)
        y_true = rng.randint(0, 2, 100)
        y_calibrated = rng.rand(100)
        auroc = PlattCalibrator.compute_auroc(y_true, y_calibrated)
        # Random predictions should be near 0.5
        assert 0.3 <= auroc <= 0.7

    def test_auroc_inverted(self) -> None:
        # Perfect inverted separation: AUROC should be ~0 (not negative)
        y_true = np.array([0, 0, 1, 1])
        y_calibrated = np.array([0.9, 0.8, 0.1, 0.2])
        auroc = PlattCalibrator.compute_auroc(y_true, y_calibrated)
        assert auroc == pytest.approx(0.0, abs=1e-6)

    def test_auroc_all_same_score(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_calibrated = np.array([0.5, 0.5, 0.5, 0.5])
        # All same scores → AUROC = 0.5 by convention
        auroc = PlattCalibrator.compute_auroc(y_true, y_calibrated)
        assert auroc == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# TemperatureCalibrator Tests
# ---------------------------------------------------------------------------

class TestTemperatureCalibrator:
    def setup_method(self) -> None:
        self.calibrator = TemperatureCalibrator()

    def test_apply_low_temperature_sharpens(self) -> None:
        # Very low temperature → sigmoid approaches step function
        p_low = TemperatureCalibrator.apply(0.5, temperature=0.1)
        p_high = TemperatureCalibrator.apply(0.5, temperature=10.0)
        # At T=0.1, raw=0.5: x=-5 → sigmoid ≈ 0.0067
        # At T=10, raw=0.5: x=-0.05 → sigmoid ≈ 0.487
        assert p_low < p_high
        assert 0.0 <= p_low <= 1.0
        assert 0.0 <= p_high <= 1.0

    def test_apply_extreme_temperatures(self) -> None:
        # T → ∞: sigmoid → 0.5 for any raw score
        p_inf = TemperatureCalibrator.apply(0.5, temperature=10000.0)
        assert p_inf == pytest.approx(0.5, abs=0.01)
        # T → 0+: sigmoid → step (1 if raw<0, 0.5 if raw=0, 0 if raw>0)
        p_zero = TemperatureCalibrator.apply(0.0, temperature=0.001)
        assert 0.0 <= p_zero <= 1.0

    def test_apply_at_extremes_raw_score(self) -> None:
        p_0 = TemperatureCalibrator.apply(0.0, temperature=1.0)
        p_1 = TemperatureCalibrator.apply(1.0, temperature=1.0)
        # sigmoid(-0) = 0.5, sigmoid(-1) < 0.5
        assert p_0 == pytest.approx(0.5)
        assert p_1 < 0.5

    def test_apply_returns_between_0_and_1(self) -> None:
        rng = np.random.RandomState(7)
        for _ in range(50):
            raw = rng.uniform(0, 1)
            temp = rng.uniform(0.05, 10.0)
            p = TemperatureCalibrator.apply(raw, temp)
            assert 0.0 <= p <= 1.0


class TestTemperatureCalibratorGridSearch:
    def setup_method(self) -> None:
        self.calibrator = TemperatureCalibrator()

    def test_find_optimal_temperature_simple(self) -> None:
        # Create data where optimal T=1.0
        raw_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        y_true = np.array([0, 0, 0, 1, 1, 1])
        optimal = self.calibrator.find_optimal_temperature(raw_scores, y_true, n_steps=20)
        assert 0.1 <= optimal <= 5.0

    def test_find_optimal_temperature_returns_in_range(self) -> None:
        rng = np.random.RandomState(99)
        raw_scores = rng.rand(50)
        y_true = (raw_scores > 0.5).astype(int)
        optimal = self.calibrator.find_optimal_temperature(
            raw_scores, y_true, temp_range=(0.2, 4.0), n_steps=30
        )
        assert 0.2 <= optimal <= 4.0

    def test_find_optimal_temperature_stable_across_runs(self) -> None:
        raw_scores = np.linspace(0.05, 0.95, 40)
        y_true = (raw_scores > 0.5).astype(int)
        # With many steps, should find similar optimum
        t1 = self.calibrator.find_optimal_temperature(raw_scores, y_true, n_steps=100)
        t2 = self.calibrator.find_optimal_temperature(raw_scores, y_true, n_steps=100)
        assert abs(t1 - t2) < 0.2  # Should be very close

    def test_find_optimal_temperature_all_correct(self) -> None:
        # All positive labels: F1 undefined/0 for any threshold → zero_division=0 handled
        raw_scores = np.array([0.1, 0.5, 0.9])
        y_true = np.array([1, 1, 1])
        optimal = self.calibrator.find_optimal_temperature(raw_scores, y_true, n_steps=10)
        assert 0.1 <= optimal <= 5.0  # Should still return something in range


# ---------------------------------------------------------------------------
# CalibrationEngine Tests
# ---------------------------------------------------------------------------

class TestCalibrationEngineInit:
    def test_init_stores_ssu_config(self) -> None:
        config = SSUConfig(alpha_consistency=0.7, alpha_verbalized=0.3)
        engine = CalibrationEngine(config)
        assert engine.ssu_config is config
        assert engine._optimal_temperature is None
        assert engine._platt is None


class TestCalibrationEngineBuildDataset:
    def test_build_dataset_returns_dataset(self) -> None:
        config = SSUConfig()
        engine = CalibrationEngine(config)
        dataset = engine.build_calibration_dataset(n_samples=20)
        assert isinstance(dataset, CalibrationDataset)
        assert len(dataset.pairs) == 20

    def test_build_dataset_all_pairs_have_scores(self) -> None:
        config = SSUConfig()
        engine = CalibrationEngine(config)
        dataset = engine.build_calibration_dataset(n_samples=50)
        for pair in dataset.pairs:
            assert 0.0 <= pair.raw_score <= 1.0
            assert isinstance(pair.is_correct, bool)

    def test_build_dataset_with_custom_n_samples(self) -> None:
        config = SSUConfig()
        engine = CalibrationEngine(config)
        for n in [1, 10, 100, 200]:
            dataset = engine.build_calibration_dataset(n_samples=n)
            assert len(dataset.pairs) == n

    def test_build_dataset_with_difficulty_dist(self) -> None:
        config = SSUConfig()
        engine = CalibrationEngine(config)
        diff_dist = [0.5] * 10
        dataset = engine.build_calibration_dataset(n_samples=10, difficulty_dist=diff_dist)
        assert len(dataset.pairs) == 10


class TestCalibrationEngineCalibrateTemperature:
    def test_calibrate_temperature_returns_float(self) -> None:
        config = SSUConfig()
        engine = CalibrationEngine(config)
        dataset = engine.build_calibration_dataset(n_samples=100)
        optimal_temp = engine.calibrate_temperature(dataset, n_steps=20)
        assert isinstance(optimal_temp, float)
        assert 0.1 <= optimal_temp <= 5.0
        assert engine._optimal_temperature == optimal_temp

    def test_calibrate_temperature_updates_internal_state(self) -> None:
        config = SSUConfig()
        engine = CalibrationEngine(config)
        dataset = engine.build_calibration_dataset(n_samples=50)
        engine.calibrate_temperature(dataset)
        assert engine._optimal_temperature is not None


class TestCalibrationEngineCalibratePlatt:
    def test_calibrate_platt_returns_fitted_platt(self) -> None:
        config = SSUConfig()
        engine = CalibrationEngine(config)
        dataset = engine.build_calibration_dataset(n_samples=100)
        platt = engine.calibrate_platt(dataset)
        assert isinstance(platt, PlattCalibrator)
        assert platt.model is not None

    def test_calibrate_platt_updates_internal_state(self) -> None:
        config = SSUConfig()
        engine = CalibrationEngine(config)
        dataset = engine.build_calibration_dataset(n_samples=50)
        engine.calibrate_platt(dataset)
        assert engine._platt is not None


class TestCalibrationEngineCalibrateFull:
    def test_calibrate_full_returns_platt_and_temp(self) -> None:
        config = SSUConfig()
        engine = CalibrationEngine(config)
        dataset = engine.build_calibration_dataset(n_samples=100)
        platt, optimal_temp = engine.calibrate_full(dataset, n_steps=15)
        assert isinstance(platt, PlattCalibrator)
        assert isinstance(optimal_temp, float)
        assert 0.1 <= optimal_temp <= 5.0

    def test_calibrate_full_updates_ssu_config(self) -> None:
        config = SSUConfig()
        engine = CalibrationEngine(config)
        dataset = engine.build_calibration_dataset(n_samples=100)
        _, optimal_temp = engine.calibrate_full(dataset, n_steps=15)
        assert engine.ssu_config.alpha_consistency == optimal_temp


class TestCalibrationEnginePersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        config = SSUConfig()
        engine = CalibrationEngine(config)
        dataset = engine.build_calibration_dataset(n_samples=100)
        platt, optimal_temp = engine.calibrate_full(dataset, n_steps=15)

        save_path = tmp_path / "calibration.json"
        engine.save_calibration(save_path)
        assert save_path.exists()

        # Load into a new engine
        config2 = SSUConfig()
        engine2 = CalibrationEngine(config2)
        engine2.load_calibration(save_path)
        assert engine2._optimal_temperature == pytest.approx(optimal_temp, abs=0.01)

    def test_load_updates_platt_model(self, tmp_path: Path) -> None:
        config = SSUConfig()
        engine = CalibrationEngine(config)
        dataset = engine.build_calibration_dataset(n_samples=100)
        engine.calibrate_full(dataset, n_steps=15)

        save_path = tmp_path / "calibration2.json"
        engine.save_calibration(save_path)

        config2 = SSUConfig()
        engine2 = CalibrationEngine(config2)
        engine2.load_calibration(save_path)
        assert engine2._platt is not None
        assert engine2._platt.model is not None


# ---------------------------------------------------------------------------
# Edge / Integration Tests
# ---------------------------------------------------------------------------

class TestCalibrationEdgeCases:
    def test_ece_with_identical_calibrated_probs(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_cal = np.array([0.5, 0.5, 0.5, 0.5])
        ece = PlattCalibrator.compute_ece(y_true, y_cal)
        # accuracy = 0.5, confidence = 0.5 → ECE = |0.5-0.5| = 0
        assert ece == pytest.approx(0.0)

    def test_auroc_with_only_positives(self) -> None:
        y_true = np.array([1, 1, 1])
        y_cal = np.array([0.3, 0.6, 0.9])
        auroc = PlattCalibrator.compute_auroc(y_true, y_cal)
        # All same label → roc_auc_score raises error typically, but should handle
        # sklearn returns 0.5 when only one class present
        assert auroc == 0.5

    def test_temperature_apply_extreme_values(self) -> None:
        # T very large → p ≈ 0.5
        p = TemperatureCalibrator.apply(0.5, temperature=1e6)
        assert p == pytest.approx(0.5, abs=0.01)
        # T very small, raw=0 → p ≈ 0.5 (at the inflection)
        p2 = TemperatureCalibrator.apply(0.0, temperature=1e-6)
        assert 0.4 <= p2 <= 0.6

    def test_dataset_split_empty_raises(self) -> None:
        dataset = CalibrationDataset()
        with pytest.raises(ValueError):
            dataset.split(ratio=0.5)

    def test_platt_calibrate_empty_dataset_raises(self) -> None:
        platt = PlattCalibrator()
        X = np.array([])
        y = np.array([])
        with pytest.raises((ValueError, RuntimeError)):
            platt.fit(X, y)
