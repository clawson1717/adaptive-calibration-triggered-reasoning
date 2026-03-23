"""Tests for the 3-SSU Engine (Step 2)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

# Ensure src/ is on the path
SRC_ROOT = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from actr.ssu import (
    ThreeSampleSSU,
    SSUSample,
    SSUConfig,
    SSUResult,
    EmbeddingSimilarity,
    VerbalizedConfidenceExtractor,
)


# ---------------------------------------------------------------------------
# VerbalizedConfidenceExtractor Tests
# ---------------------------------------------------------------------------


class TestVerbalizedConfidenceExtractor:
    """Tests for regex-based confidence extraction."""

    def setup_method(self) -> None:
        self.extractor = VerbalizedConfidenceExtractor()

    def test_extract_percentage(self) -> None:
        """Test extraction of 'I'm 85% certain'."""
        result = self.extractor.extract("I'm 85% certain")
        assert result == pytest.approx(0.85)

    def test_extract_decimal(self) -> None:
        """Test extraction of 'confidence: 0.75'."""
        result = self.extractor.extract("confidence: 0.75")
        assert result == pytest.approx(0.75)

    def test_extract_out_of_10(self) -> None:
        """Test extraction of '7 out of 10'."""
        result = self.extractor.extract("7 out of 10")
        assert result == pytest.approx(0.7)

    def test_extract_range(self) -> None:
        """Test extraction of 'between 0.6 and 0.8' returns midpoint."""
        result = self.extractor.extract("between 0.6 and 0.8")
        assert result == pytest.approx(0.7)

    def test_extract_none(self) -> None:
        """Test extraction with no confidence mentioned returns None."""
        result = self.extractor.extract("The sky is blue.")
        assert result is None

    def test_extract_betwee_n(self) -> None:
        """Test 'between 70% and 90%' returns midpoint as 0.8."""
        result = self.extractor.extract("between 70% and 90%")
        assert result == pytest.approx(0.8)

    def test_extract_certainty_suffix(self) -> None:
        """Test '75% certainty'."""
        result = self.extractor.extract("I'm 75% certain about this.")
        assert result == pytest.approx(0.75)

    def test_extract_likelihood(self) -> None:
        """Test 'likelihood: 0.65'."""
        result = self.extractor.extract("likelihood: 0.65")
        assert result == pytest.approx(0.65)

    def test_extract_probability(self) -> None:
        """Test 'probability: 0.92'."""
        result = self.extractor.extract("probability: 0.92")
        assert result == pytest.approx(0.92)

    def test_extract_no_confidence_text(self) -> None:
        """Test with text that has numbers but no confidence patterns."""
        result = self.extractor.extract("The answer is 42 and there are 100 reasons.")
        assert result is None

    def test_extract_confidence_label(self) -> None:
        """Test 'confidence: 0.91'."""
        result = self.extractor.extract("confidence: 0.91")
        assert result == pytest.approx(0.91)

    def test_extract_confidence_label_percent(self) -> None:
        """Test 'confidence: 91%'."""
        result = self.extractor.extract("confidence: 91%")
        assert result == pytest.approx(0.91)


# ---------------------------------------------------------------------------
# SSUSample Tests
# ---------------------------------------------------------------------------


class TestSSUSample:
    """Tests for the SSUSample dataclass."""

    def test_instantiation(self) -> None:
        """Test SSUSample instantiation with all fields."""
        sample = SSUSample(
            text="This is a test.",
            sample_type="standard",
            logprob=-1.5,
            verbalized_confidence=0.87,
        )
        assert sample.text == "This is a test."
        assert sample.sample_type == "standard"
        assert sample.logprob == -1.5
        assert sample.verbalized_confidence == 0.87

    def test_instantiation_optional_fields(self) -> None:
        """Test SSUSample with optional fields as None."""
        sample = SSUSample(text="Test", sample_type="high_temp")
        assert sample.logprob is None
        assert sample.verbalized_confidence is None

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        sample = SSUSample(
            text="Test text",
            sample_type="contrastive",
            logprob=-0.8,
            verbalized_confidence=0.92,
        )
        d = sample.to_dict()
        assert d["text"] == "Test text"
        assert d["sample_type"] == "contrastive"
        assert d["logprob"] == -0.8
        assert d["verbalized_confidence"] == 0.92


# ---------------------------------------------------------------------------
# SSUConfig Tests
# ---------------------------------------------------------------------------


class TestSSUConfig:
    """Tests for SSUConfig dataclass."""

    def test_default_values(self) -> None:
        """Test SSUConfig with all defaults."""
        config = SSUConfig()
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.alpha_consistency == 0.6
        assert config.alpha_verbalized == 0.4
        assert config.high_temp_delta == 0.5
        assert config.contrastive_prefix == "Alternatively: "
        assert config.n_embedding_samples == 3
        assert config.device == "cpu"

    def test_custom_values(self) -> None:
        """Test SSUConfig with custom values."""
        config = SSUConfig(
            model_name="custom-model",
            alpha_consistency=0.7,
            alpha_verbalized=0.3,
            high_temp_delta=0.8,
            contrastive_prefix="Wrong answer: ",
            n_embedding_samples=5,
            device="cuda",
        )
        assert config.model_name == "custom-model"
        assert config.alpha_consistency == 0.7
        assert config.alpha_verbalized == 0.3
        assert config.high_temp_delta == 0.8
        assert config.contrastive_prefix == "Wrong answer: "
        assert config.n_embedding_samples == 5
        assert config.device == "cuda"


# ---------------------------------------------------------------------------
# EmbeddingSimilarity Tests
# ---------------------------------------------------------------------------


class TestEmbeddingSimilarity:
    """Tests for the EmbeddingSimilarity class."""

    def test_cosine_similarity_range(self) -> None:
        """Test that cosine_similarity returns a value in [-1, 1]."""
        import torch

        sim = EmbeddingSimilarity(mock=True)
        # Identical vectors
        v = torch.ones(384)
        result = sim.cosine_similarity(v, v)
        assert -1.0 <= result <= 1.0
        # Orthogonal vectors (random)
        a = torch.randn(384)
        b = torch.randn(384)
        result = sim.cosine_similarity(a, b)
        assert -1.0 <= result <= 1.0

    def test_semantic_consistency_with_two_samples(self) -> None:
        """Test semantic_consistency with two samples returns value in [0, 1]."""
        sim = EmbeddingSimilarity(mock=True)
        samples = [
            SSUSample(text="test sample one", sample_type="standard"),
            SSUSample(text="test sample two", sample_type="high_temp"),
        ]
        score = sim.semantic_consistency(samples)
        assert 0.0 <= score <= 1.0

    def test_semantic_consistency_with_three_samples(self) -> None:
        """Test semantic_consistency with three samples returns value in [0, 1]."""
        sim = EmbeddingSimilarity(mock=True)
        samples = [
            SSUSample(text="sample one", sample_type="standard"),
            SSUSample(text="sample two", sample_type="high_temp"),
            SSUSample(text="sample three", sample_type="contrastive"),
        ]
        score = sim.semantic_consistency(samples)
        assert 0.0 <= score <= 1.0

    def test_semantic_consistency_single_sample(self) -> None:
        """Test that a single sample returns consistency of 1.0."""
        sim = EmbeddingSimilarity(mock=True)
        samples = [
            SSUSample(text="only one sample", sample_type="standard"),
        ]
        score = sim.semantic_consistency(samples)
        assert score == 1.0

    def test_identical_texts_high_consistency(self) -> None:
        """Identical texts should produce high consistency (~1.0 with mock)."""
        sim = EmbeddingSimilarity(mock=True)
        text = "This is the same text repeated"
        samples = [
            SSUSample(text=text, sample_type="standard"),
            SSUSample(text=text, sample_type="high_temp"),
            SSUSample(text=text, sample_type="contrastive"),
        ]
        score = sim.semantic_consistency(samples)
        # With mock=True and identical text, hash is identical so vector is identical
        assert score == pytest.approx(1.0)

    def test_completely_different_texts(self) -> None:
        """Completely different texts should produce low-to-variable consistency."""
        sim = EmbeddingSimilarity(mock=True)
        samples = [
            SSUSample(text="apple banana cherry", sample_type="standard"),
            SSUSample(text="zebra xylophone quantum", sample_type="high_temp"),
            SSUSample(text="the quick brown fox", sample_type="contrastive"),
        ]
        score = sim.semantic_consistency(samples)
        # With random vectors, consistency should be somewhere in [0, 1]
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# ThreeSampleSSU Tests
# ---------------------------------------------------------------------------


class TestThreeSampleSSU:
    """Tests for the main ThreeSampleSSU engine."""

    def test_run_returns_ssu_result(self) -> None:
        """Test that run() returns an SSUResult."""
        engine = ThreeSampleSSU()
        result = engine.run("What is 2+2?")
        assert isinstance(result, SSUResult)

    def test_run_result_has_all_fields(self) -> None:
        """Test that SSUResult has all required fields."""
        engine = ThreeSampleSSU()
        result = engine.run("What is the capital of France?")
        assert hasattr(result, "samples")
        assert hasattr(result, "consistency_score")
        assert hasattr(result, "verbalized_confidence")
        assert hasattr(result, "calibrated_probability")
        assert hasattr(result, "model_name_used")
        assert hasattr(result, "embedding_model_name")

    def test_run_three_samples_present(self) -> None:
        """Test that run() produces exactly 3 samples."""
        engine = ThreeSampleSSU()
        result = engine.run("Test prompt")
        assert len(result.samples) == 3
        types = {s.sample_type for s in result.samples}
        assert types == {"standard", "high_temp", "contrastive"}

    def test_consistency_score_computed(self) -> None:
        """Test that consistency_score is computed and in [0, 1]."""
        engine = ThreeSampleSSU()
        result = engine.run("Test prompt")
        assert 0.0 <= result.consistency_score <= 1.0

    def test_calibrated_probability_in_range(self) -> None:
        """Test that calibrated_probability is in [0, 1]."""
        engine = ThreeSampleSSU()
        result = engine.run("Test prompt")
        assert 0.0 <= result.calibrated_probability <= 1.0

    def test_verbalized_confidence_in_range_if_present(self) -> None:
        """Test verbalized_confidence is in [0, 1] if not None."""
        engine = ThreeSampleSSU()
        result = engine.run("Test prompt")
        if result.verbalized_confidence is not None:
            assert 0.0 <= result.verbalized_confidence <= 1.0

    def test_run_with_custom_config(self) -> None:
        """Test that run() respects custom SSUConfig."""
        config = SSUConfig(
            alpha_consistency=0.8,
            alpha_verbalized=0.2,
            model_name="custom-embedding-model",
        )
        engine = ThreeSampleSSU(config)
        result = engine.run("Test")
        assert result.embedding_model_name == "custom-embedding-model"

    def test_fuse_formula(self) -> None:
        """Test that the fusion formula matches expected output."""
        config = SSUConfig(alpha_consistency=0.6, alpha_verbalized=0.4)
        engine = ThreeSampleSSU(config)
        # With no verbalized confidence (stub samples have no confidence patterns),
        # fusion should be: 0.6 * consistency + 0.4 * 0.5
        result = engine.run("Test prompt")
        expected_floor = 0.6 * result.consistency_score + 0.4 * 0.5
        assert result.calibrated_probability == pytest.approx(expected_floor)

    def test_samples_have_correct_types(self) -> None:
        """Test that each sample has the correct sample_type."""
        engine = ThreeSampleSSU()
        result = engine.run("Test")
        type_map = {s.sample_type: s for s in result.samples}
        assert "standard" in type_map
        assert "high_temp" in type_map
        assert "contrastive" in type_map

    def test_samples_have_text(self) -> None:
        """Test that each sample has non-empty text."""
        engine = ThreeSampleSSU()
        result = engine.run("What is 2+2?")
        for sample in result.samples:
            assert isinstance(sample.text, str)
            assert len(sample.text) > 0


# ---------------------------------------------------------------------------
# SSUResult Tests
# ---------------------------------------------------------------------------


class TestSSUResult:
    """Tests for the SSUResult dataclass."""

    def test_ssu_result_instantiation(self) -> None:
        """Test SSUResult can be instantiated."""
        samples = [
            SSUSample(text="Test", sample_type="standard"),
        ]
        result = SSUResult(
            samples=samples,
            consistency_score=0.85,
            verbalized_confidence=0.9,
            calibrated_probability=0.87,
            model_name_used="test-model",
            embedding_model_name="test-embedder",
        )
        assert len(result.samples) == 1
        assert result.consistency_score == 0.85
        assert result.verbalized_confidence == 0.9
        assert result.calibrated_probability == 0.87
        assert result.model_name_used == "test-model"
        assert result.embedding_model_name == "test-embedder"

    def test_ssu_result_verbalized_none_allowed(self) -> None:
        """Test SSUResult allows verbalized_confidence to be None."""
        result = SSUResult(
            samples=[],
            consistency_score=0.5,
            verbalized_confidence=None,
            calibrated_probability=0.5,
            model_name_used="test",
            embedding_model_name="test",
        )
        assert result.verbalized_confidence is None
