"""Three-Sample Semantic Uncertainty (3-SSU) Engine for ACTR.

Generates three reasoning samples per query using different decoding strategies,
computes semantic consistency via embedding cosine similarity, extracts verbalized
confidence via regex, and fuses everything into a calibrated probability score.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

__all__ = [
    "ThreeSampleSSU",
    "SSUSample",
    "SSUConfig",
    "SSUResult",
    "EmbeddingSimilarity",
    "VerbalizedConfidenceExtractor",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SSUSample:
    """A single reasoning sample from one decoding strategy."""

    text: str
    sample_type: Literal["standard", "high_temp", "contrastive"]
    logprob: Optional[float] = None
    verbalized_confidence: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert sample to a dictionary."""
        return {
            "text": self.text,
            "sample_type": self.sample_type,
            "logprob": self.logprob,
            "verbalized_confidence": self.verbalized_confidence,
        }


@dataclass
class SSUConfig:
    """Configuration for the Three-Sample Semantic Uncertainty engine."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    alpha_consistency: float = 0.6
    alpha_verbalized: float = 0.4
    high_temp_delta: float = 0.5
    contrastive_prefix: str = "Alternatively: "
    n_embedding_samples: int = 3
    device: str = "cpu"


@dataclass
class SSUResult:
    """Result of a full 3-SSU run."""

    samples: list[SSUSample]
    consistency_score: float
    verbalized_confidence: Optional[float]
    calibrated_probability: float
    model_name_used: str
    embedding_model_name: str


# ---------------------------------------------------------------------------
# Verbalized Confidence Extractor
# ---------------------------------------------------------------------------


class VerbalizedConfidenceExtractor:
    """Extracts confidence values from text using regex patterns.

    Patterns are tried in order; the first match wins. Returns a float in [0, 1]
    or None if no confidence can be extracted.
    """

    PATTERNS: list[tuple[str, bool]] = [
        # (pattern, is_multi_value) — multi-value means "between X and Y"
        (r"confidence[:\s]+([0-9.]+)", False),
        (r"([0-9.]+)%?\s*certainty", False),
        (r"I'm\s+([0-9.]+)%?\s*(?:certain|confident|sure)", False),
        (r"([0-9.]+)\s+out\s+of\s+10\b", False),  # "7 out of 10" — X/10 scale
        (r"likelihood[:\s]+([0-9.]+)", False),
        (r"probability[:\s]+([0-9.]+)", False),
        (r"betwee?n?\s+([0-9.]+)\s*(?:%|percent)?\s*and\s+([0-9.]+)", True),
    ]

    def extract(self, text: str) -> Optional[float]:
        """Extract a confidence value from text.

        Args:
            text: The text to search for confidence expressions.

        Returns:
            A confidence value in [0, 1], or None if no confidence was found.
        """
        for pattern, is_multi in self.PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue

            if is_multi:
                # Range pattern: return the midpoint
                low_str = match.group(1)
                high_str = match.group(2)
                low_val = float(low_str)
                high_val = float(high_str)
                # Normalize to [0, 1] if values look like percentages
                if low_val > 1.0:
                    low_val /= 100.0
                if high_val > 1.0:
                    high_val /= 100.0
                midpoint = (low_val + high_val) / 2.0
                return max(0.0, min(1.0, midpoint))
            else:
                value = float(match.group(1))
                matched_text = match.group(0)
                # "X out of 10" → treat as X/10 scale
                if re.search(r"out\s+of\s+10", matched_text):
                    value = value / 10.0
                # Normalize percentage values (e.g., 87 → 0.87), but only if > 1
                elif value > 1.0:
                    value /= 100.0
                return max(0.0, min(1.0, value))

        return None


# ---------------------------------------------------------------------------
# Embedding Similarity
# ---------------------------------------------------------------------------


class EmbeddingSimilarity:
    """Computes semantic consistency between samples via embedding cosine similarity.

    When mock=True, uses deterministic random unit vectors so tests run without
    network access. Identical texts always produce identical vectors in mock mode.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        mock: bool = True,
    ) -> None:
        """Initialize the embedding similarity engine.

        Args:
            model_name: HuggingFace model name for embeddings.
            device: "cpu" or "cuda".
            mock: If True, use random unit vectors instead of real model embeddings.
        """
        self.model_name = model_name
        self.device = device
        self.mock = mock
        self._model: Optional["PreTrainedModel"] = None
        self._tokenizer: Optional["PreTrainedTokenizer"] = None

    def _load_model(self) -> None:
        """Lazily load the transformer model and tokenizer."""
        if self._model is not None:
            return

        if self.mock:
            # Don't load real model in mock mode
            return

        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get the embedding vector for a text.

        In mock mode, returns a deterministic random unit vector of dimension 384
        (matching all-MiniLM-L6-v2's embedding dimension). Identical texts always
        produce identical vectors.
        """
        if self.mock:
            # Seeded RNG so identical text → identical vector
            seed = abs(hash(text)) % (2**31)
            rng = random.Random(seed)
            vec = np.array([rng.random() for _ in range(384)], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return torch.from_numpy(vec)
        else:
            self._load_model()
            inputs = self._tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            # Mean pool over token embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.squeeze(0)

    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity between two 1D tensors.

        Returns a value in [-1, 1].
        """
        a_flat = a.flatten()
        b_flat = b.flatten()
        dot = torch.dot(a_flat, b_flat)
        norm_a = torch.norm(a_flat)
        norm_b = torch.norm(b_flat)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return (dot / (norm_a * norm_b)).item()

    def semantic_consistency(self, samples: list[SSUSample]) -> float:
        """Compute mean pairwise cosine similarity across all samples.

        The result is shifted from [-1, 1] to [0, 1].

        Args:
            samples: List of SSUSample objects.

        Returns:
            A consistency score in [0, 1], where 1 means all samples are
            semantically identical.
        """
        if len(samples) < 2:
            return 1.0

        embeddings = [self._get_embedding(s.text) for s in samples]

        n = len(embeddings)
        total_sim = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.cosine_similarity(embeddings[i], embeddings[j])
                total_sim += sim
                count += 1

        if count == 0:
            return 1.0

        mean_sim = total_sim / count
        # Shift from [-1, 1] to [0, 1]
        shifted = (mean_sim + 1.0) / 2.0
        return max(0.0, min(1.0, shifted))


# ---------------------------------------------------------------------------
# Three-Sample SSU Engine
# ---------------------------------------------------------------------------


class ThreeSampleSSU:
    """Main 3-SSU engine.

    Generates three reasoning samples (standard, high-temp, contrastive-decoding),
    computes semantic consistency, extracts verbalized confidence, and fuses them
    into a calibrated probability.
    """

    def __init__(self, config: Optional[SSUConfig] = None) -> None:
        """Initialize the SSU engine.

        Args:
            config: Optional configuration object. Uses defaults if not provided.
        """
        self.config = config if config is not None else SSUConfig()
        self._embedding_sim = EmbeddingSimilarity(
            model_name=self.config.model_name,
            device=self.config.device,
            mock=True,  # Always use mock by default to avoid downloads
        )
        self._verbal_extractor = VerbalizedConfidenceExtractor()

    def _sample_standard(self, prompt: str, config: SSUConfig) -> SSUSample:
        """Generate a standard-temperature sample (stub)."""
        text = f"[SAMPLE standard for: {prompt[:50]}]"
        return SSUSample(
            text=text,
            sample_type="standard",
            logprob=None,
            verbalized_confidence=None,
        )

    def _sample_high_temp(self, prompt: str, config: SSUConfig) -> SSUSample:
        """Generate a high-temperature sample (stub)."""
        text = f"[SAMPLE high_temp for: {prompt[:50]}]"
        return SSUSample(
            text=text,
            sample_type="high_temp",
            logprob=None,
            verbalized_confidence=None,
        )

    def _sample_contrastive(self, prompt: str, config: SSUConfig) -> SSUSample:
        """Generate a contrastive-decoding sample (stub)."""
        text = f"[SAMPLE contrastive for: {prompt[:50]}]"
        return SSUSample(
            text=text,
            sample_type="contrastive",
            logprob=None,
            verbalized_confidence=None,
        )

    def _compute_consistency(self, samples: list[SSUSample]) -> float:
        """Compute semantic consistency across all samples."""
        return self._embedding_sim.semantic_consistency(samples)

    def _extract_verbalized(self, samples: list[SSUSample]) -> Optional[float]:
        """Extract verbalized confidence from samples.

        Returns the mean of extracted confidences across samples, or None if
        no confidence could be extracted from any sample.
        """
        confidences: list[float] = []
        for sample in samples:
            conf = self._verbal_extractor.extract(sample.text)
            if conf is not None:
                confidences.append(conf)

        if not confidences:
            return None
        return sum(confidences) / len(confidences)

    def _fuse(
        self, consistency: float, verbalized: Optional[float]
    ) -> float:
        """Fuse consistency and verbalized confidence into a calibrated probability.

        Formula:
            calibrated = alpha_consistency * consistency
                      + alpha_verbalized * (verbalized if not None else 0.5)
        """
        verbalized_part = verbalized if verbalized is not None else 0.5
        return (
            self.config.alpha_consistency * consistency
            + self.config.alpha_verbalized * verbalized_part
        )

    def run(self, prompt: str) -> SSUResult:
        """Run the full 3-SSU pipeline on a prompt.

        Args:
            prompt: The input prompt/question.

        Returns:
            An SSUResult containing all samples, scores, and the fused probability.
        """
        # Generate three samples
        samples = [
            self._sample_standard(prompt, self.config),
            self._sample_high_temp(prompt, self.config),
            self._sample_contrastive(prompt, self.config),
        ]

        # Compute semantic consistency
        consistency = self._compute_consistency(samples)

        # Extract verbalized confidence
        verbalized = self._extract_verbalized(samples)

        # Fuse into calibrated probability
        calibrated = self._fuse(consistency, verbalized)

        return SSUResult(
            samples=samples,
            consistency_score=consistency,
            verbalized_confidence=verbalized,
            calibrated_probability=calibrated,
            model_name_used="stub-model",  # Will be updated when real API is integrated
            embedding_model_name=self.config.model_name,
        )
