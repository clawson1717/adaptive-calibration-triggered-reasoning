"""Stub model interfaces for SSU testing.

These allow the SSU engine to run without real API calls during testing.
Real implementations would swap these for actual API clients.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

__all__ = ["StubSampler", "StubEmbeddingModel"]


@dataclass
class StubSampler:
    """Returns a deterministic-but-varied stub response for testing.

    The sample type determines which template is used.
    """

    sample_type: str = "standard"

    def __call__(self, prompt: str, temperature: float = 0.7) -> dict:
        """Return a stub response dict.

        Args:
            prompt: The input prompt.
            temperature: The sampling temperature (used for future real implementations).

        Returns:
            A dict with keys "text" and "avg_logprob".
        """
        templates = {
            "standard": (
                "[Standard reasoning about {p}: First, we observe that... "
                "Therefore the answer is X. Confidence: 87%]"
            ),
            "high_temp": (
                "[High-temp reasoning about {p}: Looking at this differently... "
                "The answer seems to be X. I'm about 80% certain.]"
            ),
            "contrastive": (
                "[Alternatively considering {p}: Another way to think about this... "
                "The answer might be X. 0.85 certainty.]"
            ),
        }
        text = templates.get(self.sample_type, templates["standard"]).format(
            p=prompt[:40]
        )
        return {
            "text": text,
            "avg_logprob": random.uniform(-2.0, -0.5),
        }


@dataclass
class StubEmbeddingModel:
    """Returns random unit vectors for embedding-based similarity testing.

    This avoids downloading real transformer models during tests while still
    providing meaningful vector operations.
    """

    embedding_dim: int = 384  # Matches all-MiniLM-L6-v2

    def __call__(self, text: str) -> list[float]:
        """Return a random unit vector of shape (embedding_dim,).

        Args:
            text: Input text (used as a seed for reproducibility in tests).

        Returns:
            A list of floats representing a unit vector.
        """
        # Use hash of text to seed for deterministic behavior per text
        seed = hash(text) % (2**32)
        rng = random.Random(seed)
        vec = [rng.random() for _ in range(self.embedding_dim)]
        # Normalize to unit vector
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec]
