"""Configuration management for ACTR.

Provides the ACTRConfig class that holds thresholds, model settings,
calibration parameters, and other runtime configuration. Supports
loading from environment variables and from a TOML/JSON config file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

__all__ = ["ACTRConfig", "ConfidenceThresholds", "ModelSettings", "CalibrationSettings"]


@dataclass
class ConfidenceThresholds:
    """Boundary values that map a continuous confidence score to a ConfidenceTag.

    The thresholds define the cutoffs for ``low``, ``medium``, and ``high``
    confidence categories. Values must satisfy::

        0.0 <= low < medium < high <= 1.0

    Parameters
    ----------
    low : float
        Upper bound for the ``low`` confidence tag.
    medium : float
        Lower bound for ``medium``; upper bound for ``high``.
    high : float
        Lower bound for the ``high`` confidence tag.
    unknown_epsilon : float
        Confidence differences smaller than this are treated as
        ``unknown`` rather than assigned a definitive tag.
    """

    low: float = 0.4
    medium: float = 0.65
    high: float = 0.8
    unknown_epsilon: float = 0.05

    def tag_for_confidence(self, confidence: float) -> str:
        """Return the ConfidenceTag for a raw confidence score.

        Parameters
        ----------
        confidence : float
            A value in [0.0, 1.0].

        Returns
        -------
        str
            One of ``"high"``, ``"medium"``, ``"low"``, ``"unknown"``.
        """
        if not 0.0 <= confidence <= 1.0:
            return "unknown"
        if confidence < self.low - self.unknown_epsilon:
            return "low"
        if confidence >= self.high:
            return "high"
        if self.high - confidence < self.unknown_epsilon:
            return "high"
        if confidence < self.low + self.unknown_epsilon:
            return "low"
        return "medium"

    def to_dict(self) -> dict[str, float]:
        return {
            "low": self.low,
            "medium": self.medium,
            "high": self.high,
            "unknown_epsilon": self.unknown_epsilon,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfidenceThresholds:
        return cls(
            low=data.get("low", 0.4),
            medium=data.get("medium", 0.65),
            high=data.get("high", 0.8),
            unknown_epsilon=data.get("unknown_epsilon", 0.05),
        )


@dataclass
class ModelSettings:
    """Settings that control the behaviour of the underlying language model.

    Parameters
    ----------
    model_name : str
        Identifier for the model (e.g. ``"gpt-4o"``, ``"claude-3-opus"``).
    temperature : float
        Sampling temperature passed to the model API.
    max_tokens : int
        Maximum number of tokens to generate.
    top_p : float
        Nucleus sampling top-p value.
    api_base : str | None
        Optional custom base URL for the model API (useful for proxies or
        OpenAI-compatible endpoints).
    api_key : str | None
        API key for the model provider. If None, read from the
        ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` environment variables.
    timeout_seconds : float
        Request timeout in seconds.
    """

    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: float = 60.0

    def resolved_api_key(self) -> Optional[str]:
        """Return the API key, falling back to environment variables."""
        if self.api_key:
            return self.api_key
        return os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    def to_dict(self) -> dict[str, Any]:
        # Omit the actual key from serialised output for security
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "api_base": self.api_base,
            "api_key": "***" if self.api_key else None,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelSettings:
        return cls(
            model_name=data.get("model_name", "gpt-4o"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 4096),
            top_p=data.get("top_p", 1.0),
            api_base=data.get("api_base"),
            api_key=data.get("api_key"),
            timeout_seconds=data.get("timeout_seconds", 60.0),
        )


@dataclass
class CalibrationSettings:
    """Parameters that govern the ACTR calibration algorithm itself.

    Parameters
    ----------
    learning_rate : float
        How aggressively calibration adjustments move the confidence
        score per step. Must be in (0.0, 1.0].
    window_size : int
        Number of most-recent reasoning sessions to consider when
        updating the calibration model.
    min_verification_score : float
        Minimum consistency score required from self-verification to
        accept a reasoning path. Must be in [0.0, 1.0].
    enable_recursive_calibration : bool
        Whether to apply recursive (iterative) calibration when the
        confidence tag is ``"unknown"`` or ``"low"``.
    max_recursion_depth : int
        Maximum number of recursive calibration iterations.
    calibration_model_type : str
        The type of calibration model to use: ``"linear"``, ``"bayesian"``,
        or ``"histogram"``.
    """

    learning_rate: float = 0.3
    window_size: int = 100
    min_verification_score: float = 0.7
    enable_recursive_calibration: bool = True
    max_recursion_depth: int = 3
    calibration_model_type: str = "linear"

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "window_size": self.window_size,
            "min_verification_score": self.min_verification_score,
            "enable_recursive_calibration": self.enable_recursive_calibration,
            "max_recursion_depth": self.max_recursion_depth,
            "calibration_model_type": self.calibration_model_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationSettings:
        return cls(
            learning_rate=data.get("learning_rate", 0.3),
            window_size=data.get("window_size", 100),
            min_verification_score=data.get("min_verification_score", 0.7),
            enable_recursive_calibration=data.get("enable_recursive_calibration", True),
            max_recursion_depth=data.get("max_recursion_depth", 3),
            calibration_model_type=data.get("calibration_model_type", "linear"),
        )


@dataclass
class ACTRConfig:
    """Top-level configuration container for the ACTR framework.

    Aggregates confidence thresholds, model settings, and calibration
    parameters. Can be instantiated directly, loaded from a dictionary
    (e.g. from a config file), or loaded from a TOML file on disk.

    Parameters
    ----------
    thresholds : ConfidenceThresholds
        Mapping from continuous confidence scores to discrete tags.
    model : ModelSettings
        Language model connection and generation settings.
    calibration : CalibrationSettings
        Calibration algorithm hyperparameters.
    default_reasoning_mode : ReasoningMode
        The reasoning mode to use when no explicit mode is selected.
    enable_benchmarking : bool
        Whether to collect benchmarking metrics during reasoning.
    log_calibration_history : bool
        Whether to persist the full calibration history per session.
    config_file_path : Path | None
        Source file from which this config was loaded, if any.

    Examples
    --------
    >>> config = ACTRConfig()
    >>> config.thresholds.tag_for_confidence(0.85)
    'high'
    >>> config.model.model_name = "claude-3-opus"
    >>> config.calibration.learning_rate = 0.2
    """

    thresholds: ConfidenceThresholds = field(default_factory=ConfidenceThresholds)
    model: ModelSettings = field(default_factory=ModelSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    default_reasoning_mode: str = "chain_of_thought"
    enable_benchmarking: bool = False
    log_calibration_history: bool = True
    config_file_path: Optional[Path] = field(default=None, repr=False)

    def tag_for_confidence(self, confidence: float) -> str:
        """Return the discrete confidence tag for a given confidence score."""
        return self.thresholds.tag_for_confidence(confidence)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ACTRConfig:
        """Reconstruct an ACTRConfig from a dictionary."""
        thresholds = ConfidenceThresholds.from_dict(data.get("thresholds", {}))
        model = ModelSettings.from_dict(data.get("model", {}))
        calibration = CalibrationSettings.from_dict(data.get("calibration", {}))
        return cls(
            thresholds=thresholds,
            model=model,
            calibration=calibration,
            default_reasoning_mode=data.get("default_reasoning_mode", "chain_of_thought"),
            enable_benchmarking=data.get("enable_benchmarking", False),
            log_calibration_history=data.get("log_calibration_history", True),
        )

    @classmethod
    def from_toml_file(cls, path: str | Path) -> ACTRConfig:
        """Load configuration from a TOML file.

        Parameters
        ----------
        path : str | Path
            Path to a TOML configuration file.

        Returns
        -------
        ACTRConfig
        """
        import tomllib

        path = Path(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)
        config = cls.from_dict(data)
        config.config_file_path = path
        return config

    @classmethod
    def from_env(cls) -> ACTRConfig:
        """Load configuration from environment variables.

        Supported environment variables:

        - ``ACTR_MODEL_NAME`` — model identifier
        - ``ACTR_TEMPERATURE`` — float
        - ``ACTR_MAX_TOKENS`` — int
        - ``ACTR_HIGH_THRESHOLD`` — float
        - ``ACTR_MEDIUM_THRESHOLD`` — float
        - ``ACTR_LOW_THRESHOLD`` — float
        - ``ACTR_LEARNING_RATE`` — float
        - ``ACTR_API_BASE`` — string
        - ``ACTR_API_KEY`` — string

        Returns
        -------
        ACTRConfig
        """
        import shutil

        thresholds = ConfidenceThresholds(
            low=float(os.environ.get("ACTR_LOW_THRESHOLD", "0.4")),
            medium=float(os.environ.get("ACTR_MEDIUM_THRESHOLD", "0.65")),
            high=float(os.environ.get("ACTR_HIGH_THRESHOLD", "0.8")),
        )
        model = ModelSettings(
            model_name=os.environ.get("ACTR_MODEL_NAME", "gpt-4o"),
            temperature=float(os.environ.get("ACTR_TEMPERATURE", "0.7")),
            max_tokens=int(os.environ.get("ACTR_MAX_TOKENS", "4096")),
            api_base=os.environ.get("ACTR_API_BASE"),
            api_key=os.environ.get("ACTR_API_KEY"),
        )
        calibration = CalibrationSettings(
            learning_rate=float(os.environ.get("ACTR_LEARNING_RATE", "0.3")),
        )
        return cls(
            thresholds=thresholds,
            model=model,
            calibration=calibration,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration to a dictionary (API key redacted)."""
        return {
            "thresholds": self.thresholds.to_dict(),
            "model": self.model.to_dict(),
            "calibration": self.calibration.to_dict(),
            "default_reasoning_mode": self.default_reasoning_mode,
            "enable_benchmarking": self.enable_benchmarking,
            "log_calibration_history": self.log_calibration_history,
        }

    def save_toml(self, path: str | Path) -> None:
        """Write the current configuration to a TOML file."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        import datetime

        path = Path(path)
        # Basic TOML generation using a simple template
        d = self.to_dict()
        lines = [
            "# ACTR Configuration File",
            f"# Generated: {datetime.datetime.now().isoformat()}",
            "",
            "[thresholds]",
            f"low = {d['thresholds']['low']}",
            f"medium = {d['thresholds']['medium']}",
            f"high = {d['thresholds']['high']}",
            f"unknown_epsilon = {d['thresholds']['unknown_epsilon']}",
            "",
            "[model]",
            f'model_name = "{d["model"]["model_name"]}"',
            f'temperature = {d["model"]["temperature"]}',
            f'max_tokens = {d["model"]["max_tokens"]}',
            f'top_p = {d["model"]["top_p"]}',
        ]
        if d["model"]["api_base"]:
            lines.append(f'api_base = "{d["model"]["api_base"]}"')
        lines += [
            "",
            "[calibration]",
            f"learning_rate = {d['calibration']['learning_rate']}",
            f"window_size = {d['calibration']['window_size']}",
            f"min_verification_score = {d['calibration']['min_verification_score']}",
            f"enable_recursive_calibration = {str(d['calibration']['enable_recursive_calibration']).lower()}",
            f"max_recursion_depth = {d['calibration']['max_recursion_depth']}",
            f'calibration_model_type = "{d["calibration"]["calibration_model_type"]}"',
            "",
            f"default_reasoning_mode = \"{d['default_reasoning_mode']}\"",
            f"enable_benchmarking = {str(d['enable_benchmarking']).lower()}",
            f"log_calibration_history = {str(d['log_calibration_history']).lower()}",
        ]
        path.write_text("\n".join(lines))
