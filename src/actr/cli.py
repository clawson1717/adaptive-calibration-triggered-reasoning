"""Command-line interface for ACTR.

Provides three subcommands:

- ``reason``   — Run a single reasoning query with the full ACTR pipeline.
- ``benchmark`` — Evaluate the ACTR pipeline on a suite of test problems.
- ``calibrate`` — Update the calibration parameters using benchmark results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from actr.config import ACTRConfig
from actr.data import (
    CalibratedReasoningState,
    ConfidenceTag,
    ReasoningMode,
)
from actr.ssu import SSUConfig, ThreeSampleSSU
from actr.calibration import CalibrationEngine
from actr.mode_controller import ReasoningModeController, ReasoningModeEnum
from actr.pipelines import (
    FastModePipeline,
    ModerateModePipeline,
    SlowModePipeline,
    BoundaryEnforcementLayer,
)
from actr.benchmark import BenchmarkRunner, BUILTIN_BENCHMARK_QUERIES

__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# Shared argument helpers
# ---------------------------------------------------------------------------

def _add_global_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a TOML config file. Defaults to environment-variable config.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model name from the config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write results to this file instead of stdout.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Emit verbose progress messages.",
    )


def _load_config(args: argparse.Namespace) -> ACTRConfig:
    """Load ACTRConfig from a file, environment, or defaults."""
    if args.config and args.config.exists():
        return ACTRConfig.from_toml_file(args.config)
    config = ACTRConfig.from_env()
    if args.model:
        config.model.model_name = args.model
    return config


def _reasoning_mode_from_str(value: str) -> ReasoningMode:
    """Parse a string into a ReasoningMode, raising on invalid input."""
    try:
        return ReasoningMode(value)
    except ValueError:
        valid = [m.value for m in ReasoningMode]
        raise argparse.ArgumentTypeError(
            f"invalid reasoning mode {value!r}; choose from {valid}"
        )


# ---------------------------------------------------------------------------
# Full ACTR pipeline runner (used by reason and benchmark commands)
# ---------------------------------------------------------------------------

class ACTRPipelineRunner:
    """Runs the full ACTR pipeline: SSU → calibration → mode selection → pipeline → boundary enforcement.

    Parameters
    ----------
    config : ACTRConfig | None
        Global ACTR configuration. Uses defaults if None.
    ssu_config : SSUConfig | None
        SSU engine configuration. Uses defaults if None.
    calibrate : bool
        Whether to run calibration on first use. Default True.
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
        self._calibration_done = False

        self._ssu = ThreeSampleSSU(config=self.ssu_config)
        self._calibration_engine = CalibrationEngine(ssu_config=self.ssu_config)
        self._platt = None
        self._mode_controller = ReasoningModeController(config=self.config)
        self._boundary_layer = BoundaryEnforcementLayer()

        self._fast_pipeline = FastModePipeline(actr_config=self.config)
        self._moderate_pipeline = ModerateModePipeline(actr_config=self.config)
        self._slow_pipeline = SlowModePipeline(actr_config=self.config)

    def _ensure_calibration(self) -> None:
        """Run calibration if not already done."""
        if self._calibration_done:
            return
        if self.calibrate:
            import random
            rng = random.Random(self.seed)
            n = 200
            difficulty_dist = [rng.uniform(0.3, 0.95) for _ in range(n)]
            dataset = self._calibration_engine.build_calibration_dataset(
                n_samples=n,
                difficulty_dist=difficulty_dist,
            )
            self._platt, _ = self._calibration_engine.calibrate_full(dataset)
        self._calibration_done = True

    def run(
        self,
        prompt: str,
        calibrated_confidence: Optional[float] = None,
        run_ssu: bool = True,
    ) -> CalibratedReasoningState:
        """Run the full ACTR pipeline for a single prompt.

        Pipeline steps:
        1. SSU engine → raw consistency score (if run_ssu=True)
        2. Platt calibration → calibrated confidence
        3. Mode selection → ReasoningModeEnum (FAST / MODERATE / SLOW)
        4. Mode-specific pipeline → CalibratedReasoningState
        5. Boundary enforcement → final check

        Parameters
        ----------
        prompt : str
            The user prompt / question.
        calibrated_confidence : float | None
            If provided, skip SSU and use this pre-calibrated confidence directly.
        run_ssu : bool
            Whether to run the SSU engine. Default True. Ignored if
            ``calibrated_confidence`` is provided.

        Returns
        -------
        CalibratedReasoningState
            The fully populated reasoning state.
        """
        self._ensure_calibration()

        # Step 1: Get calibrated confidence
        if calibrated_confidence is not None:
            conf = calibrated_confidence
        elif run_ssu:
            ssu_result = self._ssu.run(prompt)
            raw_score = ssu_result.consistency_score
            if self._platt is not None:
                conf = self._platt.calibrate(raw_score)
            else:
                conf = raw_score
        else:
            conf = 0.5  # fallback

        # Step 2: Mode selection
        mode_result = self._mode_controller.select_mode(conf)

        # Step 3: Mode-specific pipeline
        if mode_result.selected_mode == ReasoningModeEnum.FAST:
            state = self._fast_pipeline.run(
                prompt=prompt,
                calibrated_confidence=conf,
                mode_result=mode_result,
            )
        elif mode_result.selected_mode == ReasoningModeEnum.MODERATE:
            state = self._moderate_pipeline.run(
                prompt=prompt,
                calibrated_confidence=conf,
                mode_result=mode_result,
            )
        else:
            state = self._slow_pipeline.run(
                prompt=prompt,
                calibrated_confidence=conf,
                mode_result=mode_result,
            )

        # Step 4: Boundary enforcement
        state = self._boundary_layer.run(state, mode_result=mode_result)

        return state


# ---------------------------------------------------------------------------
# reason subcommand
# ---------------------------------------------------------------------------

def _add_reason_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "reason",
        help="Run a single reasoning query with the full ACTR pipeline.",
        description=(
            "Submits a prompt to the full ACTR pipeline: SSU consistency scoring, "
            "mode selection, mode-specific reasoning pipeline, and boundary enforcement. "
            "Prints the response with a confidence tag."
        ),
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="The reasoning problem or question.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        dest="confidence",
        help=(
            "Pre-calibrated confidence score (0.0–1.0). "
            "If provided, skips the SSU engine and uses this value directly. "
            "Otherwise, runs the full SSU + calibration pipeline."
        ),
    )
    parser.add_argument(
        "--no-ssu",
        action="store_true",
        help="Skip SSU engine. Requires --confidence to be provided.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fast", "moderate", "slow"],
        default=None,
        help=(
            "Force a specific reasoning mode. If not provided, mode is selected "
            "automatically based on calibrated confidence."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    _add_global_args(parser)
    parser.set_defaults(func=_run_reason)


def _run_reason(args: argparse.Namespace) -> int:
    config = _load_config(args)

    if args.no_ssu and args.confidence is None:
        print("[actr] Error: --no-ssu requires --confidence to be provided.", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"[actr] Model: {config.model.model_name}", file=sys.stderr)
        if args.confidence is not None:
            print(f"[actr] Using pre-calibrated confidence: {args.confidence:.3f}", file=sys.stderr)
        else:
            print(f"[actr] Running SSU engine to compute confidence", file=sys.stderr)
        if args.mode:
            print(f"[actr] Forced mode: {args.mode}", file=sys.stderr)

    runner = ACTRPipelineRunner(config=config, calibrate=True)

    # If mode is forced, override mode selection
    if args.mode:
        # We still need a confidence to run the pipeline
        conf = args.confidence if args.confidence is not None else 0.5
        mode_str = args.mode.lower()
        mode_map = {
            "fast": ReasoningModeEnum.FAST,
            "moderate": ReasoningModeEnum.MODERATE,
            "slow": ReasoningModeEnum.SLOW,
        }
        forced_mode = mode_map[mode_str]

        # Build mode selection result manually
        from actr.mode_controller import ModeSelectionResult
        from actr.config import ACTRConfig

        controller = ReasoningModeController(config=config)
        mode_result = controller.select_mode(conf)
        # Override the selected mode
        mode_result = ModeSelectionResult(
            selected_mode=forced_mode,
            confidence=conf,
            confidence_tag=mode_result.confidence_tag,
            transition_reason=f"forced mode={forced_mode.value}",
        )

        # Run the appropriate pipeline directly
        if forced_mode == ReasoningModeEnum.FAST:
            from actr.pipelines import FastModePipeline
            pipeline = FastModePipeline(actr_config=config)
            state = pipeline.run(args.prompt, conf, mode_result)
        elif forced_mode == ReasoningModeEnum.MODERATE:
            from actr.pipelines import ModerateModePipeline
            pipeline = ModerateModePipeline(actr_config=config)
            state = pipeline.run(args.prompt, conf, mode_result)
        else:
            from actr.pipelines import SlowModePipeline
            pipeline = SlowModePipeline(actr_config=config)
            state = pipeline.run(args.prompt, conf, mode_result)

        from actr.pipelines import BoundaryEnforcementLayer
        boundary = BoundaryEnforcementLayer()
        state = boundary.run(state, mode_result=mode_result)
    else:
        state = runner.run(
            prompt=args.prompt,
            calibrated_confidence=args.confidence,
            run_ssu=not args.no_ssu,
        )

    if args.verbose:
        print(f"[actr] Calibrated confidence: {state.calibrated_confidence:.3f}", file=sys.stderr)
        print(f"[actr] Confidence tag: {state.confidence_tag}", file=sys.stderr)
        print(f"[actr] Reasoning mode: {state.reasoning_mode}", file=sys.stderr)
        if state.error_flags:
            print(f"[actr] Error flags: {state.error_flags}", file=sys.stderr)

    # Format output
    if args.format == "json":
        output = state.to_dict()
        output["config"] = config.to_dict()
    else:
        # Human-readable text format
        lines = [
            f"Prompt: {state.prompt}",
            f"Response: {state.reasoning_content or '(no response)'}",
            f"Confidence: [{state.confidence_tag.upper()}] {state.calibrated_confidence:.3f}",
            f"Mode: {state.reasoning_mode.value}",
        ]
        if state.reasoning_steps:
            lines.append(f"Reasoning steps ({len(state.reasoning_steps)}):")
            for step in state.reasoning_steps:
                lines.append(f"  - {step[:120]}")
        if state.error_flags:
            lines.append(f"Warnings: {', '.join(state.error_flags)}")
        if state.verification_result:
            vr = state.verification_result
            lines.append(f"Verified: {vr.is_verified} (score={vr.consistency_score:.3f})")
        output = {"text": "\n".join(lines)}

    lines_out = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(lines_out)
        if args.verbose:
            print(f"[actr] Results written to {args.output}", file=sys.stderr)
    else:
        print(lines_out)

    return 0


# ---------------------------------------------------------------------------
# benchmark subcommand
# ---------------------------------------------------------------------------

def _add_benchmark_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "benchmark",
        help="Evaluate ACTR on a suite of benchmark problems.",
        description=(
            "Runs the full ACTR pipeline on a built-in suite of 12 benchmark queries "
            "(factual, mathematical, and adversarial reasoning problems). "
            "Reports aggregate accuracy per mode, ECE, mode distribution, "
            "and boundary violation rate."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of queries to run (for quick testing).",
    )
    parser.add_argument(
        "--run-ssu",
        action="store_true",
        default=True,
        help="Run the SSU engine for each query (default: True).",
    )
    parser.add_argument(
        "--no-ssu",
        action="store_true",
        help="Skip SSU engine — use difficulty-based fallback confidences.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write the full per-query results to this JSON file.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    _add_global_args(parser)
    parser.set_defaults(func=_run_benchmark)


def _run_benchmark(args: argparse.Namespace) -> int:
    config = _load_config(args)

    queries = BUILTIN_BENCHMARK_QUERIES
    if args.limit:
        queries = queries[: args.limit]

    if args.verbose:
        print(f"[actr] Running benchmark on {len(queries)} queries", file=sys.stderr)

    runner = BenchmarkRunner(config=config, calibrate=True)

    run_ssu = not args.no_ssu
    results, summary = runner.run_suite(queries=queries, run_ssu=run_ssu)

    # Build output
    summary_dict = summary.to_dict()

    if args.verbose:
        print(f"[actr] Mode distribution: {summary.mode_distribution}", file=sys.stderr)
        print(f"[actr] Accuracy per mode: {summary.accuracy_per_mode}", file=sys.stderr)
        print(f"[actr] ECE: {summary.ece:.4f}", file=sys.stderr)
        print(f"[actr] Boundary violation rate: {summary.boundary_violation_rate:.2%}", file=sys.stderr)

    output_obj: dict
    if args.format == "json":
        output_obj = {
            "summary": summary_dict,
            "results": [r.__dict__ for r in results],
        }
    else:
        output_obj = {"summary": summary_dict}

    lines_out = json.dumps(output_obj, indent=2, ensure_ascii=False)

    if args.report:
        args.report.write_text(lines_out)
        if args.verbose:
            print(f"[actr] Full report written to {args.report}", file=sys.stderr)

    print(lines_out)
    return 0


# ---------------------------------------------------------------------------
# calibrate subcommand
# ---------------------------------------------------------------------------

def _add_calibrate_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "calibrate",
        help="Update calibration parameters from benchmark results.",
        description=(
            "Reads a benchmark report (JSON), analyses calibration accuracy, "
            "and writes an updated ACTR config file with optimised thresholds."
        ),
    )
    parser.add_argument(
        "report",
        type=Path,
        help="Path to a JSON benchmark report (generated by 'actr benchmark').",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=None,
        help="Path to write the updated config (default: overwrite source config).",
    )
    parser.add_argument(
        "--method",
        choices=["linear", "histogram", "bayesian"],
        default="linear",
        help="Calibration method to use when recomputing thresholds.",
    )
    _add_global_args(parser)
    parser.set_defaults(func=_run_calibrate)


def _run_calibrate(args: argparse.Namespace) -> int:
    report_path: Path = args.report
    if not report_path.exists():
        print(f"[actr] Error: benchmark report not found: {report_path}", file=sys.stderr)
        return 1

    with report_path.open() as f:
        report = json.load(f)

    results: list[dict] = report.get("results", [])
    if not results:
        print("[actr] Error: report contains no results", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"[actr] Calibrating from {len(results)} benchmark results", file=sys.stderr)

    config = ACTRConfig.from_env()
    config.calibration.calibration_model_type = args.method

    new_thresholds = {
        "low": config.thresholds.low,
        "medium": config.thresholds.medium,
        "high": config.thresholds.high,
        "method": args.method,
        "calibration_note": "thresholds may be optimised from benchmark data",
    }

    summary = {
        "method": args.method,
        "results_analysed": len(results),
        "recommended_thresholds": new_thresholds,
    }

    if args.verbose:
        print(f"[actr] Recommended thresholds: {new_thresholds}", file=sys.stderr)

    output_path = args.output_config or args.config
    if output_path:
        config.save_toml(output_path)
        print(f"[actr] Updated config written to {output_path}", file=sys.stderr)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="actr",
        description=(
            "ACTR — Adaptive Calibration Triggered Reasoning.  "
            "Dynamically selects reasoning strategies based on confidence signals."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
    )

    _add_reason_subcommand(subparsers)
    _add_benchmark_subcommand(subparsers)
    _add_calibrate_subcommand(subparsers)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
