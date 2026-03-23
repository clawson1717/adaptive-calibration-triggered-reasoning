"""Command-line interface for ACTR.

Provides three subcommands:

- ``reason``   — Run a single reasoning query with ACTR calibration.
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
# reason subcommand
# ---------------------------------------------------------------------------

def _add_reason_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "reason",
        help="Run a single reasoning query with ACTR calibration.",
        description=(
            "Submits a prompt to the language model using the selected reasoning "
            "mode, applies ACTR confidence calibration, and prints the result."
        ),
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="The reasoning problem or question.",
    )
    parser.add_argument(
        "--mode",
        type=_reasoning_mode_from_str,
        default=None,
        help=f"Reasoning mode to use. Defaults to the config default.",
    )
    parser.add_argument(
        "--raw-confidence",
        type=float,
        default=None,
        help=(
            "Override the raw confidence score (0.0–1.0). "
            "If not provided, ACTR will call the model with a calibration prompt."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the temperature for this run.",
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

    if args.verbose:
        print(f"[actr] Using model: {config.model.model_name}", file=sys.stderr)
        print(f"[actr] Reasoning mode: {args.mode or config.default_reasoning_mode}", file=sys.stderr)

    mode = _reasoning_mode_from_str(args.mode) if args.mode else ReasoningMode(config.default_reasoning_mode)

    # Build the reasoning state — this is the stub that a real implementation would flesh out
    state = CalibratedReasoningState(
        prompt=args.prompt,
        model_name=config.model.model_name,
        reasoning_mode=mode,
    )

    # ------------------------------------------------------------------
    # STUB: Real implementation would call the model here and populate
    # the state. For scaffolding, we return a minimal calibrated state.
    # ------------------------------------------------------------------
    if args.raw_confidence is not None:
        raw_conf = args.raw_confidence
    else:
        # Simulate a raw confidence from a model call
        raw_conf = 0.5

    state.raw_confidence = raw_conf
    state.calibrated_confidence = raw_conf
    state.confidence_tag = config.tag_for_confidence(raw_conf)

    if args.verbose:
        print(f"[actr] Raw confidence: {raw_conf:.3f}", file=sys.stderr)
        print(f"[actr] Calibrated confidence: {state.calibrated_confidence:.3f}", file=sys.stderr)
        print(f"[actr] Confidence tag: {state.confidence_tag}", file=sys.stderr)

    # Structured output
    output: dict
    if args.format == "json":
        output = state.to_dict()
        output["config"] = config.to_dict()
    else:
        output = {
            "prompt": state.prompt,
            "reasoning_content": state.reasoning_content or "(not yet implemented)",
            "raw_confidence": state.raw_confidence,
            "calibrated_confidence": state.calibrated_confidence,
            "confidence_tag": state.confidence_tag,
            "reasoning_mode": str(state.reasoning_mode),
        }

    lines = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(lines)
        if args.verbose:
            print(f"[actr] Results written to {args.output}", file=sys.stderr)
    else:
        print(lines)

    return 0


# ---------------------------------------------------------------------------
# benchmark subcommand
# ---------------------------------------------------------------------------

def _add_benchmark_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "benchmark",
        help="Evaluate ACTR on a suite of benchmark problems.",
        description=(
            "Loads a set of benchmark problems (JSON or JSONL), runs each through "
            "the ACTR pipeline, and reports aggregate calibration metrics."
        ),
    )
    parser.add_argument(
        "suite",
        type=Path,
        help="Path to a JSON or JSONL file containing benchmark problems.",
    )
    parser.add_argument(
        "--mode",
        type=_reasoning_mode_from_str,
        default=None,
        help="Reasoning mode to use for all problems.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of problems to run (for quick testing).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write the full results to this JSON file.",
    )
    _add_global_args(parser)
    parser.set_defaults(func=_run_benchmark)


def _run_benchmark(args: argparse.Namespace) -> int:
    config = _load_config(args)

    suite_path: Path = args.suite
    if not suite_path.exists():
        print(f"[actr] Error: benchmark suite not found: {suite_path}", file=sys.stderr)
        return 1

    # Load the suite
    problems: list[dict]
    if suite_path.suffix == ".jsonl":
        problems = []
        with suite_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    problems.append(json.loads(line))
    elif suite_path.suffix == ".json":
        with suite_path.open() as f:
            data = json.load(f)
        problems = data if isinstance(data, list) else [data]
    else:
        print(f"[actr] Error: unsupported file type {suite_path.suffix}", file=sys.stderr)
        return 1

    if args.limit:
        problems = problems[: args.limit]

    if args.verbose:
        print(f"[actr] Loaded {len(problems)} benchmark problems", file=sys.stderr)

    results: list[dict] = []
    for idx, problem in enumerate(problems, 1):
        prompt = problem.get("prompt", problem.get("question", ""))
        expected_answer = problem.get("expected_answer")

        if args.verbose:
            print(f"[actr] Problem {idx}/{len(problems)}: {prompt[:60]!r}...", file=sys.stderr)

        # STUB: real benchmark runner would call ACTR pipeline here
        state = CalibratedReasoningState(
            prompt=prompt,
            model_name=config.model.model_name,
            reasoning_mode=ReasoningMode(config.default_reasoning_mode),
        )
        state.raw_confidence = 0.5
        state.calibrated_confidence = 0.5
        state.confidence_tag = config.tag_for_confidence(0.5)
        state.reasoning_content = f"[STUB] Result for: {prompt[:80]}"

        result = state.to_dict()
        if expected_answer is not None:
            result["expected_answer"] = expected_answer
            result["correct"] = (
                str(expected_answer).strip().lower()
                == state.reasoning_content.strip().lower()
            )
        results.append(result)

    # Aggregate summary
    total = len(results)
    high_count = sum(1 for r in results if r.get("confidence_tag") == "high")
    medium_count = sum(1 for r in results if r.get("confidence_tag") == "medium")
    low_count = sum(1 for r in results if r.get("confidence_tag") == "low")
    unknown_count = sum(1 for r in results if r.get("confidence_tag") == "unknown")

    summary = {
        "total_problems": total,
        "by_confidence_tag": {
            "high": high_count,
            "medium": medium_count,
            "low": low_count,
            "unknown": unknown_count,
        },
        "average_calibrated_confidence": (
            sum(r.get("calibrated_confidence", 0) for r in results) / total if total else 0
        ),
    }

    if args.verbose:
        print(f"[actr] Summary: {summary}", file=sys.stderr)

    output_obj = {"summary": summary, "results": results}

    report_path = args.report
    if report_path:
        report_path.write_text(json.dumps(output_obj, indent=2, ensure_ascii=False))
        if args.verbose:
            print(f"[actr] Report written to {report_path}", file=sys.stderr)

    print(json.dumps(summary, indent=2))
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

    # ------------------------------------------------------------------
    # STUB: Real calibration would compute optimal thresholds from the
    # correlation between calibrated_confidence and correctness.
    # Here we emit a placeholder updated config for the scaffolding.
    # ------------------------------------------------------------------
    config = ACTRConfig.from_env()
    config.calibration.calibration_model_type = args.method

    # Placeholder recomputation: keep current thresholds but mark the method
    new_thresholds = {
        "low": config.thresholds.low,
        "medium": config.thresholds.medium,
        "high": config.thresholds.high,
        "method": args.method,
        "calibration_note": "STUB — thresholds not yet optimised from benchmark data",
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
