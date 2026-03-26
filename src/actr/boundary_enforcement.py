"""Box Maze Boundary Enforcement Layer for ACTR.

Provides boundary enforcement infrastructure for the adaptive calibration triggered
reasoning framework, including:
- BoundaryEnforcementConfig for configuring thresholds and propagation parameters
- BoundaryViolation for tracking constraint violations
- ReasoningStep / ReasoningContext for reasoning graph representation
- GateResult for gate decision outcomes
- ReasoningEnvelopeState for constraint propagation state
- EnforcedReasoningState wrapping CalibratedReasoningState with boundary metadata
- BoundaryEnforcementLayer for the core enforcement logic
- BoundaryViolationError for signaling safety bound violations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from actr.data import CalibratedReasoningState

__all__ = [
    "BoundaryEnforcementConfig",
    "BoundaryViolation",
    "ViolationSeverity",
    "ReasoningStep",
    "ReasoningContext",
    "GateAction",
    "GateResult",
    "ReasoningEnvelopeState",
    "EnforcedReasoningState",
    "BoundaryEnforcementLayer",
    "BoundaryViolationError",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BoundaryEnforcementConfig:
    """Configuration for the Boundary Enforcement Layer.

    Parameters
    ----------
    memory_grounding_threshold : float
        Confidence threshold below which memory grounding is triggered.
        Only ground if confidence < this value. Default 0.7.
    inference_safety_bound : float
        Minimum confidence required to safely respond. Below this bound
        the system must reject or escalate. Default 0.3.
    envelope_softness : float
        How much constraints propagate forward through the reasoning
        graph. Higher values = more aggressive propagation. Default 0.15.
    max_envelope_depth : int
        Maximum depth for envelope constraint propagation. Default 5.
    """

    memory_grounding_threshold: float = 0.7
    inference_safety_bound: float = 0.3
    envelope_softness: float = 0.15
    max_envelope_depth: int = 5


# ---------------------------------------------------------------------------
# Supporting Types
# ---------------------------------------------------------------------------


class ViolationSeverity(str, Enum):
    """Severity level for boundary violations.

    Attributes
    ----------
    INFO : Informational — no action required, logged for audit.
    WARN : Warning — should be noted but does not block reasoning.
    CRITICAL : Critical — blocks the reasoning step, requires escalation.
    """

    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


@dataclass
class BoundaryViolation:
    """A single boundary violation detected during reasoning.

    Attributes
    ----------
    reason : str
        Human-readable explanation of the violation.
    severity : ViolationSeverity
        How severe this violation is.
    step : str
        Identifier of the reasoning step where the violation occurred.
    """

    reason: str
    severity: ViolationSeverity = ViolationSeverity.INFO
    step: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "severity": self.severity.value,
            "step": self.step,
        }


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain.

    Attributes
    ----------
    step_id : str
        Unique identifier for this step.
    content : str
        The textual content of the reasoning step.
    confidence : float
        Confidence score for this step (0.0 to 1.0).
    metadata : dict[str, Any]
        Additional context for this step.
    """

    step_id: str
    content: str
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningContext:
    """Context for a reasoning gate decision.

    Attributes
    ----------
    prompt : str
        The original user prompt.
    prior_steps : list[ReasoningStep]
        Prior reasoning steps in the current chain.
    constraints : dict[str, float]
        Active constraints on the reasoning (e.g. {"accuracy": 0.9}).
    """

    prompt: str = ""
    prior_steps: list[ReasoningStep] = field(default_factory=list)
    constraints: dict[str, float] = field(default_factory=dict)


class GateAction(str, Enum):
    """Possible actions from the reasoning gate.

    Attributes
    ----------
    PASS : Proceed with the reasoning step.
    REJECT : Reject the reasoning step — unsafe to continue.
    GROUND : Ground reasoning in memory before continuing.
    ESCALATE : Escalate to a higher reasoning mode or human review.
    """

    PASS = "pass"
    REJECT = "reject"
    GROUND = "ground"
    ESCALATE = "escalate"


@dataclass
class GateResult:
    """Result of a gate decision on a reasoning step.

    Attributes
    ----------
    action : GateAction
        The action to take.
    reason : str
        Human-readable justification for the action.
    adjusted_confidence : float
        Confidence after any gate-based adjustments.
    violations : list[BoundaryViolation]
        Any violations detected during this gate evaluation.
    """

    action: GateAction
    reason: str
    adjusted_confidence: float = 0.0
    violations: list[BoundaryViolation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "adjusted_confidence": self.adjusted_confidence,
            "violations": [v.to_dict() for v in self.violations],
        }


@dataclass
class ReasoningEnvelopeState:
    """State of the constraint envelope during propagation.

    Attributes
    ----------
    active_constraints : dict[str, float]
        Currently active hard constraints.
    propagated_constraints : dict[str, float]
        Constraints that have been propagated from prior steps.
    depth : int
        Current propagation depth.
    """

    active_constraints: dict[str, float] = field(default_factory=dict)
    propagated_constraints: dict[str, float] = field(default_factory=dict)
    depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_constraints": self.active_constraints,
            "propagated_constraints": self.propagated_constraints,
            "depth": self.depth,
        }


@dataclass
class EnforcedReasoningState:
    """CalibratedReasoningState augmented with boundary enforcement metadata.

    Attributes
    ----------
    base_state : CalibratedReasoningState
        The original calibrated reasoning state.
    boundary_violations : list[BoundaryViolation]
        All violations detected during enforcement.
    envelope_state : ReasoningEnvelopeState
        Final envelope state after propagation.
    gate_decisions : list[GateResult]
        All gate decisions made during enforcement.
    enforcement_applied : bool
        Whether boundary enforcement was actually applied.
    """

    base_state: CalibratedReasoningState
    boundary_violations: list[BoundaryViolation] = field(default_factory=list)
    envelope_state: ReasoningEnvelopeState = field(default_factory=ReasoningEnvelopeState)
    gate_decisions: list[GateResult] = field(default_factory=list)
    enforcement_applied: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_state": self.base_state.to_dict(),
            "boundary_violations": [v.to_dict() for v in self.boundary_violations],
            "envelope_state": self.envelope_state.to_dict(),
            "gate_decisions": [g.to_dict() for g in self.gate_decisions],
            "enforcement_applied": self.enforcement_applied,
        }

    @property
    def has_critical_violations(self) -> bool:
        """True if any CRITICAL severity violations were detected."""
        return any(v.severity == ViolationSeverity.CRITICAL for v in self.boundary_violations)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class BoundaryViolationError(Exception):
    """Raised when a reasoning step violates safety bounds.

    Attributes
    ----------
    message : str
        The error message.
    violations : list[BoundaryViolation]
        The violations that caused the error.
    """

    def __init__(self, message: str, violations: Optional[list[BoundaryViolation]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.violations = violations or []


# ---------------------------------------------------------------------------
# BoundaryEnforcementLayer
# ---------------------------------------------------------------------------


class BoundaryEnforcementLayer:
    """Box Maze Boundary Enforcement Layer.

    This layer implements the Box Maze boundary enforcement strategy for the
    ACTR framework. It gates reasoning steps based on calibrated confidence,
    propagates soft constraints through the reasoning graph, and enforces
    safety bounds before allowing responses to be returned.

    The layer operates on the principle that reasoning confidence is bounded
    by a "box" defined by:
    - A lower safety bound (inference_safety_bound): below this, reject/escalate
    - A memory grounding threshold: below this, require grounding in memory
    - An envelope softness parameter: controls how constraints propagate

    Parameters
    ----------
    config : BoundaryEnforcementConfig | None
        Configuration for the enforcement layer. If None, uses defaults.
    """

    def __init__(self, config: Optional[BoundaryEnforcementConfig] = None) -> None:
        self.config = config if config is not None else BoundaryEnforcementConfig()

    def gate(
        self,
        reasoning_step: ReasoningStep,
        confidence: float,
        context: ReasoningContext,
    ) -> GateResult:
        """Evaluate the gate for a reasoning step.

        Makes a PASS/REJECT/GROUND/ESCALATE decision based on the confidence
        score and the reasoning context. This is the primary interface for
        evaluating individual reasoning steps.

        Decision logic:
        - p >= 0.85 → PASS (very high confidence)
        - p < inference_safety_bound (0.3) → REJECT (unsafe)
        - p < memory_grounding_threshold (0.7) → GROUND (needs grounding)
        - p < 0.15 → ESCALATE (critically low confidence)

        Parameters
        ----------
        reasoning_step : ReasoningStep
            The reasoning step to evaluate.
        confidence : float
            The calibrated confidence score (0.0 to 1.0).
        context : ReasoningContext
            The reasoning context including prior steps and constraints.

        Returns
        -------
        GateResult
            The gate decision with action, reason, adjusted confidence, and
            any violations detected.
        """
        violations: list[BoundaryViolation] = []
        adjusted_confidence = confidence

        # Critically low confidence (<= 0.15) → escalate first
        if confidence <= 0.15:
            violations.append(
                BoundaryViolation(
                    reason=f"Confidence {confidence:.3f} critically low, requires escalation",
                    severity=ViolationSeverity.WARN,
                    step=reasoning_step.step_id,
                )
            )
            return GateResult(
                action=GateAction.ESCALATE,
                reason=f"Escalated: confidence {confidence:.3f} at or below 0.15",
                adjusted_confidence=adjusted_confidence,
                violations=violations,
            )

        # Below safety bound (0.15 < p < 0.3) → reject
        if not self.check_safety_bound(confidence):
            violations.append(
                BoundaryViolation(
                    reason=f"Confidence {confidence:.3f} below safety bound {self.config.inference_safety_bound}",
                    severity=ViolationSeverity.CRITICAL,
                    step=reasoning_step.step_id,
                )
            )
            return GateResult(
                action=GateAction.REJECT,
                reason=f"Rejected: confidence {confidence:.3f} below safety bound {self.config.inference_safety_bound}",
                adjusted_confidence=adjusted_confidence,
                violations=violations,
            )

        # Below memory grounding threshold → ground in memory
        if self.should_ground_memory(confidence):
            violations.append(
                BoundaryViolation(
                    reason=f"Confidence {confidence:.3f} below memory grounding threshold {self.config.memory_grounding_threshold}",
                    severity=ViolationSeverity.INFO,
                    step=reasoning_step.step_id,
                )
            )
            return GateResult(
                action=GateAction.GROUND,
                reason=f"Ground in memory: confidence {confidence:.3f} below threshold {self.config.memory_grounding_threshold}",
                adjusted_confidence=adjusted_confidence,
                violations=violations,
            )

        # High confidence → pass
        return GateResult(
            action=GateAction.PASS,
            reason=f"Passed: confidence {confidence:.3f} within acceptable bounds",
            adjusted_confidence=adjusted_confidence,
            violations=violations,
        )

    def check_safety_bound(self, confidence: float) -> bool:
        """Check if confidence is above the inference safety bound.

        Returns True if confidence >= inference_safety_bound (above 0.3 = safe
        to respond). Below the safety bound is unsafe and requires rejection
        or escalation.

        Parameters
        ----------
        confidence : float
            The confidence score to check.

        Returns
        -------
        bool
            True if the confidence is safe to proceed, False otherwise.
        """
        return confidence >= self.config.inference_safety_bound

    def should_ground_memory(self, confidence: float) -> bool:
        """Determine whether memory grounding is needed.

        Memory grounding is only triggered when confidence is below the
        memory_grounding_threshold (0.7). Above that threshold, the
        reasoning is confident enough that memory grounding is not needed.

        Parameters
        ----------
        confidence : float
            The confidence score to evaluate.

        Returns
        -------
        bool
            True if memory grounding should be triggered, False otherwise.
        """
        return confidence < self.config.memory_grounding_threshold

    def propagate_envelope(self, state: ReasoningEnvelopeState) -> ReasoningEnvelopeState:
        """Propagate constraints forward through the reasoning graph.

        Constraints from prior steps are propagated forward with exponential
        decay based on the envelope_softness parameter. Each step inherits
        constraints from its predecessors, attenuated by depth and softness.

        The propagation stops when max_envelope_depth is reached.

        Parameters
        ----------
        state : ReasoningEnvelopeState
            The current envelope state with active and propagated constraints.

        Returns
        -------
        ReasoningEnvelopeState
            Updated envelope state with propagated constraints.
        """
        if state.depth >= self.config.max_envelope_depth:
            # Max depth reached — stop propagation
            return state

        # Build the next layer of propagated constraints
        softness = self.config.envelope_softness
        decay = 1.0 - softness

        # Copy existing propagated constraints with further decay
        propagated: dict[str, float] = {}
        for key, value in state.propagated_constraints.items():
            propagated[key] = value * decay

        # Propagate active constraints forward with depth-adjusted decay
        for key, value in state.active_constraints.items():
            # Stronger decay for active constraints that are newly propagated
            depth_decay = decay ** (state.depth + 1)
            new_value = value * depth_decay
            # Take max of existing propagated and new decayed value
            propagated[key] = max(propagated.get(key, 0.0), new_value)

        return ReasoningEnvelopeState(
            active_constraints=state.active_constraints,
            propagated_constraints=propagated,
            depth=state.depth + 1,
        )

    def enforce(self, state: CalibratedReasoningState) -> EnforcedReasoningState:
        """Apply full boundary enforcement to a reasoning state.

        This is the main entry point for boundary enforcement. It:
        1. Evaluates the gate for the reasoning step
        2. Checks the safety bound
        3. Determines if memory grounding is needed
        4. Propagates constraints through the envelope
        5. Returns an EnforcedReasoningState with all boundary metadata

        Parameters
        ----------
        state : CalibratedReasoningState
            The reasoning state to enforce.

        Returns
        -------
        EnforcedReasoningState
            The enforced reasoning state with boundary violations, envelope
            state, and gate decisions.
        """
        all_violations: list[BoundaryViolation] = []
        gate_decisions: list[GateResult] = []

        # Convert reasoning steps if available
        reasoning_steps: list[ReasoningStep] = []
        for i, step_content in enumerate(state.reasoning_steps):
            reasoning_steps.append(
                ReasoningStep(
                    step_id=f"step_{i}",
                    content=step_content,
                    confidence=state.calibrated_confidence,
                    metadata={},
                )
            )

        # If no reasoning steps but we have content, treat as single step
        if not reasoning_steps and state.reasoning_content:
            reasoning_steps.append(
                ReasoningStep(
                    step_id="step_0",
                    content=state.reasoning_content,
                    confidence=state.calibrated_confidence,
                    metadata={},
                )
            )

        # Build context
        context = ReasoningContext(
            prompt=state.prompt,
            prior_steps=reasoning_steps,
            constraints={},
        )

        # Evaluate gate for each reasoning step
        envelope_state = ReasoningEnvelopeState(
            active_constraints={},
            propagated_constraints={},
            depth=0,
        )

        for step in reasoning_steps:
            gate_result = self.gate(step, step.confidence, context)
            gate_decisions.append(gate_result)
            all_violations.extend(gate_result.violations)

            # Update envelope with propagated constraints
            if gate_result.action == GateAction.PASS:
                # Propagate envelope after a pass
                envelope_state = self.propagate_envelope(envelope_state)

        # Final envelope propagation if we passed
        if reasoning_steps and all(
            g.action in (GateAction.PASS, GateAction.GROUND) for g in gate_decisions
        ):
            envelope_state = self.propagate_envelope(envelope_state)

        return EnforcedReasoningState(
            base_state=state,
            boundary_violations=all_violations,
            envelope_state=envelope_state,
            gate_decisions=gate_decisions,
            enforcement_applied=True,
        )
