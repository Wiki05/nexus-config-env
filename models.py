# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Data models for Nexus-Config-Env.

Kubernetes SRE Remediation Environment — Action and Observation types.

The environment uses a DISCRETE MULTI-ACTION space, modelling the
actual workflow a Site Reliability Engineer follows:
  1. scan_config       → gather YAML facts
  2. read_telemetry    → gather runtime signals
  3. identify_issue    → classify root cause
  4. propose_fix       → plan the remediation
  5. apply_fix         → execute the change
  6. verify_fix        → confirm the outcome
  7. escalate          → hand off to human SRE
  8. revert_change     → undo a bad change
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ── Action Model ──────────────────────────────────────────────────────────────

class NexusAction(BaseModel):
    """
    A discrete SRE remediation action.

    The agent chooses ONE action type per step. Depending on the
    action type, additional fields (target_field, new_value, fix_type)
    may be required.

    Action space and rewards:
      scan_config    → +0.10  Analyse YAML for structural issues
      read_telemetry → +0.10  Read runtime metrics (CPU, RAM, CVE scores)
      identify_issue → +0.20  Classify the root cause (cost/security/stability)
      propose_fix    → +0.15  Plan a field change without applying it
      apply_fix      → +0.50 if correct, -0.30 if wrong field, -0.50 if bad value
      verify_fix     → +0.20  Confirm the fix was applied; ends episode if done
      escalate       → +0.05  Hand off to human SRE; ends episode
      revert_change  → -0.10  Undo last change; resets field to dirty state
    """

    action_type: Literal[
        "scan_config",
        "read_telemetry",
        "identify_issue",
        "propose_fix",
        "apply_fix",
        "verify_fix",
        "escalate",
        "revert_change",
    ] = Field(
        ...,
        description=(
            "The SRE workflow action to take. Choose one of: "
            "scan_config, read_telemetry, identify_issue, propose_fix, "
            "apply_fix, verify_fix, escalate, revert_change."
        ),
    )

    # Optional fields used by specific actions
    target_field: Optional[str] = Field(
        default=None,
        description=(
            "Dot-notation YAML path for propose_fix / apply_fix. "
            "Examples: 'resources.requests.memory', 'securityContext.privileged'."
        ),
    )
    new_value: Optional[str] = Field(
        default=None,
        description=(
            "Hardened value for apply_fix / propose_fix. "
            "Examples: '256Mi', 'false', '1000'."
        ),
    )
    fix_type: Optional[Literal["cost", "security", "stability"]] = Field(
        default=None,
        description=(
            "Issue category for identify_issue / apply_fix. "
            "One of: cost, security, stability."
        ),
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this action is being taken.",
    )


# ── Observation Model ─────────────────────────────────────────────────────────

class NexusObservation(BaseModel):
    """
    What the agent observes at each step.

    Contains:
      - The current Kubernetes YAML (possibly patched after apply_fix)
      - Real-time telemetry metrics
      - Episode progress state
      - Hints from previous actions (identified issues, proposed fixes)
    """

    config_id: str = Field(
        description="Unique scenario identifier for reproducibility.",
    )
    dirty_yaml: str = Field(
        description=(
            "Current Kubernetes manifest under evaluation. "
            "May still contain misconfigurations until apply_fix succeeds."
        ),
    )
    telemetry: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Runtime workload metrics: memory usage, CPU, CVE scores, "
            "security flags, cost estimates."
        ),
    )
    message: str = Field(
        default="",
        description=(
            "Human-readable feedback from the last action. "
            "Guides the agent on what to do next."
        ),
    )
    fixes_applied: List[str] = Field(
        default_factory=list,
        description="Fields successfully patched during this episode.",
    )
    identified_issues: List[str] = Field(
        default_factory=list,
        description="Issue types the agent has explicitly identified so far.",
    )
    proposed_field: Optional[str] = Field(
        default=None,
        description="Last field proposed by propose_fix (before applying).",
    )
    current_score: float = Field(
        default=0.0,
        description="Cumulative grader score for this episode (0.001 – 0.999).",
    )
    step: int = Field(
        default=0,
        description="Current step number within the episode.",
    )
    done: bool = Field(
        default=False,
        description="True when episode is complete.",
    )
    actions_taken: List[str] = Field(
        default_factory=list,
        description="Ordered list of action_types taken so far this episode.",
    )


# ── Step Result (for internal use, matches patterns from reference envs) ──────

class StepResult(BaseModel):
    """Internal step result wrapping observation + reward."""
    observation: NexusObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)