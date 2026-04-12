# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Data models for Nexus-Config-Env.

Simulates real-world Kubernetes infrastructure remediation.
The AI agent must inspect a malformed/vulnerable YAML config,
understand the telemetry context, and apply the correct fix.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal

try:
    from openenv.core.env_server.types import Action, Observation
    _base_action = Action
    _base_obs = Observation
except ImportError:
    _base_action = BaseModel
    _base_obs = BaseModel


class NexusAction(_base_action):
    """
    The remediation action the agent submits to the environment.

    The agent must identify:
      1. The TYPE of issue (cost / security / stability)
      2. The exact YAML field path to change
      3. The correct hardened value
    """

    fix_type: Literal["cost", "security", "stability"] = Field(
        ...,
        description=(
            "Category of the misconfiguration being fixed: "
            "'cost' for resource over-provisioning, "
            "'security' for access/privilege violations, "
            "'stability' for reliability/availability issues."
        ),
    )
    target_field: str = Field(
        ...,
        description=(
            "Dot-notation YAML path to the field to fix. "
            "Examples: 'resources.requests.memory', "
            "'securityContext.runAsUser', 'securityContext.privileged'."
        ),
    )
    new_value: str = Field(
        ...,
        description=(
            "The hardened value to apply. "
            "Examples: '256Mi', '1000', 'false'."
        ),
    )
    reasoning: str = Field(
        default="AI remediation",
        description="Brief explanation of why this fix is being applied.",
    )


class NexusObservation(_base_obs):
    """
    What the agent observes at each step of the environment.

    Contains:
      - The current (possibly broken) Kubernetes YAML
      - Real-time telemetry signals
      - Episode progress indicators
    """

    config_id: str = Field(
        description="Unique scenario identifier for reproducibility.",
    )
    dirty_yaml: str = Field(
        description=(
            "The raw Kubernetes manifest currently under evaluation. "
            "May contain misconfigurations that the agent must fix."
        ),
    )
    telemetry: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Real-time workload metrics: memory usage, CPU, security flags, "
            "cost estimates, CVE scores, etc."
        ),
    )
    fixes_applied: List[str] = Field(
        default_factory=list,
        description="List of field paths that have been successfully fixed this episode.",
    )
    current_score: float = Field(
        default=0.0,
        description="Cumulative grader score for this episode, strictly between 0 and 1.",
    )
    step: int = Field(
        default=0,
        description="Current step number within the episode.",
    )
    done: bool = Field(
        default=False,
        description="True when the episode is complete (fix applied or step budget exhausted).",
    )