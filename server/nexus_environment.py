# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Core environment implementation for Nexus-Config-Env.
Simulates real-world Kubernetes YAML misconfiguration remediation
as an RL environment following OpenEnv spec.
"""

import yaml
from copy import deepcopy
from typing import Optional

try:
    from models import NexusAction, NexusObservation
    from scenarios import SCENARIOS, GRADERS
except ImportError:
    from ..models import NexusAction, NexusObservation
    from ..scenarios import SCENARIOS, GRADERS


class NexusEnvironment:
    """
    Kubernetes Configuration Hardening Environment.

    An agent interacts with this environment by:
      1. Calling reset(task_id) to load a scenario
      2. Calling step(action) to apply a remediation
      3. Observing reward and updated observation

    The grader uses deterministic scoring:
      - Category match: +0.20
      - Field identification: +0.40 (exact) or +0.20 (partial)
      - Value correction: +0.40 (only if field is also exact)
    """

    def __init__(self):
        self.current_scenario: Optional[dict] = None
        self.current_task_id: Optional[str] = None
        self.step_count: int = 0
        self.max_steps: int = 3
        self.current_score: float = 0.0
        self.done: bool = False
        self.episode_log: list = []

    async def reset(self, task_id: str = "task_1_easy") -> NexusObservation:
        """Reset environment to start a new episode for the given task."""
        if task_id not in SCENARIOS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid: {list(SCENARIOS.keys())}"
            )
        self.current_task_id = task_id
        self.current_scenario = deepcopy(SCENARIOS[task_id][0])
        self.step_count = 0
        self.current_score = 0.0
        self.done = False
        self.episode_log = []
        return self._get_obs()

    async def step(
        self, action: NexusAction
    ) -> tuple[NexusObservation, float, bool, dict]:
        """
        Apply an agent action and return (observation, reward, done, info).

        The grader is DETERMINISTIC:
          - Partial credit for category and field matching
          - Full credit only when both field and value are exactly right
        """
        if self.current_scenario is None:
            return self._get_obs(), 0.0, False, {"error": "No active scenario. Call /reset first."}

        self.step_count += 1
        reward = 0.0
        info: dict = {}

        # ── Normalize targets from scenario ───────────────────────────────
        scenario = self.current_scenario
        target_f = scenario["target"].lower().replace("/", ".").strip()
        target_v = str(scenario["limit"]).lower().strip()
        target_type = scenario.get("type", "security").lower()

        # ── Normalize agent action ────────────────────────────────────────
        provided_f = action.target_field.lower().replace("/", ".").strip()
        provided_v = str(action.new_value).lower().strip()
        provided_type = action.fix_type.lower()

        # ── Grader: Category reward (20%) ─────────────────────────────────
        if provided_type == target_type:
            reward += 0.20
            info["category_match"] = True
        else:
            info["category_match"] = False

        # ── Grader: Field identification (up to 40%) ──────────────────────
        if provided_f == target_f:
            reward += 0.40
            info["field_match"] = "exact"
        elif target_f in provided_f or provided_f in target_f:
            reward += 0.20
            info["field_match"] = "partial"
        else:
            info["field_match"] = "none"

        # ── Grader: Value correction (40%) — only on exact field match ─────
        if provided_f == target_f and provided_v == target_v:
            reward += 0.40
            info["value_match"] = True
            self._apply_fix(action)
            self.done = True
        else:
            info["value_match"] = False

        # ── Step budget enforcement ───────────────────────────────────────
        if self.step_count >= self.max_steps:
            self.done = True
            info["message"] = "Step budget exhausted."

        # ── Clamp and accumulate score ───────────────────────────────────
        reward = round(max(0.01, min(0.99, reward)), 2)
        self.current_score = round(max(self.current_score, reward), 2)

        # ── Episode log for external grader compatibility ─────────────────
        self.episode_log.append({
            "step": self.step_count,
            "action": provided_type,
            "target_field": provided_f,
            "new_value": provided_v,
            "reward": reward,
        })

        # ── Run post-hoc grader if done ───────────────────────────────────
        if self.done and self.current_task_id in GRADERS:
            graded_score = GRADERS[self.current_task_id](self.episode_log)
            self.current_score = graded_score
            info["graded_score"] = graded_score

        return self._get_obs(done=self.done), reward, self.done, info

    # ── Private helpers ───────────────────────────────────────────────────

    def _apply_fix(self, action: NexusAction) -> None:
        """Patch the dirty_yaml in the scenario with the correct value."""
        if self.current_scenario is None:
            return
        try:
            config = yaml.safe_load(self.current_scenario["dirty_yaml"]) or {}
            keys = action.target_field.split(".")
            node = config
            for key in keys[:-1]:
                if key not in node:
                    node[key] = {}
                node = node[key]
            # Try int/bool coercion for cleaner YAML
            raw_val: str = action.new_value
            val: object
            if raw_val.lower() == "true":
                val = True
            elif raw_val.lower() == "false":
                val = False
            else:
                try:
                    val = int(raw_val)
                except ValueError:
                    val = raw_val
            node[keys[-1]] = val
            self.current_scenario["dirty_yaml"] = yaml.dump(config, default_flow_style=False)
        except Exception:
            # Fallback: simple string replacement so scenario is never broken
            self.current_scenario["dirty_yaml"] = (
                f"# Fixed by agent\n{action.target_field}: {action.new_value}\n"
            )

    def _get_obs(self, done: Optional[bool] = None) -> NexusObservation:
        """Build a NexusObservation from current environment state."""
        if self.current_scenario is None:
            return NexusObservation(
                config_id="none",
                dirty_yaml="",
                telemetry={},
                fixes_applied=[],
                current_score=0.0,
                step=0,
                done=False,
            )
        effective_done = done if done is not None else self.done
        return NexusObservation(
            config_id=self.current_scenario.get("id", "unknown"),
            dirty_yaml=self.current_scenario.get("dirty_yaml", ""),
            telemetry=self.current_scenario.get("telemetry", {}),
            fixes_applied=self.current_scenario.get("fixes_applied", []),
            current_score=self.current_score,
            step=self.step_count,
            done=effective_done,
        )