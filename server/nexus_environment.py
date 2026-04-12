# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Core environment logic for Nexus-Config-Env.

Implements the OpenEnv RL environment contract:
  reset(task_id) → NexusObservation
  step(action)   → (NexusObservation, float, bool, dict)
  _get_obs()     → NexusObservation

Supports a discrete 8-action space that mirrors real SRE workflows:
  scan_config, read_telemetry, identify_issue, propose_fix,
  apply_fix, verify_fix, escalate, revert_change
"""

import yaml
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

try:
    from models import NexusAction, NexusObservation          # type: ignore[import]
    from tasks import TASKS, SCENARIOS, GRADERS, MIN_SCORE, MAX_SCORE, _clamp  # type: ignore[import]
except ImportError:
    from ..models import NexusAction, NexusObservation         # type: ignore[import]
    from ..tasks import TASKS, SCENARIOS, GRADERS, MIN_SCORE, MAX_SCORE, _clamp  # type: ignore[import]


# ── Per-action base rewards ────────────────────────────────────────────────────
# Scaled so sum(rewards) for any episode is strictly inside (0, 1).
# Perfect 6-step run: 0.06+0.06+0.12+0.09+0.40+0.12 = 0.85 (max possible ≈ 0.92)
ACTION_REWARDS: Dict[str, float] = {
    "scan_config":    +0.06,
    "read_telemetry": +0.06,
    "identify_issue": +0.12,
    "propose_fix":    +0.09,
    "apply_fix":      +0.00,   # earned dynamically by _handle_apply_fix
    "verify_fix":     +0.12,
    "escalate":       +0.04,
    "revert_change":  -0.10,
}

# Idempotency penalty: extra reward reduced for repeated non-fix actions
REPEAT_PENALTY: float = 0.05


class NexusEnvironment:
    """
    Kubernetes Configuration Hardening RL Environment.

    Episode lifecycle:
      1. reset(task_id)  → loads scenario, returns initial observation
      2. step(action)×N  → evaluates action, returns (obs, reward, done, info)
      3. Episode ends when:
           - apply_fix with correct field+value (+done=True)
           - verify_fix after successful fix (+done=True)
           - escalate (+done=True)
           - step budget exhausted (+done=True)
    """

    def __init__(self) -> None:
        self.current_scenario: Optional[dict] = None
        self.current_task_id:  Optional[str]  = None
        self.step_count:       int   = 0
        self.max_steps:        int   = 10
        self.current_score:    float = MIN_SCORE  # never 0.0
        self.done:             bool  = False
        self.fix_applied:      bool  = False
        self.proposed_field:   Optional[str] = None
        self.identified_issues: List[str] = []
        self.actions_taken:    List[str] = []
        self.episode_log:      List[dict] = []
        self._last_yaml_backup: Optional[str] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    async def reset(self, task_id: str = "task_1_easy") -> NexusObservation:
        """Reset environment to a fresh episode for the given task."""
        if task_id not in SCENARIOS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid options: {list(SCENARIOS.keys())}"
            )

        task = TASKS[task_id]
        self.current_task_id   = task_id
        self.current_scenario  = deepcopy(SCENARIOS[task_id][0])
        self.step_count        = 0
        self.max_steps         = task.max_steps
        self.current_score     = MIN_SCORE  # must never be 0.0 — evaluator may read this
        self.done              = False
        self.fix_applied       = False
        self.proposed_field    = None
        self.identified_issues = []
        self.actions_taken     = []
        self.episode_log       = []
        self._last_yaml_backup = None

        return self._get_obs(
            message=(
                f"Episode started: {task.name}. "
                f"You have {self.max_steps} steps. "
                "Start with scan_config or read_telemetry to gather information."
            )
        )

    async def step(
        self, action: NexusAction
    ) -> Tuple[NexusObservation, float, bool, dict]:
        """
        Apply a discrete SRE action and advance the episode.

        Returns: (observation, reward, done, info)
        """
        if self.current_scenario is None:
            return (
                self._get_obs(message="No active episode. Call /reset first."),
                MIN_SCORE, False,  # never 0.0
                {"error": "No active episode. Call POST /reset first."},
            )

        self.step_count += 1
        reward  = 0.0
        info: dict = {}
        action_type = action.action_type

        # ── Repeated non-fix action penalty ───────────────────────────────
        if (action_type in self.actions_taken
                and action_type not in ("apply_fix", "verify_fix")):
            info["repeated_action"] = True
            base_reward = max(0.0, ACTION_REWARDS.get(action_type, 0.0) - REPEAT_PENALTY)
        else:
            base_reward = ACTION_REWARDS.get(action_type, 0.0)

        # ── Dispatch per action type ───────────────────────────────────────
        if action_type == "scan_config":
            reward, info = self._handle_scan_config(base_reward)

        elif action_type == "read_telemetry":
            reward, info = self._handle_read_telemetry(base_reward)

        elif action_type == "identify_issue":
            reward, info = self._handle_identify_issue(action, base_reward)

        elif action_type == "propose_fix":
            reward, info = self._handle_propose_fix(action, base_reward)

        elif action_type == "apply_fix":
            reward, info = self._handle_apply_fix(action)

            if info.get("fix_correct"):
                self.fix_applied = True

        elif action_type == "verify_fix":
            reward, info = self._handle_verify_fix(base_reward)
            if self.fix_applied:
                self.done = True
                info["message"] = "Fix verified and confirmed. Episode complete!"

        elif action_type == "escalate":
            reward = base_reward
            self.done = True
            info["message"] = (
                "Escalated to human SRE. Episode ended. "
                "Partial score based on progress so far."
            )

        elif action_type == "revert_change":
            reward, info = self._handle_revert_change(base_reward)

        # ── Update actions log ─────────────────────────────────────────────
        self.actions_taken.append(action_type)

        # ── Record episode log entry for grader ───────────────────────────
        self.episode_log.append({
            "step":         self.step_count,
            "action_type":  action_type,
            "target_field": action.target_field,
            "new_value":    action.new_value,
            "fix_type":     action.fix_type,
            "reasoning":    action.reasoning,
            "reward":       round(float(reward), 3),
        })

        # ── Step budget enforcement ────────────────────────────────────────
        if self.step_count >= self.max_steps and not self.done:
            self.done = True
            info["budget_exhausted"] = True
            info["message"] = (
                info.get("message", "")
                + f" Step budget ({self.max_steps}) exhausted."
            )

        # ── Score accumulation (cumulative, clamped to [MIN_SCORE, 0.99]) ─────
        reward = round(max(MIN_SCORE, min(0.99, reward)), 3)
        self.current_score = round(min(0.99, max(MIN_SCORE, self.current_score + reward)), 3)

        # ── Run full episode grader if done ────────────────────────────────
        if self.done and self.current_task_id in GRADERS:
            graded = GRADERS[self.current_task_id](self.episode_log)
            self.current_score = graded
            info["grader_breakdown"] = self._run_grader_breakdown()
            info["final_grader_score"] = graded

        return self._get_obs(message=info.get("message", "")), reward, self.done, info

    # ── Action handlers ────────────────────────────────────────────────────────

    def _handle_scan_config(self, base: float) -> Tuple[float, dict]:
        scenario = self.current_scenario
        dirty = scenario.get("dirty_yaml", "")
        # Identify structural issues to point the agent in the right direction
        hints = []
        if "memory:" in dirty:
            hints.append("Found memory resource allocation — check values vs telemetry")
        if "cpu:" in dirty:
            hints.append("Found CPU resource allocation — check utilisation")
        if "runAsUser:" in dirty or "securityContext:" in dirty:
            hints.append("Found securityContext block — evaluate privilege settings")
        if "privileged:" in dirty:
            hints.append("Found privileged flag — CRITICAL: evaluate immediately")
        if "hostPID:" in dirty or "hostNetwork:" in dirty:
            hints.append("Found host namespace sharing — container escape risk")
        if "readOnlyRootFilesystem:" in dirty:
            hints.append("Found filesystem permission setting — check value")

        info = {
            "action": "scan_config",
            "yaml_lines": len(dirty.splitlines()),
            "structural_hints": hints,
            "message": (
                f"Config scan complete. {len(dirty.splitlines())} YAML lines analysed. "
                f"Hints: {'; '.join(hints[:2]) or 'No obvious structural issues detected.'}"
            ),
        }
        return base, info

    def _handle_read_telemetry(self, base: float) -> Tuple[float, dict]:
        telemetry = self.current_scenario.get("telemetry", {})
        scenario_desc = self.current_scenario.get("description", "")
        info = {
            "action": "read_telemetry",
            "telemetry": telemetry,
            "scenario": scenario_desc,
            "message": (
                f"Telemetry read. Key signals: "
                + ", ".join(f"{k}={v}" for k, v in list(telemetry.items())[:4])
                + ". Cross-reference with YAML config to identify waste or risk."
            ),
        }
        return base, info

    def _handle_identify_issue(self, action: NexusAction, base: float) -> Tuple[float, dict]:
        target_type = self.current_scenario.get("type", "security").lower()
        provided_type = str(action.fix_type or "").lower()

        # Complexity bonus: security/stability issues are harder to identify than cost
        _CATEGORY_BONUS: Dict[str, float] = {"cost": 0.00, "security": 0.02, "stability": 0.01}

        if provided_type == target_type:
            category_bonus = _CATEGORY_BONUS.get(target_type, 0.0)
            reward = base + category_bonus
            self.identified_issues.append(provided_type)
            info = {
                "action": "identify_issue",
                "category_correct": True,
                "message": (
                    f"Correct! Issue category '{provided_type}' confirmed. "
                    "Next: use propose_fix to plan the field change."
                ),
            }
        else:
            # Wrong category — return floor reward (avoid exact 0.0)
            reward = MIN_SCORE
            info = {
                "action": "identify_issue",
                "category_correct": False,
                "expected_category_hint": f"Hint: review telemetry for {target_type}-type signals.",
                "message": (
                    f"Category '{provided_type}' does not match the primary issue. "
                    "Re-read telemetry carefully."
                ),
            }
        return reward, info

    def _handle_propose_fix(self, action: NexusAction, base: float) -> Tuple[float, dict]:
        target_f = self.current_scenario.get("target", "").lower().replace("/", ".").strip()
        provided_f = str(action.target_field or "").lower().replace("/", ".").strip()

        self.proposed_field = action.target_field

        if provided_f == target_f:
            reward = base
            info = {
                "action": "propose_fix",
                "field_correct": True,
                "proposed_field": action.target_field,
                "message": (
                    f"Field '{action.target_field}' is the correct target. "
                    f"Now use apply_fix with the correct new_value."
                ),
            }
        elif target_f in provided_f or provided_f in target_f:
            reward = base * 0.6
            info = {
                "action": "propose_fix",
                "field_correct": "partial",
                "proposed_field": action.target_field,
                "message": (
                    f"Field '{action.target_field}' is in the right area but not exact. "
                    "Narrow down the path further."
                ),
            }
        else:
            reward = MIN_SCORE
            info = {
                "action": "propose_fix",
                "field_correct": False,
                "message": (
                    f"Field '{action.target_field}' is not the misconfigured field. "
                    "Re-scan the config."
                ),
            }
        return reward, info

    def _handle_apply_fix(self, action: NexusAction) -> Tuple[float, dict]:
        scenario  = self.current_scenario
        target_f  = scenario.get("target", "").lower().replace("/", ".").strip()
        target_v  = str(scenario.get("limit", "")).lower().strip()

        # Guard: if target_field or new_value is missing, penalise and explain
        if not action.target_field or not action.new_value:
            return -0.10, {
                "action":      "apply_fix",
                "fix_correct": False,
                "field_match": "none",
                "value_match": False,
                "message": (
                    "apply_fix requires both target_field and new_value. "
                    "Use propose_fix first to identify the exact field and value."
                ),
            }

        provided_f = str(action.target_field or "").lower().replace("/", ".").strip()
        provided_v = str(action.new_value or "").lower().strip()

        # ── Correct field + correct value ──────────────────────────────────
        if provided_f == target_f and provided_v == target_v:
            self._patch_yaml(action)
            reward = 0.40
            info  = {
                "action":      "apply_fix",
                "fix_correct": True,
                "field_match": "exact",
                "value_match": True,
                "message":     (
                    f"Fix applied! Set {action.target_field}={action.new_value}. "
                    "Use verify_fix to confirm and complete the episode."
                ),
            }
        # ── Correct field, wrong value ─────────────────────────────────────
        elif provided_f == target_f:
            reward = -0.10
            info   = {
                "action":      "apply_fix",
                "fix_correct": False,
                "field_match": "exact",
                "value_match": False,
                "message":     (
                    f"Field is correct but value '{action.new_value}' is wrong. "
                    "Consult telemetry for the correct target value."
                ),
            }
        # ── Wrong field ────────────────────────────────────────────────────
        elif target_f in provided_f or provided_f in target_f:
            reward = -0.20
            info   = {
                "action":      "apply_fix",
                "fix_correct": False,
                "field_match": "partial",
                "value_match": False,
                "message":     (
                    f"Field '{action.target_field}' is close but not exact — "
                    "no change applied. Use propose_fix to pin down the correct path."
                ),
            }
        # ── Completely wrong ───────────────────────────────────────────────
        else:
            reward = -0.30
            info   = {
                "action":      "apply_fix",
                "fix_correct": False,
                "field_match": "none",
                "value_match": False,
                "message":     (
                    f"Wrong field '{action.target_field}'. "
                    "Re-read scan_config and read_telemetry outputs."
                ),
            }

        return reward, info

    def _handle_verify_fix(self, base: float) -> Tuple[float, dict]:
        if self.fix_applied:
            # Scale verify reward by task difficulty — harder task = higher close-out reward
            task = TASKS.get(self.current_task_id)
            difficulty = task.difficulty if task else "medium"
            _VERIFY_REWARD: Dict[str, float] = {"easy": 0.10, "medium": 0.12, "hard": 0.14}
            reward = _VERIFY_REWARD.get(difficulty, base)
            info   = {
                "action":       "verify_fix",
                "fix_applied":  True,
                "difficulty":   difficulty,
                "current_yaml": self.current_scenario.get("dirty_yaml", ""),
                "message":      "Fix verified! YAML has been patched. Episode complete.",
            }
        else:
            reward = MIN_SCORE
            info   = {
                "action":      "verify_fix",
                "fix_applied": False,
                "message":     "Nothing to verify — no successful fix has been applied yet.",
            }
        return reward, info

    def _handle_revert_change(self, base: float) -> Tuple[float, dict]:
        if self._last_yaml_backup:
            self.current_scenario["dirty_yaml"] = self._last_yaml_backup
            self._last_yaml_backup = None
            self.fix_applied = False
            info = {
                "action":  "revert_change",
                "reverted": True,
                "message": "Last change reverted. YAML restored to previous state.",
            }
        else:
            info = {
                "action":  "revert_change",
                "reverted": False,
                "message": "Nothing to revert — no changes have been made yet.",
            }
        return base, info  # base is already -0.10

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _patch_yaml(self, action: NexusAction) -> None:
        """Apply the correct fix to the scenario YAML in-place."""
        if self.current_scenario is None:
            return
        self._last_yaml_backup = self.current_scenario["dirty_yaml"]
        try:
            config = yaml.safe_load(self.current_scenario["dirty_yaml"]) or {}
            keys   = action.target_field.split(".")
            node   = config
            for k in keys[:-1]:
                if k not in node:
                    node[k] = {}
                node = node[k]

            raw: str = action.new_value
            val: object
            if raw.lower()   == "true":  val = True
            elif raw.lower() == "false": val = False
            else:
                try:    val = int(raw)
                except: val = raw

            node[keys[-1]] = val
            self.current_scenario["dirty_yaml"] = yaml.dump(config, default_flow_style=False)
            if "fixes_applied" not in self.current_scenario:
                self.current_scenario["fixes_applied"] = []
            self.current_scenario["fixes_applied"].append(action.target_field)
        except Exception:
            # Fallback — annotate the YAML
            self.current_scenario["dirty_yaml"] = (
                f"# Fixed by agent on step {self.step_count}\n"
                f"{action.target_field}: {action.new_value}\n"
            )

    def _run_grader_breakdown(self) -> dict:
        """Return detailed grader breakdown for the episode log."""
        from tasks import _grade_episode  # type: ignore[import]
        task = TASKS.get(self.current_task_id)
        if task is None:
            return {}
        return _grade_episode(self.episode_log, task)

    def _get_obs(self, message: str = "", done: Optional[bool] = None) -> NexusObservation:
        """Build a NexusObservation from current environment state."""
        if self.current_scenario is None:
            return NexusObservation(
                config_id="none", dirty_yaml="", telemetry={},
                message="No active episode.", fixes_applied=[],
                identified_issues=[], proposed_field=None,
                current_score=MIN_SCORE, step=0, done=False, actions_taken=[],
            )
        effective_done = done if done is not None else self.done
        return NexusObservation(
            config_id     = self.current_scenario.get("id", "unknown"),
            dirty_yaml    = self.current_scenario.get("dirty_yaml", ""),
            telemetry     = self.current_scenario.get("telemetry", {}),
            message       = message,
            fixes_applied = self.current_scenario.get("fixes_applied", []),
            identified_issues = self.identified_issues,
            proposed_field    = self.proposed_field,
            current_score = round(float(max(MIN_SCORE, min(MAX_SCORE, self.current_score))), 3),
            step          = self.step_count,
            done          = effective_done,
            actions_taken = list(self.actions_taken),
        )