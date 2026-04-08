from copy import deepcopy
from models import NexusAction, NexusObservation
from scenarios import SCENARIOS


class NexusEnvironment:
    def __init__(self):
        self.current_scenario = None
        self.current_task_id = None
        self.step_count = 0
        self.max_steps = 2
        self.current_score = 0.0
        self.done = False

        self.min_final_score = 0.01
        self.max_final_score = 0.99
        self.step1_reward = 0.49
        self.step2_reward = 0.50

    async def reset(self, task_id: str = "task_1_easy") -> NexusObservation:
        self.current_task_id = task_id
        self.current_scenario = deepcopy(SCENARIOS[task_id][0])
        self.step_count = 0
        self.current_score = 0.0
        self.done = False
        return self._get_obs()

    def _clamp_final_score(self, score: float) -> float:
        return min(self.max_final_score, max(self.min_final_score, score))

    async def step(self, action: NexusAction) -> tuple[NexusObservation, float, bool, dict]:
        if self.current_scenario is None:
            return self._get_obs(), 0.0, False, {"error": "No active scenario. Call reset first."}

        if self.done or self.step_count >= self.max_steps:
            return self._get_obs(done=True), 0.0, True, {
                "warning": "Task already completed. Reset to start again."
            }

        self.step_count += 1
        reward = 0.0
        info = {}

        target_f = self.current_scenario["target"].lower().replace("/", ".").strip()
        target_v = str(self.current_scenario["limit"]).lower().strip()

        provided_f = action.target_field.lower().replace("/", ".").strip()
        provided_v = str(action.new_value).lower().strip()

        field_match = (target_f in provided_f) or (provided_f in target_f)
        value_match = (provided_v == target_v)

        if self.step_count == 1:
            if field_match:
                reward = self.step1_reward
                info["message"] = "Correct field identified."
            else:
                reward = 0.0
                info["message"] = "Incorrect field."

            self.done = False

        elif self.step_count == 2:
            if field_match and value_match:
                reward = self.step2_reward
                info["message"] = "Correct value applied."
                self._apply_fix(action)
            else:
                reward = 0.0
                info["message"] = "Final fix incorrect."

            self.done = True

        self.current_score += reward

        # IMPORTANT: final task score must be strictly inside (0, 1)
        if self.done:
            self.current_score = self._clamp_final_score(self.current_score)
        else:
            self.current_score = min(self.current_score, self.max_final_score)

        return self._get_obs(done=self.done), reward, self.done, info

    def _apply_fix(self, action: NexusAction):
        target = self.current_scenario["target"]
        new_value = str(action.new_value)

        if target == "memory":
            self.current_scenario["dirty_yaml"] = (
                "resources:\n"
                "  requests:\n"
                f"    memory: '{new_value}'"
            )
            self.current_scenario["telemetry"]["recommended_mem_mb"] = 256

        elif target == "runAsUser":
            self.current_scenario["dirty_yaml"] = (
                "securityContext:\n"
                f"  runAsUser: {new_value}"
            )
            self.current_scenario["telemetry"]["is_root"] = False
            self.current_scenario["telemetry"]["user_id"] = int(new_value) if str(new_value).isdigit() else new_value

        elif target == "privileged":
            self.current_scenario["dirty_yaml"] = (
                "securityContext:\n"
                f"  privileged: {new_value}"
            )
            self.current_scenario["telemetry"]["privileged_status"] = False

    def _get_obs(self, done: bool = False) -> NexusObservation:
        return NexusObservation(
            config_id=self.current_scenario["id"] if self.current_scenario else "none",
            dirty_yaml=self.current_scenario["dirty_yaml"] if self.current_scenario else "",
            telemetry=self.current_scenario["telemetry"] if self.current_scenario else {},
            fixes_applied=[],
            current_score=self.current_score,
            step=min(self.step_count, self.max_steps),
            done=done
        )

    async def close(self):
        pass