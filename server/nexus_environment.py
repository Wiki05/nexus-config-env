from models import NexusAction, NexusObservation
from scenarios import SCENARIOS

class NexusEnvironment:
    def __init__(self):
        self.current_scenario = None
        self.step_count = 0
        self.max_steps = 2
        self.current_score = 0.0

    async def reset(self, task_id: str = "task_1_easy") -> NexusObservation:
        self.current_scenario = SCENARIOS[task_id][0]
        self.step_count = 0
        self.current_score = 0.0
        return self._get_obs()

    async def step(self, action: NexusAction) -> tuple[NexusObservation, float, bool, dict]:
        self.step_count += 1
        reward = 0.0
        done = False
        
        # Normalize fields for matching (ignore slashes and case)
        target_f = self.current_scenario["target"].lower().replace("/", ".")
        target_v = str(self.current_scenario["limit"]).lower().strip()
        
        provided_f = action.target_field.lower().replace("/", ".")
        provided_v = str(action.new_value).lower().strip()
        
        if self.step_count == 1:
            # Step 1: Just need to identify the field
            if target_f in provided_f or provided_f in target_f:
                reward = 0.50
            done = False
            
        elif self.step_count >= 2:
            # Step 2: Need the right field AND the right value
            field_match = target_f in provided_f or provided_f in target_f
            value_match = (provided_v == target_v)
            
            if field_match and value_match:
                reward = 0.50
            done = True
            
        self.current_score += reward
        self.current_score = min(self.current_score, 1.0)
        
        return self._get_obs(done), reward, done, {}

    def _get_obs(self, done: bool = False) -> NexusObservation:
        return NexusObservation(
            config_id=self.current_scenario["id"],
            dirty_yaml=self.current_scenario["dirty_yaml"],
            telemetry=self.current_scenario["telemetry"],
            fixes_applied=[],
            current_score=self.current_score,
            step=self.step_count,
            done=done
        )

    async def close(self): pass