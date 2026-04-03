from models import NexusAction, NexusObservation
from scenarios import SCENARIOS

class NexusEnvironment:
    def __init__(self):
        self.current_scenario = None
        self.step_count = 0
        self.max_steps = 4
        self.current_score = 0.0

    async def reset(self, task_id: str = "task_1_easy") -> NexusObservation:
        self.current_scenario = SCENARIOS[task_id][0]
        self.step_count = 0
        self.current_score = 0.0
        return self._get_obs()

    async def step(self, action: NexusAction) -> tuple[NexusObservation, float, bool, dict]:
        self.step_count += 1
        
        # Grading: 1.0 reward if the agent identifies the correct field to fix
        reward = 0.0
        if action.target_field in self.current_scenario["target"]:
            reward = 1.0 
            
        self.current_score += reward
        done = self.step_count >= self.max_steps or reward > 0
        
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

    async def close(self):
        pass