from pydantic import BaseModel, Field
from typing import List, Dict, Any

class NexusAction(BaseModel):
    """The agent's decision: what to fix in the cloud config."""
    fix_type: str = Field(description="Type: 'security', 'cost', or 'health'")
    target_field: str = Field(description="The YAML path to fix (e.g., 'resources.requests.memory')")
    new_value: str = Field(description="The optimized value (e.g., '256Mi')")
    reasoning: str = Field(description="Why this fix is being applied.")

class NexusObservation(BaseModel):
    """What the agent sees at each step."""
    config_id: str
    dirty_yaml: str
    telemetry: Dict[str, Any]
    fixes_applied: List[str]
    current_score: float
    step: int
    done: bool