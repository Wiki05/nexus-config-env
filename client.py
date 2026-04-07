from openenv import EnvClient
from .models import NexusAction, NexusObservation

class NexusEnv(EnvClient):
    def __init__(self, base_url: str = "http://localhost:7860"):
        super().__init__(base_url=base_url)