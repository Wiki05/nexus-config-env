from fastapi import FastAPI
from server.nexus_environment import NexusEnvironment
from models import NexusAction

app = FastAPI()
env = NexusEnvironment()

@app.post("/reset")
async def reset(task_id: str = "task_1_easy"):
    obs = await env.reset(task_id)
    return {"observation": obs}

@app.post("/step")
async def step(action: NexusAction):
    obs, reward, done, info = await env.step(action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}

@app.get("/health")
async def health():
    return {"status": "online", "project": "Nexus-Config-Env"}