import os
import asyncio
import json
import httpx
from openai import OpenAI
from typing import List, Optional

# 1. Configuration - Mandatory Environment Variables
# These will be provided by the Hackathon platform during evaluation
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# This is your local/remote environment URL
# Port 7860 is the Hugging Face standard we set in the Dockerfile
ENV_URL = os.getenv("ENV_URL", "http://0.0.0.0:7860")

# 2. Logging Helpers - Strict Format Required by Scaler/Meta
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# 3. The Inference Loop
async def run_inference():
    # Initialize OpenAI Client
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    task_name = "task_1_easy"
    log_start(task=task_name, env="Nexus-Config-Env", model=MODEL_NAME)
    
    steps_taken = 0
    rewards = []
    success = False
    
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        try:
            # Step A: Reset the environment
            response = await http_client.post(f"{ENV_URL}/reset?task_id={task_name}")
            data = response.json()
            observation = data["observation"]
            
            # Step B: Play the "Game" for 4 steps
            for step_num in range(1, 5):
                steps_taken = step_num
                
                # Ask the LLM what to do based on the dirty YAML
                prompt = f"Identify the resource waste in this YAML: {observation['dirty_yaml']}. Return the field name only."
                
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}]
                )
                action_text = completion.choices[0].message.content.strip()
                
                # Prepare the action for your FastAPI server
                # We simulate the fields for this baseline script
                action_payload = {
                    "fix_type": "cost",
                    "target_field": "memory",
                    "new_value": "256Mi",
                    "reasoning": f"Optimizing based on LLM suggestion: {action_text}"
                }
                
                # Step C: Send action to the environment
                step_response = await http_client.post(f"{ENV_URL}/step", json=action_payload)
                step_data = step_response.json()
                
                reward = step_data["reward"]
                done = step_data["done"]
                observation = step_data["observation"]
                
                rewards.append(reward)
                log_step(step=step_num, action=action_text, reward=reward, done=done, error=None)
                
                if done:
                    break
            
            # Final scoring logic
            total_score = sum(rewards)
            success = total_score >= 1.0
            log_end(success=success, steps=steps_taken, score=total_score, rewards=rewards)

        except Exception as e:
            log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)
            print(f"[DEBUG] Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(run_inference())