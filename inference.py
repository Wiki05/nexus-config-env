import os
import asyncio
import json
import httpx
from openai import OpenAI
from typing import List, Optional

# 1. Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "https://wiki05-nexus-config-env.hf.space")

# 2. Logging Helpers
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.1f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.1f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.1f} rewards={rewards_str}", flush=True)

def clean_json_response(raw_content: str):
    content = raw_content.strip()
    if content.startswith("```json"):
        content = content.replace("```json", "", 1).replace("```", "", 1).strip()
    elif content.startswith("```"):
        content = content.replace("```", "", 2).strip()
    return content

# 3. Individual Task Logic
async def run_task(task_id: str, client, http_client):
    log_start(task=task_id, env="Nexus-Config-Env", model=MODEL_NAME)
    
    steps_taken = 0
    rewards = []
    
    try:
        # Step A: Reset with Status Check
        response = await http_client.post(f"{ENV_URL}/reset?task_id={task_id}")
        
        if response.status_code != 200:
            print(f"[DEBUG] Server not ready (Status {response.status_code}). Wait for HF Space to turn GREEN.")
            log_end(success=False, steps=0, score=0.0, rewards=[])
            return

        observation = response.json()["observation"]
        
        # Step B: Solving Loop
        for step_num in range(1, 3): 
            steps_taken = step_num
            
            prompt = f"""
            Kubernetes YAML: {observation['dirty_yaml']}
            Goal: If memory is too high, fix 'memory'. If user is root, fix 'runAsUser'. If privileged is true, fix 'privileged'.
            Return ONLY a JSON object:
            "field": ("memory", "runAsUser", or "privileged"),
            "value": (the corrected value),
            "type": ("security" or "cost")
            """
            
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            
            raw_text = completion.choices[0].message.content
            json_text = clean_json_response(raw_text)
            ai_decision = json.loads(json_text)
            
            action_payload = {
                "fix_type": ai_decision.get("type", "cost"),
                "target_field": ai_decision.get("field", "unknown"),
                "new_value": str(ai_decision.get("value", "")),
                "reasoning": "Standardizing config"
            }
            
            # Step C: Send Action
            step_response = await http_client.post(f"{ENV_URL}/step", json=action_payload)
            step_data = step_response.json()
            
            reward = step_data["reward"]
            done = step_data["done"]
            observation = step_data["observation"]
            
            rewards.append(reward)
            log_step(step=step_num, action=ai_decision.get("field", "none"), reward=reward, done=done, error=None)
            
            if done: break
            
        total_score = sum(rewards)
        log_end(success=(total_score >= 1.0), steps=steps_taken, score=total_score, rewards=rewards)

    except Exception as e:
        print(f"[DEBUG] Error during task {task_id}: {e}")
        log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)

# 4. Main Loop
async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        for task in ["task_1_easy", "task_2_medium", "task_3_hard"]:
            await run_task(task, client, http_client)
            print("-" * 30)

if __name__ == "__main__":
    asyncio.run(main())