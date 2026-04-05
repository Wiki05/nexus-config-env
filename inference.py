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
ENV_URL = "http://127.0.0.1:7860" 

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.1f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.1f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.1f} rewards={rewards_str}", flush=True)

def clean_json_response(raw_content: str):
    content = raw_content.strip()
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    if start_idx != -1 and end_idx != -1:
        return content[start_idx:end_idx + 1]
    return content

async def run_task(task_id: str, client, http_client):
    log_start(task=task_id, env="Nexus-Config-Env", model=MODEL_NAME)
    steps_taken, rewards = 0, []
    
    try:
        response = await http_client.post(f"{ENV_URL}/reset?task_id={task_id}")
        observation = response.json()["observation"]
        
        for step_num in range(1, 3): 
            steps_taken = step_num
            # Simplified Prompt to prevent AI from repeating the choices
            prompt = f"""
            Analyze this Kubernetes YAML: {observation['dirty_yaml']}
            Your goal is to fix the waste or security risk.
            Return ONLY a JSON object with:
            "field": Pick ONE from ('memory', 'runAsUser', 'privileged')
            "value": The corrected value (e.g., '256Mi', '1000', or 'false')
            "type": Either 'cost' or 'security'
            """
            
            completion = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
            ai_decision = json.loads(clean_json_response(completion.choices[0].message.content))
            
            action_payload = {
                "fix_type": ai_decision.get("type", "cost"),
                "target_field": ai_decision.get("field", "unknown"),
                "new_value": str(ai_decision.get("value", "")),
                "reasoning": "Optimizing config"
            }
            
            step_res = await http_client.post(f"{ENV_URL}/step", json=action_payload)
            step_data = step_res.json()
            
            reward, done = step_data["reward"], step_data["done"]
            observation = step_data["observation"]
            rewards.append(reward)
            
            log_step(step=step_num, action=ai_decision.get("field", "none"), reward=reward, done=done, error=None)
            if done: break
            
        log_end(success=(sum(rewards) >= 1.0), steps=steps_taken, score=sum(rewards), rewards=rewards)
    except Exception as e:
        print(f"[ERROR] {e}")

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        for task in ["task_1_easy", "task_2_medium", "task_3_hard"]:
            await run_task(task, client, http_client)
            print("-" * 20)

if __name__ == "__main__":
    asyncio.run(main())