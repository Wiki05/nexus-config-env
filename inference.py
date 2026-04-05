import os
import asyncio
import json
import httpx
from openai import OpenAI
from typing import List, Optional

# 1. Configuration - Mandatory Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = "http://127.0.0.1:7860" # Local server address

def log_start(task: str, env: str, model: str):
    """Prints the [START] log exactly as required."""
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    """Prints the [STEP] log with exactly 2 decimal places for rewards."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    """Prints the [END] log with 2 decimal places for all values."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def clean_json_response(raw_content: str):
    """Safely extracts JSON from LLM text responses."""
    content = raw_content.strip()
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    if start_idx != -1 and end_idx != -1:
        return content[start_idx:end_idx + 1]
    return content

async def run_task(task_id: str, client, http_client):
    log_start(task=task_id, env="Nexus-Config-Env", model=MODEL_NAME)
    steps_taken = 0
    rewards = []
    success = False
    
    try:
        # Reset the environment for the new task
        response = await http_client.post(f"{ENV_URL}/reset?task_id={task_id}")
        if response.status_code != 200:
            raise Exception(f"Server reset failed with status {response.status_code}")
            
        observation = response.json()["observation"]
        
        # Max steps per task
        for step_num in range(1, 3): 
            steps_taken = step_num
            
            # 1. Ask the AI for a fix
            prompt = f"""
            Analyze this Kubernetes YAML: {observation['dirty_yaml']}
            Your goal is to fix the waste or security risk.
            Return ONLY a JSON object with:
            "field": Pick ONE from ('memory', 'runAsUser', 'privileged')
            "value": The corrected value (e.g., '256Mi', '1000', or 'false')
            "type": Either 'cost' or 'security'
            """
            
            completion = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            raw_ai_text = completion.choices[0].message.content
            ai_decision = json.loads(clean_json_response(raw_ai_text))
            
            # 2. Apply the fix to the environment
            action_payload = {
                "fix_type": ai_decision.get("type", "cost"),
                "target_field": ai_decision.get("field", "unknown"),
                "new_value": str(ai_decision.get("value", "")),
                "reasoning": "Standard OpenEnv Optimization"
            }
            
            step_res = await http_client.post(f"{ENV_URL}/step", json=action_payload)
            step_data = step_res.json()
            
            reward = float(step_data.get("reward", 0.0))
            done = step_data.get("done", False)
            observation = step_data.get("observation", {})
            
            rewards.append(reward)
            
            # Log this specific step
            log_step(
                step=step_num, 
                action=ai_decision.get("field", "none"), 
                reward=reward, 
                done=done, 
                error=None
            )
            
            if done:
                break
        
        # Calculate final score (normalized to 0.0 - 1.0)
        total_score = sum(rewards)
        success = total_score >= 1.0
        
    except Exception as e:
        print(f"[ERROR] Task {task_id} failed: {e}")
    finally:
        # The [END] line must always be emitted, even if the code crashes
        log_end(
            success=success, 
            steps=steps_taken, 
            score=min(max(sum(rewards), 0.0), 1.0), 
            rewards=rewards
        )

async def main():
    # Initialize OpenAI Client
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # Run the 3 mandatory tasks
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        for task in ["task_1_easy", "task_2_medium", "task_3_hard"]:
            await run_task(task, client, http_client)
            print("-" * 20)

if __name__ == "__main__":
    asyncio.run(main())