import os
import asyncio
import json
import httpx
import textwrap
from openai import OpenAI
from typing import List, Optional

# --- ⚙️ CONFIGURATION ---
# Default to HF Router if env vars are missing
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = "http://127.0.0.1:7860" 

# Initialize Client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# --- 📊 LOGGING HELPERS ---
def log_start(t): 
    print(f"[START] task={t} env=Nexus-Config-Env model={MODEL_NAME}", flush=True)

def log_step(s, a, r, d): 
    print(f"[STEP] step={s} action={a} reward={r:.2f} done={str(d).lower()} error=null", flush=True)

def log_end(sc, st, r): 
    # Ensure rewards are formatted as floats for the final string
    reward_str = ",".join([f"{x:.2f}" for x in r]) if r else ""
    print(f"[END] success={str(sc>=1.0).lower()} steps={st} score={sc:.2f} rewards={reward_str}", flush=True)

# --- 🧠 TASK EXECUTION ---
async def run_task(task_id, http_client):
    log_start(task_id)
    rewards, steps, last_field = [], 0, ""
    
    try:
        # Reset the environment for the specific task
        res = await http_client.post(f"{ENV_URL}/reset?task_id={task_id}")
        res.raise_for_status()
        obs = res.json()["observation"]
        
        for step in range(1, 3):
            steps = step
            # Construct the prompt for the LLM
            prompt = textwrap.dedent(f"""
                Task: {task_id} | Step: {step}/2
                Current YAML:
                {obs['dirty_yaml']}
                
                Telemetry: {obs['telemetry']}
                
                GOAL:
                - STEP 1: Identify the exact field path (e.g., securityContext.privileged).
                - STEP 2: Use SAME path as Step 1 and provide the fix value.
                
                FIX VALUES: 
                - Memory: '256Mi' 
                - runAsUser: '1000' 
                - privileged: 'false'
                
                Respond ONLY with a JSON object: 
                {{"field": "path.to.field", "value": "new_value", "type": "security/cost"}}
            """).strip()

            # Call the LLM (Agent)
            comp = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.1
            )
            raw = comp.choices[0].message.content
            
            # 🛡️ Robust JSON Extraction
            try:
                start_idx = raw.find('{')
                end_idx = raw.rfind('}') + 1
                data = json.loads(raw[start_idx:end_idx])
            except Exception as e:
                print(f"❌ LLM Parsing Error on step {step}: {e} | Raw: {raw}")
                break
            
            # Step 2 must target the same field identified in Step 1
            if step == 1: 
                last_field = data["field"]
            else: 
                data["field"] = last_field

            # Apply the action to the Environment
            step_res = await http_client.post(f"{ENV_URL}/step", json={
                "fix_type": data["type"], 
                "target_field": data["field"],
                "new_value": str(data["value"]), 
                "reasoning": "Hardening via Inference"
            })
            
            res_j = step_res.json()
            reward = float(res_j.get("reward", 0.0))
            rewards.append(reward)
            
            is_done = bool(res_j.get("done", False))
            log_step(step, data["field"], reward, is_done)
            
            if is_done: break
            obs = res_j.get("observation", {})

    except Exception as e:
        print(f"❌ Runtime Error: {e}")
    finally:
        total = sum(rewards)
        log_end(total, steps, rewards)

async def main():
    async with httpx.AsyncClient(timeout=30.0) as hc:
        for t in ["task_1_easy", "task_2_medium", "task_3_hard"]:
            await run_task(t, hc)
            print("-" * 20, flush=True)

if __name__ == "__main__":
    asyncio.run(main())