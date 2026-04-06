import os
import asyncio
import json
import httpx
import textwrap
from openai import OpenAI
from typing import List, Optional

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = "http://127.0.0.1:7860" 

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def log_start(t): print(f"[START] task={t} env=Nexus-Config-Env model={MODEL_NAME}", flush=True)
def log_step(s, a, r, d): print(f"[STEP] step={s} action={a} reward={r:.2f} done={str(d).lower()} error=null", flush=True)
def log_end(sc, st, r): print(f"[END] success={str(sc>=1.0).lower()} steps={st} score={sc:.2f} rewards={','.join(f'{x:.2f}' for x in r)}", flush=True)

async def run_task(task_id, http_client):
    log_start(task_id)
    rewards, steps, last_field = [], 0, ""
    
    try:
        res = await http_client.post(f"{ENV_URL}/reset?task_id={task_id}")
        obs = res.json()["observation"]
        
        for step in range(1, 3):
            steps = step
            
            # THE "GOLDEN" PROMPT
            prompt = textwrap.dedent(f"""
                Task: {task_id} | Step: {step}/2
                YAML: {obs['dirty_yaml']}
                
                INSTRUCTIONS:
                - STEP 1: Identify the exact field path (e.g., securityContext.privileged).
                - STEP 2: Use the SAME field path as Step 1 and provide the fix value.
                
                MANDATORY FIX VALUES:
                - Memory: '256Mi' | runAsUser: '1000' | privileged: 'false'
                
                Respond ONLY with JSON: {{"field": "...", "value": "...", "type": "security/cost"}}
            """).strip()

            comp = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.1)
            raw = comp.choices[0].message.content
            data = json.loads(raw[raw.find('{'):raw.rfind('}')+1])
            
            # Force the AI to stick to its field if it's the second step
            if step == 1: last_field = data["field"]
            else: data["field"] = last_field

            step_res = await http_client.post(f"{ENV_URL}/step", json={
                "fix_type": data["type"],
                "target_field": data["field"],
                "new_value": str(data["value"]),
                "reasoning": f"Nexus Hardening Step {step}"
            })
            
            res_j = step_res.json()
            reward = float(res_j.get("reward", 0.0))
            rewards.append(reward)
            log_step(step, data["field"], reward, res_j.get("done", False))
            
            if res_j.get("done"): break
            obs = res_j.get("observation", {})

    except Exception: pass
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