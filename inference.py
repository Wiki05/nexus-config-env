import os
import json
import asyncio
import textwrap
from typing import List, Optional, Tuple

import httpx
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:7860")

BENCHMARK = "Nexus-Config-Env"
TASKS = ["task_1_easy", "task_2_medium", "task_3_hard"]
MAX_STEPS = 2
TEMPERATURE = 0.1
MAX_TOKENS = 120
SUCCESS_SCORE_THRESHOLD = 1.0

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are solving Kubernetes hardening tasks.

    Return ONLY a JSON object with exactly these keys:
    {
      "field": "<one of memory | runAsUser | privileged>",
      "value": "<correct fix value>",
      "type": "<cost or security>"
    }

    Rules:
    - Use short field names only: memory, runAsUser, privileged
    - Do not return full YAML paths
    - Do not add markdown
    - Do not add explanations
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

def extract_json(raw_text: str) -> dict:
    text = (raw_text or "").strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {raw_text}")

    return json.loads(text[start:end + 1])


def normalize_field(field: str) -> str:
    field = (field or "").strip()

    mapping = {
        "memory": "memory",
        "runasuser": "runAsUser",
        "privileged": "privileged",
        "resources.requests.memory": "memory",
        "securitycontext.runasuser": "runAsUser",
        "securitycontext.privileged": "privileged",
    }

    key = field.replace("/", ".").replace(" ", "").lower()
    return mapping.get(key, field)


def build_user_prompt(task_id: str, step_num: int, observation: dict, history: List[str]) -> str:
    history_block = "\n".join(history) if history else "None"

    return textwrap.dedent(
        f"""
        Task ID: {task_id}
        Step: {step_num}/{MAX_STEPS}

        Current YAML:
        {observation.get("dirty_yaml", "")}

        Telemetry:
        {json.dumps(observation.get("telemetry", {}), ensure_ascii=False)}

        Previous actions:
        {history_block}

        Choose the correct short field name and corrected value.

        Known valid field names:
        - memory
        - runAsUser
        - privileged

        Typical correct values:
        - memory -> 256Mi
        - runAsUser -> 1000
        - privileged -> false

        Return JSON only.
        """
    ).strip()


def get_model_action(client: OpenAI, task_id: str, step_num: int, observation: dict, history: List[str]) -> dict:
    prompt = build_user_prompt(task_id, step_num, observation, history)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )

    raw = completion.choices[0].message.content or ""
    data = extract_json(raw)

    field = normalize_field(str(data.get("field", "")))
    value = str(data.get("value", "")).strip()
    fix_type = str(data.get("type", "security")).strip().lower()

    if field not in {"memory", "runAsUser", "privileged"}:
        raise ValueError(f"Invalid field from model: {field}")

    if fix_type not in {"cost", "security"}:
        fix_type = "security"

    return {
        "field": field,
        "value": value,
        "type": fix_type,
    }


def build_action_string(payload: dict) -> str:
    return f"{payload['type']}:{payload['field']}={payload['value']}"


async def reset_env(http_client: httpx.AsyncClient, task_id: str) -> dict:
    response = await http_client.post(f"{ENV_URL}/reset", params={"task_id": task_id})
    response.raise_for_status()
    body = response.json()
    return body["observation"]


async def step_env(http_client: httpx.AsyncClient, payload: dict) -> Tuple[dict, float, bool, Optional[str]]:
    response = await http_client.post(
        f"{ENV_URL}/step",
        json={
            "fix_type": payload["type"],
            "target_field": payload["field"],
            "new_value": payload["value"],
            "reasoning": "Baseline inference",
        },
    )
    response.raise_for_status()
    body = response.json()

    observation = body.get("observation", {}) or {}
    reward = float(body.get("reward", 0.0))
    done = bool(body.get("done", False))
    info = body.get("info", {}) or {}
    error = info.get("error") or info.get("warning") or None

    return observation, reward, done, error

async def run_task(task_id: str, client: OpenAI, http_client: httpx.AsyncClient) -> None:
    log_start(task_id, BENCHMARK, MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    history: List[str] = []

    try:
        observation = await reset_env(http_client, task_id)

        last_field: Optional[str] = None
        last_type: Optional[str] = None

        for step_num in range(1, MAX_STEPS + 1):
            steps_taken = step_num

            model_action = get_model_action(client, task_id, step_num, observation, history)

            if step_num == 1:
                last_field = model_action["field"]
                last_type = model_action["type"]
            else:
                if last_field:
                    model_action["field"] = last_field
                if last_type:
                    model_action["type"] = last_type

            action_str = build_action_string(model_action)

            observation, reward, done, error = await step_env(http_client, model_action)

            rewards.append(reward)
            log_step(step_num, action_str, reward, done, error)

            history.append(action_str)

            if done:
                break

        total_score = sum(rewards)
        success = total_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        total_score = sum(rewards)
        success = False

        if steps_taken == 0:
            steps_taken = 0
        else:
            log_step(steps_taken, "error", 0.00, False, str(exc))

    finally:
        log_end(success, steps_taken, sum(rewards), rewards)

async def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        for task_id in TASKS:
            await run_task(task_id, client, http_client)


if __name__ == "__main__":
    asyncio.run(main())