import os
import json
import asyncio
from typing import List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()

import httpx

ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:7860")

BENCHMARK = "Nexus-Config-Env"
TASKS = ["task_1_easy", "task_2_medium", "task_3_hard"]
MAX_STEPS = 2
SUCCESS_SCORE_THRESHOLD = 1.0


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
            "reasoning": "Deterministic baseline inference",
        },
    )
    response.raise_for_status()
    body = response.json()

    observation = body.get("observation", {}) or {}
    reward = float(body.get("reward", 0.0))
    done = bool(body.get("done", False))
    info = body.get("info", {}) or {}
    error = info.get("error") or info.get("warning") or info.get("message") or None

    return observation, reward, done, error


def choose_fix(observation: dict) -> dict:
    yaml_text = (observation.get("dirty_yaml", "") or "").lower()
    telemetry = observation.get("telemetry", {}) or {}

    if "memory" in yaml_text or "avg_mem_mb" in telemetry:
        return {
            "field": "memory",
            "value": "256Mi",
            "type": "cost",
        }

    if "runasuser" in yaml_text or telemetry.get("is_root") is True:
        return {
            "field": "runAsUser",
            "value": "1000",
            "type": "security",
        }

    if "privileged" in yaml_text or telemetry.get("privileged_status") is True:
        return {
            "field": "privileged",
            "value": "false",
            "type": "security",
        }

    raise ValueError(f"Could not determine fix from observation: {json.dumps(observation)}")


async def run_task(task_id: str, http_client: httpx.AsyncClient) -> None:
    log_start(task_id, BENCHMARK, "deterministic-baseline")

    rewards: List[float] = []
    steps_taken = 0
    success = False

    try:
        observation = await reset_env(http_client, task_id)
        chosen = choose_fix(observation)

        for step_num in range(1, MAX_STEPS + 1):
            steps_taken = step_num

            payload = {
                "field": chosen["field"],
                "type": chosen["type"],
                "value": chosen["value"],
            }

            action_str = build_action_string(payload)

            observation, reward, done, error = await step_env(http_client, payload)

            rewards.append(reward)
            log_step(step_num, action_str, reward, done, error)

            if done:
                break

        total_score = sum(rewards)
        success = total_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[ERROR] task={task_id} message={str(exc)}", flush=True)
        success = False

    finally:
        log_end(success, steps_taken, sum(rewards), rewards)


async def main() -> None:
    print(f"[DEBUG] ENV_URL={ENV_URL}", flush=True)

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        for task_id in TASKS:
            await run_task(task_id, http_client)


if __name__ == "__main__":
    asyncio.run(main())