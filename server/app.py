import sys
import json
import uvicorn
import gradio as gr
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.nexus_environment import NexusEnvironment
from models import NexusAction

app = FastAPI(title="Nexus-Config-Env", version="0.1.0")

_ui_env = NexusEnvironment()
_api_env = NexusEnvironment()

MAX_STEPS = 2


@app.get("/")
async def root():
    return RedirectResponse(url="/web")


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/metadata")
async def metadata():
    return {
        "name": "Nexus-Config-Env",
        "description": "Kubernetes YAML optimization and security hardening environment.",
        "version": "0.1.0",
        "tasks": [
            {"id": "task_1_easy", "difficulty": "easy", "name": "Ghost Hunter"},
            {"id": "task_2_medium", "difficulty": "medium", "name": "Security Patch"},
            {"id": "task_3_hard", "difficulty": "hard", "name": "Privilege Patch"},
        ],
    }


@app.post("/reset")
async def api_reset(task_id: str = "task_1_easy"):
    obs = await _api_env.reset(task_id)
    return JSONResponse(
        {
            "observation": obs.model_dump(),
            "reward": 0.0,
            "done": False,
        }
    )


@app.post("/step")
async def api_step(request: Request):
    body = await request.json()

    if _api_env.current_scenario is None:
        return JSONResponse(
            {
                "observation": None,
                "reward": 0.0,
                "done": False,
                "info": {"warning": "No active scenario. Call /reset first."},
            }
        )

    current_obs = _api_env._get_obs()
    if bool(getattr(current_obs, "done", False)) or int(getattr(current_obs, "step", 0)) >= MAX_STEPS:
        return JSONResponse(
            {
                "observation": current_obs.model_dump(),
                "reward": 0.0,
                "done": True,
                "info": {"warning": "Task already completed. Please reset to start again."},
            }
        )

    action = NexusAction(**body)
    obs, reward, done, info = await _api_env.step(action)

    return JSONResponse(
        {
            "observation": obs.model_dump(),
            "reward": round(float(reward), 2),
            "done": bool(done),
            "info": info,
        }
    )


@app.get("/state")
async def api_state():
    if _api_env.current_scenario is None:
        return {
            "has_active_task": False,
            "task_id": None,
            "step": 0,
            "observation": None,
        }

    obs = _api_env._get_obs()
    task_id = getattr(_api_env, "current_task_id", None)

    return {
        "has_active_task": True,
        "task_id": task_id,
        "step": min(int(getattr(obs, "step", 0)), MAX_STEPS),
        "observation": obs.model_dump(),
    }


@app.get("/schema")
async def api_schema():
    return {
        "action": NexusAction.model_json_schema(),
        "observation": {
            "type": "object",
            "properties": {
                "dirty_yaml": {"type": "string"},
                "telemetry": {"type": "object"},
                "step": {"type": "integer"},
                "done": {"type": "boolean"},
                "current_score": {"type": "number"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "has_active_task": {"type": "boolean"},
                "task_id": {"type": ["string", "null"]},
                "step": {"type": "integer"},
            },
        },
    }


@app.post("/mcp")
async def mcp(request: Request):
    body = await request.json()
    request_id = body.get("id")
    method = body.get("method", "")

    if method == "ping":
        result = {"status": "ok", "name": "Nexus-Config-Env"}
    else:
        result = {"status": "ok"}

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }


def _safe_json(data):
    try:
        return json.dumps(data, indent=2, default=str)
    except Exception:
        return "{}"


def _format_md(obs):
    display_step = min(int(getattr(obs, "step", 0)), MAX_STEPS)
    current_score = float(getattr(obs, "current_score", 0.0))
    done = bool(getattr(obs, "done", False)) or display_step >= MAX_STEPS
    status = "✅ COMPLETED" if done else "🚀 IN-PROGRESS"
    return (
        f"**Score:** {current_score:.2f} / 1.00"
        f" | **Step:** {display_step}/{MAX_STEPS}"
        f" | **Status:** {status}"
    )


def _build_ui_state_payload(obs):
    scenario = getattr(_ui_env, "current_scenario", None)
    task_id = getattr(_ui_env, "current_task_id", None)
    raw_step = int(getattr(obs, "step", 0))
    capped_step = min(raw_step, MAX_STEPS)

    return {
        "has_active_task": scenario is not None,
        "task_id": task_id,
        "step": capped_step,
        "raw_step": raw_step,
        "done": bool(getattr(obs, "done", False)) or capped_step >= MAX_STEPS,
        "current_score": float(getattr(obs, "current_score", 0.0)),
        "telemetry": getattr(obs, "telemetry", {}),
    }


async def ui_reset(task_id):
    try:
        obs = await _ui_env.reset(task_id)
        state_payload = _build_ui_state_payload(obs)
        return (
            _format_md(obs),
            getattr(obs, "dirty_yaml", ""),
            _safe_json(state_payload),
        )
    except Exception as e:
        return f"### ❌ Error\n{str(e)}", "", _safe_json({"error": str(e)})


async def ui_step(f_type, target, val):
    if _ui_env.current_scenario is None:
        return (
            "### ⚠️ Warning\nClick **Reset** first.",
            "",
            _safe_json({"warning": "No active scenario. Click Reset first."}),
        )

    current_obs = _ui_env._get_obs()
    current_step = int(getattr(current_obs, "step", 0))
    current_done = bool(getattr(current_obs, "done", False))

    if current_done or current_step >= MAX_STEPS:
        state_payload = _build_ui_state_payload(current_obs)
        state_payload["last_reward"] = 0.0
        state_payload["info"] = {"warning": "Task already completed. Please click Reset to start again."}
        return (
            "### ✅ Task already completed\nClick **Reset** to try again.",
            getattr(current_obs, "dirty_yaml", ""),
            _safe_json(state_payload),
        )

    try:
        action = NexusAction(
            fix_type=f_type,
            target_field=target,
            new_value=val,
            reasoning="UI Fix",
        )
        obs, reward, done, info = await _ui_env.step(action)

        state_payload = _build_ui_state_payload(obs)
        state_payload["last_reward"] = round(float(reward), 2)
        state_payload["done"] = bool(done) or state_payload["step"] >= MAX_STEPS
        state_payload["info"] = info

        return (
            _format_md(obs),
            getattr(obs, "dirty_yaml", ""),
            _safe_json(state_payload),
        )
    except Exception as e:
        return f"### ❌ Logic Error\n{str(e)}", "", _safe_json({"error": str(e)})


async def ui_get_state():
    if _ui_env.current_scenario is None:
        empty_state = {
            "has_active_task": False,
            "task_id": None,
            "step": 0,
            "raw_step": 0,
            "done": False,
            "current_score": 0.0,
            "telemetry": {},
        }
        return (
            "### 💤 System Ready\nClick **Reset** to begin.",
            "",
            _safe_json(empty_state),
        )

    try:
        obs = _ui_env._get_obs()
        state_payload = _build_ui_state_payload(obs)
        return (
            _format_md(obs),
            getattr(obs, "dirty_yaml", ""),
            _safe_json(state_payload),
        )
    except Exception as e:
        return f"### ❌ State Error\n{str(e)}", "", _safe_json({"error": str(e)})


with gr.Blocks() as demo:
    demo.title = "Nexus-Config-Env | Kubernetes Hardening"

    gr.HTML("""
    <style>
        input::-webkit-outer-spin-button,
        input::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        input[type=number] {
            -moz-appearance: textfield;
        }

        .gradio-container input[type="text"],
        .gradio-container textarea {
            background-color: #1f1f1f !important;
            color: #ffffff !important;
            border: 1px solid #444 !important;
        }
    </style>
    """)

    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            with gr.Accordion("Quick Start", open=False):
                gr.Markdown("""
                ## Quick Start

                1. Select a task from the dropdown.
                2. Click Reset to load the scenario.
                3. Read the Current YAML State.
                4. Check the Raw JSON Response.
                5. Enter Fix Category, Field Path, and New Value.
                6. Click Step.
                7. Use Get state anytime.

                ### Task IDs
                - `task_1_easy` → Ghost Hunter
                - `task_2_medium` → Security Patch
                - `task_3_hard` → Privilege Patch
                """)

            gr.Markdown("---")
            gr.Markdown("Built by **[Wiki05](https://github.com/Wiki05)** | Meta OpenEnv 2026")

        with gr.Column(scale=2):
            gr.Markdown("# Nexus-Config-Env Playground")

            with gr.Group():
                task_id = gr.Dropdown(
                    choices=["task_1_easy", "task_2_medium", "task_3_hard"],
                    label="Action (Select Hardening Task)",
                    value="task_1_easy",
                    interactive=True,
                )

                f_type = gr.Radio(
                    ["security", "cost", "stability"],
                    label="Fix Category",
                    value="security",
                )

                f_target = gr.Textbox(
                    label="Field Path",
                    placeholder="e.g., resources.requests.memory",
                    lines=1,
                    max_lines=1,
                )

                f_value = gr.Textbox(
                    label="New Value",
                    placeholder="e.g., 256Mi",
                    lines=1,
                    max_lines=1,
                )

            with gr.Row():
                btn_step = gr.Button("Step")
                btn_reset = gr.Button("Reset")
                btn_state = gr.Button("Get state")

            gr.Markdown("### Status")
            status_summary = gr.Markdown("Ready. Select a task and click Reset.")
            status_yaml = gr.Code(label="Current YAML State", language="yaml", interactive=False)

            with gr.Accordion("Raw JSON Response (Telemetry / State)", open=False):
                json_out = gr.Code(label="", language="json")

    btn_reset.click(ui_reset, inputs=[task_id], outputs=[status_summary, status_yaml, json_out])
    btn_step.click(ui_step, inputs=[f_type, f_target, f_value], outputs=[status_summary, status_yaml, json_out])
    btn_state.click(ui_get_state, inputs=[], outputs=[status_summary, status_yaml, json_out])


app = gr.mount_gradio_app(app, demo, path="/web")


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()