# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
FastAPI + Gradio server for Nexus-Config-Env.

OpenEnv-compliant endpoints:
  GET  /health     → liveness probe
  GET  /metadata   → environment description + task list
  GET  /tasks      → task list (alias)
  GET  /schema     → action + observation JSON schemas
  GET  /state      → current episode state (read-only)
  POST /reset      → start new episode for task_id
  POST /step       → submit NexusAction, receive reward + observation

Gradio playground mounted at /web.
"""

import json
import sys
import warnings
import uvicorn
import gradio as gr
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

# Suppress Gradio 6.0 deprecation warning — theme/css still work via mount_gradio_app
warnings.filterwarnings(
    "ignore",
    message="The parameters have been moved from the Blocks constructor",
    category=UserWarning,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.nexus_environment import NexusEnvironment  # type: ignore[import]  # noqa: E402
from models import NexusAction, NexusObservation        # type: ignore[import]  # noqa: E402
from tasks import SCENARIOS, TASKS                       # type: ignore[import]  # noqa: E402

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Nexus-Config-Env",
    version="0.4.0",
    description=(
        "Kubernetes SRE Remediation Environment — "
        "8-action discrete space, multi-criteria grader."
    ),
)

_ui_env  = NexusEnvironment()
_api_env = NexusEnvironment()
MAX_STEPS = 10


# ── Core OpenEnv endpoints ─────────────────────────────────────────────────────

@app.get("/")
async def root():
    return RedirectResponse(url="/web/")

@app.get("/web")
async def web_redirect():
    return RedirectResponse(url="/web/")

@app.get("/health")
async def health():
    return {"status": "healthy", "env": "nexus-config-env", "version": "0.4.0"}

@app.get("/metadata")
async def metadata():
    return {
        "name": "nexus-config-env",
        "display_name": "Nexus Config Env — Kubernetes Hardening",
        "version": "0.4.0",
        "description": (
            "Kubernetes SRE Remediation RL Environment. "
            "8-action discrete space. Multi-criteria grader with 4 dimensions: "
            "Protocol + Diagnosis + Remediation + Efficiency."
        ),
        "author": "Vignesh E",
        "action_space": "discrete (8 actions)",
        "grader": "multi_criteria_deterministic",
        "tasks": [
            {
                "id":         t.task_id,
                "name":       t.name,
                "difficulty": t.difficulty,
                "max_steps":  t.max_steps,
                "description": t.description,
            }
            for t in TASKS.values()
        ],
    }

@app.get("/tasks")
async def list_tasks():
    meta = await metadata()
    return {"tasks": meta["tasks"]}

@app.post("/reset")
async def api_reset(task_id: str = "task_1_easy"):
    if task_id not in SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid: {list(SCENARIOS.keys())}",
        )
    obs = await _api_env.reset(task_id)
    return JSONResponse({
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done":   False,
        "info":   {"message": f"Episode started: {task_id}"},
    })

@app.get("/manifest.json")
async def manifest():
    # Dummy manifest to prevent 404 errors in Gradio/Space logs
    return {"name": "Nexus-Config-Env", "short_name": "Nexus", "start_url": "/"}

@app.post("/step")
async def api_step(request: Request):
    if _api_env.current_scenario is None:
        return JSONResponse(
            {
                "observation": None,
                "reward": 0.0,
                "done":   False,
                "info":   {"warning": "No active episode. Call POST /reset first."},
            },
            status_code=400,
        )
    cur = _api_env._get_obs()
    if cur.done or cur.step >= MAX_STEPS:
        return JSONResponse({
            "observation": cur.model_dump(),
            "reward": 0.0,
            "done":   True,
            "info":   {"warning": "Episode complete. Call POST /reset to start a new one."},
        })
    try:
        body = await request.json()
    except Exception:
        body = {}

    try:
        # Handle standard OpenEnv evaluator payload format {"action": {...}}
        if "action" in body and isinstance(body["action"], dict):
            action_data = body["action"]
        else:
            action_data = dict(body)

        # Handle cases where client sends "action" key instead of "action_type"
        if "action" in action_data and "action_type" not in action_data:
            action_data["action_type"] = action_data.pop("action")

        # Strip unknown keys that Pydantic would reject — only keep known fields
        known_fields = {"action_type", "target_field", "new_value", "fix_type", "reasoning"}
        action_data = {k: v for k, v in action_data.items() if k in known_fields}

        # Sanitize fix_type — only allow valid literals or None
        valid_fix_types = {"cost", "security", "stability"}
        if action_data.get("fix_type") and str(action_data["fix_type"]).lower() in valid_fix_types:
            action_data["fix_type"] = str(action_data["fix_type"]).lower()
        else:
            action_data["fix_type"] = None

        # Ensure action_type is valid
        valid_actions = {
            "scan_config", "read_telemetry", "identify_issue", "propose_fix",
            "apply_fix", "verify_fix", "escalate", "revert_change",
        }
        if action_data.get("action_type") not in valid_actions:
            action_data["action_type"] = "scan_config"

        # Sanitize optional string fields
        for field in ("target_field", "new_value", "reasoning"):
            if action_data.get(field) is not None:
                action_data[field] = str(action_data[field]).strip() or None

        action = NexusAction(**action_data)
    except Exception as exc:
        # Never 422 — return a graceful error so the episode can continue
        cur = _api_env._get_obs() if _api_env.current_scenario else None
        return JSONResponse({
            "observation": cur.model_dump() if cur else None,
            "reward": 0.0,
            "done":   False,
            "info":   {"error": f"Invalid action payload: {exc}", "message": "Action rejected — check payload format."},
        }, status_code=200)

    obs, reward, done, info = await _api_env.step(action)
    return JSONResponse({
        "observation": obs.model_dump(),
        "reward": round(float(reward), 3),
        "done":   bool(done),
        "info":   info,
    })

@app.get("/state")
async def api_state():
    if _api_env.current_scenario is None:
        return {"has_active_task": False, "task_id": None, "step": 0, "observation": None}
    obs = _api_env._get_obs()
    return {
        "has_active_task": True,
        "task_id": _api_env.current_task_id,
        "step":    obs.step,
        "observation": obs.model_dump(),
    }

@app.get("/schema")
async def api_schema():
    return {
        "action":      NexusAction.model_json_schema(),
        "observation": NexusObservation.model_json_schema(),
    }


# ── Gradio UI helpers ──────────────────────────────────────────────────────────

def _safe_json(data) -> str:
    try:    return json.dumps(data, indent=2, default=str)
    except: return "{}"

def _fmt_status(obs, env_inst=None) -> str:
    task_max   = int(getattr(env_inst, "max_steps", MAX_STEPS)) if env_inst else MAX_STEPS
    step  = min(int(getattr(obs, "step", 0)), task_max)
    score = float(getattr(obs, "current_score", 0.0))
    done  = bool(getattr(obs, "done", False)) or step >= task_max
    icon  = "✅ COMPLETE" if done else "🔄 IN-PROGRESS"
    bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    return (
        f"**Score:** {score:.3f} / 1.000 `{bar}`\n\n"
        f"**Step:** {step}/{task_max} | **Status:** {icon}"
    )

def _build_payload(obs, env_inst) -> dict:
    step  = min(int(getattr(obs, "step", 0)), MAX_STEPS)
    done  = bool(getattr(obs, "done", False)) or step >= MAX_STEPS
    return {
        "has_active_task":  env_inst.current_scenario is not None,
        "task_id":          getattr(env_inst, "current_task_id", None),
        "step":             step,
        "done":             done,
        "current_score":    float(getattr(obs, "current_score", 0.0)),
        "actions_taken":    getattr(obs, "actions_taken", []),
        "identified_issues": getattr(obs, "identified_issues", []),
        "proposed_field":   getattr(obs, "proposed_field", None),
        "telemetry":        getattr(obs, "telemetry", {}),
        "message":          getattr(obs, "message", ""),
    }

async def ui_reset(task_id):
    try:
        obs = await _ui_env.reset(task_id)
        p   = _build_payload(obs, _ui_env)
        return _fmt_status(obs), getattr(obs, "dirty_yaml", ""), obs.message, _safe_json(p)
    except Exception as e:
        return f"❌ {e}", "", str(e), _safe_json({"error": str(e)})

async def ui_step(action_type, fix_type_val, target_field, new_value, reasoning):
    if _ui_env.current_scenario is None:
        empty = "### ⚠️ No active episode\nClick **Reset** to load a task."
        return empty, "", "Click Reset first.", _safe_json({"warning": "Call Reset first."})

    cur       = _ui_env._get_obs()
    max_steps = int(getattr(_ui_env, "max_steps", MAX_STEPS))
    if cur.done or cur.step >= max_steps:
        p = _build_payload(cur, _ui_env)
        return (
            "### ✅ Episode complete — click **Reset** to start again.",
            getattr(cur, "dirty_yaml", ""),
            "Episode already ended.",
            _safe_json(p),
        )

    try:
        action_kwargs = {
            "action_type": action_type,
            "reasoning":   reasoning or "",
        }
        if fix_type_val and fix_type_val != "none":
            action_kwargs["fix_type"] = fix_type_val
        if target_field.strip():
            action_kwargs["target_field"] = target_field.strip()
        if new_value.strip():
            action_kwargs["new_value"] = new_value.strip()

        action = NexusAction(**action_kwargs)
        obs, reward, done, info = await _ui_env.step(action)
        p = _build_payload(obs, _ui_env)
        p["last_reward"] = round(float(reward), 3)
        p["info"]        = info
        env_msg = info.get("message", "")
        return _fmt_status(obs, _ui_env), getattr(obs, "dirty_yaml", ""), env_msg, _safe_json(p)
    except Exception as e:
        return f"❌ Error: {e}", "", str(e), _safe_json({"error": str(e)})


async def ui_state():
    if _ui_env.current_scenario is None:
        return (
            "### 💤 Ready\nSelect a task and click **Reset** to begin.",
            "", "No active episode.", _safe_json({"has_active_task": False}),
        )
    obs = _ui_env._get_obs()
    p   = _build_payload(obs, _ui_env)
    return _fmt_status(obs, _ui_env), getattr(obs, "dirty_yaml", ""), obs.message, _safe_json(p)


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Nexus-Config-Env | Kubernetes Hardening",
    theme=gr.themes.Base(primary_hue="slate", neutral_hue="slate"),
    css="""
    body, .gradio-container { background: #0d1117 !important; color: #e6edf3 !important; }
    .gr-panel, .gr-box, .gr-form, .gap {
        background: #0d1117 !important;
    }
    .block { border: none !important; }

    /* Inputs */
    select, input, textarea, .wrap {
        background: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
    }
    label > span { color: #8b949e !important; font-size: 13px !important; font-weight: 500 !important; }

    /* Primary button (Step) */
    button.primary {
        background: #1a7f37 !important; color: #fff !important;
        border: none !important; border-radius: 6px !important;
        font-weight: 600 !important; font-size: 15px !important;
    }
    /* Secondary buttons (Reset, Get state) */
    button.secondary {
        background: #21262d !important; color: #c9d1d9 !important;
        border: 1px solid #30363d !important; border-radius: 6px !important;
        font-weight: 500 !important; font-size: 15px !important;
    }
    button:hover { opacity: 0.85 !important; }

    /* Code / JSON blocks */
    .code-wrap, .cm-editor, .code {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
    }

    /* Accordion */
    details > summary {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        color: #c9d1d9 !important;
        font-weight: 500 !important;
    }

    /* Left-panel code snippets */
    pre, code {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 5px !important;
        padding: 8px 10px !important;
        font-size: 12px !important;
        color: #79c0ff !important;
    }
    """,
) as demo:

    with gr.Row():
        # ── Left sidebar ────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### Quick Start")

            with gr.Accordion("Connect to this environment", open=True):
                gr.Markdown("""
Connect from Python using `OpenEnv`:
```python
from openenv import OpenEnv
env = OpenEnv.from_env("Wiki05/nexus-config-env")
obs, info = env.reset("task_1_easy")
```
Or connect directly to a running server:
```python
env = OpenEnv(base_url="http://localhost:7860")
```
""")

            with gr.Accordion("Contribute to this environment", open=False):
                gr.Markdown("""
Submit improvements via pull request on the Hugging Face Hub.
```bash
openenv fork Wiki05/nexus-config-env --repo-id <your-username>/nexus-config-env
```
Then make your changes and submit a pull request:
```bash
cd nexus-config-env
openenv push Wiki05/nexus-config-env --create-pr
```
""")

            with gr.Accordion("README", open=False):
                try:
                    with open("README.md", "r", encoding="utf-8") as f:
                        gr.Markdown(f.read())
                except FileNotFoundError:
                    gr.Markdown("README.md not found.")

        # ── Right panel ─────────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("## Playground\nClick **Reset** to start a new episode.")

            task_dd = gr.Dropdown(
                choices=["task_1_easy", "task_2_medium", "task_3_hard"],
                value="task_1_easy",
                label="Task ID",
                interactive=True,
            )
            action_dd = gr.Dropdown(
                choices=[
                    "scan_config", "read_telemetry", "identify_issue",
                    "propose_fix", "apply_fix", "verify_fix",
                    "escalate", "revert_change",
                ],
                value="scan_config",
                label="Action",
                interactive=True,
            )

            with gr.Row():
                fix_type_dd = gr.Dropdown(
                    choices=["none", "cost", "security", "stability"],
                    value="none", label="Fix Type", interactive=True,
                )
                target_tb = gr.Textbox(
                    label="Target Field",
                    placeholder="e.g., securityContext.privileged",
                    lines=1, max_lines=1,
                )
                value_tb = gr.Textbox(
                    label="New Value",
                    placeholder="e.g., false",
                    lines=1, max_lines=1,
                )

            reasoning_tb = gr.Textbox(
                label="Reasoning",
                placeholder="Enter reason...",
                lines=1, max_lines=1,
            )

            # ── Three buttons in one row — just like friend's UI ───────────
            with gr.Row():
                btn_step  = gr.Button("Step",      variant="primary",   scale=1)
                btn_reset = gr.Button("Reset",     variant="secondary", scale=1)
                btn_state = gr.Button("Get state", variant="secondary", scale=1)

            # ── Status ─────────────────────────────────────────────────────
            gr.Markdown("### Status")
            status_md = gr.Markdown("Select a task and click Reset to begin.")
            env_msg   = gr.Textbox(
                label="Environment Message", interactive=False, lines=3,
            )

            # ── Raw JSON collapsed ─────────────────────────────────────────
            with gr.Accordion("Raw JSON response", open=False):
                json_out = gr.Code(label="", language="json")

            # YAML kept hidden (still wired for future use)
            yaml_out = gr.Code(
                label="Current YAML State", language="yaml",
                interactive=False, visible=False,
            )

    # ── Event wiring ───────────────────────────────────────────────────────────
    btn_reset.click(
        ui_reset,
        inputs=[task_dd],
        outputs=[status_md, yaml_out, env_msg, json_out],
    )
    btn_step.click(
        ui_step,
        inputs=[action_dd, fix_type_dd, target_tb, value_tb, reasoning_tb],
        outputs=[status_md, yaml_out, env_msg, json_out],
    )
    btn_state.click(
        ui_state,
        inputs=[],
        outputs=[status_md, yaml_out, env_msg, json_out],
    )


app = gr.mount_gradio_app(app, demo, path="/web")


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")


if __name__ == "__main__":
    main()