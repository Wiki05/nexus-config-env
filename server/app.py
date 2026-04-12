# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
FastAPI + Gradio server for Nexus-Config-Env.

OpenEnv-spec endpoints:
  GET  /health     → liveness probe
  GET  /metadata   → environment description and task list
  GET  /tasks      → alias for /metadata tasks
  GET  /schema     → action + observation JSON schemas
  GET  /state      → current episode state (no side effects)
  POST /reset      → start a new episode for a given task_id
  POST /step       → submit a remediation action

Gradio UI is mounted at /web for human exploration.
"""

import sys
import json
import uvicorn
import gradio as gr
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.nexus_environment import NexusEnvironment
from models import NexusAction, NexusObservation
from scenarios import SCENARIOS

# ── App setup ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Nexus-Config-Env",
    version="0.3.0",
    description=(
        "Kubernetes YAML misconfiguration remediation environment. "
        "Trains AI agents to act as autonomous SREs."
    ),
)

# Separate env instances for UI and API to avoid state collision
_ui_env = NexusEnvironment()
_api_env = NexusEnvironment()

MAX_STEPS = 3


# ── Core API Routes (OpenEnv spec) ─────────────────────────────────────────

@app.get("/")
async def root():
    return RedirectResponse(url="/web/")


@app.get("/web")
async def web_root():
    return RedirectResponse(url="/web/")


@app.get("/health")
async def health():
    """Liveness probe — must return 200 for the space to pass validation."""
    return {"status": "healthy", "env": "nexus-config-env", "version": "0.3.0"}


@app.get("/metadata")
async def metadata():
    """Environment description and task catalogue."""
    return {
        "name": "nexus-config-env",
        "display_name": "Nexus Config Env — Kubernetes Hardening",
        "description": (
            "Autonomous Kubernetes YAML remediation environment. "
            "Trains AI agents to identify and fix cloud misconfigurations."
        ),
        "version": "0.3.0",
        "author": "Vignesh E",
        "tasks": [
            {
                "id": "task_1_easy",
                "name": "Ghost Hunter — Resource Cost Optimization",
                "difficulty": "easy",
                "max_steps": MAX_STEPS,
                "description": "Fix over-provisioned memory/CPU requests.",
            },
            {
                "id": "task_2_medium",
                "name": "Security Patch — Root Access Elimination",
                "difficulty": "medium",
                "max_steps": MAX_STEPS,
                "description": "Fix containers running as root or with unsafe filesystem access.",
            },
            {
                "id": "task_3_hard",
                "name": "Privilege Patch — Kernel Isolation Enforcement",
                "difficulty": "hard",
                "max_steps": MAX_STEPS,
                "description": "Disable privileged mode and host namespace sharing.",
            },
        ],
    }


@app.get("/tasks")
async def list_tasks():
    """List all available tasks (alias for /metadata tasks array)."""
    meta = await metadata()
    return {"tasks": meta["tasks"]}


@app.post("/reset")
async def api_reset(task_id: str = "task_1_easy"):
    """
    Start a new episode. Returns initial observation.

    Query param:
      task_id: one of task_1_easy, task_2_medium, task_3_hard
    """
    if task_id not in SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid: {list(SCENARIOS.keys())}",
        )
    obs = await _api_env.reset(task_id)
    return JSONResponse(
        {
            "observation": obs.model_dump(),
            "reward": 0.0,
            "done": False,
            "info": {"message": f"Episode started for {task_id}."},
        }
    )


@app.post("/step")
async def api_step(request: Request):
    """
    Submit a remediation action to the active episode.

    Body (JSON):
      fix_type:     "cost" | "security" | "stability"
      target_field: dot-notation YAML field path
      new_value:    hardened value as string
      reasoning:    optional explanation
    """
    if _api_env.current_scenario is None:
        return JSONResponse(
            {
                "observation": None,
                "reward": 0.0,
                "done": False,
                "info": {"warning": "No active scenario. Call POST /reset first."},
            },
            status_code=400,
        )

    # Guard: already at max steps or done
    current_obs = _api_env._get_obs()
    if bool(current_obs.done) or int(current_obs.step) >= MAX_STEPS:
        return JSONResponse(
            {
                "observation": current_obs.model_dump(),
                "reward": 0.0,
                "done": True,
                "info": {"warning": "Episode complete. Call POST /reset to start a new one."},
            }
        )

    try:
        body = await request.json()
        action = NexusAction(**body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action body: {exc}")

    obs, reward, done, info = await _api_env.step(action)

    return JSONResponse(
        {
            "observation": obs.model_dump(),
            "reward": round(float(reward), 3),
            "done": bool(done),
            "info": info,
        }
    )


@app.get("/state")
async def api_state():
    """Return current episode state without side effects."""
    if _api_env.current_scenario is None:
        return {
            "has_active_task": False,
            "task_id": None,
            "step": 0,
            "observation": None,
        }
    obs = _api_env._get_obs()
    return {
        "has_active_task": True,
        "task_id": getattr(_api_env, "current_task_id", None),
        "step": min(int(obs.step), MAX_STEPS),
        "observation": obs.model_dump(),
    }


@app.get("/schema")
async def api_schema():
    """Return JSON schemas for action and observation types."""
    return {
        "action": NexusAction.model_json_schema(),
        "observation": NexusObservation.model_json_schema(),
    }


# ── Gradio UI helpers ──────────────────────────────────────────────────────

def _safe_json(data) -> str:
    try:
        return json.dumps(data, indent=2, default=str)
    except Exception:
        return "{}"


def _format_status(obs) -> str:
    display_step = min(int(getattr(obs, "step", 0)), MAX_STEPS)
    score = float(getattr(obs, "current_score", 0.0))
    done = bool(getattr(obs, "done", False)) or display_step >= MAX_STEPS
    status_icon = "✅ COMPLETED" if done else "🔄 IN-PROGRESS"
    return (
        f"**Score:** {score:.2f} / 1.00"
        f" | **Step:** {display_step}/{MAX_STEPS}"
        f" | **Status:** {status_icon}"
    )


def _build_state_payload(obs, env_instance):
    task_id = getattr(env_instance, "current_task_id", None)
    step = min(int(getattr(obs, "step", 0)), MAX_STEPS)
    done = bool(getattr(obs, "done", False)) or step >= MAX_STEPS
    return {
        "has_active_task": env_instance.current_scenario is not None,
        "task_id": task_id,
        "step": step,
        "done": done,
        "current_score": float(getattr(obs, "current_score", 0.0)),
        "telemetry": getattr(obs, "telemetry", {}),
    }


async def ui_reset(task_id):
    try:
        obs = await _ui_env.reset(task_id)
        payload = _build_state_payload(obs, _ui_env)
        return _format_status(obs), getattr(obs, "dirty_yaml", ""), _safe_json(payload)
    except Exception as e:
        return f"### ❌ Error\n{e}", "", _safe_json({"error": str(e)})


async def ui_step(f_type, target, val):
    if _ui_env.current_scenario is None:
        return (
            "### ⚠️ No active episode\nClick **Reset** to start.",
            "",
            _safe_json({"warning": "Call Reset first."}),
        )

    current_obs = _ui_env._get_obs()
    if bool(current_obs.done) or int(current_obs.step) >= MAX_STEPS:
        payload = _build_state_payload(current_obs, _ui_env)
        payload["info"] = {"warning": "Episode complete. Click Reset to try again."}
        return (
            "### ✅ Episode complete — Click Reset to try again.",
            getattr(current_obs, "dirty_yaml", ""),
            _safe_json(payload),
        )

    try:
        action = NexusAction(
            fix_type=f_type,
            target_field=target,
            new_value=val,
            reasoning="UI interaction",
        )
        obs, reward, done, info = await _ui_env.step(action)
        payload = _build_state_payload(obs, _ui_env)
        payload["last_reward"] = round(float(reward), 3)
        payload["info"] = info
        return _format_status(obs), getattr(obs, "dirty_yaml", ""), _safe_json(payload)
    except Exception as e:
        return f"### ❌ Error\n{e}", "", _safe_json({"error": str(e)})


async def ui_get_state():
    if _ui_env.current_scenario is None:
        return (
            "### 💤 Ready\nSelect a task and click Reset to begin.",
            "",
            _safe_json({
                "has_active_task": False,
                "task_id": None,
                "step": 0,
                "done": False,
                "current_score": 0.0,
            }),
        )
    obs = _ui_env._get_obs()
    payload = _build_state_payload(obs, _ui_env)
    return _format_status(obs), getattr(obs, "dirty_yaml", ""), _safe_json(payload)


# ── Gradio UI ──────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Nexus-Config-Env | Kubernetes Hardening",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
    ),
) as demo:

    gr.HTML("""
    <style>
        /* ── Global ────────────────────────────── */
        body, .gradio-container {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%) !important;
            color: #e6edf3 !important;
        }
        /* ── Inputs ─────────────────────────────── */
        .gradio-container input[type="text"],
        .gradio-container textarea,
        .gradio-container select {
            background: #1c2128 !important;
            color: #e6edf3 !important;
            border: 1px solid #30363d !important;
            border-radius: 6px !important;
        }
        /* ── Buttons ─────────────────────────────── */
        .gr-button-primary {
            background: linear-gradient(90deg, #1f6feb, #388bfd) !important;
            border: none !important;
            color: #fff !important;
        }
        .gr-button-secondary {
            background: #21262d !important;
            border: 1px solid #30363d !important;
            color: #e6edf3 !important;
        }
        /* ── Hero banner ──────────────────────── */
        .hero {
            background: linear-gradient(135deg, #1f2d45, #0d1117);
            border: 1px solid #1f6feb;
            border-radius: 12px;
            padding: 28px 36px;
            margin-bottom: 16px;
        }
        .hero h1 { color: #58a6ff; margin: 0 0 8px; font-size: 2rem; }
        .hero p  { color: #8b949e; margin: 0; font-size: 0.95rem; }
        /* ── Spinner for number inputs ─────────── */
        input::-webkit-outer-spin-button,
        input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
        input[type=number] { -moz-appearance: textfield; }
    </style>
    <div class="hero">
        <h1>🛡️ Nexus-Config-Env</h1>
        <p>
            <strong>Autonomous Kubernetes Hardening via Reinforcement Learning</strong><br>
            Train AI agents to detect &amp; fix K8s misconfigurations — security vulnerabilities,
            over-provisioned resources, and policy violations. Built for the
            <strong>Meta × Scaler OpenEnv Hackathon 2026</strong>.
        </p>
    </div>
    """)

    with gr.Row():
        # ── Left panel: docs ───────────────────────────────────────────────
        with gr.Column(scale=1, variant="panel"):
            with gr.Accordion("📖 Quick Start Guide", open=False):
                gr.Markdown("""
## Quick Start

### What this environment does
Simulates real Kubernetes hardening scenarios. The AI agent must:
1. Inspect a broken YAML config
2. Read telemetry (memory, CPU, CVE scores)
3. Identify the misconfigured field
4. Apply the correct fix within the step budget

### How to use the Playground
1. Select a **Task** from the dropdown
2. Click **Reset** to load the scenario
3. Inspect the **Current YAML State** and **Telemetry**
4. Enter your fix: Category, Field Path, New Value
5. Click **Step** to apply
6. Check Score in the **Status** bar

### Task IDs
| Task ID | Name | Difficulty |
|---------|------|------------|
| `task_1_easy` | Ghost Hunter | Easy |
| `task_2_medium` | Security Patch | Medium |
| `task_3_hard` | Privilege Patch | Hard |

### Fix categories
| Value | When to use |
|-------|-------------|
| `cost` | Memory/CPU over-provisioning |
| `security` | Root access, privileged mode, writable FS |
| `stability` | Reliability / availability issues |

### Example fixes
**Easy (Ghost RAM):**
- Category: `cost`
- Field: `resources.requests.memory`
- Value: `256Mi`

**Medium (Root user):**
- Category: `security`
- Field: `securityContext.runAsUser`
- Value: `1000`

**Hard (Privileged mode):**
- Category: `security`
- Field: `securityContext.privileged`
- Value: `false`

### API Endpoints
```
GET  /health
GET  /metadata
GET  /tasks
GET  /schema
GET  /state
POST /reset?task_id=<id>
POST /step  (JSON body)
```

### Agent quickstart
```python
from openenv import OpenEnv
env = OpenEnv.from_env('Wiki05/nexus-config-env')
```
                """)

            with gr.Accordion("🏆 Reward Logic", open=True):
                gr.Markdown("""
## Grader: Deterministic Partial Credit

| Component | Points | Condition |
|-----------|--------|-----------|
| **Category** | +0.20 | Correct `fix_type` |
| **Field ID** | +0.40 | Exact `target_field` match |
| **Field partial** | +0.20 | Field contains the right key |
| **Value fix** | +0.40 | Correct `new_value` (field must also match) |
| **Max score** | **1.00** | All three correct |

This multi-layer scoring gives the AI agent a learning gradient —
it can see partial progress towards the goal, enabling effective RL training.
                """)

            gr.Markdown("---")
            gr.Markdown(
                "Built by **[Wiki05 (Vignesh E)](https://github.com/Wiki05)** "
                "| Meta × Scaler OpenEnv Hackathon 2026"
            )

        # ── Right panel: playground ────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("## 🎮 Kubernetes Hardening Playground")

            with gr.Group():
                task_id = gr.Dropdown(
                    choices=["task_1_easy", "task_2_medium", "task_3_hard"],
                    label="Hardening Task",
                    value="task_1_easy",
                    interactive=True,
                )

                f_type = gr.Radio(
                    ["cost", "security", "stability"],
                    label="Fix Category",
                    value="security",
                    interactive=True,
                )

                with gr.Row():
                    f_target = gr.Textbox(
                        label="Field Path (dot notation)",
                        placeholder="e.g., securityContext.runAsUser",
                        lines=1,
                        max_lines=1,
                    )
                    f_value = gr.Textbox(
                        label="New Value",
                        placeholder="e.g., 1000",
                        lines=1,
                        max_lines=1,
                    )

            with gr.Row():
                btn_reset = gr.Button("🔄 Reset", variant="primary")
                btn_step = gr.Button("▶️ Step", variant="secondary")
                btn_state = gr.Button("📊 Get State", variant="secondary")

            # Status bar
            gr.Markdown("### 📈 Status")
            status_summary = gr.Markdown("Select a task and click Reset to begin.")

            # YAML viewer
            status_yaml = gr.Code(
                label="Current YAML State",
                language="yaml",
                interactive=False,
            )

            # JSON telemetry
            with gr.Accordion("🔍 Raw JSON Response (Telemetry / State)", open=False):
                json_out = gr.Code(label="", language="json")

    # ── Event bindings ─────────────────────────────────────────────────────
    btn_reset.click(
        ui_reset,
        inputs=[task_id],
        outputs=[status_summary, status_yaml, json_out],
    )
    btn_step.click(
        ui_step,
        inputs=[f_type, f_target, f_value],
        outputs=[status_summary, status_yaml, json_out],
    )
    btn_state.click(
        ui_get_state,
        inputs=[],
        outputs=[status_summary, status_yaml, json_out],
    )


# Mount Gradio at /web
app = gr.mount_gradio_app(app, demo, path="/web")


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")


if __name__ == "__main__":
    main()