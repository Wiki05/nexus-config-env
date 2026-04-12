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
import uvicorn
import gradio as gr
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.nexus_environment import NexusEnvironment
from models import NexusAction, NexusObservation
from tasks import SCENARIOS, TASKS

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
        body   = await request.json()
        action = NexusAction(**body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action: {exc}")

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

def _fmt_status(obs) -> str:
    step  = min(int(getattr(obs, "step", 0)), MAX_STEPS)
    score = float(getattr(obs, "current_score", 0.0))
    done  = bool(getattr(obs, "done", False)) or step >= MAX_STEPS
    icon  = "✅ COMPLETE" if done else "🔄 IN-PROGRESS"
    bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    return (
        f"**Score:** {score:.3f} / 1.000 `{bar}`\n\n"
        f"**Step:** {step}/{MAX_STEPS} | **Status:** {icon}"
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

    cur = _ui_env._get_obs()
    if cur.done or cur.step >= MAX_STEPS:
        p = _build_payload(cur, _ui_env)
        return (
            "### ✅ Episode complete — click **Reset** to start again.",
            getattr(cur, "dirty_yaml", ""),
            "Episode already ended.",
            _safe_json(p),
        )

    try:
        # Build action — optional fields only if non-empty
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
        return _fmt_status(obs), getattr(obs, "dirty_yaml", ""), env_msg, _safe_json(p)
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
    return _fmt_status(obs), getattr(obs, "dirty_yaml", ""), obs.message, _safe_json(p)


# ── Gradio UI ──────────────────────────────────────────────────────────────────

ACTION_SPACE_HTML = """
<style>
  .as-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
  .as-table th { background:#1f2d45; color:#58a6ff; padding:8px 10px; text-align:left; }
  .as-table td { padding:7px 10px; border-bottom:1px solid #21262d; color:#c9d1d9; }
  .as-table tr:hover td { background:#161b22; }
  .rew-pos  { color:#3fb950; font-weight:700; }
  .rew-neg  { color:#f85149; font-weight:700; }
  .rew-cond { color:#d29922; font-weight:700; }
  .badge    { display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; }
  .badge-t  { background:#1f2d45; color:#58a6ff; }
  .badge-f  { background:#21262d; color:#8b949e; }
</style>
<table class='as-table'>
  <thead><tr>
    <th>Action</th><th>Reward</th><th>Terminal</th><th>Description</th>
  </tr></thead>
  <tbody>
    <tr><td><code>scan_config</code></td>    <td class='rew-pos'>+0.10</td><td><span class='badge badge-f'>No</span></td> <td>Analyse YAML structure for misconfigurations</td></tr>
    <tr><td><code>read_telemetry</code></td> <td class='rew-pos'>+0.10</td><td><span class='badge badge-f'>No</span></td> <td>Read runtime CPU, RAM, CVE score signals</td></tr>
    <tr><td><code>identify_issue</code></td> <td class='rew-pos'>+0.20</td><td><span class='badge badge-f'>No</span></td> <td>Classify root cause (cost / security / stability)</td></tr>
    <tr><td><code>propose_fix</code></td>    <td class='rew-pos'>+0.15</td><td><span class='badge badge-f'>No</span></td> <td>Plan field change without executing</td></tr>
    <tr><td><code>apply_fix</code></td>      <td class='rew-cond'>±0.50/−0.30</td><td><span class='badge badge-f'>No</span></td><td>Execute remediation — +0.50 correct, −0.30 wrong</td></tr>
    <tr><td><code>verify_fix</code></td>     <td class='rew-pos'>+0.20</td><td><span class='badge badge-t'>Yes ✓</span></td><td>Confirm fix applied — ends episode</td></tr>
    <tr><td><code>escalate</code></td>       <td class='rew-pos'>+0.05</td><td><span class='badge badge-t'>Yes ✓</span></td><td>Hand off to human SRE — ends episode</td></tr>
    <tr><td><code>revert_change</code></td>  <td class='rew-neg'>−0.10</td><td><span class='badge badge-f'>No</span></td><td>Undo last change (use only if wrong fix applied)</td></tr>
  </tbody>
</table>
"""

GRADER_HTML = """
<style>
  .gr-table { width:100%; border-collapse:collapse; font-size:0.85rem; margin-top:4px; }
  .gr-table th { background:#1a2d1a; color:#3fb950; padding:7px 10px; text-align:left; }
  .gr-table td { padding:7px 10px; border-bottom:1px solid #21262d; color:#c9d1d9; }
</style>
<table class='gr-table'>
  <thead><tr><th>Dimension</th><th>Weight</th><th>Criteria</th></tr></thead>
  <tbody>
    <tr><td><strong>Protocol</strong></td>    <td>20%</td><td>scan_config / read_telemetry used before apply_fix?</td></tr>
    <tr><td><strong>Diagnosis</strong></td>   <td>25%</td><td>Correct issue category + exact field identified?</td></tr>
    <tr><td><strong>Remediation</strong></td> <td>40%</td><td>Exact target_field + new_value applied?</td></tr>
    <tr><td><strong>Efficiency</strong></td>  <td>15%</td><td>Resolved within 50% of step budget?</td></tr>
  </tbody>
</table>
"""

HERO_HTML = """
<style>
  body, .gradio-container { background: #0d1117 !important; color: #e6edf3 !important; }
  .hero-banner {
    background: linear-gradient(135deg, #0d1930 0%, #0d1117 60%, #1a0d30 100%);
    border: 1px solid #1f6feb44;
    border-radius: 14px;
    padding: 28px 36px 22px;
    margin-bottom: 16px;
  }
  .hero-banner h1 { color: #58a6ff; font-size: 2rem; margin: 0 0 6px; }
  .hero-banner p  { color: #8b949e; margin: 0; font-size: 0.9rem; line-height: 1.6; }
  .hero-banner .badges { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
  .hero-banner .badge-pill {
    background: #21262d; border: 1px solid #30363d; border-radius: 20px;
    color: #e6edf3; font-size: 0.78rem; padding: 3px 12px;
  }
  .gradio-container input[type="text"], .gradio-container textarea {
    background: #161b22 !important; color: #e6edf3 !important;
    border: 1px solid #30363d !important; border-radius: 6px !important;
  }
  .gr-button { border-radius: 8px !important; font-weight: 600 !important; }
  input::-webkit-outer-spin-button, input::-webkit-inner-spin-button { -webkit-appearance:none; margin:0; }
  input[type=number] { -moz-appearance:textfield; }
</style>
<div class="hero-banner">
  <h1>🛡️ Nexus-Config-Env</h1>
  <p>
    <strong>Autonomous Kubernetes Hardening via Reinforcement Learning</strong><br>
    Train AI agents to act as SREs — detect security vulnerabilities,
    rightsize over-provisioned resources, and enforce policy compliance
    through an 8-action discrete workflow with multi-criteria scoring.
  </p>
  <div class="badges">
    <span class="badge-pill">🏆 Meta × Scaler OpenEnv 2026</span>
    <span class="badge-pill">⚡ 8-Action Discrete Space</span>
    <span class="badge-pill">📊 Multi-Criteria Grader</span>
    <span class="badge-pill">🔒 K8s Security Hardening</span>
    <span class="badge-pill">💰 Cloud Cost Optimisation</span>
  </div>
</div>
"""

with gr.Blocks(
    title="Nexus-Config-Env | Kubernetes Hardening",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
    ),
) as demo:

    gr.HTML(HERO_HTML)

    with gr.Row():
        # ── Left panel ─────────────────────────────────────────────────────
        with gr.Column(scale=1, variant="panel"):

            with gr.Accordion("⚡ Action Space", open=True):
                gr.HTML(ACTION_SPACE_HTML)

            with gr.Accordion("📊 Grader Logic", open=True):
                gr.HTML(GRADER_HTML)
                gr.Markdown("""
> **Deterministic + Varied**: Each episode produces a different score based on
> the agent's trajectory. Grader never returns a constant value.
> Score range: **(0.001, 0.999)** strictly.
                """)

            with gr.Accordion("📖 Quick Start", open=False):
                gr.Markdown("""
## Optimal SRE Workflow

For best score, follow this protocol every episode:

```
1. scan_config       → study the broken YAML
2. read_telemetry    → check runtime signals
3. identify_issue    → classify: cost / security
4. propose_fix       → confirm the target field
5. apply_fix         → set the hardened value
6. verify_fix        → confirm and complete
```

## Task Examples

**Easy — Ghost RAM (cost):**
```json
{"action_type":"apply_fix",
 "fix_type":"cost",
 "target_field":"resources.requests.memory",
 "new_value":"256Mi"}
```

**Medium — Root User (security):**
```json
{"action_type":"apply_fix",
 "fix_type":"security",
 "target_field":"securityContext.runAsUser",
 "new_value":"1000"}
```

**Hard — Privileged Mode (security):**
```json
{"action_type":"apply_fix",
 "fix_type":"security",
 "target_field":"securityContext.privileged",
 "new_value":"false"}
```

## API Endpoints
```
GET  /health
GET  /metadata
GET  /tasks
GET  /schema
GET  /state
POST /reset?task_id=<id>
POST /step  (JSON body)
```

## Agent Quickstart
```python
from openenv import OpenEnv
env = OpenEnv.from_env('Wiki05/nexus-config-env')
obs, _ = env.reset('task_1_easy')
```
                """)

            with gr.Accordion("🏗️ Architecture", open=False):
                gr.Markdown("""
## Environment Design

```
AI Agent (any LLM via OpenAI client)
        │
        ├── POST /reset?task_id=X
        │       → NexusObservation (dirty YAML + telemetry)
        │
        ├── POST /step  (NexusAction JSON)
        │       → reward + updated observation
        │
        └── GET  /state
                → read-only episode snapshot
```

### 4-Criteria Grader

Each episode is scored across 4 independent dimensions:

| Dimension | Max | What it measures |
|:----------|----:|:-----------------|
| Protocol  | 0.20 | Scanned before fixing? |
| Diagnosis | 0.25 | Correct category + field? |
| Remediation | 0.40 | Exact field + value fixed? |
| Efficiency | 0.15 | Completed within step budget? |
                """)

            gr.Markdown("---")
            gr.Markdown(
                "Built by **[Wiki05 (Vignesh E)](https://github.com/Wiki05)** "
                "| Meta × Scaler OpenEnv 2026"
            )

        # ── Right panel: playground ─────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("## 🎮 SRE Remediation Playground")

            with gr.Group():
                task_dd = gr.Dropdown(
                    choices=["task_1_easy", "task_2_medium", "task_3_hard"],
                    value="task_1_easy",
                    label="Hardening Task",
                    interactive=True,
                )
                action_dd = gr.Dropdown(
                    choices=[
                        "scan_config",
                        "read_telemetry",
                        "identify_issue",
                        "propose_fix",
                        "apply_fix",
                        "verify_fix",
                        "escalate",
                        "revert_change",
                    ],
                    value="scan_config",
                    label="Action Type",
                    interactive=True,
                )
                fix_type_dd = gr.Dropdown(
                    choices=["none", "cost", "security", "stability"],
                    value="none",
                    label="Fix Type (for identify_issue / apply_fix)",
                    interactive=True,
                )
                with gr.Row():
                    target_tb = gr.Textbox(
                        label="Target Field (dot notation)",
                        placeholder="e.g., securityContext.privileged",
                        lines=1, max_lines=1,
                    )
                    value_tb = gr.Textbox(
                        label="New Value",
                        placeholder="e.g., false",
                        lines=1, max_lines=1,
                    )
                reasoning_tb = gr.Textbox(
                    label="Reasoning (optional)",
                    placeholder="Why are you taking this action?",
                    lines=1, max_lines=1,
                )

            with gr.Row():
                btn_reset  = gr.Button("🔄 Reset",     variant="primary")
                btn_step   = gr.Button("▶️ Step",      variant="secondary")
                btn_state  = gr.Button("📊 Get State", variant="secondary")

            gr.Markdown("### 📈 Episode Status")
            status_md = gr.Markdown("Select a task and click Reset to begin.")
            env_msg   = gr.Textbox(label="Environment Message", interactive=False, lines=2)
            yaml_out  = gr.Code(label="Current YAML State", language="yaml", interactive=False)

            with gr.Accordion("🔍 Raw JSON Response", open=False):
                json_out = gr.Code(label="", language="json")

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