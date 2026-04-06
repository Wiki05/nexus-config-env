import os
import sys
import json
import uvicorn
import gradio as gr
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Fix path for root imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.nexus_environment import NexusEnvironment
from models import NexusAction

app = FastAPI()
_ui_env = NexusEnvironment()
_api_env = NexusEnvironment()

# --- 🚀 MANDATORY API ROUTES ---
@app.get("/health")
async def health(): return {"status": "healthy"}

@app.get("/metadata")
async def metadata(): return {"name": "Nexus-Config-Env", "version": "0.1.0"}

@app.post("/reset")
async def api_reset(task_id: str = "task_1_easy"):
    obs = await _api_env.reset(task_id)
    return JSONResponse({"observation": obs.model_dump(), "reward": 0.0, "done": False})

@app.post("/step")
async def api_step(request: Request):
    body = await request.json()
    action = NexusAction(**body)
    obs, reward, done, info = await _api_env.step(action)
    return JSONResponse({"observation": obs.model_dump(), "reward": float(reward), "done": bool(done), "info": info})

# --- 🎨 UI LOGIC (NO HARDCODED COLORS) ---
async def ui_reset(task_id):
    obs = await _ui_env.reset(task_id)
    return _format_md(obs), obs.dirty_yaml, json.dumps(obs.telemetry, indent=2)

async def ui_step(f_type, target, val):
    action = NexusAction(fix_type=f_type, target_field=target, new_value=val, reasoning="UI Fix")
    obs, reward, done, _ = await _ui_env.step(action)
    return _format_md(obs), obs.dirty_yaml, json.dumps(obs.telemetry, indent=2)

async def ui_get_state():
    if not _ui_env.current_scenario:
        return "### 💤 Ready", "", "{}"
    obs = _ui_env._get_obs()
    return _format_md(obs), obs.dirty_yaml, json.dumps(obs.telemetry, indent=2)

def _format_md(obs):
    return f"**Score:** {obs.current_score:.2f} | **Step:** {obs.step}/2 | **Status:** {'Done' if obs.done else 'Active'}"

# --- 🏗️ THE SYSTEM-DEFAULT LAYOUT ---
with gr.Blocks() as demo:
    with gr.Row():
        # SIDEBAR
        with gr.Column(scale=1, variant="panel"):
            with gr.Accordion("Quick Start", open=True):
                gr.Markdown("Connect from Python:")
                gr.Code(value="from openenv import OpenEnv\nenv = OpenEnv.from_env('Wiki05/nexus-config-env')", language="python")
                gr.Markdown("---")
                gr.Markdown("Contribute:")
                gr.Code(value="openenv fork Wiki05/nexus-config-env", language="shell")
            
            with gr.Accordion("README", open=False):
                gr.Markdown("Nexus-Config-Env: Kubernetes Hardening RL Gym.")

        # MAIN PANEL
        with gr.Column(scale=3):
            gr.Markdown("# Playground")
            gr.Markdown("Click **Reset** to start a new episode.")
            
            with gr.Group():
                task_id = gr.Dropdown(choices=["task_1_easy", "task_2_medium", "task_3_hard"], label="Action", value="task_1_easy")
                with gr.Row():
                    f_type = gr.Radio(["security", "cost"], label="Category", value="security")
                    f_target = gr.Textbox(label="Field", placeholder="securityContext.privileged")
                    f_value = gr.Textbox(label="Value", placeholder="false")

            with gr.Row():
                btn_step = gr.Button("Step", variant="primary")
                btn_reset = gr.Button("Reset")
                btn_state = gr.Button("Get state")

            gr.Markdown("### Status")
            status_summary = gr.Markdown("Click Reset to begin")
            # Using gr.Code for the YAML ensures it handles theme switching perfectly
            status_yaml = gr.Code(label="YAML Output", language="yaml", interactive=False)
            
            with gr.Accordion("Raw JSON response", open=False):
                json_out = gr.Code(label="", language="json")

    # Bindings
    btn_reset.click(ui_reset, inputs=[task_id], outputs=[status_summary, status_yaml, json_out])
    btn_step.click(ui_step, inputs=[f_type, f_target, f_value], outputs=[status_summary, status_yaml, json_out])
    btn_state.click(ui_get_state, outputs=[status_summary, status_yaml, json_out])

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)