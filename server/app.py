import os
import sys
import json
import asyncio
import uvicorn
import gradio as gr
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Ensure root imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.nexus_environment import NexusEnvironment
from models import NexusAction

# 1. Initialize FastAPI and Environments
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

# --- 🎨 UI LOGIC ---
async def ui_reset(task_id):
    obs = await _ui_env.reset(task_id)
    return _format_status(obs), json.dumps(obs.telemetry, indent=2)

async def ui_step(f_type, target, val):
    action = NexusAction(fix_type=f_type, target_field=target, new_value=val, reasoning="UI Fix")
    obs, reward, done, _ = await _ui_env.step(action)
    return _format_status(obs), json.dumps(obs.telemetry, indent=2)

async def ui_get_state():
    if not _ui_env.current_scenario:
        return "<div style='text-align:center; padding:20px; color:#94a3b8'>Click Reset to begin.</div>", "{}"
    obs = _ui_env._get_obs()
    return _format_status(obs), json.dumps(obs.telemetry, indent=2)

def _format_status(obs):
    color = "#4F46E5" if obs.current_score >= 1.0 else "#EF9F27" if obs.current_score >= 0.4 else "#E24B4A"
    return f"""
<div style='font-family:system-ui; padding:15px; background:#ffffff; border-radius:10px; border: 1px solid #e2e8f0; border-left: 5px solid {color}'>
    <div style='display:flex; justify-content:space-between; margin-bottom:10px;'>
        <b style='color:#64748b'>Status: {'✅ DONE' if obs.done else '🚀 ACTIVE'}</b>
        <b style='color:{color}'>Score: {obs.current_score:.2f} / 1.00</b>
    </div>
    <div style='font-size:12px; color:#94a3b8; margin-bottom:10px;'>Step {obs.step} of 2</div>
    <hr style='border:0; border-top:1px solid #f1f5f9; margin-bottom:10px;'>
    <pre style='background:#f8fafc; padding:10px; border-radius:5px; color:#1e293b; font-size:13px; border: 1px solid #f1f5f9'>{obs.dirty_yaml}</pre>
</div>"""

# --- 🏗️ GRADIO LAYOUT (MATCHES SCREENSHOT 1) ---
demo = gr.Blocks()
# Fixed: Apply theme as a property to avoid the TypeError in Gradio 6.0
demo.theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")

with demo:
    with gr.Row():
        # LEFT SIDEBAR
        with gr.Column(scale=1, variant="panel"):
            with gr.Accordion("🚀 Quick Start", open=True):
                gr.Markdown("Connect from Python using `OpenEnv`:")
                gr.Code(value="from openenv import OpenEnv\nenv = OpenEnv.from_env('Wiki05/nexus-config-env')", language="python")
                
                gr.Markdown("---")
                gr.Markdown("Contribute to this project:")
                gr.Code(value="openenv fork Wiki05/nexus-config-env", language="shell")
            
            with gr.Accordion("📖 README", open=False):
                gr.Markdown("### Nexus-Config-Env\nAn RL gym for Kubernetes hardening.")

        # MAIN PANEL
        with gr.Column(scale=3):
            gr.Markdown("# 🛡️ Nexus-Config-Env Playground")
            
            with gr.Tabs():
                with gr.TabItem("Playground"):
                    with gr.Group():
                        task_id = gr.Dropdown(
                            choices=["task_1_easy", "task_2_medium", "task_3_hard"], 
                            label="Action (Select Task Scenario)", 
                            value="task_1_easy"
                        )
                        with gr.Row():
                            f_type = gr.Radio(["security", "cost"], label="Fix Category", value="security")
                            f_target = gr.Textbox(label="Target Field Path", placeholder="e.g. securityContext.privileged")
                            f_value = gr.Textbox(label="Fix Value", placeholder="e.g. false")

                    # Buttons in one row like Screenshot 1
                    with gr.Row():
                        btn_step = gr.Button("🚀 Step", variant="primary")
                        btn_reset = gr.Button("🔄 Reset")
                        btn_state = gr.Button("🔍 Get state")

                    gr.Markdown("### Status")
                    status_out = gr.HTML("<div style='padding:20px; text-align:center; color:#94a3b8'>Click Reset to begin</div>")
                    
                    with gr.Accordion("Raw JSON response", open=False):
                        json_out = gr.Code(label="", language="json")

    btn_reset.click(ui_reset, inputs=[task_id], outputs=[status_out, json_out])
    btn_step.click(ui_step, inputs=[f_type, f_target, f_value], outputs=[status_out, json_out])
    btn_state.click(ui_get_state, outputs=[status_out, json_out])

app = gr.mount_gradio_app(app, demo, path="/")

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()