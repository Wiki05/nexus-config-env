import os
import sys
import asyncio
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

app = FastAPI() # Title moved to metadata
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

# --- 🎨 PLAYGROUND UI LOGIC ---
async def ui_reset(task_id):
    obs = await _ui_env.reset(task_id)
    color = "#1D9E75" if obs.current_score >= 1.0 else "#EF9F27"
    html = f"<b>Step:</b> {obs.step}/2 | <b>Score:</b> <span style='color:{color}'>{obs.current_score:.2f}</span><br><pre style='background:#1e293b; color:#38bdf8; padding:10px; border-radius:5px;'>{obs.dirty_yaml}</pre>"
    return html, json.dumps(obs.telemetry, indent=2)

async def ui_step(f_type, target, val):
    action = NexusAction(fix_type=f_type, target_field=target, new_value=val, reasoning="UI Fix")
    obs, reward, done, _ = await _ui_env.step(action)
    color = "#1D9E75" if obs.current_score >= 1.0 else "#EF9F27"
    html = f"<b>Step:</b> {obs.step}/2 | <b>Score:</b> <span style='color:{color}'>{obs.current_score:.2f}</span><br><pre style='background:#1e293b; color:#38bdf8; padding:10px; border-radius:5px;'>{obs.dirty_yaml}</pre>"
    return html, json.dumps(obs.telemetry, indent=2)

# Removed 'theme' and 'title' from here to fix the Gradio 6 error
with gr.Blocks() as demo:
    gr.Markdown("# 🛡️ Nexus-Config-Env Playground")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🚀 Quick Start")
            gr.Code(value="from openenv import OpenEnv\nenv = OpenEnv.from_env('Wiki05/nexus-config-env')", language="python")
            gr.Markdown("---")
            task_select = gr.Dropdown(choices=["task_1_easy", "task_2_medium", "task_3_hard"], label="Scenario", value="task_1_easy")
            with gr.Group():
                f_type = gr.Radio(["security", "cost"], label="Category", value="security")
                f_target = gr.Textbox(label="Field Path")
                f_value = gr.Textbox(label="Fix Value")
            btn_reset = gr.Button("Reset")
            btn_step = gr.Button("Submit Step", variant="primary")
        with gr.Column(scale=2):
            status_out = gr.HTML("Select a task and click Reset to begin.")
            json_out = gr.Code(label="Telemetry", language="json")

    btn_reset.click(ui_reset, inputs=[task_select], outputs=[status_out, json_out])
    btn_step.click(ui_step, inputs=[f_type, f_target, f_value], outputs=[status_out, json_out])

# Mount with styling configuration
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    # Final check on launch parameters
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()