import os
import sys
import json
import uvicorn
import gradio as gr
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Fix path for root imports to ensure Docker/HF pathing works
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.nexus_environment import NexusEnvironment
from models import NexusAction

# 1. Initialize FastAPI and Dual Environments
# _api_env is for the automated grader; _ui_env is for your browser testing.
app = FastAPI(title="Nexus-Config-Env")
_ui_env = NexusEnvironment()
_api_env = NexusEnvironment()

# --- 🚀 MANDATORY API ROUTES (Evaluator Endpoints) ---
@app.get("/health")
async def health(): 
    return {"status": "healthy"}

@app.get("/metadata")
async def metadata(): 
    return {"name": "Nexus-Config-Env", "version": "0.1.0"}

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

# --- 🎨 UI LOGIC (Adaptive Styling) ---
async def ui_reset(task_id):
    obs = await _ui_env.reset(task_id)
    return _format_md(obs), obs.dirty_yaml, json.dumps(obs.telemetry, indent=2)

async def ui_step(f_type, target, val):
    action = NexusAction(fix_type=f_type, target_field=target, new_value=val, reasoning="UI Fix")
    obs, reward, done, _ = await _ui_env.step(action)
    return _format_md(obs), obs.dirty_yaml, json.dumps(obs.telemetry, indent=2)

async def ui_get_state():
    if not _ui_env.current_scenario: return "### 💤 System Ready", "", "{}"
    obs = _ui_env._get_obs()
    return _format_md(obs), obs.dirty_yaml, json.dumps(obs.telemetry, indent=2)

def _format_md(obs):
    status = "✅ COMPLETED" if obs.done else "🚀 IN-PROGRESS"
    return f"**Score:** {obs.current_score:.2f} / 1.00  |  **Step:** {obs.step}/2  |  **Status:** {status}"

# --- 🏗️ GRADIO LAYOUT (Bootcamp Style) ---
with gr.Blocks(title="Nexus-Config-Env") as demo:
    with gr.Row():
        # SIDEBAR (LEFT)
        with gr.Column(scale=1, variant="panel"):
            with gr.Accordion("🚀 Quick Start", open=True):
                gr.Markdown("### Connect to Environment")
                gr.Markdown("Connect from Python using the `OpenEnv` SDK:")
                gr.Code(value="from openenv import OpenEnv\nenv = OpenEnv.from_env('Wiki05/nexus-config-env')", language="python")
                gr.Markdown("---")
                gr.Markdown("### Fork & Contribute")
                gr.Code(value="openenv fork Wiki05/nexus-config-env", language="shell")
            
            with gr.Accordion("📖 README", open=True):
                gr.Markdown("""
                ### 🛡️ Nexus-Config-Env
                Nexus-Config-Env addresses the critical need for automated security posture management in cloud-native infrastructures by training agents to prevent misconfigurations before deployment.
                
                **Observation Space:** Raw Kubernetes YAML configurations containing hidden vulnerabilities or cost inefficiencies.
                
                **Tasks Enrolled:**
                1. **Ghost Hunter (Easy):** Identifying and fixing legacy memory/CPU requests to optimize cluster footprint.
                2. **Security Patch (Medium):** Enforcing `runAsNonRoot` and removing `privileged` flags in container specs.
                3. **Stability Architect (Hard):** Correcting multi-layered infrastructure tags to ensure cluster high-availability.
                
                **Reward Function:**
                Dense reward signal: **0.50** for correct field identification; **1.00** for successful mitigation.
                """)
            
            gr.Markdown("---")
            gr.Markdown("Built by **Wiki05** | Meta PyTorch OpenEnv 2026")

        # MAIN PANEL (RIGHT)
        with gr.Column(scale=3):
            gr.Markdown("# 🚀 Nexus-Config-Env Playground")
            gr.Markdown("Select a scenario and click **Reset** to begin. Use **Step** to apply hardening fixes.")
            
            with gr.Group():
                task_id = gr.Dropdown(
                    choices=["task_1_easy", "task_2_medium", "task_3_hard"], 
                    label="Action (Select Hardening Task)", 
                    value="task_1_easy"
                )
                with gr.Row():
                    f_type = gr.Radio(["security", "cost"], label="Fix Category", value="security")
                    f_target = gr.Textbox(label="Field Path", placeholder="e.g., securityContext.runAsUser")
                    f_value = gr.Textbox(label="New Value", placeholder="e.g., 1000")

            # BUTTON ROW (Matches Bootcamp exactly)
            with gr.Row():
                btn_step = gr.Button("🚀 Step", variant="primary")
                btn_reset = gr.Button("🔄 Reset")
                btn_state = gr.Button("🔍 Get state")

            gr.Markdown("### Environment Status")
            status_summary = gr.Markdown("Click **Reset** to initialize the environment.")
            status_yaml = gr.Code(label="Current YAML State", language="yaml", interactive=False)
            
            with gr.Accordion("Raw JSON Response (Telemetry)", open=False):
                json_out = gr.Code(label="", language="json")

    # Event Bindings
    btn_reset.click(ui_reset, inputs=[task_id], outputs=[status_summary, status_yaml, json_out])
    btn_step.click(ui_step, inputs=[f_type, f_target, f_value], outputs=[status_summary, status_yaml, json_out])
    btn_state.click(ui_get_state, outputs=[status_summary, status_yaml, json_out])

# 5. Mount and Run
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    # Use 0.0.0.0 for Docker/HF compatibility
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()