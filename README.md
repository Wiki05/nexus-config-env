---
title: Nexus Config Env
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# 🚀 Nexus-Config-Env
**An OpenEnv-Compliant RL Gym for Kubernetes YAML Optimization**

Nexus-Config-Env is a Reinforcement Learning (RL) environment designed to train AI agents in **Cloud Infrastructure Tuning**. It challenges agents to solve real-world problems: reducing cloud waste (Cost) and hardening security (Compliance) within Kubernetes configurations.

---

## 🟢 Live Environment Status
- **Validation:** ✅ **6/6 Passed** (Verified with `openenv-cli`)
- **API Documentation:** [https://wiki05-nexus-config-env.hf.space/docs](https://wiki05-nexus-config-env.hf.space/docs)
- **Health Check:** [https://wiki05-nexus-config-env.hf.space/health](https://wiki05-nexus-config-env.hf.space/health)

> **Note:** This is a **Headless API** environment. The root URL (`/`) returns a 404 by design. Please use the `/docs` or `/health` endpoints to verify connectivity.

---

## 🌟 Motivation (Real-World Utility)
Cloud over-provisioning and misconfigurations cost enterprises billions annually. Nexus-Config-Env provides a standardized benchmark for AI agents to:
1. **Reduce "Ghost" Resources:** Trimming unused RAM/CPU allocations based on actual telemetry data.
2. **Security Hardening:** Automatically patching privileged containers and root-user vulnerabilities.
3. **Stability Engineering:** Ensuring industry best practices (Liveness/Readiness probes) are enforced for zero-downtime.

---

## 🛠️ Environment Design

### 👁️ Observation Space (`NexusObservation`)
The agent receives a structured state containing:
- **`dirty_yaml`**: The current Kubernetes manifest requiring optimization.
- **`telemetry`**: Mocked metrics showing actual resource usage (e.g., `avg_mem_mb`).
- **`current_score`**: Real-time reward feedback on the agent's actions.

### 🕹️ Action Space (`NexusAction`)
The agent interacts with the environment using a typed schema:
- **`fix_type`**: Categories: `cost`, `security`, or `health`.
- **`target_field`**: The YAML path to modify (e.g., `resources.requests.memory`).
- **`new_value`**: The optimized value (e.g., `256Mi`).
- **`reasoning`**: Explanation of why the fix is necessary.

---

## 🏆 Tasks & Grading (15 Scenarios)
We provide 15 scenarios across three difficulty levels, evaluated by a multi-dimensional `Grader`.

| Task | Scenarios | Objective |
| :--- | :--- | :--- |
| **Ghost Hunter** | 5 | Identify and fix 10x over-provisioned RAM/CPU limits. |
| **Security Patch** | 5 | Remove `privileged: true` and enforce `runAsNonRoot`. |
| **Stability Architect** | 5 | Implement missing health probes and balance replicas. |

---

## 🚀 Getting Started & Inference

### Local Installation
```powershell
# Clone the repo
git clone [https://github.com/Wiki05/nexus-config-env.git](https://github.com/Wiki05/nexus-config-env.git)
cd nexus-config-env

# Install dependencies
pip install -r requirements.txt

### 🏃‍♂️ Running the Baseline Agent
To reproduce the baseline results, run the following in your terminal:

```powershell
# 1. Set the live URL
$env:ENV_URL="[https://wiki05-nexus-config-env.hf.space](https://wiki05-nexus-config-env.hf.space)"

# 2. Use your OWN Hugging Face token here (Do not share your real token!)
$env:HF_TOKEN="insert_your_hf_token_here"

# 3. Run the script
python inference.py

