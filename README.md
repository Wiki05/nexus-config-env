---
title: Nexus Config Env
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
base_path: /web
---

# 🛡️ Nexus-Config-Env
### **Autonomous Kubernetes Hardening via Reinforcement Learning**

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-v1.0--compliant-green)](https://github.com/scaler-school-of-technology/openenv-spec)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Wiki05/nexus-config-env)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

**Nexus-Config-Env** is a high-fidelity Reinforcement Learning (RL) environment designed to evaluate and train AI agents in the art of **Infrastructure-as-Code (IaC) Remediation**. Built on the OpenEnv specification, it challenges agents to identify and fix critical misconfigurations in Kubernetes manifests.

---

## 🌐 Live Access & Docs
* **Production Space:** [Wiki05/nexus-config-env](https://huggingface.co/spaces/Wiki05/nexus-config-env)
* **Interactive API Docs:** `https://wiki05-nexus-config-env.hf.space/docs`
* **Health Status:** `https://wiki05-nexus-config-env.hf.space/health`

---

## 💡 The Mission: Solving the $30B Cloud Leak
Misconfigured Kubernetes clusters lead to massive security breaches and resource waste. Nexus-Config-Env focuses on three "Real-World" attack vectors:

1.  **Resource Over-provisioning (Ghost RAM):** Cutting costs by aligning requested memory with actual telemetry data.
2.  **Identity & Access (Root Containers):** Enforcing the principle of least privilege by shifting from `root` to non-privileged users.
3.  **Kernel Isolation (Privileged Mode):** Closing container-escape vulnerabilities by stripping unnecessary host capabilities.

---

## 🏗️ Environment Architecture



### 1. Observation Space (`NexusObservation`)
| Attribute | Type | Impact |
| :--- | :--- | :--- |
| `config_id` | `UUID` | Unique scenario identifier for reproducibility. |
| `dirty_yaml` | `String` | The raw, vulnerable Kubernetes manifest. |
| `telemetry` | `Dict` | Real-time usage metrics (RAM, CPU, Security Flags). |
| `step` | `Integer` | Current progress within the 2-step remediation loop. |

### 2. Action Space (`NexusAction`)
Agents interact with the environment through structured JSON actions:
* `target_field`: The exact YAML path (e.g., `securityContext.privileged`).
* `new_value`: The hardened configuration value.
* `type`: Categorization (Security, Cost, or Stability).

---

## 📊 Grader Logic & Reward Shaping
We use a **Deterministic Partial Reward** system to provide granular feedback to the agent:
* **Identification (+0.50):** Awarded when the agent correctly targets the vulnerable field.
* **Remediation (+0.50):** Awarded when the agent applies the correct hardened value.
* **Max Score:** **1.00** per task.

---

## 🛠️ Developer Quick-Start

### Local Emulation
To run this on your local machine (tested on **Asus TUF F17** / Windows 11):

```bash
# Clone and Setup
git clone [https://huggingface.co/spaces/Wiki05/nexus-config-env](https://huggingface.co/spaces/Wiki05/nexus-config-env)
cd nexus-config-env
python -m venv .venv
.venv\Scripts\activate

# Install & Launch
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 7860

## Running the AI Agent

$env:HF_TOKEN="your_token"
$env:ENV_URL="[https://wiki05-nexus-config-env.hf.space](https://wiki05-nexus-config-env.hf.space)"
python inference.py

## Author
Vignesh E

GitHub: @Wiki05

LinkedIn: vignesh-e-dev