---
title: Nexus Config Env
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🛡️ Nexus-Config-Env
### **An OpenEnv-Compliant RL Environment for Kubernetes Configuration Hardening**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Wiki05/nexus-config-env)
[![OpenAPI Docs](https://img.shields.io/badge/API-Documentation-green)](https://wiki05-nexus-config-env.hf.space/docs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Nexus-Config-Env** is a professional-grade Reinforcement Learning environment built on the **OpenEnv** specification. It evaluates AI agents on real-world Kubernetes remediation tasks: fixing security vulnerabilities, eliminating resource waste, and enforcing infrastructure best practices.

---

## 🚀 Live Environment
* **Space URL:** [huggingface.co/spaces/Wiki05/nexus-config-env](https://huggingface.co/spaces/Wiki05/nexus-config-env)
* **API Root:** `https://wiki05-nexus-config-env.hf.space`
* **Interactive Playground:** [Playground](https://wiki05-nexus-config-env.hf.space)

---

## 💡 The Problem
Cloud misconfiguration is a **$30 billion** annual problem. Nexus-Config-Env targets the three most common failures:
1.  **Ghost RAM:** Reducing over-provisioned memory to save costs.
2.  **Root Access:** Remediating `runAsUser: 0` security risks.
3.  **Privileged Containers:** Eliminating unrestricted host access.

---

## 🛠️ Technical Specification

### Observation Space (`NexusObservation`)
| Field | Type | Description |
| :--- | :--- | :--- |
| `config_id` | `str` | Unique ID for the current scenario. |
| `dirty_yaml` | `str` | The Kubernetes manifest requiring fixing. |
| `telemetry` | `dict` | Simulated runtime metrics (e.g., actual vs requested RAM). |
| `current_score` | `float` | Cumulative reward for the episode. |
| `done` | `bool` | Episode termination flag. |

### Action Space (`NexusAction`)
| Field | Type | Description |
| :--- | :--- | :--- |
| `target_field` | `str` | The YAML path to modify (e.g., `resources.requests.memory`). |
| `new_value` | `str` | The corrected value (e.g., `256Mi`). |
| `reasoning` | `str` | The agent's justification for the fix. |

---

## 📊 Evaluation & Tasks
| Task ID | Name | Objective | Max Steps |
| :--- | :--- | :--- | :---: |
| `task_1_easy` | Ghost Hunter | Reduce RAM from 8Gi to 256Mi | 2 |
| `task_2_medium` | Security Patch | Change `runAsUser` from 0 to 1000 | 2 |
| `task_3_hard` | Privilege Patch | Set `privileged` to `false` | 2 |

**Grading:** 0.50 for field identification + 0.50 for correct remediation. **Deterministic scoring.**

---

## 🏁 Quick Start

### 1. Local Setup
```bash
git clone [https://huggingface.co/spaces/Wiki05/nexus-config-env](https://huggingface.co/spaces/Wiki05/nexus-config-env)
cd nexus-config-env
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 7860


Run Baseline Agent

$env:ENV_URL="[http://127.0.0.1:7860](http://127.0.0.1:7860)"
$env:HF_TOKEN="your_token"
python inference.py


Project Structure

nexus-config-env/
├── server/
│   ├── nexus_environment.py  # Core RL Logic
│   └── app.py                # FastAPI Routes
├── Dockerfile                # Deployment Config
├── inference.py              # Grader Client (OpenAI-based)
├── openenv.yaml              # OpenEnv Metadata
└── README.md                 # Documentation


Author: Vignesh E

Built for: Meta x Scaler OpenEnv Hackathon 2026