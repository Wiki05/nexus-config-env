---
title: Nexus Config Env
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# 🛡️ Nexus-Config-Env
### **Autonomous Kubernetes Hardening via Reinforcement Learning**

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-v1.0--compliant-brightgreen)](https://github.com/meta-pytorch/OpenEnv)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Wiki05/nexus-config-env)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

**Nexus-Config-Env** is a high-fidelity Reinforcement Learning environment for **autonomous Kubernetes YAML misconfiguration remediation**. Built on the OpenEnv specification, it challenges AI agents to inspect broken cloud infrastructure configurations, interpret real-time telemetry, and apply surgical fixes — all through a structured, standardized API.

The environment simulates the daily work of a **Site Reliability Engineer (SRE)**: detecting ghost RAM allocations, eliminating root containers, closing container-escape vulnerabilities, and enforcing organizational security policies.

---

## 🌐 Live Access

| Resource | URL |
|:---------|:----|
| **Space** | [Wiki05/nexus-config-env](https://huggingface.co/spaces/Wiki05/nexus-config-env) |
| **Playground** | `https://wiki05-nexus-config-env.hf.space/web` |
| **API Docs** | `https://wiki05-nexus-config-env.hf.space/docs` |
| **Health** | `https://wiki05-nexus-config-env.hf.space/health` |

---

## 💡 Real-World Problem: The $30B Cloud Misconfiguration Crisis

Misconfigured Kubernetes clusters are responsible for billions in unnecessary cloud spend and are the root cause of the most severe security breaches. This environment targets three critical attack vectors directly from production SRE playbooks:

| # | Problem | Real Impact |
|---|---------|-------------|
| 1 | **Ghost RAM** — Apps request 32× more memory than needed | $47–$300/month per pod in wasted compute |
| 2 | **Root Containers** — Processes run as UID 0 | Full node compromise if container is breached |
| 3 | **Privileged Mode** — Containers with host kernel access | Container escape → full cluster takeover |

---

## 🏗️ Environment Architecture

```
┌─────────────────────────────────────────────────────┐
│  AI Agent (GPT-4 / Llama / Nemotron / Any LLM)     │
│                                                     │
│  POST /reset  ←──── start episode                  │
│  POST /step   ←──── apply action                   │
│  GET  /state  ←──── inspect progress               │
└──────────────────┬──────────────────────────────────┘
                   │  OpenEnv HTTP API
┌──────────────────▼──────────────────────────────────┐
│  Nexus-Config-Env (FastAPI + Gradio)                │
│                                                     │
│  Observation:  dirty YAML + telemetry signals       │
│  Action:       fix_type + target_field + new_value  │
│  Reward:       deterministic partial-credit grader  │
└─────────────────────────────────────────────────────┘
```

### Observation Space (`NexusObservation`)

| Field | Type | Description |
|:------|:-----|:------------|
| `config_id` | `str` | Unique scenario ID for reproducibility |
| `dirty_yaml` | `str` | The misconfigured Kubernetes manifest |
| `telemetry` | `dict` | Memory, CPU, CVE scores, security flags |
| `fixes_applied` | `list[str]` | Fields corrected so far |
| `current_score` | `float` | Cumulative episode score (0–1) |
| `step` | `int` | Current step in the episode |
| `done` | `bool` | True when episode is complete |

### Action Space (`NexusAction`)

```json
{
  "fix_type":     "cost | security | stability",
  "target_field": "dot-notation YAML path",
  "new_value":    "hardened value",
  "reasoning":    "brief explanation"
}
```

---

## 📊 Grader Logic & Reward Shaping

A **Deterministic Partial Credit** grader provides meaningful learning signal at every step:

| Component | Points | Condition |
|:----------|:-------|:----------|
| **Category Match** | +0.20 | Correct `fix_type` (cost/security/stability) |
| **Field Identification (exact)** | +0.40 | Exact `target_field` match |
| **Field Identification (partial)** | +0.20 | Field is in the right neighborhood |
| **Value Correction** | +0.40 | Correct `new_value` AND exact field match |
| **Max Score** | **1.00** | All three components correct |

This gradient enables efficient RL training — the agent receives partial credit for being "close", rather than a binary pass/fail signal.

Each task also has a **dedicated grader function** that validates the full episode trajectory for robust evaluation.

---

## 🎯 Tasks

### `task_1_easy` — Ghost Hunter (Cost Optimization)

**Scenario:** An API service requests 8 GB of RAM but uses only ~100 MB on average.

```yaml
# BROKEN CONFIG
resources:
  requests:
    memory: '8Gi'
  limits:
    memory: '16Gi'
```

**Telemetry:** `avg_mem_mb: 100, peak_mem_mb: 200, waste_estimate_usd_month: 47.20`

**Optimal Fix:** Set `resources.requests.memory` → `256Mi`

---

### `task_2_medium` — Security Patch (Root Access)

**Scenario:** The backend API container is running as root (UID 0) with privilege escalation enabled.

```yaml
# BROKEN CONFIG
securityContext:
  runAsUser: 0
  allowPrivilegeEscalation: true
```

**Telemetry:** `is_root: true, cve_risk_score: 9.1`

**Optimal Fix:** Set `securityContext.runAsUser` → `1000`

---

### `task_3_hard` — Privilege Patch (Container Escape Prevention)

**Scenario:** An infra-agent in `kube-system` with `privileged: true` and `hostPID: true` — a critical container-escape risk.

```yaml
# BROKEN CONFIG
securityContext:
  privileged: true
  runAsUser: 0
hostPID: true
```

**Telemetry:** `privileged_status: true, escape_risk: CRITICAL, cve_ids: [CVE-2022-0847, CVE-2019-5736]`

**Optimal Fix:** Set `securityContext.privileged` → `false`

---

## 🔌 API Reference

```bash
# Health check
GET https://wiki05-nexus-config-env.hf.space/health

# List tasks
GET https://wiki05-nexus-config-env.hf.space/tasks

# Start episode
POST https://wiki05-nexus-config-env.hf.space/reset?task_id=task_1_easy

# Submit action
POST https://wiki05-nexus-config-env.hf.space/step
Content-Type: application/json
{
  "fix_type":     "cost",
  "target_field": "resources.requests.memory",
  "new_value":    "256Mi",
  "reasoning":    "Rightsizing to match observed ~100MB usage"
}

# Get current state (no side effects)
GET https://wiki05-nexus-config-env.hf.space/state
```

---

## 🤖 Running the Baseline Agent

```bash
# Install dependencies
pip install -r requirements.txt

# Set env vars
export HF_TOKEN="your_hf_token"
export ENV_URL="https://wiki05-nexus-config-env.hf.space"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"  # or any OpenAI-compatible model

# Run inference
python inference.py
```

Expected output:
```
[INFO] Running Nexus-Config-Env baseline against https://wiki05-nexus-config-env.hf.space
[START] task=task_1_easy env=Nexus-Config-Env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=resources.requests.memory=256Mi reward=1.00 done=true error=null
[END] success=true steps=1 score=1.000 rewards=1.00
...
[SUMMARY] Average score across 3 tasks: 0.883
```

---

## 🚀 Local Development

```bash
git clone https://huggingface.co/spaces/Wiki05/nexus-config-env
cd nexus-config-env

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

Then open: http://localhost:7860/web

---

## 👤 Author

**Vignesh E** — [GitHub: @Wiki05](https://github.com/Wiki05)

Built for the **Meta × Scaler OpenEnv Hackathon 2026**