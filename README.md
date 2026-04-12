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
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Spaces-blue)](https://huggingface.co/spaces/Wiki05/nexus-config-env)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Nexus-Config-Env** is a high-fidelity Reinforcement Learning environment that trains AI agents to act as **Site Reliability Engineers (SREs)**. Given a broken Kubernetes YAML configuration and real-time telemetry, the agent must follow a structured 8-step SRE workflow to identify and remediate misconfigurations — earning rewards at each stage of the diagnostic process.

The environment features a **discrete 8-action space** and a **4-dimensional multi-criteria grader** that provides rich learning signal at every step, enabling effective RL training via partial credit rather than binary pass/fail.

---

## 🌐 Live Access

| Resource | URL |
|:---------|:----|
| **Playground** | [wiki05-nexus-config-env.hf.space/web](https://wiki05-nexus-config-env.hf.space/web) |
| **API Docs** | [wiki05-nexus-config-env.hf.space/docs](https://wiki05-nexus-config-env.hf.space/docs) |
| **Health** | [wiki05-nexus-config-env.hf.space/health](https://wiki05-nexus-config-env.hf.space/health) |
| **Tasks** | [wiki05-nexus-config-env.hf.space/tasks](https://wiki05-nexus-config-env.hf.space/tasks) |

---

## 💡 Real-World Problem: The $30B Kubernetes Misconfiguration Crisis

Misconfigured Kubernetes clusters are responsible for:
- **$30B+ yearly** in unnecessary cloud compute spend (ghost RAM / CPU hoarding)
- **Major security breaches** (Uber, Capital One) caused by over-privileged containers
- **Regulatory violations** (PCI-DSS, SOC2) from containers running as root

This environment targets **three production-level attack vectors** directly from real SRE incident playbooks:

| # | Problem | Real Impact | CVE Examples |
|---|---------|-------------|--------------|
| 1 | **Ghost RAM** — 8 Gi requested, 100 MB used | $47–$1,327/month per workload | — |
| 2 | **Root Containers** — UID 0 with privilege escalation | Host compromise on container breach | CVE-2021-4034 |
| 3 | **Privileged Mode** — host kernel + PID namespace | Full cluster takeover | CVE-2022-0847 (Dirty Pipe) |

---

## ⚡ Action Space (Discrete, 8 Actions)

Unlike simple "guess the field" environments, Nexus-Config-Env features a **structured SRE workflow** with discrete actions that mirror how real engineers remediate incidents:

| Action | Reward | Terminal | Description |
|:-------|:------:|:--------:|:------------|
| `scan_config` | +0.10 | No | Analyse YAML structure for misconfigurations |
| `read_telemetry` | +0.10 | No | Read runtime metrics (CPU, RAM, CVE scores) |
| `identify_issue` | +0.20 | No | Classify root cause (cost / security / stability) |
| `propose_fix` | +0.15 | No | Plan a field change without executing |
| `apply_fix` | **+0.50 / −0.30** | No | Execute remediation (positive if correct, negative if wrong) |
| `verify_fix` | +0.20 | **Yes** | Confirm fix applied — ends episode |
| `escalate` | +0.05 | **Yes** | Hand off to human SRE — ends episode |
| `revert_change` | −0.10 | No | Undo last change |

### Optimal SRE Protocol (maximises score)
```
scan_config → read_telemetry → identify_issue → propose_fix → apply_fix → verify_fix
```

---

## 📊 Grader: Multi-Criteria Deterministic Scoring

The grader evaluates each episode across **4 independent dimensions** for a maximum score of 1.00:

| Dimension | Weight | Criteria |
|:----------|:------:|:---------|
| **Protocol Adherence** | 20% | Used scan_config / read_telemetry before apply_fix? Identified issue before fixing? |
| **Diagnosis Accuracy** | 25% | Correct issue category (cost/security)? Identified the exact YAML field? |
| **Remediation Quality** | 40% | Exact target_field AND new_value applied? Partial credit for close attempts. |
| **Efficiency Bonus** | 15% | Resolved within 50% of step budget → full bonus; graduated reduction after. |

**Key properties:**
- ✅ **Deterministic**: Same episode trajectory → same grader score
- ✅ **Varied**: Score changes meaningfully based on agent behaviour
- ✅ **Partial credit**: No binary pass/fail — every good action is rewarded
- ✅ **Per-task graders**: Separate grader function per task ID

---

## 🎯 Tasks

### `task_1_easy` — Ghost Hunter (Cost Optimisation, 6 steps)

**Misconfig:** API service requests 8 Gi RAM, uses ~100 MB.

```yaml
# Broken
resources:
  requests:
    memory: '8Gi'
```

**Telemetry signals:** `avg_mem_mb: 98`, `waste_estimate_usd_month: 47.20`

**Correct fix:** `apply_fix(fix_type=cost, target_field=resources.requests.memory, new_value=256Mi)`

**Baseline score:** 0.843

---

### `task_2_medium` — Security Patch (Root Access, 8 steps)

**Misconfig:** Backend API running as root (UID 0) with privilege escalation enabled.

```yaml
# Broken
securityContext:
  runAsUser: 0
  allowPrivilegeEscalation: true
```

**Telemetry signals:** `is_root: true`, `cve_risk_score: 9.1`, `opa_policy_violations: [no-root-containers]`

**Correct fix:** `apply_fix(fix_type=security, target_field=securityContext.runAsUser, new_value=1000)`

**Baseline score:** 0.748

---

### `task_3_hard` — Privilege Patch (Container Escape, 10 steps)

**Misconfig:** CRITICAL — infra-agent in kube-system with `privileged: true`, `hostPID: true`, host root mounted.

```yaml
# Broken — CRITICAL escape risk
securityContext:
  privileged: true
  runAsUser: 0
hostPID: true
hostNetwork: true
```

**Telemetry signals:** `escape_risk_level: CRITICAL`, `exploitable_cves: [CVE-2022-0847, CVE-2019-5736]`

**Correct fix:** `apply_fix(fix_type=security, target_field=securityContext.privileged, new_value=false)`

**Baseline score:** 0.623

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  AI Agent (any LLM via OpenAI-compatible client)        │
│                                                         │
│  1. POST /reset?task_id=X  → NexusObservation           │
│  2. POST /step (NexusAction JSON) → reward + obs        │
│  3. Repeat until done=true                              │
│  4. GET /state → read-only episode snapshot             │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Nexus-Config-Env (FastAPI + Gradio on HF Spaces)       │
│                                                         │
│  NexusAction     (action_type + optional fields)        │
│  NexusObservation (dirty_yaml + telemetry + score)      │
│  NexusEnvironment (8-action dispatcher + YAML patcher)  │
│  Multi-Criteria Grader (4-dimensional episode scorer)   │
└─────────────────────────────────────────────────────────┘
```

### Observation Space

| Field | Type | Description |
|:------|:-----|:------------|
| `config_id` | str | Unique scenario ID |
| `dirty_yaml` | str | Current K8s manifest |
| `telemetry` | dict | CPU, RAM, CVE scores, cost estimates |
| `message` | str | Feedback from last action |
| `fixes_applied` | list | Fields patched so far |
| `identified_issues` | list | Categories agent has identified |
| `proposed_field` | str\|null | Last field targeted by propose_fix |
| `current_score` | float | Cumulative graded score (0.001–0.999) |
| `step` | int | Current step number |
| `done` | bool | Episode complete flag |
| `actions_taken` | list | Ordered action history |

---

## 🔌 API Reference

```bash
# Health check
curl https://wiki05-nexus-config-env.hf.space/health

# Start episode
curl -X POST "https://wiki05-nexus-config-env.hf.space/reset?task_id=task_1_easy"

# Submit action — scan first
curl -X POST https://wiki05-nexus-config-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "scan_config", "reasoning": "Analyse YAML structure"}'

# Submit apply_fix
curl -X POST https://wiki05-nexus-config-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type":  "apply_fix",
    "fix_type":     "cost",
    "target_field": "resources.requests.memory",
    "new_value":    "256Mi",
    "reasoning":    "Rightsizing ghost RAM: 8Gi requested vs 100MB actual usage"
  }'

# Verify fix
curl -X POST https://wiki05-nexus-config-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "verify_fix", "reasoning": "Confirm remediation applied"}'
```

---

## 📈 Baseline Results

Pre-computed with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router:

| Task | Difficulty | Avg Score |
|:-----|:----------:|:---------:|
| task_1_easy | Easy | 0.843 |
| task_2_medium | Medium | 0.748 |
| task_3_hard | Hard | 0.623 |
| **Overall** | — | **0.738** |

Full results saved to [`baseline_results.json`](./baseline_results.json).

---

## 🤖 Running the Baseline Agent

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN="your_huggingface_token"
export ENV_URL="https://wiki05-nexus-config-env.hf.space"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"  # or any OpenAI-compatible model

# Run baseline inference script
python inference.py
```

Expected output:
```
[START] task=task_1_easy env=Nexus-Config-Env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=scan_config reward=0.10 done=false error=null
[STEP] step=2 action=read_telemetry reward=0.10 done=false error=null
[STEP] step=3 action=identify_issue reward=0.20 done=false error=null
[STEP] step=4 action=propose_fix reward=0.15 done=false error=null
[STEP] step=5 action=apply_fix(resources.requests.memory=256Mi) reward=0.50 done=false error=null
[STEP] step=6 action=verify_fix reward=0.20 done=true error=null
[END] success=true steps=6 score=0.843 rewards=0.10,0.10,0.20,0.15,0.50,0.20
```

---

## 🚀 Local Development

```bash
git clone https://huggingface.co/spaces/Wiki05/nexus-config-env
cd nexus-config-env

python -m venv .venv
.venv\Scripts\activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
# Open: http://localhost:7860/web
```

---

## 👤 Author

**Vignesh E** — [GitHub: @Wiki05](https://github.com/Wiki05)

Built for the **Meta × Scaler OpenEnv Hackathon 2026**