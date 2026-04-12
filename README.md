---
title: Nexus Config Env
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
python_version: "3.11"
tags:
  - openenv
  - reinforcement-learning
  - kubernetes
  - security
  - devops
base_path: /web
---

# 🛡️ Nexus-Config-Env: Kubernetes Hardening

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)

Trains AI agents to act as **Site Reliability Engineers (SREs)** identifying and remediating Kubernetes misconfigurations. Misconfigured clusters cost companies billions in wasted cloud spend and result in critical security breaches (like container escapes and root access).

## API Endpoints (port 7860 — same as Gradio UI)

```
POST /reset     Start new episode → returns observation
POST /step      Take action → returns observation + reward + done
GET  /state     Current episode state
GET  /health    Health check → {"status": "ok"}
GET  /schema    Action + Observation schemas
GET  /docs      Swagger API docs
```

## Step payload format

```json
POST /step
{"action_type": "apply_fix", "target_field": "securityContext.privileged", "new_value": "false", "fix_type": "security"}

or flat format (for non-fix actions):
{"action_type": "scan_config"}
```

## Quick test

```bash
# Reset
curl -X POST https://wiki05-nexus-config-env.hf.space/reset?task_id=task_1_easy

# Step
curl -X POST https://wiki05-nexus-config-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "scan_config"}'
```

## Action Space

| Action | Reward | Description |
|--------|--------|-------------|
| `scan_config` | +0.10 | Analyse YAML structure for misconfigurations |
| `read_telemetry` | +0.10 | Read runtime CPU, RAM, CVE score signals |
| `identify_issue` | +0.20 | Classify root cause (cost / security / stability) |
| `propose_fix` | +0.15 | Plan field change without executing |
| `apply_fix` | **±0.50/−0.30** | Execute remediation — +0.50 correct, −0.30 wrong |
| `verify_fix` | +0.20 | Confirm fix applied — ends episode |
| `escalate` | +0.05 | Hand off to human SRE — ends episode |
| `revert_change` | **−0.10** | Undo last change (penalty) |

## 3 Tasks

| # | Difficulty | Description | Max Steps |
|---|-----------|-------------|-----------|
| 1 | Easy | Ghost RAM: Fix RAM over-provisioning (cost waste) | 6 |
| 2 | Medium | Root User: Prevent root execution (security risk) | 8 |
| 3 | Hard | Privileged Mode: Prevent container escape | 10 |

## Baseline Scores (Qwen/Qwen2.5-72B-Instruct)

| Task | Difficulty | Score |
|------|-----------|-------|
| 1 | Easy | 0.843 |
| 2 | Medium | 0.748 |
| 3 | Hard | 0.623 |

Run `python inference.py` to reproduce.

## Setup

```bash
pip install "gradio>=5.0.0" openenv-core openai httpx python-dotenv pyyaml

# HF Space secrets configuration
API_BASE_URL = https://router.huggingface.co/v1
MODEL_NAME   = Qwen/Qwen2.5-72B-Instruct
HF_TOKEN     = your_huggingface_token

python server/app.py  # Starts on localhost:7860
python inference.py   # Baseline evaluation
```

*Built for the first OpenEnv Hackathon — Meta + Scaler.*