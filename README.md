---
title: Nexus Config Env
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# Nexus-Config-Env
### An OpenEnv-Compliant RL Environment for Kubernetes Configuration Hardening

Nexus-Config-Env is a professional-grade Reinforcement Learning environment built on the OpenEnv specification. It trains and evaluates AI agents on real-world Kubernetes configuration remediation tasks — fixing cloud security vulnerabilities, eliminating resource waste, and enforcing infrastructure best practices.

This environment simulates the exact problems DevOps engineers face daily: over-provisioned memory consuming company budgets, containers running as root exposing critical attack surfaces, and unnecessary privileged access creating compliance failures.

---

## Live Environment

- **Hugging Face Space:** https://huggingface.co/spaces/Wiki05/nexus-config-env
- **API Documentation:** https://wiki05-nexus-config-env.hf.space/docs
- **Health Check:** https://wiki05-nexus-config-env.hf.space/health
- **Interactive Playground:** https://wiki05-nexus-config-env.hf.space

---

## Real-World Motivation

Cloud misconfiguration is a $30 billion annual problem. Companies running Kubernetes clusters face three recurring failures:

1. **Ghost RAM:** Developers request 8GB of memory for apps that use 100MB. The company pays for 7.9GB of wasted resources every month.
2. **Root Container Access:** Containers running as `runAsUser: 0` give attackers full system access on a single breach. 90% of cloud security incidents trace back to simple misconfigurations like this.
3. **Privileged Escalation:** `privileged: true` grants containers unrestricted host access, violating every compliance standard from SOC2 to ISO 27001.

Nexus-Config-Env provides a standardized RL benchmark where AI agents learn to detect and fix these exact problems — step by step, with shaped reward signals that reflect real operational outcomes.

---

## Environment Design

### RL Loop

    reset() → agent receives dirty YAML + telemetry
    step 1  → agent identifies the vulnerable field path → +0.50 reward
    step 2  → agent applies the correct hardened value   → +0.50 reward
    done    → episode complete, final score returned

### Observation Space (NexusObservation)

| Field | Type | Description |
|:---|:---|:---|
| config_id | str | Unique identifier for the current scenario |
| dirty_yaml | str | The broken Kubernetes YAML manifest |
| telemetry | dict | Simulated runtime metrics (actual memory usage, CPU, security flags) |
| fixes_applied | list | List of fixes the agent has applied so far |
| current_score | float | Running reward score for this episode |
| step | int | Current step number in the episode |
| done | bool | Whether the episode has ended |

### Action Space (NexusAction)

| Field | Type | Description |
|:---|:---|:---|
| fix_type | str | Category of fix: security, cost, or health |
| target_field | str | The YAML path to modify (e.g., securityContext.runAsUser) |
| new_value | str | The corrected value to apply (e.g., 1000) |
| reasoning | str | Agent's explanation for why this fix is necessary |

---

## Tasks and Grading

The environment provides 3 tasks across difficulty levels, each with 5 unique scenarios.

| Task ID | Name | Difficulty | Objective | Max Steps |
|:---|:---|:---:|:---|:---:|
| task_1_easy | Ghost Hunter | Easy | Identify and reduce over-provisioned memory from 8Gi to 256Mi | 2 |
| task_2_medium | Security Patch | Medium | Fix root user access by changing runAsUser from 0 to 1000 | 2 |
| task_3_hard | Privilege Patch | Hard | Remove unnecessary privileged container access | 2 |

### Grading Logic

Each task uses a 2-step deterministic grader:

- **Step 1 — Identification (+0.50):** Agent correctly identifies the vulnerable YAML field path
- **Step 2 — Remediation (+0.50):** Agent applies the exact correct hardened value
- **Wrong field penalty (-0.20):** Agent targets a non-existent or incorrect field
- **Final score range:** 0.0 to 1.0

All graders are purely programmatic — no AI involved in scoring. Scores are fully deterministic and reproducible.

---

## Reward Function

The reward signal is shaped across the full trajectory, not just at episode end:

    Step 1 reward: +0.50 if correct field identified, 0.0 if wrong
    Step 2 reward: +0.50 if correct value applied, 0.0 if wrong
    -0.20 penalty if field does not exist in config
    Final episode score = sum of step rewards, capped at 1.0

This partial reward design means agents receive useful learning signals at every step, not just binary success/failure at the end.

---

## Baseline Scores

Scores produced by running `inference.py` with `Qwen/Qwen2.5-72B-Instruct` via Hugging Face Inference Router:

| Task | Baseline Score |
|:---|:---:|
| task_1_easy | 0.75 |
| task_2_medium | 0.50 |
| task_3_hard | 0.50 |

---

## API Endpoints

| Endpoint | Method | Description |
|:---|:---:|:---|
| /health | GET | Returns server health status |
| /reset | POST | Initializes a new episode for a given task_id |
| /step | POST | Processes one agent action, returns observation and reward |
| /tasks | GET | Returns all available tasks and action schema |
| /grader | POST | Returns the grader score for a completed episode |
| /baseline | POST | Triggers the baseline inference script |
| /docs | GET | Interactive OpenAPI documentation |

---

## Local Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Docker (recommended for isolated testing)
- Git

### Installation
```bash
git clone https://huggingface.co/spaces/Wiki05/nexus-config-env
cd nexus-config-env
pip install -r requirements.txt
```

### Start the Server
```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 7860
```

### Validate OpenEnv Compliance
```bash
openenv validate http://localhost:7860
```

---

## Running the Baseline Agent

### Step 1 — Set Environment Variables

**Linux / Mac:**
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_huggingface_token_here"
export ENV_URL="http://127.0.0.1:7860"
```

**Windows PowerShell:**
```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN="your_huggingface_token_here"
$env:ENV_URL="http://127.0.0.1:7860"
```

### Step 2 — Run Inference
```bash
python inference.py
```

### Expected Output Format

    {"type": "START", "task_id": "task_1_easy", "model": "Qwen/Qwen2.5-72B-Instruct"}
    {"type": "STEP", "task_id": "task_1_easy", "step": 1, "reward": 0.5, "done": false}
    {"type": "STEP", "task_id": "task_1_easy", "step": 2, "reward": 0.5, "done": true}
    {"type": "END", "task_id": "task_1_easy", "score": 1.0}

---

## Docker Deployment

### Build
```bash
docker build -t nexus-config-env .
```

### Run
```bash
docker run -p 7860:7860 \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  -e HF_TOKEN="your_token_here" \
  nexus-config-env
```

### Verify
```bash
curl http://localhost:7860/health
curl http://localhost:7860/tasks
curl -X POST http://localhost:7860/reset?task_id=task_1_easy
```

---

## Pre-Submission Checklist

[x] HF Space deploys and /health returns 200
[x] openenv validate passes all checks
[x] docker build completes without errors
[x] docker run starts server successfully
[x] /reset returns valid NexusObservation
[x] /step returns reward and updated observation
[x] /tasks returns 3 tasks with action schema
[x] /grader returns score between 0.0 and 1.0
[x] inference.py runs in under 20 minutes
[x] inference.py uses OpenAI client
[x] inference.py prints START/STEP/END JSON logs
[x] All 3 tasks have graders returning 0.0 to 1.0
[x] README includes action/observation space definitions
[x] Dockerfile is at root level
[x] openenv.yaml is at root level
[x] inference.py is at root level

---

## Project Structure

nexus-config-env/
├── server/
│   ├── init.py
│   ├── app.py
│   └── nexus_environment.py
├── Dockerfile
├── client.py
├── inference.py
├── models.py
├── scenarios.py
├── openenv.yaml
├── pyproject.toml
├── requirements.txt
└── README.md

---

## Author

**Vignesh E**
- GitHub: https://github.com/Wiki05
- Email: vigneshdev1022@gmail.com
- LinkedIn: https://www.linkedin.com/in/vignesh-e-dev/
- Built for Meta x Scaler OpenEnv Hackathon 2026