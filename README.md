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

Nexus-Config-Env is a Reinforcement Learning (RL) environment designed to train AI agents in the art of **Cloud Infrastructure Tuning**. Instead of simple games, this environment challenges agents to solve real-world problems: reducing cloud waste (Cost) and hardening security (Compliance) within Kubernetes configurations.

## 🌟 Motivation (Real-World Utility)
Cloud over-provisioning and misconfigurations cost enterprises billions annually. Nexus-Config-Env provides a standardized benchmark for AI agents to:
1. **Reduce "Ghost" Resources:** Identifying and trimming unused RAM/CPU allocations.
2. **Security Hardening:** Automatically patching privileged containers and root-user vulnerabilities.
3. **Stability Engineering:** Optimizing replica sets and health probes for maximum uptime.

---

## 🛠️ Environment Design

### 👁️ Observation Space
The agent receives a `ConfigObservation` containing:
- **Raw YAML:** The current state of the Kubernetes manifest.
- **Telemetry Data:** Mocked metrics showing actual resource usage vs. requested limits.
- **Security Audit:** A list of detected CVEs or misconfigurations.

### 🕹️ Action Space
The agent can perform `ConfigAction` types:
- `PATCH_RESOURCE`: Adjust CPU/Memory limits.
- `SECURE_CONTEXT`: Update SecurityContext settings.
- `SCALABILITY_TWEAK`: Modify replica counts and strategy.

---

## 🏆 Tasks & Grading
We provide 15 scenarios across three difficulty levels. Each task is graded by a deterministic `Grader` returning a score between **0.0 and 1.0**.

| Task | Difficulty | Objective |
| :--- | :--- | :--- |
| **Ghost Hunter** | Easy | Identify and fix 10x over-provisioned RAM limits. |
| **Security Patch** | Medium | Remove `privileged: true` and set `runAsNonRoot`. |
| **Stability Architect** | Hard | Fix broken Readiness probes and unbalanced replicas. |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Docker
- OpenEnv Core

### Local Installation
```bash
# Clone the repo
git clone [https://github.com/Wiki05/nexus-config-env.git](https://github.com/Wiki05/nexus-config-env.git)
cd nexus-config-env

# Install dependencies
pip install -r requirements.txt