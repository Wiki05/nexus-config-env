# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Scenarios and Graders for Nexus-Config-Env.

3 Tasks with increasing difficulty, each with multiple sub-scenarios
to ensure score variance and prevent grader exploitation.

Task 1 (Easy)   - Resource over-provisioning (cost).      Max 3 steps.
Task 2 (Medium) - Running as root / identity risk (sec).  Max 3 steps.
Task 3 (Hard)   - Privileged container escape (sec).      Max 3 steps.

Graders return float strictly between 0.0 and 1.0 (MIN_SCORE, MAX_SCORE).

Grader design principles:
  - DETERMINISTIC: same actions -> same score
  - PARTIAL CREDIT: category + field + value layers
  - NO EXPLOIT: graders return varied scores, never always the same number
"""

from typing import List, Dict

# ── Score bounds (strict open interval as required) ──────────────────────────
MIN_SCORE: float = 0.001
MAX_SCORE: float = 0.999


def _clamp(score: float) -> float:
    return round(max(MIN_SCORE, min(MAX_SCORE, score)), 3)


# ── Scenario definitions ──────────────────────────────────────────────────────
# Each task has multiple scenarios - the environment picks the first by default.
# The `type` field MUST match the action.fix_type for category reward.

SCENARIOS: Dict[str, List[dict]] = {
    "task_1_easy": [
        {
            "id": "easy_1_ghost_ram",
            "dirty_yaml": (
                "apiVersion: v1\n"
                "kind: Pod\n"
                "metadata:\n"
                "  name: api-service\n"
                "spec:\n"
                "  containers:\n"
                "  - name: api\n"
                "    image: api:latest\n"
                "    resources:\n"
                "      requests:\n"
                "        memory: '8Gi'\n"
                "      limits:\n"
                "        memory: '16Gi'\n"
            ),
            "telemetry": {
                "avg_mem_mb": 100,
                "peak_mem_mb": 200,
                "node_ram_gb": 32,
                "waste_estimate_usd_month": 47.20,
            },
            "type": "cost",
            "target": "resources.requests.memory",
            "limit": "256Mi",
            "description": (
                "Ghost RAM: The API service uses only ~100 MB of memory but "
                "requests 8 GB. Rightsizing to 256Mi will save ~$47/month."
            ),
            "fixes_applied": [],
        },
        {
            "id": "easy_2_cpu_hoard",
            "dirty_yaml": (
                "apiVersion: v1\n"
                "kind: Pod\n"
                "metadata:\n"
                "  name: worker-job\n"
                "spec:\n"
                "  containers:\n"
                "  - name: worker\n"
                "    image: worker:2.1\n"
                "    resources:\n"
                "      requests:\n"
                "        cpu: '8'\n"
            ),
            "telemetry": {
                "avg_cpu_cores": 0.2,
                "peak_cpu_cores": 0.5,
                "cost_per_core_hour": 0.048,
            },
            "type": "cost",
            "target": "resources.requests.cpu",
            "limit": "500m",
            "description": (
                "CPU Hoarder: Worker job requests 8 CPU cores but only uses 0.2 on average. "
                "Rightsizing to 500m will significantly reduce cloud costs."
            ),
            "fixes_applied": [],
        },
    ],
    "task_2_medium": [
        {
            "id": "medium_1_root_user",
            "dirty_yaml": (
                "apiVersion: v1\n"
                "kind: Pod\n"
                "metadata:\n"
                "  name: backend-api\n"
                "spec:\n"
                "  containers:\n"
                "  - name: backend\n"
                "    image: backend:3.0\n"
                "    securityContext:\n"
                "      runAsUser: 0\n"
                "      allowPrivilegeEscalation: true\n"
            ),
            "telemetry": {
                "is_root": True,
                "user_id": 0,
                "cve_risk_score": 9.1,
                "privilege_escalation_enabled": True,
            },
            "type": "security",
            "target": "securityContext.runAsUser",
            "limit": "1000",
            "description": (
                "Root Container: The backend API is running as UID 0 (root). "
                "This violates the principle of least privilege. "
                "Fix by setting runAsUser to a non-root UID (1000)."
            ),
            "fixes_applied": [],
        },
        {
            "id": "medium_2_write_root_fs",
            "dirty_yaml": (
                "apiVersion: v1\n"
                "kind: Pod\n"
                "metadata:\n"
                "  name: data-processor\n"
                "spec:\n"
                "  containers:\n"
                "  - name: processor\n"
                "    image: processor:1.5\n"
                "    securityContext:\n"
                "      readOnlyRootFilesystem: false\n"
            ),
            "telemetry": {
                "filesystem_writes_detected": True,
                "critical_path_writes": ["/etc/passwd", "/bin"],
                "cve_score": 7.8,
            },
            "type": "security",
            "target": "securityContext.readOnlyRootFilesystem",
            "limit": "true",
            "description": (
                "Writable Root FS: The container can write to its root filesystem. "
                "Attackers can modify binaries or configs. "
                "Enable readOnlyRootFilesystem to restrict writes."
            ),
            "fixes_applied": [],
        },
    ],
    "task_3_hard": [
        {
            "id": "hard_1_privileged",
            "dirty_yaml": (
                "apiVersion: v1\n"
                "kind: Pod\n"
                "metadata:\n"
                "  name: infra-agent\n"
                "  namespace: kube-system\n"
                "spec:\n"
                "  containers:\n"
                "  - name: agent\n"
                "    image: infra-agent:latest\n"
                "    securityContext:\n"
                "      privileged: true\n"
                "      runAsUser: 0\n"
                "    hostPID: true\n"
            ),
            "telemetry": {
                "privileged_status": True,
                "host_pid_namespace": True,
                "k8s_namespace": "kube-system",
                "escape_risk": "CRITICAL",
                "cve_ids": ["CVE-2022-0847", "CVE-2019-5736"],
            },
            "type": "security",
            "target": "securityContext.privileged",
            "limit": "false",
            "description": (
                "Container Escape: infra-agent runs in kube-system with privileged=true "
                "and hostPID=true. This allows complete host takeover. "
                "Disable privileged mode immediately (set to false)."
            ),
            "fixes_applied": [],
        },
        {
            "id": "hard_2_host_network",
            "dirty_yaml": (
                "apiVersion: v1\n"
                "kind: Pod\n"
                "metadata:\n"
                "  name: network-monitor\n"
                "spec:\n"
                "  hostNetwork: true\n"
                "  containers:\n"
                "  - name: monitor\n"
                "    image: monitor:1.0\n"
                "    securityContext:\n"
                "      privileged: true\n"
            ),
            "telemetry": {
                "host_network_access": True,
                "can_sniff_cluster_traffic": True,
                "escape_risk": "CRITICAL",
            },
            "type": "security",
            "target": "securityContext.privileged",
            "limit": "false",
            "description": (
                "Network Sniffer: network-monitor has hostNetwork=true and privileged=true, "
                "allowing it to intercept all cluster traffic. "
                "Disable privileged mode to contain the blast radius."
            ),
            "fixes_applied": [],
        },
    ],
}


# ── Grader functions (deterministic, partial-credit, variance-producing) ───────

def grade_task_1(episode_log: List[Dict]) -> float:
    """
    Grader for Task 1 (Cost Optimization).
    Scoring:
      - Used 'cost' as fix_type:                +0.20
      - Targeted a resource field (memory/cpu):  +0.40
      - Applied the correct value:               +0.40
    """
    score = 0.0
    for entry in episode_log:
        if entry.get("action", "") == "cost":
            score += 0.20
        field = entry.get("target_field", "")
        if "resources" in field or "memory" in field or "cpu" in field:
            score += 0.40
        reward = entry.get("reward", 0.0)
        if reward >= 0.99:
            score += 0.40
    return _clamp(score)


def grade_task_2(episode_log: List[Dict]) -> float:
    """
    Grader for Task 2 (Security Hardening).
    Scoring:
      - Used 'security' as fix_type:             +0.20
      - Targeted a securityContext field:        +0.40
      - Applied the correct value:               +0.40
    """
    score = 0.0
    for entry in episode_log:
        if entry.get("action", "") == "security":
            score += 0.20
        field = entry.get("target_field", "")
        if "security" in field or "runasuser" in field or "filesystem" in field:
            score += 0.40
        reward = entry.get("reward", 0.0)
        if reward >= 0.99:
            score += 0.40
    return _clamp(score)


def grade_task_3(episode_log: List[Dict]) -> float:
    """
    Grader for Task 3 (Policy Enforcement — Privileged Mode).
    Scoring:
      - Used 'security' as fix_type:              +0.20
      - Targeted 'privileged' or 'hostnetwork':   +0.40
      - Applied 'false' as new_value:             +0.40
    """
    score = 0.0
    for entry in episode_log:
        if entry.get("action", "") == "security":
            score += 0.20
        field = entry.get("target_field", "")
        if "privileged" in field or "hostnetwork" in field:
            score += 0.40
        if str(entry.get("new_value", "")).lower() == "false":
            score += 0.40
    return _clamp(score)


GRADERS = {
    "task_1_easy": grade_task_1,
    "task_2_medium": grade_task_2,
    "task_3_hard": grade_task_3,
}