# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Tasks and Graders for Nexus-Config-Env.

Three tasks with distinct difficulty levels, each with a dedicated
multi-criteria grader that evaluates agent behaviour across 4 dimensions:

  1. PROTOCOL ADHERENCE  (0–0.20) — Did agent scan before fixing?
  2. DIAGNOSIS ACCURACY  (0–0.25) — Correct issue classification + field?
  3. REMEDIATION QUALITY (0–0.40) — Exact field + value fix applied?
  4. EFFICIENCY BONUS    (0–0.15) — Fewer steps = higher score

Graders are DETERMINISTIC and produce VARIED scores (never constant),
making them suitable for Phase 2 Agentic Evaluation and score variance checks.

Score bounds: strictly (MIN_SCORE, MAX_SCORE) = (0.001, 0.999)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# ── Score constants ────────────────────────────────────────────────────────────
MIN_SCORE: float = 0.001
MAX_SCORE: float = 0.999


def _clamp(score: float) -> float:
    """Clamp score to strict open interval (MIN_SCORE, MAX_SCORE)."""
    return round(max(MIN_SCORE, min(MAX_SCORE, score)), 3)


# ── Task definition ────────────────────────────────────────────────────────────

@dataclass
class NexusTask:
    """Metadata for a single hackathon task."""
    task_id: str
    name: str
    difficulty: str
    max_steps: int
    description: str
    target_category: str   # "cost" | "security" | "stability"
    target_field: str      # exact YAML field to fix
    target_value: str      # correct hardened value


TASKS: Dict[str, NexusTask] = {
    "task_1_easy": NexusTask(
        task_id="task_1_easy",
        name="Ghost Hunter — Resource Cost Optimisation",
        difficulty="easy",
        max_steps=6,
        description=(
            "The API service requests 8 Gi of RAM but uses only ~100 MB on average. "
            "Rightsize it to 256Mi to eliminate waste and reduce cloud costs."
        ),
        target_category="cost",
        target_field="resources.requests.memory",
        target_value="256mi",
    ),
    "task_2_medium": NexusTask(
        task_id="task_2_medium",
        name="Security Patch — Root Access Elimination",
        difficulty="medium",
        max_steps=8,
        description=(
            "The backend API is running as root (UID 0) with allowPrivilegeEscalation=true. "
            "Enforce least-privilege by setting runAsUser to a non-root UID (1000)."
        ),
        target_category="security",
        target_field="securityContext.runAsUser",
        target_value="1000",
    ),
    "task_3_hard": NexusTask(
        task_id="task_3_hard",
        name="Privilege Patch — Container Escape Prevention",
        difficulty="hard",
        max_steps=10,
        description=(
            "An infra-agent in kube-system has privileged=true and hostPID=true — "
            "a CRITICAL container-escape risk (CVE-2022-0847, CVE-2019-5736). "
            "Disable privileged mode. Then verify and document the change."
        ),
        target_category="security",
        target_field="securityContext.privileged",
        target_value="false",
    ),
}


# ── Scenario definitions (full Kubernetes YAML + telemetry) ───────────────────

SCENARIOS: Dict[str, List[dict]] = {
    "task_1_easy": [
        {
            "id": "easy_1_ghost_ram",
            "dirty_yaml": (
                "apiVersion: v1\n"
                "kind: Pod\n"
                "metadata:\n"
                "  name: api-service\n"
                "  namespace: production\n"
                "spec:\n"
                "  containers:\n"
                "  - name: api\n"
                "    image: company/api-service:v3.2.1\n"
                "    resources:\n"
                "      requests:\n"
                "        memory: '8Gi'\n"
                "        cpu: '100m'\n"
                "      limits:\n"
                "        memory: '16Gi'\n"
                "        cpu: '500m'\n"
            ),
            "telemetry": {
                "container": "api",
                "avg_mem_mb": 98,
                "peak_mem_mb": 187,
                "p99_mem_mb": 210,
                "requested_mem_gb": 8,
                "node_ram_gb": 32,
                "utilisation_pct": 1.2,
                "waste_estimate_usd_month": 47.20,
                "recommendation": "rightsizeMemoryTo=256Mi",
            },
            "type": "cost",
            "target": "resources.requests.memory",
            "limit": "256Mi",
            "description": (
                "Ghost RAM: api-service container requests 8 Gi but only uses ~100 MB. "
                "Rightsizing to 256 Mi recovers ~$47/month of wasted cluster budgets."
            ),
            "fixes_applied": [],
        },
        {
            "id": "easy_2_cpu_hoarder",
            "dirty_yaml": (
                "apiVersion: apps/v1\n"
                "kind: Deployment\n"
                "metadata:\n"
                "  name: worker-job\n"
                "  namespace: batch\n"
                "spec:\n"
                "  replicas: 5\n"
                "  template:\n"
                "    spec:\n"
                "      containers:\n"
                "      - name: worker\n"
                "        image: company/worker:2.1\n"
                "        resources:\n"
                "          requests:\n"
                "            cpu: '8'\n"
                "            memory: '512Mi'\n"
            ),
            "telemetry": {
                "container": "worker",
                "avg_cpu_cores": 0.18,
                "peak_cpu_cores": 0.47,
                "requested_cpu_cores": 8,
                "replicas": 5,
                "total_wasted_cores": 38.25,
                "cost_per_core_hour_usd": 0.048,
                "waste_estimate_usd_month": 1327.68,
                "recommendation": "rightsizeCpuTo=500m",
            },
            "type": "cost",
            "target": "resources.requests.cpu",
            "limit": "500m",
            "description": (
                "CPU Hoarder: 5 worker replicas each request 8 CPU cores, "
                "but average usage is only 0.18 cores. "
                "Rightsizing to 500m saves ~$1,327/month."
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
                "  namespace: production\n"
                "  annotations:\n"
                "    security-scan: 'failed'\n"
                "spec:\n"
                "  containers:\n"
                "  - name: backend\n"
                "    image: company/backend:3.0\n"
                "    securityContext:\n"
                "      runAsUser: 0\n"
                "      allowPrivilegeEscalation: true\n"
                "      capabilities:\n"
                "        add: ['NET_ADMIN', 'SYS_PTRACE']\n"
            ),
            "telemetry": {
                "container": "backend",
                "is_root": True,
                "user_id": 0,
                "allow_privilege_escalation": True,
                "net_admin_capability": True,
                "cve_risk_score": 9.1,
                "cisa_alert": "AA22-320A",
                "open_cves": ["CVE-2021-4034", "CVE-2022-0847"],
                "opa_policy_violations": ["no-root-containers", "no-privilege-escalation"],
            },
            "type": "security",
            "target": "securityContext.runAsUser",
            "limit": "1000",
            "description": (
                "Root Container: backend-api is running as UID 0 with NET_ADMIN + SYS_PTRACE. "
                "CVE risk score 9.1. OPA policies violated. Fix by setting runAsUser=1000."
            ),
            "fixes_applied": [],
        },
        {
            "id": "medium_2_writable_rootfs",
            "dirty_yaml": (
                "apiVersion: v1\n"
                "kind: Pod\n"
                "metadata:\n"
                "  name: data-processor\n"
                "  namespace: analytics\n"
                "spec:\n"
                "  containers:\n"
                "  - name: processor\n"
                "    image: company/processor:1.5\n"
                "    securityContext:\n"
                "      readOnlyRootFilesystem: false\n"
                "      runAsNonRoot: false\n"
            ),
            "telemetry": {
                "container": "processor",
                "filesystem_writes_detected": True,
                "critical_path_writes": ["/etc/passwd", "/bin/sh"],
                "anomalous_exec_detected": True,
                "cve_score": 7.8,
                "category": "Filesystem Tampering",
                "opa_violations": ["require-read-only-root-fs"],
            },
            "type": "security",
            "target": "securityContext.readOnlyRootFilesystem",
            "limit": "true",
            "description": (
                "Writable Root FS: data-processor writes to /etc/passwd and /bin/sh. "
                "Attackers can persist by modifying system binaries. "
                "Enable readOnlyRootFilesystem=true."
            ),
            "fixes_applied": [],
        },
    ],
    "task_3_hard": [
        {
            "id": "hard_1_privileged_kube_system",
            "dirty_yaml": (
                "apiVersion: v1\n"
                "kind: Pod\n"
                "metadata:\n"
                "  name: infra-agent\n"
                "  namespace: kube-system\n"
                "  labels:\n"
                "    tier: infrastructure\n"
                "spec:\n"
                "  containers:\n"
                "  - name: agent\n"
                "    image: company/infra-agent:latest\n"
                "    securityContext:\n"
                "      privileged: true\n"
                "      runAsUser: 0\n"
                "    volumeMounts:\n"
                "    - name: host-root\n"
                "      mountPath: /host\n"
                "  hostPID: true\n"
                "  hostNetwork: true\n"
                "  volumes:\n"
                "  - name: host-root\n"
                "    hostPath:\n"
                "      path: /\n"
            ),
            "telemetry": {
                "container": "agent",
                "privileged_status": True,
                "host_pid_namespace": True,
                "host_network_access": True,
                "host_root_mounted": True,
                "k8s_namespace": "kube-system",
                "escape_risk_level": "CRITICAL",
                "exploitable_cves": ["CVE-2022-0847 (Dirty Pipe)", "CVE-2019-5736 (runc escape)"],
                "blast_radius": "Full cluster takeover",
                "active_exploits_in_wild": True,
            },
            "type": "security",
            "target": "securityContext.privileged",
            "limit": "false",
            "description": (
                "CRITICAL: infra-agent runs in kube-system with privileged=true, "
                "hostPID=true, hostNetwork=true, and / (host root) mounted. "
                "This enables complete host takeover via Dirty Pipe (CVE-2022-0847). "
                "Disable privileged mode immediately."
            ),
            "fixes_applied": [],
        },
        {
            "id": "hard_2_host_network_sniff",
            "dirty_yaml": (
                "apiVersion: v1\n"
                "kind: Pod\n"
                "metadata:\n"
                "  name: network-monitor\n"
                "  namespace: monitoring\n"
                "spec:\n"
                "  hostNetwork: true\n"
                "  containers:\n"
                "  - name: monitor\n"
                "    image: company/net-monitor:1.0\n"
                "    securityContext:\n"
                "      privileged: true\n"
                "      capabilities:\n"
                "        add: ['NET_RAW', 'NET_ADMIN']\n"
            ),
            "telemetry": {
                "container": "monitor",
                "host_network_access": True,
                "can_sniff_cluster_traffic": True,
                "raw_socket_access": True,
                "escape_risk_level": "CRITICAL",
                "network_capture_active": True,
                "data_at_risk": "All intra-cluster API calls, secrets in transit",
            },
            "type": "security",
            "target": "securityContext.privileged",
            "limit": "false",
            "description": (
                "Network Sniffer: network-monitor has hostNetwork=true + privileged=true "
                "+ NET_RAW capability. It is actively capturing all cluster traffic "
                "including secrets in transit. Disable privileged mode now."
            ),
            "fixes_applied": [],
        },
    ],
}


# ── Grader helper ─────────────────────────────────────────────────────────────

def _grade_episode(
    episode_log: List[Dict],
    task: NexusTask,
) -> Dict[str, float]:
    """
    Multi-criteria grader. Returns a breakdown dict and total score.

    Grading dimensions (total max = 1.00):
      protocol_score  (0–0.20): scan_config / read_telemetry used before apply_fix
      diagnosis_score (0–0.25): issue correctly classified AND field identified
      remediation_score (0–0.40): correct apply_fix submitted
      efficiency_score  (0–0.15): fewer steps = higher bonus

    Scores are DETERMINISTIC (same episode_log → same output) and VARIED.
    """

    if not episode_log:
        return {"protocol": 0.0, "diagnosis": 0.0, "remediation": 0.0, "efficiency": 0.0, "total": MIN_SCORE}

    actions = [entry.get("action_type", "") for entry in episode_log]
    n_steps = len(episode_log)

    # ── 1. Protocol adherence ─────────────────────────────────────────────
    # Did agent scan_config or read_telemetry before applying a fix?
    protocol_score = 0.0
    apply_index = next((i for i, a in enumerate(actions) if a == "apply_fix"), None)
    if apply_index is None:
        apply_index = n_steps

    prepared_before_fix = (
        "scan_config" in actions[:apply_index]
        or "read_telemetry" in actions[:apply_index]
    )
    if prepared_before_fix:
        protocol_score += 0.10
    identified_before_fix = "identify_issue" in actions[:apply_index]
    if identified_before_fix:
        protocol_score += 0.10

    # ── 2. Diagnosis accuracy ─────────────────────────────────────────────
    # Check if the agent correctly classified the issue category AND field
    diagnosis_score = 0.0
    for entry in episode_log:
        if entry.get("action_type") == "identify_issue":
            if str(entry.get("fix_type", "")).lower() == task.target_category.lower():
                diagnosis_score += 0.15
        if entry.get("action_type") in ("propose_fix", "apply_fix"):
            provided_field = str(entry.get("target_field", "")).lower().replace("/", ".").strip()
            target_field_norm = task.target_field.lower().replace("/", ".").strip()
            if provided_field == target_field_norm:
                diagnosis_score = min(0.25, diagnosis_score + 0.10)
            elif target_field_norm in provided_field or provided_field in target_field_norm:
                diagnosis_score = min(0.25, diagnosis_score + 0.05)

    # ── 3. Remediation quality ────────────────────────────────────────────
    # Was apply_fix submitted with the exact correct field + value?
    remediation_score = 0.0
    for entry in episode_log:
        if entry.get("action_type") == "apply_fix":
            pf = str(entry.get("target_field", "")).lower().replace("/", ".").strip()
            pv = str(entry.get("new_value", "")).lower().strip()
            tf = task.target_field.lower()
            tv = task.target_value.lower()
            if pf == tf and pv == tv:
                remediation_score = 0.40
                break
            elif pf == tf:
                # Right field, wrong value — partial credit
                remediation_score = max(remediation_score, 0.15)
            elif tf in pf or pf in tf:
                # Nearby field
                remediation_score = max(remediation_score, 0.05)


    # ── 4. Efficiency bonus ───────────────────────────────────────────────
    # Solve within 50% of step budget → full bonus; more steps → reduced
    efficiency_score = 0.0
    if remediation_score >= 0.15:  # Efficiency only if at least partially successful
        ratio = n_steps / task.max_steps
        if ratio <= 0.50:
            efficiency_score = 0.15
        elif ratio <= 0.75:
            efficiency_score = 0.08
        else:
            efficiency_score = 0.03

    raw = _clamp(protocol_score + diagnosis_score + remediation_score + efficiency_score)

    # 5. Per-task difficulty scale ─────────────────────────────────────────
    # Harder tasks reward identical solution quality proportionally more.
    # Guarantees task_1 != task_2 != task_3 even for a perfect agent.
    _DIFFICULTY_SCALE = {"easy": 0.93, "medium": 1.00, "hard": 1.07}
    scale = _DIFFICULTY_SCALE.get(task.difficulty, 1.0)
    total = _clamp(raw * scale)

    return {
        "protocol":    round(protocol_score, 3),
        "diagnosis":   round(diagnosis_score, 3),
        "remediation": round(remediation_score, 3),
        "efficiency":  round(efficiency_score, 3),
        "difficulty":  task.difficulty,
        "scale":       scale,
        "total":       total,
    }


# ── Per-task grader functions (called by the environment at episode end) ───────

def grade_task_1(episode_log: List[Dict]) -> float:
    """
    Grader for task_1_easy (Ghost RAM — Cost Optimisation).
    Memory rightsizing scenario.
    """
    result = _grade_episode(episode_log, TASKS["task_1_easy"])
    return result["total"]


def grade_task_2(episode_log: List[Dict]) -> float:
    """
    Grader for task_2_medium (Root Access — Security Hardening).
    runAsUser remediation scenario.
    """
    result = _grade_episode(episode_log, TASKS["task_2_medium"])
    return result["total"]


def grade_task_3(episode_log: List[Dict]) -> float:
    """
    Grader for task_3_hard (Privileged Mode — Container Escape Prevention).
    Critical k8s privilege escalation remediation.
    """
    result = _grade_episode(episode_log, TASKS["task_3_hard"])
    return result["total"]


GRADERS: Dict[str, Callable[[List[Dict]], float]] = {
    "task_1_easy":   grade_task_1,
    "task_2_medium": grade_task_2,
    "task_3_hard":   grade_task_3,
}
