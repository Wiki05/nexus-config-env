SCENARIOS = {
    "task_1_easy": [
        {
            "id": "easy_1_ghost_ram",
            "dirty_yaml": "resources:\n  requests:\n    memory: '8Gi'",
            "telemetry": {"avg_mem_mb": 100, "peak_mem_mb": 200},
            "target": "memory", "limit": "256Mi",
            "description": "Ghost RAM: App uses 100MB but requests 8GB."
        }
    ],
    "task_2_medium": [
        {
            "id": "medium_1_root_user",
            "dirty_yaml": "securityContext:\n  runAsUser: 0",
            "telemetry": {"is_root": True, "user_id": 0},
            "target": "runAsUser", "limit": "1000",
            "description": "Security Risk: App is running as root (User 0)."
        }
    ],
    "task_3_hard": [
        {
            "id": "hard_1_privileged",
            "dirty_yaml": "securityContext:\n  privileged: true",
            "telemetry": {"privileged_status": True},
            "target": "privileged", "limit": "false",
            "description": "Vulnerability: Container has unnecessary privileged access."
        }
    ]
}