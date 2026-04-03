# scenarios.py - The 15 Dirty YAML Scenarios for our Gym

SCENARIOS = {
    "task_1_easy": [
        {
            "id": "easy_1_ghost_ram",
            "dirty_yaml": "resources:\n  requests:\n    memory: '8Gi'\n    cpu: '4000m'",
            "telemetry": {"avg_mem_mb": 95, "cost_usd": 84.50},
            "target": "memory", "limit": "256Mi",
            "description": "Ghost RAM: App uses 100MB but requests 8GB."
        }
    ],
    "task_2_medium": [
        {
            "id": "medium_1_root_user",
            "dirty_yaml": "securityContext:\n  runAsUser: 0\nresources:\n  requests:\n    memory: '4Gi'",
            "telemetry": {"avg_mem_mb": 200},
            "target": "runAsUser", "limit": "1000",
            "description": "Security Risk: App is running as root (User 0)."
        }
    ],
    "task_3_hard": [
        {
            "id": "hard_1_total_mess",
            "dirty_yaml": "securityContext:\n  runAsUser: 0\n  privileged: true\nresources:\n  requests:\n    memory: '32Gi'",
            "telemetry": {"avg_mem_mb": 300},
            "target": "privileged", "limit": "false",
            "description": "Nightmare: Root access, privileged mode, and 32GB waste."
        }
    ]
}