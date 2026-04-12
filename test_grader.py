"""Quick grader variance test — verifies all graders produce varied, in-range scores."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tasks import TASKS, _grade_episode  # type: ignore[import]  # noqa: E402

scenarios = {
    "perfect":   [
        {"action_type": "scan_config",    "fix_type": None,   "target_field": None,                        "new_value": None,    "reward": 0.10},
        {"action_type": "read_telemetry", "fix_type": None,   "target_field": None,                        "new_value": None,    "reward": 0.10},
        {"action_type": "identify_issue", "fix_type": "cost", "target_field": None,                        "new_value": None,    "reward": 0.20},
        {"action_type": "propose_fix",    "fix_type": None,   "target_field": "resources.requests.memory", "new_value": "256Mi", "reward": 0.15},
        {"action_type": "apply_fix",      "fix_type": "cost", "target_field": "resources.requests.memory", "new_value": "256mi", "reward": 0.50},
        {"action_type": "verify_fix",     "fix_type": None,   "target_field": None,                        "new_value": None,    "reward": 0.20},
    ],
    "wrong_val": [
        {"action_type": "scan_config",    "fix_type": None,   "target_field": None,                        "new_value": None,    "reward": 0.10},
        {"action_type": "identify_issue", "fix_type": "cost", "target_field": None,                        "new_value": None,    "reward": 0.20},
        {"action_type": "apply_fix",      "fix_type": "cost", "target_field": "resources.requests.memory", "new_value": "512Mi", "reward": -0.10},
    ],
    "wrong_fld": [
        {"action_type": "scan_config",    "fix_type": None,   "target_field": None,                        "new_value": None,    "reward": 0.10},
        {"action_type": "apply_fix",      "fix_type": "cost", "target_field": "resources.limits.memory",   "new_value": "256Mi", "reward": -0.30},
    ],
    "no_fix":    [
        {"action_type": "scan_config",    "fix_type": None,   "target_field": None,                        "new_value": None,    "reward": 0.10},
        {"action_type": "read_telemetry", "fix_type": None,   "target_field": None,                        "new_value": None,    "reward": 0.10},
        {"action_type": "identify_issue", "fix_type": "cost", "target_field": None,                        "new_value": None,    "reward": 0.20},
    ],
    "empty":     [],
}

print("=" * 65)
print("GRADER VARIANCE TEST - All tasks")
print("=" * 65)
all_ok = True
for task_id, task in TASKS.items():
    print(f"\n--- {task_id} ({task.difficulty}) ---")
    scores = []
    for name, log in scenarios.items():
        result = _grade_episode(log, task)
        t = result["total"]
        scores.append(t)
        in_range = 0.001 <= t <= 0.999
        print(
            f"  {name:<10}: P={result['protocol']:.2f} D={result['diagnosis']:.2f} "
            f"R={result['remediation']:.2f} E={result['efficiency']:.2f} "
            f"TOTAL={t:.3f} {'[OK]' if in_range else '[OUT OF RANGE!]'}"
        )
    unique = len(set(round(s, 3) for s in scores))
    varied = unique > 1
    in_range_all = all(0.001 <= s <= 0.999 for s in scores)
    if not varied:
        all_ok = False
    print(f"  Unique scores: {unique}/5 | Range: {min(scores):.3f}-{max(scores):.3f}")
    print(f"  All in [0.001,0.999]: {in_range_all}")
    print(f"  Grader verdict: {'PASS - VARIED' if varied else 'FAIL - SAME SCORE (DISQUALIFIED)'}")

print("\n" + "=" * 65)
print(f"Final: {'ALL GRADERS PASS' if all_ok else 'SOME GRADERS WOULD CAUSE DISQUALIFICATION'}")
print("=" * 65)
