import urllib.request
import json
import time

time.sleep(2)
base = "http://localhost:7860"

def post(url, data=None):
    body = json.dumps(data).encode() if data else b""
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())

def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())

print("=== FULL API TEST ===")
h = get(f"{base}/health")
print(f"Health: {h}")

metadata = get(f"{base}/metadata")
task_map = {t["id"]: t["max_steps"] for t in metadata.get("tasks", [])}
print(f"Metadata task max_steps: {task_map}")

expected = {"task_1_easy": 6, "task_2_medium": 8, "task_3_hard": 10}

for task_id, exp_max in expected.items():
    obs_r = post(f"{base}/reset?task_id={task_id}")
    state = get(f"{base}/state")
    actual_step = state["observation"]["step"]
    actual_done = state["observation"]["done"]

    # Do 2 valid steps
    s1 = post(f"{base}/step", {"action_type": "scan_config",    "reasoning": "scanning"})
    s2 = post(f"{base}/step", {"action_type": "read_telemetry", "reasoning": "reading"})

    state2 = get(f"{base}/state")
    step2  = state2["observation"]["step"]

    print(f"\n{task_id}:")
    print(f"  Expected max_steps = {exp_max}")
    print(f"  metadata max_steps = {task_map.get(task_id)}")
    print(f"  After reset: step={actual_step} done={actual_done}")
    print(f"  After 2 steps: step={step2}")
    ok = "OK" if task_map.get(task_id) == exp_max else "FAIL"
    print(f"  Result: {ok}")

print("\n=== DONE ===")
