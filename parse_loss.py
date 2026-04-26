import json

f = open("/home/walidsobhi/stack-4.0-adapter/checkpoint-400/trainer_state.json")
d = json.load(f)
logs = [e for e in d.get("log_history", []) if "loss" in e]

print(f"Logged: {len(logs)} steps")
first_loss = logs[0]["loss"]
last_loss = logs[-1]["loss"]
step_first = logs[0]["step"]
step_last = logs[-1]["step"]
print(f"First: {first_loss:.4f} @ step {step_first}")
print(f"Last:  {last_loss:.4f} @ step {step_last}")
print()
print("Step  | Loss    | Delta")
print("-" * 35)
prev = None
for e in logs:
    loss_val = e["loss"]
    step_val = e["step"]
    if prev:
        pct = ((prev - loss_val) / prev) * 100
        print(f"{step_val:4d} | {loss_val:.4f} | {-pct:.1f}%")
    else:
        print(f"{step_val:4d} | {loss_val:.4f} | --")
    prev = loss_val