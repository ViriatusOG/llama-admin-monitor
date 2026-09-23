
import json, os

path = os.path.expanduser("~/.config/llama-admin-monitor/presets.json")
try:
    with open(path, "r") as f:
        data = json.load(f)
        if not type(data) is list: data = []
except Exception:
    data = []

with open("docs/hardware_presets.json", "r") as f:
    new_presets = json.load(f)

existing_ids = set([p.get("id") for p in data])
for p in new_presets:
    if p["id"] not in existing_ids:
        data.append(p)

os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(data, f, indent=2)

print("Presets successfully added!")
