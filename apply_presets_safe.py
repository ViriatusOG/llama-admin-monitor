
import json, os

path = os.path.expanduser("~/.config/llama-admin-monitor/presets.json")
try:
    with open(path, "r") as f:
        data = json.load(f)
        if not type(data) is list: data = []
except Exception:
    data = []

if not data:
    print("Error: No defaults found. Run the server once to generate defaults.")
    exit(1)

# Base new presets off the very first default preset to guarantee they have all required fields for your specific version
p1 = data[0].copy()
p1.update({
    "id": "qwen-27b-iq3-mtp",
    "name": "Qwen 27B IQ3_S MTP (Single GPU)",
    "model_path": "/home/hugo/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
    "devices": "Vulkan0",
    "gpu_layers": 999,
    "context_size": 65536,
    "ctk": "q8_0",
    "ctv": "q8_0",
    "batch_size": 512,
    "ubatch_size": 512,
    "spec_type": "draft-mtp",
    "kv_offload": "",
    "no_warmup": False
})

p2 = data[0].copy()
p2.update({
    "id": "qwen-27b-q8-xl",
    "name": "Qwen 27B Q8_K_XL (Dual GPU)",
    "model_path": "/home/hugo/models/Qwen3.8-27B-UD-Q8_K_XL.gguf",
    "devices": "Vulkan0,Vulkan1",
    "gpu_layers": 999,
    "context_size": 65536,
    "ctk": "q8_0",
    "ctv": "q8_0",
    "batch_size": 512,
    "ubatch_size": 512,
    "spec_type": "",
    "kv_offload": "",
    "no_warmup": False
})

existing_ids = set([p.get("id") for p in data])
for p in [p1, p2]:
    if p["id"] not in existing_ids:
        data.append(p)

with open(path, "w") as f:
    json.dump(data, f, indent=2)

print("Safely added presets!")
