"""Load and activity-gate every adapter in the gradient sweep.

Same two hazards guarded as 14_load_and_gate.py, both observed on this box:

  1. POSTing /v1/load_lora_adapter with a path lacking adapter_model.safetensors
     returns a clean 404 but POISONS the engine: later requests 500 and the
     server dies with a GPU memory access fault. So the file is checked to
     exist BEFORE the request.

  2. vLLM returns HTTP 200 for an adapter that applies ZERO weights. An inert
     adapter scores exactly like the base model, which in a granularity
     experiment would read as "coarse partitions do not work". Every adapter
     must be shown to move logprobs before it enters the evaluation.

Adapter names and paths come from the build stats, never re-derived, so the
gate cannot test a different set than was trained.
"""
import json
import os
import sys
import urllib.error
import urllib.request

# 8107, not the frozen run's 8106: see 26_serve_sgd_sweep.sh. The 8106 server is
# may still be running and holds none of these adapters, so gating against it would
# either 404 every load or, worse, pass a gate on the wrong engine.
URL = os.environ.get("SWEEP_URL", "http://localhost:8107")
BASE = "qwen4bi"
PROBE = ("<|im_start|>user\nWhat is Hsiao Yun-Hwa's father's occupation?<|im_end|>\n"
         "<|im_start|>assistant\nHsiao Yun-Hwa's father is a civil engineer.")
STATS = sys.argv[1] if len(sys.argv) > 1 else "/work/d2l_exp/sgd_gran_build_stats.json"
# Derived from the stats filename so gating the exact-249 ladder cannot overwrite
# the exposure sweep's gate record. Both ladders must keep their own evidence.
GATE_OUT = "/work/d2l_exp/activity_gate_%s.json" % (
    os.path.basename(STATS).replace("_build_stats.json", "").replace(".json", ""))

build = json.load(open(STATS))
WANT = []
for ckey, c in sorted(build["conditions"].items()):
    for pname, a in c["adapters"].items():
        WANT.append((a["lora_name"], a["path"]))
print("gating %d adapters from %d conditions" % (len(WANT), len(build["conditions"])),
      flush=True)


def post(path, payload):
    req = urllib.request.Request(
        URL + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.read().decode()[:200]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]


def logprobs(model, text):
    req = urllib.request.Request(
        URL + "/v1/completions",
        data=json.dumps({"model": model, "prompt": text, "max_tokens": 0,
                         "echo": True, "prompt_logprobs": 0,
                         "temperature": 0.0}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        pl = json.load(r)["choices"][0]["prompt_logprobs"]
    return [list(d.values())[0]["logprob"] for d in pl if d]


loaded = set(m["id"] for m in json.load(
    urllib.request.urlopen(URL + "/v1/models", timeout=60))["data"])

for name, path in WANT:
    if name in loaded:
        continue
    sf = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(sf):
        raise SystemExit("MISSING %s -- refusing to POST (engine-poisoning guard)"
                         % sf)
    code, body = post("/v1/load_lora_adapter",
                      {"lora_name": name, "lora_path": path})
    if code != 200:
        raise SystemExit("load failed for %s: HTTP %s %s" % (name, code, body))
print("loaded all", flush=True)

b1 = logprobs(BASE, PROBE)
b2 = logprobs(BASE, PROBE)
n = min(len(b1), len(b2))
floor = max(abs(b1[i] - b2[i]) for i in range(n))
print("noise floor (base vs base) = %.10f" % floor, flush=True)

out = {"noise_floor": floor, "adapters": {}}
bad = []
for name, _ in WANT:
    v = logprobs(name, PROBE)
    m = min(n, len(v))
    delta = max(abs(b1[i] - v[i]) for i in range(m))
    active = delta > max(1e-6, floor * 10)
    out["adapters"][name] = {"delta": delta, "active": active}
    if not active:
        bad.append(name)
    print("%-32s delta %10.5f  %s" % (name, delta, "ACTIVE" if active else "INERT"),
          flush=True)

out["n_adapters"] = len(WANT)
out["n_active"] = len(WANT) - len(bad)
with open(GATE_OUT, "w") as f:
    json.dump(out, f, indent=2)

if bad:
    print("GATE_FAIL:", bad, flush=True)
    sys.exit(2)
print("GATE_OK: all %d adapters active" % len(WANT), flush=True)
