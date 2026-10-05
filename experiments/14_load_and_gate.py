"""Load every adapter needed by the matched run, then gate all of them.

Two hazards this guards against, both observed on this box:

  1. POSTing /v1/load_lora_adapter with a path that has no
     adapter_model.safetensors returns a clean 404 but POISONS the engine:
     every later request 500s and the server dies with a GPU memory access
     fault. So the file is checked to exist BEFORE the request is made.

  2. vLLM returns HTTP 200 for an adapter that applies ZERO weights. An inert
     adapter scores exactly like the base model, which in this experiment
     would read as "this memory method does not work". So every adapter must
     be shown to move logprobs before it is allowed into the evaluation.
"""
import json
import os
import sys
import urllib.error
import urllib.request

URL = "http://localhost:8106"
BASE = "qwen4bi"
PROBE = ("<|im_start|>user\nWhat is Hsiao Yun-Hwa's father's occupation?<|im_end|>\n"
         "<|im_start|>assistant\nHsiao Yun-Hwa's father is a civil engineer.")

ALL = [("sgd_pa%02d" % i, "/work/d2l_exp/sgd_perauthor/author%02d" % i)
       for i in range(10)]
ALL += [("sgd_pam%02d" % i, "/work/d2l_exp/sgd_perauthor_matched/author%02d" % i)
        for i in range(10)]
ALL += [("d2l_a%02d" % i, "/work/d2l_exp/adapters/author%02d" % i)
        for i in range(10)]
ALL += [("sgd7", "/work/d2l_exp/sgd_7mod"),
        ("sgddp", "/work/d2l_exp/sgd_dp"),
        ("d2l_all200", "/work/d2l_exp/adapters/all200")]

# Granularity ladder. Names come from the build stats rather than being
# re-derived here, so the gate can never test a different adapter set than the
# one that was actually built.
_GRAN_STATS = "/work/d2l_exp/gran_build_stats.json"
if os.path.exists(_GRAN_STATS):
    _gs = json.load(open(_GRAN_STATS))
    for _g, _v in sorted(_gs["granularities"].items(), key=lambda kv: int(kv[0])):
        for _name, _a in _v["adapters"].items():
            ALL.append(("gr_" + _name, _a["path"]))

# Optional name-prefix filter, so a newly built set can be gated without
# re-gating everything already verified in this server's lifetime.
PREFIX = sys.argv[1] if len(sys.argv) > 1 else ""
WANT = [(n, p) for n, p in ALL if n.startswith(PREFIX)]
GATE_OUT = "/work/d2l_exp/activity_gate_%s.json" % (PREFIX or "all")


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
        print("%-12s already loaded" % name, flush=True)
        continue
    sf = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(sf):
        raise SystemExit("MISSING %s -- refusing to POST (engine-poisoning guard)"
                         % sf)
    code, body = post("/v1/load_lora_adapter",
                      {"lora_name": name, "lora_path": path})
    print("%-12s load HTTP %s %s" % (name, code, body.strip()[:60]), flush=True)
    if code != 200:
        raise SystemExit("load failed for %s" % name)

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
    print("%-12s delta %10.5f  %s" % (name, delta, "ACTIVE" if active else "INERT"),
          flush=True)
    if not active:
        bad.append(name)

with open(GATE_OUT, "w") as f:
    json.dump(out, f, indent=2)

if bad:
    print("GATE_FAIL:", bad, flush=True)
    sys.exit(2)
print("GATE_OK: all %d adapters active" % len(WANT), flush=True)
