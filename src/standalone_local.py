"""Standalone answers from the 2 local 14B models (Tab.5 models #6-7).
qwen2.5:14b and qwen3:14b via ollama. Each needs ~9GB VRAM, so this runs ONLY
after vLLM stops (chain calls stop_vllm first). Short direct answers, temp 0.
Checkpointed. API-free."""
import json, time, csv, urllib.request
MODELS = [("qwen2.5-14b", "qwen2.5:14b"), ("qwen3-14b", "qwen3:14b")]
P = "Tra loi cau hoi y khoa sau day ngan gon, chinh xac.\nCau hoi: {q}\nTra loi:"
OUT = "/home/devlab1/adgr_workspace/runs/standalone.json"
rows = list(csv.DictReader(open("/home/devlab1/adgr_workspace/data/q300.csv")))
out = json.load(open(OUT))


def gen(model, prompt):
    # num_ctx=4096: the standalone answer is ~300 tokens and the question ~200.
    # Ollama's default ctx for the 14B models is 32768, which reserves ~18GB of
    # KV cache (we saw qwen3:8b occupying 10GB at 32k ctx). 4096 keeps it ~9GB.
    body = {"model": model, "prompt": prompt, "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": 300, "num_ctx": 4096}}
    for _ in range(3):
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/api/generate",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                t = (json.load(r).get("response") or "").strip()
            if len(t) >= 20:
                return t[:800]
        except Exception:
            time.sleep(5)
    return ""


for name, model in MODELS:
    todo = [qi for qi in range(len(rows))
            if not (out.get(str(qi), {}).get(name) or "").strip()]
    print(f"[{name}] todo {len(todo)}/{len(rows)}", flush=True)
    # 2 workers (was: sequential + sleep(1), ~50 min per 600 rows). Ollama
    # serves 2 concurrent same-model gens fine; ~25 min total. Lock guards
    # the shared out dict + checkpoint every 20 completions.
    import threading as _th
    from concurrent.futures import ThreadPoolExecutor as _TPE
    _lock = _th.Lock()
    _n = [0]

    def one(qi, _nm=name, _mo=model):
        t = gen(_mo, P.format(q=rows[qi]["question"]))
        with _lock:
            out.setdefault(str(qi), {})[_nm] = t
            _n[0] += 1
            if _n[0] % 20 == 0:
                json.dump(out, open(OUT, "w"), ensure_ascii=False)
                print(f"  [{_nm}] {_n[0]}/{len(todo)}", flush=True)

    with _TPE(max_workers=2) as _ex:
        list(_ex.map(one, todo))
    json.dump(out, open(OUT, "w"), ensure_ascii=False)
    print(f"[{name}] done",
          sum(1 for r in out.values() if (r.get(name) or "").strip()), "/300")
print("local standalone complete")
