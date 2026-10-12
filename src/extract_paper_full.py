"""FULL extraction on the AUTHOR's H2 chunker (15,837 chunks, 91.9% of paper's 17,236).

No arbitrary caps:
  - input : the WHOLE chunk (author chunks reach 6,895 chars; the [:3000] cut that
    the 11k run used would silently drop 4.3% of content there)
  - output: max_tokens is DERIVED from the server's max_model_len (12288) minus
    the measured prompt size, not a fixed 2000. Prompt = 4,018 tok, worst-case
    chunk = 2,300 tok -> 5,970 tok available, so 5,000 is safe and generous.
Same prompt (author's prompt.py), same model, temperature 0, checkpointed per 32.
Measured pace: 12.32 chunks/min while sharing the GPU, ~19.7 solo -> ~13.4h.
Needs the extraction prompts tmp/ext_sys.txt, tmp/ext_user.txt, tmp/ext_ex.txt
(author's extraction prompt + examples) and a vLLM server on :8000."""
import json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
R = "/home/devlab1/adgr_workspace/runs/"
W = "/home/devlab1/adgr_workspace"
VLLM = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = f"{W}/models/qwen25-awq"
MAXLEN = int(os.environ.get("MAX_MODEL_LEN", "12288"))
OUT_TOK = int(os.environ.get("OUT_TOK", "5000"))
TYPES = ["Disease", "Symptom", "Drug", "Chemical compound", "Protein", "Anatomy",
         "Biological process", "Exposure", "Diagnostic test", "Treatment method"]
SYS = open(f"{W}/tmp/ext_sys.txt").read()
USER = open(f"{W}/tmp/ext_user.txt").read()
EX = open(f"{W}/tmp/ext_ex.txt").read()


def fmt(s):
    return (s.replace("{entity_types}", str(TYPES))
            .replace("{tuple_delimiter}", "<|#|>")
            .replace("{completion_delimiter}", "<|COMPLETE|>")
            .replace("{language}", "Vietnamese").replace("{examples}", EX))


PROMPT_TOK = (len(SYS) + len(USER) + len(EX)) // 3
print(f"max_model_len={MAXLEN} prompt~{PROMPT_TOK}tok out_budget={OUT_TOK}", flush=True)

CH = f"{R}/chunks_paper.json"
OUT = f"{R}/extract_paper_full.json"
chunks = json.load(open(CH))
print("paper chunks:", len(chunks), flush=True)
try:
    out = json.load(open(OUT))
except Exception:
    out = {}
print("resume:", len(out), flush=True)


def one(ci):
    c = chunks[ci]
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": fmt(SYS)},
                         {"role": "user", "content": USER.replace("{entity_types}", str(TYPES))
                          .replace("{language}", "Vietnamese")
                          .replace("{input_text}", c["content"])}],
            "temperature": 0, "max_tokens": OUT_TOK}
    for _ in range(3):
        try:
            req = urllib.request.Request(VLLM, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1200) as r:
                d = json.load(r)
            o = d["choices"][0]["message"]["content"] or ""
            return ci, {"ents": [l for l in o.split("\n") if l.startswith("entity<|#|>")],
                        "rels": [l for l in o.split("\n") if l.startswith("relation<|#|>")],
                        "out_tok": d.get("usage", {}).get("completion_tokens"),
                        "fin": d["choices"][0].get("finish_reason"),
                        "tok": c.get("tokens", 0), "done": True}
        except Exception:
            time.sleep(10)
    return ci, {"error": "failed3x", "done": True}


todo = [i for i in range(len(chunks)) if str(i) not in out or not out[str(i)].get("done")]
print("todo:", len(todo), flush=True)
t0 = time.time()
for b in range(0, len(todo), 32):
    with ThreadPoolExecutor(max_workers=16) as ex:
        for ci, row in ex.map(one, todo[b:b + 32]):
            out[str(ci)] = row
    json.dump(out, open(OUT, "w"), ensure_ascii=False)
    done = len(out)
    rate = done / ((time.time() - t0) / 60) if time.time() > t0 else 0
    if (b // 32) % 20 == 0:
        print(f"[{b+32}/{len(todo)}] total={done}/{len(chunks)} "
              f"{rate:.1f}/min ETA={(len(chunks)-done)/max(rate,0.1)/60:.2f}h", flush=True)
print("done", len(out), "/", len(chunks))
