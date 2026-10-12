"""Qwen3-Reranker-0.6B with GPU-first selection and automatic CPU fallback.
- Device is chosen once at load from REAL free VRAM (nvidia-smi, not torch's
  per-process counters): GPU needs >=4GB free, else CPU from the start.
- GPU uses batch=4 (a batch of 16 x 8192tok blew activations past the budget
  and OOMed); CPU uses batch=16.
- If a CUDA OOM hits mid-run (e.g. another job grabs VRAM), the model moves to
  CPU and the batch is retried. Never crashes, never loses a row.
- RERANK_DEVICE env ("cuda"/"cpu") overrides the auto choice.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_model, _tok, _dev = None, None, None
MIN_GPU_FREE_MB = 4000
GPU_BATCH = int(os.environ.get("RERANK_GPU_BATCH", "4"))
CPU_BATCH = 16


def _free_mb():
    import subprocess
    try:
        out = subprocess.run("nvidia-smi --query-gpu=memory.total,memory.used "
                             "--format=csv,noheader,nounits", shell=True,
                             capture_output=True, text=True).stdout.split()
        return int(out[0].strip().rstrip(",")) - int(out[1].strip().rstrip(","))
    except Exception:
        pass
    if not torch.cuda.is_available():
        return 0
    t = torch.cuda.get_device_properties(0).total_memory
    u = torch.cuda.memory_allocated(0) + torch.cuda.memory_reserved(0)
    return (t - u) / 1024 ** 2


def _to_cuda(model):
    if not torch.cuda.is_available():
        raise RuntimeError("no cuda")
    model.to("cuda")


def load():
    global _model, _tok, _dev
    if _model is None:
        ck = os.environ.get("QWEN_RERANKER_CK", "Qwen/Qwen3-Reranker-0.6B")
        _tok = AutoTokenizer.from_pretrained(ck, padding_side="left")
        try:
            _model = AutoModelForCausalLM.from_pretrained(ck, dtype="auto").eval()
        except Exception:
            _model = AutoModelForCausalLM.from_pretrained(ck, torch_dtype="float32").eval()
        free = _free_mb()
        force = os.environ.get("RERANK_DEVICE", "").lower()
        _dev = "cpu"
        if force in ("cuda", "cpu"):
            _dev = force
        elif torch.cuda.is_available() and free >= MIN_GPU_FREE_MB:
            try:
                _to_cuda(_model)
                _dev = "cuda"
            except Exception as e:
                _dev = "cpu"
                try:
                    _model.to("cpu")
                except Exception:
                    pass
                print(f"[rerank] GPU load failed -> CPU: {repr(e)[:70]}", flush=True)
        print(f"[rerank] device={_dev} free_gpu_mb={free:.0f}", flush=True)
    return _model, _tok, _dev


def rerank(query, docs, batch=None):
    global _dev
    model, tok, dev = load()
    if batch is None:
        batch = GPU_BATCH if dev == "cuda" else CPU_BATCH
    yes, no = tok.encode("yes")[-1], tok.encode("no")[-1]
    scores = []
    i = 0
    while i < len(docs):
        chunk = docs[i:i + batch]
        texts = [f"<Instruct>: Xep hang doan van theo muc do lien quan\n"
                 f"<Query>: {query}\n<Document>: {d}" for d in chunk]
        inp = tok(texts, padding=True, truncation=True, max_length=8192,
                  return_tensors="pt").to(dev)
        try:
            with torch.no_grad():
                logits = model(**inp).logits[:, -1, :]
        except Exception as e:
            if dev == "cuda":
                print(f"[rerank] CUDA OOM mid-run -> CPU fallback: {repr(e)[:60]}",
                      flush=True)
                try:
                    model.to("cpu")
                except Exception:
                    pass
                _dev = "cpu"
                dev = "cpu"
                continue
            raise
        p = torch.softmax(logits[:, [no, yes]], dim=-1)[:, 1]
        scores.extend(p.float().cpu().tolist())
        i += batch
    return scores
