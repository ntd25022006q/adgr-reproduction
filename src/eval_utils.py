"""Shared hardened eval utils. ALL judge/gen scripts must import from here.
Rules: validate everything, retry with backoff, save raw outputs, never feed
empty answers to judges, force Vietnamese replies."""
import json, re, time, hashlib

VI = "\nTra loi HOAN TOAN bang tieng Viet (khong lan English/Indonesian). Chi tra ve JSON duy nhat, khong giai thich."


def kv(j, key):
    m = re.search(r'["\']' + key + r'["\']\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)', j or "")
    return float(m.group(1)) if m else None


def valid_ans(a, min_len=20):
    return bool(a) and len(a.strip()) >= min_len


def prompt_hash(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def ogen_retry(ogen_fn, prompt, tries=3):
    last = ""
    for i in range(tries):
        last = ogen_fn(prompt) or ""
        if valid_ans(last):
            return last, i
        time.sleep(2)
    return last, tries


def judge_retry(judge_fn, prompt, keys, tries=3):
    prompt = prompt + VI
    last = ""
    for i in range(tries):
        last = judge_fn(prompt) or ""
        vals = [(k, kv(last, k)) for k in keys]
        if all(v is not None and (k in ("supported", "total", "useful") or 0 <= v <= 1)
               for k, v in vals):
            return last, i
        time.sleep(3)
    return last, tries
