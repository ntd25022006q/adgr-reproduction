"""Rewrite vllm_32k.py -> vllm_restart.py (parameterized, with all fixes).
Fixes vs the deployed copy:
  + --enforce-eager (without it: EngineCore failed, proven 3 times)
  + wait_gpu_release() (without it: 'No available memory' race)
  + env VLLM_UTIL / VLLM_LEN / VLLM_LOG so the SAME script serves both the
    32k upgrade (before baselines_gen) and the 12k downgrade (before e2e_paper).
"""
import subprocess, sys, os, time, urllib.request, signal, fcntl
W = "/home/devlab1/adgr_workspace"
# NOTE: miniforge lives in $HOME, NOT under the workspace. A wrong W-based
# path here silently kills every server launch (shell "not found" goes to the
# captured sh() output) - this exact bug made all 32k boots die with empty logs.
P = "/home/devlab1/miniforge3/envs/unsloth/bin/python"
R = f"{W}/runs"
# ---- single-flight lock: two restart scripts racing (a stale one from a
# killed chain + the current stage) stole each other's VRAM - the 12k boot
# died with "Engine core initialization failed" while the stale 32k script
# grabbed the freed memory. Block until the other finishes, then exit 0 if
# the server we want is already up.
_LOCK = open(f"{R}/vllm_restart.lock", "w")
_WANT = os.environ.get("VLLM_LEN", "32768")


def _want_up():
    ps = os.popen("ps -eo cmd").read()
    if "vllm.entrypoints.openai.api_server" not in ps:
        return False
    if f"max-model-len {_WANT}" not in ps:
        return False
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


try:
    fcntl.flock(_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
except (BlockingIOError, OSError):
    print("another vllm_restart is running; waiting for it...", flush=True)
    fcntl.flock(_LOCK, fcntl.LOCK_EX)
    if _want_up():
        print(f"desired server (maxlen {_WANT}) already up; exiting 0", flush=True)
        sys.exit(0)
    print("waiter: desired server NOT up after other restart finished; "
          "acquiring lock and proceeding", flush=True)
    _LOCK.close()
    _LOCK = open(f"{R}/vllm_restart.lock", "w")
    fcntl.flock(_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
UTIL = os.environ.get("VLLM_UTIL", "0.72")
MLEN = os.environ.get("VLLM_LEN", "32768")
LOG = os.environ.get("VLLM_LOG", f"{R}/vllm32k.log")
# VRAM guard: 32k needs ~17GB free, 12k needs ~15.5GB free (weights 8GB +
# KV + eager overhead). If the external job spikes, WAIT instead of OOM.
NEED_FREE_MB = int(os.environ.get("VLLM_NEED_FREE_MB",
                                  "17500" if MLEN == "32768" else "16000"))
print(f"vLLM restart: util={UTIL} maxlen={MLEN} need_free={NEED_FREE_MB}MB", flush=True)
if _want_up():
    # idempotent: the desired server (same maxlen, healthy) is already up -
    # a chain re-run must not kill a good server just to rebuild it.
    print(f"server with maxlen {_WANT} already healthy; exiting 0", flush=True)
    sys.exit(0)


def wait_for_headroom(timeout_s=1800):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        out = subprocess.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader",
                             shell=True, capture_output=True, text=True).stdout.strip()
        try:
            free_mb = int(out.split()[0])
        except Exception:
            free_mb = 0
        if free_mb >= NEED_FREE_MB:
            print(f"  headroom OK: {free_mb}MB free", flush=True)
            return True
        print(f"  waiting headroom: {free_mb}MB free < {NEED_FREE_MB}MB need",
              flush=True)
        time.sleep(60)
    print("  WARN: headroom timeout, launching anyway (may OOM)", flush=True)
    return False


def sh(cmd):
    print(f"RUN {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def wait_gpu_release(timeout_s=240):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        vllm_left = any("vllm.entrypoints" in l
                        for l in subprocess.run(
                            "ps -eo cmd", shell=True,
                            capture_output=True, text=True).stdout.splitlines())
        if not vllm_left:
            print("  old vLLM fully released VRAM", flush=True)
            return True
        time.sleep(5)
    print("  WARN: timeout waiting for release, continuing anyway", flush=True)
    return False


env = ("export PATH=$HOME/miniforge3/envs/unsloth/bin:$PATH "
       "CC=$HOME/miniforge3/bin/x86_64-conda-linux-gnu-gcc "
       "CXX=$HOME/miniforge3/bin/x86_64-conda-linux-gnu-g++; ")
pids = [l.split()[0] for l in os.popen("ps -eo pid,cmd").read().splitlines()
        if "vllm.entrypoints.openai.api_server" in l]
for p in pids:
    try:
        os.kill(int(p), signal.SIGTERM)
        print("stopped vllm pid", p, flush=True)
    except Exception as e:
        print("kill err", e, flush=True)
wait_gpu_release()
time.sleep(10)
wait_for_headroom()
sh(f"{env}cd {W} && setsid nohup {P} -u -m vllm.entrypoints.openai.api_server "
   f"--model {W}/models/qwen25-awq --quantization awq --gpu-memory-utilization {UTIL} "
   f"--max-model-len {MLEN} --port 8000 --trust-remote-code --enforce-eager "
   f"> {LOG} 2>&1 < /dev/null & echo serving")
for i in range(120):
    time.sleep(10)
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10) as r:
            if r.status == 200:
                print(f"vLLM ({MLEN}) up after {(i+1)*10}s", flush=True)
                break
    except Exception:
        pass
else:
    print("vLLM did NOT come up - chain will see failures", flush=True)
    sys.exit(1)
