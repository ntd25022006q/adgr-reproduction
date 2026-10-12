"""Full crawl: 1,935 ViMedAQA article URLs -> cleaned docs. Checkpointed, polite."""
import json, re, time, urllib.request, html as ihtml
URLS = json.load(open("/home/devlab1/adgr_workspace/data/vimed_urls.json"))
OUT = "/home/devlab1/adgr_workspace/runs/crawl_full.json"
try:
    done = {d["url"]: d for d in json.load(open(OUT)) if isinstance(d, dict) and "url" in d}
except Exception:
    done = {}
print("resume:", len(done), flush=True)


def fetch(u):
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (academic research; rate-limited)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "ignore")


def clean(h):
    h = re.sub(r"(?s)<script.*?</script>|<style.*?</style>|<nav.*?</nav>|<footer.*?</footer>|<header.*?</header>", " ", h)
    h = re.sub(r"(?s)<!--.*?-->", " ", h)
    heads = []
    for tag in ["h1", "h2", "h3"]:
        for m in re.finditer(rf"<{tag}[^>]*>(.*?)</{tag}>", h, flags=re.S):
            t = ihtml.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m.group(1)))).strip()
            if len(t) > 3:
                heads.append((tag, t))
    txt = ihtml.unescape(re.sub(r"[ \t\xa0]+", " ", re.sub(r"<[^>]+>", " ", h)))
    paras = [p.strip() for p in txt.split("\n") if len(p.strip()) > 40]
    title = heads[0][1] if heads else ""
    return title, heads, paras


for i, u in enumerate(URLS):
    if u in done:
        continue
    try:
        h = fetch(u)
        title, heads, paras = clean(h)
        done[u] = {"url": u, "title": title, "heads": heads[:60], "paras": paras[:120],
                   "n_paras": len(paras)}
        print(f"[{len(done)}/{len(URLS)}] p={len(paras)}", flush=True)
    except Exception as e:
        done[u] = {"url": u, "error": repr(e)[:120]}
        print(f"[{len(done)}/{len(URLS)}] FAIL {repr(e)[:60]}", flush=True)
    json.dump(list(done.values()), open(OUT, "w"), ensure_ascii=False)
    time.sleep(1)
print("done", len(done), "ok:", sum(1 for d in done.values() if "error" not in d))
