"""Run the AUTHOR's own semantic chunker on our ordered-markdown crawl.
Purpose: verify our chunk-count/discrepancy against the paper's 17,236 chunks
(1,932 docs, 8.92 chunks/doc) using the author's exact H2 algorithm instead of
our soft-cap approximation. Writes chunks/params + stats for the report.

NOTE (external dependency, not vendored): this script imports the authors'
chunking code (lightrag.chunking_semantic + lightrag.utils.Tokenizer) from
/home/devlab1/thesis_src and a HuggingFace tokenizer, and reads
runs/crawl_md.json. It documents exactly how chunks_paper.json (15,837 chunks,
the rebuilt index behind Tables 1/3/4/6/7/8) was produced. Our own
approximation of the same H2 rule is described in the report; the 91.9% match
is measured by chunk count."""
import sys, json, os, statistics as st
sys.path.insert(0, "/home/devlab1/thesis_src")
from lightrag.chunking_semantic import semantic_markdown_chunker
from lightrag.utils import Tokenizer
from tokenizers import Tokenizer as HFTok

OUT = "/home/devlab1/adgr_workspace/runs"
docs = [d for d in json.load(open(f"{OUT}/crawl_md.json"))
        if d.get("md") and len(d["md"]) > 200]
print("docs available:", len(docs), "(of 1935 URL list)", flush=True)
tok = Tokenizer("qwen2.5", __import__("transformers").AutoTokenizer.from_pretrained(
    os.path.expanduser("~/adgr_workspace/models/qwen25-awq")))
allc = []
for i, d in enumerate(docs):
    try:
        c = semantic_markdown_chunker(tok, d["md"])
        for x in c:
            allc.append({"doc": i, "url": d["url"], "content": x["content"],
                         "tokens": x.get("tokens", 0),
                         "order": x.get("chunk_order_index", 0)})
    except Exception as e:
        print("  chunker error", d.get("url", "")[:60], repr(e)[:80], flush=True)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(docs)} chunks so far {len(allc)}", flush=True)
print(f"\ndocs={len(docs)} chunks={len(allc)} chunks/doc={len(allc)/len(docs):.2f} "
      f"(paper 8.92 on 1932 docs -> 17236)")
t = [x["tokens"] for x in allc]
print(f"chunk tokens: mean={st.mean(t):.0f} median={st.median(t):.0f} max={max(t)}")
print(f"docs with 0 chunks: {sum(1 for i,d in enumerate(docs))- len({x['doc'] for x in allc})}")
json.dump(allc, open(f"{OUT}/chunks_paper.json", "w"), ensure_ascii=False)
json.dump({"n_docs": len(docs), "n_chunks": len(allc),
           "chunks_per_doc": round(len(allc) / len(docs), 3),
           "paper_chunks_per_doc": 8.92,
           "mean_tokens": round(st.mean(t), 1), "max_tokens": max(t),
           "note": "author semantic_markdown_chunker, H2 split, 1950-tok guard-rail, 0 overlap"},
          open(f"{OUT}/chunk_paper_stats.json", "w"), indent=1)
print("wrote chunks_paper.json + chunk_paper_stats.json")
