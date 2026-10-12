"""tab7_touchup.py - reset the few transient judge_fail rows in the two arms
that fix_arms does NOT rewrite (dense_anchors, no_edge_filter), so the next
ablation_gen run re-judges only those rows. fix_arms rewrites no_rerank and
dense_only itself."""
import json, shutil
R = "/home/devlab1/adgr_workspace/runs/"
shutil.copy(R + "abl_gen.json", R + "abl_gen.json.pre_touchup_bak")
d = json.load(open(R + "abl_gen.json"))
n = 0
for arm in ["dense_anchors", "no_edge_filter"]:
    bad = [k for k, r in d.get(arm, {}).items()
           if r.get("done") and not r.get("all_ok") and not r.get("gen_failed")]
    for k in bad:
        del d[arm][k]
    n += len(bad)
    print(f"{arm}: reset {len(bad)} transient rows", flush=True)
json.dump(d, open(R + "abl_gen.json", "w"), ensure_ascii=False)
print(f"reset {n} rows total; ablation_gen will redo them (~1 min)", flush=True)
