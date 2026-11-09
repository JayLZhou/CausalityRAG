import json,sys,os,collections
import numpy as np
W="/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/"
sys.path.insert(0,W); os.chdir(W)
from causalityrag.reader import canonicalize_reader_answer as C
from causalityrag.evaluation_metrics import answer_changed, clean_correct, correctness_lost, gold_answer_spec, valid_answer
V=W+"out/qtz_option_stop_v2_20261109/"
print({f:os.path.exists(V+f) for f in ("FOLLOWUPS_COMPLETE","ERASER_COMPLETE")})
for q,f in (("0","qtz_followups4"),("1","qtz_eraser")):
    print(open(f"queue/status_gpu{q}.log").read().strip().splitlines()[-1][:70])
def load(p):
    d={}
    for l in open(p):
        r=json.loads(l); d[r["id"]]=r
    return d
m=json.load(open("out/flow_ablation_full8_20261024/quartz/manifest.json"))["config"]; rec=load(m["input"])
def lab(q,x):
    if not valid_answer(x): return None
    try: return C(rec[q]["question"],x,reader_mode="quartz")
    except Exception: return None
R=load(V+"frontier/results.jsonl"); ids=[q for q in R if lab(q,R[q].get("clean_answer","")) is not None]
F=load(V+"flow_ablation.jsonl")
el={q:bool(clean_correct("em",lab(q,R[q]["clean_answer"]),gold_answer_spec(rec[q],"quartz"),reader_mode="quartz")) for q in ids}
print("flow budget match",sum(1 for q in R if F[q].get("matched_token_budget")==R[q].get("n_modified_tokens")),"methods",list(next(iter(F.values()))["methods"]))
for k in ("direct_score","individual_flow"):
    a=l=0
    for q in ids:
        e=F[q]["methods"].get(k) or {}
        if str(e.get("status","")).startswith("protocol_violation"): continue
        ed=lab(q,e.get("edited_answer",""))
        if ed is None: continue
        cl=lab(q,R[q]["clean_answer"]); a+=int(answer_changed(cl,ed))
        if el[q]: l+=int(correctness_lost("em",cl,ed,gold_answer_spec(rec[q],"quartz"),reader_mode="quartz"))
    print(k,"ans",a,len(ids),round(100*a/len(ids),2),"em",l,sum(el.values()),round(100*l/sum(el.values()),2))
rows=[json.loads(l) for l in open(V+"suff/per_query.jsonl")]; ts=load(V+"suff/per_query_ts.jsonl")
print("suff rows",len(rows),"ts",len(ts),json.load(open(V+"suff/summary.json"))["population"])
def lab2(q,x):
    if not valid_answer(x): return None
    try: return C(q,x,reader_mode="quartz")
    except Exception: return None
Ms=["Full","ReFlow","MIRAGE","ARC-JSD"]; opt={k:[] for k in Ms+["TS"]}; bt=0
for r in rows:
    q=r["question"]; g=lab2(q,r["gold_answer"])
    for k in Ms: opt[k].append(float(g is not None and lab2(q,r["predictions"][k])==g))
    t=ts.get(r["id"])
    if t is None: bt+=1; opt["TS"].append(0.0)
    else: opt["TS"].append(float(g is not None and lab2(q,t["prediction"])==g))
print("missing ts",bt)
n=len(rows); idx=np.random.default_rng(20260804).integers(0,n,size=(10000,n))
for k,v in opt.items():
    a=np.array(v); b=a[idx].mean(1); lo,hi=np.percentile(b,[2.5,97.5]); print("suff",k,"%.2f +%.2f -%.2f"%(100*a.mean(),100*(hi-a.mean()),100*(a.mean()-lo)))
a=np.array(opt["ReFlow"]); b=np.array(opt["ARC-JSD"]); d=(a-b)[idx].mean(1); print("ReFlow-ARC %.2f [%.2f,%.2f]"%(100*(a-b).mean(),*(100*np.percentile(d,[2.5,97.5]))))
def agg(path):
    rs=[json.loads(l) for l in open(path)]
    print(path[-40:],len(rs),"keys",sorted(rs[0])[:12])
    meth=rs[0].get("methods") or rs[0].get("scores")
    print(" methods",list(meth)[:14],{k:(round(v,4) if isinstance(v,float) else v) for k,v in list(meth.items())[0][1].items()} if meth else "")
    out=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rs:
        for k,v in (r.get("methods") or r.get("scores")).items():
            if not isinstance(v,dict): continue
            for mk in ("comprehensiveness","comprehensiveness_replace","sufficiency"):
                if isinstance(v.get(mk),(int,float)): out[k][mk].append(v[mk])
    for k,v in out.items():
        print(" ",k,{mk:(round(100*sum(x)/len(x),1),len(x)) for mk,x in v.items()})
rs=[json.loads(l) for l in open(V+"eraser_quartz.jsonl")]
out=collections.defaultdict(lambda:collections.defaultdict(list)); pf=[]
for r in rs:
    pf.append(r["p_full"])
    for k,v in r["methods"].items():
        if not isinstance(v,dict): continue
        for mk in ("comprehensiveness","comprehensiveness_replace","sufficiency"):
            if isinstance(v.get(mk),(int,float)): out[k][mk].append(v[mk])
print("eraser queries",len(rs),"p_full",round(sum(pf)/len(pf),4))
for k,v in out.items():
    print("eraser",k,{mk:(round(100*sum(x)/len(x),2),len(x)) for mk,x in v.items()})
