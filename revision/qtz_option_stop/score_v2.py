import json,sys,os,collections
W="/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/"
sys.path.insert(0,W); os.chdir(W)
from causalityrag.reader import canonicalize_reader_answer as C
from causalityrag.evaluation_metrics import answer_changed, clean_correct, correctness_lost, gold_answer_spec, valid_answer
V=W+"out/qtz_option_stop_v2_20261109/"
print("complete",os.path.exists(V+"COMPLETE"))
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
O="pool_sensitivity/qtz_optstop2_miceflip/quartz/"
R=load(O+"reflow_1000.jsonl"); B=load(O+"baselines_1000.jsonl"); K=load(O+"controls_1000.jsonl")
SR={k:load(V+k+"/results.jsonl") for k in ("frontier","grid","one_shot")}
T=load("/data1/yujia/CausalityRAG/out_table3_final8_20260811/quartz/methods/reflow/results_top5_1000.jsonl")
OB=load("/data1/yujia/CausalityRAG/out_table3_final8_20260811/quartz/audits/final_top10pool_k5/baselines_1000.jsonl")
print("budget match",sum(1 for q in R if B[q].get("matched_token_budget")==R[q].get("n_modified_tokens")),"of",len(R))
print("reflow statuses",collections.Counter(r.get("evaluation_status") for r in R.values()).most_common(5),"clean==table3",sum(1 for q in R if R[q].get("clean_answer")==T[q].get("clean_answer")))
ids=[q for q in R if lab(q,R[q].get("clean_answer","")) is not None]
def flip(q,e):
    cl=lab(q,R[q]["clean_answer"]); sp=gold_answer_spec(rec[q],"quartz")
    if not e or str(e.get("status","")).startswith("protocol_violation"): return 0,0,0
    x=e.get("edited_answer","")
    if not valid_answer(x): return 0,0,0
    ed=lab(q,x)
    if ed is None: return 0,0,1
    return int(answer_changed(cl,ed)),int(correctness_lost("em",cl,ed,sp,reader_mode="quartz")),0
el={q:bool(clean_correct("em",lab(q,R[q]["clean_answer"]),gold_answer_spec(rec[q],"quartz"),reader_mode="quartz")) for q in ids}
def score(gF,gC):
    F=L=CF=CE=U=tok=0
    for q in ids:
        e=gF(q); f,l,u=flip(q,e); F+=f; L+=l if el[q] else 0; U+=u; tok+=(e or {}).get("n_modified_tokens") or 0
        if gC:
            cf,c2,_=flip(q,gC(q)); CF+=cf; CE+=c2 if el[q] else 0
    D=len(ids); DE=sum(el.values())
    return dict(ans=round(100*F/D,2),em=round(100*L/DE,2),acfr=round(100*(F-CF)/D,2),ecfr=round(100*(L-CE)/DE,2),F=F,D=D,EL=L,DE=DE,CF=CF,CE=CE,unmapped=U,meantok=round(tok/D,2))
for k,v in SR.items():
    print("search",k,json.dumps(score(lambda q:dict(v[q],status=v[q].get("evaluation_status")),None)),"calls",round(sum((v[q].get("reader_calls") or 0) for q in ids)/len(ids),2))
Fl={k:{q for q in ids if flip(q,dict(v[q],status=v[q].get("evaluation_status")))[0]} for k,v in SR.items()}
print("pairs: both",len(Fl["frontier"]&Fl["grid"]),"reflow_only",len(Fl["frontier"]-Fl["grid"]),"grid_only",len(Fl["grid"]-Fl["frontier"]),"oneshot_only",len(Fl["one_shot"]-Fl["frontier"]))
res={"reflow":score(lambda q:dict(R[q],status=R[q].get("evaluation_status")),lambda q:K[q]["methods"].get("reflow"))}
for k in ("attention","gradient_x_input","ts","mirage","arc_jsd","contextcite","rage","mice_rag"):
    res[k]=score(lambda q:B[q]["methods"].get(k),lambda q:K[q]["methods"].get(k))
rs=[score(lambda q:B[q]["methods"].get("random_seed%d"%i),lambda q:K[q]["methods"].get("random_seed%d"%i)) for i in range(5)]
res["random"]={x:round(sum(r[x] for r in rs)/5,2) for x in ("ans","em","acfr","ecfr")}
for k,v in res.items(): print(k,json.dumps(v))
same=[q for q in R if B[q]["matched_token_budget"]==OB[q]["matched_token_budget"]]
print("same budget",len(same),"changed",1000-len(same))
for k in ("attention","gradient_x_input","mirage","arc_jsd"):
    a=[q for q in same if OB[q]["methods"][k].get("selected_ids")==B[q]["methods"][k].get("selected_ids")]
    eq=sum(1 for q in a if lab(q,OB[q]["methods"][k].get("edited_answer",""))==lab(q,B[q]["methods"][k].get("edited_answer","")))
    print(k,"identical sets",len(a),"same option",eq)
t=[q for q in ids if T[q].get("selected_ids")==R[q].get("selected_ids")]
print("reflow same final set as Table III",len(t),"same option answer",sum(1 for q in t if lab(q,T[q].get("edited_answer",""))==lab(q,R[q].get("edited_answer",""))))
