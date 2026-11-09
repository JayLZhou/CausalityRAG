import json,glob,os
R='/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/'; V=R+'out/qtz_option_stop_v2_20261109/'
os.makedirs(V+'flow_scores',exist_ok=True)
rows={'direct_score':[],'individual_flow':[]}
for f in sorted(glob.glob(R+'out/flow_ablation_full8_20261024/quartz/ranking_*.json')):
    d=json.load(open(f))
    for k in rows: rows[k].append(d[k])
for k,v in rows.items():
    assert len(v)==1000 and len({r['id'] for r in v})==1000,k
    with open(V+f'flow_scores/{k}.jsonl','w') as f:
        for r in v: f.write(json.dumps(r)+'\n')
s=open(R+'a6_eraser/run_eraser.py').read()
a='    reflow, base_rows, ts_rows = _load_selections(ds)\n'
assert s.count(a)==1
s=s.replace(a,'''    if ds == "quartz":  # option-level rerun: selections at the new matched budgets
        d = f"{P}/qtz_optstop2_miceflip/quartz"
        rows = {r["id"]: r for r in load_records(f"{d}/baselines_1000.jsonl")}
        return load_records(f"{d}/reflow_1000.jsonl"), rows, rows
'''+a)
open(V+'run_eraser_v2.py','w').write(s); print('ok',len(rows['direct_score']))
