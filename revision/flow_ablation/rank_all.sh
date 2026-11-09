#!/usr/bin/env bash
# CPU-only: Graph-Score and Single-Flow rankings for all eight datasets.
R=/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1
for ds in hotpotqa musique popqa finqa 2wiki timeqa quartz triviaqa; do
  CFG=$R/out/flow_ablation_full8_20261024/configs/$ds.json
  get() { python3 -c "import json,sys;print(json.load(open('$CFG'))[sys.argv[1]])" "$1"; }
  GRAPHS=$(get graphs) POOL=$(get pool); SHA=$(sha256sum $POOL | cut -d' ' -f1)
  mkdir -p $R/a1_official/$ds
  for m in direct_score individual_flow; do
    [[ -s $R/a1_official/$ds/$m.jsonl ]] && continue
    (cd $R && nice /data1/yujia/envs/graphrag/bin/python scripts/rank_flow_ablations.py --graphs $GRAPHS       --shared-pool $POOL --expected-pool-sha256 $SHA --method $m --out $R/a1_official/$ds/$m.jsonl)       && echo "ranked $ds $m" || echo "FAILED rank $ds $m"
  done
done
echo RANK_ALL_DONE
