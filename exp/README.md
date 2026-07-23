# Experimental and historical code

This directory contains code that is not part of the stable final pipeline.
Files are retained for provenance, baseline reproduction, ablations, solver
audits, and paper figures.

Run commands from the repository root, for example:

```bash
python exp/run_mirage_topk_attack.py --help
```

## Contents

- `run_*_attack.py`: historical attack and graph-selection baselines.
- `build_unary_support.py`: ARC-JSD unary support used by an abandoned hybrid
  variant; it is not part of the final pure contribution-graph pipeline.
- `run_*_ablation.py`: attention and contribution-flow ablations.
- `prepare_*.py`: materialize size-matched or ablation interventions.
- `collect_*.py`: diagnostic data collection.
- `augment_*.py`: post-processing for staged diagnostics.
- `plot_*.py`: figure generation.
- `analyze_group_flow_oracle.py` and `group_flow_oracle.py`: historical
  grouped layer-copy MILP audit.
- `mirage.py` and `run_mirage_topk_attack.py`: MIRAGE baseline.
- `run_token_ilp.py` and `run_token_attack.py`: original token-ILP pipeline.
- `EXPERIMENTS_HOTPOTQA.md`: the earlier broad experimental protocol.
- `RESULTS_HOTPOTQA_1000.md`: the earlier hybrid HotpotQA result report.
- `verify_hf_results.py`: historical local-HF cross-check from the earlier
  dual-reader workflow; final metrics now come directly from vLLM.

Experimental tests live beside the corresponding implementation:

```bash
python -m pytest -q exp
```

The stable commands, current method, fixed-point registry procedure, and
end-to-end run instructions are documented in the top-level
[README](../Readme.md).
