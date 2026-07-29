# Experiment runners

- `run_reflow.py` computes the exact supported flow-contract frontier on the
  editable domain declared by a frozen shared replacement pool.
- `arc_jsd.py` contains the ARC-JSD implementation, and `run_arc_jsd.py`
  produces its sentence-to-token rankings without editing context.
- `run_gradient_attribution_baselines.py` produces Gradient x Input and
  Integrated Gradients token rankings.

These runners select or rank token IDs only. Counterfactual replacement and
reader verification are centralized in `scripts/evaluate_reflow.py` and
`scripts/evaluate_matched_budget_baselines.py`, ensuring that all methods use
the same frozen position-level pool.
