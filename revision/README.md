# 🧪 Revision experiments

Code for the experiments added during the paper revision. Each folder holds the
runner and summarizer scripts that produced the reported numbers; data, model
weights, and per-query outputs are not stored here.

| Folder | Experiment |
|---|---|
| `flow_ablation/` | Flow versus graph-score ablation |
| `runtime/` | Runtime and LLM-call accounting |
| `memory/` | Peak GPU memory, ReFlow versus plain inference |
| `larger_models/` | Qwen2.5-32B backbone |
| `new_baselines/` | ContextCite and the adapted MiCE-RAG and RAGE baselines |
| `faithfulness/` | Comprehensiveness and sufficiency |
| `first_order/` | Error of the first-order message contribution |
| `pool_audit/` | Sampling of the replacement pool for its audit |
| `pool_sensitivity/` | Rerun under pools from other editor LLMs |
| `transfer/` | Prompt builders and scoring for transfer to closed-source models |
| `qtz_option_stop/` | QuaRTz rerun with option-level stopping |
| `job_queue/` | GPU job definitions and the queue worker |

> [!NOTE]
> These scripts record how the runs were launched, so they keep the original
> absolute paths and GPU assignments. Review and override them before reuse.
> The portable entry points are in `exp/` and `scripts/`.
