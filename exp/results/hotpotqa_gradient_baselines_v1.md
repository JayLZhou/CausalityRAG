# HotpotQA gradient attribution baselines

This report records the matched HotpotQA run for the two external Top-5
baselines. Attribution was computed with the local Hugging Face copy of
`Qwen2.5-7B-Instruct`; counterfactual answers were verified by the existing
OpenAI-compatible vLLM reader on `yujia-server3`. Both methods use the frozen
replacement registry and the same context-unit tokenization as ReFlow.

## Ranking definitions

- **Gradient×Input-Top5** scores each context model subtoken with the absolute
  elementwise product of its input embedding and the gradient of the mean
  teacher-forced clean-answer log-likelihood. Surface-token scores are the
  mean over their mapped model subtokens.
- **Integrated-Gradients-Top5** uses the same objective and mapping, with a
  zero-embedding baseline only on context positions. Prompt and
  teacher-forced answer embeddings remain fixed. The integral uses 16
  right-endpoint steps.

The historical ranking stage filtered to registry-valid positive-score units,
took a deterministic Top-5, and left rows with fewer than five valid units
explicit. These numbers are retained for provenance but must be rerun under
the current all-token selection and on-demand replacement protocol.
Verification used
`scripts/evaluate_reader.py --ignore-remaining-flow-threshold` so the baseline
comparison measures the selected token intervention rather than ReFlow's
flow gate.

## Results

| Method | Query scope | Flips | Queries | Flip rate | Candidate coverage | Mean selected |
|---|---:|---:|---:|---:|---:|---:|
| Gradient×Input-Top5 | all 1000 | 729 | 1000 | **72.90%** | 99.90% | 4.994 |
| Gradient×Input-Top5 | clean-exact | 247 | 401 | **61.60%** | 100.00% | 4.995 |
| Integrated-Gradients-Top5 | all 1000 | 742 | 1000 | **74.20%** | 99.90% | 4.994 |
| Integrated-Gradients-Top5 | clean-exact | 255 | 401 | **63.59%** | 100.00% | 4.995 |

All-query rows include the 599 queries whose clean reader answer was not exact;
clean-exact is the primary like-for-like subset with an exact clean answer.
The single no-candidate row is counted as an unsuccessful intervention in the
all-query rate.

## Frozen server artifacts

The complete attribution and reader outputs are under
`/data1/yujia/CausalityRAG/exp/hotpotqa/gradient_baselines_v1/` on
`yujia-server3`:

```text
gradient_x_input_top5.merged.jsonl
integrated_gradients_top5.merged.jsonl
evaluation_all_queries.jsonl
evaluation_all_queries.summary.json
evaluation_clean_exact.jsonl
evaluation_clean_exact.summary.json
evaluation_ig_all_queries.jsonl
evaluation_ig_all_queries.summary.json
evaluation_ig_clean_exact.jsonl
evaluation_ig_clean_exact.summary.json
```

The attribution output is deterministic for a fixed model/runtime and frozen
artifact set; the server shard files are retained as a restartable execution
record for the IG run.
