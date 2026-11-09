"""Derive the Qwen2.5-32B runner from the model-sweep runner (A4)."""
import sys

src, dst = sys.argv[1:]
s = open(src).read()


def rep(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


rep("# Run one ReFlow backbone on server 3.",
    "# A4: Qwen2.5-32B-Instruct, adapted from run_cross_model_reflow_server3.sh (model sweep).\n"
    "# One tensor-parallel reader on both GPUs; graphs built with device_map=balanced.\n"
    "# Run one ReFlow backbone on server 3.")
rep("OUTPUT_ROOT=${OUTPUT_ROOT:-/data1/yujia/CausalityRAG/out/model_sensitivity_v1}",
    "OUTPUT_ROOT=${OUTPUT_ROOT:-/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/a4_32b/out}")
rep("DATASETS=(hotpotqa timeqa finqa musique quartz triviaqa 2wiki pubmedqa)", "DATASETS=(hotpotqa finqa popqa)")

overrides = r"""# PopQA lives in the Table 3 directory; its top-5 unit file is derived from the
# top-10 cache (a4_32b/inputs). Other datasets keep the sweep's paths.
POPQA=/data1/yujia/CausalityRAG/out_table3_final8_20260811/popqa
eval "orig_$(declare -f input_path)"
eval "orig_$(declare -f units_path)"
eval "orig_$(declare -f factual_pool_path)"
eval "orig_$(declare -f paraphrase_pool_path)"
input_path() { if [[ $1 == popqa ]]; then echo "$POPQA/retrieval/top10_1000.jsonl"; else orig_input_path "$1"; fi; }
units_path() { if [[ $1 == popqa ]]; then echo /data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/a4_32b/inputs/popqa_token_units_top5_1000.jsonl; else orig_units_path "$1"; fi; }
factual_pool_path() { if [[ $1 == popqa ]]; then echo "$POPQA/replacements/shared_pool_top10_v1/shared_pool.jsonl"; else orig_factual_pool_path "$1"; fi; }
paraphrase_pool_path() { if [[ $1 == popqa ]]; then echo "$POPQA/controls/paraphrase_pool_top5_v2.jsonl"; else orig_paraphrase_pool_path "$1"; fi; }

reader_mode() {"""
rep("reader_mode() {", overrides)

start_old = s[s.index("start_readers() {"):s.index("generate_clean_targets() {")]
start_new = r"""start_readers() {
  stop_readers
  CUDA_VISIBLE_DEVICES=0,1 setsid "$PYTHON" \
    -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name "$SERVED_MODEL" \
    --dtype bfloat16 --trust-remote-code --max-model-len 16384 --tensor-parallel-size 2 \
    --enable-chunked-prefill --max-num-seqs 32 --enforce-eager \
    --disable-log-requests --gpu-memory-utilization 0.90 --port 8000 \
    >"$LOG_ROOT/vllm_8000.log" 2>&1 < /dev/null &
  echo $! >"$RUN_ROOT/reader_8002.pid"
  wait_endpoint http://127.0.0.1:8000/v1/models
}

"""
s = s.replace(start_old, start_new)
rep('if [[ "$MODEL_KEY" == "mistral-nemo-12b" ]]; then\n        graph_args+=(--device-map balanced)',
    'if [[ "$MODEL_KEY" == "mistral-nemo-12b" || "$MODEL_KEY" == "qwen2.5-32b" ]]; then\n        graph_args+=(--device-map balanced)')
rep('if [[ "$MODEL_KEY" == "mistral-nemo-12b" ]]; then\n  run_graph_and_frontier 0,1',
    'if [[ "$MODEL_KEY" == "mistral-nemo-12b" || "$MODEL_KEY" == "qwen2.5-32b" ]]; then\n  run_graph_and_frontier 0,1')
s = s.replace("--workers 48", "--workers 32")
open(dst, "w").write(s)
print("written", dst)
