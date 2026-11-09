"""Add QuaRTz (GPU0) and TriviaQA (GPU1) to the first-flip MiCE-RAG and A6 runs."""
R = "/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1"
Q = f"{R}/queue"
a6 = lambda d: (f"a6v3_{d}\tcd {R}/a6_eraser && CUDA_VISIBLE_DEVICES=$GPU "
                f"/data1/yujia/envs/graphrag/bin/python run_eraser.py {d} eraser_v3_{d}.jsonl\n")
mf = lambda d: f"miceflip_{d}\tbash {Q}/miceflip_job.sh {d}\n"
verify_quartz = (f"a5_verify_quartz_lenient\tO={R}/a5_cfg/quartz; "
                 f"[ -s $O/baselines_1000.jsonl ] && [ ! -s $O/baselines_1000.summary.json ] && mv $O/baselines_1000.jsonl $O/baselines_1000.partial.jsonl; "
                 f"LENIENT_QUARTZ=1 bash {Q}/run_a5_cfg.sh quartz $PORT\n")
p0 = f"{Q}/queue_gpu0.txt"
lines = open(p0).read().splitlines(True)
i = next((k for k, l in enumerate(lines) if l.startswith("a3_plain_retry")), len(lines))
lines[i:i] = [f"reader_up_0h\tbash {Q}/reader.sh up\n", verify_quartz, mf("quartz"),
              f"reader_down_0h\tbash {Q}/reader.sh down\n", a6("quartz")]
open(p0, "w").writelines(lines)
p1 = f"{Q}/queue_gpu1.txt"
lines = open(p1).read().splitlines(True)
i = next(k for k, l in enumerate(lines) if l.startswith("reader_down_1e"))
lines[i:i] = [mf("triviaqa")]
lines.append(a6("triviaqa"))
open(p1, "w").writelines(lines)
for g in (0, 1):
    print(g, [l.split("\t")[0] for l in open(f"{Q}/queue_gpu{g}.txt")])
