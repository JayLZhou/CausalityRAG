"""Rebuild both GPU queues for first-flip MiCE-RAG and the A6 rerun that uses it."""
R = "/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1"
Q = f"{R}/queue"
a6 = lambda d, wait="": (f"a6v3_{d}\t{wait}cd {R}/a6_eraser && CUDA_VISIBLE_DEVICES=$GPU "
                         f"/data1/yujia/envs/graphrag/bin/python run_eraser.py {d} eraser_v3_{d}.jsonl\n")
mf = lambda d: f"miceflip_{d}\tbash {Q}/miceflip_job.sh {d}\n"
gpu0 = ([f"reader_up_0g\tbash {Q}/reader.sh up\n"] + [mf(d) for d in ("hotpotqa", "musique", "popqa", "finqa", "2wiki")]
        + [f"reader_down_0g\tbash {Q}/reader.sh down\n"] + [a6(d) for d in ("hotpotqa", "musique", "popqa", "finqa", "2wiki")]
        + [f"a3_plain_retry\tbash {R}/a3_memory/a3_plain.sh\n"])
open(f"{Q}/queue_gpu0.txt", "w").writelines(gpu0)
p = f"{Q}/queue_gpu1.txt"
lines = open(p).read().splitlines(True)
i = next(k for k, l in enumerate(lines) if l.startswith("reader_down_1e"))
lines[i:i] = [mf("timeqa")]
lines.append(a6("timeqa"))
open(p, "w").writelines(lines)
for g in (0, 1):
    print(g, [l.split("\t")[0] for l in open(f"{Q}/queue_gpu{g}.txt")])
