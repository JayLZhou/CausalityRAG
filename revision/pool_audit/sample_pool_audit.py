"""Stratified sample of shared-pool replacements for the plausibility audit.

For each dataset, draw positions uniformly from the frozen shared pool,
take the replacement every method actually uses (stable_shared_candidate,
seed 0), and render the original and edited sentence from the retrieved chunk.
"""

import argparse
import hashlib
import json
import random
import re

POOLS = {
    "hotpotqa": "/data1/yujia/CausalityRAG/out/hotpotqa/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl",
    "2wiki": "/data1/yujia/CausalityRAG/out/2wiki/replacements/shared_pool_top10_v1/shared_pool.jsonl",
    "musique": "/data1/yujia/CausalityRAG/out/musique/replacements/shared_pool_top10_v1/shared_pool.jsonl",
    "triviaqa": "/data1/yujia/CausalityRAG/out/triviaqa/replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl",
    "popqa": "/data1/yujia/CausalityRAG/out_popqa_20260810/popqa/replacements/shared_pool_top10_v1/shared_pool.jsonl",
    "timeqa": "/data1/yujia/CausalityRAG/out/timeqa/replacements/shared_pool_top10_v1/shared_pool.jsonl",
    "finqa": "/data1/yujia/CausalityRAG/out/finqa/replacements/shared_pool_top10_v1/shared_pool.jsonl",
    "quartz": "/data1/yujia/CausalityRAG/out/quartz/replacements/shared_pool_top10_v1/shared_pool.jsonl",
}
RETRIEVAL = {
    name: path.split("/replacements/")[0] + "/retrieval/top10_1000.jsonl"
    for name, path in POOLS.items()
}


def stable_shared_candidate(candidates, unit_id, seed):
    # Mirrors causalityrag.shared_replacement_pool.stable_shared_candidate.
    material = f"{seed}\0{unit_id}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return index % len(candidates)


def sentence_span(text, start, end):
    """Return (s, e) of the sentence containing text[start:end]."""
    bounds = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+", text)] + [len(text)]
    s = max(b for b in bounds if b <= start)
    e = min(b for b in bounds if b >= end)
    if e - s > 600:  # long table rows or run-on text: use a window instead
        s, e = max(0, start - 250), min(len(text), end + 250)
    return s, e


def sample_dataset(name, n, rng, seed):
    chunks = {}
    with open(RETRIEVAL[name], encoding="utf-8") as f:
        for line in f:
            for c in json.loads(line)["retrieved"]:
                chunks[c["chunk_id"]] = c["text"]
    # Reservoir sampling over position rows (pool files are up to ~1 GB).
    reservoir, seen = [], 0
    with open(POOLS[name], encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("row_kind") != "position_candidates" or not row.get("candidates"):
                continue
            if row["chunk_id"] not in chunks:
                continue
            seen += 1
            if len(reservoir) < 4 * n:
                reservoir.append(row)
            else:
                j = rng.randrange(seen)
                if j < 4 * n:
                    reservoir[j] = row
    rng.shuffle(reservoir)
    out = []
    for row in reservoir:
        text = chunks[row["chunk_id"]]
        a, b = row["chunk_char_start"], row["chunk_char_end"]
        if text[a:b] != row["surface"]:
            continue  # offset drift; skip rather than show a wrong edit
        idx = stable_shared_candidate(row["candidates"], row["unit_id"], seed)
        new = row["candidates"][idx]["new"]
        s, e = sentence_span(text, a, b)
        out.append({
            "dataset": name,
            "unit_id": row["unit_id"],
            "type": row.get("type"),
            "pos": row.get("pos"),
            "original": row["surface"],
            "replacement": new,
            "n_candidates": len(row["candidates"]),
            "candidate_index": idx,
            "sentence_original": text[s:a] + "[[" + text[a:b] + "]]" + text[b:e],
            "sentence_replaced": text[s:a] + "[[" + new + "]]" + text[b:e],
        })
        if len(out) == n:
            break
    return out, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=300)
    ap.add_argument("--sample-seed", type=int, default=20261024)
    ap.add_argument("--replacement-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    names = list(POOLS)
    base, extra = divmod(args.total, len(names))
    rows, stats = [], {}
    for i, name in enumerate(names):
        rng = random.Random(f"{args.sample_seed}:{name}")
        got, seen = sample_dataset(name, base + (i < extra), rng, args.replacement_seed)
        stats[name] = {"positions": seen, "sampled": len(got)}
        rows.extend(got)
        print(name, stats[name], flush=True)
    random.Random(args.sample_seed).shuffle(rows)  # blind annotators to dataset order
    with open(args.out, "w", encoding="utf-8") as f:
        for k, r in enumerate(rows, start=1):
            f.write(json.dumps({"audit_id": f"A{k:03d}", **r}, ensure_ascii=False) + "\n")
    with open(args.out + ".manifest.json", "w") as f:
        json.dump({"args": vars(args), "pools": POOLS, "stats": stats}, f, indent=1)


if __name__ == "__main__":
    main()
