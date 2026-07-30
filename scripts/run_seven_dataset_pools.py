"""Build frozen top-10 replacement pools for the seven transfer datasets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DATASETS = [
    (
        "2wiki",
        "/data1/yujia/RAGData/2wiki/questions/2wikimultihopqa.json",
        "/data1/yujia/RAGData/2wiki/corpus/2wikimultihopqa_corpus.json",
    ),
    (
        "pubmedqa",
        "/data1/yujia/RAGData/pubmedqa/questions/pubmedqa.json",
        "/data1/yujia/RAGData/pubmedqa/corpus/pubmedqa_corpus.json",
    ),
    (
        "timeqa",
        "/data1/yujia/RAGData/timeqa/questions/timeqa.json",
        "/data1/yujia/RAGData/timeqa/corpus/timeqa_corpus.json",
    ),
    (
        "musique",
        "/data1/yujia/RAGData/musique/questions/musique.json",
        "/data1/yujia/RAGData/musique/corpus/musique_corpus.json",
    ),
    (
        "finqa",
        "/data1/yujia/RAGData/finqa/questions/finqa.json",
        "/data1/yujia/RAGData/finqa/corpus/finqa_corpus.json",
    ),
    (
        "qasper",
        "/data1/yujia/RAGData/qasper/questions/qasper.json",
        "/data1/yujia/RAGData/qasper/corpus/qasper_corpus.json",
    ),
    (
        "cuad",
        "/data1/yujia/RAGData/cuad/questions/cuad.json",
        "/data1/yujia/RAGData/cuad/corpus/cuad_corpus.json",
    ),
]
LAYOUT = (
    "retrieval",
    "inputs",
    "replacements",
    "graphs",
    "methods/reflow",
    "methods/baselines",
    "controls",
    "audits/final_top10pool_k5",
    "logs",
)


def write_manifest(root: Path, dataset: str, status: str) -> None:
    for relative in LAYOUT:
        (root / relative).mkdir(parents=True, exist_ok=True)
    release_file = Path(__file__).resolve().parents[1] / "RELEASE_COMMIT"
    release = (
        release_file.read_text(encoding="utf-8").strip()
        if release_file.exists()
        else os.environ.get("CAUSALITYRAG_RELEASE", "development")
    )
    manifest = {
        "dataset": dataset,
        "status": status,
        "frozen_queries": 1000,
        "retrieval_top_k": 10,
        "main_experiment_k": 5,
        "derived_k_prefixes": [1, 3, 5, 10],
        "chunk_size_tokens": 384,
        "chunk_overlap_tokens": 64,
        "replacement_pool": "replacements/shared_pool_top10_v1",
        "replacement_policy": "llm_typed_counterfactual_v1",
        "release": release,
    }
    temporary = root / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, root / "manifest.json")


def run(command: list[str], *, cwd: Path) -> None:
    print("[command] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def generate_until_complete(
    *,
    python: str,
    repository: Path,
    typed_keys: Path,
    seed: Path,
    pool_dir: Path,
    max_passes: int,
) -> None:
    previous_covered = -1
    stagnant = 0
    for generation_pass in range(1, max_passes + 1):
        manifest = pool_dir / "generation.json"
        if generation_pass == 1:
            batch_size, workers, rounds, max_candidates = 24, 24, 1, 5
        elif generation_pass == 2:
            batch_size, workers, rounds, max_candidates = 12, 24, 1, 5
        elif generation_pass <= 4:
            batch_size, workers, rounds, max_candidates = 8, 16, 2, 5
        else:
            batch_size, workers, rounds, max_candidates = 1, 24, 4, 20
        run(
            [
                python,
                "scripts/generate_shared_typed_replacement_pool.py",
                "--typed-keys",
                str(typed_keys),
                "--seed",
                str(seed),
                "--out",
                str(pool_dir / "typed_candidates.jsonl"),
                "--unresolved-out",
                str(pool_dir / "unresolved.jsonl"),
                "--manifest-out",
                str(manifest),
                "--workers",
                str(workers),
                "--batch-size",
                str(batch_size),
                "--max-candidates",
                str(max_candidates),
                "--generation-rounds",
                str(rounds),
                "--attempt-offset",
                str((generation_pass - 1) * 4),
                "--llm-base-url",
                "http://127.0.0.1:8000/v1",
                "--llm-model",
                "qwen2.5-7b",
                "--spacy-base-url",
                "http://127.0.0.1:8021",
            ],
            cwd=repository,
        )
        report = json.loads(manifest.read_text(encoding="utf-8"))
        covered = int(report["typed_keys_covered"])
        unresolved = int(report["unresolved"])
        print(
            f"[pool-generation] pass={generation_pass} "
            f"covered={covered}/{report['typed_keys_total']} "
            f"unresolved={unresolved}",
            flush=True,
        )
        if unresolved == 0:
            return
        stagnant = stagnant + 1 if covered == previous_covered else 0
        previous_covered = covered
        if stagnant >= 8:
            raise RuntimeError(
                "replacement generation made no progress for eight passes; "
                "the partial pool was not frozen"
            )
    raise RuntimeError(
        f"replacement generation remains unresolved after {max_passes} passes"
    )


def build_inventory(
    *,
    python: str,
    repository: Path,
    retrieval: Path,
    units: Path,
    pool_dir: Path,
    n: int,
) -> None:
    pool_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            python,
            "scripts/build_shared_replacement_inventory.py",
            "--input",
            str(retrieval),
            "--units-cache",
            str(units),
            "--positions-out",
            str(pool_dir / "positions.jsonl"),
            "--typed-keys-out",
            str(pool_dir / "typed_keys.jsonl"),
            "--manifest-out",
            str(pool_dir / "inventory.json"),
            "--n",
            str(n),
            "--k",
            "10",
        ],
        cwd=repository,
    )


def freeze_pool(
    *,
    python: str,
    repository: Path,
    pool_dir: Path,
) -> None:
    run(
        [
            python,
            "scripts/freeze_shared_replacement_pool.py",
            "--positions",
            str(pool_dir / "positions.jsonl"),
            "--typed-candidates",
            str(pool_dir / "typed_candidates.jsonl"),
            "--out",
            str(pool_dir / "shared_pool.jsonl"),
            "--manifest-out",
            str(pool_dir / "shared_pool.manifest.json"),
        ],
        cwd=repository,
    )
    report = json.loads(
        (pool_dir / "shared_pool.manifest.json").read_text(encoding="utf-8")
    )
    if (
        int(report["unresolved_typed_positions"]) != 0
        or int(report["eligible_positions"]) != int(report["positions"])
    ):
        raise RuntimeError(f"pool failed closed audit: {report}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="/data1/yujia/CausalityRAG/out")
    parser.add_argument(
        "--tokenizer-path",
        default="/data1/yujia/models/Qwen3-Embedding-0.6B",
    )
    parser.add_argument(
        "--python",
        default="/data1/yujia/envs/graphrag/bin/python",
    )
    parser.add_argument("--max-generation-passes", type=int, default=40)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_root)
    for dataset, questions, corpus in DATASETS:
        root = out_root / dataset
        write_manifest(root, dataset, "pool_building")
        retrieval = root / "retrieval" / "top10_1000.jsonl"
        units = root / "inputs" / "token_units_top10_1000.jsonl"
        print(f"\n[dataset:start] {dataset}", flush=True)
        run(
            [
                args.python,
                "scripts/prepare_dataset_retrieval.py",
                "--dataset",
                dataset,
                "--questions",
                questions,
                "--corpus",
                corpus,
                "--out-root",
                str(root),
                "--tokenizer-path",
                args.tokenizer_path,
            ],
            cwd=repository,
        )
        units.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "/data1/yujia/envs/spacyner/bin/python",
                "scripts/build_context_units.py",
                "--input",
                str(retrieval),
                "--out",
                str(units),
                "--summary-out",
                str(root / "inputs" / "token_units_top10_1000.summary.json"),
                "--n",
                "1000",
                "--k",
                "10",
                "--workers",
                "24",
                "--backend",
                "service",
                "--spacy-base-url",
                "http://127.0.0.1:8021",
            ],
            cwd=repository,
        )

        smoke = root / "replacements" / "smoke10"
        build_inventory(
            python="/data1/yujia/envs/spacyner/bin/python",
            repository=repository,
            retrieval=retrieval,
            units=units,
            pool_dir=smoke,
            n=10,
        )
        generate_until_complete(
            python="/data1/yujia/envs/spacyner/bin/python",
            repository=repository,
            typed_keys=smoke / "typed_keys.jsonl",
            seed=smoke / "typed_candidates.jsonl",
            pool_dir=smoke,
            max_passes=args.max_generation_passes,
        )
        freeze_pool(
            python="/data1/yujia/envs/spacyner/bin/python",
            repository=repository,
            pool_dir=smoke,
        )
        print(f"[smoke:passed] {dataset}", flush=True)

        formal = root / "replacements" / "shared_pool_top10_v1"
        build_inventory(
            python="/data1/yujia/envs/spacyner/bin/python",
            repository=repository,
            retrieval=retrieval,
            units=units,
            pool_dir=formal,
            n=1000,
        )
        generate_until_complete(
            python="/data1/yujia/envs/spacyner/bin/python",
            repository=repository,
            typed_keys=formal / "typed_keys.jsonl",
            seed=smoke / "typed_candidates.jsonl",
            pool_dir=formal,
            max_passes=args.max_generation_passes,
        )
        freeze_pool(
            python="/data1/yujia/envs/spacyner/bin/python",
            repository=repository,
            pool_dir=formal,
        )
        write_manifest(root, dataset, "pool_complete")
        print(f"[dataset:pool-complete] {dataset}", flush=True)

    print("[pipeline:POOL_COMPLETE] seven datasets", flush=True)


if __name__ == "__main__":
    main()
