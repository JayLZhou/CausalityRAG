"""Freeze spaCy surface-token, sentence, NER, POS, and offset annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.linguistics import SpacyAnnotationClient
from causalityrag.token_units import context_sentence_units


_LOCAL_NLP = None


def _initialize_local_spacy(model: str) -> None:
    import spacy

    global _LOCAL_NLP
    _LOCAL_NLP = spacy.load(model)


def _annotate_local(item: tuple[int, int, dict, int, str]) -> dict:
    offset, start, record, k, model = item
    if _LOCAL_NLP is None:
        raise RuntimeError("local spaCy worker was not initialized")
    return build_cache_row(
        offset=offset,
        start=start,
        record=record,
        k=k,
        nlp=_LOCAL_NLP,
        spacy_model=model,
    )


def build_cache_row(
    *,
    offset: int,
    start: int,
    record: dict,
    k: int,
    nlp,
    spacy_model: str,
) -> dict:
    started = time.monotonic()
    units, sentences = context_sentence_units(record, k=k, nlp=nlp)
    contexts = retrieved_contexts(record)[:k]
    return {
        "index": start + offset - 1,
        "id": record_id(record),
        "units": units,
        "sentences": sentences,
        "top_k": k,
        "context_sha256": {
            str(context["chunk_id"]): hashlib.sha256(
                str(context["text"]).encode("utf-8")
            ).hexdigest()
            for context in contexts
        },
        "spacy_model": spacy_model,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--backend",
        choices=("service", "local-process"),
        default="service",
    )
    parser.add_argument(
        "--spacy-model",
        default=os.environ.get("YVETTE_SPACY_MODEL", "en_core_web_lg"),
    )
    parser.add_argument(
        "--spacy-base-url",
        default=os.environ.get(
            "CAUSALITYRAG_SPACY_BASE_URL",
            "http://127.0.0.1:8021",
        ),
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    records = load_records(args.input)[args.start:args.start + args.n]
    if args.backend == "service":
        nlp = SpacyAnnotationClient(args.spacy_base_url)
        health = nlp.health()
        if not health.get("ok"):
            raise RuntimeError("spaCy annotation service is unhealthy")
        spacy_model = str(health.get("model") or args.spacy_model)

        def annotate(item: tuple[int, dict]) -> dict:
            offset, record = item
            return build_cache_row(
                offset=offset,
                start=args.start,
                record=record,
                k=args.k,
                nlp=nlp,
                spacy_model=spacy_model,
            )

        executor = ThreadPoolExecutor(max_workers=args.workers)
        generated = executor.map(annotate, enumerate(records, 1))
    else:
        spacy_model = args.spacy_model
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialize_local_spacy,
            initargs=(spacy_model,),
        )
        generated = executor.map(
            _annotate_local,
            (
                (offset, args.start, record, args.k, spacy_model)
                for offset, record in enumerate(records, 1)
            ),
            chunksize=1,
        )

    rows = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        with executor:
            for completed, row in enumerate(generated, 1):
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                if completed <= 10 or completed % 50 == 0:
                    print(
                        f"[token-units-cache] {completed}/{len(records)} "
                        f"units={len(row['units'])} seconds={row['elapsed_seconds']}",
                        flush=True,
                    )
    elapsed = [row["elapsed_seconds"] for row in rows]
    summary = {
        "queries": len(rows),
        "top_k": args.k,
        "spacy_model": spacy_model,
        "backend": args.backend,
        "units": sum(len(row["units"]) for row in rows),
        "mean_units": (
            statistics.fmean(len(row["units"]) for row in rows)
            if rows
            else None
        ),
        "total_worker_seconds": sum(elapsed),
        "mean_seconds": statistics.fmean(elapsed) if elapsed else None,
        "workers": args.workers,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print("[token-units-cache summary]", rendered)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")


if __name__ == "__main__":
    main()
