"""Freeze canonical queries, corpus chunks, and dense top-k retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from urllib.error import HTTPError
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from causalityrag.shared_replacement_pool import file_sha256


def gold_titles(record: dict) -> list[str]:
    values: list[str] = []
    direct = record.get("gold_titles") or record.get("support_title") or []
    if isinstance(direct, str):
        direct = [direct]
    if isinstance(direct, list):
        values.extend(str(value) for value in direct if str(value).strip())
    for fact in record.get("supporting_facts", []):
        if isinstance(fact, (list, tuple)) and fact:
            values.append(str(fact[0]))
    for paragraph in record.get("paragraphs", []):
        if isinstance(paragraph, dict) and paragraph.get("is_supporting"):
            values.append(str(paragraph.get("title", "")))
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def frozen_queries(records: list[dict], n: int) -> list[dict]:
    rows = []
    seen = set()
    for record in records:
        identifier = record_id(record)
        question = str(record.get("question", "")).strip()
        answer = str(record.get("answer", "")).strip()
        if not identifier or not question or not answer or identifier in seen:
            continue
        seen.add(identifier)
        row = {
            "id": identifier,
            "question": question,
            "answer": answer,
            "gold_titles": gold_titles(record),
        }
        for key in ("answers", "answer_aliases", "answer_type", "long_answer"):
            if key in record:
                row[key] = record[key]
        rows.append(row)
        if len(rows) == n:
            break
    if len(rows) != n:
        raise ValueError(f"expected {n} valid queries, found {len(rows)}")
    return rows


def chunk_documents(
    records: list[dict],
    tokenizer,
    *,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("require chunk_size > overlap >= 0")
    chunks = []
    step = chunk_size - overlap
    for document_index, record in enumerate(records):
        title = str(record.get("title", "")).strip()
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        spans = (
            [(0, len(token_ids), text)]
            if len(token_ids) <= chunk_size
            else [
                (
                    start,
                    min(len(token_ids), start + chunk_size),
                    tokenizer.decode(
                        token_ids[start : start + chunk_size],
                        skip_special_tokens=True,
                    ).strip(),
                )
                for start in range(0, len(token_ids), step)
                if token_ids[start : start + chunk_size]
            ]
        )
        for chunk_index, (start, end, chunk_text) in enumerate(spans):
            if not chunk_text:
                continue
            material = (
                f"{document_index}\0{title}\0{start}\0{end}\0{chunk_text}"
            ).encode("utf-8")
            chunks.append({
                "chunk_id": "c" + hashlib.sha256(material).hexdigest()[:12],
                "document_index": document_index,
                "chunk_index": chunk_index,
                "token_start": start,
                "token_end": end,
                "title": title,
                "text": chunk_text,
            })
    if not chunks:
        raise ValueError("corpus produced no nonempty chunks")
    return chunks


def embed(
    texts: list[str],
    *,
    base_url: str,
    model: str,
    batch_size: int,
    label: str,
) -> np.ndarray:
    rows = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        rows.extend(
            request_embedding_batch(
                batch,
                base_url=base_url,
                model=model,
            )
        )
        completed = min(len(texts), start + len(batch))
        if completed == len(texts) or completed % (batch_size * 10) == 0:
            print(f"[embedding:{label}] {completed}/{len(texts)}", flush=True)
    matrix = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embedding service returned a zero vector")
    return matrix / norms


def request_embedding_batch(
    texts: list[str],
    *,
    base_url: str,
    model: str,
) -> list[list[float]]:
    """Embed a batch, splitting it deterministically when the server rejects it."""

    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        if error.code not in {413, 500, 502, 503, 504} or len(texts) == 1:
            raise
        middle = len(texts) // 2
        print(
            f"[embedding:split] status={error.code} "
            f"batch={len(texts)} -> {middle}+{len(texts) - middle}",
            flush=True,
        )
        return request_embedding_batch(
            texts[:middle],
            base_url=base_url,
            model=model,
        ) + request_embedding_batch(
            texts[middle:],
            base_url=base_url,
            model=model,
        )
    data = sorted(result.get("data", []), key=lambda item: int(item["index"]))
    if len(data) != len(texts):
        raise RuntimeError(
            f"embedding service returned {len(data)}/{len(texts)} rows"
        )
    return [item["embedding"] for item in data]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8017/v1")
    parser.add_argument("--embedding-model", default="Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="freeze queries and corpus chunks without running embeddings",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    root = Path(args.out_root)
    retrieval = root / "retrieval"
    index = retrieval / "index"
    index.mkdir(parents=True, exist_ok=True)
    query_path = retrieval / "queries_1000.jsonl"
    chunk_path = retrieval / "corpus_chunks_384t_o64.jsonl"
    embedding_path = index / "chunk_embeddings.npy"
    top_path = retrieval / "top10_1000.jsonl"
    summary_path = retrieval / "top10_1000.summary.json"

    queries = frozen_queries(load_records(args.questions), args.n)
    write_jsonl(query_path, queries)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
    )
    if chunk_path.exists():
        chunks = load_records(chunk_path)
        print(f"[chunks] reuse {len(chunks)} from {chunk_path}", flush=True)
    else:
        chunks = chunk_documents(
            load_records(args.corpus),
            tokenizer,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        write_jsonl(chunk_path, chunks)
        print(f"[chunks] wrote {len(chunks)} to {chunk_path}", flush=True)

    if args.chunks_only:
        print(json.dumps({
            "dataset": args.dataset,
            "queries": len(queries),
            "corpus_documents": len(load_records(args.corpus)),
            "corpus_chunks": len(chunks),
            "chunk_size_tokens": args.chunk_size,
            "chunk_overlap_tokens": args.overlap,
            "chunks_only": True,
            "queries_sha256": file_sha256(query_path),
            "chunks_sha256": file_sha256(chunk_path),
        }, indent=2, ensure_ascii=False), flush=True)
        return

    if embedding_path.exists():
        chunk_embeddings = np.load(embedding_path)
        if len(chunk_embeddings) != len(chunks):
            raise ValueError("cached chunk embeddings do not match chunk rows")
        print(f"[embedding:corpus] reuse {embedding_path}", flush=True)
    else:
        chunk_embeddings = embed(
            [
                f"{row['title']}\n{row['text']}" if row["title"] else row["text"]
                for row in chunks
            ],
            base_url=args.embedding_base_url,
            model=args.embedding_model,
            batch_size=args.embedding_batch_size,
            label="corpus",
        )
        np.save(embedding_path, chunk_embeddings)

    query_embeddings = embed(
        [row["question"] for row in queries],
        base_url=args.embedding_base_url,
        model=args.embedding_model,
        batch_size=args.embedding_batch_size,
        label="queries",
    )
    output_rows = []
    recall = 0
    for start in range(0, len(queries), 128):
        scores = query_embeddings[start : start + 128] @ chunk_embeddings.T
        for local_index, vector in enumerate(scores):
            query = queries[start + local_index]
            candidate = np.argpartition(vector, -args.top_k)[-args.top_k :]
            ranked = candidate[np.argsort(vector[candidate])[::-1]]
            retrieved = [
                {
                    "chunk_id": chunks[int(index_value)]["chunk_id"],
                    "title": chunks[int(index_value)]["title"],
                    "text": chunks[int(index_value)]["text"],
                    "rank": rank,
                    "score": float(vector[int(index_value)]),
                }
                for rank, index_value in enumerate(ranked, 1)
            ]
            normalized_gold = {
                title.casefold() for title in query["gold_titles"] if title
            }
            hit = bool(
                normalized_gold
                and any(
                    row["title"].casefold() in normalized_gold
                    for row in retrieved
                )
            )
            recall += int(hit)
            output_rows.append({**query, "retrieved": retrieved, "gold_recall_any": hit})
    write_jsonl(top_path, output_rows)
    summary = {
        "dataset": args.dataset,
        "queries": len(queries),
        "corpus_documents": len(load_records(args.corpus)),
        "corpus_chunks": len(chunks),
        "chunk_size_tokens": args.chunk_size,
        "chunk_overlap_tokens": args.overlap,
        "retrieval_top_k": args.top_k,
        "embedding_model": args.embedding_model,
        "gold_title_queries": sum(bool(row["gold_titles"]) for row in queries),
        "gold_title_recall_any": recall,
        "queries_sha256": file_sha256(query_path),
        "chunks_sha256": file_sha256(chunk_path),
        "top10_sha256": file_sha256(top_path),
    }
    with summary_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
