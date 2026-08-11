"""Normalize the line-delimited PopQA bundle used by the server-3 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def iter_json_lines(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield row


def unique_nonempty(values) -> list[str]:
    rows = []
    seen = set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if text and key not in seen:
            rows.append(text)
            seen.add(key)
    return rows


def normalize_questions(rows: list[dict]) -> list[dict]:
    normalized = []
    for index, row in enumerate(rows):
        question = str(row.get("question", "")).strip()
        raw_answers = row.get("answers", [])
        if isinstance(raw_answers, str):
            raw_answers = raw_answers.split("|")
        answers = unique_nonempty(raw_answers)
        if not answers:
            answers = unique_nonempty(str(row.get("answer", "")).split("|"))
        if not question or not answers:
            raise ValueError(f"invalid PopQA question row {index}")
        normalized.append({
            "id": f"popqa-{index:06d}",
            "question": question,
            "answer": answers[0],
            "answers": answers,
            "answer_aliases": answers[1:],
            "answer_type": "entity",
        })
    return normalized


def normalize_corpus(rows: list[dict]) -> list[dict]:
    normalized = []
    identifiers = set()
    for index, row in enumerate(rows):
        identifier = str(row.get("id", index)).strip()
        title = str(row.get("title", "")).strip()
        text = str(row.get("context", row.get("text", ""))).strip()
        if not identifier or identifier in identifiers:
            raise ValueError(f"duplicate or empty PopQA corpus id at row {index}")
        if not text:
            continue
        identifiers.add(identifier)
        normalized.append({
            "id": f"popqa-doc-{identifier}",
            "title": title,
            "text": text,
        })
    return normalized


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    args = parser.parse_args()

    source_questions = list(iter_json_lines(args.questions))
    source_corpus = list(iter_json_lines(args.corpus))
    questions = normalize_questions(source_questions)
    corpus = normalize_corpus(source_corpus)
    question_out = args.out_dir / "questions.jsonl"
    corpus_out = args.out_dir / "corpus.jsonl"
    write_jsonl(question_out, questions)
    write_jsonl(corpus_out, corpus)

    manifest = {
        "schema": "causalityrag.popqa_bundle.v1",
        "questions": len(questions),
        "source_question_rows": len(source_questions),
        "corpus_passages": len(corpus),
        "source_corpus_rows": len(source_corpus),
        "skipped_empty_corpus_passages": len(source_corpus) - len(corpus),
        "source_questions_sha256": sha256(args.questions),
        "source_corpus_sha256": sha256(args.corpus),
        "questions_sha256": sha256(question_out),
        "corpus_sha256": sha256(corpus_out),
        "canonical_answer": "first nonempty answer alias",
        "aliases_preserved": True,
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
