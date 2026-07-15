"""Token-level editable-unit proposal and support scoring."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from causalityrag.ilp import Unit, solve_budgeted_support, solve_min_cost_cover
from causalityrag.io import record_id, retrieved_contexts


TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:[.,:/-]\d+)*|[%$€£¥]+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[A-Za-z0-9]+")
NUMBER_RE = re.compile(r"^\d+(?:[.,:/-]\d+)*$")
MONEY_RE = re.compile(r"^[%$€£¥]+$")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "he", "her", "his", "in", "is",
    "it", "its", "of", "on", "or", "she", "that", "the", "their", "there", "they",
    "this", "to", "was", "were", "what", "when", "where", "which", "who", "whom",
    "whose", "why", "with",
}

RELATION_CUES = {
    "born", "died", "directed", "wrote", "written", "created", "founded", "located",
    "married", "played", "starring", "won", "served", "produced", "released", "capital",
    "member", "part", "known",
}


@dataclass(frozen=True)
class TokenUnit:
    unit_id: str
    text: str
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    chunk_id: str
    chunk_rank: int
    chunk_token_index: int
    chunk_char_start: int
    chunk_char_end: int
    sentence_id: int
    unit_type: str
    support: float
    cost: float = 1.0
    sources: list[str] = field(default_factory=list)

    def to_unit(self) -> Unit:
        return Unit(self.unit_id, self.support, self.cost, self.to_dict(include_support=False))

    def to_dict(self, include_support: bool = True) -> dict:
        out = {
            "unit_id": self.unit_id,
            "text": self.text,
            "type": self.unit_type,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "chunk_id": self.chunk_id,
            "chunk_rank": self.chunk_rank,
            "chunk_token_index": self.chunk_token_index,
            "chunk_char_start": self.chunk_char_start,
            "chunk_char_end": self.chunk_char_end,
            "sentence_id": self.sentence_id,
            "cost": round(self.cost, 6),
            "sources": list(self.sources),
        }
        if include_support:
            out["support"] = round(self.support, 6)
        return out


def build_token_units(
    record: dict,
    *,
    k: int = 5,
    support_mode: str = "proxy",
    include_stopwords: bool = False,
    question_boost: float = 2.0,
    window_boost: float = 0.5,
    cost_mode: str = "unit",
) -> list[TokenUnit]:
    contexts = retrieved_contexts(record)
    if k:
        contexts = contexts[:k]
    question = str(record.get("question", ""))
    q_terms = _terms(question)
    precomputed = record.get("token_supports") if isinstance(record.get("token_supports"), list) else None

    units = []
    global_text_offset = 0
    global_token_index = 0
    support_index = 0
    for ctx in contexts:
        text = ctx["text"]
        sentence_id = 0
        chunk_token_index = 0
        previous_content_support = 0.0
        for match in TOKEN_RE.finditer(text):
            token = match.group(0)
            if token in ".!?":
                sentence_id += 1
            unit_type = classify_token(token)
            if unit_type == "PUNCT":
                continue
            lowered = token.lower()
            is_stop = lowered in STOPWORDS
            if is_stop and not include_stopwords:
                chunk_token_index += 1
                continue

            support = None
            if support_mode == "precomputed":
                if precomputed is None or support_index >= len(precomputed):
                    raise ValueError("support_mode=precomputed requires record['token_supports'] for every token")
                support = float(precomputed[support_index])
            elif precomputed is not None and support_index < len(precomputed):
                support = float(precomputed[support_index])
            if support is None:
                support = proxy_support(
                    token,
                    unit_type,
                    chunk_rank=ctx["rank"],
                    q_terms=q_terms,
                    previous_content_support=previous_content_support,
                    question_boost=question_boost,
                    window_boost=window_boost,
                )

            if support > 0:
                sources = ["token"]
                if lowered in q_terms:
                    sources.append("question_overlap")
                if unit_type in {"NUMBER", "MONEY", "RELATION_CUE", "PROPER"}:
                    sources.append(unit_type.lower())
                cost = token_cost(token, cost_mode)
                units.append(TokenUnit(
                    unit_id=f"tok:{global_token_index}",
                    text=token,
                    token_start=global_token_index,
                    token_end=global_token_index + 1,
                    char_start=global_text_offset + match.start(),
                    char_end=global_text_offset + match.end(),
                    chunk_id=ctx["chunk_id"],
                    chunk_rank=ctx["rank"],
                    chunk_token_index=chunk_token_index,
                    chunk_char_start=match.start(),
                    chunk_char_end=match.end(),
                    sentence_id=sentence_id,
                    unit_type=unit_type,
                    support=support,
                    cost=cost,
                    sources=sources,
                ))
                if not is_stop:
                    previous_content_support = support
            chunk_token_index += 1
            global_token_index += 1
            support_index += 1
        global_text_offset += len(text) + 2
    return units


def proxy_support(
    token: str,
    unit_type: str,
    *,
    chunk_rank: int,
    q_terms: Counter,
    previous_content_support: float,
    question_boost: float,
    window_boost: float,
) -> float:
    lowered = token.lower()
    rank_weight = 1.0 / math.sqrt(max(1, chunk_rank))
    type_weight = {
        "NUMBER": 2.0,
        "MONEY": 2.0,
        "PROPER": 1.5,
        "RELATION_CUE": 1.3,
        "CONTENT": 1.0,
    }.get(unit_type, 0.5)
    support = rank_weight * type_weight
    if lowered in q_terms:
        support += question_boost * rank_weight * q_terms[lowered]
    if previous_content_support and lowered not in STOPWORDS:
        support += min(window_boost, previous_content_support * 0.1)
    return support


def classify_token(token: str) -> str:
    lowered = token.lower()
    if not WORD_RE.search(token):
        return "PUNCT"
    if MONEY_RE.match(token):
        return "MONEY"
    if NUMBER_RE.match(token):
        return "NUMBER"
    if lowered in RELATION_CUES:
        return "RELATION_CUE"
    if token[:1].isupper() and any(ch.isalpha() for ch in token):
        return "PROPER"
    if lowered in STOPWORDS:
        return "STOPWORD"
    return "CONTENT"


def token_cost(token: str, mode: str) -> float:
    if mode == "unit":
        return 1.0
    if mode == "chars":
        return max(1.0, len(token) / 4.0)
    raise ValueError(f"unknown cost mode: {mode}")


def run_token_ilp_record(
    record: dict,
    *,
    k: int = 5,
    objective: str = "min-cost",
    tau_graph: float = 0.2,
    target_support: float | None = None,
    budget: float = 5.0,
    solver: str = "auto",
    include_units: bool = False,
    top_candidates: int = 20,
    **unit_kwargs,
) -> dict:
    units = build_token_units(record, k=k, **unit_kwargs)
    ilp_units = [u.to_unit() for u in units]
    if objective == "min-cost":
        result = solve_min_cost_cover(ilp_units, target_support=target_support, tau_graph=tau_graph, solver=solver)
    elif objective == "budgeted":
        result = solve_budgeted_support(ilp_units, budget=budget, solver=solver)
    else:
        raise ValueError(f"unknown objective: {objective}")

    selected = set(result.selected_ids)
    selected_units = [u.to_dict() for u in units if u.unit_id in selected]
    top = sorted(units, key=lambda u: (-u.support / max(u.cost, 1e-9), -u.support, u.unit_id))
    contexts = retrieved_contexts(record)
    if k:
        contexts = contexts[:k]
    row = {
        "id": record_id(record),
        "question": str(record.get("question", "")),
        "clean_answer": str(record.get("clean_answer") or record.get("answer") or record.get("gold") or ""),
        "context_ids": [ctx["chunk_id"] for ctx in contexts],
        "objective": objective,
        "n_units": len(units),
        **result.to_dict(),
        "selected_units": selected_units,
        "top_candidates": [u.to_dict() for u in top[:top_candidates]],
    }
    if include_units:
        row["units"] = [u.to_dict() for u in units]
    return row


def _terms(text: str) -> Counter:
    return Counter(
        token.lower()
        for token in WORD_RE.findall(text or "")
        if token and token.lower() not in STOPWORDS
    )


def units_from_cache_row(record: dict, row: dict, *, k: int) -> list[dict]:
    identifier = record_id(record)
    if str(row.get("id", "")) != identifier:
        raise ValueError(f"token-units ID mismatch for {identifier}")
    if int(row.get("top_k", k)) != k:
        raise ValueError(f"token-units top-k mismatch for {identifier}")
    stored_hashes = row.get("context_sha256", {})
    if stored_hashes:
        current_hashes = {
            str(context["chunk_id"]): hashlib.sha256(
                str(context["text"]).encode("utf-8")
            ).hexdigest()
            for context in retrieved_contexts(record)[:k]
        }
        if stored_hashes != current_hashes:
            raise ValueError(f"token-units context hash mismatch for {identifier}")
    return list(row.get("units", []))
