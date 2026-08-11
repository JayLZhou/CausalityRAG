"""Shared metric contract for factual and control reader evaluations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from causalityrag.reader import (
    answer_token_f1,
    answers_exact_match,
    answers_match,
    normalize_answer,
)


ALIAS_DATASETS = frozenset({"popqa", "triviaqa"})
CONJUNCTIVE_DATASETS = frozenset({"timeqa"})
INVALID_ANSWERS = frozenset({
    "",
    "[]",
    "{}",
    "null",
    "none",
    "no answer",
    "no answer found",
})


def _distinct(values: Iterable[object]) -> tuple[str, ...]:
    output = []
    seen = set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if text and key not in seen:
            output.append(text)
            seen.add(key)
    return tuple(output)


def valid_answer(value: object) -> bool:
    return str(value).strip().casefold() not in INVALID_ANSWERS


def answer_changed(clean_answer: object, edited_answer: object) -> bool:
    """Return the gold-free Ans-FR event used by every benchmark."""

    return not answers_exact_match(str(clean_answer), str(edited_answer))


@dataclass(frozen=True)
class GoldAnswerSpec:
    dataset: str
    mode: str
    canonical: str
    alternatives: tuple[str, ...]
    required_components: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        if self.mode == "conjunctive":
            return bool(self.canonical and self.required_components)
        return bool(self.alternatives)


def gold_answer_spec(record: dict, dataset: str = "") -> GoldAnswerSpec:
    """Build the benchmark-specific gold protocol from a frozen input row."""

    dataset_key = dataset.strip().lower()
    canonical = str(
        record.get("answer", record.get("gold_answer", ""))
    ).strip()
    raw_answers = record.get("answers", record.get("gold_answers", []))
    listed = (
        list(raw_answers)
        if isinstance(raw_answers, (list, tuple))
        else [raw_answers]
    )

    if dataset_key in CONJUNCTIVE_DATASETS:
        components = _distinct(listed)
        if not components and canonical:
            components = _distinct(part for part in canonical.split(";") if part)
        if not canonical and components:
            canonical = "; ".join(components)
        return GoldAnswerSpec(
            dataset=dataset_key,
            mode="conjunctive",
            canonical=canonical,
            alternatives=(canonical,) if canonical else (),
            required_components=components,
        )

    aliases = record.get("answer_aliases", [])
    listed_aliases = (
        list(aliases) if isinstance(aliases, (list, tuple)) else [aliases]
    )
    if dataset_key in ALIAS_DATASETS or record.get("gold_answers") is not None:
        alternatives = _distinct([canonical, *listed, *listed_aliases])
        mode = "alternative"
    else:
        alternatives = _distinct([canonical])
        mode = "single"
    return GoldAnswerSpec(
        dataset=dataset_key,
        mode=mode,
        canonical=canonical or (alternatives[0] if alternatives else ""),
        alternatives=alternatives,
    )


def _normalized_token_counter(value: str) -> Counter[str]:
    return Counter(
        token
        for token in normalize_answer(value).split()
        if token not in {"and"}
    )


def gold_f1(answer: object, spec: GoldAnswerSpec) -> float:
    text = str(answer)
    if not spec.is_valid:
        return 0.0
    if spec.mode == "conjunctive":
        return answer_token_f1(text, spec.canonical)
    return max(answer_token_f1(text, gold) for gold in spec.alternatives)


def gold_exact(answer: object, spec: GoldAnswerSpec) -> bool:
    text = str(answer)
    if not spec.is_valid:
        return False
    if spec.mode == "conjunctive":
        return bool(
            _normalized_token_counter(text)
            and _normalized_token_counter(text)
            == _normalized_token_counter(spec.canonical)
        )
    return any(answers_exact_match(text, gold) for gold in spec.alternatives)


def gold_accuracy(
    answer: object,
    spec: GoldAnswerSpec,
    *,
    reader_mode: str = "short_answer",
) -> bool:
    text = str(answer)
    if not spec.is_valid:
        return False
    if spec.mode == "conjunctive":
        return all(
            answers_match(text, component)
            for component in spec.required_components
        )
    matcher = answers_exact_match if reader_mode == "quartz" else answers_match
    return any(matcher(text, gold) for gold in spec.alternatives)


def clean_correct(
    metric: str,
    clean_answer: object,
    spec: GoldAnswerSpec,
    *,
    reader_mode: str = "short_answer",
) -> bool:
    if metric == "f1":
        return gold_f1(clean_answer, spec) >= 1.0 - 1e-12
    if metric == "em":
        return gold_exact(clean_answer, spec)
    if metric == "acc":
        return gold_accuracy(clean_answer, spec, reader_mode=reader_mode)
    raise ValueError(f"unsupported correctness metric: {metric}")


def correctness_lost(
    metric: str,
    clean_answer: object,
    edited_answer: object,
    spec: GoldAnswerSpec,
    *,
    reader_mode: str = "short_answer",
) -> bool:
    if not clean_correct(
        metric, clean_answer, spec, reader_mode=reader_mode
    ):
        return False
    if metric == "f1":
        return gold_f1(edited_answer, spec) < gold_f1(clean_answer, spec)
    if metric == "em":
        return not gold_exact(edited_answer, spec)
    return not gold_accuracy(edited_answer, spec, reader_mode=reader_mode)
