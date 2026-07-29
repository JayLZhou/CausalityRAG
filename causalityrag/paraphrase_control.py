"""Meaning-preserving replacements for paired intervention controls."""

from __future__ import annotations

import json
import os
import urllib.request
from threading import Lock

from causalityrag.reader import parse_json_object
from causalityrag.semantic_filter import (
    is_lexical_paraphrase,
    lexical_paraphrase_candidates,
)


PARAPHRASE_POLICY = "llm_wordnet_paraphrase_control_v1"


class SentenceParaphraseClient:
    """Generate and independently audit contextual synonyms and aliases."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("CAUSALITYRAG_LLM_BASE_URL")
            or os.environ.get("YVETTE_LLM_BASE_URL")
            or "http://127.0.0.1:8000/v1"
        ).rstrip("/")
        self.model = (
            model
            or os.environ.get("CAUSALITYRAG_LLM_MODEL")
            or os.environ.get("YVETTE_LLM_MODEL")
            or "qwen2.5-7b"
        )
        self.timeout = timeout
        self._calls = 0
        self._lock = Lock()

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    def generate_many(
        self,
        targets: list[dict],
        *,
        max_candidates: int,
        attempt: int,
    ) -> dict[str, list[str]]:
        rows, aliases = _prompt_rows(targets)
        prompt = (
            "For every marked token, propose meaning-preserving replacements. "
            "The replacement must preserve the same entity, value, relation, and "
            "factual statement in this exact sentence while changing its surface "
            "wording. Use a synonym, conventional alias, abbreviation expansion, "
            "or equivalent textual rendering. It may contain up to five words. "
            "WordNet suggestions are hints, not mandatory outputs. For a year, "
            "an equivalent written-out form is valid (e.g., 1947->nineteen "
            "forty-seven). For a complete place name, a uniquely identifying "
            "description can be valid in context (e.g., London->the British "
            "capital). If the brackets cover one part of a multi-token name, the "
            "replacement must fit only that slot and must not repeat neighboring "
            "name tokens; a conventional initial is preferable when valid. "
            "Do not introduce a different fact, entity, date, number, polarity, "
            "or relation. Examples: physician->doctor, USA->United States, "
            "movie->film. Paris->London and 1947->1952 are invalid because they "
            "change information. Never include [[ or ]] in a candidate. "
            "Return an empty list only when no faithful local "
            "paraphrase exists. Return strict JSON mapping each short id to a list "
            f"of at most {max_candidates} strings.\n"
            f"Diversity attempt: {attempt}\nTargets:\n"
            + json.dumps(rows, ensure_ascii=False)
        )
        parsed = self._request(
            system=(
                "You generate contextual synonyms and aliases that preserve the "
                "original factual meaning. Output strict JSON only."
            ),
            prompt=prompt,
            temperature=min(0.8, 0.2 + 0.1 * attempt),
            max_tokens=max(128, min(4096, 20 * len(rows) * max_candidates)),
        )
        generated: dict[str, list[str]] = {}
        if not isinstance(parsed, dict):
            return generated
        for alias, values in parsed.items():
            unit_id = aliases.get(str(alias), "")
            if isinstance(values, str):
                values = [values]
            if not unit_id or not isinstance(values, list):
                continue
            original = next(
                str(target["token"])
                for target in targets
                if str(target["unit_id"]) == unit_id
            )
            seen = set()
            candidates = []
            for value in values[:max_candidates]:
                candidate = " ".join(str(value).strip().split())
                folded = candidate.casefold()
                if (
                    not candidate
                    or folded == original.casefold()
                    or folded in seen
                    or "[[" in candidate
                    or "]]" in candidate
                    or len(candidate.split()) > 7
                ):
                    continue
                seen.add(folded)
                candidates.append(candidate)
            generated[unit_id] = candidates
        return generated

    def judge_many(
        self,
        targets: list[dict],
        candidates: dict[str, list[str]],
    ) -> dict[tuple[str, str], bool]:
        items = []
        aliases: dict[str, tuple[str, str]] = {}
        for target in targets:
            unit_id = str(target["unit_id"])
            for candidate in candidates.get(unit_id, []):
                alias = str(len(items))
                aliases[alias] = (unit_id, candidate)
                items.append({
                    "id": alias,
                    "original": str(target["token"]),
                    "candidate": candidate,
                    "part_of_speech": str(target.get("pos", "")),
                    "semantic_type": str(target.get("unit_type", "")),
                    "marked_sentence": str(target["sentence"]),
                    "revised_sentence": str(target["sentence"]).replace(
                        "[[" + str(target["token"]) + "]]",
                        candidate,
                    ),
                })
        if not items:
            return {}
        prompt = (
            "Independently audit each local edit. Return true only if replacing the "
            "marked token with the candidate leaves the sentence's factual content "
            "and referents unchanged, remains grammatical, and genuinely changes "
            "surface wording. Synonyms, aliases, abbreviation expansions, and "
            "equivalent renderings are valid. Broader/narrower terms and changes to "
            "an entity, value, date, number, relation, or polarity are invalid. "
            "Reject repeated names, duplicated neighboring words, leftover marker "
            "brackets, and any candidate that does not fit the revised sentence. "
            "Return strict JSON mapping each id to true or false.\nItems:\n"
            + json.dumps(items, ensure_ascii=False)
        )
        parsed = self._request(
            system=(
                "You are a strict factual-equivalence auditor for contextual "
                "paraphrases. Output strict JSON only."
            ),
            prompt=prompt,
            temperature=0,
            max_tokens=max(128, min(4096, 12 * len(items))),
        )
        parsed = parsed if isinstance(parsed, dict) else {}
        return {
            pair: (
                parsed.get(alias) is True
                or (
                    isinstance(parsed.get(alias), dict)
                    and parsed[alias].get("valid") is True
                )
            )
            for alias, pair in aliases.items()
        }

    def _request(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> object:
        with self._lock:
            self._calls += 1
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            return parse_json_object(data["choices"][0]["message"]["content"])
        except Exception:
            return {}


def wordnet_support(target: dict, candidate: str) -> bool:
    """Return whether WordNet independently recognizes a lexical paraphrase."""

    if any(character.isspace() for character in candidate):
        return False
    return is_lexical_paraphrase(
        str(target["token"]),
        candidate,
        str(target.get("pos", "")),
        str(target.get("unit_type", "")),
    )


def _prompt_rows(targets: list[dict]) -> tuple[list[dict], dict[str, str]]:
    rows = []
    aliases = {}
    for index, target in enumerate(targets):
        alias = str(index)
        aliases[alias] = str(target["unit_id"])
        rows.append({
            "id": alias,
            "token": str(target["token"]),
            "part_of_speech": str(target.get("pos", "")),
            "semantic_type": str(target.get("unit_type", "")),
            "marked_sentence": str(target["sentence"]),
            "forbidden": list(target.get("forbidden", ())),
            "wordnet_suggestions": list(lexical_paraphrase_candidates(
                str(target["token"]), str(target.get("pos", "")), limit=2
            )),
        })
    return rows, aliases
