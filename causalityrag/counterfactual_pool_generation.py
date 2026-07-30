"""Generate multi-candidate counterfactual replacements for a frozen pool."""

from __future__ import annotations

import json
import os
import urllib.request
from threading import Lock

from causalityrag.reader import parse_json_object


POLICY = "llm_typed_counterfactual_v1"


def target_sentence(unit: dict, context: str) -> dict:
    """Return a sentence-local target span and an LLM-facing marked form."""

    start = int(unit["chunk_char_start"])
    end = int(unit["chunk_char_end"])
    sentence_start = max(
        context.rfind(".", 0, start),
        context.rfind("!", 0, start),
        context.rfind("?", 0, start),
        context.rfind("\n", 0, start),
    ) + 1
    sentence_end_candidates = [
        position
        for delimiter in (".", "!", "?", "\n")
        if (position := context.find(delimiter, end)) >= 0
    ]
    sentence_end = (
        min(sentence_end_candidates) + 1
        if sentence_end_candidates
        else len(context)
    )
    local_start = start - sentence_start
    local_end = end - sentence_start
    sentence = context[sentence_start:sentence_end]
    return {
        "sentence": sentence,
        "sentence_char_start": local_start,
        "sentence_char_end": local_end,
        "marked_sentence": (
            sentence[:local_start]
            + "[["
            + sentence[local_start:local_end]
            + "]]"
            + sentence[local_end:]
        ).strip(),
    }


def marked_target_sentence(unit: dict, context: str) -> str:
    """Mark exactly one surface-token occurrence in its containing sentence."""

    return str(target_sentence(unit, context)["marked_sentence"])


class SentenceCounterfactualClient:
    """Generate several one-token factual alternatives from marked sentences."""

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
        if not targets:
            return {}
        with self._lock:
            self._calls += 1
        rows = []
        alias_to_unit = {}
        for index, target in enumerate(targets):
            alias = str(index)
            alias_to_unit[alias] = str(target["unit_id"])
            rows.append({
                "id": alias,
                "token": str(target["token"]),
                "semantic_type_hint": (
                    str(target.get("unit_type", "")) or "unknown"
                ),
                "part_of_speech": str(target.get("pos", "")) or "unknown",
                "fine_grained_tag": str(target.get("tag", "")) or "unknown",
                "complete_entity": (
                    str(target.get("entity_text", "")) or "unknown"
                ),
                "entity_token_position": target.get("entity_token_position"),
                "marked_sentence": str(target["sentence"]),
                "forbidden": list(target.get("forbidden", ())),
            })
        prompt = (
            "For every target, generate factual counterfactual replacements for the "
            "token enclosed in [[double brackets]]. A replacement must be one token, "
            "retain the target's contextual semantic category and grammatical role, "
            "and make the sentence express an incompatible factual alternative. "
            "The revised statement must not merely restate, broaden, narrow, or "
            "generalize the original statement. It must not be a "
            "synonym, alias, paraphrase, spelling variant, inflection, generic "
            "placeholder, hypernym, hyponym, deletion, or random noise. Examples: "
            "Paris->London, "
            "1947->1952, won->lost, novelist->physicist. Invalid examples: "
            "film->movie, physician->doctor, firearms->weapons, "
            "theaters->playhouses, Paris->the French capital. Every output "
            f"array must contain exactly {max_candidates} different strings. Each "
            "individual string must be one token; 'one token' does not mean returning "
            "only one array element. Do not stop after the first candidate. For a "
            "token inside a multi-token named entity, preserve its name slot (for "
            "example, replace a surname with another surname). Type hints come from "
            "an automatic tagger and may be wrong; infer the actual type from the "
            "marked sentence and complete entity. Scientific examples include "
            "Quesnelia->Aechmea for a plant genus and dubia->fasciata for a species "
            "epithet. A geographic feature suffix may change Bay->Lake, and a "
            "measurement unit may change ft->m when grammar remains valid. "
            "Use an empty list only "
            "for function words or punctuation that carry no replaceable factual "
            "content. Return one compact STRICT JSON object mapping each short ID "
            "directly to its candidate list, for example "
            '{"0":["London","Berlin"],"1":["1952","1961"]}.\n\n'
            f"Diversity attempt: {attempt}\n"
            "Targets:\n"
            + json.dumps(rows, ensure_ascii=False)
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate meaningful factual counterfactual tokens. "
                        "Preserve type and grammar, but change information. Never "
                        "return synonyms. Output strict JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": min(1.0, 0.45 + 0.08 * attempt),
            "max_tokens": max(
                128,
                min(4096, 18 * len(targets) * max_candidates),
            ),
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
            parsed = parse_json_object(data["choices"][0]["message"]["content"])
        except Exception:
            parsed = {}
        generated: dict[str, list[str]] = {}
        if isinstance(parsed, dict):
            for alias, values in parsed.items():
                unit_id = alias_to_unit.get(str(alias), "")
                if isinstance(values, str):
                    values = [values]
                if not unit_id or not isinstance(values, list):
                    continue
                candidates = []
                seen = set()
                for value in values[:max_candidates]:
                    candidate = str(value).strip()
                    folded = candidate.casefold()
                    if (
                        not candidate
                        or folded in seen
                        or any(character.isspace() for character in candidate)
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
    ) -> dict[tuple[str, str], dict]:
        """Judge semantic validity without access to the query or answer."""

        items = []
        alias_to_pair = {}
        for target in targets:
            unit_id = str(target["unit_id"])
            for candidate in candidates.get(unit_id, []):
                alias = str(len(items))
                alias_to_pair[alias] = (unit_id, str(candidate))
                items.append({
                    "id": alias,
                    "unit_id": unit_id,
                    "original": str(target["token"]),
                    "candidate": str(candidate),
                    "semantic_type_hint": (
                        str(target.get("unit_type", "")) or "unknown"
                    ),
                    "part_of_speech": str(target.get("pos", "")) or "unknown",
                    "complete_entity": (
                        str(target.get("entity_text", "")) or "unknown"
                    ),
                    "entity_token_position": target.get(
                        "entity_token_position"
                    ),
                    "marked_sentence": str(target["sentence"]),
                })
        if not items:
            return {}
        with self._lock:
            self._calls += 1
        prompt = (
            "Audit each proposed counterfactual replacement using only its marked "
            "sentence. Mark valid=true only when all conditions hold: (1) the "
            "candidate is a real, meaningful token; (2) it has the same fine-grained "
            "contextual semantic type and grammatical role as the original; (3) it "
            "makes the revised statement express a genuinely incompatible factual "
            "alternative, rather than a restatement or change in specificity; "
            "(4) the revised "
            "sentence remains grammatical; and (5) it is not a synonym, alias, "
            "paraphrase, hypernym, hyponym, spelling variant, inflection, title, or "
            "generic placeholder. "
            "Be strict. For example, film->movie and physician->doctor are invalid "
            "synonyms; firearms->weapons and theaters->playhouses are invalid "
            "generalisations or near-paraphrases; nightmarish->terrifying, "
            "Wicked->Evil, and film->production "
            "are invalid paraphrases; Aqua->IORBx is invalid noise; Paris->London and 1947->1952 "
            "are valid when their contexts require a location and year. A conventional "
            "given name, middle name, or surname is valid for the same name slot; the "
            "modified full name need not denote an existing famous person. "
            "The automatic semantic type hint may be wrong; judge the actual role from "
            "the complete entity and marked sentence. Scientific taxon replacements "
            "such as Quesnelia->Aechmea and dubia->fasciata are valid. Geographic "
            "feature and measurement-unit replacements such as Bay->Lake and ft->m "
            "are valid when grammatically compatible. Return one "
            "compact STRICT JSON object mapping each short ID to true or false, "
            'for example {"0":true,"1":false}.\n\nItems:\n'
            + json.dumps(items, ensure_ascii=False)
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict semantic auditor for factual "
                        "counterfactual token replacements. Output JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max(128, min(4096, 12 * len(items))),
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
            parsed = parse_json_object(data["choices"][0]["message"]["content"])
        except Exception:
            parsed = {}
        judgments = {}
        parsed = parsed if isinstance(parsed, dict) else {}
        for alias, (unit_id, candidate) in alias_to_pair.items():
            raw = parsed.get(alias, False)
            valid = raw is True or (
                isinstance(raw, dict) and raw.get("valid") is True
            )
            judgments[(unit_id, candidate.casefold())] = {
                "valid": valid,
                "judge_value": raw,
            }
        return judgments

    def classify_relations_many(
        self,
        targets: list[dict],
        candidates: dict[str, list[str]],
    ) -> dict[tuple[str, str], dict]:
        """Classify each edit as counterfactual, paraphrase, or invalid."""

        items = []
        alias_to_pair = {}
        for target in targets:
            unit_id = str(target["unit_id"])
            for candidate in candidates.get(unit_id, []):
                alias = str(len(items))
                alias_to_pair[alias] = (unit_id, str(candidate))
                items.append({
                    "id": alias,
                    "original": str(target["token"]),
                    "candidate": str(candidate),
                    "semantic_type_hint": (
                        str(target.get("unit_type", "")) or "unknown"
                    ),
                    "part_of_speech": str(target.get("pos", "")) or "unknown",
                    "complete_entity": (
                        str(target.get("entity_text", "")) or "unknown"
                    ),
                    "entity_token_position": target.get(
                        "entity_token_position"
                    ),
                    "marked_sentence": str(target["sentence"]),
                })
        if not items:
            return {}
        with self._lock:
            self._calls += 1
        prompt = (
            "Classify each one-token edit using only the marked sentence. Return "
            "COUNTERFACTUAL only if replacing the bracketed token with the candidate "
            "keeps the same fine-grained semantic slot and grammar while asserting a "
            "different, incompatible factual value in that slot. Return PARAPHRASE "
            "for synonyms, aliases, inflections, hypernyms, hyponyms, broader or "
            "narrower descriptions, and edits that preserve or merely restate the "
            "original fact. Return INVALID for a different semantic role, malformed "
            "text, random noise, or an ungrammatical sentence. The automatic type "
            "hint may be wrong; infer the actual role from the complete marked "
            "sentence and complete entity. A conventional replacement for the same "
            "given-name, surname, geographic-name, organization-name, or title slot "
            "is COUNTERFACTUAL; the resulting full proper name need not denote an "
            "existing famous entity. Be conservative. "
            "Examples: Paris->London COUNTERFACTUAL; 1947->1952 COUNTERFACTUAL; "
            "won->lost COUNTERFACTUAL; film->movie PARAPHRASE; "
            "firearms->weapons PARAPHRASE; landing->harbor INVALID. Return one "
            "compact STRICT JSON object mapping each short ID to exactly one of "
            '"COUNTERFACTUAL", "PARAPHRASE", or "INVALID".\n\nItems:\n'
            + json.dumps(items, ensure_ascii=False)
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You perform conservative semantic-relation classification "
                        "for token edits. Output strict JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max(128, min(4096, 10 * len(items))),
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
            parsed = parse_json_object(data["choices"][0]["message"]["content"])
        except Exception:
            parsed = {}
        parsed = parsed if isinstance(parsed, dict) else {}
        classifications = {}
        for alias, (unit_id, candidate) in alias_to_pair.items():
            raw = parsed.get(alias, "INVALID")
            if isinstance(raw, dict):
                raw = raw.get("label", "INVALID")
            label = str(raw).strip().upper()
            if label not in {"COUNTERFACTUAL", "PARAPHRASE", "INVALID"}:
                label = "INVALID"
            classifications[(unit_id, candidate.casefold())] = {
                "label": label,
                "classifier_value": raw,
            }
        return classifications
