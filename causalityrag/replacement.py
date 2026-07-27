"""Generic contextual token replacement through the local OpenAI-compatible LLM."""

from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from causalityrag.reader import parse_json_object


_CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
_NON_SEMANTIC_TYPES = {"", "PUNCT", "STOPWORD"}
_UNTRUSTED_GENERIC_POLICIES = {
    "generic_llm_contextual",
    "generic_llm_contextual_batched",
    "generic_deterministic_fallback",
}
def is_trusted_cached_replacement(replacement: dict) -> bool:
    """Return whether a stored replacement satisfies the strict CF contract."""

    return bool(
        replacement.get("ok")
        and str(replacement.get("policy", "")) not in _UNTRUSTED_GENERIC_POLICIES
    )


class GenericReplacementClient:
    """Ask the local editor model for one non-deleting word replacement."""

    def __init__(
        self, base_url: str | None = None, model: str | None = None, timeout: int = 60
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
        self._calls_lock = Lock()

    @property
    def calls(self) -> int:
        with self._calls_lock:
            return self._calls

    def replace(
        self,
        token: str,
        context: str,
        unit_type: str = "",
        *,
        pos_hint: str = "",
        tag_hint: str = "",
        forbidden: tuple[str, ...] = (),
    ) -> dict:
        with self._calls_lock:
            self._calls += 1
        prompt = (
            "Generate one factual counterfactual token for the target word in the "
            "passage. First identify the target's fine-grained semantic type in this "
            "passage, such as person, city, year, profession, medium, genre, political "
            "status, event, or relation. Then choose another token of exactly that type "
            "which makes the passage express different information. Never use a synonym, "
            "near-synonym, paraphrase, hypernym, spelling variant, or inflection. Preserve "
            "the target's part of speech, tense, number, and capitalization. If no "
            "fine-grained type and valid alternative can be identified, use an empty "
            "replacement. Return STRICT JSON only: "
            '{"semantic_type":"...","replacement":"..."}.\n\n'
            f"Target word: {token}\n"
            f"Token type hint: {unit_type or 'unknown'}\n"
            f"Required POS: {pos_hint or 'unknown'}\n"
            f"Required fine-grained tag: {tag_hint or 'unknown'}\n"
            f"Forbidden replacements: {', '.join(forbidden) or 'none'}\n"
            f"Passage: {context}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate factual counterfactual tokens, never synonyms. "
                        "A valid replacement keeps the token's semantic type and grammar "
                        "but makes the passage express different information. Before "
                        "answering, test whether the replacement preserves the same fact "
                        "or concept; if it does, reject it and choose a genuinely different "
                        "alternative. Return an empty replacement if none exists."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 48,
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
            content = data["choices"][0]["message"]["content"]
            parsed = parse_json_object(content)
            replacement = (
                str(parsed.get("replacement", "")).strip()
                if isinstance(parsed, dict)
                else ""
            )
            semantic_type = (
                str(parsed.get("semantic_type", "")).strip()
                if isinstance(parsed, dict)
                else ""
            )
        except Exception as exc:
            return deterministic_fallback(token, unit_type, error=str(exc))
        if (
            not replacement
            or replacement.lower() == token.lower()
            or replacement.lower() in {item.lower() for item in forbidden}
            or any(char.isspace() for char in replacement)
        ):
            return deterministic_fallback(
                token, unit_type, error="invalid_llm_response"
            )
        return {
            "ok": True,
            "old": token,
            "new": replacement,
            "inferred_type": semantic_type,
            "policy": "generic_llm_contextual",
        }

    def replace_many(self, targets: list[dict]) -> dict[str, dict]:
        """Generate contextual replacements for several tokens in one request."""

        if not targets:
            return {}
        with self._calls_lock:
            self._calls += 1

        passages = {}
        target_rows = []
        for target in targets:
            passage_id = str(target["chunk_id"])
            passages[passage_id] = str(target["context"])
            target_rows.append(
                {
                    "id": str(target["unit_id"]),
                    "token": str(target["token"]),
                    "passage_id": passage_id,
                    "type": str(target.get("unit_type", "")) or "unknown",
                    "pos": str(target.get("pos_hint", "")) or "unknown",
                    "tag": str(target.get("tag_hint", "")) or "unknown",
                    "forbidden": list(target.get("forbidden", ())),
                }
            )
        prompt = (
            "Generate one counterfactual token for every listed target word. A "
            "counterfactual token has the same semantic type and grammatical role as "
            "the original token, but represents different information. Replace a person "
            "with a different person, a location with a different location, a date or "
            "number with a different value, a relation with a conflicting relation, or "
            "a content concept with a different concept of the same broad type. "
            "Return STRICT JSON only in this form: "
            '{"replacements":[{"id":"TARGET_ID","replacement":"WORD"}]}. '
            "Return exactly one item for every target ID. Each replacement must be "
            "one non-empty word, differ from the original and forbidden words, and "
            "preserve the target's broad semantic category, part of speech, tense, "
            "number, and capitalization while changing the information expressed by "
            "the passage. It must not be a synonym, near-synonym, paraphrase, spelling "
            "variant, or inflection. Return an empty replacement when no meaningful "
            "counterfactual token exists. Valid examples: Paris->London, "
            "novelist->physicist, civil->military, won->lost. Invalid examples: "
            "novelist->writer, physician->doctor, film->movie, family->household. "
            "Use each target's passage as context.\n\n"
            "Passages:\n"
            + json.dumps(passages, ensure_ascii=False)
            + "\n\nTargets:\n"
            + json.dumps(target_rows, ensure_ascii=False)
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate factual counterfactual tokens, never synonyms. "
                        "A valid replacement keeps the token's semantic type and grammar "
                        "but makes the passage express different information. Before "
                        "answering, test whether each replacement preserves the same fact "
                        "or concept; if it does, reject it and choose a genuinely different "
                        "alternative. Return an empty replacement if none exists."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max(128, min(2048, 32 * len(targets))),
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        parsed_rows = []
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            parsed = parse_json_object(data["choices"][0]["message"]["content"])
            if isinstance(parsed, dict):
                raw_rows = parsed.get("replacements", [])
                if isinstance(raw_rows, list):
                    parsed_rows = raw_rows
        except Exception:
            parsed_rows = []

        generated = {}
        for row in parsed_rows:
            if not isinstance(row, dict):
                continue
            unit_id = str(row.get("id", ""))
            replacement = str(row.get("replacement", "")).strip()
            if unit_id:
                generated[unit_id] = replacement

        results = {}
        for target in targets:
            unit_id = str(target["unit_id"])
            token = str(target["token"])
            forbidden = {
                str(item).lower() for item in target.get("forbidden", ())
            }
            replacement = generated.get(unit_id, "")
            if (
                not replacement
                or replacement.lower() == token.lower()
                or replacement.lower() in forbidden
                or any(char.isspace() for char in replacement)
            ):
                results[unit_id] = deterministic_fallback(
                    token,
                    str(target.get("unit_type", "")),
                    error="invalid_batched_llm_response",
                )
            else:
                results[unit_id] = {
                    "ok": True,
                    "old": token,
                    "new": replacement,
                    "policy": "generic_llm_contextual_batched",
                }
        return results

    def replace_many_candidates(
        self,
        targets: list[dict],
        *,
        candidates_per_target: int = 3,
    ) -> dict[str, list[dict]]:
        """Generate several distinct counterfactual candidates per target."""

        if not targets:
            return {}
        if candidates_per_target <= 0:
            raise ValueError("candidates_per_target must be positive")
        results: dict[str, list[dict]] = {
            str(target["unit_id"]): [] for target in targets
        }
        forbidden_by_id = {
            str(target["unit_id"]): {
                str(item).lower() for item in target.get("forbidden", ())
            }
            for target in targets
        }
        for target in targets:
            unit_id = str(target["unit_id"])
            forbidden_by_id[unit_id].add(str(target["token"]).lower())

        for _ in range(candidates_per_target):
            round_targets = []
            for target in targets:
                unit_id = str(target["unit_id"])
                round_targets.append(
                    {
                        **target,
                        "forbidden": tuple(sorted(forbidden_by_id[unit_id])),
                    }
                )
            with ThreadPoolExecutor(
                max_workers=min(16, len(round_targets))
            ) as executor:
                generated_rows = list(
                    executor.map(
                        lambda target: self.replace(
                            str(target["token"]),
                            str(target["context"]),
                            str(target.get("unit_type", "")),
                            pos_hint=str(target.get("pos_hint", "")),
                            tag_hint=str(target.get("tag_hint", "")),
                            forbidden=tuple(target.get("forbidden", ())),
                        ),
                        round_targets,
                    )
                )
            for target, candidate in zip(targets, generated_rows):
                unit_id = str(target["unit_id"])
                replacement = str(candidate.get("new", "")).strip()
                lowered = replacement.lower()
                if (
                    not candidate.get("ok")
                    or not replacement
                    or lowered in forbidden_by_id[unit_id]
                    or any(char.isspace() for char in replacement)
                ):
                    continue
                forbidden_by_id[unit_id].add(lowered)
                results[unit_id].append(
                    {
                        **candidate,
                        "policy": "generic_llm_candidate_pool",
                    }
                )
        return results


def generate_valid_replacement(
    unit: dict,
    context: str,
    library,
    generic_editor: GenericReplacementClient,
    nlp,
    *,
    max_generic_attempts: int = 2,
    allow_relaxed_fallback: bool = True,
) -> dict:
    """Generate and contextually validate one non-deleting token replacement."""

    rejected = []
    relaxed_candidates = []
    typed = library.replacement_for_unit(unit, context)
    if typed.get("ok"):
        validation = validate_contextual_replacement(unit, context, typed, nlp)
        if validation["valid"]:
            return {**typed, "validation": validation}
        relaxed_candidates.append((typed, validation))
        rejected.append(str(typed.get("new", "")))
    for _ in range(max_generic_attempts):
        candidate = generic_editor.replace(
            str(unit.get("text", "")),
            context,
            str(unit.get("type", "")),
            pos_hint=str(unit.get("pos", "")),
            tag_hint=str(unit.get("tag", "")),
            forbidden=tuple(rejected),
        )
        validation = validate_contextual_replacement(unit, context, candidate, nlp)
        if candidate.get("ok") and validation["valid"]:
            return {**candidate, "validation": validation}
        if candidate.get("ok"):
            relaxed_candidates.append((candidate, validation))
        rejected.append(str(candidate.get("new", "")))
    if allow_relaxed_fallback:
        fallback = deterministic_fallback(
            str(unit.get("text", "")),
            str(unit.get("type", "")),
            error="strict_contextual_validation_failed",
        )
        relaxed_candidates.append(
            (
                fallback,
                validate_contextual_replacement(
                    unit,
                    context,
                    fallback,
                    nlp,
                ),
            )
        )
        for candidate, strict_validation in relaxed_candidates:
            surface_validation = validate_surface_replacement(unit, context, candidate)
            if surface_validation["valid"]:
                return {
                    **candidate,
                    "policy": str(candidate.get("policy", "replacement")) + "_relaxed",
                    "validation": {
                        **surface_validation,
                        "strict_validation": strict_validation,
                    },
                }
    return {
        "ok": False,
        "old": str(unit.get("text", "")),
        "new": "",
        "policy": "no_valid_contextual_replacement",
        "rejected": [item for item in rejected if item],
        "validation": {"valid": False, "reason": "no_valid_candidate"},
    }


def build_selected_replacements(
    selected: list[dict],
    contexts: list[dict],
    library,
    generic_editor: GenericReplacementClient,
    nlp,
    replacement_cache: dict[str, dict] | None = None,
    *,
    allow_relaxed_fallback: bool = True,
) -> tuple[dict[str, dict], list[dict]]:
    """Resolve one frozen replacement for every selected surface token."""

    context_by_id = {
        str(context["chunk_id"]): str(context["text"]) for context in contexts
    }
    replacements = {}
    rejected = []
    for unit in selected:
        unit_id = str(unit["unit_id"])
        context = context_by_id[str(unit["chunk_id"])]
        replacement = (
            replacement_cache.get(unit_id) if replacement_cache is not None else None
        )
        if replacement is None:
            replacement = generate_valid_replacement(
                unit,
                context,
                library,
                generic_editor,
                nlp,
                allow_relaxed_fallback=allow_relaxed_fallback,
            )
            if replacement_cache is not None:
                replacement_cache[unit_id] = replacement
        if replacement.get("ok"):
            replacements[unit_id] = replacement
        else:
            rejected.append(
                {
                    **unit,
                    "replacement_failure": replacement,
                }
            )
    return replacements, rejected


def build_selected_replacements_batched(
    selected: list[dict],
    contexts: list[dict],
    library,
    generic_editor: GenericReplacementClient,
    nlp,
    replacement_cache: dict[str, dict] | None = None,
    *,
    max_generic_attempts: int = 2,
) -> tuple[dict[str, dict], list[dict]]:
    """Resolve strict replacements while batching missing LLM generations."""

    context_by_id = {
        str(context["chunk_id"]): str(context["text"]) for context in contexts
    }
    replacements = {}
    unresolved = {}
    rejected_values = {}
    typed_pending = {}

    for unit in selected:
        unit_id = str(unit["unit_id"])
        cached = (
            replacement_cache.get(unit_id)
            if replacement_cache is not None
            else None
        )
        if cached is not None:
            if cached.get("ok"):
                replacements[unit_id] = cached
            else:
                unresolved[unit_id] = unit
            continue

        context = context_by_id[str(unit["chunk_id"])]
        typed = library.replacement_for_unit(unit, context)
        if typed.get("ok"):
            typed_pending[unit_id] = (unit, context, typed)
        else:
            unresolved[unit_id] = unit

    if typed_pending:
        typed_ids = list(typed_pending)
        validations = _validate_replacement_items(
            nlp,
            [
                {
                    "unit": typed_pending[unit_id][0],
                    "context": typed_pending[unit_id][1],
                    "replacement": typed_pending[unit_id][2],
                }
                for unit_id in typed_ids
            ],
        )
        for unit_id, validation in zip(typed_ids, validations):
            unit, _, typed = typed_pending[unit_id]
            if validation["valid"]:
                replacement = {**typed, "validation": validation}
                replacements[unit_id] = replacement
                if replacement_cache is not None:
                    replacement_cache[unit_id] = replacement
            else:
                rejected_values.setdefault(unit_id, []).append(
                    str(typed.get("new", ""))
                )
                unresolved[unit_id] = unit

    for _ in range(max_generic_attempts):
        if not unresolved:
            break
        targets = []
        for unit_id, unit in unresolved.items():
            targets.append(
                {
                    "unit_id": unit_id,
                    "token": str(unit.get("text", "")),
                    "chunk_id": str(unit["chunk_id"]),
                    "context": context_by_id[str(unit["chunk_id"])],
                    "unit_type": str(unit.get("type", "")),
                    "pos_hint": str(unit.get("pos", "")),
                    "tag_hint": str(unit.get("tag", "")),
                    "forbidden": tuple(rejected_values.get(unit_id, [])),
                }
            )
        candidates = generic_editor.replace_many(targets)
        next_unresolved = {}
        unresolved_ids = list(unresolved)
        validations = _validate_replacement_items(
            nlp,
            [
                {
                    "unit": unresolved[unit_id],
                    "context": context_by_id[
                        str(unresolved[unit_id]["chunk_id"])
                    ],
                    "replacement": candidates.get(unit_id, {}),
                }
                for unit_id in unresolved_ids
            ],
        )
        for unit_id, validation in zip(unresolved_ids, validations):
            unit = unresolved[unit_id]
            candidate = candidates.get(unit_id, {})
            if candidate.get("ok") and validation["valid"]:
                replacement = {**candidate, "validation": validation}
                replacements[unit_id] = replacement
                if replacement_cache is not None:
                    replacement_cache[unit_id] = replacement
            else:
                rejected_values.setdefault(unit_id, []).append(
                    str(candidate.get("new", ""))
                )
                next_unresolved[unit_id] = unit
        unresolved = next_unresolved

    rejected = []
    for unit_id, unit in unresolved.items():
        failure = {
            "ok": False,
            "old": str(unit.get("text", "")),
            "new": "",
            "policy": "no_valid_contextual_replacement",
            "rejected": [
                value for value in rejected_values.get(unit_id, []) if value
            ],
            "validation": {"valid": False, "reason": "no_valid_candidate"},
        }
        if replacement_cache is not None:
            replacement_cache[unit_id] = failure
        rejected.append({**unit, "replacement_failure": failure})
    return replacements, rejected


def build_executable_replacements_batched(
    selected: list[dict],
    contexts: list[dict],
    library,
    generic_editor: GenericReplacementClient,
    nlp,
    replacement_cache: dict[str, dict] | None = None,
    *,
    max_generic_attempts: int = 2,
    trust_cached_generic: bool = False,
) -> tuple[dict[str, dict], list[dict]]:
    """Resolve legal edits only after a selector has proposed token positions.

    Selection is intentionally independent of replacement availability. Function
    words and tokens for which no meaningful counterfactual can be produced are
    reported as skipped; callers must not remove them from the graph and re-solve.
    """

    context_by_id = {
        str(context["chunk_id"]): str(context["text"]) for context in contexts
    }
    replacements: dict[str, dict] = {}
    skipped: list[dict] = []
    typed_pending: dict[str, tuple[dict, str, dict]] = {}
    generic_pending: dict[str, dict] = {}
    rejected_values: dict[str, list[str]] = {}

    def skip(unit: dict, reason: str, **details) -> None:
        failure = {
            "ok": False,
            "old": str(unit.get("text", "")),
            "new": "",
            "policy": "skip_after_selection",
            "validation": {"valid": False, "reason": reason, **details},
        }
        unit_id = str(unit["unit_id"])
        if replacement_cache is not None:
            replacement_cache[unit_id] = failure
        skipped.append({**unit, "replacement_failure": failure})

    for unit in selected:
        unit_id = str(unit["unit_id"])
        context = context_by_id.get(str(unit.get("chunk_id", "")), "")
        cached = replacement_cache.get(unit_id) if replacement_cache is not None else None
        if cached and cached.get("ok"):
            policy = str(cached.get("policy", ""))
            if trust_cached_generic or policy not in _UNTRUSTED_GENERIC_POLICIES:
                replacements[unit_id] = cached
                continue
        if (
            cached
            and not cached.get("ok")
            and str(cached.get("policy", "")) == "skip_after_selection"
        ):
            skipped.append({**unit, "replacement_failure": cached})
            continue

        token = str(unit.get("text", "")).strip()
        unit_type = str(unit.get("type", "")).upper()
        pos = str(unit.get("pos", "")).upper()
        inferred_type = library.infer_type(token, unit_type, context)
        if (
            not token
            or not any(char.isalnum() for char in token)
            or unit_type in _NON_SEMANTIC_TYPES
            or pos not in _CONTENT_POS
        ):
            skip(
                unit,
                "no_semantic_counterfactual",
                unit_type=unit_type,
                pos=pos,
            )
            continue

        typed = library.replacement_for_unit(unit, context)
        if typed.get("ok"):
            typed_pending[unit_id] = (unit, context, typed)
        elif inferred_type or unit_type in {"CONTENT", "PROPER"}:
            generic_pending[unit_id] = unit
        else:
            skip(
                unit,
                "no_supported_semantic_type",
                unit_type=unit_type,
                pos=pos,
            )

    if typed_pending:
        typed_ids = list(typed_pending)
        validations = _validate_replacement_items(
            nlp,
            [
                {
                    "unit": typed_pending[unit_id][0],
                    "context": typed_pending[unit_id][1],
                    "replacement": typed_pending[unit_id][2],
                }
                for unit_id in typed_ids
            ],
        )
        for unit_id, validation in zip(typed_ids, validations):
            unit, _, candidate = typed_pending[unit_id]
            if validation["valid"]:
                replacement = {**candidate, "validation": validation}
                replacements[unit_id] = replacement
                if replacement_cache is not None:
                    replacement_cache[unit_id] = replacement
            else:
                rejected_values.setdefault(unit_id, []).append(
                    str(candidate.get("new", ""))
                )
                generic_pending[unit_id] = unit

    for _ in range(max_generic_attempts):
        if not generic_pending:
            break
        candidates = generic_editor.replace_many([
            {
                "unit_id": unit_id,
                "chunk_id": str(unit["chunk_id"]),
                "token": str(unit.get("text", "")),
                "context": context_by_id[str(unit["chunk_id"])],
                "unit_type": str(unit.get("type", "")),
                "pos_hint": str(unit.get("pos", "")),
                "tag_hint": str(unit.get("tag", "")),
                "forbidden": tuple(rejected_values.get(unit_id, [])),
            }
            for unit_id, unit in generic_pending.items()
        ])
        pending_ids = list(generic_pending)
        validations = _validate_replacement_items(
            nlp,
            [
                {
                    "unit": generic_pending[unit_id],
                    "context": context_by_id[str(generic_pending[unit_id]["chunk_id"])],
                    "replacement": candidates.get(unit_id, {}),
                }
                for unit_id in pending_ids
            ],
        )
        next_pending = {}
        for unit_id, validation in zip(pending_ids, validations):
            unit = generic_pending[unit_id]
            candidate = candidates.get(unit_id, {})
            policy = str(candidate.get("policy", ""))
            if (
                candidate.get("ok")
                and validation["valid"]
                and policy != "generic_deterministic_fallback"
            ):
                replacement = {
                    **candidate,
                    "generator_policy": policy,
                    "policy": "online_llm_counterfactual",
                    "inferred_type": (
                        str(candidate.get("inferred_type", ""))
                        or str(unit.get("type", ""))
                    ),
                    "validation": validation,
                }
                replacements[unit_id] = replacement
                if replacement_cache is not None:
                    replacement_cache[unit_id] = replacement
            else:
                rejected_values.setdefault(unit_id, []).append(
                    str(candidate.get("new", ""))
                )
                next_pending[unit_id] = unit
        generic_pending = next_pending

    for unit_id, unit in generic_pending.items():
        skip(
            unit,
            "no_valid_counterfactual_candidate",
            rejected=[value for value in rejected_values.get(unit_id, []) if value],
        )
    return replacements, skipped


def _validate_replacement_items(nlp, items: list[dict]) -> list[dict]:
    if hasattr(nlp, "validate_many"):
        return nlp.validate_many(items)
    return [
        validate_contextual_replacement(
            item["unit"],
            item["context"],
            item["replacement"],
            nlp,
        )
        for item in items
    ]


def validate_surface_replacement(unit: dict, context: str, replacement: dict) -> dict:
    """Validate the intervention contract when every chunk token is endogenous."""

    new = str(replacement.get("new", ""))
    old = str(unit.get("text", ""))
    start = int(unit.get("chunk_char_start", -1))
    end = int(unit.get("chunk_char_end", -1))
    if start < 0 or end <= start or context[start:end] != old:
        return {"valid": False, "reason": "offset_mismatch"}
    if not new or new.lower() == old.lower() or any(char.isspace() for char in new):
        return {"valid": False, "reason": "invalid_surface_form"}
    return {
        "valid": True,
        "reason": "relaxed_all_tokens_endogenous",
        "original_surface": old,
        "replacement_surface": new,
    }


def validate_contextual_replacement(
    unit: dict, context: str, replacement: dict, nlp
) -> dict:
    """Require a candidate to preserve contextual POS and basic morphology."""

    if hasattr(nlp, "validate"):
        return nlp.validate(unit, context, replacement)

    new = str(replacement.get("new", ""))
    old = str(unit.get("text", ""))
    start = int(unit.get("chunk_char_start", -1))
    end = int(unit.get("chunk_char_end", -1))
    if start < 0 or end <= start or context[start:end] != old:
        return {"valid": False, "reason": "offset_mismatch"}
    if not new or new.lower() == old.lower() or any(char.isspace() for char in new):
        return {"valid": False, "reason": "invalid_surface_form"}
    left = (
        max(
            context.rfind(".", 0, start),
            context.rfind("!", 0, start),
            context.rfind("?", 0, start),
        )
        + 1
    )
    right_candidates = [
        position
        for position in (
            context.find(".", end),
            context.find("!", end),
            context.find("?", end),
        )
        if position >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else len(context)
    sentence = context[left:right]
    local_start = start - left
    revised = sentence[:local_start] + new + sentence[end - left :]
    original_doc = nlp(sentence)
    revised_doc = nlp(revised)
    original_token = next(
        (
            token
            for token in original_doc
            if token.idx <= local_start < token.idx + len(token)
        ),
        None,
    )
    revised_token = next(
        (
            token
            for token in revised_doc
            if token.idx <= local_start < token.idx + len(token)
        ),
        None,
    )
    if original_token is None or revised_token is None:
        return {"valid": False, "reason": "tokenization_mismatch"}
    details = {
        "valid": False,
        "reason": "",
        "original_pos": original_token.pos_,
        "replacement_pos": revised_token.pos_,
        "original_tag": original_token.tag_,
        "replacement_tag": revised_token.tag_,
    }
    if original_token.pos_ != revised_token.pos_:
        return {**details, "reason": "pos_mismatch"}
    strict_tag_pos = {"VERB", "AUX", "DET", "ADP", "CCONJ", "SCONJ", "PRON", "PART"}
    if (
        original_token.pos_ in strict_tag_pos
        and original_token.tag_ != revised_token.tag_
    ):
        return {**details, "reason": "tag_mismatch"}
    for feature in ("Number", "Tense", "VerbForm", "Person"):
        original_value = original_token.morph.get(feature)
        replacement_value = revised_token.morph.get(feature)
        if original_value and original_value != replacement_value:
            return {**details, "reason": f"morph_mismatch:{feature}"}
    if str(replacement.get("policy", "")).startswith("generic_llm"):
        similarity = _token_similarity(original_token, revised_token)
        if similarity is not None and similarity >= 0.72:
            return {
                **details,
                "reason": "semantic_near_duplicate",
                "semantic_similarity": similarity,
            }
    return {**details, "valid": True, "reason": "ok"}


def validate_contextual_replacements_batch(
    items: list[dict],
    nlp,
) -> list[dict]:
    """Validate many contextual replacements with one batched spaCy pass."""

    results: list[dict | None] = [None] * len(items)
    texts = []
    prepared = []
    for index, item in enumerate(items):
        unit = dict(item.get("unit", {}))
        context = str(item.get("context", ""))
        replacement = dict(item.get("replacement", {}))
        new = str(replacement.get("new", ""))
        old = str(unit.get("text", ""))
        start = int(unit.get("chunk_char_start", -1))
        end = int(unit.get("chunk_char_end", -1))
        if start < 0 or end <= start or context[start:end] != old:
            results[index] = {"valid": False, "reason": "offset_mismatch"}
            continue
        if (
            not new
            or new.lower() == old.lower()
            or any(char.isspace() for char in new)
        ):
            results[index] = {"valid": False, "reason": "invalid_surface_form"}
            continue

        left = (
            max(
                context.rfind(".", 0, start),
                context.rfind("!", 0, start),
                context.rfind("?", 0, start),
            )
            + 1
        )
        right_candidates = [
            position
            for position in (
                context.find(".", end),
                context.find("!", end),
                context.find("?", end),
            )
            if position >= 0
        ]
        right = min(right_candidates) + 1 if right_candidates else len(context)
        sentence = context[left:right]
        local_start = start - left
        revised = sentence[:local_start] + new + sentence[end - left :]
        text_offset = len(texts)
        texts.extend((sentence, revised))
        prepared.append((index, text_offset, local_start))

    docs = list(nlp.pipe(texts, batch_size=64)) if texts else []
    strict_tag_pos = {
        "VERB",
        "AUX",
        "DET",
        "ADP",
        "CCONJ",
        "SCONJ",
        "PRON",
        "PART",
    }
    for index, text_offset, local_start in prepared:
        original_doc = docs[text_offset]
        revised_doc = docs[text_offset + 1]
        original_token = next(
            (
                token
                for token in original_doc
                if token.idx <= local_start < token.idx + len(token)
            ),
            None,
        )
        revised_token = next(
            (
                token
                for token in revised_doc
                if token.idx <= local_start < token.idx + len(token)
            ),
            None,
        )
        if original_token is None or revised_token is None:
            results[index] = {
                "valid": False,
                "reason": "tokenization_mismatch",
            }
            continue
        details = {
            "valid": False,
            "reason": "",
            "original_pos": original_token.pos_,
            "replacement_pos": revised_token.pos_,
            "original_tag": original_token.tag_,
            "replacement_tag": revised_token.tag_,
        }
        if original_token.pos_ != revised_token.pos_:
            results[index] = {**details, "reason": "pos_mismatch"}
            continue
        if (
            original_token.pos_ in strict_tag_pos
            and original_token.tag_ != revised_token.tag_
        ):
            results[index] = {**details, "reason": "tag_mismatch"}
            continue
        mismatch = ""
        for feature in ("Number", "Tense", "VerbForm", "Person"):
            original_value = original_token.morph.get(feature)
            replacement_value = revised_token.morph.get(feature)
            if original_value and original_value != replacement_value:
                mismatch = f"morph_mismatch:{feature}"
                break
        if mismatch:
            results[index] = {**details, "reason": mismatch}
            continue
        replacement = dict(items[index].get("replacement", {}))
        if str(replacement.get("policy", "")).startswith("generic_llm"):
            similarity = _token_similarity(original_token, revised_token)
            if similarity is not None and similarity >= 0.72:
                results[index] = {
                    **details,
                    "reason": "semantic_near_duplicate",
                    "semantic_similarity": similarity,
                }
                continue
        results[index] = {**details, "valid": True, "reason": "ok"}

    return [
        result
        if result is not None
        else {"valid": False, "reason": "batch_validation_failed"}
        for result in results
    ]


def _token_similarity(original_token, replacement_token) -> float | None:
    if not original_token.has_vector or not replacement_token.has_vector:
        return None
    if original_token.vector_norm <= 1e-12 or replacement_token.vector_norm <= 1e-12:
        return None
    return float(original_token.similarity(replacement_token))


def deterministic_fallback(token: str, unit_type: str = "", *, error: str = "") -> dict:
    """Guarantee a non-deleting replacement when the editor response is malformed."""

    swaps = {
        "a": "the",
        "an": "the",
        "the": "a",
        "and": "or",
        "or": "and",
        "is": "was",
        "was": "is",
        "are": "were",
        "were": "are",
        "this": "that",
        "that": "this",
        "these": "those",
        "those": "these",
    }
    lowered = token.lower()
    replacement = swaps.get(lowered)
    if replacement is None:
        replacement = "different" if lowered != "different" else "changed"
    if token.isupper():
        replacement = replacement.upper()
    elif token[:1].isupper():
        replacement = replacement[:1].upper() + replacement[1:]
    return {
        "ok": True,
        "old": token,
        "new": replacement,
        "policy": "generic_deterministic_fallback",
        "fallback_error": error,
    }
