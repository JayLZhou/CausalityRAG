"""Generic contextual token replacement through the local OpenAI-compatible LLM."""

from __future__ import annotations

import json
import os
import urllib.request

from causalityrag.reader import parse_json_object


class GenericReplacementClient:
    """Ask the local editor model for one non-deleting word replacement."""

    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: int = 60) -> None:
        self.base_url = (base_url or os.environ.get("YVETTE_LLM_BASE_URL") or "http://127.0.0.1:8000/v1").rstrip("/")
        self.model = model or os.environ.get("YVETTE_LLM_MODEL") or "qwen2.5-7b"
        self.timeout = timeout

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
        prompt = (
            "Replace exactly one target word in the passage. Return STRICT JSON only: "
            '{"replacement":"..."}. The replacement must be one non-empty word, '
            "must differ from the original, must not delete text, and should preserve "
            "the target's basic part of speech, tense, number, and capitalization.\n\n"
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
                {"role": "system", "content": "You are a constrained counterfactual word editor."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 24,
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
            replacement = str(parsed.get("replacement", "")).strip() if isinstance(parsed, dict) else ""
        except Exception as exc:
            return deterministic_fallback(token, unit_type, error=str(exc))
        if (
            not replacement
            or replacement.lower() == token.lower()
            or replacement.lower() in {item.lower() for item in forbidden}
            or any(char.isspace() for char in replacement)
        ):
            return deterministic_fallback(token, unit_type, error="invalid_llm_response")
        return {"ok": True, "old": token, "new": replacement, "policy": "generic_llm_contextual"}


def generate_valid_replacement(
    unit: dict,
    context: str,
    library,
    generic_editor: GenericReplacementClient,
    nlp,
    *,
    max_generic_attempts: int = 2,
) -> dict:
    """Generate and contextually validate one non-deleting token replacement."""

    rejected = []
    typed = library.replacement_for_unit(unit, context)
    if typed.get("ok"):
        validation = validate_contextual_replacement(unit, context, typed, nlp)
        if validation["valid"]:
            return {**typed, "validation": validation}
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
        rejected.append(str(candidate.get("new", "")))
    return {
        "ok": False,
        "old": str(unit.get("text", "")),
        "new": "",
        "policy": "no_valid_contextual_replacement",
        "rejected": [item for item in rejected if item],
        "validation": {"valid": False, "reason": "no_valid_candidate"},
    }


def validate_contextual_replacement(unit: dict, context: str, replacement: dict, nlp) -> dict:
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
    left = max(context.rfind(".", 0, start), context.rfind("!", 0, start), context.rfind("?", 0, start)) + 1
    right_candidates = [
        position for position in (
            context.find(".", end), context.find("!", end), context.find("?", end)
        ) if position >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else len(context)
    sentence = context[left:right]
    local_start = start - left
    revised = sentence[:local_start] + new + sentence[end - left:]
    original_doc = nlp(sentence)
    revised_doc = nlp(revised)
    original_token = next(
        (token for token in original_doc if token.idx <= local_start < token.idx + len(token)),
        None,
    )
    revised_token = next(
        (token for token in revised_doc if token.idx <= local_start < token.idx + len(token)),
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
    if original_token.pos_ in strict_tag_pos and original_token.tag_ != revised_token.tag_:
        return {**details, "reason": "tag_mismatch"}
    for feature in ("Number", "Tense", "VerbForm", "Person"):
        original_value = original_token.morph.get(feature)
        replacement_value = revised_token.morph.get(feature)
        if original_value and original_value != replacement_value:
            return {**details, "reason": f"morph_mismatch:{feature}"}
    return {**details, "valid": True, "reason": "ok"}


def deterministic_fallback(token: str, unit_type: str = "", *, error: str = "") -> dict:
    """Guarantee a non-deleting replacement when the editor response is malformed."""

    swaps = {
        "a": "the", "an": "the", "the": "a", "and": "or", "or": "and",
        "is": "was", "was": "is", "are": "were", "were": "are",
        "this": "that", "that": "this", "these": "those", "those": "these",
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
