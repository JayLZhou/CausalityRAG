"""spaCy validation for frozen counterfactual replacement candidates."""

from __future__ import annotations


SURFACE_TYPED_COUNTERFACTUALS = {
    "PERSON",
    "NORP",
    "FAC",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "DATE",
    "TIME",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "ORDINAL",
    "CARDINAL",
    "NUMBER",
    "PROPER",
}


def validate_contextual_replacement(
    unit: dict,
    context: str,
    replacement: dict,
    nlp,
) -> dict:
    if hasattr(nlp, "validate"):
        return nlp.validate(unit, context, replacement)
    return validate_contextual_replacements_batch(
        [{"unit": unit, "context": context, "replacement": replacement}],
        nlp,
    )[0]


def validate_contextual_replacements_batch(
    items: list[dict],
    nlp,
) -> list[dict]:
    """Validate replacements with one batched spaCy pass."""

    results: list[dict | None] = [None] * len(items)
    texts = []
    prepared = []
    for index, item in enumerate(items):
        unit = dict(item.get("unit", {}))
        context = str(item.get("context", ""))
        replacement = dict(item.get("replacement", {}))
        old = str(unit.get("text", ""))
        new = str(replacement.get("new", ""))
        start = int(unit.get("chunk_char_start", -1))
        end = int(unit.get("chunk_char_end", -1))
        if start < 0 or end <= start or context[start:end] != old:
            results[index] = {"valid": False, "reason": "offset_mismatch"}
            continue
        if (
            not new
            or new.casefold() == old.casefold()
            or any(character.isspace() for character in new)
        ):
            results[index] = {
                "valid": False,
                "reason": "invalid_surface_form",
            }
            continue

        left = max(
            context.rfind(".", 0, start),
            context.rfind("!", 0, start),
            context.rfind("?", 0, start),
            context.rfind("\n", 0, start),
        ) + 1
        right_candidates = [
            position
            for delimiter in (".", "!", "?", "\n")
            if (position := context.find(delimiter, end)) >= 0
        ]
        right = (
            min(right_candidates) + 1
            if right_candidates
            else len(context)
        )
        sentence = context[left:right]
        local_start = start - left
        revised = sentence[:local_start] + new + sentence[end - left :]
        text_offset = len(texts)
        texts.extend((sentence, revised))
        prepared.append((index, text_offset, local_start))

    docs = list(nlp.pipe(texts, batch_size=64)) if texts else []
    for index, text_offset, local_start in prepared:
        original_token = _token_at(docs[text_offset], local_start)
        replacement_token = _token_at(docs[text_offset + 1], local_start)
        if original_token is None or replacement_token is None:
            results[index] = {
                "valid": False,
                "reason": "tokenization_mismatch",
            }
            continue
        details = {
            "valid": False,
            "reason": "",
            "original_pos": original_token.pos_,
            "replacement_pos": replacement_token.pos_,
            "original_tag": original_token.tag_,
            "replacement_tag": replacement_token.tag_,
        }
        error = _grammar_error(
            original_token,
            replacement_token,
            str(items[index].get("unit", {}).get("type", "")),
        )
        results[index] = (
            {**details, "reason": error}
            if error
            else {**details, "valid": True, "reason": "ok"}
        )
    return [
        result
        if result is not None
        else {"valid": False, "reason": "batch_validation_failed"}
        for result in results
    ]


def _token_at(document, offset: int):
    return next(
        (
            token
            for token in document
            if token.idx <= offset < token.idx + len(token)
        ),
        None,
    )


def _grammar_error(original, replacement, unit_type: str) -> str:
    if str(unit_type).upper() in SURFACE_TYPED_COUNTERFACTUALS:
        return ""
    if original.pos_ != replacement.pos_:
        return "pos_mismatch"
    if original.pos_ == "VERB":
        if original.tag_ in {"VBD", "VBN"}:
            return "" if replacement.tag_ in {"VBD", "VBN"} else "tag_mismatch"
        return "" if original.tag_ == replacement.tag_ else "tag_mismatch"
    if original.pos_ in {
        "AUX",
        "DET",
        "ADP",
        "CCONJ",
        "SCONJ",
        "PRON",
        "PART",
    } and original.tag_ != replacement.tag_:
        return "tag_mismatch"
    for feature in ("Number", "Tense", "VerbForm", "Person"):
        original_value = original.morph.get(feature)
        replacement_value = replacement.morph.get(feature)
        if original_value and original_value != replacement_value:
            return f"morph_mismatch:{feature}"
    return ""
