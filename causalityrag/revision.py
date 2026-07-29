"""Apply frozen non-deleting token replacements to retrieved contexts."""

from __future__ import annotations

from causalityrag.io import retrieved_contexts


def apply_token_replacements(
    record: dict,
    selected_units: list[dict],
    replacements: dict[str, dict],
    *,
    k: int = 5,
    allow_whitespace: bool = False,
    allow_case_only: bool = False,
) -> dict:
    contexts = retrieved_contexts(record)
    if k:
        contexts = contexts[:k]
    by_chunk = {context["chunk_id"]: dict(context) for context in contexts}
    edits = []
    grouped = {}
    for unit in selected_units:
        grouped.setdefault(str(unit.get("chunk_id", "")), []).append(unit)

    for chunk_id, units in grouped.items():
        context = by_chunk.get(chunk_id)
        if context is None:
            continue
        text = str(context["text"])
        for unit in sorted(
            units,
            key=lambda item: int(item.get("chunk_char_start", -1)),
            reverse=True,
        ):
            unit_id = str(unit.get("unit_id", ""))
            start = int(unit.get("chunk_char_start", -1))
            end = int(unit.get("chunk_char_end", -1))
            old = str(unit.get("text", ""))
            replacement = replacements.get(unit_id, {})
            new = str(replacement.get("new", ""))
            base = {
                "unit_id": unit_id,
                "chunk_id": chunk_id,
                "token": old,
                "chunk_char_start": start,
                "chunk_char_end": end,
            }
            if start < 0 or end <= start or text[start:end] != old:
                edits.append({
                    **base,
                    "ok": False,
                    "new": new,
                    "note": "offset_mismatch",
                })
                continue
            if (
                not new
                or (
                    not allow_case_only
                    and new.casefold() == old.casefold()
                )
                or (
                    not allow_whitespace
                    and any(character.isspace() for character in new)
                )
            ):
                edits.append({
                    **base,
                    "ok": False,
                    "new": new,
                    "note": "invalid_replacement",
                })
                continue
            text = text[:start] + new + text[end:]
            edits.append({
                **base,
                "ok": True,
                "old": old,
                "new": new,
                "policy": replacement.get("policy", ""),
                "validation": replacement.get("validation"),
                "note": "replace",
            })
        context["text"] = text

    return {
        "edited_contexts": [
            by_chunk[context["chunk_id"]] for context in contexts
        ],
        "edits": list(reversed(edits)),
        "n_edits": sum(bool(edit.get("ok")) for edit in edits),
        "n_failed_edits": sum(not edit.get("ok") for edit in edits),
    }
