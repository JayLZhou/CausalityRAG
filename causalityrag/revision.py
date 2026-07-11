"""Apply typed token revisions to retrieved contexts."""

from __future__ import annotations

from causalityrag.io import retrieved_contexts
from causalityrag.rules import TypedRuleLibrary


def apply_typed_token_revisions(
    record: dict,
    selected_units: list[dict],
    library: TypedRuleLibrary,
    *,
    k: int = 5,
    max_edits: int = 0,
) -> dict:
    contexts = retrieved_contexts(record)
    if k:
        contexts = contexts[:k]
    by_chunk = {ctx["chunk_id"]: dict(ctx) for ctx in contexts}
    chosen = list(selected_units)
    if max_edits:
        chosen = sorted(chosen, key=lambda unit: (-float(unit.get("support", 0.0)), unit.get("unit_id", "")))[:max_edits]

    edits = []
    for chunk_id, units in _group_by_chunk(chosen).items():
        context = by_chunk.get(chunk_id)
        if not context:
            continue
        text = context["text"]
        for unit in sorted(units, key=lambda item: int(item.get("chunk_char_start", -1)), reverse=True):
            start = int(unit.get("chunk_char_start", -1))
            end = int(unit.get("chunk_char_end", -1))
            old = str(unit.get("text", ""))
            if start < 0 or end <= start or text[start:end] != old:
                edits.append({**_edit_base(unit), "ok": False, "note": "offset mismatch"})
                continue
            repl = library.replacement_for_token(old, str(unit.get("type", "")))
            if not repl["ok"]:
                edits.append({**_edit_base(unit), **repl, "note": "no typed replacement"})
                continue
            text = text[:start] + repl["new"] + text[end:]
            edits.append({**_edit_base(unit), **repl, "note": ""})
        context["text"] = text

    edited_contexts = [by_chunk[ctx["chunk_id"]] for ctx in contexts]
    return {
        "edited_contexts": edited_contexts,
        "edits": list(reversed(edits)),
        "n_edits": sum(1 for edit in edits if edit.get("ok")),
        "n_failed_edits": sum(1 for edit in edits if not edit.get("ok")),
    }


def apply_token_deletions(
    record: dict,
    selected_units: list[dict],
    *,
    k: int = 5,
) -> dict:
    """Delete arbitrary selected chunk-token spans.

    This is the universal token-level intervention used when every surface
    word token is editable.  It deliberately does not consult answer text,
    types, or a replacement model.
    """

    contexts = retrieved_contexts(record)
    if k:
        contexts = contexts[:k]
    by_chunk = {ctx["chunk_id"]: dict(ctx) for ctx in contexts}
    edits = []
    for chunk_id, units in _group_by_chunk(selected_units).items():
        context = by_chunk.get(chunk_id)
        if not context:
            continue
        text = context["text"]
        for unit in sorted(units, key=lambda item: int(item.get("chunk_char_start", -1)), reverse=True):
            start = int(unit.get("chunk_char_start", -1))
            end = int(unit.get("chunk_char_end", -1))
            old = str(unit.get("text", ""))
            if start < 0 or end <= start or text[start:end] != old:
                edits.append({**_edit_base(unit), "ok": False, "new": "", "note": "offset mismatch"})
                continue
            text = text[:start] + text[end:]
            edits.append({**_edit_base(unit), "ok": True, "new": "", "note": "delete"})
        context["text"] = text

    edited_contexts = [by_chunk[ctx["chunk_id"]] for ctx in contexts]
    return {
        "edited_contexts": edited_contexts,
        "edits": list(reversed(edits)),
        "n_edits": sum(1 for edit in edits if edit.get("ok")),
        "n_failed_edits": sum(1 for edit in edits if not edit.get("ok")),
    }


def apply_token_replacements(
    record: dict,
    selected_units: list[dict],
    replacements: dict[str, dict],
    *,
    k: int = 5,
) -> dict:
    """Apply non-deleting replacements to arbitrary selected token spans."""

    contexts = retrieved_contexts(record)
    if k:
        contexts = contexts[:k]
    by_chunk = {ctx["chunk_id"]: dict(ctx) for ctx in contexts}
    edits = []
    for chunk_id, units in _group_by_chunk(selected_units).items():
        context = by_chunk.get(chunk_id)
        if not context:
            continue
        text = context["text"]
        for unit in sorted(units, key=lambda item: int(item.get("chunk_char_start", -1)), reverse=True):
            start = int(unit.get("chunk_char_start", -1))
            end = int(unit.get("chunk_char_end", -1))
            old = str(unit.get("text", ""))
            replacement = replacements.get(str(unit.get("unit_id", "")), {})
            new = str(replacement.get("new", ""))
            if start < 0 or end <= start or text[start:end] != old:
                edits.append({**_edit_base(unit), "ok": False, "new": new, "note": "offset mismatch"})
                continue
            if not new or new.lower() == old.lower() or any(char.isspace() for char in new):
                edits.append({**_edit_base(unit), "ok": False, "new": new, "note": "invalid replacement"})
                continue
            text = text[:start] + new + text[end:]
            edits.append({
                **_edit_base(unit),
                "ok": True,
                "old": old,
                "new": new,
                "policy": replacement.get("policy", ""),
                "validation": replacement.get("validation"),
                "note": "replace",
            })
        context["text"] = text

    edited_contexts = [by_chunk[ctx["chunk_id"]] for ctx in contexts]
    return {
        "edited_contexts": edited_contexts,
        "edits": list(reversed(edits)),
        "n_edits": sum(1 for edit in edits if edit.get("ok")),
        "n_failed_edits": sum(1 for edit in edits if not edit.get("ok")),
    }


def _group_by_chunk(units: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for unit in units:
        grouped.setdefault(str(unit.get("chunk_id", "")), []).append(unit)
    return grouped


def _edit_base(unit: dict) -> dict:
    return {
        "unit_id": unit.get("unit_id", ""),
        "chunk_id": unit.get("chunk_id", ""),
        "token": unit.get("text", ""),
        "chunk_char_start": unit.get("chunk_char_start"),
        "chunk_char_end": unit.get("chunk_char_end"),
        "support": unit.get("support", 0.0),
    }
