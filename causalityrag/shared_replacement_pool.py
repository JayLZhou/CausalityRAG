"""Method-independent counterfactual replacement-pool primitives."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable


POOL_SCHEMA = "causalityrag.shared_counterfactual_pool.v1"
EDITABLE_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
NON_SEMANTIC_TYPES = {"STOPWORD"}


def is_editable_unit(unit: dict) -> bool:
    """Return whether a token belongs to the shared editable domain."""

    token = str(unit.get("text", "")).strip()
    return (
        bool(token)
        and any(character.isalnum() for character in token)
        and str(unit.get("pos", "")).upper() in EDITABLE_POS
        and str(unit.get("type", "")).upper() not in NON_SEMANTIC_TYPES
    )


def typed_pool_key(unit: dict) -> str:
    """Return a stable key for context-independent candidate generation."""

    fields = {
        "surface": str(unit.get("text", "")).strip().casefold(),
        "type": str(unit.get("type", "")).upper(),
        "pos": str(unit.get("pos", "")).upper(),
        "tag": str(unit.get("tag", "")).upper(),
        "morph": _canonical_morph(unit.get("morph", "")),
        "entity_token_index": unit.get("entity_token_index"),
        "entity_token_count": unit.get("entity_token_count"),
    }
    encoded = json.dumps(
        fields,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "typed:" + hashlib.sha256(encoded).hexdigest()


def stable_shared_candidate(
    candidates: list[dict],
    *,
    unit_id: str,
    seed: int,
) -> dict:
    """Choose the same replacement for a token position in every method."""

    if not candidates:
        return {}
    material = f"{seed}\0{unit_id}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return dict(candidates[index % len(candidates)])


class FrozenSharedReplacementPool:
    """Read-only position-level pool shared by every evaluated method."""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        self._rows: dict[str, dict] = {}
        self._excluded: dict[str, dict] = {}
        with open(self.path, encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema") != POOL_SCHEMA:
                    raise ValueError(
                        f"invalid pool schema on line {line_number}"
                    )
                unit_id = str(row.get("unit_id", ""))
                candidates = row.get("candidates", [])
                if row.get("row_kind") == "excluded_position":
                    if not unit_id or candidates:
                        raise ValueError(
                            f"invalid exclusion row on line {line_number}"
                        )
                    self._excluded[unit_id] = row
                    continue
                if not unit_id or not isinstance(candidates, list) or not candidates:
                    raise ValueError(
                        f"incomplete pool row on line {line_number}"
                    )
                previous = self._rows.get(unit_id)
                if previous is not None and previous != row:
                    raise ValueError(f"conflicting pool rows for {unit_id}")
                self._rows[unit_id] = row

    def get(self, unit_id: str) -> dict | None:
        row = self._rows.get(str(unit_id))
        return dict(row) if row is not None else None

    def is_eligible(self, unit_id: str) -> bool:
        return str(unit_id) in self._rows

    def is_excluded(self, unit_id: str) -> bool:
        return str(unit_id) in self._excluded

    def exclusion(self, unit_id: str) -> dict | None:
        row = self._excluded.get(str(unit_id))
        return dict(row) if row is not None else None

    def require(self, unit_ids: Iterable[str]) -> dict[str, dict]:
        """Return rows or fail closed when any selected token is uncovered."""

        normalized = [str(unit_id) for unit_id in unit_ids]
        missing = [unit_id for unit_id in normalized if unit_id not in self._rows]
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(
                f"shared replacement pool misses {len(missing)} units: {preview}"
            )
        return {unit_id: dict(self._rows[unit_id]) for unit_id in normalized}

    def __len__(self) -> int:
        return len(self._rows)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_morph(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _canonical_morph(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return sorted(str(item) for item in value)
    return str(value)
