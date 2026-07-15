"""Typed counterfactual token replacements backed by the previous rule library."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from causalityrag.token_units import STOPWORDS, WORD_RE


TIME_TYPES = {"DATE", "TIME"}
NUMBER_TYPES = {"CARDINAL", "QUANTITY", "PERCENT", "ORDINAL", "MONEY"}
ENTITY_TYPES = {
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "LANGUAGE",
}
ONTO_TYPES = TIME_TYPES | NUMBER_TYPES | ENTITY_TYPES

TYPE_PRIORITY = [
    "PERSON", "GPE", "ORG", "LOC", "FAC", "WORK_OF_ART", "EVENT", "NORP",
    "PRODUCT", "LAW", "LANGUAGE",
]

RELATION_TOKEN_CF = {
    "born": "died",
    "died": "born",
    "directed": "produced",
    "produced": "directed",
    "wrote": "edited",
    "written": "edited",
    "created": "founded",
    "founded": "created",
    "located": "headquartered",
    "married": "divorced",
    "played": "coached",
    "starring": "featuring",
    "won": "lost",
    "served": "joined",
    "released": "announced",
    "known": "noted",
}

YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")
NUM_RE = re.compile(r"\d[\d,]*\.?\d*")
ORD_RE = re.compile(r"^(\d+)(st|nd|rd|th)$", re.I)


def load_pools(path: str | Path) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    role = {}
    for key, values in obj.get("role_pool", {}).items():
        if "\t" in key:
            rel, typ = key.split("\t", 1)
            role[(rel, typ)] = values
    return {"type_pool": obj.get("type_pool", {}), "role_pool": role}


def load_rule_metadata(path: str | Path | None) -> dict:
    if not path:
        return {}
    try:
        import yaml
    except Exception:
        return {}
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return {str(rule.get("value_type", "")).upper(): rule for rule in data.get("rules", [])}


class TypedRuleLibrary:
    """Executable token-level subset of the previous YVETTE rule library."""

    def __init__(self, pools: dict, rule_metadata: dict | None = None) -> None:
        self.type_pool = pools.get("type_pool", {})
        self.role_pool = pools.get("role_pool", {})
        self.rule_metadata = rule_metadata or {}
        self.component_types, self.component_pool = self._index_components(self.type_pool)

    @classmethod
    def from_files(cls, pools_path: str | Path, rules_path: str | Path | None = None) -> "TypedRuleLibrary":
        return cls(load_pools(pools_path), load_rule_metadata(rules_path))

    def replacement_for_token(self, token: str, unit_type: str = "", context: str = "") -> dict:
        inferred = self.infer_type(token, unit_type, context)
        if inferred == "RELATION_CUE":
            replacement = preserve_case(token, RELATION_TOKEN_CF.get(token.lower(), "changed"))
            return _replacement(token, replacement, inferred, "relation_token_swap")
        if inferred in TIME_TYPES:
            replacement = gen_time(token)
            return _replacement(token, replacement, inferred, "date_shift_same_granularity")
        if inferred in NUMBER_TYPES:
            replacement = gen_number(token)
            return _replacement(token, replacement, inferred, "numeric_shift_keep_unit")
        if inferred in ENTITY_TYPES:
            replacement = self.gen_entity_component(token, inferred)
            return _replacement(token, replacement, inferred, "same_type_pool_component")
        return _replacement(token, None, inferred or "UNKNOWN", "no_rule")

    def replacement_for_unit(self, unit: dict, context: str = "") -> dict:
        """Generate a typed replacement using contextual entity-slot metadata."""

        token = str(unit.get("text", ""))
        unit_type = str(unit.get("type", ""))
        inferred = self.infer_type(token, unit_type, context)
        if inferred in ENTITY_TYPES and unit.get("entity_token_count"):
            replacement = self.gen_entity_slot(
                token,
                inferred,
                int(unit.get("entity_token_index", 0)),
                int(unit.get("entity_token_count", 1)),
                entity_text=str(unit.get("entity_text", "")),
            )
            return _replacement(token, replacement, inferred, "same_type_entity_slot")
        return self.replacement_for_token(token, unit_type, context)

    def infer_type(self, token: str, unit_type: str = "", context: str = "") -> str:
        lowered = token.lower()
        explicit_type = unit_type.upper()
        if lowered in RELATION_TOKEN_CF:
            return "RELATION_CUE"
        if YEAR_RE.fullmatch(token):
            return "DATE"
        if unit_type == "MONEY" or token in {"$", "€", "£", "¥"}:
            return "MONEY"
        if NUM_RE.fullmatch(token) or ORD_RE.fullmatch(token):
            return "CARDINAL"
        if explicit_type in ONTO_TYPES:
            return explicit_type
        # Pool membership is not type evidence.  Common words such as "plays"
        # and "boy" can occur inside stored entity strings; inferring an entity
        # type from that coincidence produces invalid replacements.
        return ""

    def gen_entity_slot(
        self,
        token: str,
        ner_type: str,
        slot: int,
        token_count: int,
        *,
        entity_text: str = "",
    ) -> str | None:
        """Replace an entity component with the same slot of a same-type entity."""

        original_parts = WORD_RE.findall(entity_text)
        candidates: list[list[str]] = []
        for value in self.type_pool.get(ner_type, []):
            parts = WORD_RE.findall(str(value))
            if len(parts) != token_count or slot < 0 or slot >= len(parts):
                continue
            if len(original_parts) == token_count:
                if any(
                    candidate.lower() == original.lower()
                    for candidate, original in zip(parts, original_parts)
                ):
                    continue
            elif parts[slot].lower() == token.lower():
                continue
            if any(part.lower() in STOPWORDS for part in parts):
                continue
            candidates.append(parts)
        if not candidates:
            return None
        identity = entity_text or token
        replacement = candidates[
            seed(identity + ner_type + str(token_count)) % len(candidates)
        ][slot]
        return preserve_case(token, replacement)

    def gen_entity_component(self, token: str, ner_type: str) -> str | None:
        candidates = [
            value for value in self.component_pool.get(ner_type, [])
            if value.lower() != token.lower()
            and value.lower() not in STOPWORDS
            and len(value) > 1
            and bool(WORD_RE.search(value))
        ]
        if not candidates:
            values = [
                value for value in self.type_pool.get(ner_type, [])
                if value.lower() != token.lower() and len(WORD_RE.findall(value)) == 1
            ]
            candidates = values
        if not candidates:
            return None
        replacement = candidates[seed(token + ner_type) % len(candidates)]
        if " " in replacement:
            parts = [part for part in WORD_RE.findall(replacement) if part.lower() not in STOPWORDS]
            if parts:
                replacement = parts[seed(token) % len(parts)]
        return preserve_case(token, replacement)

    @staticmethod
    def _index_components(type_pool: dict) -> tuple[dict[str, Counter], dict[str, list[str]]]:
        component_types: dict[str, Counter] = defaultdict(Counter)
        component_pool: dict[str, list[str]] = defaultdict(list)
        seen_by_type: dict[str, set[str]] = defaultdict(set)
        for ner_type, values in type_pool.items():
            ner_type = str(ner_type).upper()
            if ner_type not in ENTITY_TYPES:
                continue
            for value in values:
                for part in WORD_RE.findall(str(value)):
                    key = part.lower()
                    if key in STOPWORDS or len(key) <= 1:
                        continue
                    component_types[key][ner_type] += 1
                    if key not in seen_by_type[ner_type]:
                        seen_by_type[ner_type].add(key)
                        component_pool[ner_type].append(part)
        return dict(component_types), dict(component_pool)


def _replacement(old: str, new: str | None, inferred_type: str, policy: str) -> dict:
    ok = bool(new and new.strip() and new.lower() != old.lower())
    return {
        "ok": ok,
        "old": old,
        "new": new if ok else "",
        "inferred_type": inferred_type,
        "policy": policy,
    }


def seed(value: str) -> int:
    return int(hashlib.sha1((value or "x").encode("utf-8")).hexdigest()[:8], 16)


def preserve_case(old: str, new: str) -> str:
    if old.isupper():
        return new.upper()
    if old[:1].isupper():
        return new[:1].upper() + new[1:]
    return new[:1].lower() + new[1:]


def gen_time(value: str) -> str | None:
    match = YEAR_RE.search(value or "")
    if not match:
        return None
    year = int(match.group(1))
    offset = 3 + seed(value) % 12
    new_year = year - offset if year - offset > 1000 else year + offset
    return value[:match.start()] + str(new_year) + value[match.end():]


def gen_number(value: str) -> str | None:
    text = (value or "").strip()
    ordinal = ORD_RE.match(text)
    if ordinal:
        number = int(ordinal.group(1))
        shifted = number + 2 + seed(text) % 6
        return f"{shifted}{ordinal_suffix(shifted)}"
    match = NUM_RE.search(text)
    if not match:
        return None
    raw = match.group(0).replace(",", "")
    try:
        number = float(raw)
    except ValueError:
        return None
    factor = 1.3 + (seed(text) % 40) / 100.0
    shifted = number * factor
    new_value = str(int(round(shifted))) if "." not in raw else str(round(shifted, 2))
    if "," in match.group(0):
        new_value = f"{int(float(new_value)):,}"
    return text[:match.start()] + new_value + text[match.end():]


def ordinal_suffix(number: int) -> str:
    if 11 <= number % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
