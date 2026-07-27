"""Persistent, query-keyed cache for on-demand counterfactual replacements."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import Lock


class ReplacementPool:
    """Reuse validated replacements and atomically persist newly resolved ones."""

    def __init__(self, path: str) -> None:
        if not path:
            raise ValueError("replacement pool path must be nonempty")
        self.path = Path(path)
        self._lock = Lock()
        self._rows, self._order = _load_rows(self.path)

    def cache_for(self, identifier: str) -> dict[str, dict]:
        key = str(identifier)
        row = self._rows.setdefault(key, {"id": key, "replacements": {}})
        if key not in self._order:
            self._order.append(key)
        replacements = row.setdefault("replacements", {})
        if not isinstance(replacements, dict):
            replacements = {}
            row["replacements"] = replacements
        return replacements

    def persist(
        self,
        identifier: str,
        replacements: dict[str, dict],
        *,
        source: str,
    ) -> None:
        """Merge one query cache into the latest file and replace it atomically."""

        key = str(identifier)
        with self._lock, _file_lock(self.path):
            disk_rows, disk_order = _load_rows(self.path)
            row = dict(disk_rows.get(key, self._rows.get(key, {"id": key})))
            merged = dict(row.get("replacements", {}))
            merged.update(
                {
                    str(unit_id): dict(replacement)
                    for unit_id, replacement in replacements.items()
                    if isinstance(replacement, dict)
                }
            )
            row.update(
                {
                    "id": key,
                    "replacements": merged,
                    "valid_tokens": sum(
                        bool(value.get("ok")) for value in merged.values()
                    ),
                    "invalid_tokens": sum(
                        not bool(value.get("ok")) for value in merged.values()
                    ),
                    "generation_mode": "on_demand_after_selection",
                    "last_source": str(source),
                }
            )
            disk_rows[key] = row
            if key not in disk_order:
                disk_order.append(key)
            _atomic_write(self.path, disk_rows, disk_order)
            self._rows = disk_rows
            self._order = disk_order


def _load_rows(path: Path) -> tuple[dict[str, dict], list[str]]:
    rows: dict[str, dict] = {}
    order: list[str] = []
    if not path.exists():
        return rows, order
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = str(row.get("id", ""))
            if not identifier:
                continue
            if identifier not in rows:
                order.append(identifier)
            rows[identifier] = row
    return rows, order


def _atomic_write(path: Path, rows: dict[str, dict], order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for identifier in order:
                row = rows.get(identifier)
                if row is not None:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _file_lock(path: Path):
    import fcntl

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
