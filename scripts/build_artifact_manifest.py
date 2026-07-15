"""Hash frozen experiment artifacts and bind them to code/config metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metadata-json", default="{}")
    parser.add_argument("--repository", default=".")
    args = parser.parse_args()

    metadata = json.loads(args.metadata_json)
    if not isinstance(metadata, dict):
        raise ValueError("metadata-json must decode to an object")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code": repository_state(args.repository),
        "metadata": metadata,
        "artifacts": {
            path: artifact_fingerprint(path) for path in args.artifacts
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")


def artifact_fingerprint(path: str) -> dict:
    digest = hashlib.sha256()
    lines = 0
    size = 0
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
            lines += block.count(b"\n")
    return {
        "sha256": digest.hexdigest(),
        "bytes": size,
        "lines": lines,
    }


def repository_state(path: str) -> dict:
    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ("git", "-C", path, *arguments),
            text=True,
        ).strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


if __name__ == "__main__":
    main()
