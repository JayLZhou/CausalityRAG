"""Client helpers for the isolated spaCy annotation service."""

from __future__ import annotations

import json
import os
import urllib.request


class SpacyAnnotationClient:
    def __init__(self, base_url: str | None = None, timeout: int = 60) -> None:
        self.base_url = (
            base_url
            or os.environ.get("CAUSALITYRAG_SPACY_BASE_URL")
            or "http://127.0.0.1:8021"
        ).rstrip("/")
        self.timeout = timeout

    def annotate(self, text: str) -> dict:
        return self._post("/annotate", {"text": text})

    def validate(self, unit: dict, context: str, replacement: dict) -> dict:
        return self._post(
            "/validate",
            {"unit": unit, "context": context, "replacement": replacement},
        )

    def health(self) -> dict:
        with urllib.request.urlopen(self.base_url + "/health", timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
