"""Serve spaCy NER/POS annotation to GPU environments over localhost."""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spacy

from causalityrag.replacement import validate_contextual_replacement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--model", default=os.environ.get("YVETTE_SPACY_MODEL", "en_core_web_lg"))
    args = parser.parse_args()
    nlp = spacy.load(args.model)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            self._json({"ok": True, "model": args.model})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/annotate":
                self._json(annotate(nlp, str(payload.get("text", ""))))
                return
            if self.path == "/validate":
                self._json(validate_contextual_replacement(
                    dict(payload.get("unit", {})),
                    str(payload.get("context", "")),
                    dict(payload.get("replacement", {})),
                    nlp,
                ))
                return
            self.send_error(404)

        def log_message(self, format: str, *values) -> None:
            return

        def _json(self, payload: dict) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"spaCy annotation server listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


def annotate(nlp, text: str) -> dict:
    doc = nlp(text)
    return {
        "tokens": [
            {
                "text": token.text,
                "start": token.idx,
                "end": token.idx + len(token),
                "pos": token.pos_,
                "tag": token.tag_,
                "lemma": token.lemma_,
                "morph": token.morph.to_dict(),
            }
            for token in doc
        ],
        "entities": [
            {
                "text": entity.text,
                "start": entity.start_char,
                "end": entity.end_char,
                "label": entity.label_.upper(),
                "tokens": [
                    {"start": token.idx, "end": token.idx + len(token)}
                    for token in entity if not token.is_space
                ],
            }
            for entity in doc.ents
        ],
    }


if __name__ == "__main__":
    main()
