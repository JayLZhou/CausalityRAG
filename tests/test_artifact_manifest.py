import hashlib
import tempfile

from scripts.build_artifact_manifest import artifact_fingerprint


def test_artifact_fingerprint_records_hash_size_and_lines() -> None:
    payload = b"one\ntwo\n"
    with tempfile.NamedTemporaryFile() as artifact:
        artifact.write(payload)
        artifact.flush()
        result = artifact_fingerprint(artifact.name)

    assert result == {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "lines": 2,
    }
