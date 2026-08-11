import json

import pytest

from scripts import run_seven_dataset_pools as runner
from scripts.run_seven_dataset_pools import artifact_complete


def test_artifact_complete_requires_rows_and_matching_summary(tmp_path):
    rows = tmp_path / "rows.jsonl"
    summary = tmp_path / "summary.json"
    rows.write_text('{"id":"1"}\n{"id":"2"}\n', encoding="utf-8")
    summary.write_text(
        json.dumps({"queries": 2, "retrieval_top_k": 10}),
        encoding="utf-8",
    )

    assert artifact_complete(
        rows,
        summary,
        expected_rows=2,
        top_k_field="retrieval_top_k",
    )
    assert not artifact_complete(
        rows,
        summary,
        expected_rows=3,
        top_k_field="retrieval_top_k",
    )


def test_artifact_complete_rejects_stale_summary(tmp_path):
    rows = tmp_path / "rows.jsonl"
    summary = tmp_path / "summary.json"
    rows.write_text('{"id":"1"}\n', encoding="utf-8")
    summary.write_text(
        json.dumps({"queries": 1, "top_k": 5}),
        encoding="utf-8",
    )

    assert not artifact_complete(
        rows,
        summary,
        expected_rows=1,
        top_k_field="top_k",
    )


def test_generation_tail_can_be_returned_for_common_exclusion(
    monkeypatch, tmp_path
):
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()

    def fake_run(command, *, cwd, attempts=3):
        (pool_dir / "generation.json").write_text(json.dumps({
            "typed_keys_covered": 8,
            "typed_keys_total": 10,
            "unresolved": 2,
        }))

    monkeypatch.setattr(runner, "run", fake_run)

    unresolved = runner.generate_until_complete(
        python="python",
        repository=tmp_path,
        typed_keys=tmp_path / "typed.jsonl",
        seed=tmp_path / "seed.jsonl",
        pool_dir=pool_dir,
        max_passes=1,
        allow_exclusions=True,
    )

    assert unresolved == 2


def test_generation_tail_still_fails_closed_by_default(monkeypatch, tmp_path):
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()

    def fake_run(command, *, cwd, attempts=3):
        (pool_dir / "generation.json").write_text(json.dumps({
            "typed_keys_covered": 8,
            "typed_keys_total": 10,
            "unresolved": 2,
        }))

    monkeypatch.setattr(runner, "run", fake_run)

    with pytest.raises(RuntimeError, match="remains unresolved"):
        runner.generate_until_complete(
            python="python",
            repository=tmp_path,
            typed_keys=tmp_path / "typed.jsonl",
            seed=tmp_path / "seed.jsonl",
            pool_dir=pool_dir,
            max_passes=1,
        )


def test_freeze_accepts_a_complete_partition_with_common_exclusions(
    monkeypatch, tmp_path
):
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    commands = []

    def fake_run(command, *, cwd, attempts=3):
        commands.append(command)
        (pool_dir / "shared_pool.manifest.json").write_text(json.dumps({
            "positions": 10,
            "eligible_positions": 8,
            "excluded_positions": 2,
            "unresolved_typed_positions": 2,
            "coverage": 1.0,
        }))

    monkeypatch.setattr(runner, "run", fake_run)

    runner.freeze_pool(
        python="python",
        repository=tmp_path,
        pool_dir=pool_dir,
        exclude_unresolved=True,
    )

    assert "--exclude-unresolved" in commands[0]
