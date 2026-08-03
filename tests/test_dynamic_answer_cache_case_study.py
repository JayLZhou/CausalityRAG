from scripts.run_dynamic_answer_cache_case_study import (
    PolicySpec,
    make_request_trace,
    replay_policy,
)


def _replay(spec: PolicySpec) -> dict:
    return replay_policy(
        spec=spec,
        capacity=1,
        requests=["q", "q"],
        updates={1: "q"},
        update_rows={
            "q": {
                "answer_changed": True,
                "selected_ids": ["u"],
            }
        },
        sentence_by_query={"q": {"u": "s"}},
        token_signatures={"q": ["u"]},
        sentence_signatures={"q": ["s"]},
    )


def test_lru_serves_a_resident_stale_answer() -> None:
    result = _replay(PolicySpec(name="lru"))

    assert result["resident_stale_updates"] == 1
    assert result["stale_hits"] == 1
    assert result["stale_episode_recall"] == 0


def test_reflow_token_signature_invalidates_before_reuse() -> None:
    result = _replay(
        PolicySpec(name="reflow", invalidation="token", budget=1)
    )

    assert result["update_invalidations"] == 1
    assert result["stale_hits"] == 0
    assert result["stale_episode_recall"] == 1


def test_second_version_toggle_can_refresh_an_old_cached_answer() -> None:
    result = replay_policy(
        spec=PolicySpec(name="lru"),
        capacity=1,
        requests=["q", "q", "q"],
        updates={1: "q", 2: "q"},
        update_rows={
            "q": {
                "answer_changed": True,
                "selected_ids": ["u"],
            }
        },
        sentence_by_query={"q": {"u": "s"}},
        token_signatures={"q": ["u"]},
        sentence_signatures={"q": ["s"]},
    )

    assert result["resident_stale_updates"] == 1
    assert result["stale_hits"] == 1


def test_trace_places_updates_between_repeated_requests() -> None:
    requests, updates = make_request_trace(
        ["q1", "q2"],
        n_requests=100,
        zipf_alpha=1.0,
        updates_per_query=3,
        seed=0,
    )

    for position, query_id in updates.items():
        occurrences = [i for i, value in enumerate(requests) if value == query_id]
        assert occurrences[0] < position <= occurrences[-1]

    assert len(updates) == 6
