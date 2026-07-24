import numpy as np

from exp.analyze_beta_sweep import (
    best_beta_index,
    facility_assignments,
    facility_cluster_weights,
    facility_location_greedy,
    metrics_by_beta,
    stratified_random_sample,
)


def test_facility_location_covers_two_clusters() -> None:
    similarity = np.asarray([
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.9],
        [0.1, 0.1, 0.9, 1.0],
    ])

    selected = facility_location_greedy(similarity, 2)

    assert len(set(selected) & {0, 1}) == 1
    assert len(set(selected) & {2, 3}) == 1
    assert facility_cluster_weights(similarity, selected).sum() == 4


def test_best_beta_maximizes_overall_flip_then_uses_fewer_edits() -> None:
    flips = np.asarray([
        [1, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
    ], dtype=float)
    coverage = np.ones_like(flips)
    edits = np.asarray([
        [3, 2, 1],
        [3, 2, 1],
        [3, 2, 1],
    ], dtype=float)

    metrics = metrics_by_beta(flips, coverage, edits)

    assert best_beta_index(metrics) == 1


def test_stratified_random_sample_preserves_population_weight() -> None:
    similarity = np.asarray([
        [1.0, 0.1],
        [0.9, 0.2],
        [0.8, 0.3],
        [0.1, 1.0],
    ])
    assignments = facility_assignments(similarity, [0, 1])

    selected, weights = stratified_random_sample(
        assignments,
        size=3,
        rng=np.random.default_rng(7),
    )

    assert len(selected) == 3
    assert len(set(assignments[selected])) == 2
    assert weights.sum() == 4
