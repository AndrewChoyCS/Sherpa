"""Tests for subset selection and head-to-head diversity comparison.

The fixture plants a structure where the right answer is known: a handful of distinct
behaviour groups, each containing many near-identical copies. A selection strategy that
understands diversity must pick one from each group; one that does not will load up on
duplicates. Any metric worth shipping has to separate those two outcomes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.compare import (  # noqa: E402
    METRIC_ORDER,
    SELECTION_METHODS,
    compare_subsets,
    random_baseline,
    redundancy_threshold,
    score_subset,
    select_indices,
    selection_curve,
)


# Deliberately imbalanced group sizes: one dominant behaviour with 40 near-identical
# takes, down to a tail with 2. Real teleop datasets look like this -- a few tasks are
# recorded over and over -- and the imbalance is what selection strategy has to
# overcome. With *equal* group sizes the problem is trivially easy: any random pick
# covers most groups, random scores near the ceiling, and no strategy can separate.
GROUP_SIZES = (40, 15, 8, 5, 4, 3, 3, 2)


@pytest.fixture
def planted():
    """8 well-separated groups of near-identical members, heavily imbalanced (N=80).

    Returns ``(distance_matrix, labels, tasks)``.
    """
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=10.0, size=(len(GROUP_SIZES), 6))
    points, labels = [], []
    for g, (center, count) in enumerate(zip(centers, GROUP_SIZES)):
        points.append(center + rng.normal(scale=0.01, size=(count, 6)))
        labels.extend([g] * count)
    pts = np.vstack(points)
    D = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    np.fill_diagonal(D, 0.0)
    tasks = [f"task_{g}" for g in labels]
    return D, np.array(labels), tasks


class TestSelection:
    @pytest.mark.parametrize("method", SELECTION_METHODS)
    def test_returns_n_unique_valid_indices(self, planted, method):
        D, labels, _ = planted
        idx = select_indices(method, 12, D, labels=labels)
        assert idx.size == 12
        assert len(set(idx.tolist())) == 12
        assert idx.min() >= 0 and idx.max() < D.shape[0]

    def test_size_is_clamped(self, planted):
        D, labels, _ = planted
        assert select_indices("coreset", 10_000, D, labels=labels).size == D.shape[0]
        assert select_indices("coreset", 0, D, labels=labels).size == 1

    def test_coreset_covers_every_group(self, planted):
        """Farthest-point selection should hit all 8 groups before repeating one."""
        D, labels, _ = planted
        idx = select_indices("coreset", 8, D, labels=labels)
        assert len(set(labels[idx].tolist())) == 8

    def test_stratified_covers_every_group(self, planted):
        D, labels, _ = planted
        idx = select_indices("stratified", 8, D, labels=labels)
        assert len(set(labels[idx].tolist())) == 8

    def test_stratified_requires_labels(self, planted):
        D, _, _ = planted
        with pytest.raises(ValueError, match="labels"):
            select_indices("stratified", 5, D, labels=None)

    def test_redundant_control_collapses_onto_few_groups(self, planted):
        """The adversarial baseline must behave badly, or it proves nothing."""
        D, labels, _ = planted
        idx = select_indices("redundant", 8, D, labels=labels)
        assert len(set(labels[idx].tolist())) < 8

    def test_random_is_seed_deterministic(self, planted):
        D, labels, _ = planted
        a = select_indices("random", 15, D, labels=labels, seed=7)
        b = select_indices("random", 15, D, labels=labels, seed=7)
        c = select_indices("random", 15, D, labels=labels, seed=8)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_unknown_method_rejected(self, planted):
        D, labels, _ = planted
        with pytest.raises(ValueError, match="method must be"):
            select_indices("nonsense", 5, D, labels=labels)


class TestScoring:
    def test_scores_the_submatrix(self, planted):
        D, _, _ = planted
        idx = [0, 1, 2, 30, 31]
        sub = D[np.ix_(idx, idx)]
        expected = sub[np.triu_indices(len(idx), k=1)].mean()
        assert score_subset(D, idx)["diversity_score"] == pytest.approx(expected)

    def test_coverage_counts(self, planted):
        D, labels, tasks = planted
        idx = select_indices("coreset", 8, D, labels=labels)
        metrics = score_subset(D, idx, tasks=tasks, labels=labels)
        assert metrics["n_tasks_covered"] == 8
        assert metrics["n_clusters_covered"] == 8

    def test_shared_threshold_makes_redundancy_comparable(self, planted):
        """Regression: a per-subset threshold inverted the ranking.

        Taking each subset's own 5th percentile rescales to whatever spread that subset
        has, so a perfectly spread selection still flags ~5% of itself as redundant and
        can score *worse* than a duplicate-heavy one. On real data that produced coreset
        0.88 vs random 0.50 -- backwards. One absolute cutoff from the full dataset fixes it.
        """
        D, labels, _ = planted
        cutoff = redundancy_threshold(D)
        spread = select_indices("coreset", 8, D, labels=labels)
        duplicates = select_indices("redundant", 8, D, labels=labels)

        shared_spread = score_subset(D, spread, threshold=cutoff)["redundancy_ratio"]
        shared_dupes = score_subset(D, duplicates, threshold=cutoff)["redundancy_ratio"]
        assert shared_spread == 0.0
        assert shared_dupes == 1.0

        # Without the shared cutoff the ordering is not trustworthy: the spread subset
        # reports non-zero redundancy purely because it has a bottom 5% of its own.
        assert score_subset(D, spread)["redundancy_ratio"] > 0.0

    def test_redundancy_threshold_is_positive(self, planted):
        D, _, _ = planted
        assert redundancy_threshold(D) > 0

    def test_single_episode_subset_is_safe(self, planted):
        D, _, _ = planted
        metrics = score_subset(D, [3])
        assert metrics["diversity_score"] == 0.0
        assert metrics["redundancy_ratio"] == 0.0


class TestComparison:
    def test_coreset_beats_random_and_redundant(self, planted):
        D, labels, tasks = planted
        result = compare_subsets(
            D, methods=("coreset", "random", "redundant"), subset_size=24,
            labels=labels, tasks=tasks, baseline_trials=100,
        )
        by_name = {s.name: s for s in result.subsets}
        assert by_name["coreset"].metrics["diversity_score"] > by_name["random"].metrics["diversity_score"]
        assert by_name["random"].metrics["diversity_score"] > by_name["redundant"].metrics["diversity_score"]
        assert by_name["coreset"].metrics["redundancy_ratio"] < by_name["redundant"].metrics["redundancy_ratio"]

    def test_subsets_are_equal_size(self, planted):
        """Unequal sizes would measure the size gap rather than the strategy."""
        D, labels, _ = planted
        result = compare_subsets(D, methods=("coreset", "random"), subset_size=20, labels=labels)
        assert {s.size for s in result.subsets} == {20}
        assert result.subset_size == 20

    def test_baseline_places_coreset_high(self, planted):
        D, labels, _ = planted
        result = compare_subsets(
            D, methods=("coreset", "random"), subset_size=24, labels=labels,
            baseline_trials=200,
        )
        assert result.baseline is not None
        assert result.baseline["percentile"] > 95.0
        assert result.baseline["z_score"] > 2.0

    def test_null_model_catches_a_lucky_random_draw(self, planted):
        """Why the baseline exists: one random draw can beat a principled selection.

        At a small budget a single lucky sample outscores the coreset, yet the coreset
        still sits near the top of the random *distribution*. Reporting only the
        head-to-head number would invert the conclusion here.
        """
        D, labels, _ = planted
        result = compare_subsets(
            D, methods=("coreset", "random"), subset_size=8, labels=labels,
            baseline_trials=300,
        )
        coreset, lucky_random = result.subsets
        assert lucky_random.metrics["diversity_score"] > coreset.metrics["diversity_score"]
        assert result.baseline["percentile"] > 95.0

    def test_deltas_pick_the_right_winner(self, planted):
        D, labels, tasks = planted
        result = compare_subsets(
            D, methods=("coreset", "random"), subset_size=16, labels=labels, tasks=tasks
        )
        deltas = result.deltas().set_index("metric")
        assert deltas.loc["diversity_score", "better"] == "coreset"
        # redundancy_ratio is lower-is-better; the winner logic must invert for it.
        assert deltas.loc["redundancy_ratio", "better"] in {"coreset", "tie"}

    def test_table_has_expected_columns(self, planted):
        D, labels, tasks = planted
        result = compare_subsets(
            D, methods=("coreset", "random"), subset_size=10, labels=labels, tasks=tasks
        )
        table = result.table()
        assert len(table) == 2
        for column in METRIC_ORDER:
            assert column in table.columns

    def test_baseline_can_be_disabled(self, planted):
        D, labels, _ = planted
        result = compare_subsets(
            D, methods=("coreset", "random"), subset_size=10, labels=labels,
            baseline_trials=0,
        )
        assert result.baseline is None

    def test_random_baseline_shape_and_variance(self, planted):
        D, _, _ = planted
        samples = random_baseline(D, 12, trials=50, seed=1)
        assert samples.shape == (50,)
        assert np.isfinite(samples).all()
        assert samples.std() > 0


class TestSelectionCurve:
    def test_coreset_dominates_across_budgets(self, planted):
        """The ranking must hold across budgets, not just at one lucky subset size.

        Asserted from n=12 upward: at the very smallest budgets a single random draw is
        high-variance and can win outright, which is what the null model is for.
        """
        D, labels, _ = planted
        curve = selection_curve(
            D, methods=("coreset", "random", "redundant"), sizes=(8, 12, 16, 24),
            labels=labels,
        )
        pivot = curve.pivot(index="subset_size", columns="method", values="diversity_score")
        assert (pivot["coreset"] >= pivot["random"]).loc[12:].all()
        # The adversarial control must lose everywhere, at every budget.
        assert (pivot["coreset"] > pivot["redundant"]).all()
        assert (pivot["random"] > pivot["redundant"]).all()

    def test_strategies_converge_as_budget_approaches_the_dataset(self, planted):
        """Selection only matters when the budget is a fraction of the data.

        Picking 90% of the episodes forces every strategy to include almost everything,
        so the scores collapse together. Worth pinning down because it bounds the claim:
        the coreset advantage is a small-budget result, not a universal one.
        """
        D, labels, _ = planted
        total = D.shape[0]
        curve = selection_curve(
            D, methods=("coreset", "redundant"), sizes=(int(total * 0.9),), labels=labels
        )
        scores = curve["diversity_score"].to_numpy()
        assert abs(scores[0] - scores[1]) / max(scores) < 0.25

    def test_curve_covers_requested_sizes(self, planted):
        D, labels, _ = planted
        curve = selection_curve(D, methods=("coreset",), sizes=(5, 10), labels=labels)
        assert sorted(curve["subset_size"].unique().tolist()) == [5, 10]

    def test_default_sizes_stay_within_dataset(self, planted):
        D, labels, _ = planted
        curve = selection_curve(D, methods=("coreset",), labels=labels)
        assert curve["subset_size"].max() < D.shape[0]


class TestRealData:
    """End-to-end on the fetched dataset, skipped when it is absent."""

    @pytest.mark.skipif(
        not (REPO_ROOT / "data").exists() or not any((REPO_ROOT / "data").glob("*.zarr")),
        reason="real EgoVerse episodes not fetched",
    )
    def test_coreset_beats_random_on_real_episodes(self):
        from src.pipeline import run_pipeline

        result = run_pipeline(str(REPO_ROOT / "data"), verbose=False)
        if result.n_episodes < 20:
            pytest.skip("too few episodes for a meaningful subset comparison")
        comparison = compare_subsets(
            result.distance_matrix,
            methods=("coreset", "random"),
            subset_size=result.n_episodes // 4,
            labels=result.labels,
            tasks=result.dataset.task_labels,
            baseline_trials=100,
        )
        coreset, random_pick = comparison.subsets
        assert coreset.metrics["diversity_score"] > random_pick.metrics["diversity_score"]
        assert coreset.metrics["mean_nn_distance"] > random_pick.metrics["mean_nn_distance"]
        assert comparison.baseline["percentile"] > 90.0
