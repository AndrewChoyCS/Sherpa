"""Subset selection and head-to-head diversity comparison.

This is what makes the diversity score *actionable* rather than merely descriptive.
Reporting "this dataset scores 0.183" invites the question "compared to what?". The
answer here is a ranking: build two subsets of **equal size** by different selection
strategies, score each, and show the metric separates them.

Two design points that make the comparison honest:

**Subsets must be the same size.** Several of the metrics move with N -- a larger
subset has more chances to contain a close pair, so mean nearest-neighbour distance
falls and redundancy rises. Comparing a 50-episode selection against a 200-episode one
measures the size difference, not the strategy. Sizes are equalised by default.

**A single random draw is weak evidence.** Random selection has real variance, so
beating one random sample proves little. :func:`random_baseline` resamples many times
and reports where the candidate falls in that distribution, which turns "coreset looks
better" into "coreset scores above the 99th percentile of 200 random draws".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .cluster_mapper import calculate_diversity_score, nearest_neighbor_distances, pairwise_values
from .curriculum import coreset_order

SELECTION_METHODS = ("coreset", "random", "stratified", "sequential", "redundant")

# Metrics reported per subset, in display order.
METRIC_ORDER = (
    "diversity_score",
    "median_pairwise",
    "min_pairwise",
    "mean_nn_distance",
    "redundancy_ratio",
    "n_tasks_covered",
    "n_sources_covered",
    "n_clusters_covered",
)
# Metrics where lower is better; everything else is higher-is-better.
LOWER_IS_BETTER = frozenset({"redundancy_ratio"})


@dataclass
class SubsetScore:
    """One selected subset and its diversity metrics."""

    name: str
    method: str
    indices: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(len(self.indices))


@dataclass
class ComparisonResult:
    """Head-to-head result for two or more subsets of equal size."""

    subsets: List[SubsetScore]
    subset_size: int
    baseline: Optional[Dict[str, float]] = None
    baseline_samples: Optional[np.ndarray] = None

    def table(self) -> pd.DataFrame:
        """Metrics as one row per subset, plus a delta row when exactly two."""
        rows = []
        for subset in self.subsets:
            row = {"subset": subset.name, "method": subset.method, "n": subset.size}
            row.update({k: subset.metrics.get(k) for k in METRIC_ORDER})
            rows.append(row)
        return pd.DataFrame(rows)

    def deltas(self) -> pd.DataFrame:
        """Per-metric comparison of the first two subsets, with a winner column."""
        if len(self.subsets) < 2:
            return pd.DataFrame()
        a, b = self.subsets[0], self.subsets[1]
        rows = []
        for key in METRIC_ORDER:
            va, vb = a.metrics.get(key), b.metrics.get(key)
            if va is None or vb is None:
                continue
            delta = va - vb
            pct = (delta / abs(vb) * 100.0) if vb not in (0, None) else float("nan")
            if abs(delta) < 1e-12:
                winner = "tie"
            elif (delta > 0) != (key in LOWER_IS_BETTER):
                winner = a.name
            else:
                winner = b.name
            rows.append(
                {
                    "metric": key,
                    a.name: va,
                    b.name: vb,
                    "delta": delta,
                    "pct_change": pct,
                    "better": winner,
                }
            )
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# selection strategies
# --------------------------------------------------------------------------- #
def select_indices(
    method: str,
    n: int,
    distance_matrix: np.ndarray,
    labels: Optional[Sequence[int]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Choose ``n`` episode indices by the named strategy.

    Args:
        method: One of :data:`SELECTION_METHODS`.

            ``coreset``
                Farthest-point traversal prefix -- the maximum-coverage selection.
            ``random``
                Uniform sample without replacement; the honest baseline.
            ``stratified``
                Round-robin across clusters, so every motion family is represented.
            ``sequential``
                The first ``n`` in dataset order, which in practice means whichever
                sources sort first -- the "just take the first N episodes" strategy.
            ``redundant``
                The ``n`` episodes with the *smallest* nearest-neighbour distance, i.e.
                deliberately duplicate-heavy. Included as an adversarial control: a
                metric that cannot separate this from ``coreset`` is not measuring
                diversity.
        n: Subset size, clamped to ``[1, N]``.
        distance_matrix: ``(N, N)`` DTW distances.
        labels: Cluster labels, required by ``stratified``.
        seed: Seed for ``random``.

    Returns:
        Sorted array of selected indices.
    """
    if method not in SELECTION_METHODS:
        raise ValueError(f"method must be one of {SELECTION_METHODS}, got {method!r}")

    total = int(distance_matrix.shape[0])
    n = int(np.clip(n, 1, total))

    if method == "coreset":
        chosen = coreset_order(distance_matrix)[:n]
    elif method == "random":
        chosen = np.random.default_rng(seed).choice(total, size=n, replace=False).tolist()
    elif method == "sequential":
        chosen = list(range(n))
    elif method == "redundant":
        # Ascending nearest-neighbour distance: the most-duplicated episodes first.
        chosen = np.argsort(nearest_neighbor_distances(distance_matrix))[:n].tolist()
    else:  # stratified
        if labels is None:
            raise ValueError("stratified selection requires cluster labels")
        labels_arr = np.asarray(labels)
        groups = [list(np.flatnonzero(labels_arr == g)) for g in np.unique(labels_arr)]
        # Within each cluster, take the most central episodes first so a stratified
        # pick is representative rather than a random walk through outliers.
        for group in groups:
            group.sort(key=lambda i: distance_matrix[i, group].sum())
        chosen, cursor = [], 0
        while len(chosen) < n:
            progressed = False
            for group in groups:
                if cursor < len(group):
                    chosen.append(group[cursor])
                    progressed = True
                    if len(chosen) == n:
                        break
            if not progressed:
                break
            cursor += 1

    return np.array(sorted(int(i) for i in chosen), dtype=int)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def redundancy_threshold(distance_matrix: np.ndarray, quantile: float = 0.05) -> float:
    """Absolute near-duplicate distance cutoff, derived from the *whole* dataset.

    This must be computed once globally and shared by every subset being compared. The
    tempting alternative -- taking each subset's own 5th percentile -- is not
    comparable across subsets, because it rescales to whatever spread that subset
    happens to have. A perfectly spread selection still has a bottom 5%, so it would
    report the same redundancy as a pile of duplicates, and in practice the ranking
    inverted: a coreset scored 0.88 against random's 0.50.
    """
    values = pairwise_values(distance_matrix)
    return float(np.quantile(values, quantile)) if values.size else 0.0


def score_subset(
    distance_matrix: np.ndarray,
    indices: Sequence[int],
    tasks: Optional[Sequence[str]] = None,
    sources: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[int]] = None,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Diversity metrics for one subset, computed on its own submatrix.

    Args:
        threshold: Absolute near-duplicate cutoff from :func:`redundancy_threshold`.
            Pass the *same* value for every subset in a comparison. Defaults to the
            subset's own 5th percentile, which is only meaningful in isolation.

    Coverage counts (distinct tasks/sources/clusters represented) are included because
    they are the most legible evidence that a selection is broad: "40 episodes covering
    14 of 14 tasks" lands faster than a distance in metres.
    """
    idx = np.asarray(list(indices), dtype=int)
    sub = distance_matrix[np.ix_(idx, idx)]
    values = pairwise_values(sub)

    metrics: Dict[str, float] = {
        "n": float(idx.size),
        "diversity_score": calculate_diversity_score(sub),
        "median_pairwise": float(np.median(values)) if values.size else 0.0,
        "min_pairwise": float(values.min()) if values.size else 0.0,
        "max_pairwise": float(values.max()) if values.size else 0.0,
    }
    nn = nearest_neighbor_distances(sub)
    metrics["mean_nn_distance"] = float(nn.mean()) if nn.size else 0.0
    if values.size:
        cutoff = threshold if threshold is not None else float(np.quantile(values, 0.05))
        metrics["redundancy_ratio"] = float((nn <= cutoff).mean())
    else:
        metrics["redundancy_ratio"] = 0.0

    for key, field_values in (
        ("n_tasks_covered", tasks),
        ("n_sources_covered", sources),
        ("n_clusters_covered", labels),
    ):
        if field_values is not None:
            arr = np.asarray(list(field_values))
            metrics[key] = float(len(set(arr[idx].tolist())))
    return metrics


def random_baseline(
    distance_matrix: np.ndarray,
    n: int,
    trials: int = 200,
    seed: int = 0,
    metric: str = "diversity_score",
) -> np.ndarray:
    """Distribution of a metric over ``trials`` independent random subsets of size ``n``.

    Gives the comparison a null model: without it, a single random draw could beat a
    principled selection by luck and nobody would know.
    """
    total = int(distance_matrix.shape[0])
    n = int(np.clip(n, 2, total))
    rng = np.random.default_rng(seed)
    out = np.empty(trials, dtype=np.float64)
    for t in range(trials):
        idx = rng.choice(total, size=n, replace=False)
        out[t] = score_subset(distance_matrix, idx)[metric]
    return out


def compare_subsets(
    distance_matrix: np.ndarray,
    methods: Sequence[str] = ("coreset", "random"),
    subset_size: Optional[int] = None,
    labels: Optional[Sequence[int]] = None,
    tasks: Optional[Sequence[str]] = None,
    sources: Optional[Sequence[str]] = None,
    seed: int = 42,
    baseline_trials: int = 200,
) -> ComparisonResult:
    """Score several equal-sized subsets head to head against a random null model.

    Args:
        distance_matrix: ``(N, N)`` DTW distances.
        methods: Selection strategies to compare, first two used for the delta table.
        subset_size: Episodes per subset. Defaults to a quarter of the dataset,
            floored at 2 and capped at N.
        labels: Cluster labels, for stratified selection and cluster coverage.
        tasks: Per-episode task names, for task coverage.
        sources: Per-episode source names, for source coverage.
        seed: Seed for random selection.
        baseline_trials: Random subsets drawn for the null distribution; 0 disables.

    Returns:
        A :class:`ComparisonResult`. ``baseline`` holds the null distribution summary
        and the leading subset's percentile within it.
    """
    total = int(distance_matrix.shape[0])
    size = subset_size if subset_size is not None else max(2, total // 4)
    size = int(np.clip(size, 2, total))

    # One global cutoff, shared by every subset, so redundancy is comparable.
    cutoff = redundancy_threshold(distance_matrix)

    subsets: List[SubsetScore] = []
    for method in methods:
        indices = select_indices(method, size, distance_matrix, labels=labels, seed=seed)
        subsets.append(
            SubsetScore(
                name=method,
                method=method,
                indices=indices,
                metrics=score_subset(
                    distance_matrix, indices, tasks, sources, labels, threshold=cutoff
                ),
            )
        )

    baseline = None
    samples = None
    if baseline_trials and total > size:
        samples = random_baseline(distance_matrix, size, trials=baseline_trials, seed=seed)
        best = subsets[0]
        score = best.metrics["diversity_score"]
        std = float(samples.std())
        baseline = {
            "trials": float(baseline_trials),
            "mean": float(samples.mean()),
            "std": std,
            "p05": float(np.quantile(samples, 0.05)),
            "p95": float(np.quantile(samples, 0.95)),
            "candidate": score,
            "candidate_name": best.name,
            "percentile": float((samples < score).mean() * 100.0),
            # Standard deviations above the random mean; the effect size.
            "z_score": float((score - samples.mean()) / std) if std > 1e-12 else 0.0,
        }

    return ComparisonResult(
        subsets=subsets, subset_size=size, baseline=baseline, baseline_samples=samples
    )


def selection_curve(
    distance_matrix: np.ndarray,
    methods: Sequence[str] = ("coreset", "random", "redundant"),
    sizes: Optional[Sequence[int]] = None,
    labels: Optional[Sequence[int]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Diversity score as a function of subset size, per selection strategy.

    The budget curve: for any training budget on the x-axis, it shows how much
    behavioural diversity each strategy retains. A coreset curve sitting above random
    across the whole range is the strongest single evidence that the ranking is real
    and not an artifact of one chosen size.
    """
    total = int(distance_matrix.shape[0])
    if sizes is None:
        sizes = [s for s in (5, 10, 20, 30, 50, 75, 100, 150, 200, 300) if s < total]
        sizes = sizes or [max(2, total // 2)]

    rows = []
    for method in methods:
        for size in sizes:
            idx = select_indices(method, size, distance_matrix, labels=labels, seed=seed)
            rows.append(
                {
                    "method": method,
                    "subset_size": int(size),
                    "diversity_score": score_subset(distance_matrix, idx)["diversity_score"],
                }
            )
    return pd.DataFrame(rows)
