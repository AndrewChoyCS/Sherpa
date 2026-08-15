"""Proxy metrics for a curriculum path, and baselines to judge them against.

A path that merely *exists* proves nothing. These are the four properties a good
curriculum should have, each measurable without training a model:

**Difficulty monotonicity** -- does difficulty ramp, or lurch? Reported as Spearman
correlation with training position, the fraction of non-decreasing steps, and the largest
single jump. Measured over the *introduction* sequence by default, excluding rehearsal
steps: a review clip is a deliberate dip backwards, so counting it as a monotonicity
violation would penalise the anti-forgetting mechanism for doing its job. The
review-inclusive number is reported alongside it.

**Interference** -- how often consecutive clips switch task, skill family, embodiment or
lab, and how far apart they are under DTW. This is the quantity the graph's interference
term exists to suppress.

**Coverage** -- what fraction of the skill families and task families the policy has been
exposed to before it reaches the target. A path that beelines to the goal scores
beautifully on monotonicity and interference while teaching almost nothing, so coverage is
what keeps the other two honest.

**Redundancy** -- is the path just near-duplicate clips? Uses the same 5th-percentile
near-duplicate threshold as the diversity engine's :func:`~src.cluster_mapper.find_redundant_pairs`,
so "redundant" means the same thing in both tracks.

**On the baselines.** Two random baselines are reported, and conflating them would
overstate the result:

- *Random order, same clips* re-shuffles the path's own selection. It isolates the value
  of the **ordering** alone.
- *Random subset, same size* draws fresh clips. It isolates the value of the
  **selection** as well, and is the harder comparison to win on coverage.

Difficulty-sorted and coreset-prefix baselines are also included, because they are the two
obvious things a practitioner would try instead, and a curriculum engine should have to
beat them rather than only beating chance.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .cluster_mapper import pairwise_values
from .curriculum import coreset_order
from .graph import ClipGraph
from .pathfinder import CurriculumPath

# A difficulty step larger than this counts as a "jump" -- 3x the default per-step target
# increment of 0.05, i.e. clearly beyond a smooth ramp rather than marginally over.
JUMP_THRESHOLD = 0.15

# Quantile of the full dataset's pairwise distances below which two clips count as
# near-duplicates. Matches `find_redundant_pairs`.
REDUNDANCY_QUANTILE = 0.05

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# individual metrics
# --------------------------------------------------------------------------- #
def difficulty_monotonicity(
    difficulty: Sequence[float], jump_threshold: float = JUMP_THRESHOLD
) -> Dict[str, float]:
    """How smoothly difficulty ascends along an ordering.

    Args:
        difficulty: Difficulty per training step, in order.
        jump_threshold: Step size above which a transition counts as an abrupt jump.

    Returns:
        ``spearman`` (rank correlation with position; 1.0 is perfectly ordered),
        ``frac_nondecreasing``, ``max_jump``, ``max_abs_step``, ``mean_abs_step``,
        ``n_large_jumps``. Degenerate inputs (fewer than 2 steps, or constant difficulty)
        yield 0.0 for the correlation rather than NaN, so tables stay numeric.

    Note:
        ``max_jump`` counts only the largest *upward* step, because the harm being measured
        is a policy thrown at something it is not ready for; a step down is not that. That
        makes it the wrong statistic for comparing against a shuffled baseline, though: over
        a fixed set of clips, a descending ordering spends its large rises as large *falls*
        and so posts a deceptively small ``max_jump``. ``mean_abs_step`` and
        ``max_abs_step`` are the order-fair smoothness measures -- total absolute variation
        is provably minimised by the sorted order -- so those are what the baseline
        comparison should be read on.
    """
    values = np.asarray(list(difficulty), dtype=np.float64)
    n = values.size
    out = {
        "spearman": 0.0,
        "frac_nondecreasing": 1.0,
        "max_jump": 0.0,
        "max_abs_step": 0.0,
        "mean_abs_step": 0.0,
        "n_large_jumps": 0.0,
    }
    if n < 2:
        return out

    steps = np.diff(values)
    out["frac_nondecreasing"] = float((steps >= -_EPS).mean())
    out["max_jump"] = float(steps.max()) if steps.size else 0.0
    out["max_abs_step"] = float(np.abs(steps).max())
    out["mean_abs_step"] = float(np.abs(steps).mean())
    out["n_large_jumps"] = float((steps > jump_threshold).sum())

    if float(values.max() - values.min()) > _EPS:
        from scipy.stats import spearmanr

        rho = spearmanr(values, np.arange(n)).statistic
        out["spearman"] = float(rho) if np.isfinite(rho) else 0.0
    return out


def interference_profile(
    clips: Sequence[int], nodes: pd.DataFrame, normalized_distance: np.ndarray
) -> Dict[str, float]:
    """Rate of abrupt context switches, and motion distance, between consecutive clips.

    Args:
        clips: Clip indices in training order.
        nodes: Per-clip attribute frame in dataset order.
        normalized_distance: ``(N, N)`` DTW distances rescaled to ``[0, 1]``.

    Returns:
        Per-step switch rates for task, skill family, embodiment and lab, plus mean and
        max consecutive DTW distance.
    """
    order = [int(c) for c in clips]
    out = {
        "task_switch_rate": 0.0,
        "cluster_switch_rate": 0.0,
        "embodiment_switch_rate": 0.0,
        "source_switch_rate": 0.0,
        "mean_consecutive_dtw": 0.0,
        "max_consecutive_dtw": 0.0,
    }
    if len(order) < 2:
        return out

    for column, key in (
        ("task_name", "task_switch_rate"),
        ("cluster", "cluster_switch_rate"),
        ("embodiment", "embodiment_switch_rate"),
        ("source", "source_switch_rate"),
    ):
        if column not in nodes.columns:
            continue
        values = nodes[column].astype(str).to_numpy()
        switches = [values[u] != values[v] for u, v in zip(order[:-1], order[1:])]
        out[key] = float(np.mean(switches))

    hops = np.array(
        [normalized_distance[u, v] for u, v in zip(order[:-1], order[1:])], dtype=np.float64
    )
    out["mean_consecutive_dtw"] = float(hops.mean())
    out["max_consecutive_dtw"] = float(hops.max())
    return out


def coverage(
    clips: Sequence[int],
    nodes: pd.DataFrame,
    target_index: Optional[int] = None,
) -> Dict[str, float]:
    """Skill and task exposure accumulated before the target is reached.

    Args:
        clips: Clip indices in training order.
        nodes: Per-clip attribute frame in dataset order.
        target_index: Truncate at the first occurrence of this clip. ``None`` uses the
            whole ordering, which is the right choice for baselines that may reach the
            target at a different point.

    Returns:
        ``cluster_coverage`` and ``task_coverage`` as fractions of the families present in
        the whole dataset, ``n_clips_to_target``, and ``distinct_clusters``.
    """
    order = [int(c) for c in clips]
    if target_index is not None and int(target_index) in order:
        order = order[: order.index(int(target_index)) + 1]

    out = {
        "cluster_coverage": 0.0,
        "task_coverage": 0.0,
        "n_clips_to_target": float(len(order)),
        "distinct_clusters": 0.0,
    }
    if not order:
        return out

    if "cluster" in nodes.columns:
        clusters = nodes["cluster"].astype(str).to_numpy()
        total = len(set(clusters.tolist()))
        seen = {clusters[i] for i in order}
        out["distinct_clusters"] = float(len(seen))
        out["cluster_coverage"] = float(len(seen) / total) if total else 0.0
    if "task_name" in nodes.columns:
        tasks = nodes["task_name"].astype(str).to_numpy()
        total = len(set(tasks.tolist()))
        seen = {tasks[i] for i in order}
        out["task_coverage"] = float(len(seen) / total) if total else 0.0
    return out


def redundancy(
    clips: Sequence[int],
    distance_matrix: np.ndarray,
    quantile: float = REDUNDANCY_QUANTILE,
) -> Dict[str, float]:
    """Whether an ordering is padded with near-duplicate clips.

    Args:
        clips: Clip indices in training order.
        distance_matrix: Raw ``(N, N)`` DTW distances for the **whole** dataset -- the
            near-duplicate threshold is a property of the dataset, not of the path, so
            using a submatrix here would make short paths look artificially diverse.
        quantile: Pairwise-distance quantile defining "near-duplicate".

    Returns:
        ``frac_consecutive_near_duplicate``, ``mean_pairwise_dtw`` within the ordering,
        ``distinct_ratio`` (distinct clips / steps; below 1.0 means repeats, which for a
        curriculum path is rehearsal rather than waste).
    """
    order = [int(c) for c in clips]
    out = {
        "frac_consecutive_near_duplicate": 0.0,
        "mean_pairwise_dtw": 0.0,
        "distinct_ratio": 1.0,
    }
    if len(order) < 2:
        return out

    out["distinct_ratio"] = float(len(set(order)) / len(order))

    vals = pairwise_values(distance_matrix)
    if vals.size:
        threshold = float(np.quantile(vals, quantile))
        hops = np.array(
            [distance_matrix[u, v] for u, v in zip(order[:-1], order[1:])], dtype=np.float64
        )
        out["frac_consecutive_near_duplicate"] = float((hops <= threshold).mean())

    unique = sorted(set(order))
    if len(unique) >= 2:
        sub = distance_matrix[np.ix_(unique, unique)]
        out["mean_pairwise_dtw"] = float(sub[np.triu_indices(len(unique), k=1)].mean())
    return out


# --------------------------------------------------------------------------- #
# combined report
# --------------------------------------------------------------------------- #
def path_report(
    clips: Sequence[int],
    clip_graph: ClipGraph,
    distance_matrix: np.ndarray,
    target_index: Optional[int] = None,
    is_review: Optional[Sequence[bool]] = None,
) -> Dict[str, float]:
    """All four proxy-metric families for one ordering, as a flat dict.

    Args:
        clips: Clip indices in training order.
        clip_graph: The graph, for node attributes and normalised distances.
        distance_matrix: Raw dataset DTW distances, for the redundancy threshold.
        target_index: Target clip, for coverage truncation.
        is_review: Rehearsal flags. When given, monotonicity is measured over the
            non-review steps and the review-inclusive value is reported separately as
            ``spearman_with_reviews``.

    Returns:
        Flat metric name -> value, plus ``n_steps`` and ``n_unique_clips``.
    """
    nodes = clip_graph.nodes
    order = [int(c) for c in clips]
    difficulty = nodes["difficulty"].astype(float).fillna(1.0).to_numpy()

    if is_review is not None and len(is_review) == len(order):
        introduced = [c for c, review in zip(order, is_review) if not review]
    else:
        introduced = order

    report: Dict[str, float] = {
        "n_steps": float(len(order)),
        "n_unique_clips": float(len(set(order))),
    }
    # Monotonicity and interference are both measured over the introduction sequence.
    # Rehearsal steps deliberately dip backwards in difficulty *and* deliberately switch
    # context -- that is what rehearsal is. Scoring them as violations would penalise the
    # anti-forgetting mechanism for working, and would make the path incomparable to
    # baselines, which have no rehearsal steps at all. The review-inclusive figures are
    # reported alongside, suffixed `_with_reviews`, so the cost of rehearsal stays visible.
    report.update(difficulty_monotonicity([difficulty[c] for c in introduced]))
    report.update(interference_profile(introduced, nodes, clip_graph.normalized_distance))
    if is_review is not None and len(is_review) == len(order) and any(is_review):
        full_mono = difficulty_monotonicity([difficulty[c] for c in order])
        full_interference = interference_profile(order, nodes, clip_graph.normalized_distance)
        report["spearman_with_reviews"] = full_mono["spearman"]
        report["frac_nondecreasing_with_reviews"] = full_mono["frac_nondecreasing"]
        report["task_switch_rate_with_reviews"] = full_interference["task_switch_rate"]
        report["cluster_switch_rate_with_reviews"] = full_interference["cluster_switch_rate"]
    report.update(coverage(order, nodes, target_index))
    report.update(redundancy(order, distance_matrix))
    return report


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def _mean_report(reports: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Average a set of per-seed reports key by key."""
    if not reports:
        return {}
    keys = reports[0].keys()
    return {k: float(np.mean([r[k] for r in reports])) for k in keys}


def compare_orderings(
    path: CurriculumPath,
    clip_graph: ClipGraph,
    distance_matrix: np.ndarray,
    n_seeds: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """Score the found path against four alternative orderings of the same size.

    Baselines, and what each one isolates:

    ``Random order (same clips)``
        The path's own clip selection, reshuffled. Difference from the path is
        attributable purely to **ordering**.
    ``Random subset (same size)``
        Fresh clips, randomly ordered, with the target forced in so coverage-to-target is
        comparable. Difference is attributable to **selection and ordering** together.
    ``Difficulty-sorted (same clips)``
        The obvious hand-rolled curriculum: same clips, sorted easy to hard. Nails
        monotonicity by construction, so this is the baseline the path must beat on
        *interference*, not on ramp smoothness.
    ``Coreset prefix (same size)``
        The existing farthest-point traversal from :mod:`src.curriculum` -- a
        coverage-first ordering. Strong on coverage and redundancy, weak on ramp.

    Args:
        path: The found curriculum path.
        clip_graph: The graph it was found in.
        distance_matrix: Raw dataset DTW distances.
        n_seeds: Random draws averaged for each stochastic baseline.
        random_state: Seed.

    Returns:
        One row per ordering, indexed by name, columns being the metrics from
        :func:`path_report`.
    """
    rng = np.random.default_rng(random_state)
    n_clips = clip_graph.n_clips
    selection = path.unique_clips
    size = len(selection)
    target = path.target_index

    rows: Dict[str, Dict[str, float]] = {}

    rows["Curriculum path"] = path_report(
        path.clips, clip_graph, distance_matrix, target, path.is_review
    )

    if size >= 2:
        shuffled = []
        for _ in range(n_seeds):
            order = list(rng.permutation(selection))
            shuffled.append(path_report(order, clip_graph, distance_matrix, target))
        rows["Random order (same clips)"] = _mean_report(shuffled)

        subsets = []
        others = [i for i in range(n_clips) if i != target]
        take = min(size - 1, len(others))
        for _ in range(n_seeds):
            picked = list(rng.choice(others, size=take, replace=False))
            order = list(rng.permutation(picked + [target]))
            subsets.append(path_report(order, clip_graph, distance_matrix, target))
        rows["Random subset (same size)"] = _mean_report(subsets)

        difficulty = clip_graph.nodes["difficulty"].astype(float).fillna(1.0).to_numpy()
        sorted_order = sorted(selection, key=lambda c: difficulty[c])
        rows["Difficulty-sorted (same clips)"] = path_report(
            sorted_order, clip_graph, distance_matrix, target
        )

        coreset = coreset_order(distance_matrix)[:size]
        rows["Coreset prefix (same size)"] = path_report(
            coreset, clip_graph, distance_matrix, target
        )

    frame = pd.DataFrame(rows).T
    frame.index.name = "ordering"
    return frame


def coverage_curve(
    clips: Sequence[int], nodes: pd.DataFrame, column: str = "cluster"
) -> List[int]:
    """Cumulative count of distinct families seen after each training step.

    Plotted in the dashboard: a curve that rises early and plateaus means the curriculum
    front-loads breadth, which is what should happen before difficulty ramps.
    """
    if column not in nodes.columns:
        return []
    values = nodes[column].astype(str).to_numpy()
    seen: set = set()
    out: List[int] = []
    for clip in clips:
        seen.add(values[int(clip)])
        out.append(len(seen))
    return out


# --------------------------------------------------------------------------- #
# multi-goal validation
# --------------------------------------------------------------------------- #
# Metrics whose definition does not depend on how difficulty is computed. Only these
# are safe for judging a difficulty *metric*: scoring a difficulty ordering on
# difficulty monotonicity is circular and always returns 1.0 by construction.
NON_CIRCULAR_METRICS = (
    "task_switch_rate",
    "cluster_switch_rate",
    "mean_consecutive_dtw",
    "frac_consecutive_near_duplicate",
)

# Single-feature stand-ins for difficulty, used as ablation controls. Each is a
# plausible thing someone would reach for instead of building a composite score.
NAIVE_SORT_KEYS = ("duration", "path_length", "tortuosity")


def sweep_orderings(
    clip_graph: "ClipGraph",
    distance_matrix: np.ndarray,
    path_config: Optional[object] = None,
    n_targets: int = 40,
    n_seeds: int = 10,
    random_state: int = 0,
) -> pd.DataFrame:
    """Score the path against its baselines across many goals, not one.

    A single curriculum is an anecdote. Metrics on one path vary enormously with which
    target was picked -- on this dataset the path beat difficulty-sorting on consecutive
    near-duplicates 0.400 to 0.750 for one goal, while across 40 goals the same
    comparison is 0.592 to 0.600, a tie. Any claim about the search must be made on the
    distribution.

    Args:
        clip_graph: The built graph.
        distance_matrix: Raw DTW distances for the scoped clips.
        path_config: Search/rehearsal settings; defaults to :class:`PathConfig`.
        n_targets: Goals to sample, capped at the clip count.
        n_seeds: Random draws averaged inside each stochastic baseline.
        random_state: Seed for target sampling.

    Returns:
        One row per (target, ordering), with every metric from :func:`path_report`.
    """
    from .pathfinder import PathConfig, find_curriculum_path

    config = path_config or PathConfig()
    rng = np.random.default_rng(random_state)
    total = int(clip_graph.n_clips)
    targets = rng.choice(total, size=int(min(n_targets, total)), replace=False)

    rows: List[Dict[str, float]] = []
    for target in targets:
        try:
            path = find_curriculum_path(clip_graph, int(target), config)
            frame = compare_orderings(path, clip_graph, distance_matrix, n_seeds=n_seeds)
        except Exception:  # noqa: BLE001 - an unreachable target must not abort the sweep
            continue
        if "Curriculum path" not in frame.index:
            frame = frame.T
        for name in frame.index:
            row: Dict[str, float] = {"target": int(target), "ordering": str(name)}
            for column in frame.columns:
                value = frame.loc[name, column]
                if not pd.isna(value):
                    row[str(column)] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def paired_verdict(
    sweep: pd.DataFrame,
    metric: str,
    a: str = "Curriculum path",
    b: str = "Difficulty-sorted (same clips)",
) -> Dict[str, float]:
    """Paired comparison of two orderings across goals, with a significance test.

    Pairs by target -- both orderings are scored on the same goal, so the comparison
    removes goal-to-goal variance. Uses Wilcoxon signed-rank rather than a t-test
    because these metrics are bounded ratios and far from normal.

    Returns:
        Means, the win/tie/loss split and ``p_value``. ``p_value`` is NaN when every
        pair is tied, which Wilcoxon cannot score.
    """
    left = sweep[sweep["ordering"] == a].set_index("target")[metric]
    right = sweep[sweep["ordering"] == b].set_index("target")[metric]
    joined = left.to_frame("a").join(right.to_frame("b")).dropna()

    out: Dict[str, Optional[float]] = {
        "n_goals": float(len(joined)),
        "mean_a": float(joined["a"].mean()) if len(joined) else None,
        "mean_b": float(joined["b"].mean()) if len(joined) else None,
        "a_better": float((joined["a"] < joined["b"]).sum()),
        "tied": float((joined["a"] == joined["b"]).sum()),
        "b_better": float((joined["a"] > joined["b"]).sum()),
        # None rather than NaN: `json.dump` writes a bare `NaN` literal that Python
        # reads back happily and `JSON.parse` rejects outright, so a consumer silently
        # renders nothing. Undefined here is legitimate -- Wilcoxon cannot score a
        # comparison where all pairs tie exactly, which happens on task_switch_rate.
        "p_value": None,
    }
    if len(joined) and (joined["a"] != joined["b"]).any():
        try:
            from scipy.stats import wilcoxon

            out["p_value"] = float(wilcoxon(joined["a"], joined["b"]).pvalue)
        except Exception:  # noqa: BLE001 - scipy is optional at runtime
            pass
    return out


def difficulty_ablation(
    clip_graph: "ClipGraph",
    distance_matrix: np.ndarray,
    curriculum: pd.DataFrame,
    path_config: Optional[object] = None,
    n_targets: int = 40,
    random_state: int = 0,
) -> pd.DataFrame:
    """Is the composite difficulty score better than a single naive feature?

    This is the ablation that isolates the *metric* rather than the search. For each
    goal, the same clip selection is ordered by the composite difficulty and by each
    naive stand-in (duration, path length, tortuosity) plus a random shuffle, and each
    ordering is scored **only on** :data:`NON_CIRCULAR_METRICS`.

    The restriction is the whole point. Difficulty monotonicity would be 1.0 for
    whichever key did the sorting, so it proves nothing. Task/cluster switching and
    consecutive DTW never see the difficulty score, so a composite that lowers them is
    capturing real structure rather than restating its own definition.

    Returns:
        One row per (target, sort key) with the non-circular metrics.
    """
    from .pathfinder import PathConfig, find_curriculum_path

    config = path_config or PathConfig()
    nodes = clip_graph.nodes
    episode_ids = nodes["episode_id"].tolist()
    indexed = curriculum.set_index("episode_id")

    keys: Dict[str, np.ndarray] = {
        "composite difficulty": nodes["difficulty"].astype(float).fillna(1.0).to_numpy()
    }
    for column in NAIVE_SORT_KEYS:
        if column in indexed.columns:
            keys[f"{column} only"] = indexed.loc[episode_ids, column].to_numpy(dtype=float)

    rng = np.random.default_rng(random_state)
    total = int(clip_graph.n_clips)
    targets = rng.choice(total, size=int(min(n_targets, total)), replace=False)

    rows: List[Dict[str, float]] = []
    for target in targets:
        try:
            path = find_curriculum_path(clip_graph, int(target), config)
        except Exception:  # noqa: BLE001
            continue
        selection = list(path.unique_clips)
        for name, values in keys.items():
            order = sorted(selection, key=lambda clip: values[int(clip)])
            report = path_report(order, clip_graph, distance_matrix, int(target))
            rows.append({"target": int(target), "sort_key": name, **report})
        shuffled = list(rng.permutation(selection))
        rows.append(
            {
                "target": int(target),
                "sort_key": "random order",
                **path_report(shuffled, clip_graph, distance_matrix, int(target)),
            }
        )
    return pd.DataFrame(rows)


def scope_descriptor(
    clip_graph: ClipGraph,
    task_names: Optional[Sequence[str]] = None,
    domain: Optional[str] = None,
) -> Dict[str, object]:
    """Identify the clip population a measurement was taken over.

    Proxy-metric results are *scope-dependent*, not merely noisy: the path beats a
    difficulty ordering on repeated material inside one task family (p = 0.003) and ties
    with it across the whole graph (p = 0.65). A number without its population attached
    is therefore unfalsifiable, and two readers of the same file can reach opposite
    conclusions -- which has already happened once here, when a shared artifact under
    ``reports/`` was regenerated at a different scope between two measurements.

    Every sweep artifact embeds this, so a stale or mislabelled file identifies itself.
    """
    return {
        "scope": domain or ("tasks:" + ",".join(sorted(task_names)) if task_names else "unscoped"),
        "domain": domain,
        "n_clips": int(clip_graph.n_clips),
        "task_names": sorted(task_names) if task_names else None,
        "is_scoped": bool(task_names or domain),
    }


def ablation_payload(
    ablation: pd.DataFrame,
    scope: Dict[str, object],
    sweep: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """JSON-ready difficulty-ablation summary, carrying the scope it was measured at.

    Shaped for a frontend that must *display* which population it measured rather than
    assume one. Within a single task family the naive sort keys collapse onto the
    composite -- there is barely any task switching left to differentiate -- so a
    consumer showing this next to scoped results has to say so or the two read as
    contradictory.
    """
    metrics = list(NON_CIRCULAR_METRICS)
    means = ablation.groupby("sort_key")[metrics].mean()
    # Composite first, random last; the naive proxies keep their relative order between.
    order = [k for k in ["composite difficulty"] if k in means.index]
    order += [k for k in means.index if k not in order and k != "random order"]
    order += [k for k in ["random order"] if k in means.index]

    payload: Dict[str, object] = {
        **scope,
        "n_goals": int(ablation["target"].nunique()) if len(ablation) else 0,
        "metrics": metrics,
        "lower_is_better": metrics,  # every non-circular metric here is a cost
        "rows": [
            {"sort_key": str(key), **{m: float(means.loc[key, m]) for m in metrics}}
            for key in order
        ],
    }
    if sweep is not None and len(sweep):
        payload["paired_vs_difficulty_sorted"] = {
            metric: paired_verdict(sweep, metric) for metric in ["spearman", *metrics]
        }
    return payload
