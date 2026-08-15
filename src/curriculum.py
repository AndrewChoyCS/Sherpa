"""Kinematic difficulty scoring and curriculum sequencing.

The diversity engine answers "how varied is this dataset?". A *curriculum* engine
must also answer "in what order should a policy see these episodes?", which needs a
per-episode difficulty signal that the DTW distance matrix alone does not provide.

Difficulty is derived from the raw (un-normalised) XYZ trajectory, because physical
units are exactly what makes a motion hard:

- **path_length**    total distance travelled; longer motions accumulate more error
- **tortuosity**     path length / net displacement; 1.0 is a straight reach, high
                     values mean winding, indirect motion
- **rms_jerk**       third derivative magnitude; proxy for non-smooth, snappy control
- **reversal_rate**  direction flips per timestep; counts fine corrective sub-motions
- **duration**       timestep count; longer horizons are harder to credit-assign
- **workspace_span** bounding-box diagonal; how much of the workspace is swept

These are combined via *robust* z-scores (median / MAD rather than mean / std) so a
single pathological episode cannot compress everything else into a narrow band, then
min-max mapped to ``difficulty`` in ``[0, 1]``.

Two orderings are produced, serving different training strategies:

``curriculum_rank``
    Easy-to-hard, grouped into stages. Clusters are ordered by mean difficulty and
    become stages; within a stage, episodes ascend by difficulty. This is classic
    curriculum learning -- master simple motion families before complex ones.

``coreset_rank``
    Maximum-coverage-first, via farthest-point traversal of the DTW matrix starting
    from the global medoid. Truncating at any K yields a K-episode subset that is
    near-maximally spread under the DTW metric. This is the ordering to use when
    subsampling a dataset to a training budget.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Relative contribution of each kinematic feature to the difficulty score.
#
# Weighting is deliberately tilted towards *scale-free, duration-free* features.
# Raw `path_length` and `duration` are strongly correlated with each other and vary
# by ~40x across EgoVerse sources (a 400-frame yam episode travels ~1.5 m; a
# 3000-frame aria episode travels ~60 m). Weighting them heavily makes "difficulty"
# collapse into "which source is this", which is not a curriculum signal. So the
# dominant terms are dimensionless: tortuosity, normalised jerk, reversal rate.
DEFAULT_DIFFICULTY_WEIGHTS: Dict[str, float] = {
    "log_tortuosity": 1.0,
    "log_normalized_jerk": 1.0,
    "reversal_rate": 0.75,
    "path_length": 0.5,
    "workspace_span": 0.35,
    "duration": 0.25,
}

# How to map the weighted z-score onto [0, 1].
#   "rank"   -> percentile rank; uniform spread, robust to extreme outliers
#   "minmax" -> linear rescale; preserves gaps but a single outlier compresses the rest
DIFFICULTY_SCALINGS = ("rank", "minmax")

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# per-trajectory kinematics
# --------------------------------------------------------------------------- #
def kinematic_features(trajectory: np.ndarray, dt: float = 1.0) -> Dict[str, float]:
    """Compute interpretable kinematic descriptors for one ``(T, 3)`` trajectory.

    Args:
        trajectory: Raw XYZ positions in physical units (metres).
        dt: Timestep duration. Only rescales the derivative-based features.

    Returns:
        Feature name -> value. Degenerate (very short) trajectories yield zeros
        rather than NaN so the dataframe stays numeric.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    n = traj.shape[0]

    feats: Dict[str, float] = {
        "duration": float(n),
        "path_length": 0.0,
        "net_displacement": 0.0,
        "tortuosity": 1.0,
        "log_tortuosity": 0.0,
        "mean_speed": 0.0,
        "max_speed": 0.0,
        "rms_acceleration": 0.0,
        "rms_jerk": 0.0,
        "normalized_jerk": 0.0,
        "log_normalized_jerk": 0.0,
        "reversal_rate": 0.0,
        "workspace_span": 0.0,
    }
    if n < 2:
        return feats

    deltas = np.diff(traj, axis=0)
    step = np.linalg.norm(deltas, axis=1)

    feats["path_length"] = float(step.sum())
    feats["net_displacement"] = float(np.linalg.norm(traj[-1] - traj[0]))
    feats["workspace_span"] = float(np.linalg.norm(traj.max(axis=0) - traj.min(axis=0)))

    # Tortuosity is unbounded for a closed loop (net displacement -> 0), so clamp the
    # denominator against a fraction of the path length rather than a bare epsilon.
    denom = max(feats["net_displacement"], 1e-4 * feats["path_length"], _EPS)
    feats["tortuosity"] = float(feats["path_length"] / denom)
    feats["log_tortuosity"] = float(np.log1p(feats["tortuosity"]))

    velocity = deltas / dt
    speed = np.linalg.norm(velocity, axis=1)
    feats["mean_speed"] = float(speed.mean())
    feats["max_speed"] = float(speed.max())

    if n >= 3:
        accel = np.diff(velocity, axis=0) / dt
        feats["rms_acceleration"] = float(np.sqrt((np.linalg.norm(accel, axis=1) ** 2).mean()))
        if n >= 4:
            jerk = np.diff(accel, axis=0) / dt
            sq_jerk = np.linalg.norm(jerk, axis=1) ** 2
            feats["rms_jerk"] = float(np.sqrt(sq_jerk.mean()))

            # Dimensionless normalised jerk, the standard motor-control smoothness
            # measure: sqrt( (T^5 / L^2) * integral(||jerk||^2 dt) ). Being free of
            # both length and time units, it compares a 400-frame robot episode
            # against a 3000-frame human one without unit bias.
            duration_s = n * dt
            if feats["path_length"] > _EPS and duration_s > _EPS:
                integral = float(sq_jerk.sum() * dt)
                nj = np.sqrt(integral * duration_s**5 / feats["path_length"] ** 2)
                feats["normalized_jerk"] = float(nj)
                feats["log_normalized_jerk"] = float(np.log1p(nj))

        # A "reversal" is a >90 degree turn between consecutive motion directions.
        norms = np.linalg.norm(deltas, axis=1, keepdims=True)
        moving = norms.ravel() > _EPS
        if moving.sum() >= 2:
            unit = deltas[moving] / norms[moving]
            cosines = np.einsum("ij,ij->i", unit[:-1], unit[1:])
            feats["reversal_rate"] = float((cosines < 0.0).sum() / max(len(cosines), 1))

    return feats


def _resolve_dt(dt: object, n: int) -> np.ndarray:
    """Broadcast ``dt`` to one value per trajectory.

    EgoVerse mixes 30 fps and 60 fps sources, so a per-episode timestep matters:
    a shared ``dt`` would make 60 fps episodes look twice as fast and far jerkier.
    """
    arr = np.atleast_1d(np.asarray(dt, dtype=np.float64))
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    if arr.size != n:
        raise ValueError(f"dt has {arr.size} entries but there are {n} trajectories")
    return arr


def features_dataframe(
    trajectories: Sequence[np.ndarray],
    episode_ids: Optional[Sequence[str]] = None,
    dt: object = 1.0,
) -> pd.DataFrame:
    """Kinematic features for every trajectory, one row per episode.

    Args:
        trajectories: Raw ``(T_i, C)`` XYZ trajectories in metres.
        episode_ids: Identifiers; generated if omitted.
        dt: Scalar timestep, or one timestep per trajectory (``1 / fps``).
    """
    dts = _resolve_dt(dt, len(trajectories))
    rows = [kinematic_features(t, dt=float(step)) for t, step in zip(trajectories, dts)]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(DEFAULT_DIFFICULTY_WEIGHTS))
    ids = list(episode_ids) if episode_ids is not None else [f"ep{i:04d}" for i in range(len(df))]
    df.insert(0, "episode_id", ids)
    return df


# --------------------------------------------------------------------------- #
# difficulty scoring
# --------------------------------------------------------------------------- #
def robust_zscore(values: np.ndarray) -> np.ndarray:
    """Median/MAD z-score. Outlier-resistant, and constant input maps to zeros."""
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return vals
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    if mad < _EPS:
        std = float(vals.std())
        if std < _EPS:
            return np.zeros_like(vals)
        return (vals - vals.mean()) / std
    # 1.4826 scales MAD to be a consistent estimator of sigma for normal data.
    return (vals - median) / (1.4826 * mad)


def difficulty_zscores(
    features: pd.DataFrame, weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """Weighted robust z-score per episode -- the raw difficulty signal.

    Unbounded and gap-preserving: use this when the *magnitude* of the difficulty gap
    between episodes matters. :func:`difficulty_scores` maps it onto ``[0, 1]``.
    """
    weights = weights or DEFAULT_DIFFICULTY_WEIGHTS
    n = len(features)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    accum = np.zeros(n, dtype=np.float64)
    total_weight = 0.0
    for name, weight in weights.items():
        if name not in features.columns or weight == 0:
            continue
        accum += weight * robust_zscore(features[name].to_numpy(dtype=np.float64))
        total_weight += abs(weight)
    if total_weight == 0:
        return np.zeros(n, dtype=np.float64)
    return accum / total_weight


def difficulty_scores(
    features: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    scaling: str = "rank",
) -> np.ndarray:
    """Collapse kinematic features into a ``[0, 1]`` difficulty score per episode.

    Args:
        features: Output of :func:`features_dataframe`.
        weights: Feature -> weight. Missing columns are skipped with no error.
        scaling: ``"rank"`` (default) maps to percentile rank, giving a uniform
            spread; ``"minmax"`` rescales linearly. Rank is the default because
            EgoVerse difficulty is heavily right-skewed -- a handful of very long
            human episodes otherwise compress every robot episode into a band a few
            thousandths wide, making the curriculum unreadable and un-thresholdable.

    Returns:
        ``(N,)`` scores in ``[0, 1]``; all-zeros when N < 2 or nothing varies.
    """
    if scaling not in DIFFICULTY_SCALINGS:
        raise ValueError(f"scaling must be one of {DIFFICULTY_SCALINGS}, got {scaling!r}")

    accum = difficulty_zscores(features, weights=weights)
    n = accum.size
    if n == 0:
        return accum
    if float(accum.max() - accum.min()) < _EPS:
        return np.zeros(n, dtype=np.float64)

    if scaling == "minmax":
        lo, hi = float(accum.min()), float(accum.max())
        return (accum - lo) / (hi - lo)

    # Rank scaling: average ranks so ties share a score, normalised to [0, 1].
    order = np.argsort(accum, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    if n > 1:
        # Average the ranks of tied values to avoid an arbitrary ordering.
        series = pd.Series(accum)
        ranks = series.rank(method="average").to_numpy(dtype=np.float64) - 1.0
        ranks /= n - 1
    else:
        ranks[:] = 0.0
    return ranks


# --------------------------------------------------------------------------- #
# orderings
# --------------------------------------------------------------------------- #
def coreset_order(distance_matrix: np.ndarray) -> List[int]:
    """Farthest-point traversal: maximum-coverage ordering of episodes.

    Starts at the global medoid, then repeatedly appends whichever unselected
    episode is farthest from the already-selected set. Any prefix of the result is a
    near-optimal maximally-diverse subset (the classic 2-approximation to k-center).
    """
    n = int(distance_matrix.shape[0])
    if n == 0:
        return []
    if n == 1:
        return [0]

    start = int(np.argmin(distance_matrix.sum(axis=1)))
    selected = np.zeros(n, dtype=bool)
    selected[start] = True
    order = [start]
    min_dist = distance_matrix[start].astype(np.float64).copy()

    for _ in range(n - 1):
        candidates = np.flatnonzero(~selected)
        nxt = int(candidates[np.argmax(min_dist[candidates])])
        selected[nxt] = True
        order.append(nxt)
        min_dist = np.minimum(min_dist, distance_matrix[nxt])
    return order


def build_curriculum(
    distance_matrix: np.ndarray,
    labels: Sequence[int],
    trajectories: Sequence[np.ndarray],
    episode_ids: Optional[Sequence[str]] = None,
    dt: object = 1.0,
    weights: Optional[Dict[str, float]] = None,
    scaling: str = "rank",
    metadata: Optional[Sequence[Dict[str, object]]] = None,
) -> pd.DataFrame:
    """Assemble the full curriculum table.

    Args:
        distance_matrix: ``(N, N)`` DTW distances.
        labels: Cluster label per episode.
        trajectories: Raw ``(T_i, C)`` XYZ trajectories in metres.
        episode_ids: Identifiers; generated if omitted.
        dt: Timestep duration, scalar or per-episode (``1 / fps``).
        weights: Difficulty feature weights.
        scaling: Difficulty scaling, ``"rank"`` or ``"minmax"``.
        metadata: Optional per-episode dicts (task_name, source, ...) merged in as
            columns so the curriculum is readable without a separate join.

    Returns:
        One row per episode, sorted by ``curriculum_rank``, with the kinematic
        features plus ``cluster``, ``difficulty`` (``[0, 1]``), ``difficulty_z``
        (raw, gap-preserving), ``stage`` (1-indexed, easiest motion family first),
        ``stage_position``, ``curriculum_rank``, ``coreset_rank`` and
        ``is_cluster_medoid``.
    """
    from .cluster_mapper import cluster_medoids  # local import avoids a cycle

    df = features_dataframe(trajectories, episode_ids=episode_ids, dt=dt)
    n = len(df)
    if n == 0:
        return df

    if metadata is not None and len(metadata) == n:
        for column in ("source", "task_name", "embodiment", "arm_used", "missing_frame_ratio"):
            if any(column in m for m in metadata):
                df[column] = [m.get(column) for m in metadata]

    labels_arr = np.asarray(labels, dtype=int) if len(labels) == n else np.zeros(n, dtype=int)
    df["cluster"] = labels_arr
    df["difficulty"] = difficulty_scores(df, weights=weights, scaling=scaling)
    df["difficulty_z"] = difficulty_zscores(df, weights=weights)

    # Stages: motion families ordered easiest-first by mean within-cluster difficulty.
    mean_difficulty = df.groupby("cluster")["difficulty"].mean().sort_values()
    stage_of = {int(c): i + 1 for i, c in enumerate(mean_difficulty.index)}
    df["stage"] = df["cluster"].map(stage_of).astype(int)

    medoids = set(cluster_medoids(distance_matrix, labels_arr).values()) if n else set()
    df["is_cluster_medoid"] = [i in medoids for i in range(n)]

    coreset = coreset_order(distance_matrix)
    coreset_rank = np.empty(n, dtype=int)
    for rank, idx in enumerate(coreset):
        coreset_rank[idx] = rank + 1
    df["coreset_rank"] = coreset_rank

    df = df.sort_values(["stage", "difficulty"], kind="stable").reset_index(drop=True)
    df["curriculum_rank"] = np.arange(1, n + 1)
    df["stage_position"] = df.groupby("stage").cumcount() + 1

    ordered = [
        "curriculum_rank",
        "episode_id",
        "stage",
        "stage_position",
        "cluster",
        "difficulty",
        "difficulty_z",
        "coreset_rank",
        "is_cluster_medoid",
    ]
    ordered = [c for c in ordered if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest]


def stage_summary(curriculum: pd.DataFrame) -> pd.DataFrame:
    """Per-stage rollup: episode count and difficulty range."""
    if curriculum.empty:
        return pd.DataFrame(columns=["stage", "n_episodes", "mean_difficulty"])
    grouped = (
        curriculum.groupby("stage")
        .agg(
            n_episodes=("episode_id", "count"),
            mean_difficulty=("difficulty", "mean"),
            min_difficulty=("difficulty", "min"),
            max_difficulty=("difficulty", "max"),
            mean_path_length=("path_length", "mean"),
            mean_tortuosity=("tortuosity", "mean"),
        )
        .reset_index()
    )
    return grouped
