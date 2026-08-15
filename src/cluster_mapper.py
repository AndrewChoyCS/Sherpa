"""UMAP projection, distance-matrix clustering, and quantitative diversity metrics.

Two corrections to the obvious implementation are worth calling out:

**Do not run KMeans on a distance matrix.** KMeans interprets its input as points
in Euclidean feature space and computes means of them. Feeding it an ``(N, N)``
distance matrix silently clusters "rows of distances" as N-dimensional coordinate
vectors, which is not clustering the trajectories under the DTW metric -- and the
centroids it averages into existence do not correspond to any trajectory. This
module uses average-linkage agglomerative clustering with ``metric="precomputed"``,
which consumes the DTW distances directly and is deterministic. Medoids (real
episodes) stand in for centroids.

**The mean of the full matrix understates diversity.** The diagonal is N structural
zeros, so ``mean(D)`` is deflated by a factor of ``(N-1)/N``. All statistics here
are computed over the strict upper triangle -- the N(N-1)/2 genuine pairs.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# UMAP's default spectral initialisation solves a sparse eigenproblem that requires
# k < N; it raises for small N. Below this many samples we initialise randomly.
_SPECTRAL_INIT_MIN_SAMPLES = 12


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #
def project_umap(
    distance_matrix: np.ndarray,
    n_components: int = 3,
    n_neighbors: Optional[int] = None,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Embed a precomputed distance matrix into ``n_components`` dimensions via UMAP.

    Falls back to metric MDS, then to zeros, so a tiny or degenerate dataset yields
    a plottable embedding instead of an exception.

    Args:
        distance_matrix: Symmetric ``(N, N)`` distances with a zero diagonal.
        n_components: Output dimensionality (3 for the dashboard's 3-D scatter).
        n_neighbors: UMAP locality. Defaults to ``min(15, N-1)``, floored at 2.
        min_dist: UMAP cluster tightness.
        random_state: Seed. Also forces UMAP single-threaded, making runs reproducible.

    Returns:
        ``(N, n_components)`` float64 embedding.
    """
    import umap  # imported lazily: numba JIT warm-up is slow and not always needed

    n = int(distance_matrix.shape[0])
    if n == 0:
        return np.zeros((0, n_components), dtype=np.float64)
    if n <= n_components:
        # Not enough points to define the target dimensionality meaningfully.
        emb = np.zeros((n, n_components), dtype=np.float64)
        emb[:, 0] = np.arange(n, dtype=np.float64)
        return emb

    neighbors = n_neighbors if n_neighbors is not None else min(15, n - 1)
    neighbors = int(np.clip(neighbors, 2, n - 1))
    init = "random" if n < _SPECTRAL_INIT_MIN_SAMPLES else "spectral"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reducer = umap.UMAP(
                n_components=n_components,
                metric="precomputed",
                n_neighbors=neighbors,
                min_dist=min_dist,
                init=init,
                random_state=random_state,
            )
            return np.asarray(reducer.fit_transform(distance_matrix), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"UMAP failed ({type(exc).__name__}: {exc}); falling back to MDS.", stacklevel=2
        )

    try:
        from sklearn.manifold import MDS

        mds = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=random_state,
            normalized_stress=False,
        )
        return np.asarray(mds.fit_transform(distance_matrix), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"MDS fallback also failed ({exc}); returning zeros.", stacklevel=2)
        return np.zeros((n, n_components), dtype=np.float64)


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #
def cluster_precomputed(
    distance_matrix: np.ndarray, n_clusters: int = 5, linkage: str = "average"
) -> np.ndarray:
    """Cluster trajectories directly under the DTW metric.

    Args:
        distance_matrix: Symmetric ``(N, N)`` distances.
        n_clusters: Requested clusters, clamped to ``[1, N]``.
        linkage: ``"average"``, ``"complete"`` or ``"single"``. ``"ward"`` is
            rejected because it requires Euclidean coordinates, not distances.

    Returns:
        ``(N,)`` integer cluster labels.
    """
    n = int(distance_matrix.shape[0])
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    if linkage == "ward":
        raise ValueError("ward linkage needs Euclidean coordinates; use average/complete/single")

    k = int(np.clip(n_clusters, 1, n))
    if k == 1:
        return np.zeros(n, dtype=int)

    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage=linkage)
    return model.fit_predict(distance_matrix).astype(int)


def project_and_cluster(
    distance_matrix: np.ndarray,
    n_clusters: int = 5,
    n_components: int = 3,
    linkage: str = "average",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project to ``n_components``-D and cluster, in one call.

    Returns:
        ``(embedding, labels)`` of shapes ``(N, n_components)`` and ``(N,)``.
    """
    embedding = project_umap(
        distance_matrix, n_components=n_components, random_state=random_state
    )
    labels = cluster_precomputed(distance_matrix, n_clusters=n_clusters, linkage=linkage)
    return embedding, labels


def suggest_n_clusters(
    distance_matrix: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    linkage: str = "average",
    max_cluster_fraction: float = 0.6,
) -> Tuple[int, Dict[int, float]]:
    """Pick ``k`` by maximising the silhouette score over *non-degenerate* partitions.

    Raw silhouette maximisation cannot be used on its own here, and the failure is not
    subtle. On a real 267-episode dataset, average-linkage clustering produced a ``k=2``
    partition of **266 vs 1** episodes -- one far-outlier trajectory against everything
    else -- which scored the *best* silhouette in the range, 0.63, and won. Silhouette
    rewards that: the singleton has no within-cluster distance to hurt it, and every other
    episode is far from it. The resulting labels are useless downstream, where clusters
    stand in for *skill families*: curriculum stages collapse to two, and the graph's
    interference penalty stops discriminating anything.

    The guard is on the **largest** cluster's share, not the smallest cluster's size. That
    distinction is load-bearing, and getting it wrong inverts the outcome on both datasets
    tested here. A singleton is frequently legitimate -- on the 28-episode sample *every*
    ``k`` from 3 to 10 contains one, and ``k=7`` is simultaneously the silhouette winner
    (0.478) and the best recovery of the human task labels (ARI 0.901). Rejecting
    partitions for containing a small cluster therefore throws away the right answer and
    elects ``k=2``, whose ARI is 0.264. What actually characterises the pathology is one
    cluster swallowing the dataset: 99.6% in the 266-vs-1 case, versus 25% at ``k=7``.

    All scores are still returned, including rejected ones, because the silhouette-vs-``k``
    curve is displayed and hiding the rejected peak would misrepresent the choice.

    Args:
        distance_matrix: Symmetric ``(N, N)`` distances.
        k_min: Smallest ``k`` considered.
        k_max: Largest ``k`` considered.
        linkage: Agglomerative linkage.
        max_cluster_fraction: Reject a ``k`` whose largest cluster holds more than this
            share of the episodes. Set to 1.0 to restore plain silhouette maximisation.

    Returns:
        ``(best_k, {k: silhouette})``. ``best_k`` is 1 when no ``k`` is scorable, and falls
        back to the plain silhouette winner when *no* ``k`` is non-degenerate -- which is
        itself a signal that the distance matrix is outlier-dominated rather than that some
        ``k`` is fine. :func:`diversity_report`'s ``tail_ratio`` is the check for that.
    """
    n = int(distance_matrix.shape[0])
    scores: Dict[int, float] = {}
    eligible: Dict[int, float] = {}
    dominance: Dict[int, float] = {}

    upper = min(k_max, n - 1)
    for k in range(max(2, k_min), max(2, upper) + 1):
        labels = cluster_precomputed(distance_matrix, n_clusters=k, linkage=linkage)
        if len(np.unique(labels)) < 2:
            continue
        try:
            score = float(silhouette_score(distance_matrix, labels, metric="precomputed"))
        except Exception:  # noqa: BLE001
            continue
        scores[k] = score
        counts = np.bincount(labels)
        share = float(counts.max() / counts.sum())
        dominance[k] = share
        if share <= max_cluster_fraction:
            eligible[k] = score

    if eligible:
        return max(eligible, key=eligible.get), scores
    # Nothing qualified. Falling back to the plain silhouette winner here would elect the
    # single most degenerate partition available -- silhouette *rewards* the pathology, so
    # the unguarded fallback reintroduces exactly what the filter exists to block. Fail
    # soft instead: pick the least dominated partition, breaking ties on silhouette. The
    # caller should treat this as a signal to check `tail_ratio` and reconsider
    # normalisation rather than trust the labels.
    if dominance:
        return min(dominance, key=lambda k: (dominance[k], -scores[k])), scores
    return 1, scores


def cluster_medoids(distance_matrix: np.ndarray, labels: Sequence[int]) -> Dict[int, int]:
    """Index of the most central real episode in each cluster.

    A medoid is the actual trajectory minimising summed within-cluster distance --
    the closest thing to a "representative episode" the DTW metric admits.
    """
    labels = np.asarray(labels)
    medoids: Dict[int, int] = {}
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if idx.size == 1:
            medoids[int(label)] = int(idx[0])
            continue
        sub = distance_matrix[np.ix_(idx, idx)]
        medoids[int(label)] = int(idx[np.argmin(sub.sum(axis=1))])
    return medoids


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def pairwise_values(distance_matrix: np.ndarray) -> np.ndarray:
    """The N(N-1)/2 genuine pairwise distances (strict upper triangle)."""
    n = int(distance_matrix.shape[0])
    if n < 2:
        return np.zeros(0, dtype=np.float64)
    return distance_matrix[np.triu_indices(n, k=1)].astype(np.float64)


def calculate_diversity_score(distance_matrix: np.ndarray) -> float:
    """Headline diversity score: mean pairwise DTW distance.

    Averaged over the strict upper triangle, so the zero diagonal cannot deflate it.
    Higher means the dataset spans a broader range of motion behaviours.
    """
    vals = pairwise_values(distance_matrix)
    return float(vals.mean()) if vals.size else 0.0


def nearest_neighbor_distances(distance_matrix: np.ndarray) -> np.ndarray:
    """Distance from each episode to its closest neighbour.

    This is the redundancy signal: a near-zero value means some other episode in
    the dataset already covers this behaviour.
    """
    n = int(distance_matrix.shape[0])
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    masked = distance_matrix.astype(np.float64).copy()
    np.fill_diagonal(masked, np.inf)
    return masked.min(axis=1)


def find_redundant_pairs(
    distance_matrix: np.ndarray, quantile: float = 0.05, max_pairs: int = 50
) -> List[Tuple[int, int, float]]:
    """Episode pairs closer than the ``quantile`` of the pairwise distribution.

    These are near-duplicate demonstrations -- the first thing to prune when a
    dataset is over budget.

    Returns:
        ``(i, j, distance)`` sorted closest-first, at most ``max_pairs`` entries.
    """
    vals = pairwise_values(distance_matrix)
    if vals.size == 0:
        return []
    threshold = float(np.quantile(vals, quantile))
    n = int(distance_matrix.shape[0])
    iu, ju = np.triu_indices(n, k=1)
    mask = distance_matrix[iu, ju] <= threshold
    pairs = [(int(i), int(j), float(distance_matrix[i, j])) for i, j in zip(iu[mask], ju[mask])]
    pairs.sort(key=lambda p: p[2])
    return pairs[:max_pairs]


def diversity_report(
    distance_matrix: np.ndarray, labels: Optional[Sequence[int]] = None
) -> Dict[str, float]:
    """Quantitative diversity summary for the dataset.

    Keys:
        ``n_trajectories``, ``n_pairs``
        ``diversity_score`` -- mean pairwise DTW (headline number)
        ``median_pairwise``, ``std_pairwise``, ``min_pairwise``, ``max_pairwise``
        ``dispersion`` -- coefficient of variation; how uneven the coverage is
        ``tail_ratio`` -- 99th percentile over the median. Above ~2 the matrix is
            outlier-dominated and agglomerative clustering will collapse; see
            :func:`suggest_n_clusters`.
        ``mean_nn_distance`` -- mean distance to nearest neighbour; higher = less
            redundant. Reported as ``uniqueness``-style coverage.
        ``redundancy_ratio`` -- fraction of episodes whose nearest neighbour sits in
            the closest 5% of all pairs, i.e. likely near-duplicates
        ``silhouette`` -- cluster separation under the DTW metric (needs labels)
        ``cluster_balance`` -- normalised Shannon entropy of cluster sizes, 1.0 when
            curriculum stages are evenly populated
    """
    n = int(distance_matrix.shape[0])
    vals = pairwise_values(distance_matrix)
    report: Dict[str, float] = {
        "n_trajectories": float(n),
        "n_pairs": float(vals.size),
        "diversity_score": float(vals.mean()) if vals.size else 0.0,
        "median_pairwise": float(np.median(vals)) if vals.size else 0.0,
        "std_pairwise": float(vals.std()) if vals.size else 0.0,
        "min_pairwise": float(vals.min()) if vals.size else 0.0,
        "max_pairwise": float(vals.max()) if vals.size else 0.0,
    }
    mean = report["diversity_score"]
    report["dispersion"] = float(report["std_pairwise"] / mean) if mean > 0 else 0.0

    # Heavy-tail check on the pairwise distribution: the 99th percentile over the median.
    # This is the early warning for the clustering collapse documented in
    # `suggest_n_clusters`. When a handful of episodes sit several times farther from
    # everything than a typical pair does, agglomerative linkage peels them off one at a
    # time instead of splitting the bulk, and *every* k leaves one cluster holding almost
    # the whole dataset. Measured on the same 273 episodes: 3.4 under `center`
    # normalisation, where clustering collapsed and ARI vs task_name was 0.01, against 1.2
    # under `zscore`, where clusters were balanced and ARI was 0.70. Silhouette does not
    # catch this -- it *prefers* the collapsed matrix -- so it needs its own number.
    if vals.size:
        median = float(np.median(vals))
        report["tail_ratio"] = (
            float(np.quantile(vals, 0.99) / median) if median > 0 else 0.0
        )

    nn = nearest_neighbor_distances(distance_matrix)
    report["mean_nn_distance"] = float(nn.mean()) if nn.size else 0.0
    if vals.size:
        threshold = float(np.quantile(vals, 0.05))
        report["redundancy_ratio"] = float((nn <= threshold).mean())
    else:
        report["redundancy_ratio"] = 0.0

    if labels is not None and n >= 3:
        labels_arr = np.asarray(labels)
        if 2 <= len(np.unique(labels_arr)) < n:
            try:
                report["silhouette"] = float(
                    silhouette_score(distance_matrix, labels_arr, metric="precomputed")
                )
            except Exception:  # noqa: BLE001
                pass
        counts = np.bincount(labels_arr - labels_arr.min())
        counts = counts[counts > 0]
        if counts.size > 1:
            p = counts / counts.sum()
            report["cluster_balance"] = float(
                -(p * np.log(p)).sum() / np.log(counts.size)
            )
    return report
