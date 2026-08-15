"""End-to-end orchestration for both tracks.

**Track 2** (:func:`run_pipeline`): load -> DTW -> project/cluster -> score -> sequence.
Both the CLI (``run_pipeline.py``) and the Streamlit dashboard (``app.py``) call it, so
the numbers shown in the dashboard and the numbers written to disk cannot drift apart.

**Track 1** (:func:`build_path_finder`): takes a finished Track 2 result and builds the
clip graph, goal matcher and graph layout on top of it. Layered this way because the two
tracks share every expensive computation -- the DTW matrix is the interference signal and
the kinematic difficulty score is the ramp signal -- so the path finder adds milliseconds
rather than re-deriving anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # imported lazily at runtime to keep the Track 2 path dependency-free
    from .goal_matcher import GoalMatch, GoalMatcher
    from .graph import ClipGraph, GraphConfig
    from .pathfinder import CurriculumPath, PathConfig

from .cluster_mapper import (
    cluster_medoids,
    diversity_report,
    find_redundant_pairs,
    project_and_cluster,
    suggest_n_clusters,
)
from .curriculum import build_curriculum, stage_summary
from .diversity_engine import DTWConfig, compute_dtw_matrix
from .loader import TrajectoryDataset, load_zarr_trajectories

# Metadata fields worth scoring the clustering against.
VALIDATION_FIELDS = ("task_name", "source", "embodiment")


@dataclass
class PipelineResult:
    """Everything the dashboard or a report needs, computed once."""

    dataset: TrajectoryDataset
    distance_matrix: np.ndarray
    embedding: np.ndarray
    labels: np.ndarray
    report: Dict[str, float]
    curriculum: pd.DataFrame
    stages: pd.DataFrame
    redundant_pairs: List[Tuple[int, int, float]]
    medoids: Dict[int, int]
    agreement: Dict[str, float] = field(default_factory=dict)
    agreement_support: Dict[str, int] = field(default_factory=dict)
    suggested_k: Optional[int] = None
    silhouette_by_k: Dict[int, float] = field(default_factory=dict)
    config: Optional[DTWConfig] = None

    @property
    def n_episodes(self) -> int:
        return len(self.dataset)

    def frame(self) -> pd.DataFrame:
        """Tidy per-episode table combining embedding, cluster and metadata.

        Indexed in dataset order, which is what the 3-D scatter plots against.
        """
        ds = self.dataset
        data = {
            "episode_id": ds.episode_ids,
            "UMAP_X": self.embedding[:, 0],
            "UMAP_Y": self.embedding[:, 1],
            "UMAP_Z": self.embedding[:, 2] if self.embedding.shape[1] > 2 else 0.0,
            "cluster": self.labels,
            "cluster_label": [f"Group {int(l)}" for l in self.labels],
            "n_frames": ds.lengths,
        }
        for field_name in ("source", "task_name", "embodiment", "arm_used", "missing_frame_ratio"):
            data[field_name] = ds.field_values(field_name)
        # The human-written sentence the goal matcher searches over. Defaulted to empty
        # rather than "unknown": a literal "unknown" token would become shared vocabulary
        # across every undescribed episode and dilute the TF-IDF match.
        data["task_description"] = ds.field_values("task_description", default="")
        df = pd.DataFrame(data)

        # Attach difficulty/stage by episode_id so ordering differences cannot mislabel.
        if not self.curriculum.empty:
            cols = [
                "episode_id", "difficulty", "difficulty_z", "stage", "curriculum_rank",
                "coreset_rank", "is_cluster_medoid",
            ]
            available = [c for c in cols if c in self.curriculum.columns]
            df = df.merge(self.curriculum[available], on="episode_id", how="left")
        return df


def run_pipeline(
    data_dir: str = "data",
    n_clusters: Optional[int] = None,
    arm: str = "auto",
    min_length: int = 30,
    normalize: str = "zscore",
    max_length: Optional[int] = 200,
    length_normalize: bool = True,
    sakoe_chiba_radius: Optional[int] = None,
    difficulty_scaling: str = "rank",
    linkage: str = "average",
    n_jobs: int = -1,
    cache_dir: Optional[str] = ".cache",
    random_state: int = 42,
    verbose: bool = True,
) -> PipelineResult:
    """Run the full diversity and curriculum pipeline over a directory of episodes.

    Args:
        data_dir: Directory of ``*.zarr`` EgoVerse episode stores.
        n_clusters: Curriculum groups. ``None`` selects ``k`` by maximising the silhouette
            score over the DTW metric, restricted to partitions that are not degenerate --
            see :func:`~src.cluster_mapper.suggest_n_clusters`.
        arm: Arm selection mode -- see :func:`~src.loader.load_zarr_trajectories`.
        min_length: Minimum valid frames per episode.
        normalize: DTW preprocessing mode (``center``/``zscore``/``none``). Defaults to
            ``zscore`` -- shape-only comparison -- because EgoVerse spans embodiments whose
            motion *extent* differs ~5x (a YAM arm on a tabletop against a head-mounted
            Aria recording someone walking a room). Under ``center``, which preserves
            extent, that difference dominates the distance and clustering collapses: on 273
            episodes it gave a pairwise ``tail_ratio`` of 3.4, one cluster holding 99% of
            episodes, and ARI 0.01 against ``task_name``, versus 1.2 / 25% / 0.70 under
            ``zscore``. ``center`` remains right for a single-embodiment dataset, where
            extent is skill rather than hardware -- see :mod:`src.diversity_engine`.
        max_length: Resampling cap; bounds the quadratic DTW cost.
        length_normalize: Divide pairwise costs by mean sequence length.
        sakoe_chiba_radius: Optional DTW warping band.
        difficulty_scaling: ``"rank"`` or ``"minmax"``.
        linkage: Agglomerative linkage over the precomputed distances.
        n_jobs: Parallel workers for DTW.
        cache_dir: Disk cache for the parsed dataset and the DTW matrix, each keyed by a
            content hash of its inputs; ``None`` disables both.
        random_state: Seed for UMAP.
        verbose: Print progress.

    Returns:
        A :class:`PipelineResult`. Returns an empty-but-valid result when fewer than
        two episodes survive loading, so callers can render a message instead of
        handling an exception.
    """
    dataset = load_zarr_trajectories(
        data_dir, min_length=min_length, arm=arm, verbose=verbose, cache_dir=cache_dir
    )
    n = len(dataset)
    if n < 2:
        return PipelineResult(
            dataset=dataset,
            distance_matrix=np.zeros((n, n)),
            embedding=np.zeros((n, 3)),
            labels=np.zeros(n, dtype=int),
            report=diversity_report(np.zeros((n, n))),
            curriculum=pd.DataFrame(),
            stages=pd.DataFrame(),
            redundant_pairs=[],
            medoids={},
        )

    config = DTWConfig(
        normalize=normalize,
        max_length=max_length,
        length_normalize=length_normalize,
        sakoe_chiba_radius=sakoe_chiba_radius,
        n_jobs=n_jobs,
        verbose=0,
    )
    distance_matrix = compute_dtw_matrix(dataset.trajectories, config, cache_dir=cache_dir)

    # Pass `linkage` through: selecting k under average linkage and then clustering with
    # complete/single would choose k for a partition that never gets built.
    suggested_k, silhouette_by_k = suggest_n_clusters(
        distance_matrix, 2, min(10, n - 1), linkage=linkage
    )
    k = n_clusters if n_clusters is not None else max(2, suggested_k)

    embedding, labels = project_and_cluster(
        distance_matrix, n_clusters=k, linkage=linkage, random_state=random_state
    )
    report = diversity_report(distance_matrix, labels)

    curriculum = build_curriculum(
        distance_matrix,
        labels,
        dataset.trajectories,
        dataset.episode_ids,
        dt=1.0 / dataset.fps,
        scaling=difficulty_scaling,
        metadata=dataset.metadata,
    )

    agreement, agreement_support = _label_agreement(dataset, labels)

    # A collapsed partition makes every downstream "skill family" meaningless, and nothing
    # else in the output announces it -- silhouette actually looks *better* when it happens.
    counts = np.bincount(labels)
    dominant = float(counts.max() / counts.sum())
    if verbose and (report.get("tail_ratio", 0.0) > 2.0 or dominant > 0.6):
        print(
            f"[pipeline] WARNING: clustering looks degenerate — largest cluster holds "
            f"{dominant:.0%} of episodes, pairwise tail_ratio="
            f"{report.get('tail_ratio', float('nan')):.2f}.\n"
            f"[pipeline]   With normalize={normalize!r}, distance is dominated by motion "
            "extent, which tracks embodiment rather than skill on a multi-source dataset.\n"
            "[pipeline]   Try normalize='zscore' to compare motion shape instead."
        )

    return PipelineResult(
        dataset=dataset,
        distance_matrix=distance_matrix,
        embedding=embedding,
        labels=labels,
        report=report,
        curriculum=curriculum,
        stages=stage_summary(curriculum),
        redundant_pairs=find_redundant_pairs(distance_matrix),
        medoids=cluster_medoids(distance_matrix, labels),
        agreement=agreement,
        agreement_support=agreement_support,
        suggested_k=suggested_k,
        silhouette_by_k=silhouette_by_k,
        config=config,
    )


# --------------------------------------------------------------------------- #
# Track 1: the curriculum path finder
# --------------------------------------------------------------------------- #
@dataclass
class PathFinderContext:
    """Everything needed to answer goal queries against a built graph.

    Kept separate from :class:`PipelineResult` on purpose. The expensive stage is loading
    plus the O(N^2 T^2) DTW matrix; graph construction and goal matching are milliseconds.
    Splitting them means moving an interference-weight slider in the dashboard rebuilds
    only the graph, and never re-runs DTW.

    Attributes:
        clip_graph: The directed, dual-weighted graph.
        matcher: Fitted TF-IDF goal matcher over the same clips.
        layout: ``(N, 2)`` force-directed positions for drawing.
        distance_matrix: DTW distances for the scoped clip set, aligned to the graph.
        kept_indices: Graph index -> index in the unscoped dataset. The identity mapping
            when no task scoping was applied.
        task_names: The scope that was applied, or ``None`` for the whole dataset.
    """

    clip_graph: "ClipGraph"
    matcher: "GoalMatcher"
    layout: np.ndarray
    distance_matrix: np.ndarray
    kept_indices: np.ndarray
    task_names: Optional[List[str]] = None

    def find(
        self, goal: str, path_config: Optional["PathConfig"] = None,
        target_selection: str = "hardest", target_index: Optional[int] = None,
    ) -> Tuple["GoalMatch", "CurriculumPath"]:
        """Resolve a free-text goal and return the curriculum path to it.

        Args:
            goal: Training goal in plain English.
            path_config: Search and rehearsal settings.
            target_selection: Which clip inside the matched task family to aim at.
            target_index: Explicit override, bypassing the text match for the *target*
                while still returning the match so the UI can show what it would have
                picked.

        Returns:
            ``(match, path)``.
        """
        from .goal_matcher import GoalMatch
        from .pathfinder import find_curriculum_path

        match = self.matcher.match(goal, target_selection=target_selection)
        index = int(target_index) if target_index is not None else match.target_index
        path = find_curriculum_path(self.clip_graph, index, path_config)
        return match, path


def build_path_finder(
    result: PipelineResult,
    graph_config: Optional["GraphConfig"] = None,
    task_names: Optional[Sequence[str]] = None,
    layout_seed: int = 42,
) -> PathFinderContext:
    """Build the clip graph, goal matcher and layout on top of a pipeline result.

    Args:
        result: A completed :func:`run_pipeline` result with at least two episodes.
        graph_config: Edge weighting. Defaults to :class:`~src.graph.GraphConfig`.
        task_names: Restrict the graph to these task families, so a curriculum toward
            one goal is not routed through unrelated domains. ``None`` uses everything.
        layout_seed: Seed for the force-directed layout.

    Returns:
        A :class:`PathFinderContext`.

    Raises:
        ValueError: if fewer than two episodes are available, or the scope is empty.
    """
    from .goal_matcher import GoalMatcher
    from .graph import build_clip_graph, force_directed_layout, scope_to_tasks

    if result.n_episodes < 2:
        raise ValueError(
            f"need at least 2 usable episodes to build a clip graph, got {result.n_episodes}"
        )

    frame = result.frame()
    scoped, distances, kept = scope_to_tasks(frame, result.distance_matrix, task_names)
    if len(scoped) < 2:
        raise ValueError(
            f"task scope {list(task_names or [])} leaves only {len(scoped)} clip(s); "
            "at least 2 are needed"
        )

    clip_graph = build_clip_graph(distances, scoped, graph_config)
    embedding = result.embedding[kept] if len(result.embedding) == len(frame) else None
    layout = force_directed_layout(clip_graph, embedding=embedding, seed=layout_seed)

    return PathFinderContext(
        clip_graph=clip_graph,
        matcher=GoalMatcher(scoped, distance_matrix=distances),
        layout=layout,
        distance_matrix=distances,
        kept_indices=kept,
        task_names=list(task_names) if task_names else None,
    )


def _label_agreement(
    dataset: TrajectoryDataset, labels: np.ndarray
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Adjusted Rand Index between the DTW clusters and each metadata field.

    This is the pipeline's own correctness check. ``task_name`` agreement near 1.0
    means the unsupervised DTW grouping recovered the human task labels without ever
    seeing them -- evidence the distance metric captures behaviour rather than noise.

    **Unlabelled episodes are excluded, not scored.** Not every EgoVerse source populates
    ``task_name``: on a 267-episode sample spanning four sources, 78 episodes carried no
    task at all and the loader recorded them as ``"unknown"``. Treating that placeholder as
    a real class silently asks the clustering to recover a category that means "no
    information", which drags the ARI toward zero and makes the headline validation number
    a measure of label coverage rather than of the distance metric. Episodes without a
    label are dropped from the comparison and the surviving count is reported alongside,
    so a high score cannot hide a thin sample.

    Returns:
        ``(agreement, support)`` -- the ARI per field, and how many labelled episodes each
        score was computed over.
    """
    from sklearn.metrics import adjusted_rand_score

    missing = {"unknown", "", "none", "nan", "null"}
    out: Dict[str, float] = {}
    support: Dict[str, int] = {}
    for field_name in VALIDATION_FIELDS:
        values = dataset.field_values(field_name)
        keep = [i for i, v in enumerate(values) if str(v).strip().lower() not in missing]
        support[field_name] = len(keep)
        # Two labelled classes over at least a handful of episodes, or the score is noise.
        if len(keep) < 4:
            continue
        kept_values = [values[i] for i in keep]
        if len(set(kept_values)) < 2:
            continue
        try:
            out[field_name] = float(adjusted_rand_score(kept_values, labels[keep]))
        except Exception:  # noqa: BLE001
            continue
    return out, support
