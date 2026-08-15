"""The clip knowledge graph: nodes are clips, edges encode "train B after A".

This is the substrate for Track 1 (the Curation Engine). Where the diversity engine
answers *how varied is this dataset*, the graph answers *in what order should a policy
see these clips, and which transitions are safe*.

Every edge carries a cost built from two independent signals, which is the whole idea:

**A difficulty term** so the curriculum ramps rather than lurches. Each clip already has
a rank-scaled ``difficulty`` in ``[0, 1]`` from :mod:`src.curriculum`. An edge is cheap
when it advances difficulty by roughly ``target_step`` and expensive when it either
stalls (``delta == 0``, no learning progress) or leaps (``delta >> target_step``, the
policy is thrown at something it is not ready for). Both failure modes are penalised by
the same ``|delta - target_step|`` term, which is why it is written that way rather than
as a one-sided "don't jump" penalty.

**An interference term** so consecutive clips do not jump wildly across task, skill
family, embodiment or lab. Abrupt switches are what drives catastrophic interference
during training: gradients from an unrelated task overwrite what was just learned. The
term combines the continuous DTW motion distance (already computed and cached) with
categorical mismatch penalties on the metadata EgoVerse gives us for free.

**A redundancy term**, which is the other half of that same signal and is not optional.
Interference alone rewards *minimising* distance, so its true optimum is training on the
same clip repeatedly. Measured on the 273-clip graph, that is exactly what the search
returned: a flawless difficulty ramp in which every consecutive pair was a near-duplicate
and skill coverage was 10%. So distances below a data-derived novelty floor are penalised
too. Both signals are therefore two-sided -- difficulty penalises stalling and leaping,
motion distance penalises repetition and disruption -- and the curriculum lives in the
sweet spot of each.

Two properties are deliberate and are what the tests pin down:

- **All edge costs are non-negative**, so Dijkstra is valid. Nothing here can produce a
  negative weight, which is why the search does not need Bellman-Ford.
- **Edges are directed and difficulty-ordered.** ``u -> v`` exists only when
  ``difficulty[v] >= difficulty[u] - backslide_tolerance``. The ramp is therefore
  *structural*, not merely discouraged by a cost: a path physically cannot run downhill
  more than the tolerance allows, so no weighting mistake can produce a descending
  curriculum.

A virtual ``START`` node with zero-cost edges into the easiest skill family stands in for
"wherever a fresh policy should begin", so the search picks its own entry point instead
of us hard-coding one clip.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

# The virtual source. A string, so it can never collide with an integer clip index.
START = "START"

# Metadata fields whose mismatch counts as interference, and the config key holding
# each one's penalty. Ordered most- to least-disruptive.
INTERFERENCE_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("task_name", "p_task"),
    ("cluster", "p_cluster"),
    ("embodiment", "p_embodiment"),
    ("source", "p_source"),
)

# Columns `build_clip_graph` needs on the node frame. `PipelineResult.frame()` supplies
# all of them, in dataset order, which is the order the distance matrix is indexed in.
REQUIRED_NODE_COLUMNS = ("episode_id", "difficulty", "cluster")

_EPS = 1e-12


@dataclass(frozen=True)
class GraphConfig:
    """Edge-weighting and connectivity settings for the clip graph.

    Attributes:
        k_neighbors: Candidate edges per clip, by DTW nearest neighbour. Unioned in both
            directions, so a clip's effective degree is usually higher than ``k``.
        backslide_tolerance: How far difficulty may *decrease* across an edge. Small but
            non-zero: exactly-zero tolerance fragments the graph, because rank-scaled
            difficulty is near-unique per clip.
        target_step: The per-step difficulty increment the ramp aims for. 0.05 over a
            ``[0, 1]`` range implies a comfortable ~20-step curriculum.
        w_difficulty: Weight on the ramp term.
        w_interference: Weight on the interference term.
        step_penalty: Flat per-hop cost. Without it Dijkstra will happily string together
            many near-free hops, producing a long meandering curriculum.
        backslide_penalty: Extra multiplier on difficulty *decreases* within tolerance.
        p_task, p_cluster, p_embodiment, p_source: Categorical mismatch penalties, in the
            same units as the normalised DTW distance (i.e. ~``[0, 1]``), so 0.5 for a
            task switch means "as disruptive as a maximally distant motion".
        dtw_quantile: Pairwise-distance quantile used to normalise DTW onto ``[0, 1]``.
            The 95th rather than the max, so one outlier pair cannot squash the scale.
        start_quantile: Difficulty quantile bounding the zero-cost entry pool. See
            :func:`_easy_start_clips` for why family membership alone is not enough.
        w_redundancy: Weight on the near-duplicate penalty. Set to 0 and the search will
            happily return a curriculum of near-identical clips.
        novelty_quantile: Pairwise-distance quantile defining "near-duplicate", matching
            :mod:`src.path_metrics` and :func:`find_redundant_pairs`.
    """

    k_neighbors: int = 10
    backslide_tolerance: float = 0.05
    target_step: float = 0.05
    w_difficulty: float = 1.0
    w_interference: float = 1.0
    step_penalty: float = 0.1
    backslide_penalty: float = 2.0
    p_task: float = 0.5
    p_cluster: float = 0.3
    p_embodiment: float = 0.3
    p_source: float = 0.2
    dtw_quantile: float = 0.95
    start_quantile: float = 0.10
    w_redundancy: float = 1.0
    novelty_quantile: float = 0.05

    def __post_init__(self) -> None:
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        if self.target_step <= 0:
            raise ValueError("target_step must be > 0")
        if not 0.0 < self.dtw_quantile <= 1.0:
            raise ValueError("dtw_quantile must be in (0, 1]")
        if not 0.0 < self.start_quantile <= 1.0:
            raise ValueError("start_quantile must be in (0, 1]")
        if not 0.0 <= self.novelty_quantile <= 1.0:
            raise ValueError("novelty_quantile must be in [0, 1]")
        for name in ("w_difficulty", "w_interference", "step_penalty", "backslide_penalty",
                     "p_task", "p_cluster", "p_embodiment", "p_source",
                     "backslide_tolerance", "w_redundancy"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0 to keep edge costs non-negative")

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ClipGraph:
    """A built clip graph plus everything needed to explain and draw it.

    Attributes:
        graph: Directed graph over integer clip indices, plus the ``START`` node.
        nodes: Per-clip attribute frame, in dataset order (i.e. distance-matrix order).
        normalized_distance: DTW distances rescaled onto ``[0, 1]``; the interference
            term's continuous component, and the A* heuristic's basis.
        start_clips: Clip indices reachable from ``START`` at zero cost.
        novelty_floor: Normalised DTW distance below which a transition counts as
            near-duplicate; derived from the data at ``config.novelty_quantile``.
        repairs: ``(source, destination)`` edges added purely to restore reachability;
            ``source`` is :data:`START` when the clip turned out to be an extra entry
            point. Surfaced rather than hidden -- these are the transitions the k-NN
            structure did not justify on its own.
        config: The config used.
    """

    graph: nx.DiGraph
    nodes: pd.DataFrame
    normalized_distance: np.ndarray
    start_clips: List[int]
    repairs: List[Tuple[object, int]]
    config: GraphConfig
    novelty_floor: float = 0.0

    @property
    def n_clips(self) -> int:
        return len(self.nodes)

    def episode_id(self, idx: int) -> str:
        return str(self.nodes["episode_id"].iloc[idx])

    def summary(self) -> str:
        n_edges = self.graph.number_of_edges() - len(self.start_clips)
        density = n_edges / max(self.n_clips, 1)
        return (
            f"ClipGraph(clips={self.n_clips}, edges={n_edges}, "
            f"mean_out_degree={density:.1f}, start_pool={len(self.start_clips)}, "
            f"reachability_repairs={len(self.repairs)})"
        )


# --------------------------------------------------------------------------- #
# scoping
# --------------------------------------------------------------------------- #
def scope_to_tasks(
    node_frame: pd.DataFrame,
    distance_matrix: np.ndarray,
    task_names: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Restrict the graph to one or a few related task domains.

    A curriculum toward "fold a shirt" should not be routed through utensil-sorting
    clips just because they happen to be kinematically adjacent. Scoping is applied
    here, at graph-build time, rather than at fetch time, so the Track 2 diversity
    analysis keeps the full cross-source dataset it is validated against.

    Args:
        node_frame: Per-clip attributes in dataset order.
        distance_matrix: ``(N, N)`` DTW distances in the same order.
        task_names: Task names to keep. ``None`` or empty keeps everything.

    Returns:
        ``(scoped_frame, scoped_distance_matrix, kept_indices)``. The frame is
        re-indexed from 0, so it stays aligned with the returned submatrix;
        ``kept_indices`` maps new index -> original dataset index.
    """
    n = len(node_frame)
    if not task_names:
        return node_frame.reset_index(drop=True), distance_matrix, np.arange(n)

    wanted = set(task_names)
    keep = np.flatnonzero(node_frame["task_name"].astype(str).isin(wanted).to_numpy())
    if keep.size == 0:
        raise ValueError(f"no clips match task_names={sorted(wanted)}")
    scoped = node_frame.iloc[keep].reset_index(drop=True)
    return scoped, distance_matrix[np.ix_(keep, keep)], keep


# --------------------------------------------------------------------------- #
# edge cost
# --------------------------------------------------------------------------- #
def normalize_distances(
    distance_matrix: np.ndarray, quantile: float = 0.95
) -> np.ndarray:
    """Rescale DTW distances onto ``[0, 1]`` for use as an interference cost.

    Divides by the ``quantile`` of the genuine pairwise distances (strict upper
    triangle, reusing the diversity engine's convention) and clips. Using a high
    quantile rather than the maximum keeps a single freak pair from compressing every
    other distance into near-zero, which would make the interference term inert.
    """
    from .cluster_mapper import pairwise_values

    vals = pairwise_values(distance_matrix)
    if vals.size == 0:
        return np.zeros_like(distance_matrix, dtype=np.float64)
    scale = float(np.quantile(vals, quantile))
    if scale < _EPS:
        # Every pair is identical: no interference signal to extract.
        return np.zeros_like(distance_matrix, dtype=np.float64)
    return np.clip(distance_matrix.astype(np.float64) / scale, 0.0, 1.0)


def edge_cost(
    difficulty_from: float,
    difficulty_to: float,
    normalized_dtw: float,
    mismatches: Dict[str, bool],
    config: GraphConfig,
    novelty_floor: float = 0.0,
) -> Dict[str, float]:
    """Cost of training ``to`` immediately after ``from``, broken into its terms.

    Three terms, and each of the first two is deliberately **two-sided**:

    - ``ramp`` penalises stalling (no difficulty progress) and leaping (too much) equally.
    - ``redundancy`` + ``interference`` do the same for motion distance. Interference grows
      as clips get *further* apart; redundancy grows as they get *closer* than
      ``novelty_floor``. Without the redundancy half, minimising interference is minimising
      distance, and the cheapest possible curriculum is the same clip over and over: on the
      273-clip graph the search returned a flawless difficulty ramp in which **every**
      consecutive pair was a near-duplicate, teaching essentially nothing. A step that
      shows the policy nothing new is a wasted step, so it has to cost something.

    Args:
        difficulty_from: Source clip difficulty in ``[0, 1]``.
        difficulty_to: Destination clip difficulty in ``[0, 1]``.
        normalized_dtw: Motion distance between the two clips, in ``[0, 1]``.
        mismatches: ``{field_name: True if the two clips differ}`` for the fields in
            :data:`INTERFERENCE_FIELDS`. Missing fields count as matching.
        config: Weights and penalties.
        novelty_floor: Distance below which a transition counts as near-duplicate. Derived
            from the data by :func:`build_clip_graph` at ``config.novelty_quantile``, the
            same quantile :mod:`src.path_metrics` uses to *measure* redundancy -- so the
            cost penalises exactly what the metric reports. ``0`` disables the term.

    Returns:
        ``{"ramp", "interference", "redundancy", "step", "weight"}``. Every value is
        non-negative, which is the precondition Dijkstra needs.
    """
    tau = config.target_step
    delta = float(difficulty_to) - float(difficulty_from)

    # Penalises stalls and leaps symmetrically around the target increment, then adds a
    # surcharge for going backwards at all.
    ramp = abs(delta - tau) / tau
    if delta < 0.0:
        ramp += config.backslide_penalty * (-delta) / tau

    interference = float(normalized_dtw)
    for field_name, penalty_key in INTERFERENCE_FIELDS:
        if mismatches.get(field_name, False):
            interference += float(getattr(config, penalty_key))

    # 1.0 for an identical clip, falling linearly to 0 at the novelty floor.
    redundancy = 0.0
    if novelty_floor > _EPS:
        redundancy = max(0.0, novelty_floor - float(normalized_dtw)) / novelty_floor

    weight = (
        config.w_difficulty * ramp
        + config.w_interference * interference
        + config.w_redundancy * redundancy
        + config.step_penalty
    )
    return {
        "ramp": float(ramp),
        "interference": float(interference),
        "redundancy": float(redundancy),
        "step": float(config.step_penalty),
        "weight": float(weight),
    }


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def _candidate_pairs(normalized: np.ndarray, k: int) -> List[Tuple[int, int]]:
    """Union of each clip's ``k`` nearest neighbours, as unordered pairs.

    Taking the union rather than the intersection (mutual-kNN) matters: a clip in a
    sparse region of motion space would otherwise end up isolated, and isolated clips
    are exactly the hard, late-curriculum ones we most need to route to.
    """
    n = int(normalized.shape[0])
    if n < 2:
        return []
    k = int(min(k, n - 1))
    masked = normalized.copy()
    np.fill_diagonal(masked, np.inf)
    # argpartition is O(n) per row vs O(n log n) for a full sort.
    neighbours = np.argpartition(masked, kth=k - 1, axis=1)[:, :k]
    pairs = set()
    for i in range(n):
        for j in neighbours[i]:
            j = int(j)
            if i != j:
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def _mismatch_map(nodes: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Per-field value arrays used to test whether two clips differ."""
    out: Dict[str, np.ndarray] = {}
    for field_name, _ in INTERFERENCE_FIELDS:
        if field_name in nodes.columns:
            out[field_name] = nodes[field_name].astype(str).to_numpy()
    return out


def _add_directed_edge(
    graph: nx.DiGraph,
    u: int,
    v: int,
    difficulty: np.ndarray,
    normalized: np.ndarray,
    values: Dict[str, np.ndarray],
    config: GraphConfig,
    repair: bool = False,
    novelty_floor: float = 0.0,
) -> None:
    """Add ``u -> v`` with its full cost breakdown attached."""
    mismatches = {name: bool(col[u] != col[v]) for name, col in values.items()}
    cost = edge_cost(
        float(difficulty[u]), float(difficulty[v]), float(normalized[u, v]), mismatches,
        config, novelty_floor,
    )
    graph.add_edge(
        u,
        v,
        weight=cost["weight"],
        ramp_cost=cost["ramp"],
        interference_cost=cost["interference"],
        redundancy_cost=cost["redundancy"],
        dtw=float(normalized[u, v]),
        difficulty_delta=float(difficulty[v] - difficulty[u]),
        task_switch=mismatches.get("task_name", False),
        cluster_switch=mismatches.get("cluster", False),
        embodiment_switch=mismatches.get("embodiment", False),
        source_switch=mismatches.get("source", False),
        is_repair=repair,
    )


def _easy_start_clips(nodes: pd.DataFrame, start_quantile: float = 0.10) -> List[int]:
    """Genuinely easy clips in the lowest-mean-difficulty skill family.

    Both conditions are required, and dropping either one breaks the search:

    - **Easiest family**, so the curriculum opens inside a *coherent* skill rather than on
      a scattered handful of unrelated easy motions.
    - **Bottom ``start_quantile`` of difficulty**, because family membership alone is not a
      difficulty bound. On a balanced 273-clip graph the easiest family held 106 clips
      spanning difficulty 0.0 to 0.9, and wiring all of them to ``START`` at zero cost made
      the *goal itself* an entry point -- so the search returned a one-clip "curriculum"
      with zero cost. A curriculum has to start somewhere a fresh policy could actually
      start, not merely somewhere in the right neighbourhood.

    Falls back to the bottom quantile dataset-wide, then to the single easiest clip, so a
    start pool always exists.
    """
    if nodes.empty:
        return []
    difficulty = nodes["difficulty"].astype(float)
    values = difficulty.to_numpy()
    threshold = float(np.quantile(values, start_quantile))

    easy = values <= threshold
    means = difficulty.groupby(nodes["cluster"]).mean()
    if not means.empty:
        in_easiest_family = (nodes["cluster"] == means.idxmin()).to_numpy()
        clips = np.flatnonzero(easy & in_easiest_family)
        if clips.size:
            return [int(i) for i in clips]

    clips = np.flatnonzero(easy)
    if clips.size:
        return [int(i) for i in clips]
    return [int(values.argmin())]


def _repair_reachability(
    graph: nx.DiGraph,
    difficulty: np.ndarray,
    normalized: np.ndarray,
    values: Dict[str, np.ndarray],
    config: GraphConfig,
    start_clips: List[int],
    novelty_floor: float = 0.0,
) -> List[Tuple[object, int]]:
    """Add the fewest edges needed for ``START`` to reach every clip.

    k-NN thresholded by the difficulty rule can leave clips with no incoming edge from
    the reachable set. Each repair links such a clip to its closest reachable clip by
    DTW -- the least disruptive transition available.

    Crucially, a repair **still obeys the difficulty rule**: the source is chosen only
    among reachable clips that are no harder than the destination (within tolerance).
    Allowing repairs to ignore it would silently destroy the structural ramp guarantee --
    an early version did exactly that and produced a single edge dropping difficulty by
    0.75, which no amount of weight tuning could have prevented.

    When *no* reachable clip is easy enough to lead into any remaining one, those clips
    are not badly connected -- they are additional entry points, so the easiest of them
    is wired to ``START`` and joins the start pool.

    Args:
        start_clips: The zero-cost entry pool, appended to in place when a clip has to
            be attached directly to ``START``.

    Returns:
        ``(source, destination)`` per repair; ``source`` is :data:`START` for the
        entry-point case. Returned so they can be reported rather than quietly inflating
        apparent connectivity.
    """
    n = int(normalized.shape[0])
    tolerance = config.backslide_tolerance
    repairs: List[Tuple[object, int]] = []
    reachable = nx.descendants(graph, START)

    # Bounded by n: each iteration makes at least one more clip reachable.
    for _ in range(n):
        missing = [i for i in range(n) if i not in reachable]
        if not missing:
            break

        reached = sorted(reachable)
        best: Optional[Tuple[float, int, int]] = None
        for dst in missing:
            allowed = [s for s in reached if difficulty[s] <= difficulty[dst] + tolerance]
            if not allowed:
                continue
            costs = normalized[allowed, dst]
            cheapest = int(np.argmin(costs))
            candidate = (float(costs[cheapest]), allowed[cheapest], dst)
            if best is None or candidate[0] < best[0]:
                best = candidate

        if best is None:
            dst = min(missing, key=lambda i: difficulty[i])
            graph.add_edge(
                START, dst, weight=0.0, ramp_cost=0.0, interference_cost=0.0,
                redundancy_cost=0.0, dtw=0.0,
                difficulty_delta=0.0, task_switch=False, cluster_switch=False,
                embodiment_switch=False, source_switch=False, is_repair=True,
            )
            start_clips.append(int(dst))
            repairs.append((START, int(dst)))
        else:
            _, src, dst = best
            _add_directed_edge(
                graph, src, dst, difficulty, normalized, values, config, repair=True,
                novelty_floor=novelty_floor,
            )
            repairs.append((int(src), int(dst)))

        # Everything newly downstream of dst is reachable too.
        reachable.add(dst)
        reachable |= nx.descendants(graph, dst)
    return repairs


def build_clip_graph(
    distance_matrix: np.ndarray,
    node_frame: pd.DataFrame,
    config: Optional[GraphConfig] = None,
) -> ClipGraph:
    """Build the directed, dual-weighted clip graph.

    Args:
        distance_matrix: ``(N, N)`` DTW distances from :func:`~src.diversity_engine.compute_dtw_matrix`.
        node_frame: One row per clip **in the same order as the distance matrix** --
            :meth:`~src.pipeline.PipelineResult.frame` produces exactly this. Must carry
            ``episode_id``, ``difficulty`` and ``cluster``; ``task_name``, ``embodiment``
            and ``source`` are used for interference penalties when present.
        config: Edge weighting. Defaults to :class:`GraphConfig`.

    Returns:
        A :class:`ClipGraph` in which every clip is reachable from :data:`START`.

    Raises:
        ValueError: if the frame and matrix disagree on size, required columns are
            missing, or fewer than two clips are supplied.
    """
    config = config or GraphConfig()
    nodes = node_frame.reset_index(drop=True)
    n = len(nodes)

    if distance_matrix.shape != (n, n):
        raise ValueError(
            f"distance_matrix is {distance_matrix.shape} but node_frame has {n} rows; "
            "they must be in the same order"
        )
    missing_cols = [c for c in REQUIRED_NODE_COLUMNS if c not in nodes.columns]
    if missing_cols:
        raise ValueError(f"node_frame is missing required columns: {missing_cols}")
    if n < 2:
        raise ValueError(f"need at least 2 clips to build a graph, got {n}")

    # difficulty can arrive with NaN if the curriculum merge missed a clip; treat an
    # unknown clip as maximally hard rather than as trivially easy, which would let it
    # contaminate the start pool.
    difficulty = nodes["difficulty"].astype(float).fillna(1.0).to_numpy()
    normalized = normalize_distances(distance_matrix, config.dtw_quantile)
    values = _mismatch_map(nodes)

    # Derived from the data, at the same quantile `src.path_metrics` uses to *measure*
    # redundancy, so the edge cost penalises precisely what the metric reports.
    from .cluster_mapper import pairwise_values

    novelty_pairs = pairwise_values(normalized)
    novelty_floor = (
        float(np.quantile(novelty_pairs, config.novelty_quantile))
        if novelty_pairs.size and config.novelty_quantile > 0
        else 0.0
    )

    graph = nx.DiGraph()
    for i in range(n):
        row = nodes.iloc[i]
        graph.add_node(
            i,
            **{
                "episode_id": str(row["episode_id"]),
                "difficulty": float(difficulty[i]),
                "difficulty_z": float(row.get("difficulty_z", np.nan)),
                "cluster": int(row["cluster"]),
                "stage": int(row["stage"]) if not pd.isna(row.get("stage", np.nan)) else -1,
                "task_name": str(row.get("task_name", "unknown")),
                "embodiment": str(row.get("embodiment", "unknown")),
                "source": str(row.get("source", "unknown")),
                "n_frames": int(row.get("n_frames", 0) or 0),
            },
        )

    # Directed edges, keeping only those that do not run meaningfully downhill.
    for i, j in _candidate_pairs(normalized, config.k_neighbors):
        if difficulty[j] >= difficulty[i] - config.backslide_tolerance:
            _add_directed_edge(
                graph, i, j, difficulty, normalized, values, config,
                novelty_floor=novelty_floor,
            )
        if difficulty[i] >= difficulty[j] - config.backslide_tolerance:
            _add_directed_edge(
                graph, j, i, difficulty, normalized, values, config,
                novelty_floor=novelty_floor,
            )

    start_clips = _easy_start_clips(nodes, config.start_quantile)
    graph.add_node(START, episode_id=START, difficulty=0.0, cluster=-1, task_name=START)
    for i in start_clips:
        # Zero cost: entering the curriculum is free, so the search is free to choose
        # which easy clip to open with.
        graph.add_edge(
            START, i, weight=0.0, ramp_cost=0.0, interference_cost=0.0,
            redundancy_cost=0.0, dtw=0.0,
            difficulty_delta=0.0, task_switch=False, cluster_switch=False,
            embodiment_switch=False, source_switch=False, is_repair=False,
        )

    repairs = _repair_reachability(
        graph, difficulty, normalized, values, config, start_clips, novelty_floor
    )

    return ClipGraph(
        graph=graph,
        nodes=nodes,
        normalized_distance=normalized,
        start_clips=start_clips,
        repairs=repairs,
        config=config,
        novelty_floor=novelty_floor,
    )


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
def force_directed_layout(
    clip_graph: ClipGraph,
    embedding: Optional[np.ndarray] = None,
    seed: int = 42,
    iterations: int = 60,
) -> np.ndarray:
    """2-D force-directed positions for the clips, for drawing the graph.

    Seeded from the existing UMAP embedding when available. That matters for more than
    determinism: a cold spring layout of a k-NN graph converges to an arbitrary
    rotation each run, so the picture would reshuffle on every rerun and the user would
    lose their mental map. Starting from UMAP also means the drawn layout agrees with
    the diversity map in the Track 2 tabs.

    Args:
        clip_graph: The built graph. ``START`` is excluded from the layout.
        embedding: ``(N, >=2)`` UMAP coordinates in dataset order, or ``None``.
        seed: Spring-layout seed.
        iterations: Spring-layout iterations.

    Returns:
        ``(N, 2)`` float64 positions.
    """
    clips = [i for i in clip_graph.graph.nodes if i != START]
    undirected = clip_graph.graph.subgraph(clips).to_undirected()
    n = clip_graph.n_clips

    init: Optional[Dict[int, np.ndarray]] = None
    if embedding is not None and len(embedding) == n and np.asarray(embedding).shape[1] >= 2:
        coords = np.asarray(embedding, dtype=np.float64)[:, :2].copy()
        span = np.ptp(coords, axis=0)
        span[span < _EPS] = 1.0
        # spring_layout expects roughly unit-scale positions.
        coords = 2.0 * (coords - coords.min(axis=0)) / span - 1.0
        init = {i: coords[i] for i in range(n)}

    positions = nx.spring_layout(
        undirected, pos=init, seed=seed, iterations=iterations, weight=None
    )
    out = np.zeros((n, 2), dtype=np.float64)
    for i, pos in positions.items():
        out[int(i)] = pos
    return out
