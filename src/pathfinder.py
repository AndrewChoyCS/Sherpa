"""Shortest-path search over the clip graph, plus rehearsal ("review") insertion.

Given a target clip, the curriculum is the cheapest route to it from the virtual
``START``. Because every edge cost in :mod:`src.graph` is non-negative, Dijkstra is
exact here and needs no relaxation of negative cycles.

**Why the cheapest path is the right curriculum.** The edge cost is
``difficulty-ramp + interference + redundancy + step``. Minimising its sum over a route to
the target simultaneously minimises total ramp roughness, total cross-task/embodiment
disruption, wasted near-duplicate steps, and curriculum length. There is no separate
objective to reconcile -- the ordering problem is the routing problem.

**Review nodes.** Reaching the target is not enough: a policy that walked a long path
away from its early skills has probably forgotten them. Rehearsal is the standard
mitigation, so every ``review_every`` clips the path revisits one clip it has already
seen. The pick is deliberate rather than random: the skill family absent longest, and
within it the clip *farthest* from where the curriculum currently is under DTW -- i.e.
the material most exposed to interference from recent training. Cluster medoids win ties,
being the most representative clip of their family.

This is a cheap **proxy** for rehearsal-based anti-forgetting, not a validated
intervention. It reproduces the structure of experience replay (interleave old samples,
prioritise the most-forgotten) without any evidence here that it improves a real training
run -- measuring that would need the training experiment this build does not include.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .graph import START, ClipGraph, edge_cost

SEARCH_METHODS = ("dijkstra", "astar")


@dataclass(frozen=True)
class PathConfig:
    """Search and rehearsal settings.

    Attributes:
        review_every: Insert a rehearsal clip after every this many newly-introduced
            clips. ``0`` disables review insertion entirely.
        search: ``"dijkstra"`` (exact) or ``"astar"``. See :func:`find_curriculum_path`
            for why Dijkstra is the default.
        max_reviews: Hard cap on inserted rehearsal clips, so a very long path cannot
            become mostly review.
    """

    review_every: int = 4
    search: str = "dijkstra"
    max_reviews: int = 12

    def __post_init__(self) -> None:
        if self.search not in SEARCH_METHODS:
            raise ValueError(f"search must be one of {SEARCH_METHODS}, got {self.search!r}")
        if self.review_every < 0:
            raise ValueError("review_every must be >= 0")


@dataclass
class CurriculumPath:
    """An ordered curriculum: the searched route plus any inserted rehearsal clips.

    Attributes:
        clips: Clip indices in training order, including review repeats.
        is_review: Parallel flags; ``True`` marks a rehearsal repeat of an earlier clip.
        table: One row per training step, with metadata and the per-transition cost
            breakdown -- this is what the dashboard renders and the CLI writes to CSV.
        route: The raw searched route (no reviews, no ``START``).
        target_index: The clip the search was aimed at.
        search_cost: Total edge weight along the searched route, from the search itself.
        cost_terms: Searched-route totals split into ``ramp``, ``interference``,
            ``redundancy`` and ``step``.
        method: Which search actually ran.
    """

    clips: List[int]
    is_review: List[bool]
    table: pd.DataFrame
    route: List[int]
    target_index: int
    search_cost: float
    cost_terms: Dict[str, float] = field(default_factory=dict)
    method: str = "dijkstra"

    @property
    def n_steps(self) -> int:
        return len(self.clips)

    @property
    def n_reviews(self) -> int:
        return int(sum(self.is_review))

    @property
    def unique_clips(self) -> List[int]:
        """Distinct clips in first-appearance order -- the actual data subset selected."""
        seen: List[int] = []
        for clip in self.clips:
            if clip not in seen:
                seen.append(clip)
        return seen

    def summary(self) -> str:
        return (
            f"CurriculumPath({len(self.route)} clips + {self.n_reviews} reviews, "
            f"cost={self.search_cost:.2f}, method={self.method})"
        )


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def _astar_heuristic(clip_graph: ClipGraph, target: int):
    """Remaining-interference estimate: normalised DTW from a node to the target.

    Not a provable lower bound on the true remaining cost -- a route can accumulate ramp
    and step costs this ignores, but it can also *overestimate* when the direct DTW
    distance exceeds the cost of a smooth multi-hop route, so A* here is
    epsilon-approximate rather than optimal. It is offered because the pitch names A*;
    Dijkstra stays the default because at a few hundred nodes it is already
    sub-millisecond, making the approximation a pure downside.
    """
    normalized = clip_graph.normalized_distance
    weight = clip_graph.config.w_interference

    def heuristic(node, _goal) -> float:
        if node == START:
            return 0.0
        return float(weight * normalized[int(node), target])

    return heuristic


def search_route(
    clip_graph: ClipGraph, target_index: int, method: str = "dijkstra"
) -> Tuple[List[int], float]:
    """Cheapest route from ``START`` to ``target_index``.

    Returns:
        ``(clip_indices, total_weight)`` with ``START`` stripped off the front.

    Raises:
        ValueError: if the target is not a clip in the graph.
        networkx.NetworkXNoPath: if the target is unreachable, which
            :func:`~src.graph.build_clip_graph`'s reachability repair should prevent.
    """
    graph = clip_graph.graph
    if target_index not in graph or target_index == START:
        raise ValueError(f"target_index {target_index!r} is not a clip in the graph")

    if method == "astar":
        route = nx.astar_path(
            graph, START, target_index,
            heuristic=_astar_heuristic(clip_graph, target_index), weight="weight",
        )
    else:
        route = nx.dijkstra_path(graph, START, target_index, weight="weight")

    total = sum(
        float(graph[u][v]["weight"]) for u, v in zip(route[:-1], route[1:])
    )
    return [int(node) for node in route if node != START], float(total)


# --------------------------------------------------------------------------- #
# rehearsal insertion
# --------------------------------------------------------------------------- #
def _pick_review(
    candidates: Sequence[int],
    current: int,
    positions: Dict[int, int],
    clip_graph: ClipGraph,
) -> Optional[int]:
    """Choose which already-seen clip to rehearse.

    Args:
        candidates: Clips already introduced, excluding the one just placed.
        current: The clip the curriculum has just reached.
        positions: Clip -> most recent position in the emitted sequence, used to find
            which skill family has gone longest without being touched.
        clip_graph: Supplies clusters, medoid flags and DTW distances.

    Returns:
        The clip to rehearse, or ``None`` when nothing is eligible yet.
    """
    if not candidates:
        return None

    nodes = clip_graph.nodes
    normalized = clip_graph.normalized_distance
    clusters = nodes["cluster"].to_numpy()
    medoids = (
        nodes["is_cluster_medoid"].fillna(False).to_numpy()
        if "is_cluster_medoid" in nodes.columns
        else np.zeros(len(nodes), dtype=bool)
    )

    # Most-neglected skill family: the one whose latest appearance is furthest back.
    last_seen: Dict[int, int] = {}
    for clip in candidates:
        cluster = int(clusters[clip])
        last_seen[cluster] = max(last_seen.get(cluster, -1), positions.get(clip, -1))
    stalest = min(last_seen, key=lambda c: last_seen[c])

    pool = [c for c in candidates if int(clusters[c]) == stalest]
    # Within the family: farthest from the current clip (most interference exposure),
    # with medoids winning ties as the family's most representative sample.
    return max(pool, key=lambda c: (float(normalized[current, c]), bool(medoids[c]), -c))


def insert_reviews(
    route: Sequence[int], clip_graph: ClipGraph, config: PathConfig
) -> Tuple[List[int], List[bool]]:
    """Interleave rehearsal repeats into a searched route.

    Returns:
        ``(clips, is_review)`` where ``clips`` may repeat earlier entries. Review clips
        are always drawn from clips already introduced *earlier in the same sequence*,
        never from unseen ones -- a rehearsal of unseen material would not be rehearsal.
    """
    clips: List[int] = []
    is_review: List[bool] = []
    if not route:
        return clips, is_review
    if config.review_every <= 0:
        return [int(c) for c in route], [False] * len(route)

    introduced: List[int] = []
    positions: Dict[int, int] = {}
    n_reviews = 0

    for count, clip in enumerate(route, start=1):
        clip = int(clip)
        clips.append(clip)
        is_review.append(False)
        positions[clip] = len(clips) - 1
        if clip not in introduced:
            introduced.append(clip)

        # Never review after the last clip: the curriculum has to culminate on the goal,
        # not trail off into rehearsal.
        due = count % config.review_every == 0 and count < len(route)
        if not due or n_reviews >= config.max_reviews:
            continue
        # Rehearsing the clip we just trained on would be a no-op, so exclude it.
        pick = _pick_review([c for c in introduced if c != clip], clip, positions, clip_graph)
        if pick is None:
            continue
        clips.append(pick)
        is_review.append(True)
        positions[pick] = len(clips) - 1
        n_reviews += 1

    return clips, is_review


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def _transition_costs(
    clip_graph: ClipGraph, u: int, v: int
) -> Dict[str, float]:
    """Cost breakdown for training ``v`` after ``u``, whether or not that edge exists.

    Review insertions create transitions the k-NN graph never proposed. Scoring them
    with the same cost function keeps the table's columns comparable across searched and
    inserted steps; ``on_graph_edge`` distinguishes the two.
    """
    graph = clip_graph.graph
    nodes = clip_graph.nodes
    on_edge = graph.has_edge(u, v)

    fields = {}
    for column, key in (
        ("task_name", "task_switch"),
        ("cluster", "cluster_switch"),
        ("embodiment", "embodiment_switch"),
        ("source", "source_switch"),
    ):
        if column in nodes.columns:
            fields[key] = bool(nodes[column].iloc[u] != nodes[column].iloc[v])
        else:
            fields[key] = False

    difficulty = nodes["difficulty"].astype(float).fillna(1.0).to_numpy()
    cost = edge_cost(
        float(difficulty[u]),
        float(difficulty[v]),
        float(clip_graph.normalized_distance[u, v]),
        {
            "task_name": fields["task_switch"],
            "cluster": fields["cluster_switch"],
            "embodiment": fields["embodiment_switch"],
            "source": fields["source_switch"],
        },
        clip_graph.config,
        clip_graph.novelty_floor,
    )
    return {
        "edge_weight": cost["weight"],
        "ramp_cost": cost["ramp"],
        "interference_cost": cost["interference"],
        "redundancy_cost": cost["redundancy"],
        "dtw_from_prev": float(clip_graph.normalized_distance[u, v]),
        "difficulty_delta": float(difficulty[v] - difficulty[u]),
        "on_graph_edge": on_edge,
        **fields,
    }


def _build_table(
    clip_graph: ClipGraph, clips: Sequence[int], is_review: Sequence[bool]
) -> pd.DataFrame:
    """One row per training step, with metadata and per-transition costs."""
    nodes = clip_graph.nodes
    first_seen: Dict[int, int] = {}
    rows: List[Dict[str, object]] = []

    for step, (clip, review) in enumerate(zip(clips, is_review), start=1):
        row: Dict[str, object] = {
            "step": step,
            "clip_index": int(clip),
            "episode_id": str(nodes["episode_id"].iloc[clip]),
            "is_review": bool(review),
            "reviews_step": first_seen.get(int(clip)) if review else None,
        }
        for column in (
            "task_name", "task_description", "stage", "cluster", "difficulty",
            "difficulty_z", "embodiment", "source", "n_frames",
        ):
            if column in nodes.columns:
                row[column] = nodes[column].iloc[clip]
        if step == 1:
            row.update(
                edge_weight=np.nan, ramp_cost=np.nan, interference_cost=np.nan,
                redundancy_cost=np.nan,
                dtw_from_prev=np.nan, difficulty_delta=np.nan, on_graph_edge=True,
                task_switch=False, cluster_switch=False, embodiment_switch=False,
                source_switch=False,
            )
        else:
            costs = _transition_costs(clip_graph, int(clips[step - 2]), int(clip))
            if review:
                # A rehearsal step is *meant* to drop back in difficulty, so its ramp
                # cost -- and therefore the composite weight -- has no meaning and would
                # read as a huge violation. Blanked rather than reported. The interference
                # and DTW columns stay, because those costs are real: interleaving old
                # material genuinely does switch context.
                costs["ramp_cost"] = np.nan
                costs["edge_weight"] = np.nan
            row.update(costs)
        rows.append(row)
        first_seen.setdefault(int(clip), step)

    return pd.DataFrame(rows)


def find_curriculum_path(
    clip_graph: ClipGraph,
    target_index: int,
    config: Optional[PathConfig] = None,
) -> CurriculumPath:
    """Find and assemble the curriculum path to ``target_index``.

    Args:
        clip_graph: Built graph from :func:`~src.graph.build_clip_graph`.
        target_index: Clip to route to, e.g. from
            :meth:`~src.goal_matcher.GoalMatcher.match`.
        config: Search and rehearsal settings.

    Returns:
        A :class:`CurriculumPath` whose ``table`` is ready to display or write out.
    """
    config = config or PathConfig()
    route, search_cost = search_route(clip_graph, int(target_index), config.search)
    clips, is_review = insert_reviews(route, clip_graph, config)
    table = _build_table(clip_graph, clips, is_review)

    # Split the searched cost into its terms, using the graph's own edge attributes so
    # this reports what the search actually optimised.
    graph = clip_graph.graph
    terms = {"ramp": 0.0, "interference": 0.0, "redundancy": 0.0, "step": 0.0}
    sequence = [START] + list(route)
    for u, v in zip(sequence[:-1], sequence[1:]):
        data = graph[u][v]
        terms["ramp"] += clip_graph.config.w_difficulty * float(data["ramp_cost"])
        terms["interference"] += clip_graph.config.w_interference * float(
            data["interference_cost"]
        )
        terms["redundancy"] += clip_graph.config.w_redundancy * float(
            data.get("redundancy_cost", 0.0)
        )
        if u != START:
            terms["step"] += float(clip_graph.config.step_penalty)

    return CurriculumPath(
        clips=[int(c) for c in clips],
        is_review=[bool(r) for r in is_review],
        table=table,
        route=[int(c) for c in route],
        target_index=int(target_index),
        search_cost=float(search_cost),
        cost_terms=terms,
        method=config.search,
    )
