"""FastAPI wrapper over the EgoVerse curriculum pipeline.

Run it:
    uvicorn server.api:app --port 8000 --reload

Ports 8501/8502 are left alone -- the Streamlit dashboard uses those.

**Layering.** The expensive stage is loading plus the ``O(N^2 T^2)`` DTW matrix;
graph construction, goal matching and Dijkstra are milliseconds. Those are cached
separately, exactly as ``app.py`` does with ``st.cache_data`` /
``st.cache_resource``, so dragging an interference weight in the browser rebuilds
only the graph and never re-runs DTW. The DTW matrix itself is additionally
cached to ``.cache/`` by content hash inside :mod:`src.diversity_engine`, so even
a cold server re-uses whatever the CLI already computed.

**No numeric logic lives here.** Every endpoint delegates to :mod:`src`, so a
number in the browser and the same number in ``reports/metrics.json`` come from
one code path and cannot drift.
"""

from __future__ import annotations

import warnings
from pathlib import Path
import inspect
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from find_path import DOMAIN_PRESETS
from src.cluster_mapper import find_redundant_pairs
from src.graph import START, GraphConfig
from src.path_metrics import compare_orderings, coverage_curve, path_report
from src.pathfinder import PathConfig
from src.pipeline import PathFinderContext, PipelineResult, build_path_finder, run_pipeline

from .serialize import decimate, frame_records, jsonable

warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="EgoVerse Curriculum Engine",
    description="JSON surface over the trajectory diversity and curriculum path finder.",
    version="0.1.0",
)

# The Vite dev server is a different origin (5173 -> 8000). In production the
# built frontend is served from this same app, where CORS is irrelevant.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# request models
# --------------------------------------------------------------------------- #
# Pulled from the function signature rather than restated here. Restating them
# means the API silently pins whatever the defaults were on the day it was written:
# when `normalize` changed from "center" to "zscore" upstream, a hardcoded default
# would have kept forcing the old value and quietly served the wrong numbers.
PIPELINE_DEFAULTS: Dict[str, object] = {
    name: parameter.default
    for name, parameter in inspect.signature(run_pipeline).parameters.items()
    if parameter.default is not inspect.Parameter.empty
}

# Settings a caller may override. Anything else stays at the pipeline's own value.
TUNABLE = (
    "data_dir", "arm", "min_length", "normalize", "max_length", "length_normalize",
    "sakoe_chiba_radius", "linkage", "difficulty_scaling", "n_clusters",
)


class RunRequest(BaseModel):
    """Ingestion and DTW settings -- the parameters that force a pipeline re-run.

    Every field defaults to ``None`` meaning *use the pipeline's own default*, so
    this layer never pins a value the pipeline has since changed.
    """

    data_dir: Optional[str] = None
    arm: Optional[str] = None
    min_length: Optional[int] = None
    normalize: Optional[str] = None
    max_length: Optional[int] = None
    length_normalize: Optional[bool] = None
    sakoe_chiba_radius: Optional[int] = None
    linkage: Optional[str] = None
    difficulty_scaling: Optional[str] = None
    n_clusters: Optional[int] = None

    def effective(self) -> Dict[str, object]:
        """Resolved settings: caller overrides on top of the pipeline's defaults.

        ``n_clusters=None`` is meaningful to the pipeline (choose k by silhouette),
        so it is passed through rather than treated as "unset".
        """
        resolved = {name: PIPELINE_DEFAULTS.get(name) for name in TUNABLE}
        for name in TUNABLE:
            value = getattr(self, name)
            if value is not None:
                resolved[name] = value
        return resolved

    def key(self) -> Tuple:
        """Cache key over the resolved settings, not the raw request."""
        effective = self.effective()
        return tuple(effective[name] for name in TUNABLE)


# Same reasoning as PIPELINE_DEFAULTS: read the dataclass rather than restate it, so
# a new weight or a changed default arrives here without an edit.
GRAPH_FIELDS = tuple(GraphConfig.__dataclass_fields__)


class GraphRequest(RunRequest):
    """Adds edge weighting and task scoping -- cheap to change, no DTW re-run.

    Any ``GraphConfig`` field may be overridden by name; unset fields keep the
    dataclass default. ``w_redundancy`` and ``novelty_quantile`` matter as much as
    the interference weight: without the redundancy term, minimising interference
    means minimising distance, and the cheapest route is the same clip repeatedly.
    """

    graph: Dict[str, float] = Field(default_factory=dict)
    tasks: Optional[List[str]] = None
    # A curated task group from find_path.py. Imported rather than restated so the
    # browser demo and `--domain` on the CLI can never drift apart.
    domain: Optional[str] = None

    def scope(self, present: Optional[set] = None) -> Optional[List[str]]:
        """Resolve the task scope: explicit `tasks` wins, else the domain preset.

        A preset is intersected with the tasks the dataset actually contains, so it
        never fails outright just because one of its tasks was not sampled -- the same
        rule `find_path.py` applies.
        """
        if self.tasks:
            return list(self.tasks)
        if not self.domain:
            return None
        if self.domain not in DOMAIN_PRESETS:
            raise HTTPException(
                status_code=422,
                detail=f"unknown domain {self.domain!r}; allowed: {sorted(DOMAIN_PRESETS)}",
            )
        preset = list(DOMAIN_PRESETS[self.domain])
        if present is None:
            return preset
        scoped = [task for task in preset if task in present]
        # Falling back to the whole graph beats erroring: the preset is a
        # convenience, not a constraint the caller asked to be enforced.
        return scoped or None

    def graph_config(self) -> GraphConfig:
        overrides = {k: v for k, v in self.graph.items() if k in GRAPH_FIELDS}
        unknown = set(self.graph) - set(GRAPH_FIELDS)
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"unknown graph settings {sorted(unknown)}; allowed: {list(GRAPH_FIELDS)}",
            )
        try:
            return GraphConfig(**overrides)  # type: ignore[arg-type]
        except (ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    def graph_key(self) -> Tuple:
        return (
            self.key(),
            tuple(sorted((k, v) for k, v in self.graph.items() if k in GRAPH_FIELDS)),
            tuple(self.tasks or ()),
            self.domain or "",
        )


class PathRequest(GraphRequest):
    """Adds the goal query and rehearsal settings."""

    goal: str = Field(default="teach the robot to fold a shirt", min_length=1)
    review_every: int = 4
    max_reviews: int = 12
    search: str = "dijkstra"
    target_selection: str = "hardest"
    target_index: Optional[int] = None
    # Baseline count. 50 matches find_path.py's default so the browser and the
    # CLI report the same comparison numbers.
    seeds: int = 50


# --------------------------------------------------------------------------- #
# caches
# --------------------------------------------------------------------------- #
_PIPELINE_CACHE: Dict[Tuple, PipelineResult] = {}
_CONTEXT_CACHE: Dict[Tuple, PathFinderContext] = {}
_CACHE_LIMIT = 8


def _evict(cache: Dict) -> None:
    """Bound cache growth. Insertion-ordered dicts make this FIFO."""
    while len(cache) > _CACHE_LIMIT:
        cache.pop(next(iter(cache)))


def _pipeline(request: RunRequest) -> PipelineResult:
    key = request.key()
    if key not in _PIPELINE_CACHE:
        result = run_pipeline(verbose=False, **request.effective())  # type: ignore[arg-type]
        if result.n_episodes < 2:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Only {result.n_episodes} usable episode(s) in "
                    f"{request.effective()['data_dir']!r}; "
                    "at least 2 are needed for a pairwise distance matrix. Fetch episodes "
                    "with scripts/fetch_egoverse_data.py, or generate synthetic ones with "
                    "scripts/generate_synthetic_data.py."
                ),
            )
        _PIPELINE_CACHE[key] = result
        _evict(_PIPELINE_CACHE)
    return _PIPELINE_CACHE[key]


def _context(request: GraphRequest) -> PathFinderContext:
    key = request.graph_key()
    if key not in _CONTEXT_CACHE:
        result = _pipeline(request)
        # The scope is intersected with the tasks this dataset actually has, which
        # needs the loaded frame -- hence resolving it here rather than on the model.
        present = set(result.frame()["task_name"].astype(str))
        try:
            _CONTEXT_CACHE[key] = build_path_finder(
                result, request.graph_config(), task_names=request.scope(present)
            )
        except ValueError as error:  # empty task scope, or <2 clips
            raise HTTPException(status_code=422, detail=str(error)) from error
        _evict(_CONTEXT_CACHE)
    return _CONTEXT_CACHE[key]


# --------------------------------------------------------------------------- #
# snapshot
# --------------------------------------------------------------------------- #
def _trajectory_previews(result: PipelineResult) -> List[dict]:
    """Decimated XY polylines for the hero plot, one per episode.

    Centred per episode but **not** rescaled per episode: extent is signal here.
    Under the pipeline's default ``center`` normalisation a 5 cm nudge and a 50 cm
    sweep stay far apart because they are different skills, so scaling each path
    to fill its own box would erase exactly what the plot is meant to show. The
    frontend applies one shared scale across all 28.
    """
    ds = result.dataset
    # workspace_span and path_length live on the curriculum table, keyed by id.
    extra: Dict[str, Dict[str, float]] = {}
    if not result.curriculum.empty:
        columns = [c for c in ("workspace_span", "path_length") if c in result.curriculum.columns]
        if columns:
            extra = (
                result.curriculum.set_index("episode_id")[columns]
                .astype(float)
                .to_dict(orient="index")
            )

    # Decimate harder as the dataset grows: 273 episodes at 240 points each is a
    # multi-megabyte first paint for strokes a few hundred pixels wide. The hero
    # only plots a bounded subset anyway.
    max_points = 240 if len(ds.trajectories) <= 60 else 120

    previews: List[dict] = []
    for i, trajectory in enumerate(ds.trajectories):
        points = decimate(np.asarray(trajectory, dtype=float)[:, :2], max_points)
        centre = points.mean(axis=0)
        centred = points - centre
        metadata = ds.metadata[i] if i < len(ds.metadata) else {}
        episode_id = ds.episode_ids[i]
        stats = extra.get(episode_id, {})
        previews.append(
            {
                "episode_id": episode_id,
                "source": str(metadata.get("source", "unknown")),
                "task_name": str(metadata.get("task_name", "unknown")),
                "n_frames": int(len(trajectory)),
                "fps": float(metadata.get("fps") or 30.0),
                "span": float(stats.get("workspace_span", float(np.abs(centred).max()))),
                "path_length": float(stats.get("path_length", 0.0)),
                "points": [[float(x), float(y)] for x, y in centred],
            }
        )
    return previews


def _agreement_matrix(result: PipelineResult) -> dict:
    """Cluster x task_name counts.

    This is what makes the ARI claim legible instead of asserted: a reviewer can
    see which unsupervised group corresponds to which human task label, and where
    it disagrees, rather than being handed a single number to trust.
    """
    frame = result.frame()
    if "task_name" not in frame.columns:
        return {"clusters": [], "tasks": [], "counts": [], "excluded": 0}

    # Unlabelled episodes are excluded, matching what the ARI is computed over. If they
    # were kept, "unknown" would appear as a task label spanning every group and the
    # picture would contradict the number printed beside it -- the table would show a
    # conflated clustering while the ARI reported a clean one.
    labels = frame["task_name"].astype(str)
    labelled = frame[~labels.str.lower().isin(("unknown", "nan", ""))]
    excluded = len(frame) - len(labelled)
    if labelled.empty:
        return {"clusters": [], "tasks": [], "counts": [], "excluded": excluded}

    table = pd.crosstab(labelled["cluster"], labelled["task_name"])
    return {
        "clusters": [int(c) for c in table.index.tolist()],
        "tasks": [str(t) for t in table.columns.tolist()],
        "counts": [[int(v) for v in row] for row in table.to_numpy()],
        "excluded": excluded,
    }


def _snapshot(request: RunRequest) -> dict:
    result = _pipeline(request)
    frame = result.frame()
    return jsonable(
        {
            # The resolved settings actually used, not the raw request, so the UI
            # displays what ran rather than what was asked for.
            "config": request.effective(),
            "diversity_metrics": result.report,
            "agreement": result.agreement,
            # How many episodes each ARI was computed over. Essential context: an ARI
            # scored on a small cleanly-labelled subset is not comparable to one
            # scored on a large partly-unlabelled set, and 78 of the current episodes
            # carry no task_name at all.
            "agreement_support": result.agreement_support,
            "n_clusters": int(len(set(result.labels.tolist()))),
            "suggested_k": result.suggested_k,
            "silhouette_by_k": result.silhouette_by_k,
            "n_episodes": result.n_episodes,
            "n_skipped": len(result.dataset.skipped),
            "episodes": frame_records(frame),
            "stages": frame_records(result.stages),
            "skipped": [
                {"episode_id": episode_id, "reason": reason}
                for episode_id, reason in result.dataset.skipped
            ],
            "trajectories": _trajectory_previews(result),
            "sources": sorted(frame["source"].astype(str).unique().tolist()),
            "tasks": sorted(frame["task_name"].astype(str).unique().tolist()),
            "agreement_matrix": _agreement_matrix(result),
            "comparison": _comparison(result),
        }
    )


def _comparison(result) -> dict:
    """Head-to-head subset scores, the budget curve and the random null model.

    Embedded in the snapshot rather than fetched separately so the static export --
    the version that runs with no backend -- still shows the ranking result.
    """
    from src.compare import compare_subsets, selection_curve

    total = result.n_episodes
    if total < 8:
        return {}

    size = max(2, total // 4)
    comparison = compare_subsets(
        result.distance_matrix,
        methods=("coreset", "random", "stratified", "redundant"),
        subset_size=size,
        labels=result.labels,
        tasks=result.dataset.task_labels,
        sources=result.dataset.field_values("source"),
    )
    curve = selection_curve(
        result.distance_matrix,
        methods=("coreset", "stratified", "random", "redundant"),
        labels=result.labels,
    )
    episode_ids = result.dataset.episode_ids
    return {
        "subset_size": comparison.subset_size,
        "n_tasks_total": len(set(result.dataset.task_labels)),
        "subsets": [
            {
                "name": subset.name,
                "metrics": subset.metrics,
                "episode_ids": [episode_ids[i] for i in subset.indices],
            }
            for subset in comparison.subsets
        ],
        "deltas": frame_records(comparison.deltas()),
        "curve": frame_records(curve),
        "baseline": comparison.baseline or {},
        # Down-sampled for the histogram; the full 200 samples are not needed to draw it.
        "baseline_samples": (
            [float(v) for v in comparison.baseline_samples]
            if comparison.baseline_samples is not None
            else []
        ),
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "cached_pipelines": len(_PIPELINE_CACHE)}


@app.get("/api/domains")
def domains(data_dir: Optional[str] = None) -> dict:
    """The curated task groups, each narrowed to what this dataset actually holds.

    Reported with the clip count per domain so the UI can say how large a scoped
    graph is before the reader commits to running it.
    """
    result = _pipeline(RunRequest(data_dir=data_dir))
    frame = result.frame()
    counts = frame["task_name"].astype(str).value_counts().to_dict()
    return jsonable(
        {
            name: {
                "tasks": [task for task in tasks if task in counts],
                "n_clips": sum(counts.get(task, 0) for task in tasks),
            }
            for name, tasks in DOMAIN_PRESETS.items()
        }
    )


@app.get("/api/snapshot")
def snapshot() -> dict:
    """The default-config run. Same defaults as ``run_pipeline.py``."""
    return _snapshot(RunRequest())


@app.post("/api/run")
def run(request: RunRequest) -> dict:
    """Re-run with different ingestion/DTW settings. Slow only on a cache miss."""
    return _snapshot(request)


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #
@app.post("/api/graph")
def graph(request: GraphRequest) -> dict:
    """Nodes, edges and layout for the clip graph.

    Node/edge data is read off ``clip_graph.graph`` and ``ctx.layout`` directly
    rather than out of ``src.graph_plot.path_graph_figure``, which returns a
    Plotly figure -- the frontend draws its own SVG and has no use for a figure
    spec.
    """
    context = _context(request)
    clip_graph = context.clip_graph
    nodes = clip_graph.nodes
    layout = context.layout

    node_payload = []
    for i in range(len(nodes)):
        attributes = clip_graph.graph.nodes[i]
        node_payload.append(
            {
                "index": i,
                "episode_id": attributes["episode_id"],
                "x": float(layout[i, 0]),
                "y": float(layout[i, 1]),
                "difficulty": attributes["difficulty"],
                "cluster": attributes["cluster"],
                "stage": attributes.get("stage", -1),
                "task_name": attributes.get("task_name", "unknown"),
                "source": attributes.get("source", "unknown"),
                "embodiment": attributes.get("embodiment", "unknown"),
                "n_frames": attributes.get("n_frames", 0),
            }
        )

    edge_payload = []
    for u, v, data in clip_graph.graph.edges(data=True):
        # START is a virtual node with no layout position; its zero-cost entry
        # edges are reported as `start_clips` instead of drawn.
        if u == START or v == START:
            continue
        edge_payload.append(
            {
                "from": int(u),
                "to": int(v),
                "weight": data.get("weight"),
                "ramp": data.get("ramp_cost"),
                "interference": data.get("interference_cost"),
                # The near-duplicate penalty. Without it the cheapest route is the
                # same clip over and over, so it is part of the cost breakdown.
                "redundancy": data.get("redundancy_cost"),
                "dtw": data.get("dtw"),
                "is_repair": bool(data.get("is_repair", False)),
            }
        )

    n_edges = len(edge_payload)
    return jsonable(
        {
            "nodes": node_payload,
            "edges": edge_payload,
            "start_clips": clip_graph.start_clips,
            "repairs": [[str(a), int(b)] for a, b in clip_graph.repairs],
            "n_edges": n_edges,
            "mean_out_degree": n_edges / max(len(nodes), 1),
        }
    )


# --------------------------------------------------------------------------- #
# path
# --------------------------------------------------------------------------- #
@app.post("/api/path")
def path(request: PathRequest) -> dict:
    """Resolve a plain-English goal and return the curriculum route to it."""
    context = _context(request)
    try:
        path_config = PathConfig(
            review_every=request.review_every,
            search=request.search,
            max_reviews=request.max_reviews,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    try:
        match, curriculum_path = context.find(
            request.goal,
            path_config,
            target_selection=request.target_selection,
            target_index=request.target_index,
        )
    except (ValueError, KeyError, IndexError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    report = path_report(
        curriculum_path.clips,
        context.clip_graph,
        context.distance_matrix,
        target_index=curriculum_path.target_index,
        is_review=curriculum_path.is_review,
    )
    comparison = compare_orderings(
        curriculum_path,
        context.clip_graph,
        context.distance_matrix,
        n_seeds=request.seeds,
    )
    curve = coverage_curve(curriculum_path.clips, context.clip_graph.nodes)

    return jsonable(
        {
            "match": {
                "query": match.query,
                "target_index": match.target_index,
                "task_name": match.task_name,
                "score": match.score,
                "margin": match.margin,
                "is_confident": match.is_confident,
                "note": match.note,
                "candidates": [
                    {
                        "task_name": c.task_name,
                        "score": c.score,
                        "clip_index": c.clip_index,
                        "episode_id": c.episode_id,
                        "task_description": c.task_description,
                        "n_clips": c.n_clips,
                    }
                    for c in match.candidates
                ],
            },
            "steps": frame_records(curriculum_path.table),
            "route": [int(c) for c in curriculum_path.route],
            "target_index": curriculum_path.target_index,
            "search_cost": curriculum_path.search_cost,
            "cost_terms": curriculum_path.cost_terms,
            "method": curriculum_path.method,
            "n_reviews": curriculum_path.n_reviews,
            "report": report,
            "comparison": {
                str(index): {str(k): v for k, v in row.items()}
                for index, row in comparison.to_dict(orient="index").items()
            },
            "coverage_curve": [int(v) for v in curve],
        }
    )


# --------------------------------------------------------------------------- #
# supporting views
# --------------------------------------------------------------------------- #
@app.post("/api/redundancy")
def redundancy(request: RunRequest) -> List[dict]:
    """Near-duplicate episode pairs, closest first."""
    result = _pipeline(request)
    ids = result.dataset.episode_ids
    pairs = find_redundant_pairs(result.distance_matrix)
    return jsonable(
        [{"a": ids[i], "b": ids[j], "distance": float(d)} for i, j, d in pairs]
    )


@app.post("/api/matrix")
def matrix(request: RunRequest) -> Response:
    """The DTW matrix as raw float32, row-major.

    Binary rather than JSON because this is the one payload that grows
    quadratically -- 4.2 GB at the full 23k-episode dataset. The frontend drops
    the buffer straight into a canvas with no parse step.
    """
    result = _pipeline(request)
    values = np.ascontiguousarray(result.distance_matrix, dtype=np.float32)
    return Response(
        content=values.tobytes(),
        media_type="application/octet-stream",
        headers={
            "x-matrix-n": str(values.shape[0]),
            "x-episode-order": ",".join(result.dataset.episode_ids),
        },
    )


@app.get("/api/trajectory/{episode_id}")
def trajectory(episode_id: str, data_dir: Optional[str] = None) -> dict:
    """Full XYZ polyline for one episode, for the inspector."""
    result = _pipeline(RunRequest(data_dir=data_dir))
    try:
        index = result.dataset.episode_ids.index(episode_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=f"unknown episode {episode_id!r}") from error

    points = np.asarray(result.dataset.trajectories[index], dtype=float)
    metadata = result.dataset.metadata[index]
    return jsonable(
        {
            "episode_id": episode_id,
            "fps": float(metadata.get("fps") or 30.0),
            "arm_used": metadata.get("arm_used"),
            "unit_scale": metadata.get("unit_scale"),
            "missing_frame_ratio": metadata.get("missing_frame_ratio"),
            "points": [[float(v) for v in row[:3]] for row in decimate(points, 1200)],
        }
    )


# --------------------------------------------------------------------------- #
# built frontend
# --------------------------------------------------------------------------- #
# Mounted last so it cannot shadow /api. Absent during development, when Vite
# serves the frontend on 5173 and proxies /api here.
_DIST = REPO_ROOT / "web" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="web")


def build_snapshot(data_dir: Optional[str] = None) -> dict:
    """Snapshot payload without going through HTTP. Used by the export script."""
    return _snapshot(RunRequest(data_dir=data_dir))


__all__ = ["app", "build_snapshot"]
