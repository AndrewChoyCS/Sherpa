"""Interactive Streamlit dashboard for the EgoVerse curriculum engine.

Tab 1-2 are Track 1 (the curriculum path finder); tabs 3-8 are Track 2 (trajectory
diversity scoring). Both are driven from one pipeline run, so the graph's edge
weights and the diversity metrics are derived from the same DTW matrix.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.compare import (
    SELECTION_METHODS,
    ComparisonResult,
    compare_subsets,
    selection_curve,
)
from src.graph import GraphConfig
from src.graph_plot import path_graph_figure
from src.path_metrics import compare_orderings, coverage_curve
from src.pathfinder import PathConfig, find_curriculum_path
from src.pipeline import PathFinderContext, PipelineResult, build_path_finder, run_pipeline

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="EgoVerse Curriculum Path Finder", layout="wide", page_icon="🤖"
)

STRETCH = "stretch"  # Streamlit >=1.49 replacement for use_container_width=True


# --------------------------------------------------------------------------- #
# cached pipeline
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=8)
def cached_pipeline(
    data_dir: str,
    n_clusters: Optional[int],
    arm: str,
    min_length: int,
    normalize: str,
    max_length: Optional[int],
    length_normalize: bool,
    sakoe_chiba_radius: Optional[int],
    linkage: str,
    difficulty_scaling: str,
) -> PipelineResult:
    """Run the pipeline, memoised on the parameter tuple."""
    return run_pipeline(
        data_dir=data_dir,
        n_clusters=n_clusters,
        arm=arm,
        min_length=min_length,
        normalize=normalize,
        max_length=max_length,
        length_normalize=length_normalize,
        sakoe_chiba_radius=sakoe_chiba_radius,
        linkage=linkage,
        difficulty_scaling=difficulty_scaling,
        verbose=False,
    )


@st.cache_resource(show_spinner=False, max_entries=16)
def cached_path_finder(
    _result: PipelineResult,
    pipeline_key: str,
    graph_config: GraphConfig,
    task_scope: tuple,
) -> PathFinderContext:
    """Build the clip graph, layout and goal matcher, memoised on the graph settings.

    ``_result`` is underscore-prefixed so Streamlit skips hashing it -- a
    ``PipelineResult`` holds NumPy arrays and a fitted dataset and is expensive to hash.
    ``pipeline_key`` stands in for it in the cache key instead, so changing an ingestion
    setting still invalidates this cache.

    Cached separately from the pipeline on purpose: graph construction is milliseconds
    while the DTW matrix is the expensive part, so moving an interference slider must not
    re-run DTW.
    """
    return build_path_finder(
        _result, graph_config, task_names=list(task_scope) or None
    )


@st.cache_data(show_spinner=False, max_entries=32)
def cached_comparison(
    _result: PipelineResult,
    pipeline_key: str,
    subset_size: int,
    methods: tuple,
) -> ComparisonResult:
    """Score the subsets, memoised on the selection settings.

    ``_result`` is underscore-prefixed so Streamlit skips hashing its NumPy arrays;
    ``pipeline_key`` carries the ingestion settings into the cache key instead.
    """
    return compare_subsets(
        _result.distance_matrix,
        methods=methods,
        subset_size=subset_size,
        labels=_result.labels,
        tasks=_result.dataset.task_labels,
        sources=_result.dataset.field_values("source"),
    )


@st.cache_data(show_spinner=False, max_entries=8)
def cached_selection_curve(
    _result: PipelineResult, pipeline_key: str
) -> pd.DataFrame:
    """Diversity-vs-budget curve per strategy, memoised on the ingestion settings."""
    return selection_curve(
        _result.distance_matrix,
        methods=("coreset", "stratified", "random", "redundant"),
        labels=_result.labels,
    )


def _chart_theme() -> str:
    """Match the graph palette to the active Streamlit theme."""
    try:
        return "dark" if st.get_option("theme.base") == "dark" else "light"
    except Exception:  # noqa: BLE001 - theme option is not guaranteed to exist
        return "light"


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("Configuration")
data_path = st.sidebar.text_input("Episode directory", "data")

st.sidebar.subheader("Ingestion")
arm_mode = st.sidebar.selectbox(
    "End-effector arm",
    ("auto", "left", "right", "both"),
    help=(
        "auto: use the more active arm per episode, always 3-D. "
        "both: concatenate to 6-D and keep only bimanual episodes. "
        "DTW requires a consistent channel count, so 'auto' is the safe default "
        "for a dataset that mixes single-arm and bimanual sources."
    ),
)
min_length = st.sidebar.slider("Minimum valid frames", 10, 300, 30, step=10)

st.sidebar.subheader("DTW distance")
normalize = st.sidebar.selectbox(
    "Normalisation",
    ("zscore", "center", "none"),
    help=(
        "zscore (default): remove placement AND scale, comparing pure motion shape. "
        "Required across embodiments, whose motion extent differs ~5x — under 'center' "
        "that difference dominates the distance and clustering collapses. "
        "center: keep motion extent, so a 5 cm nudge and a 50 cm sweep stay far apart; "
        "correct for a single-embodiment dataset. none: raw world coordinates."
    ),
)
max_length = st.sidebar.slider(
    "Resample cap (frames)",
    50,
    600,
    200,
    step=50,
    help="DTW is O(N²·T²). EgoVerse episodes run to ~3,800 frames, so capping T is "
    "what makes this interactive.",
)
length_normalize = st.sidebar.checkbox(
    "Length-normalise distances",
    value=True,
    help="Divide each pair by its mean sequence length, so long episodes are not "
    "scored as 'diverse' purely for having more timesteps.",
)
band = st.sidebar.slider("Sakoe-Chiba radius (0 = off)", 0, 100, 0, step=5)

st.sidebar.subheader("Curriculum")
auto_k = st.sidebar.checkbox("Auto-select group count", value=True)
n_clusters = None
if not auto_k:
    n_clusters = st.sidebar.slider("Curriculum groups", 2, 12, 6)
linkage = st.sidebar.selectbox("Linkage", ("average", "complete", "single"))
difficulty_scaling = st.sidebar.selectbox("Difficulty scaling", ("rank", "minmax"))

st.sidebar.subheader("Path finder (Track 1)")
st.sidebar.caption(
    "These rebuild the graph only — they never re-run DTW, so they are instant."
)
k_neighbors = st.sidebar.slider(
    "Graph neighbours (k)", 4, 24, 10,
    help="Candidate transitions per clip, by DTW nearest neighbour. Higher gives the "
    "search more routes to choose from, at the cost of a denser picture.",
)
w_difficulty = st.sidebar.slider(
    "Difficulty-ramp weight", 0.0, 3.0, 1.0, 0.1,
    help="How hard to insist that difficulty rises by a steady increment each step.",
)
w_interference = st.sidebar.slider(
    "Interference weight", 0.0, 3.0, 1.0, 0.1,
    help="How hard to avoid switching task, skill family, embodiment or lab between "
    "consecutive clips. Turning this down broadens skill coverage but makes the "
    "curriculum jumpier — that trade-off is the point, and the Validation tab measures it.",
)
target_step = st.sidebar.slider(
    "Target difficulty step", 0.01, 0.20, 0.05, 0.01,
    help="The per-step difficulty increment the ramp aims for. Smaller means a longer, "
    "gentler curriculum.",
)
step_penalty = st.sidebar.slider(
    "Per-hop penalty", 0.0, 1.0, 0.1, 0.05,
    help="Flat cost per training step; raises it to get a shorter, more direct curriculum.",
)
review_every = st.sidebar.slider(
    "Insert a review every N clips", 0, 10, 4,
    help="Rehearsal of an already-seen clip, as a proxy for replay-based anti-forgetting. "
    "0 disables it.",
)
search_method = st.sidebar.selectbox(
    "Search", ("dijkstra", "astar"),
    help="Dijkstra is exact. A* uses a DTW-to-target heuristic that is not provably "
    "admissible, so it can return a slightly worse path; at this graph size it is not faster.",
)

run = st.sidebar.button("Run pipeline", type="primary", width=STRETCH)
st.sidebar.caption(
    "The DTW matrix is cached to `.cache/` by content hash, so re-runs with the same "
    "ingestion settings are instant."
)

# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #
st.title("EgoVerse Curriculum Path Finder & Diversity Engine")
st.markdown(
    "**Track 1 — the Curation Engine.** Type a training goal in plain English and the "
    "app finds an ordered path through a knowledge graph of clips: a curriculum that "
    "ramps difficulty smoothly and avoids the abrupt task/skill switches that cause "
    "catastrophic interference during training.\n\n"
    "**Track 2 — non-text quantitative diversity scoring.** Measures how much distinct "
    "manipulation behaviour a dataset actually contains, by comparing end-effector "
    "trajectories under Dynamic Time Warping, then sequences the episodes into a "
    "difficulty-ordered training curriculum.\n\n"
    "The two share a substrate: the DTW distance matrix is the interference signal on "
    "the graph's edges, and the kinematic difficulty score is the ramp signal."
)

if not run and "result" not in st.session_state:
    st.info(
        "Set your options in the sidebar and press **Run pipeline**.\n\n"
        "No episodes yet? Fetch a sample from the EgoVerse R2 bucket:\n"
        "```bash\n"
        "python scripts/fetch_egoverse_data.py --sources yam scale aria --limit 40\n"
        "```"
    )
    st.stop()

if run:
    with st.spinner("Loading episodes, computing DTW matrix, sequencing curriculum..."):
        st.session_state["result"] = cached_pipeline(
            data_path,
            n_clusters,
            arm_mode,
            min_length,
            normalize,
            max_length,
            length_normalize,
            band or None,
            linkage,
            difficulty_scaling,
        )

result: PipelineResult = st.session_state["result"]
ds = result.dataset

if result.n_episodes < 2:
    st.error(
        f"Only {result.n_episodes} usable episode(s) found in `{data_path}`. "
        "At least 2 are needed for a pairwise distance matrix."
    )
    if ds.skipped:
        st.subheader("Skipped episodes")
        st.dataframe(
            pd.DataFrame(ds.skipped, columns=["episode_id", "reason"]), width=STRETCH
        )
    st.stop()

df = result.frame()
report = result.report

# --------------------------------------------------------------------------- #
# headline metrics
# --------------------------------------------------------------------------- #
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Episodes analysed", result.n_episodes, delta=f"-{len(ds.skipped)} skipped")
c2.metric(
    "Diversity score",
    f"{report['diversity_score']:.5f}",
    help="Mean pairwise DTW distance over the strict upper triangle.",
)
c3.metric(
    "Redundancy",
    f"{report['redundancy_ratio']:.0%}",
    help="Share of episodes whose nearest neighbour falls in the closest 5% of all "
    "pairs — i.e. near-duplicate demonstrations.",
)
c4.metric(
    "Cluster silhouette",
    f"{report.get('silhouette', float('nan')):.3f}",
    help="Separation of the curriculum groups under the DTW metric.",
)
c5.metric(
    "Curriculum balance",
    f"{report.get('cluster_balance', float('nan')):.3f}",
    help="Normalised entropy of stage sizes; 1.0 means evenly populated stages.",
)

if result.agreement:
    task_ari = result.agreement.get("task_name")
    if task_ari is not None:
        support = result.agreement_support.get("task_name", result.n_episodes)
        st.success(
            f"**Validation — Adjusted Rand Index vs. ground-truth `task_name`: "
            f"{task_ari:.3f}** over {support} of {result.n_episodes} episodes. "
            "The clustering never sees the task labels, so recovering "
            "them from end-effector motion alone is direct evidence that the DTW metric "
            "captures behaviour rather than noise."
        )
        if support < result.n_episodes:
            st.caption(
                f"{result.n_episodes - support} episode(s) carry no `task_name` and are "
                "excluded from the score — not every EgoVerse source populates it, and "
                "scoring against a placeholder label would measure label coverage rather "
                "than the distance metric."
            )
            # An aggregate ARI hides *where* the labels are. A group containing no labelled
            # episodes contributed nothing to the score, so its grouping is unvalidated
            # rather than validated — worth stating, because the headline number cannot
            # show it.
            missing_labels = {"unknown", "", "none", "nan", "null"}
            labelled = ~(
                df["task_name"].astype(str).str.strip().str.lower().isin(missing_labels)
            )
            per_cluster = labelled.groupby(df["cluster"]).agg(["sum", "count"])
            unvalidated = per_cluster[per_cluster["sum"] == 0]
            thinnest = per_cluster[per_cluster["sum"] > 0].eval("sum / count").min()
            if not unvalidated.empty:
                st.caption(
                    f"Label coverage is uneven across groups: **{len(unvalidated)} of "
                    f"{len(per_cluster)} groups contain no labelled episode at all** "
                    f"({int(unvalidated['count'].sum())} episodes), so their grouping is "
                    "neither confirmed nor contradicted by the ARI above. The most thinly "
                    f"validated remaining group rests on {thinnest:.0%} labelled episodes."
                )

tabs = st.tabs(
    [
        "⚖️ Subset A/B",
        "🔎 Path finder",
        "📈 Path validation",
        "Diversity map",
        "Curriculum",
        "Distance structure",
        "Trajectory inspector",
        "Redundancy",
        "Data quality",
    ]
)

# --------------------------------------------------------------------------- #
# Track 1 — build the graph once, shared by the path finder and validation tabs
# --------------------------------------------------------------------------- #
graph_config = GraphConfig(
    k_neighbors=k_neighbors,
    w_difficulty=w_difficulty,
    w_interference=w_interference,
    target_step=target_step,
    step_penalty=step_penalty,
)
path_config = PathConfig(review_every=review_every, search=search_method)
pipeline_key = "|".join(
    str(x) for x in (data_path, arm_mode, min_length, normalize, max_length,
                     length_normalize, band, linkage, difficulty_scaling, n_clusters)
)
all_tasks = sorted(df["task_name"].astype(str).unique())

# --------------------------------------------------------------------------- #
# 0. subset A/B  (Track 2 headline deliverable: a score that ranks two subsets)
# --------------------------------------------------------------------------- #
with tabs[0]:
    st.subheader("Does the diversity score actually rank two subsets?")
    st.markdown(
        "A diversity score is only useful if it can *choose*. Here two equal-sized "
        "subsets are selected by different strategies and scored head to head. "
        "`coreset` is farthest-point selection over the DTW matrix; `random` is the "
        "honest baseline; `redundant` is an adversarial control that deliberately picks "
        "near-duplicates — a metric that cannot separate it from `coreset` is not "
        "measuring diversity."
    )

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])
    with ctrl1:
        method_a = st.selectbox("Subset A", SELECTION_METHODS, index=0, key="cmp_a")
    with ctrl2:
        method_b = st.selectbox("Subset B", SELECTION_METHODS, index=1, key="cmp_b")
    with ctrl3:
        max_size = max(2, result.n_episodes - 1)
        default_size = int(np.clip(result.n_episodes // 4, 2, max_size))
        subset_size = st.slider(
            "Episodes per subset", 2, max_size, default_size,
            help="Both subsets are the same size on purpose: several of these metrics "
            "move with N, so unequal sizes would measure the size gap, not the strategy.",
        )

    extra = [m for m in ("stratified", "redundant") if m not in (method_a, method_b)]
    comparison = cached_comparison(
        result, pipeline_key, subset_size, (method_a, method_b, *extra)
    )
    a_score, b_score = comparison.subsets[0], comparison.subsets[1]

    m1, m2, m3, m4 = st.columns(4)
    delta_pct = (
        (a_score.metrics["diversity_score"] - b_score.metrics["diversity_score"])
        / abs(b_score.metrics["diversity_score"]) * 100.0
        if b_score.metrics["diversity_score"] else 0.0
    )
    m1.metric(f"{method_a} diversity", f"{a_score.metrics['diversity_score']:.4f}",
              delta=f"{delta_pct:+.1f}% vs {method_b}")
    m2.metric(f"{method_b} diversity", f"{b_score.metrics['diversity_score']:.4f}")
    m3.metric(
        f"{method_a} redundancy", f"{a_score.metrics['redundancy_ratio']:.0%}",
        delta=f"{(a_score.metrics['redundancy_ratio'] - b_score.metrics['redundancy_ratio']):+.0%}",
        delta_color="inverse",
        help="Share of the subset whose nearest neighbour is a near-duplicate, using one "
        "absolute cutoff shared by both subsets.",
    )
    m4.metric(
        f"{method_a} task coverage",
        f"{int(a_score.metrics.get('n_tasks_covered', 0))} / {df['task_name'].nunique()}",
        delta=f"{int(a_score.metrics.get('n_tasks_covered', 0) - b_score.metrics.get('n_tasks_covered', 0)):+d}",
    )

    if comparison.baseline:
        base = comparison.baseline
        st.success(
            f"**{base['candidate_name']} scores {base['candidate']:.4f} against a random "
            f"baseline of {base['mean']:.4f} ± {base['std']:.4f} over "
            f"{int(base['trials'])} draws — the {base['percentile']:.0f}th percentile, "
            f"{base['z_score']:.1f}σ above the mean.** Beating a single random draw would "
            "prove little; beating the whole distribution is the ranking claim."
        )

    left, right = st.columns([3, 2])
    with left:
        table = comparison.table()
        melted = table.melt(
            id_vars="subset",
            value_vars=["diversity_score", "mean_nn_distance", "redundancy_ratio"],
            var_name="metric", value_name="value",
        )
        bar = px.bar(
            melted, x="metric", y="value", color="subset", barmode="group",
            title=f"Selection strategies compared at n={comparison.subset_size}",
        )
        bar.update_layout(height=380)
        st.plotly_chart(bar, width=STRETCH)
    with right:
        if comparison.baseline_samples is not None:
            hist = px.histogram(
                pd.DataFrame({"random subset diversity": comparison.baseline_samples}),
                x="random subset diversity", nbins=40,
                title="Null model: random subsets",
            )
            hist.add_vline(
                x=comparison.baseline["candidate"], line_dash="dash", line_color="#22c55e",
                annotation_text=comparison.baseline["candidate_name"],
            )
            hist.update_layout(height=380, showlegend=False)
            st.plotly_chart(hist, width=STRETCH)

    st.markdown("**Per-metric verdict**")
    deltas = comparison.deltas()
    st.dataframe(deltas, width=STRETCH, hide_index=True)

    st.markdown("**Budget curve** — diversity retained at every training budget")
    curve = cached_selection_curve(result, pipeline_key)
    curve_fig = px.line(
        curve, x="subset_size", y="diversity_score", color="method", markers=True,
        title="Diversity vs subset size, by selection strategy",
    )
    curve_fig.update_layout(height=380)
    st.plotly_chart(curve_fig, width=STRETCH)
    st.caption(
        "Read this as: for a given episode budget on the x-axis, how much behavioural "
        "diversity does each strategy keep. Coreset staying above random across the "
        "whole range is what shows the ranking is real rather than an artifact of one "
        "chosen subset size."
    )

    with st.expander("Which episodes were selected"):
        pick_df = pd.DataFrame(
            {
                "episode_id": [ds.episode_ids[i] for i in a_score.indices],
                "task_name": [ds.metadata[i].get("task_name") for i in a_score.indices],
                "source": [ds.metadata[i].get("source") for i in a_score.indices],
            }
        )
        st.dataframe(pick_df, width=STRETCH, height=320)
        st.download_button(
            f"Download {method_a}_subset.csv",
            pick_df.to_csv(index=False).encode(),
            f"{method_a}_subset.csv",
            "text/csv",
        )

# --------------------------------------------------------------------------- #
# 1. path finder  (Track 1)
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("Type a training goal")
    goal_col, scope_col = st.columns([3, 2])
    with goal_col:
        goal = st.text_input(
            "Training goal",
            value="teach the robot to fold a shirt",
            key="goal_text",
            help="Plain English. Matched against every clip's task name and the "
            "human-written task description, using TF-IDF over character and word n-grams.",
        )
    with scope_col:
        scope = st.multiselect(
            "Restrict to task domains (optional)",
            all_tasks,
            default=[],
            help="Scope the graph so the curriculum is routed only within related task "
            "domains, instead of through whatever happens to be kinematically adjacent.",
        )

    context = None
    try:
        context = cached_path_finder(result, pipeline_key, graph_config, tuple(scope))
    except ValueError as exc:
        st.error(f"Could not build the clip graph: {exc}")

    if context is not None:
        clip_graph = context.clip_graph
        match = context.matcher.match(goal)

        # ---- what the text matched, and how to override it ----
        # Lexical matching has a real failure mode (see src/goal_matcher.py), so the
        # ranked alternates are a first-class control rather than a hidden diagnostic.
        info, override = st.columns([3, 2])
        with info:
            if match.is_confident:
                st.success(
                    f"**{match.task_name}** — matched with score {match.score:.2f} and a "
                    f"{match.margin:.0%} lead over the runner-up."
                )
            else:
                st.warning(f"**{match.task_name}** — {match.note}")
        with override:
            # One selector at a time. Two widgets both writing `target_index` would fight
            # over Streamlit's per-key state, and the loser would silently go stale.
            source = st.radio(
                "Target from",
                ("Matched task", "A specific clip"),
                horizontal=True,
                key="target_source",
            )

        if source == "Matched task":
            options = [c.clip_index for c in match.candidates] or [match.target_index]
            labels = {c.clip_index: c.label() for c in match.candidates}
            target_index = st.selectbox(
                "Target (override if the match is wrong)",
                options,
                index=0,
                format_func=lambda i: labels.get(i, clip_graph.episode_id(i)),
                key="target_from_match",
            )
        else:
            target_index = st.selectbox(
                "Target clip",
                list(range(clip_graph.n_clips)),
                format_func=lambda i: (
                    f"{clip_graph.episode_id(i)} — {clip_graph.nodes['task_name'].iloc[i]} "
                    f"(difficulty {clip_graph.nodes['difficulty'].iloc[i]:.2f})"
                ),
                key="target_from_clip",
            )

        path = find_curriculum_path(clip_graph, int(target_index), path_config)
        st.session_state["path"] = path
        st.session_state["context"] = context

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clips in curriculum", len(path.route))
        m2.metric("Review insertions", path.n_reviews)
        m3.metric(
            "Path cost", f"{path.search_cost:.1f}",
            help="Total edge weight; the sum of the ramp, interference and per-hop terms.",
        )
        m4.metric(
            "Graph", f"{clip_graph.n_clips} clips",
            delta=f"{clip_graph.graph.number_of_edges() - len(clip_graph.start_clips)} edges",
            delta_color="off",
        )
        if clip_graph.repairs:
            st.caption(
                f"{len(clip_graph.repairs)} edge(s) were added purely to keep every clip "
                "reachable — the k-nearest-neighbour structure did not justify them on its "
                "own, so a path routed through one is less well supported by the data."
            )

        # ---- the graph ----
        event = st.plotly_chart(
            path_graph_figure(
                clip_graph, context.layout, path,
                theme=_chart_theme(),
                title=f"“{goal}” → {clip_graph.episode_id(path.target_index)}",
            ),
            key="path_graph",
            on_select="rerun",
            selection_mode=("points",),
            width=STRETCH,
        )
        st.caption(
            "Node fill is difficulty (pale → dark), so a correct curriculum visibly walks "
            "from light to dark. Faint grey lines are candidate transitions the search "
            "could have taken; the orange route is the one it chose. Diamonds are rehearsal "
            "steps, the star is the goal. Click any node to inspect it."
        )

        # ---- click-through detail ----
        selected = None
        try:
            points = event.selection["points"] if event and event.selection else []
            for point in points:
                data = point.get("customdata")
                if data is not None:
                    selected = int(data[0] if isinstance(data, (list, tuple)) else data)
                    break
        except Exception:  # noqa: BLE001 - selection payload shape varies by version
            selected = None

        if selected is not None:
            row = clip_graph.nodes.iloc[selected]
            st.markdown(f"#### {row['episode_id']}")
            d1, d2 = st.columns([1, 2])
            with d1:
                st.write(
                    {
                        "task": row.get("task_name"),
                        "description": row.get("task_description"),
                        "difficulty": round(float(row["difficulty"]), 3),
                        "skill family": int(row["cluster"]),
                        "embodiment": row.get("embodiment"),
                        "lab / source": row.get("source"),
                        "frames": int(row.get("n_frames", 0) or 0),
                        "on path": selected in path.clips,
                    }
                )
            with d2:
                original = int(context.kept_indices[selected])
                traj = ds.trajectories[original]
                detail = go.Figure(
                    go.Scatter3d(
                        x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                        mode="lines", line=dict(width=4),
                        name=str(row["episode_id"]),
                    )
                )
                detail.update_layout(
                    height=380, margin=dict(l=0, r=0, t=30, b=0),
                    title="End-effector path (metres)",
                    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z",
                               aspectmode="data"),
                )
                st.plotly_chart(detail, width=STRETCH)
        else:
            st.info("Click a node on the graph to inspect that clip.")

        # ---- the ordered curriculum ----
        st.subheader("Ordered curriculum")
        st.markdown(
            "Train in this order. Rows flagged `is_review` are rehearsal repeats of a "
            "clip seen earlier — their `ramp_cost` and `edge_weight` are blank because a "
            "deliberate step *backwards* in difficulty has no ramp to score."
        )
        display = [
            c for c in (
                "step", "episode_id", "task_name", "difficulty", "stage", "is_review",
                "reviews_step", "edge_weight", "ramp_cost", "interference_cost",
                "dtw_from_prev", "difficulty_delta", "task_switch", "cluster_switch",
                "on_graph_edge", "embodiment", "source", "n_frames",
            ) if c in path.table.columns
        ]
        st.dataframe(path.table[display], width=STRETCH, height=420)
        st.download_button(
            "Download path.csv",
            path.table.to_csv(index=False).encode(),
            "path.csv",
            "text/csv",
        )

# --------------------------------------------------------------------------- #
# 2. path validation  (Track 1)
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("Does the path actually behave like a curriculum?")
    path = st.session_state.get("path")
    context = st.session_state.get("context")

    if path is None or context is None:
        st.info("Find a path on the **Path finder** tab first.")
    else:
        clip_graph = context.clip_graph
        comparison = compare_orderings(path, clip_graph, context.distance_matrix)

        st.markdown(
            "Every baseline is the **same size** as the found path, so none of this is a "
            "length effect. What each one isolates:\n\n"
            "- **Random order (same clips)** — the path's own selection, reshuffled. "
            "Isolates the value of the *ordering*.\n"
            "- **Random subset (same size)** — fresh clips, with the target forced in. "
            "Isolates *selection and ordering* together.\n"
            "- **Difficulty-sorted (same clips)** — the obvious hand-rolled curriculum. It "
            "nails monotonicity by construction, so the path has to win on *interference*, "
            "not on ramp smoothness.\n"
            "- **Coreset prefix (same size)** — the Track 2 farthest-point ordering, which "
            "maximises coverage. Expect it to beat the path on coverage and lose on ramp: "
            "that contrast is the whole argument for curriculum ordering over pure coverage."
        )

        pretty = {
            "spearman": "Difficulty monotonicity (ρ) ↑",
            "frac_nondecreasing": "Non-decreasing steps ↑",
            "max_jump": "Largest difficulty jump ↓",
            "mean_abs_step": "Mean |difficulty step| ↓",
            "task_switch_rate": "Task switch rate ↓",
            "cluster_switch_rate": "Skill-family switch rate ↓",
            "mean_consecutive_dtw": "Mean consecutive DTW ↓",
            "cluster_coverage": "Skill coverage to target ↑",
            "task_coverage": "Task coverage to target ↑",
            "frac_consecutive_near_duplicate": "Consecutive near-duplicates ↓",
            "mean_pairwise_dtw": "Mean pairwise DTW in set ↑",
        }
        shown = [c for c in pretty if c in comparison.columns]
        table = comparison[shown].rename(columns=pretty).T
        st.dataframe(table.style.format("{:.3f}"), width=STRETCH)
        st.caption(
            "**Read smoothness on _Mean |difficulty step|_, not on _Largest difficulty "
            "jump_.** The latter counts only upward steps — the harm it is meant to catch — "
            "so over a fixed clip set a *descending* ordering posts a deceptively small "
            "value. Total absolute variation is the order-fair measure, and the sorted "
            "order provably minimises it.\n\n"
            "↑ / ↓ marks which direction is better. Monotonicity and switch rates are "
            "measured over the *introduction* sequence, excluding rehearsal steps — a "
            "review is meant to step backwards and switch context, and the baselines have "
            "no reviews, so counting them would compare unlike things."
        )

        difficulty = clip_graph.nodes["difficulty"].astype(float).to_numpy()
        c1, c2 = st.columns(2)
        with c1:
            ramp = pd.DataFrame(
                {
                    "step": np.arange(1, path.n_steps + 1),
                    "difficulty": [difficulty[c] for c in path.clips],
                    "kind": ["review" if r else "new clip" for r in path.is_review],
                }
            )
            fig = px.line(ramp, x="step", y="difficulty", markers=False,
                          title="Difficulty along the curriculum")
            fig.add_scatter(
                x=ramp["step"], y=ramp["difficulty"], mode="markers",
                marker=dict(size=9, symbol=[
                    "diamond" if k == "review" else "circle" for k in ramp["kind"]
                ]),
                name="step", showlegend=False,
            )
            fig.update_layout(height=340)
            st.plotly_chart(fig, width=STRETCH)
            st.caption("Diamonds are rehearsal steps — the dips are intentional.")
        with c2:
            curve = coverage_curve(path.clips, clip_graph.nodes, "cluster")
            total = int(clip_graph.nodes["cluster"].nunique())
            cov = pd.DataFrame(
                {"step": np.arange(1, len(curve) + 1), "skill families seen": curve}
            )
            fig = px.line(cov, x="step", y="skill families seen",
                          title=f"Skill coverage accumulated (of {total} families)")
            fig.add_hline(y=total, line_dash="dash",
                          annotation_text="all families")
            fig.update_layout(height=340, yaxis_range=[0, total + 0.5])
            st.plotly_chart(fig, width=STRETCH)
            st.caption(
                "A curve that plateaus early and low means the interference term is "
                "keeping the curriculum inside one family. Lower the interference weight "
                "in the sidebar to trade smoothness for breadth."
            )

        hops = pd.DataFrame(
            {
                "step": np.arange(2, path.n_steps + 1),
                "consecutive DTW": [
                    float(clip_graph.normalized_distance[u, v])
                    for u, v in zip(path.clips[:-1], path.clips[1:])
                ],
            }
        )
        fig = px.bar(hops, x="step", y="consecutive DTW",
                     title="Motion distance between consecutive clips (lower = less interference)")
        fig.update_layout(height=300)
        st.plotly_chart(fig, width=STRETCH)

        with st.expander("Cost breakdown of the searched route"):
            st.json(
                {
                    "search_method": path.method,
                    "total_cost": path.search_cost,
                    "cost_terms": path.cost_terms,
                    "n_clips": len(path.route),
                    "n_reviews": path.n_reviews,
                    "graph_config": clip_graph.config.as_dict(),
                    "task_scope": context.task_names,
                    "reachability_repairs": len(clip_graph.repairs),
                }
            )

# --------------------------------------------------------------------------- #
# 3. diversity map
# --------------------------------------------------------------------------- #
with tabs[3]:
    left, right = st.columns([3, 1])
    with right:
        color_by = st.selectbox(
            "Colour by",
            ("cluster_label", "task_name", "source", "embodiment", "difficulty", "stage"),
        )
        size_by_difficulty = st.checkbox("Size by difficulty", value=False)
    with left:
        fig = px.scatter_3d(
            df,
            x="UMAP_X",
            y="UMAP_Y",
            z="UMAP_Z",
            color=color_by,
            size="difficulty" if size_by_difficulty and "difficulty" in df else None,
            hover_data=["episode_id", "task_name", "source", "n_frames", "difficulty", "stage"],
            title="Trajectory diversity map — UMAP of the DTW distance matrix",
        )
        fig.update_traces(marker=dict(opacity=0.85))
        if not size_by_difficulty:
            fig.update_traces(marker=dict(size=6))
        fig.update_layout(height=680, legend_title_text=color_by)
        st.plotly_chart(fig, width=STRETCH)
    st.caption(
        "Each point is one episode. Proximity means similar end-effector motion under "
        "DTW. UMAP consumes the precomputed distance matrix directly, so the layout "
        "reflects DTW geometry — not Euclidean distance between raw coordinates."
    )

# --------------------------------------------------------------------------- #
# 4. curriculum
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("Training order")
    st.markdown(
        "Two orderings, for two different strategies. **Curriculum rank** groups "
        "episodes into stages by motion family and ascends by difficulty inside each "
        "stage. **Coreset rank** is a farthest-point traversal of the DTW matrix: "
        "truncate it at any *K* to get a near-maximally diverse *K*-episode subset — "
        "the ordering to use when subsampling to a training budget."
    )

    if not result.stages.empty:
        stage_fig = px.bar(
            result.stages,
            x="stage",
            y="n_episodes",
            color="mean_difficulty",
            color_continuous_scale="Viridis",
            title="Episodes per curriculum stage (stage 1 = easiest motion family)",
            labels={"n_episodes": "episodes", "mean_difficulty": "mean difficulty"},
        )
        stage_fig.update_layout(height=320)
        st.plotly_chart(stage_fig, width=STRETCH)

    scatter = px.scatter(
        df.sort_values("curriculum_rank"),
        x="curriculum_rank",
        y="difficulty",
        color="cluster_label",
        hover_data=["episode_id", "task_name", "n_frames"],
        title="Difficulty along the curriculum",
    )
    scatter.update_traces(marker=dict(size=10, opacity=0.85))
    scatter.update_layout(height=340)
    st.plotly_chart(scatter, width=STRETCH)

    display_cols = [
        c
        for c in (
            "curriculum_rank",
            "episode_id",
            "stage",
            "difficulty",
            "difficulty_z",
            "coreset_rank",
            "is_cluster_medoid",
            "task_name",
            "source",
            "duration",
            "path_length",
            "tortuosity",
            "normalized_jerk",
            "reversal_rate",
        )
        if c in result.curriculum.columns
    ]
    st.dataframe(result.curriculum[display_cols], width=STRETCH, height=420)
    st.download_button(
        "Download curriculum.csv",
        result.curriculum.to_csv(index=False).encode(),
        "curriculum.csv",
        "text/csv",
    )

    with st.expander("Coreset order — maximum-coverage subset selection"):
        coreset = df.sort_values("coreset_rank")[
            ["coreset_rank", "episode_id", "task_name", "cluster_label", "difficulty"]
        ]
        st.markdown(
            "Reading top-down, each next episode is the one farthest from everything "
            "already picked. Note how it alternates between tasks — that is the "
            "coverage-first behaviour a budget-limited training set wants."
        )
        st.dataframe(coreset, width=STRETCH, height=320)

# --------------------------------------------------------------------------- #
# 5. distance structure
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("DTW distance matrix")
    order_by = st.radio(
        "Order rows by", ("curriculum stage", "cluster", "dataset order"), horizontal=True
    )
    if order_by == "curriculum stage" and "curriculum_rank" in df:
        order = np.argsort(df["curriculum_rank"].to_numpy())
    elif order_by == "cluster":
        order = np.argsort(result.labels, kind="stable")
    else:
        order = np.arange(result.n_episodes)

    matrix = result.distance_matrix[np.ix_(order, order)]
    labels_ordered = [ds.episode_ids[i] for i in order]
    heat = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels_ordered,
            y=labels_ordered,
            colorscale="Magma_r",
            colorbar=dict(title="DTW"),
        )
    )
    heat.update_layout(
        height=620,
        title="Pairwise DTW distances — dark blocks are groups of similar behaviour",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )
    st.plotly_chart(heat, width=STRETCH)
    st.caption(
        "Sorted by curriculum stage, a well-separated dataset shows dark blocks on the "
        "diagonal (within-group similarity) and bright off-diagonal regions (between-group "
        "difference). Large dark off-diagonal areas indicate stages that are not really distinct."
    )

    col1, col2 = st.columns(2)
    with col1:
        vals = result.distance_matrix[np.triu_indices(result.n_episodes, k=1)]
        hist = px.histogram(
            pd.DataFrame({"pairwise DTW distance": vals}),
            x="pairwise DTW distance",
            nbins=40,
            title="Distribution of pairwise distances",
        )
        hist.update_layout(height=330)
        st.plotly_chart(hist, width=STRETCH)
    with col2:
        if result.silhouette_by_k:
            sil = pd.DataFrame(
                {
                    "k": list(result.silhouette_by_k),
                    "silhouette": list(result.silhouette_by_k.values()),
                }
            )
            sil_fig = px.line(
                sil, x="k", y="silhouette", markers=True, title="Group count selection"
            )
            sil_fig.add_vline(
                x=result.suggested_k,
                line_dash="dash",
                annotation_text=f"chosen k={result.suggested_k}",
            )
            sil_fig.update_layout(height=330)
            st.plotly_chart(sil_fig, width=STRETCH)

# --------------------------------------------------------------------------- #
# 6. trajectory inspector
# --------------------------------------------------------------------------- #
with tabs[6]:
    st.subheader("Actual end-effector paths")
    st.markdown(
        "The map above is an abstraction. This is the underlying motion, so the "
        "grouping can be checked by eye rather than taken on trust."
    )
    mode = st.radio(
        "Show", ("One episode per curriculum stage", "Pick episodes manually"), horizontal=True
    )

    if mode == "One episode per curriculum stage":
        # Cluster medoids are the most representative real episode per group.
        picks = []
        for cluster, idx in sorted(result.medoids.items()):
            picks.append((f"stage medoid — {ds.episode_ids[idx]}", idx))
    else:
        chosen = st.multiselect(
            "Episodes",
            options=list(range(result.n_episodes)),
            default=list(range(min(4, result.n_episodes))),
            format_func=lambda i: ds.episode_ids[i],
        )
        picks = [(ds.episode_ids[i], i) for i in chosen]

    if picks:
        traj_fig = go.Figure()
        for name, idx in picks:
            traj = ds.trajectories[idx]
            meta = ds.metadata[idx]
            traj_fig.add_trace(
                go.Scatter3d(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    z=traj[:, 2],
                    mode="lines",
                    name=f"{name[:38]} ({meta.get('task_name')})",
                    line=dict(width=4),
                )
            )
        traj_fig.update_layout(
            height=660,
            title="End-effector XYZ paths (metres)",
            scene=dict(
                xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)", aspectmode="data"
            ),
        )
        st.plotly_chart(traj_fig, width=STRETCH)
    else:
        st.info("Select at least one episode.")

# --------------------------------------------------------------------------- #
# 7. redundancy
# --------------------------------------------------------------------------- #
with tabs[7]:
    st.subheader("Near-duplicate episodes")
    st.markdown(
        "Pairs in the closest 5% of the pairwise distribution. These are the first "
        "candidates to prune when a dataset is over budget, and a distance of exactly "
        "0 means two byte-identical demonstrations are being stored twice."
    )
    if result.redundant_pairs:
        rows = [
            {
                "DTW distance": round(dist, 6),
                "episode A": ds.episode_ids[i],
                "episode B": ds.episode_ids[j],
                "task A": ds.metadata[i].get("task_name"),
                "task B": ds.metadata[j].get("task_name"),
                "same task": ds.metadata[i].get("task_name") == ds.metadata[j].get("task_name"),
            }
            for i, j, dist in result.redundant_pairs
        ]
        red_df = pd.DataFrame(rows)
        exact = red_df[red_df["DTW distance"] <= 1e-9]
        if not exact.empty:
            st.warning(
                f"**{len(exact)} pair(s) at distance ~0** — duplicated episodes, not merely "
                "similar ones. Worth removing at the source."
            )
        st.dataframe(red_df, width=STRETCH, height=420)
        st.download_button(
            "Download redundant_pairs.csv",
            red_df.to_csv(index=False).encode(),
            "redundant_pairs.csv",
            "text/csv",
        )
    else:
        st.success("No near-duplicate pairs below the threshold.")

    nn = np.sort(np.where(np.eye(result.n_episodes, dtype=bool), np.inf, result.distance_matrix).min(axis=1))
    nn_fig = px.bar(
        pd.DataFrame({"episode (sorted)": np.arange(len(nn)), "nearest-neighbour DTW": nn}),
        x="episode (sorted)",
        y="nearest-neighbour DTW",
        title="Distance to nearest neighbour — low bars are redundant episodes",
    )
    nn_fig.update_layout(height=330)
    st.plotly_chart(nn_fig, width=STRETCH)

# --------------------------------------------------------------------------- #
# 8. data quality
# --------------------------------------------------------------------------- #
with tabs[8]:
    st.subheader("Ingestion and data quality")
    st.markdown(
        "Curation depends on knowing what was thrown away and why. Real EgoVerse "
        "episodes carry chunk-padded arrays, missing-frame sentinels, unpopulated pose "
        "streams and inconsistent units — all handled at load time and reported here."
    )

    q1, q2, q3 = st.columns(3)
    q1.metric("Loaded", result.n_episodes)
    q2.metric("Skipped", len(ds.skipped))
    mm = sum(1 for m in ds.metadata if float(m.get("unit_scale") or 1.0) != 1.0)
    q3.metric("Unit-converted (mm→m)", mm)

    meta_df = pd.DataFrame(ds.metadata)
    meta_df.insert(0, "episode_id", ds.episode_ids)
    meta_df["n_frames_used"] = ds.lengths
    st.dataframe(meta_df, width=STRETCH, height=340)

    if ds.skipped:
        st.subheader(f"Skipped episodes ({len(ds.skipped)})")
        st.dataframe(
            pd.DataFrame(ds.skipped, columns=["episode_id", "reason"]),
            width=STRETCH,
            height=280,
        )

    counts = pd.DataFrame({"source": ds.field_values("source")}).value_counts().reset_index()
    counts.columns = ["source", "episodes"]
    comp = px.pie(counts, names="source", values="episodes", title="Episodes by source", hole=0.45)
    comp.update_layout(height=360)
    st.plotly_chart(comp, width=STRETCH)

    with st.expander("Run configuration"):
        st.json(
            {
                "data_dir": data_path,
                "arm": arm_mode,
                "min_length": min_length,
                "normalize": normalize,
                "max_length": max_length,
                "length_normalize": length_normalize,
                "sakoe_chiba_radius": band or None,
                "linkage": linkage,
                "difficulty_scaling": difficulty_scaling,
                "n_clusters": n_clusters or f"auto (k={result.suggested_k})",
                "diversity_metrics": report,
                "cluster_label_agreement_ari": result.agreement,
                "cluster_label_agreement_support": result.agreement_support,
            }
        )
