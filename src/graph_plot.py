"""Rendering the clip graph with a curriculum path highlighted on it.

Shared by the dashboard and the headless CLI so the interactive view and the exported
HTML cannot drift apart.

**Encoding choices**, in the order they were decided:

*Form.* A node-link diagram, because the question the picture answers is "what route did
the curriculum take through the dataset" -- that is topology, not magnitude, and no bar or
line chart shows a route.

*Colour.* Node fill encodes **difficulty**, which is a continuous magnitude, so it gets a
single-hue sequential ramp (blue, light to dark) rather than categorical hues. This is
also the encoding that makes the demo legible: a correct curriculum path visibly walks
from pale to dark as it advances, so the ramp is verifiable at a glance instead of only in
the metrics table. Skill family is *not* colour-encoded -- with 7+ families, categorical
hues cannot stay distinguishable under colour-vision deficiency in a scatter-like form
where any two marks can end up adjacent. Family lives in the hover text, and the layout
already groups families spatially.

The path uses one accent hue (orange), validated against the blue ramp at CVD ΔE 25.3
(protan) and normal-vision ΔE 29.0 -- unambiguous for every reader. Review and target
marks reuse that same accent and are distinguished by **symbol**, not by new hues, so the
palette never grows past one categorical slot.

The node ramp deliberately starts at the ramp's step 250 rather than step 100: the lightest
step measures 1.29:1 against a light surface, so the easiest clips -- the ones the
curriculum starts on -- would have faded into the background. Every mark also carries a
thin surface-coloured ring so overlapping nodes stay separable in dense regions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from .graph import ClipGraph
from .pathfinder import CurriculumPath

# Sequential blue ramp for difficulty, floored at step 250 so low-difficulty nodes stay
# visible against the surface. Dark mode mirrors it, capped before the steps that recede
# into a dark surface.
DIFFICULTY_RAMP_LIGHT = (
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)
DIFFICULTY_RAMP_DARK = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
)

THEMES: Dict[str, Dict[str, object]] = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "gridline": "#e1e0d9",
        "accent": "#eb6834",
        "ramp": DIFFICULTY_RAMP_LIGHT,
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "muted": "#898781",
        "gridline": "#2c2c2a",
        "accent": "#d95926",
        "ramp": DIFFICULTY_RAMP_DARK,
    },
}

# Marks are >= 8px; path marks sit above them and so need a little more room.
BASE_MARKER_SIZE = 11
PATH_RING_SIZE = 20


def _colorscale(ramp: Sequence[str]) -> List[List[object]]:
    stops = np.linspace(0.0, 1.0, len(ramp))
    return [[float(s), c] for s, c in zip(stops, ramp)]


def _hover_text(clip_graph: ClipGraph) -> List[str]:
    """Per-node tooltip. Carries the identity information colour deliberately does not."""
    nodes = clip_graph.nodes
    lines: List[str] = []
    for i in range(len(nodes)):
        row = nodes.iloc[i]
        parts = [f"<b>{row['episode_id']}</b>"]
        if "task_name" in nodes.columns:
            parts.append(f"task: {row['task_name']}")
        if "task_description" in nodes.columns and str(row.get("task_description", "")):
            parts.append(f"“{str(row['task_description'])[:70]}”")
        parts.append(f"difficulty: {float(row['difficulty']):.3f}")
        parts.append(f"skill family: {int(row['cluster'])}")
        if "stage" in nodes.columns:
            parts.append(f"stage: {row['stage']}")
        for column, label in (("embodiment", "embodiment"), ("source", "lab"),
                              ("n_frames", "frames")):
            if column in nodes.columns:
                parts.append(f"{label}: {row[column]}")
        lines.append("<br>".join(parts))
    return lines


def _edge_segments(clip_graph: ClipGraph, layout: np.ndarray):
    """Background edges as one trace, using None separators to break the polyline."""
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    for u, v in clip_graph.graph.edges():
        if u == "START" or v == "START":
            continue
        xs.extend([layout[int(u), 0], layout[int(v), 0], None])
        ys.extend([layout[int(u), 1], layout[int(v), 1], None])
    return xs, ys


def path_graph_figure(
    clip_graph: ClipGraph,
    layout: np.ndarray,
    path: Optional[CurriculumPath] = None,
    theme: str = "light",
    title: Optional[str] = None,
    show_step_labels: bool = True,
) -> go.Figure:
    """Draw the clip graph, optionally with a curriculum path highlighted.

    Args:
        clip_graph: The built graph.
        layout: ``(N, 2)`` force-directed positions from
            :func:`~src.graph.force_directed_layout`.
        path: Path to highlight. ``None`` draws the bare graph.
        theme: ``"light"`` or ``"dark"``; picks the validated palette for that surface.
        title: Figure title.
        show_step_labels: Number the path nodes directly on the plot. Selective direct
            labelling -- only path marks are labelled, never every node.

    Returns:
        A Plotly figure. Node ``customdata`` carries the clip index, so a click handler
        can resolve which clip was selected.
    """
    palette = THEMES.get(theme, THEMES["light"])
    accent = str(palette["accent"])
    surface = str(palette["surface"])
    nodes = clip_graph.nodes
    difficulty = nodes["difficulty"].astype(float).fillna(1.0).to_numpy()
    n = len(nodes)

    figure = go.Figure()

    # ---- background edges: recessive, non-interactive ----
    edge_x, edge_y = _edge_segments(clip_graph, layout)
    figure.add_trace(
        go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(color=str(palette["gridline"]), width=1),
            hoverinfo="skip", showlegend=False, name="candidate transitions",
        )
    )

    # ---- all clips ----
    figure.add_trace(
        go.Scatter(
            x=layout[:, 0], y=layout[:, 1], mode="markers",
            marker=dict(
                size=BASE_MARKER_SIZE,
                color=difficulty,
                colorscale=_colorscale(palette["ramp"]),
                cmin=0.0, cmax=1.0,
                line=dict(color=surface, width=1.5),
                colorbar=dict(
                    title=dict(text="difficulty", side="right"),
                    thickness=12, len=0.55, outlinewidth=0,
                    tickfont=dict(color=str(palette["muted"])),
                ),
            ),
            customdata=np.arange(n),
            text=_hover_text(clip_graph),
            hovertemplate="%{text}<extra></extra>",
            name="Clips", showlegend=True,
        )
    )

    if path is None or not path.clips:
        _finalize(figure, palette, title)
        return figure

    # ---- the path, drawn over the graph ----
    steps = [int(c) for c in path.clips]
    px_, py_ = layout[steps, 0], layout[steps, 1]

    figure.add_trace(
        go.Scatter(
            x=px_, y=py_, mode="lines",
            line=dict(color=accent, width=3),
            hoverinfo="skip", name="Curriculum path", showlegend=True,
        )
    )

    # Direction arrows. The route is an *ordering*, so which way it runs is the whole
    # point; an undirected polyline would leave that unreadable.
    for (x0, y0), (x1, y1) in zip(zip(px_[:-1], py_[:-1]), zip(px_[1:], py_[1:])):
        if np.hypot(x1 - x0, y1 - y0) < 1e-9:
            continue
        figure.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=2,
            arrowcolor=accent, opacity=0.9, text="",
            # Stop short of the marks at both ends, or the arrowhead lands underneath
            # the path ring and the direction becomes invisible.
            standoff=PATH_RING_SIZE * 0.65, startstandoff=PATH_RING_SIZE * 0.65,
        )

    introduced = [c for c, review in zip(steps, path.is_review) if not review]
    reviews = [c for c, review in zip(steps, path.is_review) if review]

    figure.add_trace(
        go.Scatter(
            x=layout[introduced, 0], y=layout[introduced, 1], mode="markers",
            marker=dict(size=PATH_RING_SIZE, symbol="circle-open",
                        color=accent, line=dict(color=accent, width=2.5)),
            customdata=np.array(introduced),
            hoverinfo="skip", name="On path", showlegend=True,
        )
    )
    if reviews:
        # Symbol, not a new hue: the palette stays at one categorical slot.
        figure.add_trace(
            go.Scatter(
                x=layout[reviews, 0], y=layout[reviews, 1], mode="markers",
                marker=dict(size=PATH_RING_SIZE + 4, symbol="diamond-open",
                            color=accent, line=dict(color=accent, width=2.5)),
                customdata=np.array(reviews),
                hoverinfo="skip", name="Review (rehearsal)", showlegend=True,
            )
        )

    target = int(path.target_index)
    figure.add_trace(
        go.Scatter(
            x=[layout[target, 0]], y=[layout[target, 1]], mode="markers",
            marker=dict(size=PATH_RING_SIZE + 8, symbol="star-open",
                        color=accent, line=dict(color=accent, width=2.5)),
            hoverinfo="skip", name="Goal", showlegend=True,
        )
    )

    if show_step_labels:
        # Selective direct labels: the training order, on path marks only. A rehearsed
        # clip occupies one position but two steps, so its labels are merged ("1,5")
        # rather than stacked illegibly on top of each other.
        label_of: Dict[int, List[str]] = {}
        for step_number, clip in enumerate(steps, start=1):
            label_of.setdefault(clip, []).append(str(step_number))
        clips_labelled = list(label_of)
        # Lift the labels clear of the rings; textposition alone leaves them overlapping
        # a 20px mark, which is what made them unreadable.
        offset = 0.045 * float(np.ptp(layout[:, 1]) or 1.0)
        figure.add_trace(
            go.Scatter(
                x=layout[clips_labelled, 0], y=layout[clips_labelled, 1] + offset,
                mode="text",
                text=[",".join(label_of[c]) for c in clips_labelled],
                textposition="top center",
                textfont=dict(size=12, color=str(palette["text_primary"])),
                hoverinfo="skip", showlegend=False,
            )
        )

    _finalize(figure, palette, title)
    return figure


def _finalize(figure: go.Figure, palette: Dict[str, object], title: Optional[str]) -> None:
    """Strip the axes and apply surface/ink tokens. A node-link layout has no scale."""
    blank = dict(showgrid=False, zeroline=False, showticklabels=False, visible=False)
    figure.update_layout(
        title=dict(text=title or "", font=dict(color=str(palette["text_primary"]))),
        xaxis=blank, yaxis=blank,
        plot_bgcolor=str(palette["surface"]),
        paper_bgcolor=str(palette["surface"]),
        font=dict(color=str(palette["text_secondary"])),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            font=dict(color=str(palette["text_secondary"])),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=10, t=88, b=10),
        height=640,
        hoverlabel=dict(align="left"),
        dragmode="pan",
    )
