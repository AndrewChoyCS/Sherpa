"""Tests for the Track 1 curriculum path finder.

The properties pinned down here are the ones the design depends on rather than merely
the ones that are easy to assert:

- every edge cost is non-negative, which is what makes Dijkstra valid;
- the difficulty ramp is *structural* -- no path can descend more than the tolerance
  allows, regardless of how the weights are tuned;
- ``START`` reaches every clip, so no goal is unroutable;
- rehearsal steps only ever repeat clips already introduced earlier in the same path;
- the found path actually beats a random ordering of its own clips on monotonicity.

Fixtures reuse ``scripts/generate_synthetic_data.py``, so the graph is built over
episodes in the real Zarr v3 schema (chunk padding, sentinel frames, millimetre units)
rather than idealised arrays. Tests needing real episodes skip when ``data/`` is empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_synthetic_data import generate_dataset  # noqa: E402

from src.goal_matcher import GoalMatcher, build_documents  # noqa: E402
from src.graph import (  # noqa: E402
    START,
    GraphConfig,
    build_clip_graph,
    edge_cost,
    force_directed_layout,
    normalize_distances,
    scope_to_tasks,
)
from src.path_metrics import (  # noqa: E402
    compare_orderings,
    coverage,
    coverage_curve,
    difficulty_monotonicity,
    interference_profile,
    path_report,
    redundancy,
)
from src.pathfinder import (  # noqa: E402
    PathConfig,
    find_curriculum_path,
    insert_reviews,
    search_route,
)
from src.pipeline import build_path_finder, run_pipeline  # noqa: E402

REAL_DATA_DIR = REPO_ROOT / "data"
requires_real_data = pytest.mark.skipif(
    not REAL_DATA_DIR.exists() or not any(REAL_DATA_DIR.glob("*.zarr")),
    reason="real EgoVerse episodes not fetched; run scripts/fetch_egoverse_data.py",
)


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("synth_graph")
    generate_dataset(out, n_episodes=24, n_duplicates=3, seed=11, inject_edge_cases=True)
    return out


@pytest.fixture(scope="module")
def result(synth_dir):
    return run_pipeline(str(synth_dir), n_clusters=4, cache_dir=None, verbose=False)


@pytest.fixture(scope="module")
def context(result):
    return build_path_finder(result)


@pytest.fixture(scope="module")
def clip_graph(context):
    return context.clip_graph


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #
class TestClipGraph:
    def test_all_edge_weights_non_negative(self, clip_graph):
        """Dijkstra's precondition. If this fails the search is silently unsound."""
        weights = [d["weight"] for _, _, d in clip_graph.graph.edges(data=True)]
        assert weights
        assert min(weights) >= 0.0

    def test_edges_never_descend_beyond_tolerance(self, clip_graph):
        """The ramp is structural, not just discouraged by cost."""
        tolerance = clip_graph.config.backslide_tolerance
        difficulty = clip_graph.nodes["difficulty"].astype(float).to_numpy()
        for u, v in clip_graph.graph.edges():
            if u == START:
                continue
            assert difficulty[v] >= difficulty[u] - tolerance - 1e-9

    def test_start_reaches_every_clip(self, clip_graph):
        """No goal may be unroutable, or the app can answer a query with an error."""
        import networkx as nx

        reachable = nx.descendants(clip_graph.graph, START)
        assert len(reachable) == clip_graph.n_clips

    def test_build_is_deterministic(self, result):
        first = build_path_finder(result).clip_graph
        second = build_path_finder(result).clip_graph
        # Node ids mix the START string with integer clip indices, so sort by str.
        as_text = lambda g: sorted(f"{u}->{v}" for u, v in g.edges())  # noqa: E731
        assert as_text(first.graph) == as_text(second.graph)
        assert first.start_clips == second.start_clips
        assert first.repairs == second.repairs

    def test_start_pool_is_easy_in_absolute_terms(self, clip_graph):
        """Entry points must be genuinely easy, not merely in the easiest family.

        Family membership alone put 106 of 273 clips -- including the goal -- into the
        zero-cost entry pool, which let the search return a one-clip curriculum.
        """
        nodes = clip_graph.nodes
        values = nodes["difficulty"].astype(float).to_numpy()
        quantile = clip_graph.config.start_quantile
        threshold = float(np.quantile(values, quantile))
        # Repairs may attach an extra entry point, so bound the pool rather than fix it.
        repaired = {dst for src, dst in clip_graph.repairs if src == START}
        for clip in clip_graph.start_clips:
            if clip in repaired:
                continue
            assert values[clip] <= threshold + 1e-9
        assert len(clip_graph.start_clips) < clip_graph.n_clips

    def test_repairs_are_flagged_and_reported(self, clip_graph):
        """Repair edges must be auditable, not silently inflate apparent connectivity."""
        flagged = {
            (u, v) for u, v, d in clip_graph.graph.edges(data=True) if d.get("is_repair")
        }
        assert flagged == {(u, v) for u, v in clip_graph.repairs}

    def test_rejects_frame_matrix_size_mismatch(self, clip_graph):
        frame = clip_graph.nodes.iloc[:-1]
        with pytest.raises(ValueError, match="same order"):
            build_clip_graph(clip_graph.normalized_distance, frame)

    def test_rejects_missing_columns(self, clip_graph):
        frame = clip_graph.nodes.drop(columns=["difficulty"])
        with pytest.raises(ValueError, match="missing required columns"):
            build_clip_graph(clip_graph.normalized_distance, frame)

    def test_rejects_single_clip(self):
        frame = pd.DataFrame({"episode_id": ["a"], "difficulty": [0.1], "cluster": [0]})
        with pytest.raises(ValueError, match="at least 2 clips"):
            build_clip_graph(np.zeros((1, 1)), frame)

    def test_layout_shape_and_determinism(self, clip_graph, context):
        first = force_directed_layout(clip_graph, embedding=None, seed=3)
        second = force_directed_layout(clip_graph, embedding=None, seed=3)
        assert first.shape == (clip_graph.n_clips, 2)
        assert np.allclose(first, second)
        assert np.isfinite(context.layout).all()


class TestEdgeCost:
    CONFIG = GraphConfig(target_step=0.05, step_penalty=0.1)

    def _cost(self, d_from, d_to, dtw=0.0, **mismatches):
        return edge_cost(d_from, d_to, dtw, mismatches, self.CONFIG)

    def test_perfect_ramp_costs_only_the_step_penalty(self):
        cost = self._cost(0.20, 0.25)
        assert cost["ramp"] == pytest.approx(0.0)
        assert cost["interference"] == pytest.approx(0.0)
        assert cost["weight"] == pytest.approx(self.CONFIG.step_penalty)

    def test_stalling_and_leaping_are_penalised_symmetrically(self):
        """|delta - tau| is deliberately two-sided: no progress is as wrong as too much."""
        stall = self._cost(0.20, 0.20)["ramp"]
        leap = self._cost(0.20, 0.30)["ramp"]
        assert stall == pytest.approx(leap)
        assert stall > 0

    def test_backsliding_costs_more_than_an_equal_forward_error(self):
        back = self._cost(0.20, 0.15)["ramp"]
        forward = self._cost(0.20, 0.35)["ramp"]  # same |delta - tau| magnitude
        assert back > forward

    def test_categorical_mismatches_accumulate(self):
        base = self._cost(0.20, 0.25, dtw=0.3)["interference"]
        switched = self._cost(
            0.20, 0.25, dtw=0.3, task_name=True, cluster=True, embodiment=True, source=True
        )["interference"]
        expected = base + sum(
            (self.CONFIG.p_task, self.CONFIG.p_cluster,
             self.CONFIG.p_embodiment, self.CONFIG.p_source)
        )
        assert switched == pytest.approx(expected)

    def test_never_negative_across_a_sweep(self):
        for d_from in np.linspace(0, 1, 11):
            for d_to in np.linspace(0, 1, 11):
                for dtw in (0.0, 0.5, 1.0):
                    assert self._cost(float(d_from), float(d_to), dtw)["weight"] >= 0.0

    def test_config_rejects_negative_weights(self):
        with pytest.raises(ValueError, match="non-negative"):
            GraphConfig(w_interference=-1.0)
        with pytest.raises(ValueError, match="target_step"):
            GraphConfig(target_step=0.0)


class TestDistanceNormalisation:
    def test_bounded_to_unit_interval(self, clip_graph):
        normalized = clip_graph.normalized_distance
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_identical_clips_yield_zeros(self):
        assert np.allclose(normalize_distances(np.zeros((4, 4))), 0.0)


class TestScoping:
    def test_scope_subsets_frame_and_matrix(self, context):
        frame = context.clip_graph.nodes
        wanted = [str(frame["task_name"].iloc[0])]
        scoped, matrix, kept = scope_to_tasks(frame, context.distance_matrix, wanted)
        assert set(scoped["task_name"]) == set(wanted)
        assert matrix.shape == (len(scoped), len(scoped))
        assert len(kept) == len(scoped)

    def test_empty_scope_keeps_everything(self, context):
        frame = context.clip_graph.nodes
        scoped, matrix, kept = scope_to_tasks(frame, context.distance_matrix, None)
        assert len(scoped) == len(frame)
        assert matrix.shape == context.distance_matrix.shape

    def test_unmatched_scope_raises(self, context):
        with pytest.raises(ValueError, match="no clips match"):
            scope_to_tasks(context.clip_graph.nodes, context.distance_matrix, ["nope"])


# --------------------------------------------------------------------------- #
# goal matching
# --------------------------------------------------------------------------- #
class TestGoalMatcher:
    def test_matches_the_named_archetype(self, context):
        """A goal phrased in natural English must find the right task family."""
        match = context.matcher.match("teach it to stir the pot in circles")
        assert match.task_name == "stir"
        assert match.score > 0

    def test_alternates_are_ranked_descending(self, context):
        match = context.matcher.match("wipe the table surface")
        scores = [c.score for c in match.candidates]
        assert scores == sorted(scores, reverse=True)
        assert len(scores) <= 5

    def test_empty_query_degrades_gracefully(self, context):
        match = context.matcher.match("")
        assert not match.is_confident
        assert match.note
        assert 0 <= match.target_index < context.clip_graph.n_clips

    def test_out_of_vocabulary_query_degrades_gracefully(self, context):
        match = context.matcher.match("qqqzzzxxx wobblefrump")
        assert not match.is_confident
        assert match.score == pytest.approx(0.0)
        assert 0 <= match.target_index < context.clip_graph.n_clips

    def test_target_selection_modes_differ(self, context):
        nodes = context.clip_graph.nodes
        hardest = context.matcher.match("stir", target_selection="hardest").target_index
        easiest = context.matcher.match("stir", target_selection="easiest").target_index
        assert (
            float(nodes["difficulty"].iloc[hardest])
            >= float(nodes["difficulty"].iloc[easiest])
        )

    def test_rejects_unknown_target_selection(self, context):
        with pytest.raises(ValueError, match="target_selection"):
            context.matcher.match("stir", target_selection="sideways")

    def test_documents_expand_snake_case(self):
        frame = pd.DataFrame(
            {
                "episode_id": ["a"],
                "task_name": ["yam_fold_tshirt"],
                "task_description": ["fold the black t shirt"],
                "difficulty": [0.5],
            }
        )
        assert "yam fold tshirt" in build_documents(frame)[0]

    def test_survives_missing_language_metadata(self):
        """No task_description at all must not raise -- it just cannot match well."""
        frame = pd.DataFrame(
            {
                "episode_id": ["a", "b"],
                "task_name": ["", ""],
                "difficulty": [0.2, 0.8],
            }
        )
        match = GoalMatcher(frame).match("anything")
        assert not match.is_confident


# --------------------------------------------------------------------------- #
# path search
# --------------------------------------------------------------------------- #
class TestPathfinder:
    def test_path_ends_exactly_at_the_target(self, clip_graph):
        """The curriculum must culminate on the goal, never trail off into rehearsal."""
        for target in (2, 5, clip_graph.n_clips - 1):
            for review_every in (0, 1, 2, 4):
                path = find_curriculum_path(
                    clip_graph, target, PathConfig(review_every=review_every)
                )
                assert path.route[-1] == target
                assert path.clips[-1] == target
                assert not path.is_review[-1]

    def test_path_starts_in_the_easy_pool(self, clip_graph):
        path = find_curriculum_path(clip_graph, clip_graph.n_clips - 1)
        assert path.clips[0] in clip_graph.start_clips

    def test_introductions_never_descend_beyond_tolerance(self, clip_graph):
        """The headline guarantee: the ramp cannot run backwards."""
        difficulty = clip_graph.nodes["difficulty"].astype(float).to_numpy()
        path = find_curriculum_path(clip_graph, clip_graph.n_clips - 1)
        introduced = [c for c, r in zip(path.clips, path.is_review) if not r]
        steps = np.diff([difficulty[c] for c in introduced])
        if steps.size:
            assert steps.min() >= -clip_graph.config.backslide_tolerance - 1e-9

    def test_reviews_only_repeat_already_seen_clips(self, clip_graph):
        path = find_curriculum_path(
            clip_graph, clip_graph.n_clips - 1, PathConfig(review_every=2)
        )
        seen = set()
        for clip, review in zip(path.clips, path.is_review):
            if review:
                assert clip in seen, "a review must rehearse something already introduced"
            seen.add(clip)

    def test_review_every_zero_inserts_none(self, clip_graph):
        path = find_curriculum_path(
            clip_graph, clip_graph.n_clips - 1, PathConfig(review_every=0)
        )
        assert path.n_reviews == 0
        assert path.clips == path.route

    def test_reviews_respect_the_cap(self, clip_graph):
        path = find_curriculum_path(
            clip_graph, clip_graph.n_clips - 1, PathConfig(review_every=1, max_reviews=2)
        )
        assert path.n_reviews <= 2

    def test_astar_returns_a_valid_path(self, clip_graph):
        target = clip_graph.n_clips - 1
        astar = find_curriculum_path(clip_graph, target, PathConfig(search="astar"))
        assert astar.route[-1] == target
        assert astar.clips[0] in clip_graph.start_clips

    def test_table_has_one_row_per_step(self, clip_graph):
        path = find_curriculum_path(clip_graph, clip_graph.n_clips - 1)
        assert len(path.table) == path.n_steps
        assert list(path.table["step"]) == list(range(1, path.n_steps + 1))

    def test_review_rows_blank_the_ramp_cost(self, clip_graph):
        """A deliberate dip has no ramp to score, so reporting one would mislead."""
        path = find_curriculum_path(
            clip_graph, clip_graph.n_clips - 1, PathConfig(review_every=2)
        )
        reviews = path.table[path.table["is_review"]]
        if not reviews.empty:
            assert reviews["ramp_cost"].isna().all()
            assert reviews["edge_weight"].isna().all()
            # Interference is real for a review: interleaving does switch context.
            assert reviews["interference_cost"].notna().all()

    def test_cost_terms_sum_to_the_search_cost(self, clip_graph):
        path = find_curriculum_path(clip_graph, clip_graph.n_clips - 1)
        assert sum(path.cost_terms.values()) == pytest.approx(path.search_cost, rel=1e-6)

    def test_rejects_a_non_clip_target(self, clip_graph):
        with pytest.raises(ValueError, match="not a clip"):
            search_route(clip_graph, START)
        with pytest.raises(ValueError, match="not a clip"):
            search_route(clip_graph, 10_000)

    def test_rejects_bad_config(self):
        with pytest.raises(ValueError, match="search must be"):
            PathConfig(search="bfs")
        with pytest.raises(ValueError, match="review_every"):
            PathConfig(review_every=-1)

    def test_insert_reviews_on_empty_route(self, clip_graph):
        clips, flags = insert_reviews([], clip_graph, PathConfig())
        assert clips == [] and flags == []


# --------------------------------------------------------------------------- #
# proxy metrics
# --------------------------------------------------------------------------- #
class TestPathMetrics:
    def test_perfect_ramp_scores_one(self):
        report = difficulty_monotonicity([0.0, 0.1, 0.2, 0.3, 0.4])
        assert report["spearman"] == pytest.approx(1.0)
        assert report["frac_nondecreasing"] == pytest.approx(1.0)
        assert report["n_large_jumps"] == 0

    def test_descending_order_scores_minus_one(self):
        report = difficulty_monotonicity([0.4, 0.3, 0.2, 0.1])
        assert report["spearman"] == pytest.approx(-1.0)
        assert report["frac_nondecreasing"] == pytest.approx(0.0)

    def test_large_jumps_are_counted(self):
        report = difficulty_monotonicity([0.0, 0.05, 0.9])
        assert report["n_large_jumps"] == 1
        assert report["max_jump"] == pytest.approx(0.85)

    def test_degenerate_inputs_stay_numeric(self):
        for values in ([], [0.5], [0.5, 0.5, 0.5]):
            report = difficulty_monotonicity(values)
            assert all(np.isfinite(v) for v in report.values())

    def test_interference_counts_switches(self, clip_graph):
        nodes = clip_graph.nodes
        same_task = nodes.index[nodes["task_name"] == nodes["task_name"].iloc[0]].tolist()
        if len(same_task) >= 2:
            profile = interference_profile(
                same_task[:2], nodes, clip_graph.normalized_distance
            )
            assert profile["task_switch_rate"] == pytest.approx(0.0)

    def test_coverage_truncates_at_the_target(self, clip_graph):
        nodes = clip_graph.nodes
        order = list(range(clip_graph.n_clips))
        truncated = coverage(order, nodes, target_index=order[2])
        full = coverage(order, nodes, target_index=None)
        assert truncated["n_clips_to_target"] == 3
        assert truncated["cluster_coverage"] <= full["cluster_coverage"]

    def test_coverage_curve_is_monotonic(self, clip_graph):
        curve = coverage_curve(list(range(clip_graph.n_clips)), clip_graph.nodes)
        assert curve == sorted(curve)
        assert curve[-1] == clip_graph.nodes["cluster"].nunique()

    def test_redundancy_reports_repeats(self, clip_graph, context):
        report = redundancy([0, 1, 0, 1], context.distance_matrix)
        assert report["distinct_ratio"] == pytest.approx(0.5)

    def test_redundancy_of_a_single_clip_is_neutral(self, context):
        report = redundancy([3], context.distance_matrix)
        assert report["distinct_ratio"] == pytest.approx(1.0)

    def test_path_report_covers_every_family(self, clip_graph, context):
        path = find_curriculum_path(clip_graph, clip_graph.n_clips - 1)
        report = path_report(
            path.clips, clip_graph, context.distance_matrix, path.target_index, path.is_review
        )
        for key in (
            "spearman", "frac_nondecreasing", "task_switch_rate",
            "cluster_coverage", "frac_consecutive_near_duplicate", "n_steps",
        ):
            assert key in report and np.isfinite(report[key])

    def test_comparison_has_a_row_per_ordering(self, clip_graph, context):
        path = find_curriculum_path(clip_graph, clip_graph.n_clips - 1)
        frame = compare_orderings(path, clip_graph, context.distance_matrix, n_seeds=5)
        assert "Curriculum path" in frame.index
        assert "Random order (same clips)" in frame.index
        assert "Coreset prefix (same size)" in frame.index
        assert frame.loc["Curriculum path", "spearman"] is not None

    def test_path_beats_random_ordering_on_monotonicity(self, clip_graph, context):
        """The claim the whole design rests on, asserted rather than eyeballed.

        Compared against a reshuffle of the path's *own* clips, so this isolates the
        value of the ordering rather than of the selection.
        """
        path = find_curriculum_path(clip_graph, clip_graph.n_clips - 1)
        frame = compare_orderings(path, clip_graph, context.distance_matrix, n_seeds=20)
        assert (
            frame.loc["Curriculum path", "spearman"]
            > frame.loc["Random order (same clips)", "spearman"]
        )
        # Total absolute variation, not max_jump: over a fixed clip set a descending
        # ordering posts a small max *upward* step while being maximally jagged, so
        # max_jump cannot separate a good ordering from a reversed one.
        assert (
            frame.loc["Curriculum path", "mean_abs_step"]
            <= frame.loc["Random order (same clips)", "mean_abs_step"]
        )
        assert (
            frame.loc["Curriculum path", "max_abs_step"]
            <= frame.loc["Random order (same clips)", "max_abs_step"]
        )


# --------------------------------------------------------------------------- #
# integration
# --------------------------------------------------------------------------- #
class TestPathFinderIntegration:
    def test_end_to_end_on_synthetic_data(self, context):
        match, path = context.find("stir the pot")
        assert path.route
        assert path.clips[-1] == match.target_index or path.clips[-1] == path.target_index
        assert len(path.table) == path.n_steps

    def test_scoped_graph_routes_only_within_scope(self, result):
        tasks = ["stir", "wipe"]
        scoped = build_path_finder(result, task_names=tasks)
        assert set(scoped.clip_graph.nodes["task_name"]) <= set(tasks)
        path = find_curriculum_path(scoped.clip_graph, scoped.clip_graph.n_clips - 1)
        assert all(
            scoped.clip_graph.nodes["task_name"].iloc[c] in tasks for c in path.clips
        )

    def test_build_path_finder_needs_two_episodes(self, tmp_path):
        empty = run_pipeline(str(tmp_path), cache_dir=None, verbose=False)
        with pytest.raises(ValueError, match="at least 2"):
            build_path_finder(empty)

    def test_cli_writes_artifacts(self, synth_dir, tmp_path, monkeypatch):
        import find_path

        out = tmp_path / "cli_out"
        monkeypatch.setattr(
            sys, "argv",
            ["find_path.py", "stir the pot", "--data-dir", str(synth_dir),
             "--out", str(out), "--no-cache", "--quiet", "--seeds", "5"],
        )
        assert find_path.main() == 0
        for name in ("path.csv", "path_metrics.json", "path_graph.html"):
            assert (out / name).exists(), f"{name} was not written"

    @requires_real_data
    def test_real_data_goal_match(self):
        """On the real dataset, the demo query must resolve to the folding task."""
        real = run_pipeline(str(REAL_DATA_DIR), cache_dir=None, verbose=False)
        ctx = build_path_finder(real)
        match = ctx.matcher.match("teach the robot to fold a shirt")
        assert "fold" in match.task_name.lower(), match.task_name
        assert match.is_confident

    @requires_real_data
    def test_real_data_path_reaches_target(self):
        real = run_pipeline(str(REAL_DATA_DIR), cache_dir=None, verbose=False)
        ctx = build_path_finder(real)
        match, path = ctx.find("teach the robot to fold a shirt")
        assert path.clips[-1] == path.target_index
        assert path.clips[0] in ctx.clip_graph.start_clips


class TestPathFinderDashboard:
    def test_new_tabs_render(self, synth_dir):
        """Streamlit renders exceptions into the page rather than raising them."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=900)
        at.run()
        assert not at.exception

        at.text_input[0].set_value(str(synth_dir))
        at.button[0].click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert len(at.tabs) == 9
        assert any("Path finder" in t.label for t in at.tabs)
        assert any("Path validation" in t.label for t in at.tabs)
        assert any(m.label == "Clips in curriculum" for m in at.metric)
