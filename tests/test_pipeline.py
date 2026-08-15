"""Tests for the EgoVerse diversity & curriculum engine.

Fixtures are written in the *real* `processed_v3` schema (Zarr v3, dot-separated
keys, chunk-padded arrays, sentinel frames) via
``scripts/generate_synthetic_data.py``, so the tests exercise the same quirks the
production data has rather than an idealised format.

Tests that need the real dataset are skipped when ``data/`` is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_synthetic_data import build_pose, generate_dataset, make_xyz, write_episode  # noqa: E402

from src.cluster_mapper import (  # noqa: E402
    calculate_diversity_score,
    cluster_medoids,
    cluster_precomputed,
    diversity_report,
    find_redundant_pairs,
    nearest_neighbor_distances,
    pairwise_values,
    project_umap,
    suggest_n_clusters,
)
from src.curriculum import (  # noqa: E402
    build_curriculum,
    coreset_order,
    difficulty_scores,
    features_dataframe,
    kinematic_features,
    robust_zscore,
)
from src.diversity_engine import (  # noqa: E402
    DTWConfig,
    compute_dtw_matrix,
    normalize_trajectory,
    preprocess_trajectories,
    resample_trajectory,
)
from src.loader import clean_pose_array, detect_unit_scale, load_zarr_trajectories  # noqa: E402

REAL_DATA_DIR = REPO_ROOT / "data"
requires_real_data = pytest.mark.skipif(
    not REAL_DATA_DIR.exists() or not any(REAL_DATA_DIR.glob("*.zarr")),
    reason="real EgoVerse episodes not fetched; run scripts/fetch_egoverse_data.py",
)


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory) -> Path:
    """A synthetic dataset including pathological episodes."""
    out = tmp_path_factory.mktemp("synth")
    generate_dataset(out, n_episodes=16, n_duplicates=3, seed=7, inject_edge_cases=True)
    return out


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #
class TestLoader:
    def test_truncates_chunk_padding(self, tmp_path):
        """The zero tail past total_frames must never reach the trajectory.

        This is the single most consequential loader behaviour: every real episode is
        chunk-padded, and reading the tail appends a fabricated collapse to the origin.
        """
        xyz = make_xyz("reach", np.random.default_rng(0), steps=250)
        path = write_episode(
            tmp_path / "ep.zarr", {"right.obs_ee_pose": build_pose(xyz)}, chunk_pad=True
        )
        import zarr

        stored = zarr.open_group(str(path), mode="r")["right.obs_ee_pose"]
        assert stored.shape[0] == 300, "fixture should be padded to a chunk boundary"

        ds = load_zarr_trajectories(str(tmp_path), verbose=False)
        assert len(ds) == 1
        assert ds.trajectories[0].shape[0] == 250
        # The padded tail is exactly zero; a correctly truncated trajectory is not.
        assert not np.allclose(ds.trajectories[0][-1], 0.0)

    def test_drops_missing_frame_sentinels(self, tmp_path):
        rng = np.random.default_rng(1)
        xyz = make_xyz("wipe", rng, steps=400)
        write_episode(
            tmp_path / "ep.zarr",
            {"right.obs_ee_pose": build_pose(xyz, missing_ratio=0.25, rng=rng)},
        )
        ds = load_zarr_trajectories(str(tmp_path), verbose=False)
        assert len(ds) == 1
        traj = ds.trajectories[0]
        assert traj.shape[0] == pytest.approx(400 * 0.75, abs=2)
        assert not np.any(np.all(traj == 0.0, axis=1)), "no sentinel rows may survive"
        assert ds.metadata[0]["missing_frame_ratio"] == pytest.approx(0.25, abs=0.02)

    def test_rejects_mostly_missing_and_dead_streams(self, synth_dir):
        ds = load_zarr_trajectories(str(synth_dir), verbose=False)
        skipped = {ep: reason for ep, reason in ds.skipped}
        assert any("edge_dead" in ep for ep in skipped)
        assert any("edge_mostly_missing" in ep for ep in skipped)
        assert all("edge_dead" not in ep for ep in ds.episode_ids)

    def test_detects_millimetre_units(self, synth_dir):
        ds = load_zarr_trajectories(str(synth_dir), verbose=False)
        idx = [i for i, e in enumerate(ds.episode_ids) if "millimetres" in e]
        assert idx, "millimetre edge-case episode should load"
        meta = ds.metadata[idx[0]]
        assert meta["unit_scale"] == pytest.approx(1e-3)
        # After conversion the workspace span must be metre-scale, not 1000x that.
        span = np.linalg.norm(ds.trajectories[idx[0]].max(0) - ds.trajectories[idx[0]].min(0))
        assert 0.01 < span < 5.0

    def test_detect_unit_scale(self):
        assert detect_unit_scale(np.array([[0.3, 0.1, 0.2]])) == 1.0
        assert detect_unit_scale(np.array([[300.0, 100.0, 200.0]])) == 1e-3

    def test_clean_pose_array_raises_on_static(self):
        static = np.tile(np.array([0.5, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0]), (200, 1))
        with pytest.raises(ValueError, match="static"):
            clean_pose_array(static, 200, min_length=30)

    def test_arm_modes(self, synth_dir):
        auto = load_zarr_trajectories(str(synth_dir), arm="auto", verbose=False)
        both = load_zarr_trajectories(str(synth_dir), arm="both", verbose=False)
        assert all(t.shape[1] == 3 for t in auto.trajectories)
        assert all(t.shape[1] == 6 for t in both.trajectories)
        # 'both' requires two usable arms, so it can only ever load fewer episodes.
        assert len(both) <= len(auto)
        assert {m["arm_used"] for m in both.metadata} <= {"both"}

    def test_auto_picks_more_active_arm(self, tmp_path):
        rng = np.random.default_rng(3)
        active = make_xyz("zigzag_search", rng, steps=300)
        # A barely-moving arm: above the static threshold, but far less active.
        quiet = np.cumsum(rng.normal(0, 2e-4, (300, 3)), axis=0) + np.array([0.4, 0.0, 0.2])
        write_episode(
            tmp_path / "ep.zarr",
            {
                "left.obs_ee_pose": build_pose(quiet),
                "right.obs_ee_pose": build_pose(active),
            },
        )
        ds = load_zarr_trajectories(str(tmp_path), arm="auto", verbose=False)
        assert ds.metadata[0]["arm_used"] == "right"

    def test_missing_directory_is_not_fatal(self):
        ds = load_zarr_trajectories("/nonexistent/path", verbose=False)
        assert len(ds) == 0 and ds.skipped == []

    def test_metadata_and_labels(self, synth_dir):
        ds = load_zarr_trajectories(str(synth_dir), verbose=False)
        assert len(ds.task_labels) == len(ds)
        assert set(ds.field_values("source")) == {"synth"}
        assert (ds.fps > 0).all()


# --------------------------------------------------------------------------- #
# diversity engine
# --------------------------------------------------------------------------- #
class TestDiversityEngine:
    @pytest.fixture
    def trajectories(self):
        rng = np.random.default_rng(0)
        return [make_xyz(a, rng, steps=n) for a, n in
                zip(("reach", "wipe", "stir", "zigzag_search"), (120, 200, 160, 240))]

    def test_matrix_is_metric_shaped(self, trajectories):
        D = compute_dtw_matrix(trajectories, DTWConfig(n_jobs=1))
        assert D.shape == (4, 4)
        assert np.allclose(D, D.T), "distance matrix must be symmetric"
        assert np.allclose(np.diag(D), 0.0), "diagonal must be exactly zero"
        assert np.isfinite(D).all() and (D >= 0).all()

    def test_identical_trajectories_have_zero_distance(self):
        rng = np.random.default_rng(2)
        xyz = make_xyz("pick_place", rng, steps=150)
        D = compute_dtw_matrix([xyz, xyz.copy()], DTWConfig(n_jobs=1))
        assert D[0, 1] == pytest.approx(0.0, abs=1e-9)

    def test_empty_and_single_input(self):
        assert compute_dtw_matrix([]).shape == (0, 0)
        assert compute_dtw_matrix([np.zeros((10, 3))]).shape == (1, 1)

    def test_resample_preserves_endpoints_and_shape(self):
        traj = make_xyz("reach", np.random.default_rng(4), steps=500)
        out = resample_trajectory(traj, 100)
        assert out.shape == (100, 3)
        assert np.allclose(out[0], traj[0]) and np.allclose(out[-1], traj[-1])

    def test_max_length_caps_cost(self, trajectories):
        processed = preprocess_trajectories(trajectories, DTWConfig(max_length=50))
        assert all(p.shape[0] <= 50 for p in processed)

    def test_normalize_modes(self):
        traj = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 9.0], [5.0, 6.0, 3.0]])
        assert np.allclose(normalize_trajectory(traj, "center").mean(axis=0), 0.0)
        assert np.allclose(normalize_trajectory(traj, "zscore").std(axis=0), 1.0)
        assert np.allclose(normalize_trajectory(traj, "none"), traj)

    def test_zscore_handles_zero_variance_axis(self):
        """A frozen axis must not produce NaN/inf, which would poison the whole matrix."""
        traj = np.column_stack([np.linspace(0, 1, 50), np.zeros(50), np.zeros(50)])
        out = normalize_trajectory(traj, "zscore")
        assert np.isfinite(out).all()

    def test_centering_removes_workspace_offset(self):
        """Two identical motions in different workspace corners must coincide."""
        rng = np.random.default_rng(5)
        xyz = make_xyz("wipe", rng, steps=200)
        shifted = xyz + np.array([10.0, -5.0, 3.0])
        centered = compute_dtw_matrix([xyz, shifted], DTWConfig(normalize="center", n_jobs=1))
        raw = compute_dtw_matrix([xyz, shifted], DTWConfig(normalize="none", n_jobs=1))
        assert centered[0, 1] == pytest.approx(0.0, abs=1e-9)
        # Uncentred, the pair is separated by the workspace offset alone. Compare
        # against the offset rather than a magic constant, since length normalisation
        # rescales the absolute magnitude.
        # Sqrt length normalisation gives the distance a physical interpretation:
        # two paths separated by a constant offset d cost d*sqrt(T)/sqrt(T) == d.
        # The normalised DTW distance is therefore in metres, like the input.
        offset = np.linalg.norm([10.0, -5.0, 3.0])
        assert raw[0, 1] == pytest.approx(offset, rel=0.02)

    def test_zscore_is_scale_invariant_but_center_is_not(self):
        rng = np.random.default_rng(6)
        xyz = make_xyz("stir", rng, steps=200)
        scaled = (xyz - xyz.mean(0)) * 10.0 + xyz.mean(0)
        z = compute_dtw_matrix([xyz, scaled], DTWConfig(normalize="zscore", n_jobs=1))
        c = compute_dtw_matrix([xyz, scaled], DTWConfig(normalize="center", n_jobs=1))
        assert z[0, 1] == pytest.approx(0.0, abs=1e-8)
        assert c[0, 1] > 1e-3, "centering must retain motion extent"

    def test_length_normalization_is_length_invariant(self):
        """The same pair of shapes must score the same distance at any duration.

        tslearn's DTW is a root-sum-square, so raw cost grows as sqrt(T). Dividing by
        sqrt(mean length) cancels that; dividing by the mean length itself would
        overcorrect and make long episodes look artificially similar.
        """
        rng = np.random.default_rng(0)
        a, b = make_xyz("reach", rng, steps=200), make_xyz("wipe", rng, steps=200)
        cfg = DTWConfig(length_normalize=True, max_length=None, n_jobs=1)

        distances = []
        for length in (100, 400, 1600):
            pair = [resample_trajectory(a, length), resample_trajectory(b, length)]
            distances.append(compute_dtw_matrix(pair, cfg)[0, 1])

        spread = (max(distances) - min(distances)) / np.mean(distances)
        assert spread < 0.05, f"distance drifted {spread:.1%} across a 16x length range"

    def test_unnormalized_distance_grows_with_length(self):
        """The bias that length normalisation exists to remove."""
        rng = np.random.default_rng(0)
        a, b = make_xyz("reach", rng, steps=200), make_xyz("wipe", rng, steps=200)
        cfg = DTWConfig(length_normalize=False, max_length=None, n_jobs=1)
        short = compute_dtw_matrix(
            [resample_trajectory(a, 100), resample_trajectory(b, 100)], cfg
        )[0, 1]
        long = compute_dtw_matrix(
            [resample_trajectory(a, 1600), resample_trajectory(b, 1600)], cfg
        )[0, 1]
        assert long > 3 * short

    def test_disk_cache_round_trip(self, trajectories, tmp_path):
        cfg = DTWConfig(n_jobs=1)
        first = compute_dtw_matrix(trajectories, cfg, cache_dir=str(tmp_path))
        cached = list(tmp_path.glob("dtw_*.npy"))
        assert len(cached) == 1
        second = compute_dtw_matrix(trajectories, cfg, cache_dir=str(tmp_path))
        assert np.array_equal(first, second)

    def test_cache_key_tracks_config(self, trajectories, tmp_path):
        compute_dtw_matrix(trajectories, DTWConfig(normalize="center", n_jobs=1), cache_dir=str(tmp_path))
        compute_dtw_matrix(trajectories, DTWConfig(normalize="zscore", n_jobs=1), cache_dir=str(tmp_path))
        assert len(list(tmp_path.glob("dtw_*.npy"))) == 2, "config must change the cache key"

    def test_rejects_invalid_config(self):
        with pytest.raises(ValueError):
            DTWConfig(normalize="bogus")


# --------------------------------------------------------------------------- #
# cluster mapper
# --------------------------------------------------------------------------- #
class TestClusterMapper:
    @pytest.fixture
    def distance_matrix(self):
        """Three tight, well-separated groups."""
        rng = np.random.default_rng(0)
        centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        pts = np.vstack([c + rng.normal(0, 0.2, (6, 2)) for c in centers])
        D = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
        np.fill_diagonal(D, 0.0)
        return D

    def test_diversity_score_excludes_diagonal(self, distance_matrix):
        """The naive mean over the full matrix is deflated by the N structural zeros."""
        n = distance_matrix.shape[0]
        naive = float(distance_matrix.mean())
        correct = calculate_diversity_score(distance_matrix)
        assert correct > naive
        assert correct == pytest.approx(distance_matrix.sum() / (n * (n - 1)))

    def test_pairwise_values_count(self, distance_matrix):
        n = distance_matrix.shape[0]
        assert pairwise_values(distance_matrix).size == n * (n - 1) // 2

    def test_clustering_recovers_groups(self, distance_matrix):
        labels = cluster_precomputed(distance_matrix, n_clusters=3)
        assert len(np.unique(labels)) == 3
        # Each block of 6 consecutive points is one planted group.
        for block in range(3):
            assert len(set(labels[block * 6 : (block + 1) * 6])) == 1

    def test_ward_linkage_rejected(self, distance_matrix):
        """Ward needs Euclidean coordinates; silently accepting a distance matrix
        would produce meaningless clusters."""
        with pytest.raises(ValueError, match="ward"):
            cluster_precomputed(distance_matrix, 3, linkage="ward")

    def test_cluster_count_clamped(self, distance_matrix):
        n = distance_matrix.shape[0]
        assert len(cluster_precomputed(distance_matrix, n_clusters=999)) == n
        assert len(np.unique(cluster_precomputed(distance_matrix, n_clusters=1))) == 1

    def test_suggest_n_clusters_finds_planted_structure(self, distance_matrix):
        best_k, scores = suggest_n_clusters(distance_matrix, 2, 6)
        assert best_k == 3
        assert set(scores) <= {2, 3, 4, 5, 6}

    def test_medoids_are_real_episodes(self, distance_matrix):
        labels = cluster_precomputed(distance_matrix, 3)
        medoids = cluster_medoids(distance_matrix, labels)
        assert len(medoids) == 3
        for label, idx in medoids.items():
            assert labels[idx] == label

    def test_umap_small_sample_does_not_crash(self):
        """UMAP's spectral init raises for tiny N; the guard must fall back cleanly."""
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(4, 2))
        D = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
        np.fill_diagonal(D, 0.0)
        emb = project_umap(D, n_components=3)
        assert emb.shape == (4, 3) and np.isfinite(emb).all()

    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_umap_degenerate_sizes(self, n):
        emb = project_umap(np.zeros((n, n)), n_components=3)
        assert emb.shape == (n, 3)

    def test_nearest_neighbor_and_redundancy(self):
        pts = np.array([[0.0], [0.001], [5.0], [10.0]])
        D = np.abs(pts - pts.T)
        nn = nearest_neighbor_distances(D)
        assert nn[0] == pytest.approx(0.001) and nn[3] == pytest.approx(5.0)
        pairs = find_redundant_pairs(D, quantile=0.2)
        assert pairs and pairs[0][:2] == (0, 1)

    def test_diversity_report_keys(self, distance_matrix):
        labels = cluster_precomputed(distance_matrix, 3)
        report = diversity_report(distance_matrix, labels)
        for key in ("diversity_score", "median_pairwise", "dispersion",
                    "mean_nn_distance", "redundancy_ratio", "silhouette", "cluster_balance"):
            assert key in report and np.isfinite(report[key])
        assert 0.0 <= report["cluster_balance"] <= 1.0 + 1e-9

    def test_empty_report_is_safe(self):
        report = diversity_report(np.zeros((0, 0)))
        assert report["diversity_score"] == 0.0


# --------------------------------------------------------------------------- #
# curriculum
# --------------------------------------------------------------------------- #
class TestCurriculum:
    def test_straight_line_is_minimally_tortuous(self):
        line = np.column_stack([np.linspace(0, 1, 100), np.zeros(100), np.zeros(100)])
        feats = kinematic_features(line)
        assert feats["tortuosity"] == pytest.approx(1.0, abs=1e-6)
        assert feats["path_length"] == pytest.approx(1.0, abs=1e-6)
        assert feats["reversal_rate"] == pytest.approx(0.0)

    def test_zigzag_has_reversals(self):
        t = np.arange(100)
        zig = np.column_stack([t * 0.001, ((-1.0) ** t) * 0.05, np.zeros(100)])
        assert kinematic_features(zig)["reversal_rate"] > 0.5

    def test_closed_loop_tortuosity_is_finite(self):
        """Net displacement ~0 would divide by zero without the clamp."""
        theta = np.linspace(0, 2 * np.pi, 200)
        circle = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(200)])
        feats = kinematic_features(circle)
        assert np.isfinite(feats["tortuosity"]) and feats["tortuosity"] > 10

    def test_normalized_jerk_is_scale_and_duration_free(self):
        """The point of the dimensionless metric: same motion, different units/rate."""
        rng = np.random.default_rng(0)
        xyz = make_xyz("pick_place", rng, steps=300)
        base = kinematic_features(xyz, dt=1 / 30)["log_normalized_jerk"]
        scaled = kinematic_features(xyz * 100.0, dt=1 / 30)["log_normalized_jerk"]
        assert scaled == pytest.approx(base, rel=1e-6)

    def test_short_trajectories_do_not_produce_nan(self):
        for n in (1, 2, 3):
            feats = kinematic_features(np.zeros((n, 3)))
            assert all(np.isfinite(v) for v in feats.values())

    def test_robust_zscore_handles_constant_input(self):
        assert np.allclose(robust_zscore(np.full(10, 3.0)), 0.0)

    def test_rank_scaling_spreads_uniformly(self):
        """Min-max collapses a skewed distribution; rank scaling is why we default to it."""
        import pandas as pd

        skewed = pd.DataFrame({"path_length": np.array([1, 2, 3, 4, 5, 1000.0])})
        weights = {"path_length": 1.0}
        ranked = difficulty_scores(skewed, weights, scaling="rank")
        minmax = difficulty_scores(skewed, weights, scaling="minmax")
        assert ranked.min() == 0.0 and ranked.max() == 1.0
        assert np.all(np.diff(np.sort(ranked)) > 0.05), "ranks must be evenly spread"
        assert np.sort(minmax)[-2] < 0.05, "min-max compresses everything below the outlier"

    def test_difficulty_bounds(self):
        rng = np.random.default_rng(0)
        trajs = [make_xyz(a, rng, steps=200) for a in
                 ("reach", "wipe", "stir", "zigzag_search", "pour")]
        scores = difficulty_scores(features_dataframe(trajs))
        assert scores.min() >= 0.0 and scores.max() <= 1.0

    def test_coreset_order_is_a_permutation(self):
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(12, 2))
        D = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
        np.fill_diagonal(D, 0.0)
        order = coreset_order(D)
        assert sorted(order) == list(range(12))

    def test_coreset_picks_far_points_first(self):
        """A prefix of the coreset order should cover the extremes, not one cluster."""
        pts = np.array([[0.0], [0.1], [0.2], [50.0], [100.0]])
        D = np.abs(pts - pts.T)
        order = coreset_order(D)
        assert set(order[:3]) >= {3, 4}, "the two far outliers must be picked early"

    def test_dt_broadcasting(self):
        rng = np.random.default_rng(0)
        trajs = [make_xyz("reach", rng, steps=100) for _ in range(3)]
        df = features_dataframe(trajs, dt=np.array([1 / 30, 1 / 60, 1 / 30]))
        assert len(df) == 3
        with pytest.raises(ValueError, match="dt has"):
            features_dataframe(trajs, dt=np.array([1 / 30, 1 / 60]))

    def test_build_curriculum_structure(self):
        rng = np.random.default_rng(0)
        trajs = [make_xyz(a, rng, steps=150) for a in
                 ("reach", "reach", "wipe", "wipe", "zigzag_search", "zigzag_search")]
        D = compute_dtw_matrix(trajs, DTWConfig(n_jobs=1))
        labels = cluster_precomputed(D, 3)
        ids = [f"ep{i}" for i in range(6)]
        cur = build_curriculum(D, labels, trajs, ids, dt=1 / 30)

        assert list(cur["curriculum_rank"]) == list(range(1, 7))
        assert sorted(cur["coreset_rank"]) == list(range(1, 7))
        assert cur["stage"].min() == 1
        assert set(cur["episode_id"]) == set(ids)
        assert int(cur["is_cluster_medoid"].sum()) == len(np.unique(labels))
        # Stages must be ordered easiest-first by mean difficulty.
        means = cur.groupby("stage")["difficulty"].mean()
        assert list(means) == sorted(means)

    def test_build_curriculum_merges_metadata(self):
        rng = np.random.default_rng(0)
        trajs = [make_xyz("reach", rng, steps=120) for _ in range(3)]
        D = compute_dtw_matrix(trajs, DTWConfig(n_jobs=1))
        meta = [{"task_name": f"t{i}", "source": "synth"} for i in range(3)]
        cur = build_curriculum(D, np.zeros(3, int), trajs, ["a", "b", "c"], metadata=meta)
        assert "task_name" in cur.columns and set(cur["source"]) == {"synth"}


# --------------------------------------------------------------------------- #
# integration
# --------------------------------------------------------------------------- #
class TestIntegration:
    def test_synthetic_end_to_end(self, synth_dir):
        from src.pipeline import run_pipeline

        result = run_pipeline(str(synth_dir), n_clusters=4, cache_dir=None, verbose=False)
        assert result.n_episodes >= 10
        assert result.distance_matrix.shape == (result.n_episodes,) * 2
        assert result.embedding.shape == (result.n_episodes, 3)
        assert len(result.curriculum) == result.n_episodes
        assert result.report["diversity_score"] > 0
        frame = result.frame()
        assert len(frame) == result.n_episodes
        assert not frame["difficulty"].isna().any()

    def test_pipeline_handles_empty_directory(self, tmp_path):
        from src.pipeline import run_pipeline

        result = run_pipeline(str(tmp_path), cache_dir=None, verbose=False)
        assert result.n_episodes == 0
        assert result.curriculum.empty

    @requires_real_data
    def test_real_data_end_to_end(self):
        from src.pipeline import run_pipeline

        result = run_pipeline(str(REAL_DATA_DIR), cache_dir=None, verbose=False)
        assert result.n_episodes >= 2
        assert np.allclose(result.distance_matrix, result.distance_matrix.T)
        assert len(result.curriculum) == result.n_episodes

    @requires_real_data
    def test_real_data_clustering_recovers_task_labels(self):
        """The pipeline's headline claim, asserted rather than eyeballed.

        Clustering is unsupervised over end-effector motion, so agreement with the
        human-authored `task_name` is evidence the DTW metric tracks behaviour.

        Uses `zscore` rather than the `center` default, because on a multi-embodiment
        dataset `center` does not work -- see
        `test_center_normalisation_is_outlier_dominated_at_scale` immediately below, which
        pins that failure down. Motion *extent* differs ~5x between a YAM arm confined to a
        tabletop and a head-mounted Aria recording someone walking a room, so under
        `center` (which preserves extent) distance measures embodiment rather than skill.
        """
        from src.pipeline import run_pipeline

        result = run_pipeline(
            str(REAL_DATA_DIR), normalize="zscore", cache_dir=".cache", verbose=False
        )
        ari = result.agreement.get("task_name")
        assert ari is not None
        assert ari > 0.5, f"expected clusters to track task labels, got ARI={ari:.3f}"
        # And the labels must be usable downstream, not just agree on paper.
        counts = np.bincount(result.labels)
        assert counts.max() / counts.sum() <= 0.6, (
            f"one cluster holds {counts.max()}/{counts.sum()} episodes; skill families "
            "would carry no information"
        )

    @requires_real_data
    def test_center_normalisation_is_outlier_dominated_at_scale(self):
        """Regression guard on a real limitation, so it cannot be rediscovered silently.

        `center` normalisation preserves motion extent, which is correct within one
        embodiment and wrong across several. On this dataset it yields a heavy-tailed
        distance distribution, and agglomerative linkage then peels off single outliers
        instead of splitting the bulk, so every k leaves one cluster holding nearly
        everything. `tail_ratio` is the detector; silhouette is not -- it *prefers* this
        matrix. If a future change fixes `center` at scale, this test should start failing
        and be removed.
        """
        from src.pipeline import run_pipeline

        result = run_pipeline(
            str(REAL_DATA_DIR), normalize="center", cache_dir=".cache", verbose=False
        )
        if result.n_episodes < 100:
            pytest.skip("this pathology only appears on the scaled multi-source dataset")
        assert result.report["tail_ratio"] > 2.0, (
            "expected a heavy-tailed pairwise distribution under `center`; "
            f"got tail_ratio={result.report['tail_ratio']:.2f}"
        )
        counts = np.bincount(result.labels)
        assert counts.max() / counts.sum() > 0.6, (
            "expected the documented clustering collapse under `center` at scale"
        )


class TestDashboard:
    def test_app_runs_without_exceptions(self, synth_dir):
        """Executes app.py headlessly; Streamlit swallows errors into the page."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=600)
        at.run()
        assert not at.exception

        at.text_input[0].set_value(str(synth_dir))
        at.button[0].click().run()
        assert not at.exception, [e.value for e in at.exception]
        # Subset A/B, then the two Track 1 path-finder tabs, then six Track 2 tabs.
        assert len(at.tabs) == 9
        assert at.tabs[0].label.endswith("Subset A/B")
        assert any(m.label == "Episodes analysed" for m in at.metric)
