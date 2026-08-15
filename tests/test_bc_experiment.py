"""Tests for the behaviour-cloning ordering experiment.

The experiment exists to answer "does the curriculum order actually help a model", so
the tests concentrate on the things that would silently invalidate that answer: a
validation set that moves between arms, ordering that isn't really applied, or a summary
that reports a win when the arms are tied.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.bc_experiment import (  # noqa: E402
    ORDERINGS,
    ArmResult,
    TrainConfig,
    build_windows,
    order_episodes,
    split_episodes,
    summarise,
    verdict,
    write_report,
)

torch = pytest.importorskip("torch", reason="torch not installed")


def _traj(n: int, seed: int = 0) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).normal(size=(n, 3)), axis=0)


class TestWindows:
    def test_shapes_and_count(self):
        x, y = build_windows(_traj(100), window=16, horizon=1)
        # 99 deltas, minus the window, minus the horizon, plus one.
        assert x.shape == (99 - 16 - 1 + 1, 48)
        assert y.shape == (99 - 16 - 1 + 1, 3)

    def test_inputs_are_deltas_not_positions(self):
        """The model must see motion, or absolute workspace placement dominates."""
        traj = _traj(60, seed=1)
        x_a, y_a = build_windows(traj, 8, 1)
        x_b, y_b = build_windows(traj + np.array([100.0, -50.0, 7.0]), 8, 1)
        assert np.allclose(x_a, x_b, atol=1e-5)
        assert np.allclose(y_a, y_b, atol=1e-5)

    @pytest.mark.parametrize("n", [0, 1, 5, 16])
    def test_too_short_returns_empty_not_garbage(self, n):
        x, y = build_windows(np.zeros((n, 3)), window=16, horizon=1)
        assert x.shape[0] == 0 and y.shape[0] == 0
        assert x.shape[1] == 48

    def test_targets_follow_inputs_in_time(self):
        traj = _traj(40, seed=2)
        deltas = np.diff(traj, axis=0)
        x, y = build_windows(traj, 4, 1)
        assert np.allclose(x[0], deltas[:4].reshape(-1), atol=1e-5)
        assert np.allclose(y[0], deltas[4], atol=1e-5)


class TestSplit:
    def test_no_overlap_and_full_cover(self):
        train, val = split_episodes(100, 0.2)
        assert len(set(train) & set(val)) == 0
        assert len(train) + len(val) == 100
        assert len(val) == 20

    def test_is_deterministic_across_calls(self):
        """The held-out set must not move between arms or seeds.

        If it did, a difference between orderings would be confounded with which
        episodes happened to be held out, and the experiment would measure nothing.
        """
        assert np.array_equal(split_episodes(273, 0.2)[1], split_episodes(273, 0.2)[1])

    def test_always_yields_at_least_one_val_episode(self):
        assert len(split_episodes(3, 0.01)[1]) >= 1


class TestOrdering:
    @pytest.fixture
    def ranks(self):
        return np.arange(50, dtype=float)[::-1].copy()

    def test_anti_curriculum_is_the_exact_reverse(self, ranks):
        train = list(range(50))
        forward = order_episodes("curriculum", train, ranks, seed=0)
        backward = order_episodes("anti_curriculum", train, ranks, seed=0)
        assert forward == list(reversed(backward))

    def test_curriculum_follows_rank_order(self, ranks):
        order = order_episodes("curriculum", list(range(50)), ranks, seed=0)
        assert list(ranks[order]) == sorted(ranks[order])

    def test_shuffled_depends_on_seed(self, ranks):
        train = list(range(50))
        assert order_episodes("shuffled", train, ranks, 1) != order_episodes(
            "shuffled", train, ranks, 2
        )

    def test_every_ordering_is_a_permutation(self, ranks):
        train = list(range(50))
        for ordering in ORDERINGS:
            assert sorted(order_episodes(ordering, train, ranks, 0)) == train

    def test_ordered_arms_require_ranks(self):
        with pytest.raises(ValueError, match="curriculum ranks"):
            order_episodes("curriculum", [0, 1, 2], None, seed=0)

    def test_unknown_ordering_rejected(self, ranks):
        with pytest.raises(ValueError, match="unknown ordering"):
            order_episodes("nonsense", [0, 1], ranks, 0)


class TestSummary:
    def _arms(self, curriculum_losses, shuffled_losses):
        out = []
        for seed, value in enumerate(curriculum_losses):
            out.append(
                ArmResult("curriculum", seed, first_pass_auc=value, final_val_loss=value)
            )
        for seed, value in enumerate(shuffled_losses):
            out.append(
                ArmResult("shuffled", seed, first_pass_auc=value, final_val_loss=value)
            )
        return out

    def test_pairs_by_seed(self):
        arms = self._arms([0.1, 0.2, 0.3], [0.2, 0.3, 0.4])
        entry = summarise(arms)["paired_vs_shuffled"]["curriculum"]["first_pass_auc"]
        assert entry["n_seeds"] == 3
        assert entry["wins"] == 3 and entry["losses"] == 0
        assert entry["mean_delta"] < 0

    def test_reports_a_loss_as_a_loss(self):
        arms = self._arms([0.4, 0.5], [0.1, 0.2])
        entry = summarise(arms)["paired_vs_shuffled"]["curriculum"]["first_pass_auc"]
        assert entry["wins"] == 0 and entry["losses"] == 2

    def test_verdict_reports_null_result_honestly(self):
        """A tie must not be dressed up as a win -- the likely outcome here."""
        arms = self._arms([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
        text = verdict(summarise(arms))
        assert "No significant ordering effect" in text

    def test_verdict_reports_curriculum_being_worse(self):
        arms = self._arms([0.4] * 6, [0.1] * 6)
        summary = summarise(arms)
        entry = summary["paired_vs_shuffled"]["curriculum"]["first_pass_auc"]
        assert entry["mean_delta"] > 0
        if entry.get("p_value") is not None and entry["p_value"] < 0.05:
            assert "worse" in verdict(summary)

    def test_no_baseline_is_not_fatal(self):
        arms = [ArmResult("curriculum", 0, first_pass_auc=0.1)]
        assert summarise(arms)["paired_vs_shuffled"] == {}
        assert "No paired comparison" in verdict(summarise(arms))


class TestReport:
    def test_written_json_is_strict(self, tmp_path):
        """Regression: a bare NaN parses in Python and throws in the browser.

        An earlier artifact in this repo emitted `NaN` for an undefined p-value; Python
        read it back happily while `JSON.parse` rejected the whole file, so the section
        consuming it silently never rendered. Writing with `allow_nan=False` turns that
        into a loud failure at write time instead.
        """
        arms = [
            ArmResult("curriculum", 0, first_pass_auc=0.1, final_val_loss=0.1),
            ArmResult("shuffled", 0, first_pass_auc=0.2, final_val_loss=0.2),
        ]
        path = tmp_path / "curves.json"
        write_report(arms, TrainConfig(), str(path))
        text = path.read_text()
        assert "NaN" not in text and "Infinity" not in text
        json.loads(text)  # strict parse, as a browser would

    def test_report_contains_verdict_and_runs(self, tmp_path):
        arms = [ArmResult(o, 0, first_pass_auc=0.1) for o in ("curriculum", "shuffled")]
        payload = write_report(arms, TrainConfig(), str(tmp_path / "c.json"))
        assert payload["n_runs"] == 2
        assert isinstance(payload["verdict"], str) and payload["verdict"]


class TestTraining:
    """A real (tiny) training run, to catch shape and device bugs end to end."""

    def test_arm_trains_and_produces_a_curve(self):
        from src.bc_experiment import run_arm

        trajectories = [_traj(200, seed=i) for i in range(12)]
        ranks = list(range(12))
        config = TrainConfig(
            window=8, hidden=32, depth=2, epochs=1, eval_points=4, device="cpu"
        )
        arm = run_arm(trajectories, "curriculum", seed=0, curriculum_rank=ranks, config=config)
        assert arm.val_loss and all(np.isfinite(arm.val_loss))
        assert arm.first_pass_auc > 0
        assert arm.final_val_loss > 0
        assert len(arm.steps) == len(arm.val_loss)

    def test_same_seed_and_ordering_is_reproducible(self):
        from src.bc_experiment import run_arm

        trajectories = [_traj(200, seed=i) for i in range(10)]
        ranks = list(range(10))
        config = TrainConfig(
            window=8, hidden=32, depth=2, epochs=1, eval_points=3, device="cpu"
        )
        a = run_arm(trajectories, "curriculum", 0, ranks, config)
        b = run_arm(trajectories, "curriculum", 0, ranks, config)
        assert a.final_val_loss == pytest.approx(b.final_val_loss, rel=1e-6)
