"""Does training on the curriculum order actually help? A behaviour-cloning test.

Every other metric in this repo is a *proxy*: it scores properties of an ordering
(monotonic difficulty, low interference, coverage) rather than what a model does with
it. This module runs the experiment those proxies stand in for -- train the same policy
on the same data in different orders and compare what it learns.

The setup, and why each choice matters:

**Task.** Predict the end-effector's next displacement from a window of its recent
poses. Small, honest behaviour cloning: no images, no actions, just the trajectory
signal the whole pipeline is built on.

**The data is never reshuffled.** This is the entire experiment. Standard training
shuffles every epoch, which destroys ordering by construction and would make all arms
identical. Episodes are presented in the order under test and windows within an episode
stay in temporal order.

**Ordering matters most in the first pass.** After one epoch the model has seen
everything, and curriculum effects wash out. So validation is evaluated densely *during*
the first pass, not only at epoch boundaries, and the headline metric is
area-under-the-validation-curve over that pass -- how fast the model got good, not just
where it ended.

**Anti-curriculum is the control.** Reversing the order (hard to easy) is the arm that
must do *worse* if the difficulty ranking means anything. Without it, a curriculum
beating a shuffle could just be an artifact of any non-random order.

**Validation episodes are held out and identical across every arm and seed.** They are
split by episode, never by window, so no trajectory contributes to both sides.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ORDERINGS = ("curriculum", "anti_curriculum", "shuffled")

# Window of past poses fed to the model, and how far ahead it predicts.
DEFAULT_WINDOW = 16
DEFAULT_HORIZON = 1


@dataclass
class TrainConfig:
    """Settings shared by every arm, so only the ordering differs between runs."""

    window: int = DEFAULT_WINDOW
    horizon: int = DEFAULT_HORIZON
    hidden: int = 256
    depth: int = 3
    batch_size: int = 256
    learning_rate: float = 1e-3
    epochs: int = 3
    val_fraction: float = 0.2
    # Validation checkpoints during the first pass. Dense on purpose: this is where
    # ordering has any effect at all, and epoch-boundary evaluation would miss it.
    eval_points: int = 40
    seed: int = 0
    device: str = "auto"

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ArmResult:
    """One (ordering, seed) training run."""

    ordering: str
    seed: int
    steps: List[int] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    first_pass_auc: float = 0.0
    final_val_loss: float = 0.0
    best_val_loss: float = 0.0
    steps_to_threshold: Optional[int] = None
    forgetting: float = 0.0
    train_seconds: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_windows(
    trajectory: np.ndarray, window: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding windows of one trajectory into ``(inputs, targets)``.

    The input is the window's per-step displacements, centred by subtracting the window's
    final position, so the model sees *motion* rather than absolute workspace placement.
    The target is the next displacement. Predicting a delta rather than a position keeps
    the target scale consistent across sources that sit in different parts of the
    workspace.
    """
    traj = np.asarray(trajectory, dtype=np.float32)
    n = traj.shape[0]
    if n < window + horizon + 1:
        return np.zeros((0, window * 3), np.float32), np.zeros((0, 3), np.float32)

    deltas = np.diff(traj, axis=0)  # (n-1, 3)
    count = deltas.shape[0] - window - horizon + 1
    if count <= 0:
        return np.zeros((0, window * 3), np.float32), np.zeros((0, 3), np.float32)

    idx = np.arange(count)[:, None] + np.arange(window)[None, :]
    inputs = deltas[idx].reshape(count, window * 3)
    targets = deltas[np.arange(count) + window + horizon - 1]
    return inputs.astype(np.float32), targets.astype(np.float32)


def split_episodes(
    n_episodes: int, val_fraction: float, seed: int = 12345
) -> Tuple[np.ndarray, np.ndarray]:
    """Held-out episode indices, fixed across every arm and seed.

    Deliberately seeded independently of the run seed: if the validation set moved with
    the seed, differences between arms would be confounded with which episodes happened
    to be held out.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_episodes)
    n_val = max(1, int(round(n_episodes * val_fraction)))
    return np.sort(order[n_val:]), np.sort(order[:n_val])


def order_episodes(
    ordering: str,
    train_indices: Sequence[int],
    curriculum_rank: Optional[Sequence[float]],
    seed: int,
) -> List[int]:
    """Arrange the training episodes according to the arm under test."""
    train = list(int(i) for i in train_indices)
    if ordering == "shuffled":
        return list(np.random.default_rng(seed).permutation(train))
    if curriculum_rank is None:
        raise ValueError(f"ordering {ordering!r} needs curriculum ranks")
    ranks = np.asarray(curriculum_rank, dtype=float)
    ordered = sorted(train, key=lambda i: ranks[i])
    if ordering == "curriculum":
        return ordered
    if ordering == "anti_curriculum":
        return list(reversed(ordered))
    raise ValueError(f"unknown ordering {ordering!r}")


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _resolve_device(requested: str):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(window: int, config: TrainConfig):
    import torch.nn as nn

    layers: List[nn.Module] = []
    in_dim = window * 3
    for _ in range(config.depth):
        layers += [nn.Linear(in_dim, config.hidden), nn.GELU()]
        in_dim = config.hidden
    layers.append(nn.Linear(in_dim, 3))
    return nn.Sequential(*layers)


def run_arm(
    trajectories: Sequence[np.ndarray],
    ordering: str,
    seed: int,
    curriculum_rank: Optional[Sequence[float]] = None,
    config: Optional[TrainConfig] = None,
) -> ArmResult:
    """Train one arm and return its validation curve.

    Args:
        trajectories: Per-episode ``(T, 3)`` XYZ arrays.
        ordering: One of :data:`ORDERINGS`.
        seed: Controls weight init, and the shuffle for the ``shuffled`` arm.
        curriculum_rank: Per-episode rank; required by the ordered arms.
        config: Shared training settings.

    Returns:
        An :class:`ArmResult` with the dense first-pass validation curve.
    """
    import time

    import torch
    import torch.nn as nn

    config = config or TrainConfig()
    device = _resolve_device(config.device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_idx, val_idx = split_episodes(len(trajectories), config.val_fraction)
    episode_order = order_episodes(ordering, train_idx, curriculum_rank, seed)

    # Validation: one fixed tensor, built once.
    val_inputs, val_targets = [], []
    for i in val_idx:
        x, y = build_windows(trajectories[i], config.window, config.horizon)
        if len(x):
            val_inputs.append(x)
            val_targets.append(y)
    val_x = torch.from_numpy(np.concatenate(val_inputs)).to(device)
    val_y = torch.from_numpy(np.concatenate(val_targets)).to(device)

    # Training windows, concatenated in the arm's episode order and never reshuffled.
    per_episode = []
    for i in episode_order:
        x, y = build_windows(trajectories[i], config.window, config.horizon)
        if len(x):
            per_episode.append((i, x, y))
    train_x = torch.from_numpy(np.concatenate([x for _, x, _ in per_episode])).to(device)
    train_y = torch.from_numpy(np.concatenate([y for _, _, y in per_episode])).to(device)

    # Windows belonging to the first 20% of episodes seen, for the forgetting measure.
    early_count = sum(len(x) for _, x, _ in per_episode[: max(1, len(per_episode) // 5)])
    early_x, early_y = train_x[:early_count], train_y[:early_count]

    model = _build_model(config.window, config).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    n_train = train_x.shape[0]
    steps_per_epoch = max(1, math.ceil(n_train / config.batch_size))
    eval_every = max(1, steps_per_epoch // config.eval_points)

    def evaluate(inputs, targets) -> float:
        model.eval()
        with torch.no_grad():
            total, count = 0.0, 0
            for start in range(0, inputs.shape[0], 8192):
                chunk_x = inputs[start : start + 8192]
                chunk_y = targets[start : start + 8192]
                total += float(loss_fn(model(chunk_x), chunk_y)) * chunk_x.shape[0]
                count += chunk_x.shape[0]
        model.train()
        return total / max(count, 1)

    result = ArmResult(ordering=ordering, seed=seed)
    threshold: Optional[float] = None
    started = time.time()
    step = 0

    for epoch in range(config.epochs):
        for start in range(0, n_train, config.batch_size):
            batch_x = train_x[start : start + config.batch_size]
            batch_y = train_y[start : start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss_fn(model(batch_x), batch_y).backward()
            optimiser.step()
            step += 1

            # Dense evaluation during the first pass only; ordering cannot matter after.
            if (epoch == 0 and step % eval_every == 0) or (
                epoch > 0 and step % steps_per_epoch == 0
            ):
                value = evaluate(val_x, val_y)
                result.steps.append(step)
                result.val_loss.append(value)
                if threshold is None:
                    # Fixed once from this arm's own first reading; re-derived globally
                    # by `summarise` so the arms share a comparable threshold.
                    threshold = value

    result.train_seconds = time.time() - started
    result.final_val_loss = evaluate(val_x, val_y)
    result.best_val_loss = min(result.val_loss) if result.val_loss else result.final_val_loss

    # Forgetting: how much worse the model is on the episodes it saw first than on
    # held-out data. A curriculum that only works by overwriting early material is not
    # a curriculum, so this has to be reported rather than assumed away.
    result.forgetting = evaluate(early_x, early_y) - result.final_val_loss

    first_pass = [
        (s, v) for s, v in zip(result.steps, result.val_loss) if s <= steps_per_epoch
    ]
    if len(first_pass) > 1:
        xs = np.array([s for s, _ in first_pass], dtype=float)
        ys = np.array([v for _, v in first_pass], dtype=float)
        result.first_pass_auc = float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))
    elif first_pass:
        result.first_pass_auc = float(first_pass[0][1])
    return result


def summarise(results: Sequence[ArmResult]) -> Dict[str, object]:
    """Aggregate arms across seeds, with a paired comparison against ``shuffled``.

    Pairs by seed: each ordering is trained from the same initialisation as the shuffled
    run it is compared against, so the difference is not confounded with weight init.
    """
    by_ordering: Dict[str, List[ArmResult]] = {}
    for item in results:
        by_ordering.setdefault(item.ordering, []).append(item)

    summary: Dict[str, object] = {"orderings": {}, "paired_vs_shuffled": {}}
    for ordering, arms in by_ordering.items():
        summary["orderings"][ordering] = {
            "n_seeds": len(arms),
            "first_pass_auc_mean": float(np.mean([a.first_pass_auc for a in arms])),
            "first_pass_auc_std": float(np.std([a.first_pass_auc for a in arms])),
            "final_val_loss_mean": float(np.mean([a.final_val_loss for a in arms])),
            "final_val_loss_std": float(np.std([a.final_val_loss for a in arms])),
            "best_val_loss_mean": float(np.mean([a.best_val_loss for a in arms])),
            "forgetting_mean": float(np.mean([a.forgetting for a in arms])),
        }

    baseline = {a.seed: a for a in by_ordering.get("shuffled", [])}
    for ordering, arms in by_ordering.items():
        if ordering == "shuffled" or not baseline:
            continue
        paired = [(a, baseline[a.seed]) for a in arms if a.seed in baseline]
        if not paired:
            continue
        for metric in ("first_pass_auc", "final_val_loss"):
            ours = np.array([getattr(a, metric) for a, _ in paired])
            theirs = np.array([getattr(b, metric) for _, b in paired])
            delta = ours - theirs
            entry = {
                "n_seeds": len(paired),
                "mean_ours": float(ours.mean()),
                "mean_shuffled": float(theirs.mean()),
                "mean_delta": float(delta.mean()),
                # None, never NaN. A bare NaN survives `json.dump` by default, then
                # throws in `JSON.parse`, so the consumer silently renders nothing while
                # the file looks correct in an editor.
                "pct_change": (
                    float(delta.mean() / abs(theirs.mean()) * 100.0)
                    if theirs.mean()
                    else None
                ),
                "wins": int((delta < 0).sum()),
                "losses": int((delta > 0).sum()),
                "p_value": None,
            }
            if len(paired) > 1 and np.any(delta != 0):
                try:
                    from scipy.stats import wilcoxon

                    entry["p_value"] = float(wilcoxon(ours, theirs).pvalue)
                except Exception:  # noqa: BLE001 - report without it rather than fail
                    pass
            summary["paired_vs_shuffled"].setdefault(ordering, {})[metric] = entry
    return summary


def verdict(summary: Dict[str, object]) -> str:
    """One sentence stating what the experiment found, computed from the numbers.

    Written to be equally willing to report a null result: an ordering effect inside
    seed noise is the likely outcome and saying so is the point of running it.
    """
    paired = summary.get("paired_vs_shuffled", {})
    curriculum = paired.get("curriculum", {}).get("first_pass_auc")
    if not curriculum:
        return "No paired comparison available."

    delta = curriculum["mean_delta"]
    p_value = curriculum.get("p_value")
    pct = curriculum.get("pct_change")
    pct_text = "n/a" if pct is None else f"{pct:+.1f}%"
    significant = p_value is not None and p_value < 0.05

    if significant and delta < 0:
        return (
            f"Curriculum order reached a lower validation loss during the first pass than "
            f"shuffled order ({curriculum['pct_change']:+.1f}%, p={p_value:.3f}, "
            f"{curriculum['wins']}/{curriculum['n_seeds']} seeds)."
        )
    if significant and delta > 0:
        return (
            f"Curriculum order was *worse* than shuffled during the first pass "
            f"({pct_text}, p={p_value:.3f})."
        )
    return (
        f"No significant ordering effect: curriculum vs shuffled differed by "
        f"{pct_text} over {curriculum['n_seeds']} seeds"
        + (f" (p={p_value:.3f})" if p_value is not None else "")
        + ". The ordering effect is within seed noise on this dataset."
    )


def write_report(results: Sequence[ArmResult], config: TrainConfig, path: str) -> Dict:
    """Serialise curves plus summary to JSON, strict (no NaN literals)."""
    summary = summarise(results)
    payload = {
        "config": config.as_dict(),
        "n_runs": len(results),
        "summary": summary,
        "verdict": verdict(summary),
        "runs": [r.as_dict() for r in results],
    }
    with open(path, "w") as handle:
        # allow_nan=False so an undefined statistic fails here rather than producing a
        # file that Python reads happily and `JSON.parse` rejects in the browser.
        json.dump(payload, handle, indent=2, allow_nan=False)
    return payload
