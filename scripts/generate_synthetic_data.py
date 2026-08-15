#!/usr/bin/env python
"""Generate synthetic episodes in the *real* EgoVerse `processed_v3` schema.

Useful for offline development, CI, and reproducing edge cases without R2 credentials.
The test suite imports :func:`write_episode` from this module, so there is a single
implementation of "what an EgoVerse episode looks like on disk".

Every schema quirk found in the real dataset is reproduced faithfully, because code
that only ever sees clean fixtures will break on the real thing:

- Zarr **v3** groups with flat, dot-separated keys (``left.obs_ee_pose``).
- Pose arrays of shape ``(T, 7)``: XYZ then a unit quaternion.
- Arrays **chunk-padded** beyond ``total_frames`` with a zero tail.
- Optional mid-episode **missing-frame sentinels** (``[0,0,0,1,0,0,0]``).
- Optional **millimetre** units instead of metres.
- Group ``attrs`` carrying ``total_frames``, ``fps``, ``task_name``,
  ``task_description``, ``embodiment`` and ``features``.

Examples:
    python scripts/generate_synthetic_data.py --n-episodes 40 --out data_synth
    python scripts/generate_synthetic_data.py --out data_synth --inject-edge-cases
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import zarr

ARCHETYPES = ("reach", "pick_place", "wipe", "stir", "zigzag_search", "pour")

# Tabletop workspace bounds in metres.
X_RANGE = (0.30, 0.70)
Y_RANGE = (-0.30, 0.30)
Z_RANGE = (0.05, 0.50)

# Real stores round the array length up to a chunk boundary and zero-fill the tail.
CHUNK = 100


def _minimum_jerk(steps: int) -> np.ndarray:
    """Minimum-jerk time scaling on [0, 1] -- the canonical smooth reach profile."""
    t = np.linspace(0.0, 1.0, steps)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _sample_point(rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE), rng.uniform(*Z_RANGE)]
    )


def make_xyz(archetype: str, rng: np.random.Generator, steps: Optional[int] = None) -> np.ndarray:
    """Synthesise one ``(T, 3)`` XYZ trajectory for a motion archetype."""
    steps = int(steps or rng.integers(120, 600))
    s = _minimum_jerk(steps)
    progress = np.linspace(0.0, 1.0, steps)
    start = _sample_point(rng)

    if archetype == "reach":
        goal = _sample_point(rng)
        xyz = start[None, :] + s[:, None] * (goal - start)[None, :]

    elif archetype == "pick_place":
        goal = _sample_point(rng)
        xyz = start[None, :] + s[:, None] * (goal - start)[None, :]
        xyz[:, 2] += rng.uniform(0.12, 0.28) * np.sin(np.pi * s)

    elif archetype == "wipe":
        cycles, amp = rng.uniform(2.0, 5.0), rng.uniform(0.08, 0.22)
        xyz = np.column_stack(
            [
                start[0] + rng.uniform(0.02, 0.10) * progress,
                start[1] + amp * np.sin(2 * np.pi * cycles * progress),
                np.full(steps, start[2]) + rng.normal(0, 0.002, steps),
            ]
        )

    elif archetype == "stir":
        cycles, radius = rng.uniform(1.5, 4.0), rng.uniform(0.04, 0.12)
        theta = 2 * np.pi * cycles * progress
        xyz = np.column_stack(
            [
                start[0] + radius * np.cos(theta),
                start[1] + radius * np.sin(theta),
                start[2] + 0.01 * np.sin(4 * np.pi * progress),
            ]
        )

    elif archetype == "zigzag_search":
        n_legs = int(rng.integers(5, 11))
        waypoints = np.stack([_sample_point(rng) for _ in range(n_legs)])
        leg_t = np.linspace(0.0, n_legs - 1, steps)
        idx = np.clip(leg_t.astype(int), 0, n_legs - 2)
        frac = leg_t - idx
        xyz = waypoints[idx] + frac[:, None] * (waypoints[idx + 1] - waypoints[idx])

    elif archetype == "pour":
        goal = start + rng.normal(0, 0.02, 3)
        approach = max(2, int(steps * 0.35))
        xyz = np.repeat(goal[None, :], steps, axis=0)
        xyz[:approach] = start[None, :] + _minimum_jerk(approach)[:, None] * (goal - start)[None, :]
        xyz[approach:, 2] += 0.015 * np.sin(np.linspace(0, np.pi, steps - approach))

    else:
        raise ValueError(f"unknown archetype {archetype!r}")

    return xyz + rng.normal(0.0, 0.0015, xyz.shape)


def _unit_quats(steps: int, spin: float) -> np.ndarray:
    """Smoothly varying unit quaternions in ``[qw, qx, qy, qz]`` order."""
    half = (spin * np.linspace(0, 1, steps) * np.pi) / 2.0
    quat = np.column_stack([np.cos(half), np.zeros(steps), np.zeros(steps), np.sin(half)])
    return quat / np.linalg.norm(quat, axis=1, keepdims=True)


def build_pose(
    xyz: np.ndarray,
    spin: float = 0.5,
    missing_ratio: float = 0.0,
    unit_scale: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Assemble a ``(T, 7)`` pose array, optionally injecting missing-frame sentinels."""
    steps = xyz.shape[0]
    pose = np.column_stack([xyz * unit_scale, _unit_quats(steps, spin)])
    if missing_ratio > 0:
        rng = rng or np.random.default_rng(0)
        n_missing = int(steps * missing_ratio)
        idx = rng.choice(steps, size=n_missing, replace=False)
        pose[idx] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    return pose


def write_episode(
    path: str | Path,
    poses: Dict[str, np.ndarray],
    task_name: str = "synthetic_task",
    embodiment: str = "synthetic_bimanual",
    fps: int = 30,
    chunk_pad: bool = True,
    total_frames: Optional[int] = None,
    extra_attrs: Optional[Dict[str, object]] = None,
) -> Path:
    """Write one episode store in the real EgoVerse Zarr v3 schema.

    Args:
        path: Destination ``*.zarr`` directory.
        poses: Key -> ``(T, 7)`` array, e.g. ``{"right.obs_ee_pose": pose}``.
        task_name: Ground-truth task label, stored in ``attrs``.
        embodiment: Embodiment tag, stored in ``attrs``.
        fps: Frame rate, stored in ``attrs``.
        chunk_pad: Reproduce the real chunk-padding behaviour by extending each array
            to a chunk boundary with a zero tail, while ``total_frames`` records the
            true length. Loaders that ignore ``total_frames`` read the fabricated tail.
        total_frames: Override the recorded frame count.
        extra_attrs: Additional group attributes.

    Returns:
        The path written.
    """
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)

    lengths = {k: v.shape[0] for k, v in poses.items()}
    true_frames = int(total_frames or (min(lengths.values()) if lengths else 0))

    root = zarr.open_group(str(path), mode="w")
    features: Dict[str, Dict[str, object]] = {}
    for key, arr in poses.items():
        data = np.asarray(arr, dtype=np.float64)
        if chunk_pad:
            padded_len = int(np.ceil(data.shape[0] / CHUNK) * CHUNK)
            if padded_len > data.shape[0]:
                tail = np.zeros((padded_len - data.shape[0], data.shape[1]))
                data = np.concatenate([data, tail], axis=0)
        node = root.create_array(
            key, shape=data.shape, chunks=(min(CHUNK, data.shape[0]), data.shape[1]),
            dtype="float64",
        )
        node[:] = data
        features[key] = {"dtype": "float64", "shape": [data.shape[1]], "names": ["dim_0"]}

    root.attrs["total_frames"] = true_frames
    root.attrs["fps"] = fps
    root.attrs["task_name"] = task_name
    root.attrs["task_description"] = f"synthetic {task_name}"
    root.attrs["embodiment"] = embodiment
    root.attrs["features"] = features
    for key, value in (extra_attrs or {}).items():
        root.attrs[key] = value
    return path


def generate_dataset(
    out_dir: str | Path,
    n_episodes: int = 40,
    n_duplicates: int = 4,
    seed: int = 42,
    inject_edge_cases: bool = False,
) -> List[Path]:
    """Write a synthetic dataset, optionally including pathological episodes."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Deliberately imbalanced, as real teleop datasets over-represent easy skills.
    weights = np.array([0.28, 0.22, 0.16, 0.14, 0.12, 0.08])
    weights /= weights.sum()

    written: List[Path] = []
    generated: List[Tuple[str, np.ndarray]] = []
    for i in range(n_episodes):
        archetype = str(rng.choice(ARCHETYPES, p=weights))
        xyz = make_xyz(archetype, rng)
        generated.append((archetype, xyz))
        # Mix single-arm and bimanual sources, as the real dataset does.
        bimanual = rng.random() < 0.6
        poses = {"right.obs_ee_pose": build_pose(xyz, spin=rng.uniform(-1, 1), rng=rng)}
        if bimanual:
            other = make_xyz(str(rng.choice(ARCHETYPES, p=weights)), rng, steps=xyz.shape[0])
            poses["left.obs_ee_pose"] = build_pose(other, spin=rng.uniform(-1, 1), rng=rng)
        written.append(
            write_episode(
                out / f"synth__{archetype}__{i:04d}.zarr",
                poses,
                task_name=archetype,
                embodiment="synthetic_bimanual" if bimanual else "synthetic_right_arm",
                fps=int(rng.choice([30, 60])),
            )
        )

    # Near-duplicates, to exercise redundancy detection.
    for k in range(n_duplicates):
        archetype, xyz = generated[int(rng.integers(0, len(generated)))]
        jittered = xyz + rng.normal(0.0, 0.0005, xyz.shape)
        written.append(
            write_episode(
                out / f"synth__dup__{k:04d}.zarr",
                {"right.obs_ee_pose": build_pose(jittered, rng=rng)},
                task_name=archetype,
                embodiment="synthetic_right_arm",
            )
        )

    if inject_edge_cases:
        # A dead pose stream, as every real `eva` episode exhibits.
        dead = np.tile(np.array([0.0, 0.0, 0.0, -0.5, 0.5, -0.5, 0.5]), (400, 1))
        written.append(
            write_episode(out / "synth__edge_dead.zarr", {"right.obs_ee_pose": dead},
                          task_name="dead_stream", embodiment="synthetic_right_arm")
        )
        # Millimetre units, as the human motion-capture sources use.
        xyz = make_xyz("reach", rng, steps=300)
        written.append(
            write_episode(out / "synth__edge_millimetres.zarr",
                          {"right.obs_ee_pose": build_pose(xyz, unit_scale=1000.0, rng=rng)},
                          task_name="mm_units", embodiment="synthetic_right_arm")
        )
        # Heavy mid-episode frame dropout.
        xyz = make_xyz("wipe", rng, steps=400)
        written.append(
            write_episode(out / "synth__edge_dropout.zarr",
                          {"right.obs_ee_pose": build_pose(xyz, missing_ratio=0.3, rng=rng)},
                          task_name="dropout", embodiment="synthetic_right_arm")
        )
        # Mostly-missing episode, which must be rejected outright.
        xyz = make_xyz("reach", rng, steps=300)
        written.append(
            write_episode(out / "synth__edge_mostly_missing.zarr",
                          {"right.obs_ee_pose": build_pose(xyz, missing_ratio=0.8, rng=rng)},
                          task_name="mostly_missing", embodiment="synthetic_right_arm")
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data_synth")
    parser.add_argument("--n-episodes", type=int, default=40)
    parser.add_argument("--n-duplicates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--inject-edge-cases",
        action="store_true",
        help="also write dead-stream, millimetre, dropout and mostly-missing episodes",
    )
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)

    written = generate_dataset(
        out, args.n_episodes, args.n_duplicates, args.seed, args.inject_edge_cases
    )
    print(f"Wrote {len(written)} synthetic episodes to '{out}' (real processed_v3 schema).")
    print("Now run:  python run_pipeline.py --data-dir " + str(out))


if __name__ == "__main__":
    main()
