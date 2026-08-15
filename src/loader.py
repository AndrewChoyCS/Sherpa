"""Zarr ingestion for real EgoVerse `processed_v3` episode stores.

This module is written against the *actual* dataset schema, verified by inspection
rather than assumed. The relevant facts, each of which requires handling:

**Zarr v3, flat dot-separated keys.** An episode is a Zarr v3 group whose arrays are
named ``left.obs_ee_pose``, ``right.obs_ee_pose``, ``images.front_1`` and so on --
a flat namespace, not nested groups. Group ``attrs`` carry ``total_frames``, ``fps``,
``task_name``, ``task_description``, ``embodiment`` and a ``features`` dict.

**Pose layout is ``[x, y, z, qw, qx, qy, qz]``.** Shape ``(T, 7)``; the trailing four
columns are a unit quaternion. Only XYZ is used here.

**Arrays are chunk-padded past the end of the episode.** An episode with
``total_frames=2808`` is stored in a length-2900 array, the tail being zeros. Every
episode inspected exhibits this. Reading the array without truncating to
``total_frames`` appends a fabricated jump to the origin, which would dominate any
distance metric. Truncation is mandatory, not defensive.

**All-zero XYZ rows are missing-data sentinels.** Frames where the pose is exactly
``[0,0,0,1,0,0,0]`` (identity) mean "not tracked this frame", and appear mid-episode
in 13-25% of frames for some human-teleop sources. Treated as real coordinates they
create enormous spurious excursions to the world origin. They are dropped, and the
ratio is retained as a data-quality signal.

**Units are inconsistent across sources.** Robot sources (``yam``) record metres
(|xyz| ~ 0.2-0.3); human motion-capture sources (``scale``) record millimetres
(|xyz| ~ 200). Left unconverted, cross-source distances measure unit mismatch rather
than behaviour. ``unit_scale="auto"`` rescales millimetre-scale episodes to metres.

**Some episodes are entirely degenerate.** Every ``eva`` episode inspected has a
constant ``right.obs_ee_pose`` of ``[0,0,0,-0.5,0.5,-0.5,0.5]`` for all frames -- the
pose stream was never populated. And in some bimanual episodes one arm is static to
within 0.4 mm. Degenerate arms are rejected; degenerate episodes are skipped with a
recorded reason rather than silently producing a zero-length trajectory.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import zarr

# Per-arm end-effector pose keys in the real schema.
ARM_KEYS: Dict[str, str] = {
    "left": "left.obs_ee_pose",
    "right": "right.obs_ee_pose",
}
# Fallback aliases, tried when the canonical per-arm keys are absent.
FALLBACK_POSE_KEYS: Tuple[str, ...] = (
    "obs_ee_pose",
    "ee_pose",
    "obs_head_pose",
    "left.cmd_ee_pose",
    "right.cmd_ee_pose",
)

ARM_MODES = ("auto", "left", "right", "both")

DEFAULT_MIN_LENGTH = 30
# Median |xyz| above this implies the episode is in millimetres, not metres.
MM_DETECTION_THRESHOLD = 10.0
# An arm whose positional standard deviation is below this (metres) is not moving.
STATIC_ARM_STD_M = 1e-3
# Reject an episode if more than this fraction of frames are missing-data sentinels.
MAX_MISSING_RATIO = 0.5


@dataclass
class TrajectoryDataset:
    """Loaded end-effector trajectories plus provenance and data-quality metadata.

    Attributes:
        trajectories: Cleaned ``(T_i, C)`` float64 arrays in metres. ``C`` is 3 for
            single-arm modes and 6 for ``arm="both"`` (left XYZ then right XYZ).
        episode_ids: Identifier per trajectory, derived from the store directory name.
        metadata: Per-episode dict with ``task_name``, ``embodiment``, ``fps``,
            ``source``, ``arm_used``, ``missing_frame_ratio``, ``unit_scale`` and
            ``n_raw_frames``.
        skipped: ``(episode_id, reason)`` for every rejected episode.
    """

    trajectories: List[np.ndarray] = field(default_factory=list)
    episode_ids: List[str] = field(default_factory=list)
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.trajectories)

    @property
    def lengths(self) -> np.ndarray:
        return np.array([t.shape[0] for t in self.trajectories], dtype=int)

    @property
    def fps(self) -> np.ndarray:
        """Frames per second per episode, defaulting to 30 where unrecorded."""
        return np.array([float(m.get("fps") or 30.0) for m in self.metadata])

    def field_values(self, name: str, default: str = "unknown") -> List[str]:
        """Metadata column as a list of strings, for grouping and colouring."""
        return [str(m.get(name, default) or default) for m in self.metadata]

    @property
    def task_labels(self) -> List[str]:
        """Ground-truth task names, used to score clustering via Adjusted Rand Index."""
        return self.field_values("task_name")

    def as_tuple(self) -> Tuple[List[np.ndarray], List[str]]:
        return self.trajectories, self.episode_ids

    def summary(self) -> str:
        if not len(self):
            return "TrajectoryDataset(empty)"
        lens = self.lengths
        n_sources = len(set(self.field_values("source")))
        n_tasks = len(set(self.task_labels))
        return (
            f"TrajectoryDataset(n={len(self)}, dims={self.trajectories[0].shape[1]}, "
            f"sources={n_sources}, tasks={n_tasks}, "
            f"frames min/median/max={lens.min()}/{int(np.median(lens))}/{lens.max()}, "
            f"skipped={len(self.skipped)})"
        )


# --------------------------------------------------------------------------- #
# cleaning primitives
# --------------------------------------------------------------------------- #
def _missing_mask(pose: np.ndarray) -> np.ndarray:
    """True where a frame is a missing-data sentinel (exactly zero XYZ)."""
    return np.all(pose[:, :3] == 0.0, axis=1)


def detect_unit_scale(xyz: np.ndarray) -> float:
    """Return the factor converting ``xyz`` to metres.

    Human motion-capture sources in EgoVerse record millimetres while robot sources
    record metres. A median position magnitude above ``MM_DETECTION_THRESHOLD``
    metres is not physically plausible for a tabletop workspace, so it implies
    millimetres.
    """
    if xyz.size == 0:
        return 1.0
    magnitude = float(np.median(np.linalg.norm(xyz, axis=1)))
    return 1e-3 if magnitude > MM_DETECTION_THRESHOLD else 1.0


def clean_pose_array(
    raw: np.ndarray,
    total_frames: Optional[int],
    min_length: int,
    unit_scale: str = "auto",
) -> Tuple[np.ndarray, float, float]:
    """Truncate, de-sentinel and unit-normalise one ``(T, >=3)`` pose array.

    Returns:
        ``(xyz_metres, missing_ratio, applied_scale)``.

    Raises:
        ValueError: if the result is too short, mostly missing, or non-moving.
    """
    arr = np.asarray(raw)
    if arr.ndim == 1:
        raise ValueError(f"pose array is 1-D with shape {arr.shape}")
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] < 3:
        raise ValueError(f"pose array has {arr.shape[1]} channels, need >= 3")

    # Chunk padding: the stored array runs past the real end of the episode.
    if total_frames is not None and 0 < total_frames <= arr.shape[0]:
        arr = arr[:total_frames]
    arr = np.asarray(arr, dtype=np.float64)

    n_raw = arr.shape[0]
    if n_raw == 0:
        raise ValueError("episode has zero frames")

    missing = _missing_mask(arr) | ~np.isfinite(arr[:, :3]).all(axis=1)
    missing_ratio = float(missing.mean())
    if missing_ratio > MAX_MISSING_RATIO:
        raise ValueError(f"{missing_ratio:.0%} of frames are missing-data sentinels")

    xyz = arr[~missing, :3]
    if xyz.shape[0] < min_length:
        raise ValueError(f"only {xyz.shape[0]} valid frames (min_length={min_length})")

    scale = detect_unit_scale(xyz) if unit_scale == "auto" else float(unit_scale)
    xyz = xyz * scale

    if float(np.max(xyz.std(axis=0))) < STATIC_ARM_STD_M:
        raise ValueError(
            f"end-effector is static (max per-axis std "
            f"{float(np.max(xyz.std(axis=0))):.2e} m); pose stream likely unpopulated"
        )
    return np.ascontiguousarray(xyz), missing_ratio, scale


# --------------------------------------------------------------------------- #
# store inspection
# --------------------------------------------------------------------------- #
def _array_keys(group: Any) -> List[str]:
    try:
        return sorted(name for name, _ in group.arrays())
    except Exception:  # noqa: BLE001
        return sorted(k for k in group.keys())


def _read_arm(
    group: Any,
    key: str,
    total_frames: Optional[int],
    min_length: int,
    unit_scale: str,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float], Optional[str]]:
    """Read and clean one arm. Returns ``(xyz, missing_ratio, scale, error)``."""
    try:
        raw = np.asarray(group[key][:])
    except Exception as exc:  # noqa: BLE001
        return None, None, None, f"unreadable {key}: {exc}"
    try:
        xyz, missing_ratio, scale = clean_pose_array(raw, total_frames, min_length, unit_scale)
    except ValueError as exc:
        return None, None, None, f"{key}: {exc}"
    return xyz, missing_ratio, scale, None


def _episode_id(entry: str) -> str:
    """Strip repeated ``.zarr`` suffixes from a store directory name."""
    name = entry
    while name.endswith(".zarr"):
        name = name[: -len(".zarr")]
    return name


def _source_of(episode_id: str) -> str:
    """Dataset source, i.e. the part before the first path separator token."""
    return episode_id.split("__", 1)[0] if "__" in episode_id else "unknown"


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def _store_fingerprint(
    stores: Sequence[Tuple[str, str]], min_length: int, arm: str, unit_scale: str
) -> str:
    """Content hash over the store inventory plus the result-affecting loader options.

    Keyed on file *sizes* rather than mtimes: copying a dataset between machines or
    syncing it out of a network volume rewrites mtimes without changing the data, and a
    cache that missed on every sync would be pointless.
    """
    h = hashlib.sha256()
    h.update(f"v1|{min_length}|{arm}|{unit_scale}".encode())
    for store_path, ep_id in stores:
        h.update(ep_id.encode())
        for root, _dirs, files in os.walk(store_path):
            for name in sorted(files):
                try:
                    size = os.path.getsize(os.path.join(root, name))
                except OSError:
                    size = -1
                h.update(f"{name}:{size}".encode())
    return h.hexdigest()[:16]


def load_zarr_trajectories(
    data_dir: str,
    min_length: int = DEFAULT_MIN_LENGTH,
    arm: str = "auto",
    unit_scale: str = "auto",
    verbose: bool = True,
    cache_dir: Optional[str] = None,
) -> TrajectoryDataset:
    """Load end-effector XYZ trajectories from a directory of EgoVerse zarr stores.

    Args:
        data_dir: Directory of ``*.zarr`` episode stores (see
            ``scripts/fetch_egoverse_data.py``), or a single store.
        min_length: Reject episodes with fewer valid frames than this.
        arm: ``"auto"`` picks the more *active* arm per episode, yielding a 3-D
            trajectory for every episode -- necessary because DTW can only compare
            series with matching channel counts, and some sources are single-arm.
            ``"left"``/``"right"`` force one arm. ``"both"`` concatenates into a 6-D
            trajectory and skips episodes lacking two usable arms.
        unit_scale: ``"auto"`` detects millimetre sources and converts to metres;
            or pass a numeric factor as a string/float.
        verbose: Print a load summary and skip reasons.
        cache_dir: If given, memoise the parsed dataset there, keyed by a content hash of
            the store inventory and these options. Each episode is a Zarr *group* of many
            small files, so ingestion is dominated by per-file latency rather than by
            parsing -- measured at 337 s for 320 stores on a network volume versus 21 s
            for the entire DTW matrix over the same episodes. Collapsing that into one
            file read is therefore a much larger win than any compute optimisation.
            ``None`` disables it, which is the default so existing callers are unchanged.

    Returns:
        A :class:`TrajectoryDataset`. Per-episode failures are recorded in
        ``.skipped`` rather than raised, so one bad store cannot abort a run.

    Note:
        The cache is a pickle, so ``cache_dir`` must be a location you control -- the
        same trust assumption the DTW ``.npy`` cache already makes.
    """
    if arm not in ARM_MODES:
        raise ValueError(f"arm must be one of {ARM_MODES}, got {arm!r}")

    out = TrajectoryDataset()
    if not os.path.exists(data_dir):
        if verbose:
            print(f"[loader] '{data_dir}' not found; returning empty dataset.")
        return out

    if os.path.isdir(data_dir):
        entries = sorted(e for e in os.listdir(data_dir) if e.endswith(".zarr"))
        stores = [(os.path.join(data_dir, e), _episode_id(e)) for e in entries]
        if not stores:
            stores = [(data_dir, _episode_id(os.path.basename(data_dir.rstrip(os.sep))))]
    else:
        stores = [(data_dir, _episode_id(os.path.basename(data_dir)))]

    cache_path: Optional[str] = None
    if cache_dir:
        fingerprint = _store_fingerprint(stores, min_length, arm, unit_scale)
        cache_path = os.path.join(cache_dir, f"dataset_{fingerprint}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as handle:
                    cached = pickle.load(handle)
                if isinstance(cached, TrajectoryDataset):
                    if verbose:
                        print(f"[loader] cache hit {os.path.basename(cache_path)} "
                              f"({len(cached)} episodes)")
                    return cached
            except Exception as exc:  # noqa: BLE001 - a corrupt cache must not be fatal
                warnings.warn(
                    f"ignoring unreadable dataset cache {cache_path}: {exc}", stacklevel=2
                )

    for store_path, ep_id in stores:
        try:
            group = zarr.open_group(store_path, mode="r")
        except Exception as exc:  # noqa: BLE001
            out.skipped.append((ep_id, f"could not open store: {exc}"))
            continue

        attrs = dict(group.attrs)
        total_frames = attrs.get("total_frames")
        available = set(_array_keys(group))

        # Read every usable arm once, then decide which to keep.
        arms: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []
        for side, key in ARM_KEYS.items():
            if key not in available:
                continue
            xyz, missing_ratio, scale, error = _read_arm(
                group, key, total_frames, min_length, unit_scale
            )
            if error:
                errors.append(error)
                continue
            arms[side] = {
                "xyz": xyz,
                "missing_ratio": missing_ratio,
                "scale": scale,
                # Path length is the activity measure: it distinguishes a genuinely
                # moving arm from one that merely jitters in place.
                "activity": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
            }

        if not arms:
            for key in FALLBACK_POSE_KEYS:
                if key not in available:
                    continue
                xyz, missing_ratio, scale, error = _read_arm(
                    group, key, total_frames, min_length, unit_scale
                )
                if error:
                    errors.append(error)
                    continue
                arms[key] = {
                    "xyz": xyz,
                    "missing_ratio": missing_ratio,
                    "scale": scale,
                    "activity": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
                }
                break

        if not arms:
            reason = "; ".join(errors) if errors else f"no pose arrays in {sorted(available)}"
            out.skipped.append((ep_id, reason))
            continue

        # Assemble the trajectory according to the requested arm mode.
        if arm == "both":
            if "left" not in arms or "right" not in arms:
                out.skipped.append(
                    (ep_id, f"arm='both' needs two usable arms, have {sorted(arms)}")
                )
                continue
            left, right = arms["left"]["xyz"], arms["right"]["xyz"]
            # Sentinel dropping can desynchronise the two arms; trim to the shorter.
            n = min(left.shape[0], right.shape[0])
            traj = np.concatenate([left[:n], right[:n]], axis=1)
            arm_used = "both"
            missing_ratio = max(arms["left"]["missing_ratio"], arms["right"]["missing_ratio"])
            scale = arms["left"]["scale"]
        elif arm in ("left", "right"):
            if arm not in arms:
                out.skipped.append((ep_id, f"arm='{arm}' unavailable, have {sorted(arms)}"))
                continue
            chosen = arms[arm]
            traj, arm_used = chosen["xyz"], arm
            missing_ratio, scale = chosen["missing_ratio"], chosen["scale"]
        else:  # auto -> most active arm
            arm_used = max(arms, key=lambda side: arms[side]["activity"])
            chosen = arms[arm_used]
            traj = chosen["xyz"]
            missing_ratio, scale = chosen["missing_ratio"], chosen["scale"]

        out.trajectories.append(traj)
        out.episode_ids.append(ep_id)
        out.metadata.append(
            {
                "source": _source_of(ep_id),
                "task_name": attrs.get("task_name", "unknown"),
                "task_description": attrs.get("task_description", ""),
                "embodiment": attrs.get("embodiment", "unknown"),
                "fps": attrs.get("fps", 30),
                "arm_used": arm_used,
                "arms_available": ",".join(sorted(arms)),
                "missing_frame_ratio": round(float(missing_ratio or 0.0), 4),
                "unit_scale": scale,
                "n_raw_frames": int(total_frames or traj.shape[0]),
                "n_valid_frames": int(traj.shape[0]),
            }
        )

    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            # Write-then-rename, so an interrupted write cannot leave a torn cache that
            # the next run would have to detect and discard.
            temporary = f"{cache_path}.tmp{os.getpid()}"
            with open(temporary, "wb") as handle:
                pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, cache_path)
        except Exception as exc:  # noqa: BLE001 - caching is an optimisation, not a duty
            warnings.warn(f"could not write dataset cache {cache_path}: {exc}", stacklevel=2)

    if verbose:
        print(f"[loader] {out.summary()}")
        if out.skipped:
            print(f"[loader] skipped {len(out.skipped)} episodes:")
            for ep_id, reason in out.skipped[:8]:
                print(f"[loader]   {ep_id}: {reason}")
            if len(out.skipped) > 8:
                print(f"[loader]   ... and {len(out.skipped) - 8} more")

    return out
