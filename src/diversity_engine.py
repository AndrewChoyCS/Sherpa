"""Pairwise Dynamic Time Warping distance matrix over end-effector trajectories.

Design notes that differ from a naive implementation, and why:

**Normalisation is a choice, not a detail.** Raw XYZ trajectories are dominated by
*where* in the workspace the motion happened, which swamps the *shape* of the
motion. Three modes are offered:

- ``"center"`` (default) subtracts each trajectory's own mean, removing absolute
  placement while preserving motion *extent*. A 5 cm nudge and a 50 cm sweep stay
  far apart -- which is correct, they are genuinely different skills.
- ``"zscore"`` additionally divides by per-axis standard deviation. This makes the
  5 cm nudge and the 50 cm sweep *identical*. Use it only when you want pure
  shape similarity and consider scale a nuisance parameter.
- ``"none"`` keeps absolute world coordinates.

**DTW cost grows with sequence length, as sqrt(T).** tslearn's DTW returns the
*root* of the summed squared local distances along the warping path, so aligning two
length-T series separated by a constant offset ``d`` costs ``d * sqrt(T)`` -- not
``d * T``. Left uncorrected, "diversity" partly degenerates into "variance in episode
duration", which matters here because EgoVerse episodes span 295 to 3,712 frames.

``length_normalize`` therefore divides each pair by the **square root** of its mean
sequence length. Dividing by the mean length itself (the intuitive choice) over-corrects
and makes longer episodes score as systematically *less* distant. Measured on a fixed
pair of trajectories resampled from T=100 to T=1600, ``cost/sqrt(T)`` stays flat within
0.5% while ``cost/T`` falls by a factor of four.

**Cost is O(N^2 * T^2).** ``max_length`` resamples long episodes down (preserving
shape via linear interpolation over normalised time) and ``sakoe_chiba_radius``
optionally restricts the warping band. Both are large constant-factor savings.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence

import numpy as np
from tslearn.metrics import cdist_dtw
from tslearn.utils import to_time_series_dataset

NORMALIZE_MODES = ("center", "zscore", "none")

# Bumped whenever the *meaning* of a config value changes rather than its value, so
# that on-disk caches written by an older definition are not silently reused.
# v2: length_normalize divides by sqrt(mean length) instead of mean length.
_FINGERPRINT_VERSION = 2


@dataclass(frozen=True)
class DTWConfig:
    """Configuration for the DTW distance computation.

    Attributes:
        normalize: One of ``"center"``, ``"zscore"``, ``"none"``. See module docstring.
        max_length: Resample any trajectory longer than this down to this many
            timesteps. ``None`` disables resampling. Quadratic cost saver.
        length_normalize: Divide each pairwise cost by the pair's mean sequence
            length so long episodes are not spuriously "diverse".
        sakoe_chiba_radius: Warping band radius. ``None`` means unconstrained DTW.
        n_jobs: Parallel workers passed to tslearn (``-1`` = all cores).
        verbose: tslearn verbosity for progress output.
    """

    normalize: str = "center"
    max_length: Optional[int] = 200
    length_normalize: bool = True
    sakoe_chiba_radius: Optional[int] = None
    n_jobs: int = -1
    verbose: int = 0

    def __post_init__(self) -> None:
        if self.normalize not in NORMALIZE_MODES:
            raise ValueError(f"normalize must be one of {NORMALIZE_MODES}, got {self.normalize!r}")
        if self.max_length is not None and self.max_length < 2:
            raise ValueError("max_length must be >= 2 or None")

    def fingerprint_fields(self) -> dict:
        """Config fields that affect the *result* (excludes scheduling knobs)."""
        d = asdict(self)
        d.pop("n_jobs", None)
        d.pop("verbose", None)
        return d


# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #
def resample_trajectory(traj: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample a ``(T, D)`` trajectory onto ``target_length`` timesteps.

    Interpolation happens over normalised time so trajectory *shape* is preserved
    while the timestep count changes.
    """
    traj = np.asarray(traj, dtype=np.float64)
    n = traj.shape[0]
    if n == target_length:
        return traj
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, target_length)
    return np.stack([np.interp(dst, src, traj[:, d]) for d in range(traj.shape[1])], axis=1)


def normalize_trajectory(traj: np.ndarray, mode: str) -> np.ndarray:
    """Apply per-trajectory normalisation. See module docstring for semantics."""
    traj = np.asarray(traj, dtype=np.float64)
    if mode == "none":
        return traj
    centered = traj - traj.mean(axis=0, keepdims=True)
    if mode == "center":
        return centered
    if mode == "zscore":
        std = centered.std(axis=0, keepdims=True)
        # A perfectly flat axis has zero std; dividing would emit NaN/inf. Leave it
        # at zero -- it carries no shape information either way.
        std = np.where(std < 1e-12, 1.0, std)
        return centered / std
    raise ValueError(f"unknown normalize mode {mode!r}")


def preprocess_trajectories(
    trajectories: Sequence[np.ndarray], config: DTWConfig
) -> List[np.ndarray]:
    """Resample-then-normalise every trajectory according to ``config``."""
    out: List[np.ndarray] = []
    for traj in trajectories:
        arr = np.asarray(traj, dtype=np.float64)
        if config.max_length is not None and arr.shape[0] > config.max_length:
            arr = resample_trajectory(arr, config.max_length)
        out.append(normalize_trajectory(arr, config.normalize))
    return out


# --------------------------------------------------------------------------- #
# caching
# --------------------------------------------------------------------------- #
def dataset_fingerprint(trajectories: Sequence[np.ndarray], config: DTWConfig) -> str:
    """Content hash over trajectory bytes + result-affecting config."""
    h = hashlib.sha256()
    h.update(f"v{_FINGERPRINT_VERSION}".encode())
    h.update(repr(sorted(config.fingerprint_fields().items())).encode())
    for traj in trajectories:
        arr = np.ascontiguousarray(np.asarray(traj, dtype=np.float64))
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()[:16]


def _cache_path(cache_dir: str, fingerprint: str) -> str:
    return os.path.join(cache_dir, f"dtw_{fingerprint}.npy")


# --------------------------------------------------------------------------- #
# main computation
# --------------------------------------------------------------------------- #
def compute_dtw_matrix(
    trajectories: Sequence[np.ndarray],
    config: Optional[DTWConfig] = None,
    cache_dir: Optional[str] = None,
) -> np.ndarray:
    """Compute the symmetric pairwise DTW distance matrix.

    Args:
        trajectories: Variable-length ``(T_i, D)`` arrays.
        config: Preprocessing and DTW settings. Defaults to :class:`DTWConfig`.
        cache_dir: If given, memoise the matrix to disk keyed by a content hash of
            the trajectories and the result-affecting config.

    Returns:
        ``(N, N)`` float64 matrix, zero diagonal, symmetric. ``(0, 0)`` for empty
        input and ``(1, 1)`` zeros for a single trajectory.
    """
    config = config or DTWConfig()
    n = len(trajectories)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if n == 1:
        return np.zeros((1, 1), dtype=np.float64)

    fingerprint = dataset_fingerprint(trajectories, config)
    if cache_dir:
        path = _cache_path(cache_dir, fingerprint)
        if os.path.exists(path):
            try:
                cached = np.load(path)
                if cached.shape == (n, n):
                    print(f"[dtw] cache hit {os.path.basename(path)}")
                    return cached
            except Exception as exc:  # noqa: BLE001 - a corrupt cache must not be fatal
                warnings.warn(f"ignoring unreadable DTW cache {path}: {exc}", stacklevel=2)

    processed = preprocess_trajectories(trajectories, config)
    # to_time_series_dataset pads short series with NaN; tslearn's DTW treats the
    # NaN tail as "series ended", so variable lengths are handled natively.
    dataset = to_time_series_dataset(processed)

    print(
        f"[dtw] computing {n}x{n} DTW matrix "
        f"(normalize={config.normalize}, max_length={config.max_length}, "
        f"length_normalize={config.length_normalize})..."
    )
    matrix = _cdist_dtw_safe(dataset, config)

    # tslearn returns a float32/float64 array that is symmetric up to numerical
    # noise; enforce exact symmetry and a zero diagonal so downstream consumers
    # (UMAP precomputed, silhouette) never trip their validation checks.
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)

    if config.length_normalize:
        # sqrt, not the mean length itself: tslearn's DTW is a root-sum-square, so
        # cost scales as sqrt(path length). See the module docstring.
        lengths = np.array([p.shape[0] for p in processed], dtype=np.float64)
        denom = np.sqrt(0.5 * (lengths[:, None] + lengths[None, :]))
        matrix = matrix / denom
        np.fill_diagonal(matrix, 0.0)

    if not np.isfinite(matrix).all():
        raise ValueError(
            "DTW matrix contains non-finite values; check trajectories for NaN/inf."
        )

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(_cache_path(cache_dir, fingerprint), matrix)

    return matrix


def _cdist_dtw_safe(dataset: np.ndarray, config: DTWConfig) -> np.ndarray:
    """Run ``cdist_dtw``, degrading to single-threaded on a parallel-backend failure.

    joblib's process backend can fail inside constrained hosts (Streamlit reruns,
    sandboxes, notebooks). A diversity report is worth more than a stack trace, so
    fall back rather than propagate.
    """
    kwargs = dict(
        global_constraint="sakoe_chiba" if config.sakoe_chiba_radius else None,
        sakoe_chiba_radius=config.sakoe_chiba_radius,
        verbose=config.verbose,
    )
    try:
        return cdist_dtw(dataset, n_jobs=config.n_jobs, **kwargs)
    except Exception as exc:  # noqa: BLE001
        if config.n_jobs in (1, None):
            raise
        warnings.warn(
            f"parallel DTW failed ({type(exc).__name__}: {exc}); retrying single-threaded.",
            stacklevel=2,
        )
        return cdist_dtw(dataset, n_jobs=1, **kwargs)
