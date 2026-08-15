"""NumPy/pandas -> JSON conversion.

The one thing this module exists for is **NaN**. JSON has no NaN literal, and
this pipeline produces NaN legitimately and often:

- ``CurriculumPath.table`` has no ``edge_weight``, ``ramp_cost`` or
  ``difficulty_delta`` on step 1, because nothing precedes it.
- The same table deliberately blanks ``ramp_cost`` and ``edge_weight`` on every
  review row, since a rehearsal step is *meant* to drop in difficulty and its
  ramp cost would read as a huge violation.
- ``diversity_report`` returns NaN for ``silhouette`` and ``cluster_balance``
  when there are too few clusters to define them.

FastAPI's default encoder emits a bare ``NaN`` token for these, which is invalid
JSON: ``JSON.parse`` rejects it and the frontend fails on a response that looked
fine in curl. Everything is therefore funnelled through :func:`jsonable`, which
maps non-finite floats to ``None`` so the UI can render them as an em dash --
the honest reading, where ``0.0`` would assert a measurement never taken.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def jsonable(value: Any) -> Any:
    """Recursively convert NumPy/pandas values to strict-JSON-safe Python.

    Non-finite floats (NaN, +/-Inf) become ``None``. NumPy scalars become Python
    scalars. DataFrames become lists of row dicts. Everything else passes
    through unchanged.
    """
    # Order matters: np.bool_ is not a subclass of bool, and np.integer must be
    # checked before np.floating would coerce it.
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return frame_records(value)
    if isinstance(value, pd.Series):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        # Keys must be strings in JSON; cluster ids and k values arrive as ints.
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """A DataFrame as JSON-safe row dicts.

    ``DataFrame.to_dict("records")`` alone is not enough: it preserves NaN and
    NumPy scalar types, both of which break strict JSON.
    """
    if frame is None or frame.empty:
        return []
    # object_hook-free path: convert column-wise, which is much faster than
    # per-cell for the wide curriculum table.
    clean = frame.astype(object).where(pd.notna(frame), None)
    return [{str(k): jsonable(v) for k, v in row.items()} for row in clean.to_dict("records")]


def decimate(points: np.ndarray, max_points: int = 240) -> np.ndarray:
    """Evenly subsample a ``(T, C)`` polyline to at most ``max_points`` rows.

    Real episodes reach 3,712 frames and the hero plot draws 28 of them at once.
    Sending every frame would be megabytes of JSON to render strokes a few
    hundred pixels wide, where the extra samples are invisible.

    Endpoints are always kept, so the start and end of the motion are exact --
    which matters because net displacement between them is a reported feature.
    """
    n = len(points)
    if n <= max_points:
        return np.asarray(points, dtype=float)
    index = np.linspace(0, n - 1, max_points).round().astype(int)
    return np.asarray(points, dtype=float)[index]
