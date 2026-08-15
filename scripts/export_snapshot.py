#!/usr/bin/env python
"""Export the default pipeline run to ``web/public/snapshot.json``.

The web frontend first-paints from this file rather than waiting on a pipeline
run, so the narrative page shows real numbers immediately and stays readable with
no server running at all. Interactive parts -- goal queries, graph weights --
still need ``uvicorn server.api:app``.

Regenerate it whenever the episode set or the pipeline defaults change:

    python scripts/export_snapshot.py
    python scripts/export_snapshot.py --data-dir data_synth --out web/public/snapshot.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from server.api import build_snapshot  # noqa: E402  - needs the path shim above


def publish_ablation(source: str, destination: Path) -> bool:
    """Copy a difficulty-ablation payload to the web app, made browser-safe.

    This exists because the payload is written by a different tool and arrives with
    bare ``NaN`` literals in it -- a Wilcoxon p-value is genuinely undefined when all
    40 goals tie exactly, and Python's ``json`` emits and accepts ``NaN`` happily.
    JavaScript's ``JSON.parse`` does not: the browser throws and the section silently
    fails to render, while the file looks perfectly fine in an editor.

    So it is re-serialised through the same ``jsonable`` helper the API uses, which
    maps non-finite floats to ``null``, and written with ``allow_nan=False`` so a leak
    fails here rather than in front of a reader.
    """
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = REPO_ROOT / source_path
    if not source_path.exists():
        print(f"  (no ablation payload at {source}; skipping)")
        return False

    from server.serialize import jsonable

    payload = jsonable(json.loads(source_path.read_text(), parse_constant=lambda _: None))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, allow_nan=False))
    scope = payload.get("scope", "?") if isinstance(payload, dict) else "?"
    print(f"  ablation:  {source_path.name} -> {destination.name}  (scope: {scope})")

    # Loud, because it is silently wrong rather than broken. Inside one task family
    # there is barely any task-switching left to differentiate, so every
    # difficulty-based sort key posts the same number and the section renders four
    # identical bars -- which reads as "the metric does nothing" when the unscoped
    # run separates the keys cleanly.
    if isinstance(payload, dict) and payload.get("is_scoped"):
        print(
            f"  WARNING: that payload is scoped to '{payload.get('domain')}'. The ablation\n"
            f"           collapses within a single domain and will look like a null result.\n"
            f"           Regenerate unscoped:  python find_path.py \"<goal>\" --sweep 40 --out <dir>"
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="directory of .zarr episodes")
    parser.add_argument(
        "--out",
        default="web/public/snapshot.json",
        help="output path for the snapshot JSON",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="pretty-print with this indent; default is compact, which is ~40%% smaller",
    )
    parser.add_argument(
        "--ablation",
        default="reports/difficulty_ablation_unscoped.json",
        help="difficulty-ablation payload to publish to web/public/ablation.json; "
        "must be an UNSCOPED sweep or the comparison collapses (see publish_ablation)",
    )
    args = parser.parse_args()

    print(f"Running the default pipeline over {args.data_dir!r} ...", flush=True)
    try:
        payload = build_snapshot(args.data_dir)
    except Exception as error:  # noqa: BLE001 - report and exit rather than traceback
        # The common case is an empty data/ directory, which is a setup problem
        # rather than a bug, so it gets an actionable message instead of a stack.
        print(f"\nCould not build a snapshot: {error}", file=sys.stderr)
        print(
            "\nFetch real episodes:\n"
            "  python scripts/fetch_egoverse_data.py --sources yam scale aria --limit 40\n"
            "or generate synthetic ones in the same on-disk schema:\n"
            "  python scripts/generate_synthetic_data.py --out data_synth\n"
            "  python scripts/export_snapshot.py --data-dir data_synth",
            file=sys.stderr,
        )
        return 1

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False is the point: server/serialize.py has already mapped NaN to
    # None, so if any slips through, this raises here at export time rather than
    # producing a file that JSON.parse rejects in the browser.
    out_path.write_text(json.dumps(payload, indent=args.indent, allow_nan=False))

    publish_ablation(args.ablation, out_path.parent / "ablation.json")

    size_kb = out_path.stat().st_size / 1024
    metrics = payload["diversity_metrics"]
    ari = payload["agreement"].get("task_name")
    print(
        f"\nWrote {out_path.relative_to(REPO_ROOT)}  ({size_kb:,.0f} KB)\n"
        f"  episodes        {payload['n_episodes']} usable / {payload['n_skipped']} rejected\n"
        f"  diversity       {metrics['diversity_score']:.5f} m\n"
        f"  redundancy      {metrics['redundancy_ratio']:.1%}\n"
        f"  clusters        {payload['n_clusters']} (suggested {payload['suggested_k']})\n"
        f"  ARI vs task     {ari if ari is None else f'{ari:.3f}'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
