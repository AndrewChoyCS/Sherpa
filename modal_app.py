"""Modal offload for the slow parts of the EgoVerse curriculum pipeline.

Two things are slow locally, for different reasons, and Modal fixes both differently:

**Fetching episodes is network-bound, not CPU-bound.** Discovery has to page through R2
prefix listings one source at a time, and a source holding tens of thousands of episode
prefixes takes minutes to enumerate from a laptop -- long enough that an unscoped fetch
looks like a hang, and long enough to hit read timeouts. Here each source is discovered in
its own container and the downloads are then fanned out across containers, so wall-clock is
the *slowest single source* rather than the sum of all of them, over R2-adjacent
datacentre networking rather than home broadband.

**The DTW matrix is CPU-bound and quadratic.** ``O(N^2 * T^2)`` over a few hundred clips is
seconds on many cores and minutes on few. :func:`run_pipeline_remote` runs it on a wide
container and persists the result to a volume cache keyed by the same content hash the
local code uses, so a later local run reuses it instead of recomputing.

Episodes are *pose-only* (~5 KB each, versus ~300 MB with the JPEG streams), so a
300-episode dataset is a couple of megabytes. That is why the split works: the heavy work
happens remotely, and the artifact that comes back is small enough to just download and
keep using locally.

Two persistent volumes:
    egoverse-data    fetched episodes (``/data/episodes``) and artifacts (``/data/reports``)
    egoverse-cache   DTW matrices, keyed by trajectory content hash

Usage:
    # one-off: fetch episodes into the volume, in parallel, then pull them down
    modal run modal_app.py::fetch --limit 320
    modal run modal_app.py::sync_down

    # run the quadratic part remotely and bring the artifacts back
    modal run modal_app.py::pipeline
    modal run modal_app.py::find --goal "teach the robot to fold a shirt"

    # inspect what is in the volume
    modal run modal_app.py::status
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import modal

APP_NAME = "egoverse-curriculum"

# Sources carrying real manipulation pose data. The bucket also holds `test_*`, `*_old`
# and `proc_test_*` variants; including them spends the discovery budget on duplicates.
DEFAULT_SOURCES = ("yam", "scale", "aria", "mecka")

DATA_ROOT = "/data"
CACHE_ROOT = "/cache"
EPISODES_DIR = f"{DATA_ROOT}/episodes"
REPORTS_DIR = f"{DATA_ROOT}/reports"

# The episode set the deployed web app serves. Kept separate from EPISODES_DIR, which
# `fetch` overwrites: the browser's first paint comes from a `snapshot.json` exported
# against one specific dataset, so the live API has to read that same set or the static
# page and the interactive controls quietly disagree about the numbers.
WEB_EPISODES_DIR = f"{DATA_ROOT}/episodes_web"
# Matching DTW/dataset caches. Separate from the batch cache at CACHE_ROOT for the same
# reason: these are keyed by a content hash of WEB_EPISODES_DIR's trajectories, so mixing
# them with another episode set's entries just produces misses.
WEB_CACHE_DIR = f"{CACHE_ROOT}/.cache"

data_volume = modal.Volume.from_name("egoverse-data", create_if_missing=True)
cache_volume = modal.Volume.from_name("egoverse-cache", create_if_missing=True)
r2_secret = modal.Secret.from_name("egoverse-r2")

# The repo's own source is added to the image so remote code is the *same* code the local
# CLI and dashboard run -- results cannot diverge between local and remote.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .env({"OMP_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"})
    .add_local_dir("src", "/root/src")
    .add_local_dir("scripts", "/root/scripts")
    .add_local_file("run_pipeline.py", "/root/run_pipeline.py")
    .add_local_file("find_path.py", "/root/find_path.py")
    # The web surface. `server/api.py` imports DOMAIN_PRESETS from find_path and reads
    # its defaults out of run_pipeline's signature, so both files above are load-bearing
    # here even though nothing runs them as CLIs.
    .add_local_dir("server", "/root/server")
    .add_local_dir("web/dist", "/root/web/dist")
)

app = modal.App(APP_NAME)


# --------------------------------------------------------------------------- #
# helpers shared by the remote functions
# --------------------------------------------------------------------------- #
def _allocated_cpus(default: int = 16) -> int:
    """CPUs actually allocated to this container, not the host's core count.

    ``os.cpu_count()`` reports the host inside a container, so joblib's ``n_jobs=-1``
    over-subscribes badly. The cgroup v2 quota is the real allocation.
    """
    try:
        quota = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota[0] != "max":
            return max(1, int(int(quota[0]) / int(quota[1])))
    except Exception:  # noqa: BLE001 - fall back rather than fail the job
        pass
    return min(default, os.cpu_count() or default)


def _r2_client():
    """R2 client built from the Modal secret rather than ``~/.egoverse_env``."""
    sys.path.insert(0, "/root/scripts")
    from fetch_egoverse_data import make_r2_client

    return make_r2_client(dict(os.environ))


def _tar_bytes(root: Path, arcname: str) -> bytes:
    """Tar a directory into memory. Pose-only episodes are small enough for this."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        if root.exists():
            archive.add(str(root), arcname=arcname)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# 1. discovery, one container per source
# --------------------------------------------------------------------------- #
@app.function(image=image, secrets=[r2_secret], timeout=3600, max_containers=16)
def discover_source(source: str, max_episodes: Optional[int], prefix: str) -> List[str]:
    """Enumerate episode prefixes for one source.

    Runs per-source so the walk of a huge source overlaps with the small ones instead of
    serialising behind them.
    """
    sys.path.insert(0, "/root/scripts")
    from fetch_egoverse_data import discover_episodes

    client = _r2_client()
    bucket = os.environ.get("BUCKET", "rldb")
    started = time.time()
    episodes = discover_episodes(
        client, bucket, prefix, [source], max_per_source=max_episodes
    )
    print(
        f"[discover] {source}: {len(episodes)} episodes in {time.time() - started:.1f}s"
    )
    return [ep for ep, _ in episodes]


# --------------------------------------------------------------------------- #
# 2. download, fanned out across containers
# --------------------------------------------------------------------------- #
@app.function(
    image=image,
    secrets=[r2_secret],
    volumes={DATA_ROOT: data_volume},
    timeout=3600,
    max_containers=32,
)
def fetch_batch(prefixes: List[str], root_prefix: str) -> Dict[str, object]:
    """Download the pose arrays for a batch of episodes into the data volume."""
    sys.path.insert(0, "/root/scripts")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from fetch_egoverse_data import POSE_KEYS, fetch_episode, local_name

    client = _r2_client()
    bucket = os.environ.get("BUCKET", "rldb")
    out_dir = Path(EPISODES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, failed, total_bytes = 0, [], 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = {}
        for prefix in prefixes:
            name = local_name(prefix, "", root_prefix)
            dest = out_dir / f"{name}.zarr"
            futures[
                pool.submit(fetch_episode, client, bucket, prefix, dest, list(POSE_KEYS))
            ] = prefix
        for future in as_completed(futures):
            prefix, n_objects, n_bytes, error = future.result()
            if error:
                failed.append((prefix, error))
            else:
                ok += 1
                total_bytes += n_bytes

    data_volume.commit()
    print(f"[fetch] {ok} ok, {len(failed)} failed, {total_bytes / 1e6:.1f} MB")
    return {"ok": ok, "failed": failed[:10], "n_failed": len(failed), "bytes": total_bytes}


# --------------------------------------------------------------------------- #
# 3. the quadratic part
# --------------------------------------------------------------------------- #
@app.function(
    image=image,
    volumes={DATA_ROOT: data_volume, CACHE_ROOT: cache_volume},
    cpu=16.0,
    memory=16384,
    timeout=7200,
)
def run_pipeline_remote(config: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Run the Track 2 pipeline on a wide container, caching the DTW matrix on a volume.

    Returns the small artifacts inline (CSV/JSON text and the packed matrices) so the
    caller can write them locally; the same files are also left in ``/data/reports``.
    """
    sys.path.insert(0, "/root")
    import numpy as np

    from src.pipeline import run_pipeline

    settings: Dict[str, object] = {
        "data_dir": EPISODES_DIR,
        "cache_dir": CACHE_ROOT,
        # Explicit, not -1. joblib's -1 resolves via os.cpu_count(), which inside a
        # container reports the *host's* cores, not the cgroup allocation -- so -1 spawns
        # dozens of workers onto 16 CPUs and the DTW matrix thrashes instead of scaling.
        "n_jobs": _allocated_cpus(),
        "verbose": True,
    }
    settings.update(config or {})
    print(f"[pipeline] settings: { {k: v for k, v in settings.items()} }")

    started = time.time()
    result = run_pipeline(**settings)  # type: ignore[arg-type]
    elapsed = time.time() - started
    print(f"[pipeline] {result.n_episodes} episodes in {elapsed:.1f}s")

    reports = Path(REPORTS_DIR)
    reports.mkdir(parents=True, exist_ok=True)
    result.curriculum.to_csv(reports / "curriculum.csv", index=False)
    result.stages.to_csv(reports / "stages.csv", index=False)
    result.frame().to_csv(reports / "episodes.csv", index=False)
    np.save(reports / "dtw_matrix.npy", result.distance_matrix)
    np.save(reports / "embedding.npy", result.embedding)

    metrics = {
        "diversity_metrics": result.report,
        "cluster_label_agreement_ari": result.agreement,
        "suggested_k": result.suggested_k,
        "n_episodes_loaded": result.n_episodes,
        "n_episodes_skipped": len(result.dataset.skipped),
        "skipped": [{"episode_id": e, "reason": r} for e, r in result.dataset.skipped],
        "elapsed_seconds": elapsed,
        "config": {k: v for k, v in settings.items() if k != "verbose"},
    }
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    data_volume.commit()
    cache_volume.commit()
    return {
        "metrics": metrics,
        "n_episodes": result.n_episodes,
        "elapsed_seconds": elapsed,
        "reports_tar": _tar_bytes(reports, "reports"),
    }


@app.function(
    image=image,
    volumes={DATA_ROOT: data_volume, CACHE_ROOT: cache_volume},
    cpu=16.0,
    memory=16384,
    timeout=7200,
)
def find_path_remote(
    goal: str,
    graph_overrides: Optional[Dict[str, object]] = None,
    review_every: int = 4,
    task_scope: Optional[Sequence[str]] = None,
    n_seeds: int = 50,
) -> Dict[str, object]:
    """Resolve a goal and search for its curriculum path, remotely.

    Uses the same :mod:`src` code as the local CLI, so the returned metrics are directly
    comparable to ``python find_path.py``.
    """
    sys.path.insert(0, "/root")
    from src.graph import GraphConfig
    from src.path_metrics import compare_orderings, path_report
    from src.pathfinder import PathConfig
    from src.pipeline import build_path_finder, run_pipeline

    result = run_pipeline(data_dir=EPISODES_DIR, cache_dir=CACHE_ROOT, verbose=True)
    if result.n_episodes < 2:
        return {"error": f"only {result.n_episodes} usable episode(s) in the volume"}

    graph_config = GraphConfig(**(graph_overrides or {}))
    context = build_path_finder(result, graph_config, task_names=task_scope)
    match, path = context.find(goal, PathConfig(review_every=review_every))

    report = path_report(
        path.clips, context.clip_graph, context.distance_matrix,
        path.target_index, path.is_review,
    )
    comparison = compare_orderings(
        path, context.clip_graph, context.distance_matrix, n_seeds=n_seeds
    )

    reports = Path(REPORTS_DIR)
    reports.mkdir(parents=True, exist_ok=True)
    path.table.to_csv(reports / "path.csv", index=False)
    payload = {
        "goal": goal,
        "matched_task": match.task_name,
        "match_score": match.score,
        "match_margin": match.margin,
        "match_is_confident": match.is_confident,
        "match_note": match.note,
        "target_episode": context.clip_graph.episode_id(path.target_index),
        "n_clips": len(path.route),
        "n_reviews": path.n_reviews,
        "search_cost": path.search_cost,
        "cost_terms": path.cost_terms,
        "graph": context.clip_graph.summary(),
        "path_metrics": report,
        "ordering_comparison": json.loads(comparison.to_json(orient="index")),
    }
    (reports / "path_metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    data_volume.commit()
    cache_volume.commit()

    payload["path_csv"] = path.table.to_csv(index=False)
    return payload


# --------------------------------------------------------------------------- #
# 4. moving data back and forth
# --------------------------------------------------------------------------- #
@app.function(
    image=image,
    volumes={DATA_ROOT: data_volume, CACHE_ROOT: cache_volume},
    cpu=16.0,
    memory=16384,
    timeout=7200,
)
def probe(max_length: int = 200) -> Dict[str, object]:
    """Time the stages separately, so a slow run can be attributed rather than guessed at.

    Volume reads and the quadratic DTW fail in different ways and are fixed differently;
    a single wall-clock number cannot tell them apart.
    """
    sys.path.insert(0, "/root")
    from src.diversity_engine import DTWConfig, compute_dtw_matrix
    from src.loader import load_zarr_trajectories

    cpus = _allocated_cpus()
    timings: Dict[str, object] = {"allocated_cpus": cpus, "host_cpu_count": os.cpu_count()}

    started = time.time()
    dataset = load_zarr_trajectories(EPISODES_DIR, verbose=True, cache_dir=CACHE_ROOT)
    timings["load_seconds"] = round(time.time() - started, 2)
    timings["n_episodes"] = len(dataset)
    timings["n_skipped"] = len(dataset.skipped)
    if len(dataset):
        lengths = dataset.lengths
        timings["frames_min_median_max"] = [
            int(lengths.min()), int(lengths.median() if hasattr(lengths, "median") else
                                    __import__("numpy").median(lengths)), int(lengths.max())
        ]

    started = time.time()
    matrix = compute_dtw_matrix(
        dataset.trajectories,
        DTWConfig(max_length=max_length, n_jobs=cpus),
        cache_dir=CACHE_ROOT,
    )
    timings["dtw_seconds"] = round(time.time() - started, 2)
    n = matrix.shape[0]
    pairs = n * (n - 1) // 2
    timings["n_pairs"] = pairs
    if pairs:
        timings["ms_per_pair"] = round(1000 * timings["dtw_seconds"] / pairs, 4)
    cache_volume.commit()
    return timings


@app.function(image=image, volumes={DATA_ROOT: data_volume}, timeout=1800)
def snapshot_episodes() -> bytes:
    """Tar the fetched episodes for download. A few MB, because poses only."""
    return _tar_bytes(Path(EPISODES_DIR), "episodes")


@app.function(image=image, volumes={DATA_ROOT: data_volume, CACHE_ROOT: cache_volume})
def volume_status() -> Dict[str, object]:
    """Episode and artifact counts in the volumes."""
    episodes = sorted(Path(EPISODES_DIR).glob("*.zarr")) if Path(EPISODES_DIR).exists() else []
    by_source: Dict[str, int] = {}
    for path in episodes:
        by_source.setdefault(path.name.split("__", 1)[0], 0)
        by_source[path.name.split("__", 1)[0]] += 1
    reports = sorted(p.name for p in Path(REPORTS_DIR).glob("*")) if Path(REPORTS_DIR).exists() else []
    caches = sorted(p.name for p in Path(CACHE_ROOT).glob("dtw_*.npy")) if Path(CACHE_ROOT).exists() else []
    return {"n_episodes": len(episodes), "by_source": by_source,
            "reports": reports, "dtw_caches": caches}


# --------------------------------------------------------------------------- #
# the deployed web app
# --------------------------------------------------------------------------- #
@app.function(
    image=image,
    volumes={DATA_ROOT: data_volume, CACHE_ROOT: cache_volume},
    cpu=4.0,
    memory=8192,
    scaledown_window=600,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    """Serve Sherpa -- the React frontend and the JSON API -- as one ASGI app.

    ``server/api.py`` already mounts ``web/dist`` at ``/`` when the directory exists, so
    a single process covers both surfaces and the frontend's relative ``/api`` calls are
    same-origin. Nothing here adds behaviour; it only points the pipeline at the volumes.
    """
    sys.path.insert(0, "/root")
    os.chdir("/root")

    # `run_pipeline`'s defaults are the *relative* strings "data" and ".cache"
    # (src/pipeline.py), and `cache_dir` is not in the API's TUNABLE list, so a request
    # cannot redirect it. Symlinking the volumes onto those two names means the deployed
    # app and the local CLI run byte-identical code paths rather than a forked config.
    for link, target in ((Path("/root/data"), WEB_EPISODES_DIR),
                         (Path("/root/.cache"), WEB_CACHE_DIR)):
        if not link.exists():
            link.symlink_to(target)

    # The API calls run_pipeline without n_jobs, so it inherits n_jobs=-1, which joblib
    # resolves through os.cpu_count() -- the *host's* cores, not the cgroup's. Same
    # oversubscription trap the batch functions avoid with _allocated_cpus().
    os.environ["LOKY_MAX_CPU_COUNT"] = str(_allocated_cpus(4))

    from server.api import app as api

    return api


# --------------------------------------------------------------------------- #
# local entrypoints
# --------------------------------------------------------------------------- #
def _chunks(items: Sequence[str], size: int) -> List[List[str]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _balance(per_source: Dict[str, List[str]], limit: int) -> List[str]:
    """Round-robin across sources so one huge source cannot crowd out the rest.

    Mirrors ``_balanced_take`` in the local fetcher: a diversity tool that silently drew
    everything from one source would report a misleadingly low diversity score.
    """
    for values in per_source.values():
        values.sort()
    picked: List[str] = []
    index = 0
    order = sorted(per_source)
    while len(picked) < limit:
        progressed = False
        for source in order:
            group = per_source[source]
            if index < len(group):
                picked.append(group[index])
                progressed = True
                if len(picked) == limit:
                    return picked
        if not progressed:
            break
        index += 1
    return picked


@app.local_entrypoint()
def fetch(
    limit: int = 320,
    sources: str = ",".join(DEFAULT_SOURCES),
    prefix: str = "processed_v3/",
    batch_size: int = 20,
):
    """Discover and download episodes into the data volume, in parallel."""
    source_list = [s.strip() for s in sources.split(",") if s.strip()]
    budget = max(4 * limit, 50)
    print(f"Discovering up to {budget} episodes per source across {source_list} ...")

    started = time.time()
    per_source: Dict[str, List[str]] = {}
    args = [(source, budget, prefix) for source in source_list]
    for source, found in zip(
        source_list, discover_source.starmap(args, order_outputs=True)
    ):
        per_source[source] = list(found)
        print(f"  {source}: {len(found)} episodes")
    print(f"Discovery took {time.time() - started:.1f}s")

    selected = _balance(per_source, limit)
    print(f"Selected {len(selected)} episodes, balanced round-robin across sources.")

    started = time.time()
    ok = failed = 0
    total_bytes = 0
    batches = _chunks(selected, batch_size)
    for outcome in fetch_batch.starmap([(b, prefix) for b in batches]):
        ok += int(outcome["ok"])
        failed += int(outcome["n_failed"])
        total_bytes += int(outcome["bytes"])
    print(
        f"\nFetched {ok} episodes ({failed} failed / no pose arrays), "
        f"{total_bytes / 1e6:.1f} MB, in {time.time() - started:.1f}s"
    )
    print(json.dumps(volume_status.remote(), indent=2)[:800])


@app.local_entrypoint()
def sync_down(dest: str = "data"):
    """Download the volume's episodes into a local directory for the Streamlit app."""
    print("Packing episodes from the volume ...")
    blob = snapshot_episodes.remote()
    target = Path(dest)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        members = [m for m in archive.getmembers() if m.name != "episodes"]
        for member in members:
            # Strip the leading "episodes/" so stores land directly in `dest`.
            member.name = member.name.split("/", 1)[1] if "/" in member.name else member.name
            archive.extract(member, path=target, filter="data")
    stores = sorted(p.name for p in target.glob("*.zarr"))
    print(f"Wrote {len(stores)} episode stores ({len(blob) / 1e6:.1f} MB compressed) to {target}/")


@app.local_entrypoint()
def pipeline(out: str = "reports"):
    """Run the Track 2 pipeline remotely and write its artifacts locally."""
    outcome = run_pipeline_remote.remote()
    print(f"{outcome['n_episodes']} episodes in {outcome['elapsed_seconds']:.1f}s")
    print(json.dumps(outcome["metrics"]["diversity_metrics"], indent=2))
    print(json.dumps(outcome["metrics"]["cluster_label_agreement_ari"], indent=2))

    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(outcome["reports_tar"]), mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.name == "reports":
                continue
            member.name = member.name.split("/", 1)[1] if "/" in member.name else member.name
            archive.extract(member, path=target, filter="data")
    print(f"Artifacts written to {target}/")


@app.local_entrypoint()
def find(
    goal: str = "teach the robot to fold a shirt",
    review_every: int = 4,
    w_interference: float = 1.0,
    scope: str = "",
    out: str = "reports",
):
    """Resolve a goal and search for its curriculum path remotely."""
    payload = find_path_remote.remote(
        goal,
        graph_overrides={"w_interference": w_interference},
        review_every=review_every,
        task_scope=[s.strip() for s in scope.split(",") if s.strip()] or None,
    )
    if "error" in payload:
        print("ERROR:", payload["error"])
        return

    print(f"Goal            : {goal!r}")
    print(f"Matched task    : {payload['matched_task']} "
          f"(score {payload['match_score']:.2f}, lead {payload['match_margin']:.0%})")
    if not payload["match_is_confident"]:
        print("  !", payload["match_note"])
    print(f"Target episode  : {payload['target_episode']}")
    print(f"Graph           : {payload['graph']}")
    print(f"Path            : {payload['n_clips']} clips + {payload['n_reviews']} reviews, "
          f"cost {payload['search_cost']:.2f}")

    import pandas as pd

    comparison = pd.DataFrame(payload["ordering_comparison"]).T
    columns = [
        c for c in ("spearman", "frac_nondecreasing", "max_jump", "task_switch_rate",
                    "cluster_switch_rate", "cluster_coverage", "task_coverage",
                    "frac_consecutive_near_duplicate")
        if c in comparison.columns
    ]
    print("\nProxy metrics vs. baselines:")
    print(comparison[columns].astype(float).round(3).to_string())

    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    (target / "path.csv").write_text(payload.pop("path_csv"))
    (target / "path_metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nArtifacts written to {target}/")


@app.local_entrypoint()
def status():
    """Print what the volumes currently hold."""
    print(json.dumps(volume_status.remote(), indent=2))
