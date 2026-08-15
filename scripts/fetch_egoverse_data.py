#!/usr/bin/env python
"""Fetch end-effector pose data for EgoVerse episodes from the Cloudflare R2 bucket.

The full ``processed_v3`` episodes are ~300 MB each because they embed JPEG camera
streams. This diversity engine only needs the end-effector pose arrays, which are
~5 KB compressed per episode -- roughly a 60,000x saving. So rather than syncing
whole stores, this script downloads a *key subset*:

    <episode>.zarr/zarr.json              (group metadata: task_name, total_frames, fps)
    <episode>.zarr/left.obs_ee_pose/**    (T, 7) pose array, if the arm exists
    <episode>.zarr/right.obs_ee_pose/**

The result is still a valid Zarr v3 group -- just one holding fewer arrays -- so the
normal loader reads it with no special-casing. Episode metadata advertises keys that
were not downloaded; the loader tolerates that by inspecting real arrays, not
``features``.

Credentials come from ``~/.egoverse_env`` (written by
``egomimic/utils/aws/aws/setup_secret.sh``). Note R2 rejects an ``X-Amz-Security-Token``
header, so any session token in that file is deliberately ignored.

Examples:
    # 200 episodes spread evenly across every source
    python scripts/fetch_egoverse_data.py --limit 200

    # only bimanual sources, into a separate directory
    python scripts/fetch_egoverse_data.py --sources yam mecka --limit 80 --out data_bimanual

    # see what is available without downloading
    python scripts/fetch_egoverse_data.py --list-sources
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_ENV_FILE = os.path.expanduser("~/.egoverse_env")
DEFAULT_ROOT_PREFIX = "processed_v3/"

# Keys worth downloading. Pose arrays are tiny; everything else is images.
POSE_KEYS = ("left.obs_ee_pose", "right.obs_ee_pose")
# Optional extras, downloaded only with --include-extras.
EXTRA_KEYS = (
    "left.obs_gripper",
    "right.obs_gripper",
    "obs_head_pose",
    "left.cmd_ee_pose",
    "right.cmd_ee_pose",
)


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #
def load_egoverse_env(path: str = DEFAULT_ENV_FILE) -> Dict[str, str]:
    """Parse the shell-style ``~/.egoverse_env`` file into a dict."""
    if not os.path.exists(path):
        raise SystemExit(
            f"Credentials file '{path}' not found.\n"
            "Run the EgoVerse setup first:\n"
            "  bash egomimic/utils/aws/setup_secret.sh"
        )
    env: Dict[str, str] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip("'\"")
    return env


def make_r2_client(env: Dict[str, str]):
    """Build a boto3 S3 client pointed at the R2 endpoint."""
    import boto3
    from botocore.config import Config

    endpoint = env.get("R2_ENDPOINT_URL") or env.get("S3_ENDPOINT_URL") or env.get(
        "AWS_ENDPOINT_URL_S3"
    )
    if not endpoint:
        raise SystemExit("No R2 endpoint URL found in the environment file.")

    access_key = env.get("R2_ACCESS_KEY_ID")
    secret_key = env.get("R2_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise SystemExit("R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY missing.")

    # R2 is not STS-aware: passing a session token makes it reject the request with
    # InvalidArgument: X-Amz-Security-Token. Omit it even when present in the file.
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            # Listing a prefix holding tens of thousands of episodes routinely exceeds
            # botocore's 60 s default, and a ReadTimeoutError there aborts the whole run
            # after several minutes of successful paging. Adaptive retries also back off
            # rather than hammering R2 when it throttles.
            retries={"max_attempts": 8, "mode": "adaptive"},
            connect_timeout=15,
            read_timeout=120,
            max_pool_connections=32,
        ),
    )


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def list_prefixes(
    client, bucket: str, prefix: str, limit: Optional[int] = None
) -> List[str]:
    """One level of "directory" prefixes below ``prefix``.

    Args:
        limit: Stop paging once this many prefixes have been collected. Some sources hold
            tens of thousands of episode prefixes, and paging all of them takes minutes
            per source; when the caller only needs a few hundred, that work is wasted.
    """
    paginator = client.get_paginator("list_objects_v2")
    out: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for entry in page.get("CommonPrefixes", []):
            out.append(entry["Prefix"])
        if limit is not None and len(out) >= limit:
            break
    return out


def _walk_source(
    client,
    bucket: str,
    root_prefix: str,
    root: str,
    max_depth: int,
    max_episodes: Optional[int],
) -> List[Tuple[str, str]]:
    """Breadth-first walk of one source, stopping early once ``max_episodes`` are found."""
    episodes: List[Tuple[str, str]] = []
    frontier = [(root, 0)]
    while frontier:
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
        prefix, depth = frontier.pop(0)
        if prefix.rstrip("/").endswith(".zarr"):
            source = prefix[len(root_prefix) :].rstrip("/").rsplit("/", 1)[0]
            episodes.append((prefix, source or "root"))
            continue
        if depth >= max_depth:
            continue
        remaining = None if max_episodes is None else max_episodes - len(episodes)
        for child in list_prefixes(client, bucket, prefix, limit=remaining):
            frontier.append((child, depth + 1))
    return episodes


def discover_episodes(
    client,
    bucket: str,
    root_prefix: str,
    sources: Optional[Iterable[str]] = None,
    max_depth: int = 3,
    max_per_source: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Walk prefixes to find ``*.zarr/`` episode stores.

    Episodes live at varying depths -- ``processed_v3/eva/<ts>.zarr/`` but
    ``processed_v3/yam/<task>/<ts>.zarr/`` -- so this walks levels and stops
    descending as soon as a prefix is itself a store. That keeps the listing cost
    proportional to the number of episodes, instead of listing the millions of
    image chunk objects a recursive listing would return.

    Args:
        max_per_source: Stop discovering a source once this many episodes are found.
            Without it, discovery enumerates all ~23k episodes in the bucket before
            ``--limit`` is applied, which dominates runtime and is what makes an
            unscoped fetch appear to hang. ``None`` enumerates everything.

    Returns:
        ``(episode_prefix, source_label)`` pairs.
    """
    if sources:
        roots = [f"{root_prefix}{s.strip('/')}/" for s in sources]
    else:
        roots = list_prefixes(client, bucket, root_prefix)

    episodes: List[Tuple[str, str]] = []
    for root in roots:
        episodes.extend(
            _walk_source(client, bucket, root_prefix, root, max_depth, max_per_source)
        )
    return episodes


def _balanced_take(
    episodes: List[Tuple[str, str]], limit: Optional[int]
) -> List[Tuple[str, str]]:
    """Round-robin across sources so one huge source cannot crowd out the rest.

    A dataset-diversity tool that silently sampled 200 episodes from a single source
    would report a misleadingly low diversity score, so balance is the default.
    """
    if limit is None or limit >= len(episodes):
        return episodes
    by_source: Dict[str, List[Tuple[str, str]]] = {}
    for ep, source in episodes:
        by_source.setdefault(source, []).append((ep, source))
    for group in by_source.values():
        group.sort()

    picked: List[Tuple[str, str]] = []
    order = sorted(by_source)
    idx = 0
    while len(picked) < limit:
        progressed = False
        for source in order:
            group = by_source[source]
            if idx < len(group):
                picked.append(group[idx])
                progressed = True
                if len(picked) == limit:
                    return picked
        if not progressed:
            break
        idx += 1
    return picked


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def local_name(episode_prefix: str, source: str, root_prefix: str) -> str:
    """Flatten an episode prefix into a single local ``*.zarr`` directory name."""
    rel = episode_prefix[len(root_prefix) :].rstrip("/")
    return rel.replace("/", "__")


def fetch_episode(
    client,
    bucket: str,
    episode_prefix: str,
    dest_dir: Path,
    keys: Iterable[str],
) -> Tuple[str, int, int, Optional[str]]:
    """Download ``zarr.json`` plus the requested key subtrees for one episode.

    Returns:
        ``(episode_prefix, n_objects, n_bytes, error)``.
    """
    n_objects = 0
    n_bytes = 0
    try:
        targets = [f"{episode_prefix}zarr.json"] + [
            f"{episode_prefix}{key}/" for key in keys
        ]
        paginator = client.get_paginator("list_objects_v2")
        to_download: List[Tuple[str, int]] = []
        for target in targets:
            if target.endswith("/"):
                for page in paginator.paginate(Bucket=bucket, Prefix=target):
                    for obj in page.get("Contents", []):
                        to_download.append((obj["Key"], obj["Size"]))
            else:
                to_download.append((target, 0))

        if not any(k.endswith("zarr.json") for k, _ in to_download):
            return episode_prefix, 0, 0, "no zarr.json"
        if len(to_download) <= 1:
            return episode_prefix, 0, 0, "no pose arrays present"

        for key, size in to_download:
            rel = key[len(episode_prefix) :]
            out_path = dest_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(out_path))
            n_objects += 1
            n_bytes += size or out_path.stat().st_size
        return episode_prefix, n_objects, n_bytes, None
    except Exception as exc:  # noqa: BLE001 - report and continue with other episodes
        return episode_prefix, n_objects, n_bytes, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="data", help="local output directory")
    parser.add_argument("--limit", type=int, default=200, help="max episodes to fetch")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="restrict to these processed_v3 sources (default: all, balanced)",
    )
    parser.add_argument("--prefix", default=DEFAULT_ROOT_PREFIX, help="bucket root prefix")
    parser.add_argument("--workers", type=int, default=16, help="parallel download workers")
    parser.add_argument(
        "--include-extras",
        action="store_true",
        help="also fetch gripper/head/cmd pose arrays",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="stop discovering a source after this many episodes (default: 4x --limit). "
        "Bounds listing cost; pass 0 to enumerate the whole bucket.",
    )
    parser.add_argument(
        "--list-sources", action="store_true", help="print available sources and exit"
    )
    parser.add_argument(
        "--clean", action="store_true", help="delete the output directory first"
    )
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    args = parser.parse_args()

    env = load_egoverse_env(args.env_file)
    bucket = env.get("BUCKET", "rldb")
    client = make_r2_client(env)

    if args.list_sources:
        print(f"Sources under s3://{bucket}/{args.prefix}:")
        for prefix in list_prefixes(client, bucket, args.prefix):
            print(f"  {prefix[len(args.prefix):].rstrip('/')}")
        return

    # Discover a healthy multiple of what we need, so the round-robin balancing still has
    # room to choose, without paging every episode in the bucket.
    if args.max_per_source is None:
        max_per_source = max(4 * args.limit, 50) if args.limit else None
    else:
        max_per_source = args.max_per_source or None

    print(f"Discovering episodes under s3://{bucket}/{args.prefix} ...")
    episodes = discover_episodes(
        client, bucket, args.prefix, args.sources, max_per_source=max_per_source
    )
    if not episodes:
        raise SystemExit("No .zarr episodes discovered. Check --sources/--prefix.")

    counts: Dict[str, int] = {}
    for _, source in episodes:
        counts[source] = counts.get(source, 0) + 1
    print(f"Found {len(episodes)} episodes across {len(counts)} sources.")

    selected = _balanced_take(episodes, args.limit)
    if len(selected) < len(episodes):
        print(f"Selecting {len(selected)} episodes, balanced round-robin across sources.")

    out_dir = Path(args.out)
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = list(POSE_KEYS) + (list(EXTRA_KEYS) if args.include_extras else [])

    ok = 0
    failed: List[Tuple[str, str]] = []
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for episode_prefix, source in selected:
            dest = out_dir / f"{local_name(episode_prefix, source, args.prefix)}.zarr"
            futures[
                pool.submit(fetch_episode, client, bucket, episode_prefix, dest, keys)
            ] = episode_prefix

        for i, future in enumerate(as_completed(futures), start=1):
            episode_prefix, n_objects, n_bytes, error = future.result()
            if error:
                failed.append((episode_prefix, error))
            else:
                ok += 1
                total_bytes += n_bytes
            if i % 25 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} processed ({ok} ok, {len(failed)} failed)")

    print(f"\nFetched {ok} episodes into '{out_dir}' ({total_bytes / 1e6:.1f} MB).")
    if failed:
        print(f"{len(failed)} episodes failed or had no pose arrays:")
        for episode_prefix, error in failed[:10]:
            print(f"  {episode_prefix}: {error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
