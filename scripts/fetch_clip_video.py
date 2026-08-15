#!/usr/bin/env python
"""Download the front-camera stream for specific episodes and write watchable video.

The main fetcher (:mod:`scripts.fetch_egoverse_data`) deliberately never touches the
camera keys -- pose is ~5 KB an episode against ~50 MB for one camera, which is what
makes a 273-episode dataset tractable. This script is the opt-in escape hatch for the
handful of clips in a curriculum you actually want to watch.

**Why a decode step is needed.** ``images.front_1`` is not a file you can copy and
play. It is a 1-D Zarr array of ``variable_length_bytes``, one encoded frame per
entry, zstd-compressed (and shard-indexed on ``yam``). So the frames have to be read
through Zarr, written out, and muxed into a container.

Usage:
    # one curriculum's clips, in training order, as mp4
    python scripts/fetch_clip_video.py --episodes-from /tmp/ex_garments/path.csv

    # or name them explicitly
    python scripts/fetch_clip_video.py yam__fold_clothes__2026-07-08-22-09-05-861000

    # keep the JPEGs instead of muxing
    python scripts/fetch_clip_video.py <episode_id> --format frames

Outputs, per episode, under ``--out`` (default ``video/``):
    <episode_id>.mp4          if ffmpeg is available and --format mp4 (the default)
    <episode_id>/frame_*.jpg  if --format frames, or if ffmpeg is missing

Credentials come from ``~/.egoverse_env``, same as the pose fetcher.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fetch_egoverse_data import (  # noqa: E402
    DEFAULT_ROOT_PREFIX,
    load_egoverse_env,
    make_r2_client,
)

CAMERA_KEYS = ("images.front_1", "images.left_wrist", "images.right_wrist")


def prefix_for(episode_id: str, root: str = DEFAULT_ROOT_PREFIX) -> str:
    """Map an episode id back to its R2 prefix.

    Ids are ``source__task__timestamp`` where the source groups by task (``yam``), and
    ``source__timestamp`` where it does not (``scale``, ``aria``, ``mecka``). That is
    the same split the loader uses when it derives the id from the store name.
    """
    parts = episode_id.split("__")
    if len(parts) == 3:
        source, task, stamp = parts
        return f"{root}{source}/{task}/{stamp}.zarr/"
    if len(parts) == 2:
        source, stamp = parts
        return f"{root}{source}/{stamp}.zarr/"
    raise ValueError(f"cannot parse episode id {episode_id!r}")


def episodes_from_csv(path: Path) -> List[str]:
    """Episode ids in training order from a ``path.csv`` written by find_path.py.

    Duplicates are dropped while keeping first-appearance order: a rehearsal step
    repeats a clip already in the list, and downloading it twice is wasted bandwidth.
    """
    import csv

    seen: List[str] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            episode = row.get("episode_id")
            if episode and episode not in seen:
                seen.append(episode)
    return seen


def download_camera(client, bucket: str, episode_id: str, key: str, dest: Path) -> int:
    """Mirror one camera array's objects to ``dest``. Returns bytes fetched."""
    prefix = f"{prefix_for(episode_id)}{key}/"
    paginator = client.get_paginator("list_objects_v2")
    total = 0
    found = False
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            found = True
            relative = obj["Key"][len(prefix) :]
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, obj["Key"], str(target))
            total += obj["Size"]
    if not found:
        raise FileNotFoundError(f"{episode_id}: no objects under {prefix}")
    return total


def _frame_bytes(value) -> Optional[bytes]:
    """Unwrap one ``variable_length_bytes`` element to raw bytes.

    Zarr hands these back double-wrapped: a 0-d ``object`` array whose contents are
    themselves a 0-d fixed-width bytes array (``|S43830``). One ``.tolist()`` peels
    each layer, so this loops rather than calling ``.item()`` once -- which returns
    the inner array, not the bytes, and silently yields zero frames.
    """
    import numpy as np

    for _ in range(4):
        if isinstance(value, (bytes, bytearray, np.bytes_)):
            return bytes(value)
        if isinstance(value, np.ndarray):
            value = value.tolist() if value.shape == () else value.reshape(-1)[0]
            continue
        return None
    return None


def decode_frames(array_dir: Path) -> List[bytes]:
    """Read the encoded frames out of a downloaded ``images.*`` Zarr array.

    The whole array is read in one go: per-element reads re-open the shard index on
    every access, which is minutes rather than seconds over a 1,300-frame clip.
    """
    import zarr

    array = zarr.open_array(str(array_dir), mode="r")
    raw = array[:]
    frames: List[bytes] = []
    for value in raw:
        blob = _frame_bytes(value)
        if blob:
            frames.append(blob)
    return frames


def write_mp4(
    frames: List[bytes], out_path: Path, fps: float, width: Optional[int] = None, crf: int = 28
) -> bool:
    """Mux encoded frames into an mp4. Returns False if ffmpeg is unavailable.

    Defaults are tuned for a web demo rather than for archival: ``crf 28`` and an
    optional downscale turn ~60 MB of source JPEGs into a couple of MB, which is the
    difference between a page that plays instantly and one that stalls. ``faststart``
    moves the index to the front so a browser can begin playing before the whole file
    has arrived.
    """
    if shutil.which("ffmpeg") is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, blob in enumerate(frames):
            (tmp_dir / f"frame_{i:06d}.jpg").write_bytes(blob)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", f"{fps:g}",
            "-i", str(tmp_dir / "frame_%06d.jpg"),
            "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        ]
        if width:
            # -2 keeps the aspect ratio and forces an even height, which yuv420p needs.
            command += ["-vf", f"scale={width}:-2"]
        command.append(str(out_path))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    ffmpeg failed: {result.stderr.strip()[:300]}", file=sys.stderr)
            return False
    return True


def write_frames(frames: List[bytes], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, blob in enumerate(frames):
        (out_dir / f"frame_{i:06d}.jpg").write_bytes(blob)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episodes", nargs="*", help="episode ids to fetch")
    parser.add_argument(
        "--episodes-from",
        type=Path,
        help="a path.csv from find_path.py; fetches its clips in training order",
    )
    parser.add_argument("--camera", default="images.front_1", choices=CAMERA_KEYS)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "video")
    parser.add_argument("--fps", type=float, default=30.0, help="playback rate for the mp4")
    parser.add_argument("--width", type=int, default=640, help="downscale to this width; 0 keeps source size")
    parser.add_argument("--crf", type=int, default=28, help="x264 quality; lower is better and bigger")
    parser.add_argument("--format", default="mp4", choices=("mp4", "frames"))
    parser.add_argument(
        "--keep-zarr", action="store_true", help="keep the downloaded array, not just the video"
    )
    parser.add_argument("--dry-run", action="store_true", help="list sizes and exit, download nothing")
    args = parser.parse_args()

    episodes: List[str] = list(args.episodes)
    if args.episodes_from:
        episodes = episodes_from_csv(args.episodes_from) + [
            e for e in episodes if e not in episodes_from_csv(args.episodes_from)
        ]
    if not episodes:
        parser.error("give episode ids, or --episodes-from a path.csv")

    env = load_egoverse_env()
    bucket = env.get("BUCKET")
    if not bucket:
        raise SystemExit("BUCKET missing from ~/.egoverse_env")
    client = make_r2_client(env)

    # A size check first: one camera runs 20-70 MB an episode, and it is worth
    # seeing the total before committing to the download.
    paginator = client.get_paginator("list_objects_v2")
    plan = []
    for episode in episodes:
        prefix = f"{prefix_for(episode)}{args.camera}/"
        size = sum(
            obj["Size"]
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
            for obj in page.get("Contents", [])
        )
        plan.append((episode, size))
        print(f"  {episode:52} {size / 1e6:8.2f} MB")
    total = sum(size for _, size in plan)
    print(f"  {'TOTAL':52} {total / 1e6:8.2f} MB  ({len(plan)} clips, {args.camera})")

    if args.dry_run:
        return 0
    if any(size == 0 for _, size in plan):
        print("\nSome clips have no objects under that camera key; aborting.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for position, (episode, _) in enumerate(plan, start=1):
        print(f"\n[{position}/{len(plan)}] {episode}")
        work = args.out / f"_{episode}" / args.camera
        download_camera(client, bucket, episode, args.camera, work)
        frames = decode_frames(work)
        print(f"    {len(frames)} frames decoded")

        if args.format == "mp4" and write_mp4(
            frames, args.out / f"{episode}.mp4", args.fps, args.width or None, args.crf
        ):
            print(f"    wrote {args.out / f'{episode}.mp4'}")
        else:
            write_frames(frames, args.out / episode)
            print(f"    wrote {len(frames)} jpgs to {args.out / episode}/")

        if not args.keep_zarr:
            shutil.rmtree(args.out / f"_{episode}", ignore_errors=True)

    write_manifest(args.out)
    print(f"\nDone. Output in {args.out}/")
    return 0


def write_manifest(out_dir: Path) -> Path:
    """List the clips present, so the frontend knows which have video.

    Written by scanning the directory rather than from this run's episode list, so a
    manifest stays correct when clips are fetched across several invocations. The
    frontend treats a missing entry as "no video" and falls back to the plotted path,
    which is why this must describe what is actually on disk.
    """
    import json

    clips = {}
    for mp4 in sorted(out_dir.glob("*.mp4")):
        clips[mp4.stem] = {"src": f"{out_dir.name}/{mp4.name}", "bytes": mp4.stat().st_size}
    manifest = out_dir / "index.json"
    manifest.write_text(json.dumps(clips, indent=2))
    print(f"  manifest: {len(clips)} clip(s) -> {manifest}")
    return manifest


if __name__ == "__main__":
    raise SystemExit(main())
