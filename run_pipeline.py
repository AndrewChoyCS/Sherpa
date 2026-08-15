#!/usr/bin/env python
"""Headless CLI for the EgoVerse diversity and curriculum pipeline.

Runs the same code path as the dashboard and writes durable artifacts, so results can
be produced in CI or on a training box with no browser.

Outputs written to ``--out``:
    curriculum.csv        per-episode difficulty, stage, curriculum and coreset order
    stages.csv            per-stage rollup
    metrics.json          diversity metrics, cluster-vs-label agreement, run config
    dtw_matrix.npy        the (N, N) DTW distance matrix
    embedding.npy         the (N, 3) UMAP embedding
    episodes.csv          tidy per-episode table (embedding + cluster + metadata)
    diversity_map.html    standalone interactive 3-D scatter

Examples:
    python run_pipeline.py
    python run_pipeline.py --data-dir data --clusters 6 --out reports
    python run_pipeline.py --normalize zscore --max-length 300 --arm both
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from src.pipeline import run_pipeline


def _write_html_map(result, path: Path) -> None:
    """Write a standalone interactive 3-D diversity map."""
    import plotly.express as px

    df = result.frame()
    fig = px.scatter_3d(
        df,
        x="UMAP_X",
        y="UMAP_Y",
        z="UMAP_Z",
        color="cluster_label",
        symbol="source" if df["source"].nunique() > 1 else None,
        hover_data=["episode_id", "task_name", "embodiment", "n_frames", "difficulty", "stage"],
        title="EgoVerse Trajectory Diversity Map (DTW + UMAP)",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.85))
    fig.update_layout(height=760, legend_title_text="Curriculum group")
    fig.write_html(str(path), include_plotlyjs="cdn")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", default="data", help="directory of .zarr episodes")
    parser.add_argument("--out", default="reports", help="output directory for artifacts")
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="curriculum groups (default: choose k by silhouette score)",
    )
    parser.add_argument("--arm", default="auto", choices=("auto", "left", "right", "both"))
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument(
        "--normalize", default="zscore", choices=("center", "zscore", "none"),
        help="shape-only (zscore) by default; `center` keeps motion extent and "
        "collapses on multi-embodiment data — see README",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="resample cap; 0 disables resampling (quadratically slower)",
    )
    parser.add_argument("--no-length-normalize", action="store_true")
    parser.add_argument(
        "--sakoe-chiba-radius", type=int, default=None, help="DTW warping band radius"
    )
    parser.add_argument("--linkage", default="average", choices=("average", "complete", "single"))
    parser.add_argument("--difficulty-scaling", default="rank", choices=("rank", "minmax"))
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-cache", action="store_true", help="ignore the DTW disk cache")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)

    result = run_pipeline(
        data_dir=args.data_dir,
        n_clusters=args.clusters,
        arm=args.arm,
        min_length=args.min_length,
        normalize=args.normalize,
        max_length=args.max_length or None,
        length_normalize=not args.no_length_normalize,
        sakoe_chiba_radius=args.sakoe_chiba_radius,
        difficulty_scaling=args.difficulty_scaling,
        linkage=args.linkage,
        n_jobs=args.n_jobs,
        cache_dir=None if args.no_cache else ".cache",
        verbose=not args.quiet,
    )

    if result.n_episodes < 2:
        print(
            f"\nOnly {result.n_episodes} usable episode(s) in '{args.data_dir}'.\n"
            "Fetch data first:\n"
            "  python scripts/fetch_egoverse_data.py --sources yam scale aria --limit 40",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result.curriculum.to_csv(out_dir / "curriculum.csv", index=False)
    result.stages.to_csv(out_dir / "stages.csv", index=False)
    result.frame().to_csv(out_dir / "episodes.csv", index=False)
    np.save(out_dir / "dtw_matrix.npy", result.distance_matrix)
    np.save(out_dir / "embedding.npy", result.embedding)

    metrics = {
        "diversity_metrics": result.report,
        "cluster_label_agreement_ari": result.agreement,
        "cluster_label_agreement_support": result.agreement_support,
        "n_clusters": int(len(set(result.labels.tolist()))),
        "suggested_k": result.suggested_k,
        "silhouette_by_k": {str(k): v for k, v in result.silhouette_by_k.items()},
        "n_episodes_loaded": result.n_episodes,
        "n_episodes_skipped": len(result.dataset.skipped),
        "skipped": [{"episode_id": e, "reason": r} for e, r in result.dataset.skipped],
        "config": {
            "data_dir": args.data_dir,
            "arm": args.arm,
            "min_length": args.min_length,
            "normalize": args.normalize,
            "max_length": args.max_length or None,
            "length_normalize": not args.no_length_normalize,
            "sakoe_chiba_radius": args.sakoe_chiba_radius,
            "linkage": args.linkage,
            "difficulty_scaling": args.difficulty_scaling,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_html_map(result, out_dir / "diversity_map.html")

    # ---- console report ----
    report = result.report
    print("\n" + "=" * 72)
    print("EgoVerse Diversity & Curriculum Report")
    print("=" * 72)
    print(f"Episodes analysed      : {result.n_episodes} ({len(result.dataset.skipped)} skipped)")
    print(f"Curriculum groups      : {len(set(result.labels.tolist()))} (silhouette-suggested k={result.suggested_k})")
    print(f"Diversity score        : {report['diversity_score']:.5f}  (mean pairwise DTW)")
    print(f"Median pairwise        : {report['median_pairwise']:.5f}")
    print(f"Dispersion (CV)        : {report['dispersion']:.3f}")
    print(f"Mean NN distance       : {report['mean_nn_distance']:.5f}  (higher = less redundant)")
    print(f"Redundancy ratio       : {report['redundancy_ratio']:.1%}  of episodes near-duplicate")
    if "silhouette" in report:
        print(f"Cluster silhouette     : {report['silhouette']:.3f}")
    if "cluster_balance" in report:
        print(f"Curriculum balance     : {report['cluster_balance']:.3f}  (1.0 = even stages)")

    if result.agreement:
        print("\nCluster agreement with metadata (Adjusted Rand Index):")
        for name, score in sorted(result.agreement.items(), key=lambda kv: -kv[1]):
            support = result.agreement_support.get(name, result.n_episodes)
            print(f"  {name:<14} {score:>6.3f}   (over {support} labelled episodes)")
        unlabelled = {
            name: result.n_episodes - count
            for name, count in result.agreement_support.items()
            if count < result.n_episodes
        }
        if unlabelled:
            print("  note: episodes without a label are excluded from the score above:")
            for name, count in sorted(unlabelled.items()):
                print(f"    {name}: {count} unlabelled")

    if not result.stages.empty:
        print("\nCurriculum stages (easiest first):")
        for row in result.stages.itertuples():
            print(
                f"  stage {row.stage}: {row.n_episodes:>3} episodes  "
                f"difficulty {row.min_difficulty:.2f}-{row.max_difficulty:.2f}  "
                f"mean path {row.mean_path_length:.2f} m"
            )

    if result.redundant_pairs:
        print(f"\nClosest near-duplicate pairs ({len(result.redundant_pairs)} below the 5th percentile):")
        ids = result.dataset.episode_ids
        for i, j, dist in result.redundant_pairs[:5]:
            print(f"  {dist:.5f}  {ids[i]}  <->  {ids[j]}")

    # ---- subset ranking: does the score choose, not just describe? ----
    if result.n_episodes >= 8:
        from src.compare import compare_subsets

        comparison = compare_subsets(
            result.distance_matrix,
            methods=("coreset", "random", "stratified", "redundant"),
            subset_size=max(2, result.n_episodes // 4),
            labels=result.labels,
            tasks=result.dataset.task_labels,
            sources=result.dataset.field_values("source"),
        )
        table = comparison.table()
        table.to_csv(out_dir / "subset_comparison.csv", index=False)
        comparison.deltas().to_csv(out_dir / "subset_deltas.csv", index=False)
        for subset in comparison.subsets:
            metrics["subset_comparison"] = metrics.get("subset_comparison", {})
            metrics["subset_comparison"][subset.name] = subset.metrics
        metrics["subset_comparison_baseline"] = comparison.baseline or {}
        metrics["subset_size"] = comparison.subset_size
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        print(f"\nSubset ranking (n={comparison.subset_size} each, equal sizes):")
        print(
            f"  {'strategy':<12}{'diversity':>11}{'nn dist':>10}"
            f"{'redundant':>11}{'tasks':>8}"
        )
        for subset in comparison.subsets:
            m = subset.metrics
            print(
                f"  {subset.name:<12}{m['diversity_score']:>11.4f}"
                f"{m['mean_nn_distance']:>10.3f}{m['redundancy_ratio']:>10.0%}"
                f"{int(m.get('n_tasks_covered', 0)):>8}"
            )
        if comparison.baseline:
            base = comparison.baseline
            print(
                f"  -> {base['candidate_name']} sits at the {base['percentile']:.0f}th "
                f"percentile of {int(base['trials'])} random draws "
                f"({base['z_score']:.1f} sigma above the random mean)."
            )

    print(f"\nArtifacts written to '{out_dir}/'")
    print(f"  open {out_dir / 'diversity_map.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
