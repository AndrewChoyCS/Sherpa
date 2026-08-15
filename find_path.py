#!/usr/bin/env python
"""Headless CLI for the EgoVerse curriculum path finder (Track 1).

Takes a training goal in plain English, resolves it to a target clip, and searches the
clip graph for an ordered curriculum that reaches it -- ramping difficulty smoothly and
avoiding abrupt task/skill/embodiment switches along the way.

Runs the same code path as the dashboard's Path Finder tab, so a result produced in CI
matches what the browser shows.

Outputs written to ``--out``:
    path.csv            the ordered curriculum, one row per training step
    path_metrics.json   proxy metrics, baseline comparison, goal match, run config
    path_graph.html     standalone interactive graph with the path highlighted

Examples:
    # the featured demo: a garment-folding curriculum that crosses embodiments
    python find_path.py "teach the robot to fold a shirt" --domain garments

    python find_path.py "pack the items into the box" --domain containers
    python find_path.py "pick up the hat" --scope-to-match --review-every 3
    python find_path.py "sort the utensils" --w-interference 0.5 --out reports
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.goal_matcher import GoalMatch
from src.graph import GraphConfig
from src.graph_plot import path_graph_figure
from src.path_metrics import (
    NON_CIRCULAR_METRICS,
    ablation_payload,
    compare_orderings,
    difficulty_ablation,
    paired_verdict,
    path_report,
    scope_descriptor,
    sweep_orderings,
)
from src.pathfinder import PathConfig
from src.pipeline import build_path_finder, run_pipeline

# Metrics worth printing to the console, in reading order, with a friendlier label and
# whether higher is better.
HEADLINE_METRICS = (
    ("spearman", "Difficulty monotonicity (rho)", True),
    ("frac_nondecreasing", "Non-decreasing steps", True),
    ("max_jump", "Largest difficulty jump", False),
    ("mean_abs_step", "Mean |difficulty step|", False),
    ("task_switch_rate", "Task switch rate", False),
    ("cluster_switch_rate", "Skill-family switch rate", False),
    ("mean_consecutive_dtw", "Mean consecutive DTW", False),
    ("cluster_coverage", "Skill coverage to target", True),
    ("task_coverage", "Task coverage to target", True),
    ("frac_consecutive_near_duplicate", "Consecutive near-duplicates", False),
)


# Curated task groupings, so the intended scoped demo is one flag rather than five task
# names. The pitch calls for a graph "scoped to one or a few related task domains", and
# scoping matters for more than tidiness: on the full 273-clip graph the near-duplicate
# threshold is the 5th percentile of a distribution dominated by cross-embodiment pairs,
# so every within-task transition reads as a duplicate and the redundancy metric saturates
# at 1.0. Scoped to one domain, that threshold becomes meaningful again.
DOMAIN_PRESETS = {
    "garments": (
        "fold_clothes",
        "yam_fold_tshirt",
        "freeform_sort_laundry_by_type_and_color",
        "freeform_hang_shirt_on_hanger_and_place_on_rack",
        "flagship_folding_clothes",
    ),
    "containers": (
        "yam_pack_items",
        "flagship_bagging_groceries",
        "freeform_placing_utensils_into_a_drawer",
        "freeform_place_rubberbands_in_ziploc",
        "object_in_container",
    ),
}


def _match_payload(match: GoalMatch) -> dict:
    return {
        "query": match.query,
        "matched_task": match.task_name,
        "target_index": match.target_index,
        "score": match.score,
        "margin": match.margin,
        "is_confident": match.is_confident,
        "note": match.note,
        "alternates": [
            {
                "task_name": c.task_name,
                "score": c.score,
                "clip_index": c.clip_index,
                "episode_id": c.episode_id,
                "n_clips": c.n_clips,
            }
            for c in match.candidates
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("goal", help="training goal in plain English")
    parser.add_argument("--data-dir", default="data", help="directory of .zarr episodes")
    parser.add_argument("--out", default="reports", help="output directory for artifacts")

    ingest = parser.add_argument_group("ingestion (shared with run_pipeline.py)")
    ingest.add_argument("--arm", default="auto", choices=("auto", "left", "right", "both"))
    ingest.add_argument("--min-length", type=int, default=30)
    ingest.add_argument(
        "--normalize", default="zscore", choices=("center", "zscore", "none"),
        help="shape-only (zscore) by default; `center` keeps motion extent and "
        "collapses on multi-embodiment data — see README",
    )
    ingest.add_argument("--max-length", type=int, default=200)
    ingest.add_argument("--clusters", type=int, default=None)
    ingest.add_argument("--no-cache", action="store_true", help="ignore the DTW disk cache")

    graph = parser.add_argument_group("graph weighting")
    graph.add_argument("--k-neighbors", type=int, default=10)
    graph.add_argument("--w-difficulty", type=float, default=1.0,
                       help="weight on the difficulty-ramp term")
    graph.add_argument("--w-interference", type=float, default=1.0,
                       help="weight on the interference term; lower broadens coverage")
    graph.add_argument("--target-step", type=float, default=0.05,
                       help="difficulty increment the ramp aims for per step")
    graph.add_argument("--step-penalty", type=float, default=0.1)
    graph.add_argument(
        "--scope-to-match", action="store_true",
        help="restrict the graph to the matched task family and route only within it",
    )
    graph.add_argument(
        "--tasks", nargs="*", default=None,
        help="restrict the graph to these task names (overrides --domain/--scope-to-match)",
    )
    graph.add_argument(
        "--domain", choices=sorted(DOMAIN_PRESETS), default=None,
        help="restrict the graph to a curated group of related tasks",
    )

    search = parser.add_argument_group("search and rehearsal")
    search.add_argument("--search", default="dijkstra", choices=("dijkstra", "astar"))
    search.add_argument("--review-every", type=int, default=4,
                        help="insert a rehearsal clip every N clips; 0 disables")
    search.add_argument("--max-reviews", type=int, default=12)
    search.add_argument("--target-selection", default="hardest",
                        choices=("hardest", "medoid", "easiest"))
    validate = parser.add_argument_group("multi-goal validation")
    validate.add_argument(
        "--sweep",
        type=int,
        default=0,
        metavar="N",
        help="re-run the baseline comparison across N sampled goals and write "
        "ordering_sweep.csv, difficulty_ablation.csv and a paired significance test. "
        "Single-goal metrics vary enormously with the target, so any claim about the "
        "search needs the distribution. 40 is a reasonable N.",
    )
    search.add_argument("--seeds", type=int, default=50,
                        help="random draws averaged per stochastic baseline")
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
        cache_dir=None if args.no_cache else ".cache",
        verbose=not args.quiet,
    )
    if result.n_episodes < 2:
        print(
            f"\nOnly {result.n_episodes} usable episode(s) in '{args.data_dir}'.\n"
            "Fetch data first:\n"
            "  python scripts/fetch_egoverse_data.py --limit 300",
            file=sys.stderr,
        )
        return 1

    graph_config = GraphConfig(
        k_neighbors=args.k_neighbors,
        w_difficulty=args.w_difficulty,
        w_interference=args.w_interference,
        target_step=args.target_step,
        step_penalty=args.step_penalty,
    )
    path_config = PathConfig(
        review_every=args.review_every,
        search=args.search,
        max_reviews=args.max_reviews,
    )

    # Scoping to the matched family needs the match, which needs a matcher, which needs a
    # graph. So match against the unscoped set first, then rebuild scoped if asked.
    context = build_path_finder(result, graph_config)
    match = context.matcher.match(args.goal, target_selection=args.target_selection)

    tasks = args.tasks
    if tasks is None and args.domain:
        # Keep only preset tasks that this dataset actually contains, so a preset never
        # fails outright just because one of its tasks was not sampled.
        present = set(context.clip_graph.nodes["task_name"].astype(str))
        tasks = [t for t in DOMAIN_PRESETS[args.domain] if t in present]
        if not tasks:
            print(f"none of the --domain {args.domain} tasks are present; "
                  "routing over the full graph", file=sys.stderr)
            tasks = None
    if tasks is None and args.scope_to_match:
        tasks = [match.task_name]
    if tasks:
        context = build_path_finder(result, graph_config, task_names=tasks)
        match = context.matcher.match(args.goal, target_selection=args.target_selection)

    path = context.find(args.goal, path_config, args.target_selection, match.target_index)[1]

    clip_graph = context.clip_graph
    report = path_report(
        path.clips, clip_graph, context.distance_matrix, path.target_index, path.is_review
    )
    comparison = compare_orderings(
        path, clip_graph, context.distance_matrix, n_seeds=args.seeds
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path.table.to_csv(out_dir / "path.csv", index=False)

    figure = path_graph_figure(
        clip_graph, context.layout, path,
        title=f"Curriculum path — “{args.goal}” → {match.task_name}",
    )
    figure.write_html(str(out_dir / "path_graph.html"), include_plotlyjs="cdn")

    payload = {
        "goal_match": _match_payload(match),
        "path_metrics": report,
        "ordering_comparison": json.loads(comparison.to_json(orient="index")),
        "path": {
            "n_steps": path.n_steps,
            "n_unique_clips": len(path.unique_clips),
            "n_reviews": path.n_reviews,
            "search_cost": path.search_cost,
            "cost_terms": path.cost_terms,
            "method": path.method,
            "episode_ids": [clip_graph.episode_id(c) for c in path.clips],
        },
        "graph": {
            "n_clips": clip_graph.n_clips,
            "n_edges": clip_graph.graph.number_of_edges() - len(clip_graph.start_clips),
            "start_pool": len(clip_graph.start_clips),
            "reachability_repairs": len(clip_graph.repairs),
            "task_scope": context.task_names,
        },
        "config": {
            "data_dir": args.data_dir,
            "graph": graph_config.as_dict(),
            "search": {
                "method": args.search,
                "review_every": args.review_every,
                "max_reviews": args.max_reviews,
                "target_selection": args.target_selection,
                "baseline_seeds": args.seeds,
            },
        },
    }
    # ---- multi-goal validation ----
    # One path is an anecdote: proxy metrics swing hard with which target was chosen.
    # This re-runs the whole comparison across many goals and pairs the results, which
    # is the only basis on which the search itself can be credited or discredited.
    sweep_summary = None
    if args.sweep:
        print(f"\nSweeping {args.sweep} goals for the paired comparison...")
        sweep = sweep_orderings(
            clip_graph,
            context.distance_matrix,
            path_config,
            n_targets=args.sweep,
            n_seeds=max(5, args.seeds // 5),
        )
        ablation = difficulty_ablation(
            clip_graph,
            context.distance_matrix,
            result.curriculum,
            path_config,
            n_targets=args.sweep,
        )
        # Stamp the population onto every artifact. `reports/` is shared mutable state,
        # and these results are scope-dependent rather than merely noisy, so a file that
        # does not say what it measured can be read as contradicting a correct claim.
        scope = scope_descriptor(clip_graph, context.task_names, getattr(args, "domain", None))
        for frame in (sweep, ablation):
            frame.insert(0, "scope", scope["scope"])
            frame.insert(1, "n_clips_in_scope", scope["n_clips"])
        sweep.to_csv(out_dir / "ordering_sweep.csv", index=False)
        ablation.to_csv(out_dir / "difficulty_ablation.csv", index=False)
        (out_dir / "difficulty_ablation.json").write_text(
            json.dumps(ablation_payload(ablation, scope, sweep), indent=2, default=str)
        )

        metrics_to_test = ["spearman", *NON_CIRCULAR_METRICS]
        sweep_summary = {
            **scope,
            "n_goals": int(sweep["target"].nunique()),
            "ordering_means": sweep.groupby("ordering")[metrics_to_test].mean().to_dict(),
            "path_vs_difficulty_sorted": {
                metric: paired_verdict(sweep, metric) for metric in metrics_to_test
            },
            "difficulty_ablation_means": (
                ablation.groupby("sort_key")[list(NON_CIRCULAR_METRICS)].mean().to_dict()
            ),
        }
        payload["multi_goal_sweep"] = sweep_summary

    (out_dir / "path_metrics.json").write_text(json.dumps(payload, indent=2, default=str))

    # ---- console report ----
    print("\n" + "=" * 78)
    print("EgoVerse Curriculum Path Finder")
    print("=" * 78)
    print(f"Goal                   : {args.goal!r}")
    print(f"Matched task           : {match.task_name}  "
          f"(score {match.score:.2f}, lead {match.margin:.0%})")
    if not match.is_confident:
        print(f"  ! {match.note}")
    print(f"  alternates           : "
          + ", ".join(f"{c.task_name} ({c.score:.2f})" for c in match.candidates[1:4]))
    print(f"Target clip            : {clip_graph.episode_id(path.target_index)}")
    print(f"Graph                  : {clip_graph.summary()}")
    print(f"Path                   : {len(path.route)} clips + {path.n_reviews} reviews, "
          f"cost {path.search_cost:.2f} "
          f"(ramp {path.cost_terms.get('ramp', 0):.1f}, "
          f"interference {path.cost_terms.get('interference', 0):.1f})")

    print("\nOrdered curriculum:")
    for row in path.table.itertuples():
        marker = "  ~review~ " if row.is_review else "          "
        print(f"  {row.step:>3}. {marker}d={row.difficulty:.3f}  "
              f"{str(row.task_name)[:26]:<26}  {row.episode_id}")

    print("\nProxy metrics vs. baselines (same size; see README for what each isolates):")
    available = [(k, label, hi) for k, label, hi in HEADLINE_METRICS if k in comparison.columns]
    width = max(len(label) for _, label, _ in available)
    header = " " * (width + 2) + "".join(f"{name[:22]:>24}" for name in comparison.index)
    print(header)
    for key, label, higher_better in available:
        arrow = "^" if higher_better else "v"
        cells = "".join(f"{comparison.loc[name, key]:>24.3f}" for name in comparison.index)
        print(f"  {label:<{width}}{cells}   ({arrow} better)")

    if sweep_summary:
        n_goals = sweep_summary["n_goals"]
        print(f"\nPaired across {n_goals} goals — path vs. difficulty-sorted:")
        print(f"  {'metric':<34}{'path':>9}{'sorted':>9}{'W/T/L':>12}{'p':>8}")
        for metric, verdict in sweep_summary["path_vs_difficulty_sorted"].items():
            wtl = f"{int(verdict['a_better'])}/{int(verdict['tied'])}/{int(verdict['b_better'])}"
            p_value = verdict["p_value"]
            p_text = "  tied" if p_value is None else f"{p_value:.3f}"
            print(
                f"  {metric:<34}{verdict['mean_a']:>9.3f}{verdict['mean_b']:>9.3f}"
                f"{wtl:>12}{p_text:>8}"
            )
        print(
            "  Read this before crediting the search: where W/T/L is dominated by ties\n"
            "  and p is far from significance, the path is reproducing a plain\n"
            "  difficulty ordering rather than improving on it."
        )

        print(f"\nDifficulty-metric ablation across {n_goals} goals")
        print("  (scored only on metrics independent of the difficulty definition):")
        means = sweep_summary["difficulty_ablation_means"]
        keys = list(next(iter(means.values())).keys())
        print("  " + f"{'sort key':<24}" + "".join(f"{m[:20]:>22}" for m in NON_CIRCULAR_METRICS))
        for key in keys:
            cells = "".join(f"{means[m][key]:>22.3f}" for m in NON_CIRCULAR_METRICS)
            print(f"  {key:<24}{cells}")

    print(f"\nArtifacts written to '{out_dir}/'")
    print(f"  open {out_dir / 'path_graph.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
