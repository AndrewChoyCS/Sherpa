# EgoVerse Curriculum Path Finder & Diversity Engine

Two submissions over one substrate.

**Track 1 — the Curation Engine.** Type a training goal in plain English. The app finds
and highlights an ordered path through a knowledge graph of clips: a curriculum that ramps
difficulty smoothly and avoids the abrupt task/skill switches that cause catastrophic
interference during training. Dumping all data in at once, or ordering it randomly, is
inefficient and can actively hurt the model; this picks *and sequences* a subset instead.

**Track 2 — non-text quantitative diversity scoring.** Measures how much genuinely
distinct manipulation behaviour a dataset contains, by comparing end-effector trajectories
under Dynamic Time Warping, then sequences the episodes into a difficulty-ordered
curriculum.

They share every expensive computation, which is the reason they live together: the **DTW
distance matrix is the interference signal** on the graph's edges, and the **kinematic
difficulty score is the ramp signal**. Track 1 adds milliseconds on top of Track 2, and
re-derives nothing.

On 273 real episodes across four sources, the unsupervised clustering recovers the
human-authored `task_name` groupings with an **Adjusted Rand Index of 0.653** over the 195
episodes that carry a label — evidence the distance metric tracks behaviour rather than
noise. (78 episodes, all of `mecka`, carry no `task_name` at all and are excluded rather
than scored against a placeholder; see *Known limitations*.)

**Live:** <https://andrewchoy--egoverse-curriculum-web.modal.run> — Sherpa, running the
real pipeline on the real episodes. See *Deploying* for how it is hosted.

---

## The 60-second demo

```bash
python find_path.py "teach the robot to fold a shirt" --domain garments
```

A garment-folding curriculum over 131 clips spanning **three embodiments**. It opens on
*human* demonstrations, crosses to the *robot*, and ramps to the goal:

| # | difficulty | task | embodiment |
|---|---|---|---|
| 1 | 0.000 | `freeform_hang_shirt_on_hanger…` | human (scale) |
| 2 | 0.018 | `freeform_sort_laundry_by_type…` | human (scale) |
| 3 | 0.062 | `yam_fold_tshirt` | robot (yam) |
| 4 | 0.125 | `yam_fold_tshirt` | robot |
| 5 | ↩ *review* 0.000 | `freeform_hang_shirt_on_hanger…` | human |
| 6–9 | 0.173 → 0.375 | `yam_fold_tshirt` | robot |
| 10 | ↩ *review* 0.000 | `freeform_hang_shirt_on_hanger…` | human |
| 11 | **0.441** | `yam_fold_tshirt` ← **goal** | robot |

Nobody labelled that ordering. It is the cheapest route through the graph, and the
`↩ review` rows are rehearsal insertions pulling back the earliest skill.

### Does it actually behave like a curriculum?

Measured against four baselines of **identical size**, so none of this is a length effect
(`reports/path_metrics.json`, ↑/↓ marks the better direction):

| Proxy metric | **Path** | Random order (same clips) | Random subset (same size) | Difficulty-sorted | Coreset prefix |
|---|---|---|---|---|---|
| Difficulty monotonicity ρ ↑ | **1.000** | 0.078 | 0.005 | 1.000 | 0.033 |
| Non-decreasing steps ↑ | **1.000** | 0.502 | 0.492 | 1.000 | 0.500 |
| Mean \|difficulty step\| ↓ | **0.055** | 0.190 | 0.372 | 0.055 | 0.757 |
| Max \|difficulty step\| ↓ | **0.088** | 0.381 | 0.772 | 0.088 | 0.963 |
| Task switch rate ↓ | **0.250** | 0.405 | 0.560 | 0.250 | 1.000 |
| Skill-family switch rate ↓ | **0.125** | 0.223 | 0.510 | 0.125 | 1.000 |
| Mean consecutive DTW ↓ | **0.428** | 0.600 | 0.822 | 0.428 | 0.980 |
| Skill coverage to target ↑ | **0.400** | 0.324 | 0.388 | 0.400 | 1.000 |
| Task coverage to target ↑ | **0.600** | 0.424 | 0.436 | 0.600 | 1.000 |
| Consecutive near-duplicates ↓ | 0.400 | **0.255** | 0.070 | 0.750 | 0.000 |

**The path wins 8 of 9 against a reshuffle of its own clips, and loses one.** Reading it
honestly:

- The two random baselines answer different questions and must not be conflated.
  *Random order (same clips)* reshuffles the path's own selection, isolating the value of
  the **ordering**. *Random subset (same size)* draws fresh clips, isolating **selection**
  too.
- **Difficulty-sorted is the baseline that matters.** It nails monotonicity by
  construction, so ramp smoothness is not where the path earns its keep — it ties there.
  The path wins on **redundancy: 0.400 vs 0.750**. Same ramp quality, far less wasted
  repetition.
- **Coreset prefix is the instructive contrast.** It maximises coverage (1.000) and has the
  worst possible ramp (ρ = 0.033) and switches on every step. That gap *is* the argument
  for curriculum ordering over pure coverage selection.
- **The path loses on consecutive near-duplicates (0.400 vs 0.255).** This is the real cost
  of smoothness and it is not tuned away: minimising interference pulls consecutive clips
  together, and the redundancy penalty offsets it without erasing it. Lower
  `--w-interference` to trade smoothness for breadth; the dashboard's Validation tab
  recomputes the whole table live.

---

## Quickstart

```bash
# 1. environment
conda create -n egoverse python=3.11 -y && conda activate egoverse
pip install -r requirements.txt

# 2. fetch real episodes from the EgoVerse R2 bucket
#    (requires ~/.egoverse_env, written by egomimic/utils/aws/setup_secret.sh)
python scripts/fetch_egoverse_data.py --sources yam scale aria mecka --limit 320

# 3a. Track 1 — find a curriculum for a goal
python find_path.py "teach the robot to fold a shirt" --domain garments

# 3b. Track 2 — diversity report, headless
python run_pipeline.py

# 3c. interactive dashboard (both tracks, 8 tabs)
streamlit run app.py

# 3d. or Sherpa, the web frontend — see "Sherpa" below
python scripts/export_snapshot.py
uvicorn server.api:app --port 8000 &
cd web && npm install && npm run dev      # http://localhost:5173
```

No credentials? Generate synthetic episodes in the identical on-disk schema:

```bash
python scripts/generate_synthetic_data.py --out data_synth --inject-edge-cases
python find_path.py "stir the pot" --data-dir data_synth
```

**Fetching is slow from a laptop and fast on Modal** — see *Running on Modal* below. An
unscoped `--limit 320` discovery walk took over 40 minutes locally and hit R2 read
timeouts; the same fetch on Modal took **211 seconds**.

---

## What it produces

**Track 1 — `find_path.py` / the Path Finder tab**

| Output | Meaning |
|---|---|
| **Ordered curriculum** | `path.csv` — one row per training step, with the per-transition cost split into its ramp / interference / redundancy terms, and rehearsal steps flagged. |
| **Goal match** | The task family the free-text goal resolved to, its score, its lead over the runner-up, and the ranked alternates. |
| **Proxy metrics** | Monotonicity, interference, coverage and redundancy, each against four same-size baselines. |
| **Interactive graph** | `path_graph.html` — the clip graph with the ordered path lit up on it. |

**Track 2 — `run_pipeline.py` / the diversity tabs**

| Output | Meaning |
|---|---|
| **Diversity score** | Mean pairwise DTW distance over the strict upper triangle. |
| **Redundancy ratio** | Share of episodes whose nearest neighbour lies in the closest 5% of all pairs, i.e. near-duplicate demonstrations. |
| **Mean NN distance** | Average distance to the nearest neighbour; a direct coverage measure. |
| **`tail_ratio`** | 99th percentile / median of the pairwise distances. **Above ~2 the matrix is outlier-dominated and clustering will collapse** — see *Method notes*. |
| **Curriculum rank** | Easy→hard training order, grouped into stages by motion family. |
| **Coreset rank** | Farthest-point traversal; truncate at any *K* for a near-maximally diverse *K*-episode subset. |
| **Cluster/label ARI** | Agreement between unsupervised clusters and `task_name`, with the labelled-episode count it was computed over. |
| **Subset ranking** | Equal-sized subsets picked by different strategies, scored head to head against a random null model. |

### Does the score actually rank two subsets?

A diversity score is only useful if it can *choose*. On the 273-episode dataset, four
equal-sized selections of 68 episodes each:

| strategy | diversity | NN distance | near-duplicates | tasks covered |
|---|---|---|---|---|
| `coreset` | **1.9749** | **1.709** | **0%** | **12 / 14** |
| `stratified` | 1.9729 | 1.469 | 38% | 7 / 14 |
| `random` | 1.8779 | 1.146 | 56% | 12 / 14 |
| `redundant` (control) | 1.3330 | 0.480 | 100% | 4 / 14 |

`coreset` sits at the **100th percentile of 200 random draws, 3.7σ above the random
mean**. Three things make this a comparison rather than a favourable anecdote:

- **Equal subset sizes.** Several of these metrics move with N — a larger subset has
  more chances to contain a close pair — so unequal sizes would measure the size gap
  rather than the strategy.
- **An adversarial control.** `redundant` deliberately selects near-duplicates. A metric
  that cannot rank it last is not measuring diversity.
- **A null model, not a single draw.** Random selection has real variance, and at small
  budgets one lucky draw genuinely beats the coreset outright — there is a test pinning
  exactly that case. Only the percentile within the whole distribution supports the claim.

Reproduce with `python run_pipeline.py` (prints the table, writes
`reports/subset_comparison.csv`), or the **Subset A/B** tab in the dashboard.

---

## Architecture

```
src/loader.py            Zarr v3 ingestion, cleaning, unit normalisation, dataset cache
src/diversity_engine.py  preprocessing + pairwise DTW distance matrix (disk-cached)
src/cluster_mapper.py    UMAP projection, precomputed-metric clustering, metrics
src/curriculum.py        kinematic difficulty scoring, stage + coreset sequencing
src/pipeline.py          orchestration for both tracks

  Track 1 ---------------------------------------------------------------------
src/graph.py             the clip graph: dual-weighted directed edges, START, layout
src/goal_matcher.py      free-text goal -> target clip, via TF-IDF over task language
src/pathfinder.py        Dijkstra/A* search + rehearsal ("review") insertion
src/path_metrics.py      the four proxy metrics, and the four baselines
src/graph_plot.py        the force-directed graph figure, path highlighted

app.py                   Streamlit dashboard (8 tabs: 2 for Track 1, 6 for Track 2)
find_path.py             Track 1 headless CLI
run_pipeline.py          Track 2 headless CLI
modal_app.py             Modal offload: parallel fetch + remote pipeline, on volumes
scripts/fetch_egoverse_data.py     R2 downloader (pose keys only, bounded discovery)
scripts/generate_synthetic_data.py synthetic data in the real schema
tests/test_pipeline.py             Track 2 tests
tests/test_pathfinder.py           Track 1 tests

  Web frontend ----------------------------------------------------------------
server/api.py            FastAPI JSON surface over src/ (no numeric logic of its own)
server/serialize.py      NumPy/pandas -> strict JSON; maps NaN to null
web/                     Sherpa: React + Vite frontend (Route page + Workbench)
scripts/export_snapshot.py          writes web/public/snapshot.json for first paint
tests/test_api.py                   API tests, on synthetic data
```

---

## How Track 1 works

**The graph.** One node per clip. Candidate edges are each clip's *k* DTW nearest
neighbours, unioned in both directions. An edge `u → v` exists only when
`difficulty[v] ≥ difficulty[u] − backslide_tolerance`, so **the ramp is structural rather
than merely discouraged by a cost** — no weighting mistake can produce a descending
curriculum. Every cost is non-negative, which is what makes Dijkstra exact here.

**The edge cost has three terms, and two of them are deliberately two-sided:**

```
w(u→v) = w_difficulty · ramp + w_interference · interference + w_redundancy · redundancy + step

ramp         = |Δdifficulty − τ| / τ           two-sided: stalling is as wrong as leaping
                 + backslide_penalty · max(0, −Δdifficulty) / τ
interference = DTW(u,v) normalised to [0,1]    grows as clips get FURTHER apart
                 + penalties for switching task / skill family / embodiment / lab
redundancy   = max(0, novelty_floor − DTW) / novelty_floor   grows as clips get CLOSER
step         = flat per-hop cost               keeps the curriculum from meandering
```

The redundancy term is not decoration. Interference alone rewards *minimising* distance, so
its true optimum is training on the same clip repeatedly — and that is exactly what the
search returned before the term existed: a flawless difficulty ramp in which every
consecutive pair was a near-duplicate and skill coverage was 10%. A step that shows the
policy nothing new is a wasted step, so it has to cost something. `novelty_floor` is derived
from the data at the *same* 5th percentile that `path_metrics` uses to **measure**
redundancy, so the cost penalises precisely what the metric reports.

**Where it starts.** A virtual `START` node with zero-cost edges into the easy end of the
easiest skill family. Both halves are required: family membership alone put 106 of 273 clips
— *including the goal* — into the entry pool, and the search returned a one-clip
"curriculum" at zero cost. `start_quantile` bounds it to the bottom decile of difficulty.

**Goal input.** Free text, matched by TF-IDF cosine similarity against each clip's
`task_name` + `task_description`. Character n-grams (3–5, `char_wb`) and word n-grams (1–2)
are summed, because they fail differently: "shirt" matches "tshirt" only at the character
level. See *Known limitations* — this is lexical, not semantic, and it has a measured
failure mode.

**Rehearsal.** Every `review_every` clips, the path revisits one it has already seen: the
skill family absent longest, and within it the clip *farthest* from where the curriculum
currently is — the material most exposed to interference from recent training. Cluster
medoids break ties. This is a cheap **proxy** for replay-based anti-forgetting, with no
training evidence behind it in this build.

---

## Running on Modal

The two slow things are slow for different reasons, and Modal fixes them differently.

```bash
modal deploy modal_app.py
modal run modal_app.py::fetch --limit 320    # parallel discovery + download
modal run modal_app.py::sync_down            # pull episodes local (~62 MB)
modal run modal_app.py::pipeline             # quadratic DTW on 16 cores
modal run modal_app.py::find --goal "teach the robot to fold a shirt"
modal run modal_app.py::probe                # time the stages separately
modal run modal_app.py::status
```

**Fetching is network-bound.** Discovery pages R2 prefix listings one source at a time, and
a source holding tens of thousands of episode prefixes takes minutes to enumerate from a
laptop. Each source is discovered in its own container and the downloads are then fanned out
across containers, so wall-clock is the *slowest single source* rather than the sum:
**320 episodes in 211 s**, versus 40+ minutes and a read timeout locally.

**The DTW matrix is CPU-bound and quadratic**, and is cached to a volume by content hash.
Two persistent volumes: `egoverse-data` (episodes + artifacts) and `egoverse-cache` (DTW
matrices and parsed datasets).

`probe` exists because a single wall-clock number cannot tell those two apart, and guessing
wrong wastes the optimisation. Measured on 320 episodes / 16 cores:

| stage | before | after | what changed |
|---|---|---|---|
| load 320 zarr stores | **337 s** | **16.6 s** | dataset disk cache (`loader.py`) |
| DTW, 35,511 pairs | 20.7 s | 0.25 s | existing matrix cache |

Ingestion, not compute, was 94% of the runtime — each episode is a Zarr *group* of many
small files, so loading is dominated by per-file latency on a network volume. Collapsing it
into one file read was worth far more than any compute optimisation. Also note `n_jobs=-1`
is wrong inside a container: joblib resolves it via `os.cpu_count()`, which reports the
**host's** 32 cores rather than the cgroup's 16, and oversubscribes.

Modal needs the R2 credentials to reach the bucket. `modal_app.py` reads them from a Modal
secret named `egoverse-r2` holding only the four keys the client needs — the session token
is deliberately excluded, since R2 rejects `X-Amz-Security-Token` anyway.

---

## Sherpa — the web frontend

A purpose-built browser UI for the same pipeline, aimed at someone evaluating the work
rather than operating it. Two surfaces: **Route**, which makes the argument in reading
order, and **Workbench**, with the parameter rail and the analytical views.

The framing is the name. A sherpa knows the route, sets the pace, and calls the rest
stops — which is what this does, and it is not decoration: the difficulty score genuinely
runs 0 → 1, and rehearsal steps genuinely drop back. So a curriculum is drawn as an
**ascent profile** — altitude is difficulty, each stage is a step of the climb, the goal is
the summit, and rehearsal steps are camps below the ridge. Identical data to a
difficulty-by-step chart; the shape just says what the numbers mean.

Colour carries exactly two facts. **Embodiment** — human demonstrations in violet, robot in
blue — so the human→robot crossing in the garments route is visible without a legend. And
one warm hue reserved for **the route and its summit**, used nowhere else.

```bash
python scripts/export_snapshot.py          # writes web/public/snapshot.json
uvicorn server.api:app --port 8000         # the JSON API
cd web && npm install && npm run dev       # http://localhost:5173
```

For a single-process deployment, build the frontend and let the API serve it — `server/api.py`
mounts `web/dist` at `/` when it exists, so `http://localhost:8000` then serves both:

```bash
cd web && npm run build && cd ..
uvicorn server.api:app --port 8000
```

Ports 8501/8502 are left to Streamlit.

**Hybrid loading.** The narrative page first-paints from the exported `snapshot.json`, so it
is never empty and never waits on a pipeline run. Goal queries and graph-weight changes hit
the API, which is milliseconds — the DTW matrix is already cached to `.cache/` by content
hash. Only an ingestion or DTW parameter change forces a re-run, and that is fast on a cache
hit. With the API down the page still reads; the interactive controls say they need it.

**It cannot disagree with the CLI.** `server/api.py` holds no numeric logic — every endpoint
delegates to `src/`, and `RunRequest` reads its defaults out of `run_pipeline`'s own signature
rather than restating them, so a changed pipeline default propagates instead of being pinned
to whatever it was when the API was written. Verified by comparing 13 values from
`POST /api/run` against a direct in-process `run_pipeline()` call: identical to 1e-12.

| Endpoint | Returns |
|---|---|
| `GET /api/snapshot` | the default run: metrics, ARI + support counts, episodes, stages, rejections, decimated trajectories |
| `POST /api/run` | same shape, with ingestion/DTW overrides |
| `POST /api/graph` | clip graph nodes, edges with the full cost breakdown, force-directed layout, reachability repairs |
| `GET /api/domains` | the curated task groups, narrowed to what this dataset holds, with a clip count each |
| `POST /api/path` | goal → match, ordered steps, cost terms, proxy metrics, baseline comparison, coverage curve. Accepts `domain` for the same scoping `--domain` applies |
| `POST /api/matrix` | the DTW matrix as raw `float32` (binary, not JSON — this is the payload that grows quadratically) |
| `POST /api/redundancy` | near-duplicate pairs, closest first |
| `GET /api/trajectory/{id}` | full XYZ polyline for one episode |

**Two things that would otherwise break silently.** JSON has no `NaN` literal, and this
pipeline produces NaN legitimately and often — step 1 of a curriculum has no incoming edge
weight, review rows deliberately blank their ramp cost, `silhouette` is undefined below three
clusters. FastAPI's encoder emits a bare `NaN`, which `curl` shows happily and `JSON.parse`
rejects; everything therefore goes through `server/serialize.py`, and `tests/test_api.py`
re-parses every response with `parse_constant` so a leak fails a test rather than the browser.
Trajectories are also decimated server-side, since 273 episodes at full length is megabytes
of JSON to draw strokes a few hundred pixels wide.

**Worked examples.** The page carries the two demo curricula — `garments` / "teach the robot
to fold a shirt" and `containers` / "pack the items into the box" — as one-click cards that
run the live search rather than showing a canned transcript, and each prints the equivalent
`find_path.py` command beside it. `DOMAIN_PRESETS` is imported from `find_path.py`, so the
demo and the CLI cannot disagree about what a domain contains; the browser reproduces the
CLI's 11-step garments curriculum exactly, episode for episode.

**Real footage on the example cards.** Each worked example shows the actual front-camera
video for three of its clips — where the curriculum starts, halfway, and the goal it ends on.
The garments card makes the embodiment crossing visible without a caption: the first clip is
a human egocentric view, the last is the YAM arm folding a shirt.

Those six clips are fetched deliberately, not by default:

```bash
python scripts/fetch_clip_video.py --episodes-from reports/path.csv --dry-run   # price it
python scripts/fetch_clip_video.py <episode_id> ... --out web/public/clips      # fetch + encode
```

`images.front_1` is not a file you can copy and play — it is a 1-D Zarr array of
`variable_length_bytes`, one JPEG per entry, zstd-compressed and shard-indexed on `yam`. The
script reads it through Zarr, muxes with ffmpeg at `crf 28` scaled to 640px, and writes an
`index.json` manifest. That ratio is why this is viable: **287 MB of source JPEGs became 8.7 MB
of mp4** across the six demo clips. The frontend reads the manifest and falls back to the
plotted path for any episode without footage, so the page works identically with an empty
`web/public/clips/`.

**Playing a curriculum.** Every step plays in training order as an end-effector path at 40×
real time — one shared scale across the path with a 10 cm reference bar, rehearsal steps
marked with the step they repeat. Motion rather than video is the honest default here: it is
what the pipeline actually reasons about, and pose is ~5 KB an episode against ~50 MB for one
camera, which is what keeps 273 episodes at 84 MB on disk.

**No charting library.** Every visual is hand-drawn canvas or SVG — Plotly is what `app.py`
already uses, and its default look is what this replaces. The whole bundle is ~62 KB gzipped.

The page reports what the run actually shows rather than a number written into the markup:
the agreement headline is derived from the measured ARI, so a degenerate clustering reads as
one, and `tail_ratio > 2` raises an explicit warning that the matrix is outlier-dominated. The
hero plots the most mutually distinct episodes by coreset rank and states how many were held
back, rather than silently truncating.

---

## Deploying

Live at <https://andrewchoy--egoverse-curriculum-web.modal.run>, as a single Modal ASGI
app — `modal_app.py::web` returns the FastAPI app unchanged, and `server/api.py` already
mounts `web/dist` at `/`, so one container serves the SPA and the API on one origin and
the frontend's relative `/api` calls need no configuration.

```bash
python scripts/export_snapshot.py     # refresh the first-paint snapshot
cd web && npm run build && cd ..      # web/dist is uploaded at deploy time
modal deploy modal_app.py
```

**The data lives on the volumes, not in the image or in git.** `episodes_web/` on
`egoverse-data` holds the exact 333 stores this README's numbers come from, and the
matching DTW/dataset caches sit on `egoverse-cache`. The container symlinks `/root/data`
and `/root/.cache` onto them, so `run_pipeline`'s own relative defaults resolve to the
volumes and the deployed app runs byte-identical code to the CLI. Those two directories
are deliberately kept apart from the `episodes/` set that `::fetch` overwrites: the
browser first-paints from a `snapshot.json` exported against one specific dataset, and a
live API reading a *different* set would quietly disagree with the static page.

Nothing here needs credentials — the R2 secret is only for the offline fetch path.

**Cold start is ~95 s**, then ~0.5 s warm. The page itself is unaffected: it first-paints
entirely from the bundled `snapshot.json` with no server call, so only the interactive
controls wait on a sleeping container. Set `min_containers=1` on the `web` function to
keep one alive if that matters.

---

## Data notes — what the real dataset actually looks like

The schema was verified by inspecting real episodes, not assumed. Each of these
required explicit handling, and each would silently corrupt results if ignored:

- **Zarr v3, flat dot-separated keys.** Episodes are Zarr *v3* groups (`zarr.json`
  root, `c/0/0` chunks) with arrays named `left.obs_ee_pose`, `right.obs_ee_pose`,
  `images.front_1`. Zarr 2.x cannot read them.
- **Pose layout is `[x, y, z, qw, qx, qy, qz]`** — `(T, 7)`, quaternion in the last
  four columns. Only XYZ is used.
- **Arrays are chunk-padded past the episode end.** An episode with
  `total_frames=2808` is stored in a length-2900 array with a zero tail. *Every*
  episode inspected does this. Reading the array without truncating to `total_frames`
  appends a fabricated collapse to the world origin, which dominates any distance
  metric. Truncation is mandatory.
- **All-zero XYZ rows are missing-data sentinels**, not positions. They appear
  mid-episode in 13–25% of frames for some human-teleop sources. Treated as real
  coordinates they create huge spurious excursions. They are dropped, and the ratio is
  surfaced as a data-quality signal.
- **Units are inconsistent.** Robot sources (`yam`) record metres (|xyz| ≈ 0.2–0.3);
  human motion-capture (`scale`) records millimetres (|xyz| ≈ 200). Detected per
  episode and converted to metres, with the applied factor recorded.
- **Some episodes are entirely degenerate.** All 7 `eva` episodes sampled have a
  constant `right.obs_ee_pose` of `[0,0,0,-0.5,0.5,-0.5,0.5]` — the pose stream was
  never populated. Some bimanual episodes have one arm static to within 0.4 mm.
  These are rejected with a recorded reason rather than silently producing garbage.
- **Arm availability varies.** Some sources are bimanual, some single-arm. Since DTW
  requires a consistent channel count, `arm="auto"` (default) selects the *more
  active* arm per episode, always yielding 3-D. `arm="both"` gives 6-D but keeps only
  bimanual episodes.

On the 333 stores currently fetched this yields **273 usable and 60 correctly rejected**
episodes — an 82% yield. Rejections are recorded with a reason and surfaced in the Data
Quality tab, never dropped silently.

Only pose arrays are downloaded, never the JPEG camera streams: ~5 KB versus ~300 MB
per episode, a ~60,000× saving.

---

## Method notes — choices that materially change the answer

**Clustering runs on the distance matrix directly.** `KMeans` cannot be used here: it
interprets its input as points in Euclidean feature space and averages them, so
feeding it an `(N, N)` distance matrix clusters *rows of distances* as N-dimensional
coordinates — not the trajectories under the DTW metric, and its centroids correspond
to no real episode. This uses average-linkage agglomerative clustering with
`metric="precomputed"`, which consumes DTW distances directly and is deterministic.
Medoids (real episodes) stand in for centroids.

**Diversity is averaged over the strict upper triangle.** The diagonal is N structural
zeros; `mean(D)` over the full matrix is deflated by a factor of `(N-1)/N`.

**The redundancy cutoff is absolute and shared across compared subsets**, derived once
from the full dataset. Recomputing each subset's own 5th percentile is not comparable:
it rescales to whatever spread that subset happens to have, so a perfectly spread
selection still flags ~5% of itself. That bug inverted the ranking — coreset scored 0.88
against random's 0.50, the wrong way round — before a regression test pinned it.

**DTW cost grows as √T, not T.** tslearn's DTW returns the *root* of summed squared
local distances, so two length-*T* series separated by constant offset *d* cost
`d·√T`. Length normalisation therefore divides by **√(mean length)**. Dividing by the
mean length itself — the intuitive choice — overcorrects, making long episodes score
as systematically *less* distant. Measured on a fixed pair resampled from T=100 to
T=1600, `cost/√T` is flat within 0.5% while `cost/T` falls 4×. A useful side effect:
the normalised distance comes out in metres.

**Normalisation is the single most consequential setting, and the right answer changed
when the dataset grew.** `center` removes absolute workspace placement but keeps motion
*extent*, so a 5 cm nudge and a 50 cm sweep stay far apart — correct **within one
embodiment**, where extent is skill. Across embodiments it is wrong, because extent is
then hardware: a YAM arm spans 0.68 m on a tabletop while a head-mounted Aria recording
someone walking a room spans 3.4 m, a ~5× difference that swamps every behavioural signal.

On 273 episodes across four sources, changing *only* the normalisation:

| | `center` | `zscore` |
|---|---|---|
| `tail_ratio` (99th pctile / median) | **3.4** | 1.2 |
| Largest cluster's share, best *k* | **≥ 0.85** at every *k* | 0.25 |
| ARI vs `task_name` | **0.013** | **0.698** |
| Silhouette | 0.58 | 0.17 |

Under `center` a handful of large-extent episodes sit several times farther from everything
than a typical pair does. Agglomerative linkage then peels them off **one at a time**
instead of splitting the bulk, so *every* `k` from 2 to 12 leaves one cluster holding ≥85%
of the data — and this is a property of the distance matrix, not of the linkage: `average`,
`complete` and `single` all fail, with `complete` still leaving 227 of 267 in one cluster at
`k=12`. Since clusters stand in for **skill families**, that silently destroys the
interference penalty, the coverage metric and rehearsal selection all at once.

So **`zscore` is now the default.** `center` remains available and remains right for a
single-embodiment dataset.

Note the last row: **silhouette prefers the broken configuration.** It rates the collapsed
matrix 0.58 against 0.17 for the one that recovers the task labels, because a partition of
"one far outlier vs everything else" has an excellent silhouette by construction. Silhouette
is therefore useless for this choice, which is why `tail_ratio` exists as a separate
detector and why `run_pipeline` prints an explicit warning when it exceeds 2 or one cluster
exceeds 60%. A low silhouette on the good run is not a problem — those clusters genuinely
overlap in shape space.

**`k` is chosen among non-degenerate partitions only.** Plain silhouette maximisation picked
a **266-vs-1** split as the best of the range. The guard is on the *largest* cluster's share,
not the smallest cluster's size, and that distinction inverts the outcome: on the 28-episode
sample every `k` from 3 to 10 contains a singleton, and `k=7` is simultaneously the
silhouette winner and the best label recovery (ARI 0.901), so rejecting partitions for
containing a small cluster discards the right answer and elects `k=2` (ARI 0.264). What
characterises the pathology is one cluster *swallowing the dataset* — 99.6% versus 25%. When
no `k` qualifies, the fallback returns the **least** dominated partition rather than the
plain silhouette winner, which would have re-elected the exact pathology the guard exists to
block.

**Difficulty is scale- and duration-free.** Raw path length varies ~40× across sources
(a 400-frame `yam` episode travels ~1.5 m; a 3000-frame `aria` episode ~60 m), so
weighting it heavily collapses "difficulty" into "which source is this". The dominant
terms are dimensionless: log tortuosity, **normalised jerk**
(`√(T⁵/L² · ∫‖jerk‖²dt)`, the standard motor-control smoothness measure), and reversal
rate. Scores are then **rank**-scaled by default, because the distribution is heavily
right-skewed and min-max would compress the great majority of episodes into a band a
few thousandths wide, making the curriculum unreadable and un-thresholdable.

**Timesteps are per-episode.** EgoVerse mixes 30 and 60 fps; a shared `dt` would make
60 fps episodes look twice as fast and far jerkier.

---

## Performance

DTW is `O(N²·T²)`. Two knobs bound it: `--max-length` (resamples long episodes; real
ones reach 8,523 frames) and `--sakoe-chiba-radius` (restricts the warping band). The
matrix is cached to `.cache/` keyed by a content hash of the trajectories plus the
result-affecting config, so re-runs are instant.

Measured at `max_length=200`: **~0.131 ms/pair** on 8 local cores, and **0.58 ms/pair**
on a 16-core Modal container over 35,511 pairs (longer episodes there — median 2,250
frames against the earlier sample's ~1,000).

**Ingestion, not DTW, is the real bottleneck at this size.** 320 zarr stores took 337 s
to load from a network volume against 20.7 s for the entire distance matrix, because each
episode is a Zarr *group* of many small files and the cost is per-file latency. Both the
parsed dataset and the matrix are now disk-cached (`--no-cache` disables), which takes the
load to 16.6 s.

| Episodes | Pairs | Est. time (8 cores) | Matrix size (float64) |
|---|---|---|---|
| 40 | 780 | 0.1 s | 13 KB |
| 273 (current sample) | 37 K | ~5 s | 0.6 MB |
| 1,000 | 500 K | ~65 s | 8 MB |
| 5,000 | 12.5 M | ~27 min | 200 MB |
| 23,034 (full `processed_v3`) | 265 M | ~9.6 h | 4.2 GB |

The full dataset is the point at which distributed compute pays for itself — the
pairwise matrix is embarrassingly parallel over row blocks.

---

## Tests

```bash
pytest tests/ -q     # 142 passed
```

Fixtures are written in the *real* schema (Zarr v3, chunk padding, sentinel frames,
millimetre units, dead streams) rather than an idealised one, so the tests exercise
the quirks production data actually has. Tests needing real episodes skip
automatically when `data/` is empty. The dashboard is executed headlessly via
Streamlit's `AppTest`, since Streamlit renders exceptions into the page rather than
raising them.

The Track 1 suite pins the properties the design depends on rather than the ones that are
easy to assert: every edge cost is non-negative (Dijkstra's precondition); no path descends
beyond the tolerance; `START` reaches every clip; rehearsal only ever repeats clips already
introduced; the curriculum always ends on the goal; and the found path beats a reshuffle of
its own clips on monotonicity and on total absolute variation.

Two tests encode *limitations* rather than features, so they cannot be rediscovered
silently: `test_center_normalisation_is_outlier_dominated_at_scale` asserts the collapse
documented above, and should start failing — and be deleted — if `center` is ever fixed at
scale.

The API suite (`tests/test_api.py`) runs against synthetic episodes, so it needs no fetched
data and no network. Its load-bearing assertion is that every response parses as *strict*
JSON: `json.loads` accepts a bare `NaN` by default, so the tests pass `parse_constant` to
reject it, which is the only way the NaN-to-null contract stays honest. It also pins the
distinctions that are easy to lose — step 1 reports `null` rather than `0.0` for its
non-existent incoming edge, review rows blank their ramp cost but keep their interference
cost, and step count equals route length plus rehearsals. It skips cleanly if `fastapi` or
`httpx2` is absent.

---

## Known limitations

**Track 1**

- **Goal matching is lexical, not semantic.** TF-IDF has a measured failure mode: on the
  sample dataset, *"tidy up the table"* retrieves `pick_hat`, matching on the shared bigram
  "up the" rather than on meaning. Nothing in TF-IDF can know that "tidy" relates to "sort
  utensils". This is why every match carries a confidence and a lead-over-runner-up, and why
  the ranked alternates are a visible override control rather than a hidden diagnostic. A
  sentence encoder would fix it at the cost of a ~90 MB model download.
- **Rehearsal has no training evidence behind it.** The review nodes reproduce the
  *structure* of experience replay, and the metrics confirm they interleave earlier skills as
  intended, but nothing here shows they reduce forgetting. That needs the training run this
  build does not include.
- **The interference penalties are hand-set, not learned.** `p_task=0.5`, `p_cluster=0.3` and
  the rest are plausible orderings of disruptiveness, not measured ones.
- **Task, skill family and lab are near-redundant on this dataset.** Each `task_name` maps to
  exactly one cluster and one source, so the four categorical penalties largely double-count
  a single underlying distinction. On a dataset where one task appears across several labs
  they would separate.
- **The A\* heuristic is not provably admissible.** Remaining normalised DTW to the target can
  exceed the true remaining cost, so A\* is ε-approximate. Dijkstra is the default and is
  exact; at a few hundred nodes it is sub-millisecond, which makes the approximation a pure
  downside.
- **The redundancy metric saturates on an unscoped graph.** "Near-duplicate" is the 5th
  percentile of the pairwise distribution, and on the full 273-clip set that distribution is
  dominated by cross-embodiment pairs — so *every* within-task transition reads as a
  duplicate and the metric pins at 1.0. Scoping to a domain (`--domain garments`) restores
  its meaning. This is a limitation of a global-percentile threshold, not of the path.
- **Smoothness and breadth genuinely trade off.** The path is measurably more redundant than
  a random reshuffle (0.400 vs 0.255). Lowering `--w-interference` buys coverage and costs
  smoothness; there is no setting that wins both, and the comparison table is shown either
  way rather than tuned until it looks good.

**Track 2**

- **Unit detection is a heuristic** (median |xyz| > 10 ⇒ millimetres). It is recorded
  per episode in the Data Quality tab so it can be audited, but an episode recorded in
  millimetres yet confined near the origin would be misread as metres.
- **Only end-effector position is used.** Orientation, gripper state and force are
  ignored, so two motions tracing the same path with different wrist rotation or
  grasp timing are treated as identical.
- **Cross-embodiment comparison is approximate.** A human hand and a YAM arm sweeping
  the same path score as similar, though their control problems differ.
- **`arm="auto"` can pick different arms for different episodes**, so a left-handed and
  a right-handed execution of one task may not align as closely as they should.
- **The ARI headline is weaker than it looks on a small sample.** 0.653 over 195 labelled
  episodes is a more honest number than the 0.901 previously reported over 28 — that came
  from a small, cleanly-labelled sample. 78 of 273 episodes (all of `mecka`) carry no
  `task_name` at all and are excluded from the score rather than compared against the literal
  string `"unknown"`, which would make the metric a measure of label coverage.
- **Label coverage is uneven across groups, which an aggregate ARI cannot show.** 3 of the 10
  groups contain no labelled episode at all, so their grouping is neither confirmed nor
  contradicted by the score — though they hold only 8 episodes between them. The sharper gap
  is group 4: **21 episodes whose grouping rests on a single labelled example.** The
  dashboard reports this under the validation banner.
- **Agglomerative clustering and UMAP are `O(N²)` in memory.** Beyond ~20k episodes the
  4.2 GB distance matrix becomes the binding constraint, and clustering a coreset
  rather than the full set is the practical path.
- **`center` normalisation does not work on multi-embodiment data** — see *Method notes*.

---

## Not included

- **No training run.** The proxy metrics are proxies. Nothing here measures whether a policy
  trained in this order performs better, and the repo contains no simulator, so task success
  is not measurable at all. Note also that curriculum-ordering effects are fragile and often
  vanish outside constructed settings; a null result would not be surprising.
- **Only `obs_ee_pose` is fetched.** The `cmd_*`, joint and gripper streams exist in the
  schema and would be needed for true action prediction; the current fetch is pose-only,
  which is what makes it ~190 KB rather than ~300 MB per episode.
