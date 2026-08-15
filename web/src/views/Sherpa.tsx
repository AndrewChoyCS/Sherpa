/**
 * Sherpa — the narrative page.
 *
 * Reading order is the argument: what it is → the two worked routes → ask it for
 * your own → what the route is made of → how it decides. Each band makes one claim
 * and shows its evidence immediately.
 *
 * The hero is the demo. The goal input sits in it, so the first thing on screen is
 * the thing the product does rather than a description of it.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AscentProfile } from '../viz/AscentProfile'
import { TerrainField, embodimentOf } from '../viz/TerrainField'
import { PathFilmstrip } from '../viz/PathFilmstrip'
import { BaselineLedger } from '../viz/RampPlot'
import { RejectionLedger } from '../viz/RejectionLedger'
import { RouteGraph } from '../viz/RouteGraph'
import { WorkedExamples, WORKED_EXAMPLES } from '../viz/WorkedExamples'
import { rankingVerdict, SubsetCompare } from '../viz/SubsetCompare'
import { DifficultyAblation } from '../viz/DifficultyAblation'
import type { AblationPayload } from '../viz/DifficultyAblation'
import { loadAblation } from '../lib/ablation'
import type { WorkedExample } from '../viz/WorkedExamples'
import { ApiError, buildGraph, findPath, loadDomains } from '../lib/api'
import type { DomainInfo } from '../lib/api'
import { loadClips } from '../lib/clips'
import type { ClipManifest } from '../lib/clips'
import { int, metres, num, pct, shortId } from '../lib/format'
import type { GraphPayload, PathPayload, PathStep, Snapshot } from '../lib/types'

/**
 * `mean_abs_step` rather than `max_jump` for smoothness: `max_jump` counts only the
 * largest *upward* step, so an ordering that mostly descends posts a deceptively
 * small value. Mean absolute step is order-fair.
 */
const BASELINE_METRICS = [
  { key: 'spearman', label: 'ρ ramp', higherIsBetter: true },
  { key: 'mean_abs_step', label: 'mean |step|', higherIsBetter: false },
  { key: 'task_switch_rate', label: 'task switch', higherIsBetter: false },
  { key: 'cluster_switch_rate', label: 'skill switch', higherIsBetter: false },
  { key: 'mean_consecutive_dtw', label: 'consec. DTW', higherIsBetter: false },
  { key: 'frac_consecutive_near_duplicate', label: 'near-dupes', higherIsBetter: false },
  { key: 'cluster_coverage', label: 'coverage', higherIsBetter: true },
]

const TERRAIN_MAX = 96

export interface SherpaProps {
  snapshot: Snapshot
  onOpenWorkbench: () => void
}

export function Sherpa({ snapshot, onOpenWorkbench }: SherpaProps) {
  const [goal, setGoal] = useState('teach the robot to fold a shirt')
  const [path, setPath] = useState<PathPayload | null>(null)
  const [graph, setGraph] = useState<GraphPayload | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hovered, setHovered] = useState<number | null>(null)
  const [stage, setStage] = useState<PathStep | null>(null)
  const [domains, setDomains] = useState<Record<string, DomainInfo> | null>(null)
  const [clips, setClips] = useState<ClipManifest>({})
  const [ablation, setAblation] = useState<AblationPayload | null>(null)
  const [previews, setPreviews] = useState<Record<string, PathPayload>>({})
  const [activeExample, setActiveExample] = useState<string | null>(null)
  // Which scope the *displayed* route was searched under. The baseline claim below
  // differs by scope, so the copy has to know which run it is describing.
  const [activeDomain, setActiveDomain] = useState<string | null>(null)
  const routeRef = useRef<HTMLElement | null>(null)

  const metrics = snapshot.diversity_metrics

  const embodiments = useMemo(
    () => new Map(snapshot.episodes.map((e) => [e.episode_id, e.embodiment])),
    [snapshot.episodes],
  )

  const trajectoryById = useMemo(
    () => new Map(snapshot.trajectories.map((t) => [t.episode_id, t])),
    [snapshot.trajectories],
  )

  /** Terrain is bounded and picked by coreset rank — the most mutually distinct
      motions rather than whichever loaded first. */
  const terrain = useMemo(() => {
    if (snapshot.trajectories.length <= TERRAIN_MAX) return snapshot.trajectories
    const rank = new Map(
      snapshot.episodes.map((e) => [e.episode_id, e.coreset_rank ?? Number.MAX_SAFE_INTEGER]),
    )
    return [...snapshot.trajectories]
      .sort((a, b) => (rank.get(a.episode_id) ?? 0) - (rank.get(b.episode_id) ?? 0))
      .slice(0, TERRAIN_MAX)
  }, [snapshot.trajectories, snapshot.episodes])

  const humanCount = useMemo(
    () =>
      snapshot.episodes.filter((e) => embodimentOf(e.source, e.embodiment) === 'human').length,
    [snapshot.episodes],
  )

  const runQuery = useCallback(
    async (query: string, domain: string | null = null) => {
      if (!query.trim()) return
      setBusy(true)
      setError(null)
      try {
        // The graph must carry the same scope as the path, or the route would be
        // drawn over a graph it was not searched on.
        const scope = { data_dir: snapshot.config.data_dir, domain }
        setActiveDomain(domain)
        const [nextGraph, nextPath] = await Promise.all([
          buildGraph(scope),
          findPath({ ...scope, goal: query }),
        ])
        setGraph(nextGraph)
        setPath(nextPath)
      } catch (thrown) {
        setError(thrown instanceof ApiError ? thrown.message : 'Could not reach the route server.')
      } finally {
        setBusy(false)
      }
    },
    [snapshot.config.data_dir],
  )

  const runExample = useCallback(
    (example: WorkedExample) => {
      setGoal(example.goal)
      setActiveExample(example.id)
      const url = new URL(window.location.href)
      url.searchParams.set('goal', example.goal)
      url.searchParams.set('domain', example.domain)
      window.history.replaceState(null, '', url)
      void runQuery(example.goal, example.domain).then(() => {
        const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
        routeRef.current?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' })
      })
    },
    [runQuery],
  )

  const submitGoal = (event: React.FormEvent) => {
    event.preventDefault()
    if (busy) return
    setActiveExample(null)
    const url = new URL(window.location.href)
    url.searchParams.set('goal', goal)
    url.searchParams.delete('domain')
    window.history.replaceState(null, '', url)
    void runQuery(goal)
  }

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const linked = params.get('goal')
    if (!linked) return
    const domain = params.get('domain')
    setGoal(linked)
    setActiveExample(
      WORKED_EXAMPLES.find((e) => e.goal === linked && e.domain === domain)?.id ?? null,
    )
    void runQuery(linked, domain)
  }, [runQuery])

  useEffect(() => {
    loadClips().then(setClips)
  }, [])

  // Absent payload simply hides the section, so the page works with an empty
  // web/public/.
  useEffect(() => {
    loadAblation().then(setAblation)
  }, [])

  useEffect(() => {
    loadDomains().then(setDomains).catch(() => undefined)
  }, [])

  // Resolve both example routes up front so each card shows a real ascent rather
  // than a promise. Cheap: the distance matrix is cached, leaving graph build plus
  // Dijkstra. Failure is quiet — the cards still run.
  useEffect(() => {
    let cancelled = false
    Promise.all(
      WORKED_EXAMPLES.map((example) =>
        findPath({
          goal: example.goal,
          domain: example.domain,
          data_dir: snapshot.config.data_dir,
          seeds: 0,
        })
          .then((result) => [example.id, result] as const)
          .catch(() => null),
      ),
    ).then((results) => {
      if (cancelled) return
      setPreviews(Object.fromEntries(results.filter(Boolean) as [string, PathPayload][]))
    })
    return () => {
      cancelled = true
    }
  }, [snapshot.config.data_dir])

  const shown = stage ?? null

  return (
    <>
      {/* ================= hero ================= */}
      <header className="hero">
        <div className="hero__terrain">
          <TerrainField trajectories={terrain} embodiments={embodiments} opacity={0.8} />
        </div>
        <div className="hero__scrim" aria-hidden="true" />

        <div className="shell hero__inner">
          <p className="label hero__eyebrow">Curriculum routing for robot training data</p>
          <h1 className="hero-name">Sherpa</h1>
          <p className="lead hero__lead">Type a training goal. Watch the optimal path light up.</p>

          <form className="ask" onSubmit={submitGoal}>
            <label className="visually-hidden" htmlFor="goal">
              Training goal
            </label>
            <input
              id="goal"
              className="ask__input"
              value={goal}
              onChange={(event) => setGoal(event.target.value)}
              placeholder="teach the robot to fold a shirt"
              autoComplete="off"
              spellCheck={false}
            />
            <button className="btn ask__go" type="submit" disabled={busy}>
              {busy ? 'routing' : 'find route'}
            </button>
          </form>

          <dl className="hero__stats">
            <div>
              <dt className="label">Clips mapped</dt>
              <dd className="readout readout-md">{snapshot.n_episodes}</dd>
            </div>
            <div>
              <dt className="label">Human / robot</dt>
              <dd className="readout readout-md">
                {humanCount} / {snapshot.n_episodes - humanCount}
              </dd>
            </div>
            <div>
              <dt className="label">Frames read</dt>
              <dd className="readout readout-md">{int(totalFrames(snapshot))}</dd>
            </div>
            <div>
              <dt className="label">Pairs compared</dt>
              <dd className="readout readout-md">{int(metrics.n_pairs)}</dd>
            </div>
          </dl>
        </div>
      </header>

      {/* ================= what it is ================= */}
      <section className="band" aria-labelledby="what-title">
        <div className="shell stack">
          <p className="label">What Sherpa does</p>
          <h2 className="display" id="what-title">
            It turns a pile of egocentric clips into a route you can train on.
          </h2>
          <div className="two-up">
            <p className="prose">
              Sherpa reads EgoVerse's egocentric clips and builds a knowledge graph from how the
              motions actually behave — no task labels, no video, just the end-effector path.
              Type a training goal and it finds and highlights the optimal curriculum path:
              ordered by difficulty, and structured to avoid catastrophic interference during
              training.
            </p>
            <p className="prose">
              {/* Two claims were pulled here. "Wastes most of the data" sounds measured
                  and is not, and "before the model forgets it" asserts the rehearsal
                  works — which src/pathfinder.py explicitly calls an unvalidated proxy.
                  The structural argument stands on its own without either. */}
              Dumping every clip in at once, or shuffling them, throws away structure the data
              already has. A route uses it: it starts where the skill is easiest, climbs one
              step at a time, and interleaves earlier material as it goes — the shape of
              experience replay.
            </p>
          </div>
        </div>
      </section>

      {/* ================= worked routes ================= */}
      <section className="band band--tight" aria-labelledby="ex-title">
        <div className="shell stack">
          <p className="label">Two routes it found</p>
          <h2 className="title" id="ex-title">
            Worked examples
          </h2>
          <p className="prose">
            Each runs the search live against the clips currently loaded, scoped to a group of
            related tasks. Scope matters: across the whole graph a goal can be reached through
            unrelated skills, and these groups are what keep a route inside one family.
          </p>
          <WorkedExamples
            domains={domains}
            previews={previews}
            trajectories={trajectoryById}
            clips={clips}
            activeId={activeExample}
            busy={busy}
            onRun={runExample}
          />
        </div>
      </section>

      {/* ================= the route ================= */}
      <section className="band" aria-labelledby="route-title" ref={routeRef}>
        <div className="shell stack">
          <p className="label">The climb</p>
          <h2 className="display" id="route-title">
            {path ? goalHeading(path) : 'Ask for a route.'}
          </h2>

          {error && (
            <p className="notice" role="status">
              {error}
            </p>
          )}

          {!path && !error && (
            <p className="prose">
              Type a goal above, or open one of the worked examples. The search matches your words
              against every clip's task name and description, then finds the cheapest route to it —
              where cost is difficulty ramp plus interference plus redundancy.
            </p>
          )}

          {path && (
            <>
              <AscentProfile steps={path.steps} onHover={setStage} />

              <div className="readouts">
                <Readout label="Stages" value={String(path.route.length)} />
                <Readout label="Rehearsals" value={String(path.n_reviews)} />
                <Readout label="Monotonic ρ" value={num(path.report.spearman, 3)} />
                <Readout label="Route cost" value={num(path.search_cost, 2)} />
                <Readout label="Ends on" value={path.match.task_name.replace(/_/g, ' ')} wide />
              </div>

              <p className="prose">
                {shown ? (
                  <>
                    <strong>Stage {shown.step}</strong> · {shown.task_name.replace(/_/g, ' ')} ·
                    difficulty {num(shown.difficulty, 3)}
                    {shown.is_review && <> · rehearsal of stage {shown.reviews_step}</>}
                  </>
                ) : (
                  <>
                    Cost breaks into{' '}
                    {Object.entries(path.cost_terms)
                      .filter(([term]) => term !== 'weight')
                      .map(([term, value]) => `${term} ${num(value, 2)}`)
                      .join(' · ')}
                    . Hover a stage on the ridge to read it.
                  </>
                )}
              </p>

              {path.route.length === 1 && (
                <p className="notice" role="status">
                  The search reached this goal in one step — it is already an entry clip, so there
                  is no climb. Try a goal in a harder skill family.
                </p>
              )}

              {path.steps.length > 1 && (
                <>
                  <hr className="rule" />
                  <p className="label">Every stage, in order</p>
                  <PathFilmstrip steps={path.steps} trajectories={trajectoryById} />
                </>
              )}

              {graph && (
                <>
                  <hr className="rule" />
                  <p className="label">The route on the graph</p>
                  <p className="prose">
                    Every admissible transition the {graph.nodes.length}-clip graph allows, with the
                    chosen route lit. Hairlines are context; the warm line is the answer.
                  </p>
                  <div className="graph-frame">
                    <RouteGraph
                      graph={graph}
                      route={path.route}
                      steps={path.steps}
                      targetIndex={path.target_index}
                      onHover={setHovered}
                      hovered={hovered}
                    />
                  </div>
                </>
              )}

              {Object.keys(path.comparison).length > 1 && (
                <>
                  <hr className="rule" />
                  <p className="label">Against baselines of identical size</p>
                  <BaselineLedger comparison={path.comparison} metrics={BASELINE_METRICS} />
                  {/* The honest reading of this column depends on the scope the route was
                      searched under, and the page can show either — worked examples are
                      scoped to a task family, typed goals run over the whole graph. Both
                      figures come from the same 40-goal sweep via
                      `find_path.py --sweep 40`. Asserting one of them unconditionally was
                      wrong in exactly half the cases. */}
                  <p className="prose">
                    Read the difficulty-sorted column carefully: it ties on ramp smoothness by
                    construction, and on task and skill switching it ties exactly.{' '}
                    {activeDomain ? (
                      <>
                        Inside a task family the route does earn its keep on{' '}
                        <strong>repeated material</strong> — across a 40-goal sweep it posts
                        0.098 consecutive near-duplicates against 0.162, winning 11 and losing
                        none (p = 0.003).
                      </>
                    ) : (
                      <>
                        Across the whole graph, though, a 40-goal sweep has the two tied on
                        every proxy metric including repetition (0.592 against 0.600, p = 0.65)
                        — unscoped, the search <strong>matches</strong> a plain difficulty
                        ordering rather than beating it. Scope it to one task family and the
                        redundancy gap becomes real.
                      </>
                    )}{' '}
                    Either way it adds <strong>rehearsal</strong>, which sorting has no notion
                    of: {path.n_reviews} of these {path.steps.length} stages exist to revisit
                    earlier material. The random columns show what ordering is worth at all.
                  </p>
                </>
              )}
            </>
          )}
        </div>
      </section>

      {/* ================= can the score rank? =================
          Component and payload authored in the other session; placed here rather
          than rebuilt. Guarded, because the field is optional in the snapshot. */}
      {snapshot.comparison?.subsets?.length ? (
        <section className="band band--tight" aria-labelledby="rank-title">
          <div className="shell stack">
            {/* Both lines are derived from the measured null model, so this section
                cannot claim a win the numbers do not support. */}
            <p className="label">{rankingVerdict(snapshot.comparison).eyebrow}</p>
            <h2 className="title" id="rank-title">
              {rankingVerdict(snapshot.comparison).heading}
            </h2>
            <p className="prose">
              A score that only describes a dataset is not yet useful. This picks equal-sized
              subsets by different strategies and scores them head to head — including a control
              built to lose, and the full distribution of random draws behind it.
            </p>
            <SubsetCompare comparison={snapshot.comparison} />
          </div>
        </section>
      ) : null}

      {/* ================= what got thrown out ================= */}
      <section className="band band--tight" aria-labelledby="rej-title">
        <div className="shell stack">
          <p className="label">Before any of it counts</p>
          <h2 className="title" id="rej-title">
            {snapshot.n_skipped} clips were refused, each for a stated reason
          </h2>
          <p className="prose">
            Clips where the pose stream was never populated, or where an arm sat still to within a
            fraction of a millimetre. Treated as real coordinates they would dominate every distance
            in the matrix, so they are rejected with the measurement that disqualified them.
          </p>
          <RejectionLedger skipped={snapshot.skipped} />
        </div>
      </section>

      {/* ================= how it decides ================= */}
      <section className="band" aria-labelledby="how-title">
        <div className="shell stack">
          <p className="label">How it decides</p>
          <h2 className="display" id="how-title">
            Choices that change the answer.
          </h2>
          <dl className="notes">
            <div>
              <dt>The cheapest path is the curriculum.</dt>
              <dd>
                Edge cost is difficulty ramp plus interference plus redundancy plus step length.
                Minimising that sum over a route simultaneously minimises ramp roughness,
                cross-task disruption and wasted repetition — so ordering and routing are the same
                problem, not two objectives to reconcile.
              </dd>
            </div>
            <div>
              <dt>Rehearsal is scheduled, not random.</dt>
              <dd>
                Every few stages the route revisits one clip it has already taught: the skill family
                absent longest, and within it the clip furthest from where training currently is. It
                mirrors experience replay, and it is a proxy for anti-forgetting rather than a
                validated intervention.
              </dd>
            </div>
            <div>
              <dt>Distance is measured on shape, not on where the arm was.</dt>
              <dd>
                Motions are compared under dynamic time warping after normalisation. Comparing raw
                extent across embodiments measures hardware — a tabletop arm against a head-worn
                camera on someone walking a room — rather than skill.
              </dd>
            </div>
            <div>
              <dt>Difficulty is scale- and duration-free.</dt>
              <dd>
                Raw path length varies about 40× across sources, so weighting it heavily would
                collapse "difficulty" into "which rig recorded this". The dominant terms are
                dimensionless: log tortuosity, normalised jerk, and reversal rate.
              </dd>
            </div>
          </dl>

          {ablation && (
            <>
              <hr className="rule" />
              <div className="stack">
                <p className="label">Is the difficulty score doing the work?</p>
                <p className="prose">
                  The same clips, ordered by one signal at a time. Every metric below is blind
                  to the difficulty score — they count how often the ordering jumps between
                  tasks and skill families, so they cannot simply restate its definition back
                  at it. Sorting by the composite beats each raw ingredient, and beats random.
                </p>
                <DifficultyAblation payload={ablation} metric="task_switch_rate" />
                {/* Composite ranks 1st on three of the four metrics in this payload and
                    LAST on the fourth. Choosing only the flattering one would repeat the
                    mistake this page has already had to correct twice. */}
                <p className="prose">
                  It is not a clean sweep. On consecutive near-duplicates the composite is the
                  worst of the five — {num(ablationRow(ablation, 'composite difficulty', 'frac_consecutive_near_duplicate'), 3)} against random's{' '}
                  {num(ablationRow(ablation, 'random order', 'frac_consecutive_near_duplicate'), 3)} — because
                  grouping clips by difficulty puts similar clips next to each other, which is
                  the same property that makes it good at avoiding task switches. That cost is
                  why the route search carries a separate redundancy penalty rather than
                  trusting the ordering alone.
                </p>
              </div>
            </>
          )}

          <hr className="rule" />

          <div className="readouts">
            <Readout label="Diversity" value={metres(metrics.diversity_score, 3)} />
            <Readout label="Redundancy" value={pct(metrics.redundancy_ratio, 1)} />
            <Readout label="Mean NN distance" value={metres(metrics.mean_nn_distance, 3)} />
            <Readout label="Skill families" value={String(snapshot.n_clusters)} />
          </div>

          {/* The load-bearing check, kept as one line rather than a section: every
              claim above depends on the distance metric measuring behaviour, and
              this is the only number that establishes it does. */}
          {typeof snapshot.agreement?.task_name === 'number' && (
            <p className="prose">
              <strong>The grouping was never told the answers.</strong> Clustering these clips
              purely on motion recovers the human-written task labels at an Adjusted Rand Index
              of <strong>{num(snapshot.agreement.task_name, 3)}</strong>
              {typeof snapshot.agreement_support?.task_name === 'number' && (
                <> across the {snapshot.agreement_support.task_name} clips that carry a label</>
              )}
              . Chance is 0. That is the evidence the distance metric tracks behaviour rather
              than noise — everything else on this page rests on it.
            </p>
          )}

          <p>
            <button className="btn btn--ghost" type="button" onClick={onOpenWorkbench}>
              open workbench
            </button>
          </p>
        </div>
      </section>

      <footer className="foot">
        <div className="shell foot__inner">
          <span className="mono">Sherpa</span>
          <span className="unit">Andrew Choy</span>
          <span className="unit">Day Shift Hackathon</span>
          <span className="unit foot__src">{snapshot.config.data_dir}/ · {snapshot.n_episodes} clips</span>
        </div>
      </footer>
    </>
  )
}

function Readout({ label, value, wide }: { label: string; value: string; wide?: boolean }) {
  return (
    <div className="readouts__cell" data-wide={wide}>
      <p className="label">{label}</p>
      <p className="readout readout-lg">{value}</p>
    </div>
  )
}

/** One cell out of the ablation payload, by row name and metric. */
function ablationRow(payload: AblationPayload, sortKey: string, metric: string): number | null {
  const row = payload.rows?.find((r) => r.sort_key === sortKey)
  const value = row ? Number(row[metric]) : NaN
  return Number.isFinite(value) ? value : null
}

function goalHeading(path: PathPayload): string {
  const stages = path.route.length
  return `${stages} stage${stages === 1 ? '' : 's'} to ${path.match.task_name.replace(/_/g, ' ')}.`
}

function totalFrames(snapshot: Snapshot): number {
  return snapshot.trajectories.reduce((sum, t) => sum + t.n_frames, 0)
}

export { shortId }
