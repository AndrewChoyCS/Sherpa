/**
 * The ranking claim: can the diversity score choose between two subsets?
 *
 * A score that only describes ("this dataset measures 1.90 m") is not yet useful.
 * The deliverable is a score that *ranks*, so this draws the head-to-head: equal-sized
 * subsets picked by different strategies, scored against each other and against a null
 * model of random draws.
 *
 * Three deliberate choices:
 *
 * Both subsets are the same size. Several of these metrics move with N — a bigger
 * subset has more chances to contain a close pair — so unequal sizes would measure the
 * size gap rather than the strategy.
 *
 * A `redundant` adversarial control is shown alongside. It deliberately picks
 * near-duplicates, and a metric that cannot rank it last is not measuring diversity.
 * Including a baseline designed to lose is what makes the win legible.
 *
 * The null model is drawn, not summarised. A single random draw can beat a principled
 * selection by luck — at small budgets it demonstrably does — so the strip plots the
 * whole distribution of random subsets and marks where the coreset falls. That turns
 * "coreset looks better" into a percentile.
 *
 * The verdict line is computed from the numbers rather than written into the page, so
 * it cannot go stale if the episode set or the selection changes.
 */

import { num, pct, int } from '../lib/format'
import type { SubsetComparison } from '../lib/types'

export interface SubsetCompareProps {
  comparison: SubsetComparison
}

/** Colour per strategy: one hero, one honest baseline, one designed to lose. */
// Recoloured for the dark aurora ground (the previous values were tuned for a
// light background and went muddy). Semantics are unchanged: the principled
// selection carries the page's one warm accent, the adversarial control is the
// only red, and the chance baseline stays neutral.
const STRATEGY_HEX: Record<string, string> = {
  coreset: '#f0be7c',
  stratified: '#a57bd8',
  random: '#7b869e',
  redundant: '#e2685f',
}

const STRATEGY_NOTE: Record<string, string> = {
  coreset: 'farthest-point selection over the DTW matrix',
  stratified: 'round-robin across motion families',
  random: 'uniform sample — the honest baseline',
  redundant: 'adversarial control: picks near-duplicates',
}

function hexFor(name: string): string {
  return STRATEGY_HEX[name] ?? '#7b869e'
}

/**
 * Headline derived from the measured result.
 *
 * Guards on the null model rather than the head-to-head number: beating one random
 * draw is not evidence, and at small budgets the single draw can even win outright
 * while the coreset still sits at the top of the distribution.
 */
export interface RankingVerdict {
  eyebrow: string
  /** Matches `agreementVerdict`'s field name. */
  heading: string
  /**
   * Alias of `heading`, kept so the Sherpa view and the older log view can consume
   * this during the frontend rebrand. Drop once only one caller remains.
   */
  headline: string
  holds: boolean
}

const RANKING_EYEBROW = 'Can the score choose between two subsets?'

function verdict(heading: string, holds: boolean): RankingVerdict {
  return { eyebrow: RANKING_EYEBROW, heading, headline: heading, holds }
}

export function rankingVerdict(comparison: SubsetComparison): RankingVerdict {
  const percentile = comparison.baseline?.percentile ?? 0
  const z = comparison.baseline?.z_score ?? 0

  if (percentile >= 95 && z >= 2) {
    return verdict('It ranks a principled selection above chance.', true)
  }
  if (percentile >= 80) {
    return verdict('It leans the right way, but not decisively.', false)
  }
  return verdict('It does not separate the selections on this run.', false)
}

/** Horizontal bar, so long strategy names stay readable on one line. */
function MetricBars({
  comparison,
  metric,
  label,
  format,
  lowerIsBetter = false,
}: {
  comparison: SubsetComparison
  metric: string
  label: string
  format: (v: number) => string
  lowerIsBetter?: boolean
}) {
  const values = comparison.subsets.map((s) => s.metrics[metric] ?? 0)
  const max = Math.max(...values, 1e-9)
  const best = lowerIsBetter ? Math.min(...values) : Math.max(...values)

  return (
    <div className="cmp-metric">
      <p className="eyebrow">{label}</p>
      {comparison.subsets.map((subset) => {
        const value = subset.metrics[metric] ?? 0
        const width = Math.max(1, (value / max) * 100)
        const isBest = Math.abs(value - best) < 1e-12
        return (
          <div className="cmp-row" key={subset.name}>
            <span className="cmp-row__name mono">{subset.name}</span>
            <span className="cmp-row__track">
              <span
                className="cmp-row__fill"
                style={{ width: `${width}%`, background: hexFor(subset.name) }}
              />
            </span>
            <span className="cmp-row__value mono" data-best={isBest || undefined}>
              {format(value)}
            </span>
          </div>
        )
      })}
    </div>
  )
}

/** The random null model as a strip of ticks, with the candidate marked. */
function NullModel({ comparison }: { comparison: SubsetComparison }) {
  const samples = comparison.baseline_samples ?? []
  const baseline = comparison.baseline
  if (!baseline || samples.length === 0) return null

  const width = 460
  const height = 92
  const lo = Math.min(...samples, baseline.candidate)
  const hi = Math.max(...samples, baseline.candidate)
  const span = hi - lo || 1
  const x = (v: number) => 8 + ((v - lo) / span) * (width - 16)

  return (
    <figure className="cmp-null">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Random subset null model">
        <line x1={8} y1={58} x2={width - 8} y2={58} stroke="var(--rule)" strokeWidth={1} />
        {samples.map((value, i) => (
          <line
            key={i}
            x1={x(value)}
            x2={x(value)}
            y1={44}
            y2={58}
            stroke="#8a8f94"
            strokeWidth={1}
            opacity={0.32}
          />
        ))}
        <line
          x1={x(baseline.mean)}
          x2={x(baseline.mean)}
          y1={38}
          y2={64}
          stroke="#5b6165"
          strokeWidth={1.5}
          strokeDasharray="3 2"
        />
        <line
          x1={x(baseline.candidate)}
          x2={x(baseline.candidate)}
          y1={20}
          y2={68}
          stroke={hexFor(baseline.candidate_name)}
          strokeWidth={2.5}
        />
        <text x={x(baseline.candidate)} y={14} textAnchor="middle" className="cmp-null__tag">
          {baseline.candidate_name}
        </text>
        <text x={x(baseline.mean)} y={80} textAnchor="middle" className="cmp-null__tick">
          random mean
        </text>
      </svg>
      <figcaption className="prose caption">
        {int(baseline.trials)} random subsets of {comparison.subset_size} episodes.{' '}
        {baseline.candidate_name} lands at the {num(baseline.percentile, 0)}th percentile,{' '}
        {num(baseline.z_score, 1)}σ above the random mean.
      </figcaption>
    </figure>
  )
}

/** Diversity retained at each training budget, per strategy. */
function BudgetCurve({ comparison }: { comparison: SubsetComparison }) {
  const rows = comparison.curve ?? []
  if (rows.length === 0) return null

  const width = 460
  const height = 220
  const pad = { left: 44, right: 12, top: 14, bottom: 34 }
  const sizes = rows.map((r) => r.subset_size)
  const scores = rows.map((r) => r.diversity_score)
  const xMin = Math.min(...sizes)
  const xMax = Math.max(...sizes)
  const yMin = Math.min(...scores)
  const yMax = Math.max(...scores)
  const xSpan = xMax - xMin || 1
  const ySpan = yMax - yMin || 1

  const px = (v: number) => pad.left + ((v - xMin) / xSpan) * (width - pad.left - pad.right)
  const py = (v: number) => height - pad.bottom - ((v - yMin) / ySpan) * (height - pad.top - pad.bottom)

  const methods = Array.from(new Set(rows.map((r) => r.method)))

  return (
    <figure className="cmp-curve">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Diversity by training budget">
        <line
          x1={pad.left}
          y1={height - pad.bottom}
          x2={width - pad.right}
          y2={height - pad.bottom}
          stroke="var(--rule)"
        />
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} stroke="var(--rule)" />
        {methods.map((method) => {
          const series = rows
            .filter((r) => r.method === method)
            .sort((a, b) => a.subset_size - b.subset_size)
          const d = series
            .map((r, i) => `${i === 0 ? 'M' : 'L'}${px(r.subset_size)},${py(r.diversity_score)}`)
            .join(' ')
          return (
            <g key={method}>
              <path d={d} fill="none" stroke={hexFor(method)} strokeWidth={2} />
              {series.map((r) => (
                <circle
                  key={r.subset_size}
                  cx={px(r.subset_size)}
                  cy={py(r.diversity_score)}
                  r={2.5}
                  fill={hexFor(method)}
                />
              ))}
            </g>
          )
        })}
        <text x={pad.left} y={height - 8} className="cmp-null__tick">
          {xMin}
        </text>
        <text x={width - pad.right} y={height - 8} textAnchor="end" className="cmp-null__tick">
          {xMax} episodes
        </text>
      </svg>
      <figcaption className="prose caption">
        For a given episode budget, how much behavioural diversity each strategy keeps.
        Coreset staying above random across the range is what shows the ranking is real
        rather than an artifact of one chosen subset size.
      </figcaption>
    </figure>
  )
}

export function SubsetCompare({ comparison }: SubsetCompareProps) {
  if (!comparison?.subsets?.length) return null

  const byName = new Map(comparison.subsets.map((s) => [s.name, s]))
  const coreset = byName.get('coreset') ?? comparison.subsets[0]
  const random = byName.get('random') ?? comparison.subsets[1]
  const lead =
    random && random.metrics.diversity_score
      ? ((coreset.metrics.diversity_score - random.metrics.diversity_score) /
          Math.abs(random.metrics.diversity_score)) *
        100
      : 0

  return (
    <div className="cmp">
      <div className="cmp__legend">
        {comparison.subsets.map((subset) => (
          <span className="cmp__key" key={subset.name}>
            <i style={{ background: hexFor(subset.name) }} />
            <b className="mono">{subset.name}</b>
            <span>{STRATEGY_NOTE[subset.name] ?? ''}</span>
          </span>
        ))}
      </div>

      <div className="cmp__grid">
        <MetricBars
          comparison={comparison}
          metric="diversity_score"
          label="diversity score (mean pairwise DTW, metres)"
          format={(v) => num(v, 4)}
        />
        <MetricBars
          comparison={comparison}
          metric="mean_nn_distance"
          label="nearest-neighbour distance — higher means less duplication"
          format={(v) => num(v, 3)}
        />
        <MetricBars
          comparison={comparison}
          metric="redundancy_ratio"
          label="near-duplicate share — lower is better"
          format={(v) => pct(v, 0)}
          lowerIsBetter
        />
        <MetricBars
          comparison={comparison}
          metric="n_tasks_covered"
          label={`tasks covered of ${comparison.n_tasks_total}`}
          format={(v) => int(v)}
        />
      </div>

      <div className="cmp__panels">
        <NullModel comparison={comparison} />
        <BudgetCurve comparison={comparison} />
      </div>

      <p className="prose caption">
        Both subsets hold {comparison.subset_size} episodes — equal sizes on purpose,
        since these metrics move with N and unequal ones would measure the size gap
        instead of the strategy. Coreset leads random by {num(lead, 1)}% on diversity
        while carrying {pct(coreset.metrics.redundancy_ratio, 0)} near-duplicates
        against random&rsquo;s {pct(random?.metrics.redundancy_ratio, 0)}.
      </p>
    </div>
  )
}
