/**
 * Does the composite difficulty score beat a single naive feature?
 *
 * This isolates the *metric* rather than the search. The same clip selection is ordered
 * by the composite difficulty and by each obvious stand-in someone might reach for
 * instead — episode duration, path length, tortuosity — plus a random shuffle, and every
 * ordering is scored on the same measures.
 *
 * The scoring is restricted to metrics **independent of how difficulty is defined**, and
 * that restriction is the entire argument. Scoring a difficulty ordering on difficulty
 * monotonicity returns 1.0 for whichever key did the sorting, which proves nothing.
 * Task switching, cluster switching and consecutive DTW never see the difficulty score,
 * so a composite that lowers them is capturing real structure rather than restating its
 * own definition.
 *
 * The scope is rendered, not assumed. These results are scope-dependent rather than
 * merely noisy: across the whole graph the composite clearly beats every naive proxy,
 * while inside a single task family there is barely any task switching left to
 * differentiate and the keys collapse onto each other. A reader seeing this next to
 * domain-scoped results elsewhere on the page would otherwise read the two as
 * contradictory, so the component states which population it measured and says plainly
 * when that population cannot support the comparison.
 */

import { num } from '../lib/format'

/** Matches `ablation_payload()` in `src/path_metrics.py`. */
export interface AblationRow {
  sort_key: string
  [metric: string]: number | string
}

export interface AblationPayload {
  /** `"unscoped"`, a domain name, or `"tasks:a,b,c"`. */
  scope: string
  domain: string | null
  n_clips: number
  task_names: string[] | null
  is_scoped: boolean
  n_goals: number
  metrics: string[]
  /** Metrics where a lower value is better; all non-circular ones here are costs. */
  lower_is_better: string[]
  rows: AblationRow[]
}

export interface DifficultyAblationProps {
  payload: AblationPayload
  /** Metric to plot. Defaults to the first, `task_switch_rate`. */
  metric?: string
}

const METRIC_LABEL: Record<string, string> = {
  task_switch_rate: 'task switches per step',
  cluster_switch_rate: 'skill-family switches per step',
  mean_consecutive_dtw: 'motion distance between consecutive clips',
  frac_consecutive_near_duplicate: 'consecutive near-duplicates',
}

const COMPOSITE = 'composite difficulty'
const RANDOM = 'random order'

function hexFor(sortKey: string): string {
  if (sortKey === COMPOSITE) return 'var(--summit, #f0be7c)'
  if (sortKey === RANDOM) return 'var(--haze, #7b869e)'
  return 'var(--plume, #a57bd8)'
}

/**
 * Headline computed from the payload.
 *
 * Reports collapse explicitly rather than dressing a tie as a win: inside one task
 * family the naive keys land on the same value as the composite, and claiming a result
 * there would be false.
 */
export function ablationVerdict(payload: AblationPayload, metric: string): {
  eyebrow: string
  heading: string
  holds: boolean
} {
  const rows = payload.rows ?? []
  const composite = rows.find((r) => r.sort_key === COMPOSITE)
  const naive = rows.filter((r) => r.sort_key !== COMPOSITE && r.sort_key !== RANDOM)
  const eyebrow = 'Is the composite difficulty score doing real work?'

  if (!composite || naive.length === 0) {
    return { eyebrow, heading: 'Not enough orderings to compare.', holds: false }
  }

  const value = Number(composite[metric])
  const best = Math.min(...naive.map((r) => Number(r[metric])))
  const margin = (best - value) / (best || 1)

  if (margin < 0.01) {
    return {
      eyebrow,
      heading: payload.is_scoped
        ? 'Inside one task family, the naive proxies do just as well.'
        : 'The composite ties with the naive proxies here.',
      holds: false,
    }
  }
  return {
    eyebrow,
    heading: `It beats every single-feature proxy by ${num(margin * 100, 0)}%.`,
    holds: true,
  }
}

export function DifficultyAblation({ payload, metric }: DifficultyAblationProps) {
  if (!payload?.rows?.length) return null

  const chosen = metric ?? payload.metrics?.[0] ?? 'task_switch_rate'
  const rows = payload.rows
  const values = rows.map((r) => Number(r[chosen]))
  const max = Math.max(...values, 1e-9)
  const best = Math.min(...values)
  const verdict = ablationVerdict(payload, chosen)

  return (
    <div className="abl">
      <p className="abl__scope mono">
        measured over{' '}
        <b>{payload.is_scoped ? `the ${payload.scope} domain` : 'the full graph, unscoped'}</b>{' '}
        — {payload.n_clips} clips, {payload.n_goals} goals
      </p>

      {payload.is_scoped && (
        <p className="notice" role="status">
          Scoped results understate this comparison. Within a single task family there is
          almost no task switching left to differentiate, so every difficulty-based key
          collapses onto the same value. The unscoped run is the one that separates them.
        </p>
      )}

      <p className="eyebrow">{METRIC_LABEL[chosen] ?? chosen} — lower is better</p>

      {rows.map((row) => {
        const value = Number(row[chosen])
        const width = Math.max(1, (value / max) * 100)
        const isBest = Math.abs(value - best) < 1e-9
        return (
          <div className="abl-row" key={row.sort_key}>
            <span className="abl-row__name mono">{row.sort_key}</span>
            <span className="abl-row__track">
              <span
                className="abl-row__fill"
                style={{ width: `${width}%`, background: hexFor(row.sort_key) }}
              />
            </span>
            <span className="abl-row__value mono" data-best={isBest || undefined}>
              {num(value, 3)}
            </span>
          </div>
        )
      })}

      <p className="prose caption">
        Every ordering here contains the same clips; only the sort key differs. The
        metrics shown never see the difficulty score, so they cannot restate its own
        definition — which is what makes this evidence about the metric rather than a
        tautology. {verdict.holds ? verdict.heading : ''}
      </p>
    </div>
  )
}
