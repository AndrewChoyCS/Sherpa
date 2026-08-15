/**
 * Difficulty against training step — the curriculum's ramp.
 *
 * Drawn as a step plot rather than a smooth line because difficulty is piecewise
 * constant: a clip has one difficulty for as long as it is being trained on, and
 * interpolating between clips would draw values the curriculum never visits.
 *
 * Review steps are drawn as ochre drops rather than hidden. A rehearsal step is
 * *meant* to fall back in difficulty, and `path_report` measures monotonicity over
 * the introduction sequence for exactly that reason — so the dips are the
 * mechanism working, not violations, and the plot should show them.
 */

import { num } from '../lib/format'
import type { PathStep } from '../lib/types'

export interface RampPlotProps {
  steps: PathStep[]
  /** Spearman rho over the introduction sequence, for the corner readout. */
  spearman?: number | null
}

// A pixel-scale coordinate system rather than a 0-100 one. With a 100-unit-wide
// viewBox stretched across a 1,440px band every length is multiplied by ~14, so a
// 0.45-unit stroke lands at 6px and reads as a slab. Working near 1 unit = 1 px
// keeps stroke weights and marker radii meaning what they say.
const W = 1200
const H = 210
const PAD_L = 24
const PAD_R = 16
const PAD_V = 26

export function RampPlot({ steps, spearman }: RampPlotProps) {
  if (steps.length === 0) return null

  const innerW = W - PAD_L - PAD_R
  const innerH = H - PAD_V * 2
  const stepW = innerW / Math.max(1, steps.length)

  const x = (i: number) => PAD_L + i * stepW
  const y = (v: number) => PAD_V + innerH - v * innerH

  // Step outline across the introduction sequence only; review steps get their
  // own marks so they cannot be mistaken for a ramp violation.
  const introduced = steps.filter((s) => !s.is_review)
  const d: string[] = []
  introduced.forEach((step, i) => {
    const index = steps.indexOf(step)
    const px = x(index)
    const py = y(step.difficulty ?? 0)
    d.push(`${i === 0 ? 'M' : 'L'}${px.toFixed(2)},${py.toFixed(2)}`)
    d.push(`L${(px + stepW).toFixed(2)},${py.toFixed(2)}`)
  })

  return (
    <figure className="ramp">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Difficulty by training step across ${steps.length} steps.`}
      >
        {/* baseline + ceiling: difficulty is rank-scaled onto [0,1], so both
            bounds are meaningful and worth drawing. */}
        <line className="ramp__axis" x1={PAD_L} y1={y(0)} x2={W - PAD_R} y2={y(0)} />
        <line className="ramp__axis ramp__axis--faint" x1={PAD_L} y1={y(1)} x2={W - PAD_R} y2={y(1)} />

        <path className="ramp__line" d={d.join(' ')} />

        {steps.map((step, i) => {
          const cx = x(i) + stepW / 2
          const cy = y(step.difficulty ?? 0)
          return step.is_review ? (
            <g key={i} className="ramp__review">
              <line x1={cx} y1={y(0)} x2={cx} y2={cy} />
              <circle cx={cx} cy={cy} r={4} />
            </g>
          ) : (
            <circle key={i} className="ramp__stop" cx={cx} cy={cy} r={3.5} />
          )
        })}
      </svg>
      <figcaption className="ramp__caption">
        <span className="eyebrow">difficulty 0 → 1 by step</span>
        {spearman !== undefined && (
          <span className="mono">
            &rho; {num(spearman, 3)}
            <span className="unit"> monotonicity</span>
          </span>
        )}
        <span className="mono ramp__key">
          <span className="ramp__key-review" /> review
        </span>
      </figcaption>
    </figure>
  )
}

/**
 * The path against its baselines.
 *
 * `compare_orderings` returns aggregate metrics per ordering, not per-step
 * difficulty, so this is a ledger rather than a set of small-multiple curves —
 * plotting would require data the comparison does not produce.
 */
export interface BaselineLedgerProps {
  comparison: Record<string, Record<string, number | null>>
  metrics: { key: string; label: string; higherIsBetter: boolean; digits?: number }[]
}

export function BaselineLedger({ comparison, metrics }: BaselineLedgerProps) {
  const rows = Object.keys(comparison)
  if (rows.length === 0) return null

  // The path's own row is whichever key the comparison labels as the curriculum;
  // it is highlighted so a reader sees at a glance which line is the claim.
  const pathRow = rows.find((r) => /curriculum|path/i.test(r)) ?? rows[0]

  return (
    <table className="ledger">
      <thead>
        <tr>
          <th scope="col">ordering</th>
          {metrics.map((m) => (
            <th scope="col" key={m.key} className="num">
              {m.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row} data-primary={row === pathRow}>
            <td>{row}</td>
            {metrics.map((m) => {
              const value = comparison[row]?.[m.key]
              const best = bestFor(comparison, m)
              const isBest = value !== null && value !== undefined && value === best
              return (
                <td key={m.key} className="num" data-best={isBest}>
                  {num(value, m.digits ?? 3)}
                </td>
              )
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function bestFor(
  comparison: Record<string, Record<string, number | null>>,
  metric: { key: string; higherIsBetter: boolean },
): number | null {
  const values = Object.values(comparison)
    .map((row) => row?.[metric.key])
    .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  if (values.length === 0) return null
  return metric.higherIsBetter ? Math.max(...values) : Math.min(...values)
}
