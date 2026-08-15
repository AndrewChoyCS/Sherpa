/**
 * The ascent — the signature element.
 *
 * This is the same data a difficulty-by-step chart would show, drawn as the thing
 * it actually describes: a route climbing to a summit. Difficulty is altitude, each
 * training step is a stage of the climb, and the goal is the peak. That is not a
 * metaphor laid over the numbers — the pipeline's difficulty score genuinely runs
 * 0 → 1 and the search genuinely ends on the hardest clip.
 *
 * Rehearsal steps are the reason this reads better as a ridgeline than as a chart.
 * A review deliberately drops back to easier material so nothing is forgotten — an
 * acclimatisation halt, not a failure to climb. Drawn as a descent to a marked
 * camp, it looks like what it is; drawn on a monotonicity chart it looks like a
 * violation, which is why the metrics measure it separately.
 *
 * Stages are joined with straight segments rather than a smoothed curve: difficulty
 * is piecewise constant per clip, and interpolating would draw altitudes the
 * curriculum never visits.
 */

import { useMemo } from 'react'
import type { PathStep } from '../lib/types'
import { num } from '../lib/format'

const W = 1200
const H = 340
const PAD_X = 40
const PAD_TOP = 54
const BASE = H - 46

export interface AscentProfileProps {
  steps: PathStep[]
  /** Draws the route in as it resolves. */
  animate?: boolean
  /** Reports the stage under the pointer. */
  onHover?: (step: PathStep | null) => void
  compact?: boolean
}

export function AscentProfile({ steps, animate = true, onHover, compact = false }: AscentProfileProps) {
  const id = useMemo(() => `asc-${Math.abs(hash(steps.map((s) => s.step + s.episode_id).join()))}`, [steps])

  const points = useMemo(() => {
    if (steps.length === 0) return []
    const span = Math.max(1, steps.length - 1)
    return steps.map((step, i) => ({
      step,
      x: PAD_X + (i / span) * (W - PAD_X * 2),
      y: BASE - (step.difficulty ?? 0) * (BASE - PAD_TOP),
    }))
  }, [steps])

  if (points.length === 0) return null

  const ridge = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')
  const massif = `${ridge} L${points[points.length - 1].x.toFixed(1)},${BASE} L${points[0].x.toFixed(1)},${BASE} Z`
  const summit = points[points.length - 1]

  return (
    <figure className={compact ? 'ascent ascent--compact' : 'ascent'}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Ascent profile: ${steps.length} training stages climbing from difficulty ${num(
          steps[0].difficulty,
          2,
        )} to ${num(summit.step.difficulty, 2)}.`}
      >
        <defs>
          <linearGradient id={`${id}-face`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--aurora)" stopOpacity="0.34" />
            <stop offset="100%" stopColor="var(--aurora)" stopOpacity="0" />
          </linearGradient>
          <linearGradient id={`${id}-ridge`} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="var(--plume)" />
            <stop offset="62%" stopColor="var(--aurora)" />
            <stop offset="100%" stopColor="var(--summit)" />
          </linearGradient>
          <filter id={`${id}-glow`} x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="6" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* altitude gridlines — difficulty is rank-scaled onto [0,1], so 0, ½ and 1
            are all real reference altitudes rather than arbitrary ticks */}
        <g className="ascent__grid">
          {[0, 0.5, 1].map((level) => {
            const y = BASE - level * (BASE - PAD_TOP)
            return (
              <g key={level}>
                <line x1={PAD_X} y1={y} x2={W - PAD_X} y2={y} />
                <text x={PAD_X - 10} y={y + 4} textAnchor="end">
                  {level.toFixed(1)}
                </text>
              </g>
            )
          })}
        </g>

        <g key={id} data-animate={animate}>
          <path className="ascent__face" d={massif} fill={`url(#${id}-face)`} />
          <path
            className="ascent__ridge"
            d={ridge}
            pathLength={100}
            fill="none"
            stroke={`url(#${id}-ridge)`}
            filter={`url(#${id}-glow)`}
          />
        </g>

        {/* stages */}
        <g className="ascent__stages">
          {points.map((p, i) => {
            const isSummit = i === points.length - 1
            const review = p.step.is_review
            return (
              <g
                key={`${p.step.step}-${i}`}
                className="ascent__stage"
                style={{ '--stage-delay': `${i * 70}ms` } as React.CSSProperties}
                data-summit={isSummit}
                data-review={review}
                onPointerEnter={() => onHover?.(p.step)}
                onPointerLeave={() => onHover?.(null)}
              >
                {/* a dropline gives every stage a footing on the base, which is what
                    makes the review dips legible as camps rather than as noise */}
                <line x1={p.x} y1={p.y} x2={p.x} y2={BASE} className="ascent__drop" />
                {review ? (
                  <circle cx={p.x} cy={p.y} r={7} className="ascent__camp" />
                ) : (
                  <circle cx={p.x} cy={p.y} r={isSummit ? 9 : 5} className="ascent__node" />
                )}
                {isSummit && <circle cx={p.x} cy={p.y} r={17} className="ascent__halo" />}
                <circle cx={p.x} cy={p.y} r={16} className="ascent__hit" />
              </g>
            )
          })}
        </g>

        {/* the summit is the one thing labelled, because it is the one thing asked for */}
        {(
          <g className="ascent__summit-label">
            <text x={summit.x} y={summit.y - 30} textAnchor="end">
              summit
            </text>
            <text x={summit.x} y={summit.y - 30} textAnchor="end" dy="14" className="ascent__summit-d">
              {num(summit.step.difficulty, 3)}
            </text>
          </g>
        )}

        <line className="ascent__base" x1={PAD_X} y1={BASE} x2={W - PAD_X} y2={BASE} />
      </svg>

      <figcaption className="ascent__caption">
        <span className="label">altitude = difficulty</span>
        <span className="ascent__key">
          <span className="ascent__key-camp" /> rehearsal camp
        </span>
        <span className="ascent__key">
          <span className="ascent__key-summit" /> goal
        </span>
      </figcaption>
    </figure>
  )
}

/** Stable id per step-set, so gradient defs never collide between instances. */
function hash(text: string): number {
  let h = 0
  for (let i = 0; i < text.length; i += 1) h = (h * 31 + text.charCodeAt(i)) | 0
  return h
}
