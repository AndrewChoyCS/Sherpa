/**
 * The experiment the proxy metrics stand in for: a policy actually trained.
 *
 * Everything else on this page scores an *ordering* — monotonic difficulty, low
 * interference, coverage. None of it shows a model learning anything. This trains the
 * same behaviour-cloning policy three times on identical data, changing only the order
 * episodes arrive in, and plots what happened.
 *
 * The result is mixed, and the component is built to say so rather than to flatter it.
 * Curriculum order does **not** train faster — it lost to shuffling on every seed during
 * the first pass. What it does is finish lower, and finish far lower than the same
 * ordering reversed. So the headline is the *anti-curriculum gap*, not a speed-up, and
 * the layout puts that gap at the centre.
 *
 * Why anti-curriculum earns its place on the chart: curriculum and anti-curriculum have
 * near-identical first-pass curves, which tells you the early penalty is the cost of
 * ordering data at all instead of shuffling — the standard non-IID batching effect —
 * rather than a failure of this particular ordering. Without the reversed arm you would
 * read that penalty as "the curriculum is bad". With it, the curriculum-specific effect
 * is visible where it actually lives: at the end.
 *
 * The band is min-to-max across seeds, not a standard error. With 8 seeds a full range
 * is honest about spread in a way a shrinking error bar is not, and it makes overlapping
 * arms visibly overlapping.
 */

import { num } from '../lib/format'

/** Matches `curve_payload()` in `src/bc_experiment.py`. */
export interface CurveSeries {
  ordering: string
  n_seeds: number
  steps: number[]
  mean: number[]
  lo: number[]
  hi: number[]
  final_val_loss: number
  first_pass_auc: number
  forgetting: number
}

export interface PairedEntry {
  n_seeds: number
  mean_ours: number
  mean_shuffled: number
  mean_delta: number
  pct_change: number | null
  wins: number
  losses: number
  p_value: number | null
}

export interface TrainingCurvesPayload {
  series: CurveSeries[]
  summary: {
    orderings: Record<string, Record<string, number>>
    paired_vs_shuffled: Record<string, Record<string, PairedEntry>>
  }
  verdict: string
  n_runs: number
  config?: Record<string, unknown>
}

export interface TrainingCurvesProps {
  payload: TrainingCurvesPayload
}

const ARM_HEX: Record<string, string> = {
  curriculum: 'var(--summit, #f0be7c)',
  anti_curriculum: 'var(--ember, #e2685f)',
  shuffled: 'var(--haze, #7b869e)',
}

const ARM_LABEL: Record<string, string> = {
  curriculum: 'curriculum (easy → hard)',
  anti_curriculum: 'reversed (hard → easy)',
  shuffled: 'shuffled (standard practice)',
}

function hexFor(ordering: string): string {
  return ARM_HEX[ordering] ?? 'var(--haze, #7b869e)'
}

function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined) return 'n/a'
  return `${value >= 0 ? '+' : ''}${num(value, digits)}%`
}

/**
 * Headline computed from the paired statistics.
 *
 * Deliberately checks the speed-up claim first and reports it failing, because that is
 * the claim a reader assumes is being made. Leading with the final-loss win would be
 * technically true and materially misleading.
 */
export function trainingVerdict(payload: TrainingCurvesPayload): {
  eyebrow: string
  heading: string
  holds: boolean
} {
  const paired = payload.summary?.paired_vs_shuffled ?? {}
  const speed = paired.curriculum?.first_pass_auc
  const final = paired.curriculum?.final_val_loss
  const eyebrow = 'We trained a policy. Did the order matter?'

  const slower = speed && speed.mean_delta > 0 && (speed.p_value ?? 1) < 0.05
  const betterEnd = final && final.mean_delta < 0 && (final.p_value ?? 1) < 0.05

  if (slower && betterEnd) {
    return {
      eyebrow,
      heading: 'Not a speed-up. It ends lower, and far above its own reverse.',
      holds: true,
    }
  }
  if (slower) {
    return { eyebrow, heading: 'Ordering cost early convergence and did not repay it.', holds: false }
  }
  if (betterEnd) {
    return { eyebrow, heading: 'Curriculum order reached a lower final loss.', holds: true }
  }
  return { eyebrow, heading: 'The ordering effect sits inside seed noise.', holds: false }
}

function CurveChart({ series }: { series: CurveSeries[] }) {
  const width = 620
  const height = 300
  const pad = { left: 62, right: 14, top: 16, bottom: 40 }

  const allSteps = series.flatMap((s) => s.steps)
  const allValues = series.flatMap((s) => [...s.lo, ...s.hi])
  if (!allSteps.length || !allValues.length) return null

  const xMin = Math.min(...allSteps)
  const xMax = Math.max(...allSteps)
  const yMin = Math.min(...allValues)
  const yMax = Math.max(...allValues)
  const xSpan = xMax - xMin || 1
  const ySpan = yMax - yMin || 1

  const px = (v: number) => pad.left + ((v - xMin) / xSpan) * (width - pad.left - pad.right)
  const py = (v: number) =>
    height - pad.bottom - ((v - yMin) / ySpan) * (height - pad.top - pad.bottom)

  // First pass ends where the densely-sampled region does; marking it explains why the
  // curves are evaluated so heavily on the left.
  const firstPassEnd = Math.min(...series.map((s) => s.steps[Math.min(39, s.steps.length - 1)]))

  return (
    <figure className="curves">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Validation loss by training order">
        <line
          x1={pad.left}
          y1={height - pad.bottom}
          x2={width - pad.right}
          y2={height - pad.bottom}
          stroke="var(--rule)"
        />
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} stroke="var(--rule)" />

        <line
          x1={px(firstPassEnd)}
          x2={px(firstPassEnd)}
          y1={pad.top}
          y2={height - pad.bottom}
          stroke="var(--rule)"
          strokeDasharray="3 3"
        />
        <text x={px(firstPassEnd) + 4} y={pad.top + 10} className="curves__tick">
          end of first pass
        </text>

        {series.map((s) => {
          const band =
            s.steps.map((step, i) => `${i === 0 ? 'M' : 'L'}${px(step)},${py(s.hi[i])}`).join(' ') +
            ' ' +
            s.steps
              .map((_, i) => `L${px(s.steps[s.steps.length - 1 - i])},${py(s.lo[s.lo.length - 1 - i])}`)
              .join(' ') +
            ' Z'
          const line = s.steps
            .map((step, i) => `${i === 0 ? 'M' : 'L'}${px(step)},${py(s.mean[i])}`)
            .join(' ')
          return (
            <g key={s.ordering}>
              <path d={band} fill={hexFor(s.ordering)} opacity={0.14} stroke="none" />
              <path d={line} fill="none" stroke={hexFor(s.ordering)} strokeWidth={2} />
            </g>
          )
        })}

        <text x={pad.left} y={height - 12} className="curves__tick">
          {xMin}
        </text>
        <text x={width - pad.right} y={height - 12} textAnchor="end" className="curves__tick">
          {xMax} steps
        </text>
        <text x={8} y={pad.top + 8} className="curves__tick">
          {num(yMax, 6)}
        </text>
        <text x={8} y={height - pad.bottom} className="curves__tick">
          {num(yMin, 6)}
        </text>
      </svg>
      <figcaption className="prose caption">
        Validation loss against training step, mean of 8 seeds with the min–max band across
        seeds. Same data, same architecture, same initialisations — only the order episodes
        arrive in differs, and the data is never reshuffled.
      </figcaption>
    </figure>
  )
}

export function TrainingCurves({ payload }: TrainingCurvesProps) {
  if (!payload?.series?.length) return null

  const byName = new Map(payload.series.map((s) => [s.ordering, s]))
  const curriculum = byName.get('curriculum')
  const anti = byName.get('anti_curriculum')
  const paired = payload.summary?.paired_vs_shuffled ?? {}

  const antiGap =
    curriculum && anti
      ? ((anti.final_val_loss - curriculum.final_val_loss) / curriculum.final_val_loss) * 100
      : null

  return (
    <div className="curves-block">
      <div className="curves__legend">
        {payload.series.map((s) => (
          <span className="curves__key" key={s.ordering}>
            <i style={{ background: hexFor(s.ordering) }} />
            <b className="mono">{ARM_LABEL[s.ordering] ?? s.ordering}</b>
          </span>
        ))}
      </div>

      <CurveChart series={payload.series} />

      <div className="curves__stats">
        <div className="curves__stat">
          <span className="eyebrow">first pass vs shuffled</span>
          <b>{pct(paired.curriculum?.first_pass_auc?.pct_change)}</b>
          <span className="curves__note">
            worse — lost {paired.curriculum?.first_pass_auc?.losses ?? 0}/
            {paired.curriculum?.first_pass_auc?.n_seeds ?? 0} seeds
          </span>
        </div>
        <div className="curves__stat">
          <span className="eyebrow">final loss vs shuffled</span>
          <b>{pct(paired.curriculum?.final_val_loss?.pct_change)}</b>
          <span className="curves__note">
            better, {paired.curriculum?.final_val_loss?.wins ?? 0}/
            {paired.curriculum?.final_val_loss?.n_seeds ?? 0} seeds, p=
            {num(paired.curriculum?.final_val_loss?.p_value ?? 0, 3)}
          </span>
        </div>
        <div className="curves__stat curves__stat--hero">
          <span className="eyebrow">vs its own reverse</span>
          <b>{antiGap === null ? 'n/a' : `${num(antiGap, 0)}%`}</b>
          <span className="curves__note">reversed ordering finishes this much worse</span>
        </div>
        <div className="curves__stat">
          <span className="eyebrow">forgetting</span>
          <b>{anti && anti.forgetting > 0 ? 'reversed only' : 'none'}</b>
          <span className="curves__note">
            reversed drifts on early material; curriculum does not
          </span>
        </div>
      </div>

      <p className="prose">
        <b>Curriculum order is not a speed-up.</b> It lost to shuffling on every seed
        during the first pass ({pct(paired.curriculum?.first_pass_auc?.pct_change)}, p=
        {num(paired.curriculum?.first_pass_auc?.p_value ?? 0, 3)}). But the reversed
        ordering loses by almost exactly the same amount early
        ({pct(paired.anti_curriculum?.first_pass_auc?.pct_change)}), which places that
        penalty on <i>ordering data at all</i> rather than on this ordering — batches drawn
        in sequence are correlated, and shuffling exists to break that.
      </p>
      <p className="prose">
        The curriculum-specific effect appears at the end:{' '}
        {pct(paired.curriculum?.final_val_loss?.pct_change)} against shuffled, and{' '}
        {antiGap === null ? '' : `${num(antiGap, 0)}% `}
        against the same episodes in reverse — which also induces measurable forgetting.
        Running the ranking backwards hurts; that is the evidence the ranking carries real
        signal.
      </p>
      <p className="prose caption">
        {payload.n_runs} runs, 8 seeds per arm, paired by seed so differences are not
        confounded with initialisation. Task is next-displacement prediction from a window
        of end-effector poses — real training on real episodes, but not a robot success
        rate.
      </p>
    </div>
  )
}
