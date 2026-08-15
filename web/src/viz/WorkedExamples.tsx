/**
 * Two worked examples, wired to the live finder.
 *
 * A reviewer arriving cold does not know what to type, and a free-text box with no
 * suggestions is a dead end — so these are the two curricula the project is
 * demonstrated on, one click each. They are not screenshots or transcripts: pressing
 * one runs the same `/api/path` call the input box does, so what appears is computed
 * now, against whatever episodes are currently loaded.
 *
 * Each is scoped to a curated task group (`--domain` on the CLI). The scope matters:
 * routing "fold a shirt" across the whole graph lets the search wander through
 * unrelated domains, and the presets are what keep a curriculum inside a coherent
 * skill family. The presets themselves are imported from `find_path.py` server-side,
 * so this demo and the CLI cannot disagree about what "garments" means.
 */

import { useEffect, useRef, useState } from 'react'
import { ClipPlot, CLIP_VIEW, halfExtentOf, useDrawIn } from './ClipPlot'
import { clipUrl } from '../lib/clips'
import type { ClipEntry, ClipManifest } from '../lib/clips'
import { num } from '../lib/format'
import type { DomainInfo } from '../lib/api'
import type { PathPayload, PathStep, TrajectoryPreview } from '../lib/types'

export interface WorkedExample {
  id: string
  goal: string
  domain: string
  /** What this example is meant to show — one line, no salesmanship. */
  point: string
  cliParams: string
}

export const WORKED_EXAMPLES: WorkedExample[] = [
  {
    id: 'garments',
    goal: 'teach the robot to fold a shirt',
    domain: 'garments',
    point:
      'Opens on human demonstrations and crosses to the robot partway up the ramp, so the curriculum spans three embodiments rather than staying inside one.',
    cliParams: '--domain garments',
  },
  {
    id: 'containers',
    goal: 'pack the items into the box',
    domain: 'containers',
    point:
      'A tighter graph and a shorter route, ending on the hardest packing clip. Shows the same machinery on a task family with far fewer clips.',
    cliParams: '--domain containers',
  },
]

/**
 * Three clips from a curriculum: where it starts, where it is halfway, and the goal.
 *
 * Rehearsal steps are excluded from the picks. A review repeats an earlier clip, so
 * choosing one as the "middle" would show the same motion twice and understate the
 * ramp; they stay in the played sequence, just not in this three-clip summary.
 *
 * All three share one scale, so the ramp is legible as shape and size rather than
 * only as a number underneath.
 */
function pickThree(steps: PathStep[]): { label: string; step: PathStep }[] {
  const introduced = steps.filter((step) => !step.is_review)
  if (introduced.length === 0) return []
  if (introduced.length <= 2) {
    return introduced.map((step, i) => ({ label: i === 0 ? 'first' : 'hardest', step }))
  }
  const middle = introduced[Math.floor((introduced.length - 1) / 2)]
  return [
    { label: 'first', step: introduced[0] },
    { label: 'middle', step: middle },
    { label: 'hardest', step: introduced[introduced.length - 1] },
  ]
}

function ExamplePreview({
  path,
  trajectories,
  clips,
}: {
  path: PathPayload
  trajectories: Map<string, TrajectoryPreview>
  clips: ClipManifest
}) {
  const picks = pickThree(path.steps)
  const motions = picks.map((pick) => trajectories.get(pick.step.episode_id))
  const scale = (CLIP_VIEW - 16) / 2 / halfExtentOf(motions)

  if (picks.length === 0) return null

  return (
    <ol className="preview">
      {picks.map((pick, i) => (
        <PreviewCell
          key={pick.step.step}
          index={i}
          label={pick.label}
          step={pick.step}
          trajectory={motions[i]}
          scale={scale}
          total={path.steps.length}
          clip={clips[pick.step.episode_id]}
        />
      ))}
    </ol>
  )
}

function PreviewCell({
  index,
  label,
  step,
  trajectory,
  scale,
  total,
  clip,
}: {
  index: number
  label: string
  step: PathStep
  trajectory: TrajectoryPreview | undefined
  scale: number
  total: number
  clip: ClipEntry | undefined
}) {
  // Staggered so the three draw left to right, which is the direction the ramp runs.
  // Only used for the fallback plot; video needs no draw-in.
  const progress = useDrawIn(index, 850, Boolean(trajectory) && !clip)

  return (
    <li className="preview__cell">
      <p className="eyebrow preview__label">{label}</p>
      {clip ? (
        <ClipVideo clip={clip} label={`${label} clip: ${step.task_name}`} />
      ) : (
        <ClipPlot
          trajectory={trajectory}
          progress={progress}
          scale={scale}
          strokeWidth={1.1}
          className="clipplot preview__plot"
        />
      )}
      <p className="preview__d readout readout-sm">{num(step.difficulty, 3)}</p>
      <p className="preview__meta">
        step {step.step}/{total} · {step.source}
      </p>
    </li>
  )
}

/**
 * A fetched camera clip, looping silently.
 *
 * Autoplay is suppressed under `prefers-reduced-motion`; the clip still loads and can
 * be started by clicking it, so the footage is never simply unavailable. Click always
 * toggles, because six looping videos on one screen is a lot to have no control over.
 */
function ClipVideo({ clip, label }: { clip: ClipEntry; label: string }) {
  const ref = useRef<HTMLVideoElement>(null)
  const [playing, setPlaying] = useState(true)

  useEffect(() => {
    const video = ref.current
    if (!video) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      video.pause()
      setPlaying(false)
    }
  }, [])

  const toggle = () => {
    const video = ref.current
    if (!video) return
    if (video.paused) {
      void video.play()
      setPlaying(true)
    } else {
      video.pause()
      setPlaying(false)
    }
  }

  return (
    <button type="button" className="preview__video" onClick={toggle} aria-label={`${label} — ${playing ? 'pause' : 'play'}`}>
      <video
        ref={ref}
        src={clipUrl(clip)}
        muted
        loop
        playsInline
        autoPlay
        preload="metadata"
        aria-label={label}
      />
      {!playing && <span className="preview__playmark" aria-hidden="true">▶</span>}
    </button>
  )
}

export interface WorkedExamplesProps {
  domains: Record<string, DomainInfo> | null
  /** Resolved curricula, keyed by example id; absent until the API answers. */
  previews: Record<string, PathPayload>
  trajectories: Map<string, TrajectoryPreview>
  /** Fetched camera clips, keyed by episode id; empty when none were pulled. */
  clips: ClipManifest
  /** Which example is currently loaded, if any. */
  activeId: string | null
  busy: boolean
  onRun: (example: WorkedExample) => void
}

export function WorkedExamples({
  domains,
  previews,
  trajectories,
  clips,
  activeId,
  busy,
  onRun,
}: WorkedExamplesProps) {
  return (
    <ul className="examples">
      {WORKED_EXAMPLES.map((example) => {
        const info = domains?.[example.domain]
        const preview = previews[example.id]
        const active = activeId === example.id
        return (
          <li key={example.id} className="examples__card" data-active={active}>
            <p className="eyebrow">
              domain · {example.domain}
              {info && <> · {info.n_clips} clips</>}
            </p>
            <p className="examples__goal mono">{example.goal}</p>
            <p className="prose examples__point">{example.point}</p>

            {/* The curriculum's shape, before the reader commits to running it:
                where it starts, halfway, and the goal it ends on. */}
            {preview ? (
              <>
                <ExamplePreview path={preview} trajectories={trajectories} clips={clips} />
                <p className="examples__summary mono">
                  {preview.route.length} clips + {preview.n_reviews} rehearsal
                  {preview.n_reviews === 1 ? '' : 's'} · ρ {num(preview.report.spearman, 3)} ·
                  ends on {preview.match.task_name.replace(/_/g, ' ')}
                </p>
              </>
            ) : (
              <p className="examples__summary unit">
                Curriculum loads from the server; start it with{' '}
                <code>uvicorn server.api:app --port 8000</code>.
              </p>
            )}

            {info && (
              <ul className="examples__tasks">
                {info.tasks.map((task) => (
                  <li key={task}>{task.replace(/_/g, ' ')}</li>
                ))}
              </ul>
            )}

            <div className="examples__actions">
              <button
                className="btn"
                type="button"
                onClick={() => onRun(example)}
                disabled={busy}
                aria-current={active}
              >
                {busy && active ? 'routing…' : active ? 'shown below' : 'run this example'}
              </button>
            </div>

            {/* The equivalent command, so the browser demo is traceable back to
                something reproducible outside it. */}
            <code className="examples__cli">
              python find_path.py "{example.goal}" {example.cliParams}
            </code>
          </li>
        )
      })}
    </ul>
  )
}
