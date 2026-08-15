/**
 * The curriculum, played in order.
 *
 * There is no video to show. EgoVerse episodes carry JPEG camera streams, but this
 * project deliberately never downloads them — `fetch_egoverse_data.py` pulls
 * `POSE_KEYS` only, which is ~5 KB an episode against ~300 MB with the frames. So
 * what plays here is the thing the pipeline actually reasons about: the end-effector
 * path, drawn at a duration-proportional rate, one clip after another in training
 * order.
 *
 * Two decisions worth stating:
 *
 * 1. **One scale across this curriculum, not across the dataset.** Every clip in the
 *    strip and the viewer shares a single metres-per-pixel scale, so a small motion
 *    looks small and the comparison between steps survives — motion extent is part of
 *    what makes one clip harder than another.
 *
 *    The scale is set by the largest clip *in this path*, not the largest in the
 *    dataset. Dataset-wide, extent runs to ~1.6 m because some sources are head-worn
 *    cameras on someone walking a room; against that, a tabletop fold renders at a
 *    tenth of the box and the view stops showing you the clip at all. A 10 cm
 *    reference bar sits under the viewer so the absolute scale stays readable.
 *
 * 2. **Rehearsal steps are in the sequence, marked, not filtered.** A review step is
 *    a real training step that revisits an earlier clip, so it plays in its turn and
 *    is labelled with the step it repeats. Dropping them would misrepresent the
 *    curriculum; leaving them unmarked would make it look like the route doubles back.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ClipPlot, CLIP_VIEW, halfExtentOf } from './ClipPlot'
import { frames, num, shortId } from '../lib/format'
import type { PathStep, TrajectoryPreview } from '../lib/types'

/** Real seconds per animation second — the same convention the hero prints. */
const PLAY_SPEED = 40
const MIN_MS = 400
const PAD = 10

export interface PathFilmstripProps {
  steps: PathStep[]
  /** Every episode's decimated path, keyed by episode_id. */
  trajectories: Map<string, TrajectoryPreview>
}

export function PathFilmstrip({ steps, trajectories }: PathFilmstripProps) {
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [progress, setProgress] = useState(1)
  const rafRef = useRef(0)

  const playable = steps.filter((step) => trajectories.has(step.episode_id))

  /** Largest half-extent among this path's clips; sets the shared scale. */
  const halfExtent = useMemo(
    () => halfExtentOf(playable.map((step) => trajectories.get(step.episode_id))),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [steps, trajectories],
  )

  const current = playable[Math.min(index, playable.length - 1)]
  const currentTrajectory = current ? trajectories.get(current.episode_id) : undefined

  /** Draw duration for a clip, proportional to its real length. */
  const durationFor = useCallback(
    (trajectory: TrajectoryPreview | undefined) => {
      if (!trajectory) return MIN_MS
      const seconds = trajectory.n_frames / (trajectory.fps || 30)
      return Math.max(MIN_MS, (seconds / PLAY_SPEED) * 1000)
    },
    [],
  )

  // One clip's draw-in. On completion, advance if still playing; stop at the end
  // rather than looping, because a curriculum has a last step.
  useEffect(() => {
    cancelAnimationFrame(rafRef.current)
    if (!playing || !currentTrajectory) return

    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduce) {
      setProgress(1)
      const timer = window.setTimeout(() => {
        if (index < playable.length - 1) setIndex(index + 1)
        else setPlaying(false)
      }, 700)
      return () => window.clearTimeout(timer)
    }

    const duration = durationFor(currentTrajectory)
    let start = 0
    const frame = (now: number) => {
      if (!start) start = now
      const t = Math.min(1, (now - start) / duration)
      setProgress(t)
      if (t < 1) {
        rafRef.current = requestAnimationFrame(frame)
      } else if (index < playable.length - 1) {
        setIndex(index + 1)
      } else {
        setPlaying(false)
      }
    }
    rafRef.current = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(rafRef.current)
  }, [playing, index, currentTrajectory, durationFor, playable.length])

  // Selecting a step by hand shows it complete; play is what animates.
  const select = (next: number) => {
    setPlaying(false)
    setIndex(next)
    setProgress(1)
  }

  const play = () => {
    if (playing) return setPlaying(false)
    // Restarting from the end replays from the top, which is what a play button
    // is expected to do once a sequence has finished.
    if (index >= playable.length - 1 && progress >= 1) setIndex(0)
    setProgress(0)
    setPlaying(true)
  }

  const scale = (CLIP_VIEW - PAD * 2) / 2 / (halfExtent || 1)

  if (playable.length === 0) {
    return (
      <p className="notice">
        No motion is loaded for these clips, so the sequence cannot be played.
      </p>
    )
  }

  return (
    <div className="strip">
      {/* ---- viewer ---- */}
      <div className="strip__stage">
        <ClipPlot
          trajectory={currentTrajectory}
          progress={progress}
          scale={scale}
          strokeWidth={0.5}
          className="clipplot clipplot--large"
        />
        <div className="strip__scalebar" aria-hidden="true">
          <span className="strip__scalebar-line" style={{ width: `${(0.1 * scale * 100) / CLIP_VIEW}%` }} />
          <span className="unit">10 cm</span>
        </div>
      </div>

      {/* ---- readout ---- */}
      <div className="strip__meta stack stack--tight">
        <div>
          <p className="eyebrow">
            step {current?.step} of {steps.length}
            {current?.is_review && <> · rehearsal of step {current.reviews_step}</>}
          </p>
          <p className="readout readout-md">{current?.task_name.replace(/_/g, ' ')}</p>
          <p className="mono strip__id">{current && shortId(current.episode_id)}</p>
        </div>

        <dl className="strip__spec">
          <dt>difficulty</dt>
          <dd>{num(current?.difficulty, 3)}</dd>
          <dt>frames</dt>
          <dd>{frames(currentTrajectory?.n_frames)}</dd>
          <dt>duration</dt>
          <dd>
            {currentTrajectory
              ? `${(currentTrajectory.n_frames / (currentTrajectory.fps || 30)).toFixed(1)} s`
              : '—'}
          </dd>
          <dt>path length</dt>
          <dd>{num(currentTrajectory?.path_length, 2)} m</dd>
          <dt>source</dt>
          <dd>{current?.source}</dd>
          <dt>embodiment</dt>
          <dd>{current?.embodiment?.replace(/_/g, ' ')}</dd>
        </dl>

        <div className="strip__transport">
          <button
            className="btn btn--ghost"
            type="button"
            onClick={() => select(Math.max(0, index - 1))}
            disabled={index === 0}
            aria-label="Previous step"
          >
            ‹ prev
          </button>
          <button className="btn" type="button" onClick={play}>
            {playing ? '❙❙ pause' : '▶ play in order'}
          </button>
          <button
            className="btn btn--ghost"
            type="button"
            onClick={() => select(Math.min(playable.length - 1, index + 1))}
            disabled={index >= playable.length - 1}
            aria-label="Next step"
          >
            next ›
          </button>
        </div>
        <p className="unit">
          Plays at {PLAY_SPEED}× real time. No camera frames are downloaded for this dataset —
          this is the end-effector path the curriculum is actually built from.
        </p>
      </div>

      {/* ---- strip ---- */}
      <ol className="strip__reel">
        {playable.map((step, i) => {
          const trajectory = trajectories.get(step.episode_id)
          return (
            <li key={`${step.step}-${step.episode_id}`}>
              <button
                type="button"
                className="strip__cell"
                aria-current={i === index}
                data-review={step.is_review}
                onClick={() => select(i)}
                title={`Step ${step.step}: ${step.task_name}${
                  step.is_review ? ` (rehearsal of step ${step.reviews_step})` : ''
                }`}
              >
                <span className="strip__cellnum">
                  {step.step}
                  {step.is_review && <span className="strip__rev">◉</span>}
                </span>
                <ClipPlot
                  trajectory={trajectory}
                  progress={i === index ? progress : 1}
                  scale={scale}
                />
                <span className="strip__celld">{num(step.difficulty, 2)}</span>
              </button>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
