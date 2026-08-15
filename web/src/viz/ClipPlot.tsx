/**
 * One episode's end-effector path, optionally revealed up to `progress`.
 *
 * Shared by the filmstrip and the worked-example cards so a clip looks identical
 * wherever it appears — same projection, same registration cross, same pen.
 *
 * The caller supplies the scale rather than the plot fitting itself to its box.
 * That is the whole point: a set of clips drawn to one shared scale can be compared
 * by eye, and a set of individually fitted clips cannot.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { penHex } from '../lib/pens'
import type { TrajectoryPreview } from '../lib/types'

export const CLIP_VIEW = 100

export interface ClipPlotProps {
  trajectory: TrajectoryPreview | undefined
  /** 0–1 fraction of the path to draw. */
  progress: number
  /** Pixels (in viewBox units) per metre. */
  scale: number
  strokeWidth?: number
  className?: string
}

export function ClipPlot({
  trajectory,
  progress,
  scale,
  strokeWidth = 0.9,
  className = 'clipplot',
}: ClipPlotProps) {
  const path = useMemo(() => {
    if (!trajectory) return ''
    const points = trajectory.points
    const upTo = Math.max(2, Math.round(Math.min(1, progress) * points.length))
    const centre = CLIP_VIEW / 2
    let d = ''
    for (let i = 0; i < upTo && i < points.length; i += 1) {
      const [x, y] = points[i]
      // Screen y grows downward; negate so the plot reads as a normal chart.
      d += `${i === 0 ? 'M' : 'L'}${(centre + x * scale).toFixed(2)},${(centre - y * scale).toFixed(2)}`
    }
    return d
  }, [trajectory, progress, scale])

  if (!trajectory) return <span className="strip__missing unit">no motion</span>

  const hex = penHex(trajectory.source)
  const tip = tipOf(path)

  return (
    <svg
      className={className}
      viewBox={`0 0 ${CLIP_VIEW} ${CLIP_VIEW}`}
      role="img"
      aria-label={`End-effector path for ${trajectory.episode_id}`}
    >
      <g className="clipplot__reg">
        <line x1={CLIP_VIEW / 2 - 2} y1={CLIP_VIEW / 2} x2={CLIP_VIEW / 2 + 2} y2={CLIP_VIEW / 2} />
        <line x1={CLIP_VIEW / 2} y1={CLIP_VIEW / 2 - 2} x2={CLIP_VIEW / 2} y2={CLIP_VIEW / 2 + 2} />
      </g>
      <path
        d={path}
        fill="none"
        stroke={hex}
        strokeWidth={strokeWidth}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {/* The pen's current position, so a partly drawn path reads as in progress
          rather than as a short one. */}
      {progress < 1 && tip && <circle r={strokeWidth * 1.8} fill={hex} cx={tip.x} cy={tip.y} />}
    </svg>
  )
}

function tipOf(path: string): { x: number; y: number } | null {
  const cut = Math.max(path.lastIndexOf('L'), path.lastIndexOf('M'))
  if (cut < 0) return null
  const [x, y] = path.slice(cut + 1).split(',').map(Number)
  return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null
}

/**
 * Drives a one-shot draw-in, staggered by `index`.
 *
 * Returns 1 immediately under `prefers-reduced-motion`, so the plot is simply
 * complete rather than animating fast.
 */
export function useDrawIn(index = 0, duration = 900, ready = true): number {
  const [progress, setProgress] = useState(0)
  const raf = useRef(0)

  useEffect(() => {
    if (!ready) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setProgress(1)
      return
    }
    const delay = index * 220
    let start = 0
    const step = (now: number) => {
      if (!start) start = now
      const elapsed = now - start - delay
      setProgress(elapsed <= 0 ? 0 : Math.min(1, elapsed / duration))
      if (elapsed < duration) raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf.current)
  }, [index, duration, ready])

  return progress
}

/** Largest half-extent across a set of clips; the shared scale's denominator. */
export function halfExtentOf(clips: (TrajectoryPreview | undefined)[]): number {
  let max = 0
  for (const clip of clips) {
    if (!clip) continue
    for (const [x, y] of clip.points) max = Math.max(max, Math.abs(x), Math.abs(y))
  }
  return max || 1
}
