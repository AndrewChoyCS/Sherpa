/**
 * The terrain behind the name: every episode's end-effector path at once.
 *
 * This is the raw material the whole system reasons over — a few hundred recorded
 * motions, each one a few kilobytes of pose and nothing else. Drawing them together
 * on load is the page's one orchestrated moment: the ground the route will later be
 * found through, appearing before anything is asked of it.
 *
 * Held deliberately quiet. It sits behind the headline at low opacity, and colour
 * carries one fact only — whether a human or a robot made the motion.
 *
 * Two things are preserved from the data rather than styled in:
 *   - **One shared scale.** A path that fills its cell really did travel further.
 *   - **Duration-proportional draw time.** All strokes start together, so short
 *     clips finish almost at once and the long ones are still moving at the end.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import type { TrajectoryPreview } from '../lib/types'

const MAX_DRAW_MS = 2600
const CELL_PAD = 0.12

/** Human demonstrations in plume, robot in aurora — the only fact colour carries. */
export function embodimentOf(source: string, embodiment?: string): 'human' | 'robot' {
  if (embodiment) return /human|hand|ego/i.test(embodiment) ? 'human' : 'robot'
  return /scale|aria|mecka/i.test(source) ? 'human' : 'robot'
}

const HUES = { human: 'rgba(165,123,216,', robot: 'rgba(74,99,231,' }

export interface TerrainFieldProps {
  trajectories: TrajectoryPreview[]
  /** episode_id → embodiment string, for colouring. */
  embodiments: Map<string, string>
  columns?: number
  opacity?: number
}

export function TerrainField({
  trajectories,
  embodiments,
  columns,
  opacity = 0.5,
}: TerrainFieldProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)

  /** 85th percentile of extent: keying to the single largest leaves everything else
      a tenth of its cell, because a few sources are head-worn cameras on someone
      walking a room. */
  const halfExtent = useMemo(() => {
    const extents = trajectories
      .map((t) => t.points.reduce((m, [x, y]) => Math.max(m, Math.abs(x), Math.abs(y)), 0))
      .sort((a, b) => a - b)
    if (!extents.length) return 1
    return extents[Math.min(extents.length - 1, Math.floor(extents.length * 0.85))] || 1
  }, [trajectories])

  const durations = useMemo(() => {
    const seconds = trajectories.map((t) => t.n_frames / (t.fps || 30))
    const longest = Math.max(1e-6, ...seconds)
    return seconds.map((s) => Math.max(120, (s / longest) * MAX_DRAW_MS))
  }, [trajectories])

  useEffect(() => {
    const element = wrapRef.current
    if (!element) return
    const observer = new ResizeObserver((entries) => setWidth(entries[0].contentRect.width))
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  const cols = columns ?? (width < 640 ? 6 : width < 1000 ? 9 : 12)
  const rows = Math.max(1, Math.ceil(trajectories.length / cols))
  const cell = width > 0 ? width / cols : 0
  const height = cell * rows

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || cell === 0 || trajectories.length === 0) return
    const context = canvas.getContext('2d')
    if (!context) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.round(width * dpr)
    canvas.height = Math.round(height * dpr)
    canvas.style.height = `${height}px`
    context.setTransform(dpr, 0, 0, dpr, 0, 0)

    const inner = cell * (1 - CELL_PAD * 2)
    const scale = inner / 2 / halfExtent

    const draw = (index: number, upTo: number) => {
      const trajectory = trajectories[index]
      const points = trajectory.points
      if (upTo < 2) return
      const cx = (index % cols) * cell + cell / 2
      const cy = Math.floor(index / cols) * cell + cell / 2
      const kind = embodimentOf(trajectory.source, embodiments.get(trajectory.episode_id))
      context.strokeStyle = `${HUES[kind]}${opacity})`
      context.lineWidth = 0.85
      context.lineJoin = 'round'
      context.lineCap = 'round'
      context.beginPath()
      for (let i = 0; i < upTo && i < points.length; i += 1) {
        const [x, y] = points[i]
        const px = cx + x * scale
        const py = cy - y * scale
        if (i === 0) context.moveTo(px, py)
        else context.lineTo(px, py)
      }
      context.stroke()
    }

    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    context.clearRect(0, 0, width, height)

    if (reduce) {
      trajectories.forEach((t, i) => draw(i, t.points.length))
      return
    }

    let raf = 0
    let start = 0
    const frame = (now: number) => {
      if (!start) start = now
      const elapsed = now - start
      context.clearRect(0, 0, width, height)
      let done = true
      trajectories.forEach((trajectory, index) => {
        const t = Math.min(1, elapsed / durations[index])
        draw(index, Math.max(2, Math.round(t * trajectory.points.length)))
        if (t < 1) done = false
      })
      if (!done) raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [trajectories, embodiments, cell, cols, width, height, halfExtent, durations, opacity])

  return (
    <div ref={wrapRef} className="terrain" aria-hidden="true">
      <canvas ref={canvasRef} style={{ width: '100%', display: 'block' }} />
    </div>
  )
}
