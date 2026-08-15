/**
 * The DTW distance matrix.
 *
 * Drawn on canvas from the raw float32 buffer, one rect per pair. A single-hue
 * ink ramp is used rather than a rainbow scale: the quantity is unsigned distance
 * with no meaningful midpoint, so lightness alone carries it, and a multi-hue
 * scale would invent categories the data does not have.
 *
 * Rows are in dataset order — which is also the distance matrix's order — so this
 * is the matrix as the pipeline holds it, not a re-sorted view. The diagonal is N
 * structural zeros, which is why the reported diversity score averages the strict
 * upper triangle instead of the whole matrix.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { shortId } from '../lib/format'

export interface DtwMatrixProps {
  n: number
  values: Float32Array
  episodeIds: string[]
  onSelect?: (episodeId: string) => void
}

const MAX_PX = 560

export function DtwMatrix({ n, values, episodeIds, onSelect }: DtwMatrixProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [probe, setProbe] = useState<{ i: number; j: number; value: number } | null>(null)

  /** Scale to the largest off-diagonal distance; the diagonal is structural zero. */
  const max = useMemo(() => {
    let out = 0
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const v = values[i * n + j]
        if (Number.isFinite(v) && v > out) out = v
      }
    }
    return out || 1
  }, [n, values])

  const size = Math.min(MAX_PX, 640)
  const cell = n > 0 ? size / n : 0

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || n === 0) return
    const context = canvas.getContext('2d')
    if (!context) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.round(size * dpr)
    canvas.height = Math.round(size * dpr)
    context.setTransform(dpr, 0, 0, dpr, 0, 0)
    context.clearRect(0, 0, size, size)

    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        const value = values[i * n + j]
        // Near distances are dark, far distances are near-paper: a dense block of
        // ink reads immediately as "these episodes are near-duplicates".
        const t = Number.isFinite(value) ? Math.min(1, value / max) : 1
        const lightness = 14 + t * 78
        context.fillStyle = `hsl(200 12% ${lightness}%)`
        context.fillRect(j * cell, i * cell, Math.ceil(cell), Math.ceil(cell))
      }
    }
  }, [n, values, max, cell, size])

  const handleMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (cell === 0) return
    const rect = event.currentTarget.getBoundingClientRect()
    const j = Math.floor((event.clientX - rect.left) / cell)
    const i = Math.floor((event.clientY - rect.top) / cell)
    if (i < 0 || j < 0 || i >= n || j >= n) return setProbe(null)
    setProbe({ i, j, value: values[i * n + j] })
  }

  return (
    <div className="matrix">
      <canvas
        ref={canvasRef}
        style={{ width: size, height: size, display: 'block' }}
        onPointerMove={handleMove}
        onPointerLeave={() => setProbe(null)}
        onClick={() => probe && onSelect?.(episodeIds[probe.i])}
        role="img"
        aria-label={`${n} by ${n} dynamic time warping distance matrix.`}
      />
      <div className="matrix__probe mono" aria-live="polite">
        {probe ? (
          <>
            <span>{shortId(episodeIds[probe.i] ?? '')}</span>
            <span>{shortId(episodeIds[probe.j] ?? '')}</span>
            <span className="matrix__value">{probe.value.toFixed(4)} m</span>
          </>
        ) : (
          <span className="unit">
            dark = near · light = far · diagonal is structural zero
          </span>
        )}
      </div>
    </div>
  )
}
