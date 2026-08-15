/**
 * The clip graph with the chosen route drawn over it.
 *
 * The k-NN graph is the substrate and stays hairline — it is context, not the
 * answer. The route is the only ochre on the page, so what the search picked is
 * unmistakable without a legend. Stops are numbered because here the order
 * genuinely is the result; numbering anything else on this page would be
 * decoration.
 *
 * Review stops get an open ring rather than a filled stop: a rehearsal step
 * revisits a clip already introduced, so drawing it as another stop on the line
 * would imply the route doubles back, which it does not. The line follows the
 * searched route; rings mark which clips get revisited.
 */

import { useMemo } from 'react'
import { penVar } from '../lib/pens'
import { shortId } from '../lib/format'
import type { GraphPayload, PathStep } from '../lib/types'

export interface RouteGraphProps {
  graph: GraphPayload
  route: number[]
  steps: PathStep[]
  targetIndex: number | null
  /** Reports the clip under the pointer so the caller can annotate it. */
  onHover?: (clipIndex: number | null) => void
  hovered?: number | null
}

const VIEW = 100
const PAD = 6

export function RouteGraph({ graph, route, steps, targetIndex, onHover, hovered }: RouteGraphProps) {
  const positions = useMemo(() => {
    const xs = graph.nodes.map((n) => n.x)
    const ys = graph.nodes.map((n) => n.y)
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)
    const spanX = maxX - minX || 1
    const spanY = maxY - minY || 1
    // One scale for both axes so the force layout is not sheared; the shorter
    // axis is centred in the leftover space.
    const span = Math.max(spanX, spanY)
    const inner = VIEW - PAD * 2
    const offsetX = (span - spanX) / 2
    const offsetY = (span - spanY) / 2
    const map = new Map<number, { x: number; y: number }>()
    graph.nodes.forEach((node) => {
      map.set(node.index, {
        x: PAD + ((node.x - minX + offsetX) / span) * inner,
        y: PAD + inner - ((node.y - minY + offsetY) / span) * inner,
      })
    })
    return map
  }, [graph.nodes])

  const nodeById = useMemo(() => new Map(graph.nodes.map((n) => [n.index, n])), [graph.nodes])

  // Marks shrink as the clip count grows. Route stops do not shrink as far, because
  // they have to stay findable against a few hundred background clips.
  const shrink = Math.min(1, Math.sqrt(60 / Math.max(1, graph.nodes.length)))
  const rBase = Math.max(0.5, 1.25 * shrink)
  const rRoute = Math.max(1.3, 1.9 * shrink)

  const routePoints = route
    .map((clip) => positions.get(clip))
    .filter((p): p is { x: number; y: number } => Boolean(p))

  const routeD = routePoints.length
    ? routePoints.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(' ')
    : ''

  const reviewClips = new Set(steps.filter((s) => s.is_review).map((s) => s.clip_index))
  const routeOrder = new Map(route.map((clip, i) => [clip, i + 1]))

  return (
    <svg
      className="routegraph"
      viewBox={`0 0 ${VIEW} ${VIEW}`}
      role="img"
      aria-label={`Clip graph of ${graph.nodes.length} episodes with a ${route.length}-stop curriculum route highlighted.`}
    >
      {/* substrate: every admissible transition the k-NN structure allows */}
      <g className="routegraph__edges">
        {graph.edges.map((edge, i) => {
          const a = positions.get(edge.from)
          const b = positions.get(edge.to)
          if (!a || !b) return null
          return (
            <line
              key={i}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              strokeDasharray={edge.is_repair ? '1 1' : undefined}
            />
          )
        })}
      </g>

      {/* the route */}
      {routeD && (
        <path
          className="routegraph__route"
          d={routeD}
          pathLength={100}
          key={route.join('-')} /* remount re-triggers the draw-in */
        />
      )}

      {/* clips */}
      <g className="routegraph__nodes">
        {graph.nodes.map((node) => {
          const p = positions.get(node.index)
          if (!p) return null
          const onRoute = routeOrder.has(node.index)
          const isTarget = node.index === targetIndex
          return (
            <circle
              key={node.index}
              cx={p.x}
              cy={p.y}
              r={onRoute ? rRoute : rBase}
              fill={onRoute ? 'var(--route)' : penVar(node.source)}
              opacity={onRoute || hovered === node.index ? 1 : 0.55}
              stroke={isTarget ? 'var(--ink)' : 'none'}
              strokeWidth={isTarget ? 0.8 : 0}
              onPointerEnter={() => onHover?.(node.index)}
              onPointerLeave={() => onHover?.(null)}
            >
              <title>
                {shortId(node.episode_id)} · {node.task_name} · difficulty{' '}
                {node.difficulty.toFixed(3)}
              </title>
            </circle>
          )
        })}
      </g>

      {/* rehearsal markers */}
      <g className="routegraph__reviews">
        {[...reviewClips].map((clip) => {
          const p = positions.get(clip)
          if (!p) return null
          return <circle key={clip} cx={p.x} cy={p.y} r={rRoute + 1.4} />
        })}
      </g>

      {/* stop numbers — the order IS the result, so it is labelled */}
      <g className="routegraph__stops">
        {route.map((clip, i) => {
          const p = positions.get(clip)
          if (!p) return null
          return (
            <text key={clip} x={p.x} y={p.y - 3} textAnchor="middle">
              {i + 1}
            </text>
          )
        })}
      </g>

      {/* target mark: a plotter registration square, not another dot */}
      {targetIndex !== null &&
        positions.get(targetIndex) &&
        (() => {
          const p = positions.get(targetIndex)!
          const node = nodeById.get(targetIndex)
          return (
            <g className="routegraph__target">
              <rect x={p.x - 3.4} y={p.y - 3.4} width={6.8} height={6.8} />
              <title>Target: {node ? shortId(node.episode_id) : targetIndex}</title>
            </g>
          )
        })()}
    </svg>
  )
}
