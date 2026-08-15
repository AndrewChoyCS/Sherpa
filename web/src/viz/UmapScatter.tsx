/**
 * The diversity map: UMAP of the DTW metric, coloured by source pen.
 *
 * Two dimensions, not three. The pipeline computes a 3-D embedding, but a 3-D
 * scatter on a flat page has to be rotated to be read at all, and the third axis
 * carries no more truth than the first two — UMAP distances are not metric in any
 * direction. Difficulty is mapped to radius instead, which is a real per-episode
 * quantity and needs no rotation to compare.
 */

import { useMemo } from 'react'
import { penVar } from '../lib/pens'
import { num, shortId } from '../lib/format'
import type { EpisodeRow } from '../lib/types'

export interface UmapScatterProps {
  episodes: EpisodeRow[]
  selected?: string | null
  onSelect?: (episodeId: string) => void
}

const VIEW = 100
const PAD = 5

export function UmapScatter({ episodes, selected, onSelect }: UmapScatterProps) {
  const points = useMemo(() => {
    if (episodes.length === 0) return []
    const xs = episodes.map((e) => e.UMAP_X)
    const ys = episodes.map((e) => e.UMAP_Y)
    const minX = Math.min(...xs)
    const minY = Math.min(...ys)
    const span = Math.max(Math.max(...xs) - minX, Math.max(...ys) - minY) || 1
    const inner = VIEW - PAD * 2
    // Marks shrink as the set grows, or 273 episodes render as four solid blobs
    // and the difficulty encoding stops being readable at all.
    const scale = Math.min(1, Math.sqrt(40 / Math.max(1, episodes.length)))
    return episodes.map((episode) => ({
      episode,
      x: PAD + ((episode.UMAP_X - minX) / span) * inner,
      y: PAD + inner - ((episode.UMAP_Y - minY) / span) * inner,
      r: (0.9 + (episode.difficulty ?? 0) * 2.0) * scale,
    }))
  }, [episodes])

  return (
    <svg
      className="umap"
      viewBox={`0 0 ${VIEW} ${VIEW}`}
      role="img"
      aria-label={`UMAP projection of ${episodes.length} episodes under the DTW metric; radius encodes difficulty.`}
    >
      {points.map(({ episode, x, y, r }) => (
        <circle
          key={episode.episode_id}
          cx={x}
          cy={y}
          r={r}
          fill={penVar(episode.source)}
          opacity={selected && selected !== episode.episode_id ? 0.25 : 0.7}
          stroke={episode.is_cluster_medoid ? 'var(--ink)' : 'none'}
          strokeWidth={episode.is_cluster_medoid ? 0.6 : 0}
          onClick={() => onSelect?.(episode.episode_id)}
        >
          <title>
            {shortId(episode.episode_id)} · {episode.task_name} · group {episode.cluster} ·
            difficulty {num(episode.difficulty, 3)}
            {episode.is_cluster_medoid ? ' · group medoid' : ''}
          </title>
        </circle>
      ))}
    </svg>
  )
}
