/**
 * API client.
 *
 * Hybrid loading, as agreed: the narrative page first-paints from a static
 * `snapshot.json` written by `scripts/export_snapshot.py`, so it is never empty
 * and never waits on a pipeline run. Goal queries and graph-weight changes go
 * to the live server, which is milliseconds — the expensive DTW matrix is
 * already cached to `.cache/` by content hash.
 *
 * If the live server is down the page still reads: the snapshot renders and the
 * interactive controls report that they need the server, rather than failing
 * silently or showing a broken chart.
 */

import type { GraphPayload, PathPayload, RedundantPair, RunConfig, Snapshot } from './types'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch {
    throw new ApiError('Server not reachable. Start it with: uvicorn server.api:app --port 8000')
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => '')
    throw new ApiError(detail || `${path} failed`, response.status)
  }
  return (await response.json()) as T
}

async function get<T>(path: string): Promise<T> {
  let response: Response
  try {
    response = await fetch(path)
  } catch {
    throw new ApiError('Server not reachable. Start it with: uvicorn server.api:app --port 8000')
  }
  if (!response.ok) throw new ApiError(`${path} failed`, response.status)
  return (await response.json()) as T
}

/**
 * The default run. Tries the static export first because it is instant and
 * needs no server; falls back to the live endpoint so a fresh checkout without
 * an exported snapshot still works.
 */
export async function loadSnapshot(): Promise<{ snapshot: Snapshot; live: boolean }> {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}snapshot.json`)
    if (response.ok) {
      return { snapshot: (await response.json()) as Snapshot, live: false }
    }
  } catch {
    /* fall through to the live endpoint */
  }
  return { snapshot: await get<Snapshot>('/api/snapshot'), live: true }
}

/** Re-run ingestion/DTW. The only genuinely slow call, and only on a cache miss. */
export function runPipeline(config: Partial<RunConfig>): Promise<Snapshot> {
  return post<Snapshot>('/api/run', config)
}

export interface GraphRequest extends Partial<RunConfig> {
  /** Any GraphConfig field by name; unknown keys are rejected server-side. */
  graph?: Record<string, number>
  tasks?: string[] | null
  /** A curated task group from find_path.py's DOMAIN_PRESETS, e.g. "garments". */
  domain?: string | null
}

export interface DomainInfo {
  tasks: string[]
  n_clips: number
}

/** The curated task groups, already narrowed to what this dataset holds. */
export function loadDomains(): Promise<Record<string, DomainInfo>> {
  return get<Record<string, DomainInfo>>('/api/domains')
}

export function buildGraph(request: GraphRequest): Promise<GraphPayload> {
  return post<GraphPayload>('/api/graph', request)
}

export interface PathRequest extends GraphRequest {
  goal: string
  review_every?: number
  max_reviews?: number
  search?: 'dijkstra' | 'astar'
  target_selection?: 'hardest' | 'medoid' | 'easiest'
  target_index?: number | null
  seeds?: number
}

export function findPath(request: PathRequest): Promise<PathPayload> {
  return post<PathPayload>('/api/path', request)
}

export function redundantPairs(config: Partial<RunConfig>): Promise<RedundantPair[]> {
  return post<RedundantPair[]>('/api/redundancy', config)
}

/**
 * The DTW matrix as raw float32 rather than JSON. At 23k episodes the matrix is
 * 4.2 GB; even at demo size, binary is a fraction of the JSON cost and drops
 * straight into a canvas without a parse step.
 */
export async function loadMatrix(config: Partial<RunConfig>): Promise<{ n: number; values: Float32Array }> {
  let response: Response
  try {
    response = await fetch('/api/matrix', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(config),
    })
  } catch {
    throw new ApiError('Server not reachable. Start it with: uvicorn server.api:app --port 8000')
  }
  if (!response.ok) throw new ApiError('/api/matrix failed', response.status)
  const n = Number(response.headers.get('x-matrix-n') ?? '0')
  const buffer = await response.arrayBuffer()
  return { n, values: new Float32Array(buffer) }
}

export async function loadTrajectory(episodeId: string): Promise<{
  episode_id: string
  points: [number, number, number][]
  fps: number
}> {
  return get(`/api/trajectory/${encodeURIComponent(episodeId)}`)
}
