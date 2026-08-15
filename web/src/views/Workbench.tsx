/**
 * The workbench: parameters on the left, analysis on the right.
 *
 * Split by cost, not by topic. Ingestion and DTW settings force a pipeline re-run
 * and so sit behind an explicit "run" button; graph weights and the goal are
 * milliseconds and apply immediately. Mixing the two behind one button would make
 * every cheap change feel expensive, and mixing them behind none would re-run DTW
 * on a slider drag.
 */

import { useEffect, useMemo, useState } from 'react'
import { ApiError, loadMatrix, redundantPairs, runPipeline } from '../lib/api'
import { DtwMatrix } from '../viz/DtwMatrix'
import { UmapScatter } from '../viz/UmapScatter'
import { int, metres, num, pct, shortId } from '../lib/format'
import { loadedPens } from '../lib/pens'
import type { RedundantPair, RunConfig, Snapshot } from '../lib/types'

export interface WorkbenchProps {
  initial: Snapshot
}

export function Workbench({ initial }: WorkbenchProps) {
  const [snapshot, setSnapshot] = useState<Snapshot>(initial)
  const [config, setConfig] = useState<RunConfig>(initial.config)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [matrix, setMatrix] = useState<{ n: number; values: Float32Array } | null>(null)
  const [pairs, setPairs] = useState<RedundantPair[] | null>(null)
  const [selected, setSelected] = useState<string | null>(null)

  const dirty = useMemo(
    () => JSON.stringify(config) !== JSON.stringify(snapshot.config),
    [config, snapshot.config],
  )

  // The matrix and the redundancy list both describe the *current* snapshot, so
  // they refetch whenever it changes and never linger from a previous run.
  useEffect(() => {
    let cancelled = false
    setMatrix(null)
    setPairs(null)
    loadMatrix(snapshot.config)
      .then((result) => !cancelled && setMatrix(result))
      .catch(() => undefined)
    redundantPairs(snapshot.config)
      .then((result) => !cancelled && setPairs(result))
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [snapshot.config])

  const run = async () => {
    setBusy(true)
    setError(null)
    try {
      setSnapshot(await runPipeline(config))
    } catch (thrown) {
      setError(thrown instanceof ApiError ? thrown.message : 'Pipeline run failed.')
    } finally {
      setBusy(false)
    }
  }

  const set = <K extends keyof RunConfig>(key: K, value: RunConfig[K]) =>
    setConfig((previous) => ({ ...previous, [key]: value }))

  const metrics = snapshot.diversity_metrics
  const pens = loadedPens(snapshot.sources)
  const episodeOrder = snapshot.episodes.map((e) => e.episode_id)

  return (
    <div className="bench stock">
      {/* ---------------- control rail ---------------- */}
      <aside className="bench__rail" aria-label="Pipeline parameters">
        <p className="eyebrow">Ingestion</p>
        <div className="field">
          <label htmlFor="wb-dir">Episode directory</label>
          <input
            id="wb-dir"
            type="text"
            value={config.data_dir}
            onChange={(event) => set('data_dir', event.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="wb-arm">End-effector arm</label>
          <select
            id="wb-arm"
            value={config.arm}
            onChange={(event) => set('arm', event.target.value as RunConfig['arm'])}
          >
            <option value="auto">auto · more active arm</option>
            <option value="left">left</option>
            <option value="right">right</option>
            <option value="both">both · 6-D bimanual</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="wb-min">Minimum valid frames · {config.min_length}</label>
          <input
            id="wb-min"
            type="range"
            min={10}
            max={300}
            step={10}
            value={config.min_length}
            onChange={(event) => set('min_length', Number(event.target.value))}
          />
        </div>

        <p className="eyebrow">DTW distance</p>
        <div className="field">
          <label htmlFor="wb-norm">Normalisation</label>
          <select
            id="wb-norm"
            value={config.normalize}
            onChange={(event) => set('normalize', event.target.value as RunConfig['normalize'])}
          >
            <option value="center">center · keep extent</option>
            <option value="zscore">zscore · pure shape</option>
            <option value="none">none · raw world</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="wb-cap">Resample cap · {config.max_length ?? '—'} frames</label>
          <input
            id="wb-cap"
            type="range"
            min={50}
            max={600}
            step={50}
            value={config.max_length ?? 200}
            onChange={(event) => set('max_length', Number(event.target.value))}
          />
        </div>
        <div className="field field--row">
          <input
            id="wb-lennorm"
            type="checkbox"
            checked={config.length_normalize}
            onChange={(event) => set('length_normalize', event.target.checked)}
          />
          <label htmlFor="wb-lennorm">Length-normalise (÷√T, gives metres)</label>
        </div>

        <p className="eyebrow">Curriculum</p>
        <div className="field">
          <label htmlFor="wb-k">
            Groups · {config.n_clusters ?? `auto (${snapshot.n_clusters})`}
          </label>
          <input
            id="wb-k"
            type="range"
            min={0}
            max={10}
            step={1}
            value={config.n_clusters ?? 0}
            onChange={(event) =>
              set('n_clusters', Number(event.target.value) === 0 ? null : Number(event.target.value))
            }
          />
          <span className="unit">0 = choose k by silhouette</span>
        </div>
        <div className="field">
          <label htmlFor="wb-scaling">Difficulty scaling</label>
          <select
            id="wb-scaling"
            value={config.difficulty_scaling}
            onChange={(event) =>
              set('difficulty_scaling', event.target.value as RunConfig['difficulty_scaling'])
            }
          >
            <option value="rank">rank</option>
            <option value="minmax">minmax</option>
          </select>
        </div>

        <button className="btn" type="button" onClick={run} disabled={busy || !dirty}>
          {busy ? 'running…' : dirty ? 'run pipeline' : 'up to date'}
        </button>
        <p className="unit">
          The DTW matrix is cached to <code>.cache/</code> by content hash, so a repeat of any earlier
          setting returns immediately.
        </p>
        {error && <p className="notice">{error}</p>}
      </aside>

      {/* ---------------- analysis ---------------- */}
      <div className="bench__body">
        <section className="band band--tight">
          <div className="titleblock">
            <div>
              <span className="eyebrow">episodes</span>
              <span className="readout readout-md">{snapshot.n_episodes}</span>
            </div>
            <div>
              <span className="eyebrow">rejected</span>
              <span className="readout readout-md">{snapshot.n_skipped}</span>
            </div>
            <div>
              <span className="eyebrow">diversity</span>
              <span className="readout readout-md">{metres(metrics.diversity_score, 5)}</span>
            </div>
            <div>
              <span className="eyebrow">redundancy</span>
              <span className="readout readout-md">{pct(metrics.redundancy_ratio, 1)}</span>
            </div>
            <div>
              <span className="eyebrow">silhouette</span>
              <span className="readout readout-md">{num(metrics.silhouette, 3)}</span>
            </div>
            <div>
              <span className="eyebrow">ARI vs task</span>
              <span className="readout readout-md">{num(snapshot.agreement.task_name, 3)}</span>
            </div>
          </div>
        </section>

        <section className="band band--tight" aria-labelledby="wb-map">
          <p className="eyebrow" id="wb-map">
            Diversity map · UMAP of the DTW metric
          </p>
          <UmapScatter
            episodes={snapshot.episodes}
            selected={selected}
            onSelect={setSelected}
          />
          <ul className="penkey">
            {pens.map((pen) => (
              <li key={pen.source}>
                <span className="penkey__swatch" style={{ background: pen.hex }} />
                {pen.source}
              </li>
            ))}
          </ul>
        </section>

        <section className="band band--tight" aria-labelledby="wb-matrix">
          <p className="eyebrow" id="wb-matrix">
            DTW distance matrix · {snapshot.n_episodes}² pairs, {int(metrics.n_pairs)} distinct
          </p>
          {matrix ? (
            <DtwMatrix
              n={matrix.n}
              values={matrix.values}
              episodeIds={episodeOrder}
              onSelect={setSelected}
            />
          ) : (
            <p className="unit">Matrix needs the server: uvicorn server.api:app --port 8000</p>
          )}
        </section>

        <section className="band band--tight" aria-labelledby="wb-stages">
          <p className="eyebrow" id="wb-stages">
            Curriculum stages
          </p>
          <table className="ledger">
            <thead>
              <tr>
                <th scope="col">stage</th>
                <th scope="col" className="num">
                  episodes
                </th>
                <th scope="col" className="num">
                  mean difficulty
                </th>
                <th scope="col" className="num">
                  min
                </th>
                <th scope="col" className="num">
                  max
                </th>
                <th scope="col" className="num">
                  mean path length
                </th>
              </tr>
            </thead>
            <tbody>
              {snapshot.stages.map((stage) => (
                <tr key={stage.stage}>
                  <td>{stage.stage}</td>
                  <td className="num">{stage.n_episodes}</td>
                  <td className="num">{num(stage.mean_difficulty, 3)}</td>
                  <td className="num">{num(stage.min_difficulty, 3)}</td>
                  <td className="num">{num(stage.max_difficulty, 3)}</td>
                  <td className="num">{metres(stage.mean_path_length, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="band band--tight" aria-labelledby="wb-redundancy">
          <p className="eyebrow" id="wb-redundancy">
            Near-duplicate pairs · closest 5% of all distances
          </p>
          {pairs === null ? (
            <p className="unit">Needs the server.</p>
          ) : pairs.length === 0 ? (
            <p className="prose">No pair falls inside the near-duplicate threshold.</p>
          ) : (
            <table className="ledger">
              <thead>
                <tr>
                  <th scope="col">episode A</th>
                  <th scope="col">episode B</th>
                  <th scope="col" className="num">
                    DTW
                  </th>
                </tr>
              </thead>
              <tbody>
                {pairs.slice(0, 20).map((pair) => (
                  <tr key={`${pair.a}-${pair.b}`}>
                    <td>{shortId(pair.a)}</td>
                    <td>{shortId(pair.b)}</td>
                    <td className="num">{num(pair.distance, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {pairs && pairs.length > 20 && (
            <p className="unit">Showing the 20 closest of {pairs.length} reported pairs.</p>
          )}
        </section>

        <section className="band band--tight" aria-labelledby="wb-episodes">
          <p className="eyebrow" id="wb-episodes">
            Episodes · curriculum order
          </p>
          <table className="ledger">
            <thead>
              <tr>
                <th scope="col" className="num">
                  rank
                </th>
                <th scope="col">episode</th>
                <th scope="col">task</th>
                <th scope="col" className="num">
                  group
                </th>
                <th scope="col" className="num">
                  stage
                </th>
                <th scope="col" className="num">
                  difficulty
                </th>
                <th scope="col" className="num">
                  frames
                </th>
                <th scope="col" className="num">
                  coreset
                </th>
              </tr>
            </thead>
            <tbody>
              {[...snapshot.episodes]
                .sort((a, b) => (a.curriculum_rank ?? 0) - (b.curriculum_rank ?? 0))
                .map((episode) => (
                  <tr
                    key={episode.episode_id}
                    data-primary={episode.episode_id === selected}
                    onClick={() => setSelected(episode.episode_id)}
                  >
                    <td className="num">{episode.curriculum_rank ?? '—'}</td>
                    <td>{shortId(episode.episode_id)}</td>
                    <td>{episode.task_name.replace(/_/g, ' ')}</td>
                    <td className="num">{episode.cluster}</td>
                    <td className="num">{episode.stage ?? '—'}</td>
                    <td className="num">{num(episode.difficulty, 3)}</td>
                    <td className="num">{int(episode.n_frames)}</td>
                    <td className="num">{episode.coreset_rank ?? '—'}</td>
                  </tr>
                ))}
            </tbody>
          </table>
          <p className="unit">
            Coreset rank is a farthest-point traversal: truncate at any K for a near-maximally
            diverse K-episode subset.
          </p>
        </section>
      </div>
    </div>
  )
}
