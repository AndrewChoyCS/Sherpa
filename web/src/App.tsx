import { useEffect, useState } from 'react'
import { Sherpa } from './views/Sherpa'
import { Workbench } from './views/Workbench'
import { loadSnapshot } from './lib/api'
import type { Snapshot } from './lib/types'

type View = 'route' | 'workbench'

export default function App() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [view, setView] = useState<View>(() =>
    new URLSearchParams(window.location.search).get('view') === 'workbench' ? 'workbench' : 'route',
  )

  const show = (next: View) => {
    setView(next)
    const url = new URL(window.location.href)
    if (next === 'route') url.searchParams.delete('view')
    else url.searchParams.set('view', next)
    window.history.replaceState(null, '', url)
  }

  useEffect(() => {
    loadSnapshot()
      .then(({ snapshot: loaded }) => setSnapshot(loaded))
      .catch((thrown) =>
        setError(thrown instanceof Error ? thrown.message : 'Could not load a snapshot.'),
      )
  }, [])

  return (
    <>
      {/* Atmosphere sits behind everything and never moves with the content. */}
      <div className="sky" aria-hidden="true" />
      <div className="grain" aria-hidden="true" />

      {error ? (
        <main className="boot">
          <div className="panel stack--tight">
            <p className="label">No data loaded</p>
            <p className="prose">{error}</p>
            <p className="prose">
              Export a snapshot with <code>python scripts/export_snapshot.py</code>, or start the
              server with <code>uvicorn server.api:app --port 8000</code>.
            </p>
          </div>
        </main>
      ) : !snapshot ? (
        // The snapshot is a local file and resolves in milliseconds; a skeleton
        // would flash and read as noise.
        <main className="boot">
          <p className="label">Loading terrain…</p>
        </main>
      ) : (
        <>
          <header className="topbar">
            <p className="topbar__mark">Sherpa</p>
            <nav className="topbar__nav">
              <button
                type="button"
                className="tab"
                aria-current={view === 'route'}
                onClick={() => show('route')}
              >
                Route
              </button>
              <button
                type="button"
                className="tab"
                aria-current={view === 'workbench'}
                onClick={() => show('workbench')}
              >
                Workbench
              </button>
            </nav>
            <p className="topbar__meta unit">{snapshot.n_episodes} clips</p>
          </header>
          <main>
            {view === 'route' ? (
              <Sherpa snapshot={snapshot} onOpenWorkbench={() => show('workbench')} />
            ) : (
              <Workbench initial={snapshot} />
            )}
          </main>
        </>
      )}
    </>
  )
}
