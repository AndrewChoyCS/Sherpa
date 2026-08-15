/**
 * What the loader refused, grouped by failure kind.
 *
 * Listing every rejection verbatim does not scale: at 60 rejections the reasons are
 * near-identical strings differing only in a per-axis standard deviation, and they
 * ran to thousands of pixels that buried the rest of the page. The reader's actual
 * questions are "how many, and for what reasons" — so the default view answers
 * those, with the full per-episode list kept one click away rather than removed.
 *
 * Kinds are parsed from the reason text rather than from a status code, because the
 * loader reports prose. That is a little fragile by nature, so anything unmatched
 * falls through to its own "other" group with the raw string intact instead of being
 * silently dropped.
 */

import { shortId } from '../lib/format'
import type { SkippedRow } from '../lib/types'

interface Kind {
  key: string
  label: string
  test: RegExp
}

const KINDS: Kind[] = [
  {
    key: 'sentinel',
    label: 'Pose stream all missing-data sentinels',
    test: /missing-data sentinel/i,
  },
  {
    key: 'static',
    label: 'End-effector static — pose stream never populated',
    test: /is static/i,
  },
  { key: 'short', label: 'Too few valid frames', test: /too short|min_length|valid frame/i },
]

function kindsOf(reason: string): string[] {
  const found = KINDS.filter((kind) => kind.test.test(reason)).map((kind) => kind.key)
  return found.length > 0 ? found : ['other']
}

function labelFor(key: string): string {
  return KINDS.find((kind) => kind.key === key)?.label ?? 'Other'
}

export interface RejectionLedgerProps {
  skipped: SkippedRow[]
}

export function RejectionLedger({ skipped }: RejectionLedgerProps) {
  if (skipped.length === 0) {
    return <p className="prose">No episode was rejected.</p>
  }

  // An episode can fail on both arms for different reasons, so it can appear under
  // more than one kind; the counts therefore describe failures, not a partition.
  const groups = new Map<string, SkippedRow[]>()
  for (const row of skipped) {
    for (const key of kindsOf(row.reason)) {
      const bucket = groups.get(key) ?? []
      bucket.push(row)
      groups.set(key, bucket)
    }
  }

  const ordered = [...groups.entries()].sort((a, b) => b[1].length - a[1].length)

  return (
    <div className="stack stack--tight">
      <table className="ledger">
        <thead>
          <tr>
            <th scope="col" className="num">
              n
            </th>
            <th scope="col">failure</th>
            <th scope="col">example measurement</th>
          </tr>
        </thead>
        <tbody>
          {ordered.map(([key, rows]) => (
            <tr key={key}>
              <td className="num">{rows.length}</td>
              <td>{labelFor(key)}</td>
              <td className="reason">{firstMeasurement(rows[0].reason)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <details className="disclose">
        <summary>All {skipped.length} rejected episodes, with full reasons</summary>
        <table className="ledger">
          <thead>
            <tr>
              <th scope="col">episode</th>
              <th scope="col">reason</th>
            </tr>
          </thead>
          <tbody>
            {skipped.map((row) => (
              <tr key={row.episode_id}>
                <td>{shortId(row.episode_id)}</td>
                <td className="reason">{row.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  )
}

/**
 * The measured value that disqualified the episode, which is the part worth showing
 * — the surrounding prose is identical across every row in a group.
 */
function firstMeasurement(reason: string): string {
  const std = reason.match(/max per-axis std ([0-9.eE+-]+\s*m)/)
  if (std) return `max per-axis std ${std[1]}`
  const pct = reason.match(/(\d+)% of frames/)
  if (pct) return `${pct[1]}% of frames`
  return reason.length > 90 ? `${reason.slice(0, 90)}…` : reason
}
