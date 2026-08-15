/**
 * Number and label formatting.
 *
 * One rule throughout: a missing number renders as an em dash, never as 0 and
 * never as "NaN". Step 1 of a curriculum has no incoming edge weight and a
 * review step has no meaningful ramp cost, so blank is the correct reading —
 * printing 0.00 there would assert a measurement that was never taken.
 */

const DASH = '—'

export function num(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH
  return value.toFixed(digits)
}

export function pct(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH
  return `${(value * 100).toFixed(digits)}%`
}

export function metres(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH
  return `${value.toFixed(digits)} m`
}

export function int(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH
  return Math.round(value).toLocaleString('en-US')
}

/**
 * Episode IDs run to 45 characters and are mostly timestamp. The source prefix
 * and the seconds-level tail are what distinguish two episodes of the same
 * task, so both ends are kept and the middle is elided.
 */
export function shortId(episodeId: string, tail = 6): string {
  const [head, ...rest] = episodeId.split('__')
  if (rest.length === 0) return episodeId
  const last = rest[rest.length - 1]
  const task = rest.length > 1 ? rest.slice(0, -1).join('__') : null
  const stamp = last.length > tail ? `…${last.slice(-tail)}` : last
  return task ? `${head}/${task}/${stamp}` : `${head}/${stamp}`
}

/** Frame count as the plotter log prints it. */
export function frames(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return `${DASH} fr`
  return `${Math.round(n).toLocaleString('en-US')} fr`
}

export function titleCase(text: string): string {
  return text.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

export const EM_DASH = DASH
