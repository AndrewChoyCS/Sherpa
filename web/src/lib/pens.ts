/**
 * The pen carousel.
 *
 * A plotter holds a fixed set of pens and swaps between them; the log records
 * every swap. Here one pen is one data source, so colour answers "where did
 * this motion come from" without a legend lookup. Sources beyond the carousel
 * fall back to graphite rather than inventing a fifth pen.
 */

export const PEN_ORDER = ['yam', 'scale', 'aria', 'mecka'] as const

const PEN_HEX: Record<string, string> = {
  yam: '#1f4fa8',
  scale: '#b23a2e',
  aria: '#1e7a5a',
  mecka: '#63409a',
}

const GRAPHITE = '#5d666b'

/** Hex for a source, for canvas contexts that cannot read CSS custom properties. */
export function penHex(source: string): string {
  return PEN_HEX[source] ?? GRAPHITE
}

/** `var(--pen-*)` for a source, for SVG and DOM. */
export function penVar(source: string): string {
  return source in PEN_HEX ? `var(--pen-${penKey(source)})` : 'var(--ink-45)'
}

function penKey(source: string): string {
  switch (source) {
    case 'yam':
      return 'blue'
    case 'scale':
      return 'red'
    case 'aria':
      return 'green'
    case 'mecka':
      return 'violet'
    default:
      return 'blue'
  }
}

/** 1-based carousel slot, or null when the source is off-carousel. */
export function penNumber(source: string): number | null {
  const i = PEN_ORDER.indexOf(source as (typeof PEN_ORDER)[number])
  return i === -1 ? null : i + 1
}

/** `PEN 2` / `PEN —`, as the margin log prints it. */
export function penLabel(source: string): string {
  const n = penNumber(source)
  return n === null ? 'PEN —' : `PEN ${n}`
}

/**
 * Pens actually loaded for a dataset, in carousel order, with off-carousel
 * sources appended. Drives the hero's pen list, so it reflects the data rather
 * than a hardcoded four.
 */
export function loadedPens(sources: string[]): { source: string; hex: string; slot: number | null }[] {
  const present = new Set(sources)
  const onCarousel = PEN_ORDER.filter((s) => present.has(s)).map((s) => ({
    source: s as string,
    hex: penHex(s),
    slot: penNumber(s),
  }))
  const offCarousel = sources
    .filter((s) => !PEN_ORDER.includes(s as (typeof PEN_ORDER)[number]))
    .sort()
    .map((s) => ({ source: s, hex: GRAPHITE, slot: null }))
  return [...onCarousel, ...offCarousel]
}

export const ROUTE_HEX = '#c07f00'
export const INK_HEX = '#14191c'
export const RULE_HEX = '#b9c6cc'
