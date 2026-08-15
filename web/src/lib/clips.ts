/**
 * The camera clips that have been fetched for the demo.
 *
 * Most episodes have no video and never will: `fetch_egoverse_data.py` downloads pose
 * keys only, which is what keeps a 273-episode dataset at 84 MB. A handful are pulled
 * deliberately by `scripts/fetch_clip_video.py` for the worked examples, and this
 * manifest is how the UI knows which. Anything absent falls back to the plotted path,
 * so the page works identically with an empty clips directory.
 */

export interface ClipEntry {
  src: string
  bytes: number
}

export type ClipManifest = Record<string, ClipEntry>

/** Loads the manifest, treating any failure as "no clips fetched". */
export async function loadClips(): Promise<ClipManifest> {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}clips/index.json`)
    if (!response.ok) return {}
    return (await response.json()) as ClipManifest
  } catch {
    return {}
  }
}

/** Resolves a manifest entry to a URL the browser can load. */
export function clipUrl(entry: ClipEntry): string {
  return `${import.meta.env.BASE_URL}${entry.src}`
}
