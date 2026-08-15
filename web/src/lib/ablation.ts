/**
 * The difficulty-ablation payload.
 *
 * Published by `scripts/export_snapshot.py --ablation`, which re-serialises it to be
 * browser-safe — the source file carries bare `NaN` p-values (undefined when all 40
 * goals tie), which `JSON.parse` rejects outright.
 *
 * Absent payload means the section simply does not render, so the page works with an
 * empty `web/public/`.
 */

import type { AblationPayload } from '../viz/DifficultyAblation'

export async function loadAblation(): Promise<AblationPayload | null> {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}ablation.json`)
    if (!response.ok) return null
    return (await response.json()) as AblationPayload
  } catch {
    return null
  }
}
