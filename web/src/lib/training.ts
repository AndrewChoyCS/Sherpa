/**
 * The trained-policy result.
 *
 * Absent payload hides the section, so the page works with an empty `web/public/`.
 */

import type { TrainingCurvesPayload } from '../viz/TrainingCurves'

export async function loadTraining(): Promise<TrainingCurvesPayload | null> {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}training_curves.json`)
    if (!response.ok) return null
    return (await response.json()) as TrainingCurvesPayload
  } catch {
    return null
  }
}
