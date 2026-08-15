/**
 * The JSON contract between server/api.py and this frontend.
 *
 * Every numeric field that Python can emit as NaN is typed `number | null`
 * here, because server/serialize.py maps NaN and +/-Inf to null. That is not
 * defensive noise: `CurriculumPath.table` genuinely has no edge weight on step
 * 1 (nothing precedes it) and deliberately blanks `ramp_cost` on review rows,
 * so null is the honest value and the UI renders it as an em dash.
 */

export interface RunConfig {
  data_dir: string
  arm: 'auto' | 'left' | 'right' | 'both'
  min_length: number
  normalize: 'center' | 'zscore' | 'none'
  max_length: number | null
  length_normalize: boolean
  sakoe_chiba_radius: number | null
  linkage: string
  difficulty_scaling: 'rank' | 'minmax'
  n_clusters: number | null
}

export interface DiversityMetrics {
  n_trajectories: number
  n_pairs: number
  diversity_score: number
  median_pairwise: number
  std_pairwise: number
  min_pairwise: number
  max_pairwise: number
  dispersion: number
  mean_nn_distance: number
  redundancy_ratio: number
  silhouette: number | null
  cluster_balance: number | null
  /**
   * 99th percentile pairwise distance over the median. Above ~2 the matrix is
   * outlier-dominated and clustering will collapse into one blob plus singletons —
   * the one number that separates "clusters genuinely overlap" from "clustering is
   * broken". Note silhouette is *anti*-correlated with correctness across
   * normalisation modes, so it cannot play this role.
   */
  tail_ratio?: number | null
}

export interface EpisodeRow {
  episode_id: string
  UMAP_X: number
  UMAP_Y: number
  UMAP_Z: number
  cluster: number
  cluster_label: string
  n_frames: number
  source: string
  task_name: string
  embodiment: string
  arm_used: string
  missing_frame_ratio: number | null
  task_description: string
  difficulty: number | null
  difficulty_z: number | null
  stage: number | null
  curriculum_rank: number | null
  coreset_rank: number | null
  is_cluster_medoid: boolean | null
}

/** A decimated end-effector path for the hero plot. */
export interface TrajectoryPreview {
  episode_id: string
  source: string
  task_name: string
  /** True frame count, before decimation — what the margin log reports. */
  n_frames: number
  fps: number
  /** Workspace span in metres; drives the shared scale. */
  span: number
  path_length: number
  /** Centred XY in metres, decimated to <=240 points. */
  points: [number, number][]
}

export interface StageRow {
  stage: number
  n_episodes: number
  mean_difficulty: number
  min_difficulty: number
  max_difficulty: number
  mean_path_length: number
  mean_tortuosity: number
}

export interface SkippedRow {
  episode_id: string
  reason: string
}

export interface Snapshot {
  config: RunConfig
  diversity_metrics: DiversityMetrics
  /** Adjusted Rand Index per metadata field; `task_name` is the headline. */
  agreement: Record<string, number>
  /**
   * Episodes each ARI was computed over. Unlabelled episodes are excluded from the
   * score rather than lumped under "unknown", so the support count is what makes
   * the number comparable between runs.
   */
  agreement_support: Record<string, number>
  n_clusters: number
  suggested_k: number | null
  silhouette_by_k: Record<string, number>
  n_episodes: number
  n_skipped: number
  episodes: EpisodeRow[]
  stages: StageRow[]
  skipped: SkippedRow[]
  trajectories: TrajectoryPreview[]
  sources: string[]
  tasks: string[]
  /** Cluster x task_name counts, for the agreement block. */
  agreement_matrix: {
    clusters: number[]
    tasks: string[]
    counts: number[][]
    /** Episodes dropped from the table for carrying no task_name. */
    excluded: number
  }
  /**
   * Head-to-head subset scores. Empty when the dataset is too small (< 8 episodes)
   * for a subset comparison to mean anything.
   */
  comparison: SubsetComparison
}

// --------------------------------------------------------------------------
// subset comparison
// --------------------------------------------------------------------------
export interface SubsetScore {
  /** Selection strategy: coreset, random, stratified or redundant. */
  name: string
  metrics: Record<string, number>
  episode_ids: string[]
}

export interface SubsetBaseline {
  trials: number
  mean: number
  std: number
  p05: number
  p95: number
  /** Score of the strategy being tested against the random distribution. */
  candidate: number
  candidate_name: string
  /** Where the candidate falls within the random draws, 0-100. */
  percentile: number
  /** Standard deviations above the random mean; the effect size. */
  z_score: number
}

export interface SubsetComparison {
  /** Episodes per subset. Equal across subsets, since these metrics move with N. */
  subset_size: number
  n_tasks_total: number
  subsets: SubsetScore[]
  deltas: Record<string, unknown>[]
  curve: { method: string; subset_size: number; diversity_score: number }[]
  baseline: SubsetBaseline | null
  baseline_samples: number[]
}

// --------------------------------------------------------------------------
// graph
// --------------------------------------------------------------------------
export interface GraphNode {
  index: number
  episode_id: string
  x: number
  y: number
  difficulty: number
  cluster: number
  stage: number
  task_name: string
  source: string
  embodiment: string
  n_frames: number
}

export interface GraphEdge {
  from: number
  to: number
  weight: number
  ramp: number
  interference: number
  /** Near-duplicate penalty; without it the cheapest route repeats one clip. */
  redundancy: number
  dtw: number
  is_repair: boolean
}

export interface GraphPayload {
  nodes: GraphNode[]
  edges: GraphEdge[]
  start_clips: number[]
  /** Edges added only to restore reachability — surfaced, not hidden. */
  repairs: [string | number, number][]
  n_edges: number
  mean_out_degree: number
}

// --------------------------------------------------------------------------
// path
// --------------------------------------------------------------------------
export interface GoalCandidate {
  task_name: string
  score: number
  clip_index: number
  episode_id: string
  task_description: string
  n_clips: number
}

export interface GoalMatch {
  query: string
  target_index: number
  task_name: string
  score: number
  margin: number
  is_confident: boolean
  note: string
  candidates: GoalCandidate[]
}

export interface PathStep {
  step: number
  clip_index: number
  episode_id: string
  is_review: boolean
  reviews_step: number | null
  task_name: string
  task_description: string
  stage: number | null
  cluster: number
  difficulty: number | null
  difficulty_z: number | null
  source: string
  embodiment: string
  n_frames: number
  edge_weight: number | null
  ramp_cost: number | null
  interference_cost: number | null
  dtw_from_prev: number | null
  difficulty_delta: number | null
  on_graph_edge: boolean
  task_switch: boolean
  cluster_switch: boolean
  embodiment_switch: boolean
  source_switch: boolean
}

export interface PathPayload {
  match: GoalMatch
  steps: PathStep[]
  route: number[]
  target_index: number
  search_cost: number
  cost_terms: Record<string, number>
  method: string
  n_reviews: number
  report: Record<string, number | null>
  /** Row key -> metric -> value; rows are the path and its baselines. */
  comparison: Record<string, Record<string, number | null>>
  /** Cumulative distinct skill families seen, by step. */
  coverage_curve: number[]
}

export interface RedundantPair {
  a: string
  b: string
  distance: number
}
