export interface circuit_task_spec {
  task_id: string
  circuit_type: 'low_pass' | 'high_pass'
  target_hz: number
  initial_r_ohms: number
  initial_c_farads: number
  min_r_ohms: number
  max_r_ohms: number
  min_c_farads: number
  max_c_farads: number
  max_steps: number
  success_tolerance_pct: number
  cost_weight: number
  step_weight: number
}

export interface playback_frame {
  step: number
  action: string
  action_label: string
  action_symbol: string
  reward: number | null
  best_score: number
  note: string
  current_r_ohms: number
  current_c_farads: number
  current_hz: number
  normalized_error: number
  current_cost: number
  remaining_steps: number
  delta_hz: number
  within_tolerance: boolean
}

export interface episode_summary {
  score: number
  success: boolean
  steps_used: number
  achieved_hz: number
  best_error: number
  best_cost: number
  best_r_ohms: number
  best_c_farads: number
}

export interface baseline_comparison {
  baseline_name: string
  task_id: string
  label: string
  score: number
  success: boolean
  steps_used: number
  evaluations: number | null
  achieved_hz: number
  current_r_ohms: number
  current_c_farads: number
  normalized_error: number
  normalized_cost: number
}

export interface ui_playback_payload {
  task: circuit_task_spec
  frames: playback_frame[]
  summary: episode_summary | null
  comparisons: baseline_comparison[]
}

export interface ui_catalog_response {
  default_task_id: string
  task_ids: string[]
  tasks: circuit_task_spec[]
  action_scale_factor: number
}
