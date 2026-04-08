import { useEffect as use_effect, useState as use_state } from 'react'
import './app.css'
import { get_catalog, get_episode, get_preview } from './api_client'
import type { circuit_task_spec, ui_catalog_response, ui_playback_payload } from './ui_types'
import { clamp, format_capacitance, format_frequency, format_resistance, task_title } from './ui_helpers'
import { baselines_card } from './components/baselines_card'
import { chart_card } from './components/chart_card'
import { controls_panel } from './components/controls_panel'
import { score_card } from './components/score_card'
import { story_card } from './components/story_card'
import { telemetry_card } from './components/telemetry_card'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { AlertTriangle, CircuitBoard } from 'lucide-react'

type loading_stage = 'catalog' | 'preview' | 'episode' | null

export function app() {
  const [catalog, set_catalog] = use_state<ui_catalog_response | null>(null)
  const [selected_task_id, set_selected_task_id] = use_state('')
  const [payload, set_payload] = use_state<ui_playback_payload | null>(null)
  const [loading_stage, set_loading_stage] = use_state<loading_stage>('catalog')
  const [active_frame_index, set_active_frame_index] = use_state(0)
  const [playback_seconds, set_playback_seconds] = use_state(0.35)
  const [is_playing, set_is_playing] = use_state(false)
  const [run_completed, set_run_completed] = use_state(false)
  const [error, set_error] = use_state<string | null>(null)

  use_effect(() => {
    const controller = new AbortController()

    async function load_catalog() {
      try {
        set_loading_stage('catalog')
        const next_catalog = await get_catalog(controller.signal)
        set_catalog(next_catalog)
        set_selected_task_id(next_catalog.default_task_id)
        set_error(null)
      } catch (caught_error) {
        if (!controller.signal.aborted) {
          set_error(get_error_message(caught_error))
        }
      } finally {
        if (!controller.signal.aborted) {
          set_loading_stage(null)
        }
      }
    }

    void load_catalog()

    return () => {
      controller.abort()
    }
  }, [])

  use_effect(() => {
    if (!selected_task_id) {
      return
    }

    const controller = new AbortController()

    async function load_preview() {
      try {
        set_loading_stage('preview')
        set_is_playing(false)
        set_run_completed(false)
        set_active_frame_index(0)
        set_error(null)
        const next_preview = await get_preview(selected_task_id, controller.signal)
        set_payload(next_preview)
      } catch (caught_error) {
        if (!controller.signal.aborted) {
          set_error(get_error_message(caught_error))
        }
      } finally {
        if (!controller.signal.aborted) {
          set_loading_stage(null)
        }
      }
    }

    void load_preview()

    return () => {
      controller.abort()
    }
  }, [selected_task_id])

  use_effect(() => {
    if (!is_playing || !payload) {
      return
    }

    const last_frame_index = payload.frames.length - 1
    if (active_frame_index >= last_frame_index) {
      set_is_playing(false)
      set_run_completed(true)
      return
    }

    const timeout_id = window.setTimeout(() => {
      set_active_frame_index((current_index) => {
        const next_index = Math.min(current_index + 1, last_frame_index)

        if (next_index >= last_frame_index) {
          set_is_playing(false)
          set_run_completed(true)
        }

        return next_index
      })
    }, playback_seconds * 1000)

    return () => {
      window.clearTimeout(timeout_id)
    }
  }, [active_frame_index, is_playing, payload, playback_seconds])

  async function handle_run_episode() {
    if (!selected_task_id) {
      return
    }

    try {
      set_loading_stage('episode')
      set_error(null)
      set_is_playing(false)
      set_run_completed(false)
      set_active_frame_index(0)
      const episode_payload = await get_episode(selected_task_id)
      set_payload(episode_payload)

      if (episode_payload.frames.length > 1) {
        set_is_playing(true)
      } else {
        set_run_completed(true)
      }
    } catch (caught_error) {
      set_error(get_error_message(caught_error))
    } finally {
      set_loading_stage(null)
    }
  }

  function handle_scrub_step(next_index: number) {
    if (!payload) {
      return
    }

    const safe_index = clamp(next_index, 0, payload.frames.length - 1)
    set_is_playing(false)
    set_active_frame_index(safe_index)

    if (safe_index === payload.frames.length - 1 && payload.summary) {
      set_run_completed(true)
    }
  }

  const selected_task: circuit_task_spec | null =
    payload?.task ??
    catalog?.tasks.find((task) => task.task_id === selected_task_id) ??
    null
  const active_frame = payload
    ? payload.frames[Math.min(active_frame_index, payload.frames.length - 1)]
    : null
  const step_range_max = payload ? Math.max(payload.frames.length - 1, 0) : 0
  const is_busy = loading_stage !== null || is_playing
  const show_summary = Boolean(run_completed && payload?.summary)
  const show_comparisons = Boolean(run_completed && payload?.comparisons.length)

  const status_label = is_playing
    ? 'playing'
    : loading_stage === 'episode'
      ? 'running episode'
      : loading_stage === 'preview'
        ? 'loading preview'
        : loading_stage === 'catalog'
          ? 'loading catalog'
          : 'ready'

  return (
    <div className="min-h-screen bg-surface text-ink">
      <header className="sticky top-0 z-10 flex items-center justify-between gap-4 border-b border-line bg-surface/95 px-6 py-3 backdrop-blur-sm">
        <div className="flex min-w-0 items-center gap-3">
          <span className="grid h-7 w-7 place-items-center rounded-md border border-line-strong bg-panel text-brand">
            <CircuitBoard className="size-4" aria-hidden="true" />
          </span>
          <span className="text-[0.92rem] font-semibold tracking-tight text-ink">CircuitRL</span>
          <span className="ml-2 hidden border-l border-line pl-3 text-[0.78rem] text-muted-foreground sm:inline">
            deterministic RC tuning benchmark console
          </span>
        </div>
        <Badge variant={is_busy ? 'warn' : 'ok'} aria-live="polite">
          <span
            className={
              'size-1.5 rounded-full ' +
              (is_busy
                ? 'bg-warn shadow-[0_0_0_3px_rgba(214,163,107,0.18)] animate_pulse_dot'
                : 'bg-ok shadow-[0_0_0_3px_rgba(127,179,157,0.16)]')
            }
            aria-hidden="true"
          />
          {status_label}
        </Badge>
      </header>

      <main
        className="mx-auto grid max-w-[1480px] items-start md:grid-cols-[17rem_minmax(0,1fr)]"
        aria-busy={is_busy}
      >
        {controls_panel({
          catalog,
          is_busy,
          is_playing,
          loading_stage,
          playback_seconds,
          selected_task,
          selected_task_id,
          active_frame_index,
          step_range_max,
          on_playback_seconds_change: set_playback_seconds,
          on_run_episode: handle_run_episode,
          on_scrub_step: handle_scrub_step,
          on_task_change: set_selected_task_id,
        })}

        <div className="grid min-w-0 gap-4 px-6 pb-12 pt-5">
          {error ? (
            <Alert variant="destructive">
              <AlertTriangle className="size-4" />
              <AlertTitle>Failed to load fresh benchmark data.</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}

          {task_header({ task: selected_task, action_scale_factor: catalog?.action_scale_factor ?? 1.2 })}

          {telemetry_card({ frame: active_frame, task: selected_task })}

          <div className="grid gap-4 lg:grid-cols-[minmax(0,1.55fr)_minmax(0,1fr)]">
            {chart_card({ active_frame_index, payload })}
            {story_card({ active_frame_index, payload })}
          </div>

          {score_card({
            summary: show_summary ? payload?.summary ?? null : null,
            task: selected_task,
          })}
          {baselines_card({
            comparisons: show_comparisons ? payload?.comparisons ?? [] : [],
          })}
        </div>
      </main>
    </div>
  )
}

interface task_header_props {
  task: circuit_task_spec | null
  action_scale_factor: number
}

function task_header({ task, action_scale_factor }: task_header_props) {
  if (!task) {
    return (
      <section className="flex items-start justify-between gap-4 border-b border-line pb-4">
        <div>
          <h1 className="m-0 mb-1 text-[1.25rem] font-semibold tracking-tight text-ink">Loading task…</h1>
          <div className="text-[0.78rem] text-muted">Waiting for the deterministic catalog.</div>
        </div>
      </section>
    )
  }

  return (
    <section className="flex items-start justify-between gap-4 border-b border-line pb-4">
      <div className="min-w-0">
        <h1 className="m-0 mb-1.5 text-[1.25rem] font-semibold tracking-tight text-ink">
          {task_title(task)}
        </h1>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[0.78rem] text-muted">
          <span>target <strong className="font-mono font-medium text-ink-dim">{format_frequency(task.target_hz)}</strong></span>
          <span>tolerance <strong className="font-mono font-medium text-ink-dim">±{task.success_tolerance_pct.toFixed(1)}%</strong></span>
          <span>budget <strong className="font-mono font-medium text-ink-dim">{task.max_steps} steps</strong></span>
          <span>action scale <strong className="font-mono font-medium text-ink-dim">×{action_scale_factor.toFixed(1)}</strong></span>
          <span>start <strong className="font-mono font-medium text-ink-dim">{format_resistance(task.initial_r_ohms)} · {format_capacitance(task.initial_c_farads)}</strong></span>
        </div>
      </div>
      <span className="whitespace-nowrap rounded border border-line-strong px-2 py-1 font-mono text-[0.74rem] text-muted">
        {task.task_id}
      </span>
    </section>
  )
}

function get_error_message(caught_error: unknown): string {
  if (caught_error instanceof Error) {
    return caught_error.message
  }

  return 'Unexpected error while communicating with the CircuitRL backend.'
}
