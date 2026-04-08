import type { circuit_task_spec, ui_catalog_response } from '../ui_types'
import { format_frequency, task_title } from '../ui_helpers'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Separator } from '@/components/ui/separator'

type loading_stage = 'catalog' | 'preview' | 'episode' | null

interface controls_panel_props {
  catalog: ui_catalog_response | null
  is_busy: boolean
  is_playing: boolean
  loading_stage: loading_stage
  playback_seconds: number
  selected_task: circuit_task_spec | null
  selected_task_id: string
  active_frame_index: number
  step_range_max: number
  on_playback_seconds_change: (value: number) => void
  on_run_episode: () => void
  on_scrub_step: (value: number) => void
  on_task_change: (task_id: string) => void
}

export function controls_panel({
  catalog,
  is_busy,
  is_playing,
  loading_stage,
  playback_seconds,
  selected_task,
  selected_task_id,
  active_frame_index,
  step_range_max,
  on_playback_seconds_change,
  on_run_episode,
  on_scrub_step,
  on_task_change,
}: controls_panel_props) {
  const run_label =
    loading_stage === 'episode' ? 'Running…' : is_playing ? 'Playing' : 'Run episode'
  const current_step = Math.min(active_frame_index, step_range_max)

  return (
    <aside
      aria-label="Benchmark controls"
      className="sticky top-[3.4rem] hidden h-[calc(100vh-3.4rem)] self-start overflow-y-auto border-r border-line p-5 md:block"
    >
      <section className="mb-6 grid gap-2">
        <Label htmlFor="task_select">Task</Label>
        <Select
          value={selected_task_id || undefined}
          onValueChange={on_task_change}
          disabled={!catalog || is_busy}
        >
          <SelectTrigger id="task_select">
            <SelectValue placeholder="Select a task" />
          </SelectTrigger>
          <SelectContent>
            {catalog?.task_ids.map((task_id) => (
              <SelectItem key={task_id} value={task_id}>
                {task_id}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-[0.74rem] leading-snug text-muted-foreground">
          {selected_task ? task_title(selected_task) : 'Loading deterministic catalog…'}
        </p>
      </section>

      <Button
        className="mb-6 w-full"
        size="lg"
        onClick={on_run_episode}
        disabled={!selected_task_id || is_busy}
      >
        {run_label}
      </Button>

      <section className="mb-6 grid gap-3">
        <div className="flex items-baseline justify-between gap-2">
          <Label htmlFor="cadence">Playback cadence</Label>
          <span className="font-mono text-[0.74rem] text-ink-dim">
            {playback_seconds.toFixed(2)}s / step
          </span>
        </div>
        <Slider
          id="cadence"
          min={0.15}
          max={0.95}
          step={0.05}
          value={[playback_seconds]}
          onValueChange={(values) => on_playback_seconds_change(values[0] ?? playback_seconds)}
          disabled={is_busy}
        />
      </section>

      <section className="mb-6 grid gap-3">
        <div className="flex items-baseline justify-between gap-2">
          <Label htmlFor="step_scrub">Episode step</Label>
          <span className="font-mono text-[0.74rem] text-ink-dim">
            {step_range_max > 0 ? `${current_step} / ${step_range_max}` : '—'}
          </span>
        </div>
        <Slider
          id="step_scrub"
          min={0}
          max={Math.max(step_range_max, 0)}
          step={1}
          value={[current_step]}
          onValueChange={(values) => on_scrub_step(values[0] ?? 0)}
          disabled={step_range_max === 0}
        />
        <p className="text-[0.74rem] leading-snug text-muted-foreground">
          Drag after a run to inspect any visited frame without rerunning the backend.
        </p>
      </section>

      <Separator className="my-5" />

      <section className="grid gap-3">
        <Label>Selection</Label>
        <dl className="grid gap-1.5 font-mono text-[0.74rem] text-ink-dim">
          <MetaRow label="id" value={selected_task?.task_id ?? '—'} />
          <MetaRow
            label="filter"
            value={
              selected_task
                ? selected_task.circuit_type === 'low_pass'
                  ? 'low-pass'
                  : 'high-pass'
                : '—'
            }
          />
          <MetaRow
            label="target"
            value={selected_task ? format_frequency(selected_task.target_hz) : '—'}
          />
          <MetaRow
            label="tolerance"
            value={
              selected_task ? `±${selected_task.success_tolerance_pct.toFixed(1)}%` : '—'
            }
          />
          <MetaRow
            label="budget"
            value={selected_task ? `${selected_task.max_steps} steps` : '—'}
          />
        </dl>
      </section>
    </aside>
  )
}

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-2">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="m-0 truncate">{value}</dd>
    </div>
  )
}
