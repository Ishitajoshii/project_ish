import { empty_panel } from './empty_panel'
import type { circuit_task_spec, playback_frame } from '../ui_types'
import {
  format_capacitance,
  format_frequency,
  format_percent,
  format_resistance,
  format_score,
  get_signal_tone,
  normalize_log_value,
} from '../ui_helpers'
import { Card, CardHeader, CardKicker, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { cn } from '@/lib/utils'

interface telemetry_card_props {
  frame: playback_frame | null
  task: circuit_task_spec | null
}

export function telemetry_card({ frame, task }: telemetry_card_props) {
  if (!frame || !task) {
    return empty_panel({
      title: 'Live telemetry',
      body: 'Telemetry will populate once a task is loaded. Pick a task or press Run episode to start the playback.',
    })
  }

  const tone = get_signal_tone(task, frame)
  const closeness = Math.max(0, 1 - frame.normalized_error)
  const step_ratio = frame.step / Math.max(task.max_steps, 1)
  const r_position = normalize_log_value(frame.current_r_ohms, task.min_r_ohms, task.max_r_ohms)
  const c_position = normalize_log_value(frame.current_c_farads, task.min_c_farads, task.max_c_farads)
  const reward_value = frame.reward === null ? 'Staged' : format_score(frame.reward)

  const progress_tone =
    tone.tone_class_name === 'ok' ? 'ok' : tone.tone_class_name === 'warn' ? 'warn' : 'ok'

  return (
    <Card>
      <CardHeader>
        <div>
          <CardKicker>Live telemetry</CardKicker>
          <CardTitle className="mt-1">
            Step {frame.step} of {task.max_steps}
            <span className="ml-2 font-mono text-[0.78rem] font-normal text-muted-foreground">
              {frame.action_label}
            </span>
          </CardTitle>
        </div>
        <Badge variant={tone.tone_class_name as 'ok' | 'warn' | 'soft'}>{tone.tone_label}</Badge>
      </CardHeader>

      <div className="grid grid-cols-1 border-t border-line sm:grid-cols-2 lg:grid-cols-4">
        <TelemetryCell
          label="Resistor"
          value={format_resistance(frame.current_r_ohms)}
          meta={`range ${format_percent(r_position)}`}
          className="border-b border-line lg:border-b-0 lg:border-r"
        />
        <TelemetryCell
          label="Capacitor"
          value={format_capacitance(frame.current_c_farads)}
          meta={`range ${format_percent(c_position)}`}
          className="border-b border-line sm:border-r lg:border-b-0"
        />
        <TelemetryCell
          label="Target vs achieved"
          className="border-b border-line lg:border-b-0 lg:border-r"
        >
          <div className="flex items-baseline gap-2 font-mono text-ink">
            <span className="text-[1.05rem]">{format_frequency(frame.current_hz)}</span>
            <span className="text-[0.74rem] text-muted-foreground">
              / {format_frequency(task.target_hz)}
            </span>
          </div>
          <p className="mt-1 font-mono text-[0.72rem] text-muted-foreground">
            Δ {format_frequency(frame.delta_hz)} · err {format_percent(frame.normalized_error)}
          </p>
          <Progress value={closeness * 100} tone={progress_tone} className="mt-2" />
        </TelemetryCell>
        <TelemetryCell
          label="Reward / budget"
          value={reward_value}
          meta={`best ${format_score(frame.best_score)} · used ${format_percent(step_ratio)} · ${frame.remaining_steps} left`}
        />
      </div>
    </Card>
  )
}

interface telemetry_cell_props {
  label: string
  value?: string
  meta?: string
  className?: string
  children?: React.ReactNode
}

function TelemetryCell({ label, value, meta, className, children }: telemetry_cell_props) {
  return (
    <div className={cn('min-w-0 px-4 py-3.5', className)}>
      <div className="mb-1.5 text-[0.66rem] font-semibold uppercase tracking-[0.09em] text-muted-foreground">
        {label}
      </div>
      {children ?? (
        <div className="font-mono text-[1.05rem] leading-tight text-ink">{value}</div>
      )}
      {meta ? (
        <p className="mt-1 font-mono text-[0.72rem] leading-snug text-muted-foreground">{meta}</p>
      ) : null}
    </div>
  )
}
