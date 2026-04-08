import { empty_panel } from './empty_panel'
import type { circuit_task_spec, episode_summary } from '../ui_types'
import {
  format_capacitance,
  format_frequency,
  format_percent,
  format_resistance,
  format_score,
} from '../ui_helpers'
import { Card, CardHeader, CardKicker, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface score_card_props {
  summary: episode_summary | null
  task: circuit_task_spec | null
}

export function score_card({ summary, task }: score_card_props) {
  if (!summary || !task) {
    return empty_panel({
      title: 'Final score',
      body: 'Run the episode to reveal the best score, achieved cutoff, and the strongest RC pair chosen along the trajectory.',
    })
  }

  return (
    <Card>
      <CardHeader>
        <div>
          <CardKicker>Final score</CardKicker>
          <CardTitle className="mt-1">Best state observed over the episode</CardTitle>
        </div>
        <Badge variant={summary.success ? 'ok' : 'warn'}>
          {summary.success ? 'Benchmark-clearing' : 'Below pass line'}
        </Badge>
      </CardHeader>

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <div className="border-b border-line p-5 lg:border-b-0 lg:border-r">
          <div className="font-mono text-[2.4rem] font-semibold leading-none tracking-tight text-ink">
            {format_score(summary.score)}
          </div>
          <p className="mt-2 font-mono text-[0.76rem] text-muted-foreground">
            {summary.steps_used} steps · best err {format_percent(summary.best_error)}
          </p>
          <p className="mt-1 font-mono text-[0.76rem] text-muted-foreground">
            target {format_frequency(task.target_hz)} → {format_frequency(summary.achieved_hz)}
          </p>
        </div>

        <dl className="grid grid-cols-1 sm:grid-cols-2">
          <ScoreKV
            label="Best resistor"
            value={format_resistance(summary.best_r_ohms)}
            className="border-b border-line sm:border-r"
          />
          <ScoreKV
            label="Best capacitor"
            value={format_capacitance(summary.best_c_farads)}
            className="border-b border-line"
          />
          <ScoreKV
            label="Normalized cost"
            value={format_percent(summary.best_cost)}
            className="sm:border-r"
          />
          <ScoreKV label="Achieved cutoff" value={format_frequency(summary.achieved_hz)} />
        </dl>
      </div>
    </Card>
  )
}

function ScoreKV({
  label,
  value,
  className,
}: {
  label: string
  value: string
  className?: string
}) {
  return (
    <div className={`min-w-0 px-4 py-3.5 ${className ?? ''}`}>
      <dt className="mb-1 text-[0.66rem] font-semibold uppercase tracking-[0.09em] text-muted-foreground">
        {label}
      </dt>
      <dd className="m-0 font-mono text-[1.02rem] text-ink">{value}</dd>
    </div>
  )
}
