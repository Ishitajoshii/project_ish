import { empty_panel } from './empty_panel'
import type { ui_playback_payload } from '../ui_types'
import { format_percent } from '../ui_helpers'
import { Card, CardContent, CardHeader, CardKicker, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

interface story_card_props {
  active_frame_index: number
  payload: ui_playback_payload | null
}

export function story_card({ active_frame_index, payload }: story_card_props) {
  if (!payload) {
    return empty_panel({
      title: 'Step playback',
      body: 'Each step committed during playback will be explained here, with a scrubbable action timeline.',
    })
  }

  const frame = payload.frames[Math.min(active_frame_index, payload.frames.length - 1)]
  const step_intro =
    frame.step === 0
      ? 'Controller staged. No action committed yet — every legal move is still scorable.'
      : `Step ${frame.step} — ${frame.action_label} reshapes the RC product.`
  const best_score = Math.max(...payload.frames.map((sample) => sample.best_score))

  return (
    <Card>
      <CardHeader>
        <div>
          <CardKicker>Step playback</CardKicker>
          <CardTitle className="mt-1">Narrative trace</CardTitle>
        </div>
        <Badge variant="outline" className="normal-case">
          {payload.task.circuit_type === 'low_pass' ? 'low-pass' : 'high-pass'}
        </Badge>
      </CardHeader>

      <CardContent className="grid gap-3">
        <p className="m-0 font-mono text-[0.72rem] text-muted-foreground">step {frame.step}</p>
        <h4 className="m-0 text-[0.92rem] font-semibold leading-snug text-ink">{step_intro}</h4>
        <p className="m-0 text-[0.84rem] leading-relaxed text-ink-dim">{frame.note}</p>

        <div className="mt-1 flex flex-wrap gap-1.5 border-t border-line pt-3">
          {payload.frames.slice(0, active_frame_index + 1).map((sample) => {
            const is_active = sample.step === active_frame_index
            const is_best = sample.best_score === best_score && sample.step !== 0
            return (
              <span
                key={`${sample.step}-${sample.action}`}
                className={cn(
                  'inline-flex cursor-default items-center gap-1 rounded border border-line-strong bg-inset px-1.5 py-1 font-mono text-[0.7rem] text-muted-foreground',
                  is_active && 'border-brand bg-brand/10 text-ink',
                  is_best && !is_active && 'border-ok/40',
                )}
              >
                <strong
                  className={cn(
                    'font-semibold text-ink-dim',
                    is_active && 'text-brand',
                    is_best && 'text-ok',
                  )}
                >
                  {sample.action_symbol}
                </strong>
                <span>
                  s{sample.step} · {format_percent(sample.normalized_error)}
                </span>
              </span>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
