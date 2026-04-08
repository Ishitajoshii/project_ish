import { empty_panel } from './empty_panel'
import type { ui_playback_payload } from '../ui_types'
import { clamp, format_percent } from '../ui_helpers'
import { Card, CardContent, CardHeader, CardKicker, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface chart_card_props {
  active_frame_index: number
  payload: ui_playback_payload | null
}

export function chart_card({ active_frame_index, payload }: chart_card_props) {
  if (!payload) {
    return empty_panel({
      title: 'Error vs step',
      body: 'The error trace draws as the playback advances. Run the episode to start plotting the trajectory.',
    })
  }

  const { frames, task } = payload
  const visible_frames = frames.slice(0, active_frame_index + 1)
  const width = 780
  const height = 320
  const left = 44
  const right = 16
  const top = 18
  const bottom = 34
  const plot_width = width - left - right
  const plot_height = height - top - bottom
  const error_values = visible_frames.map((frame) => frame.normalized_error * 100)
  const max_error =
    Math.max(Math.max(...error_values, 0), task.success_tolerance_pct, 6) * 1.15

  const x_for = (step: number) => left + (step / Math.max(task.max_steps, 1)) * plot_width
  const y_for = (error_percent: number) =>
    top + (1 - clamp(error_percent / max_error, 0, 1)) * plot_height

  const polyline_points = visible_frames
    .map(
      (frame) =>
        `${x_for(frame.step).toFixed(2)},${y_for(frame.normalized_error * 100).toFixed(2)}`,
    )
    .join(' ')
  const last_visible = visible_frames[visible_frames.length - 1]
  const area_points = `${left.toFixed(2)},${(height - bottom).toFixed(2)} ${polyline_points} ${x_for(
    last_visible.step,
  ).toFixed(2)},${(height - bottom).toFixed(2)}`
  const tolerance_y = y_for(task.success_tolerance_pct)
  const grid_lines = Array.from({ length: 5 }, (_, index) => {
    const value = (max_error / 4) * index
    return { value, y: y_for(value) }
  })
  const best_visible_score = Math.max(...visible_frames.map((frame) => frame.best_score))

  return (
    <Card className="chart_panel">
      <CardHeader>
        <div>
          <CardKicker>Error vs step</CardKicker>
          <CardTitle className="mt-1">Normalized error trace</CardTitle>
        </div>
        <Badge variant="default" className="normal-case">
          err {format_percent(last_visible.normalized_error)}
        </Badge>
      </CardHeader>
      <CardContent className="px-2 pb-3 pt-3">
        <svg
          className="block h-auto w-full"
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label="Normalized error versus step chart"
        >
          <rect
            x={left}
            y={top}
            width={plot_width}
            height={plot_height}
            rx="4"
            fill="rgba(255,255,255,0.015)"
            stroke="rgba(255,255,255,0.04)"
          />

          {grid_lines.map(({ value, y }) => (
            <g key={value}>
              <line
                x1={left}
                y1={y}
                x2={width - right}
                y2={y}
                stroke="rgba(255,255,255,0.055)"
                strokeWidth="1"
              />
              <text
                x={left - 8}
                y={y + 4}
                textAnchor="end"
                fill="#878a82"
                fontSize="10"
                fontFamily="IBM Plex Mono, monospace"
              >
                {value.toFixed(0)}%
              </text>
            </g>
          ))}

          <line
            x1={left}
            y1={tolerance_y}
            x2={width - right}
            y2={tolerance_y}
            stroke="#7fb39d"
            strokeWidth="1"
            strokeDasharray="4 4"
          />
          <text
            x={width - right - 4}
            y={tolerance_y - 6}
            textAnchor="end"
            fill="#7fb39d"
            fontSize="10"
            fontFamily="IBM Plex Mono, monospace"
          >
            tol {task.success_tolerance_pct.toFixed(1)}%
          </text>

          <polygon points={area_points} fill="rgba(201,139,95,0.1)" />
          <polyline
            points={polyline_points}
            fill="none"
            stroke="#c98b5f"
            strokeWidth="1.75"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {visible_frames.map((frame) => {
            const is_active = frame.step === active_frame_index
            const is_best = frame.best_score === best_visible_score && frame.step !== 0
            return (
              <circle
                key={`${frame.step}-${frame.action}`}
                cx={x_for(frame.step)}
                cy={y_for(frame.normalized_error * 100)}
                r={is_active ? 4 : 2.5}
                fill={is_active ? '#c98b5f' : is_best ? '#7fb39d' : '#ecede9'}
                stroke="#0e0f0d"
                strokeWidth="1.5"
              />
            )
          })}

          {Array.from({ length: task.max_steps + 1 }, (_, step) => step)
            .filter((step) => step === 0 || step === task.max_steps || step % 2 === 0)
            .map((step) => (
              <text
                key={step}
                x={x_for(step)}
                y={height - 12}
                textAnchor="middle"
                fill="#878a82"
                fontSize="10"
                fontFamily="IBM Plex Mono, monospace"
              >
                {step}
              </text>
            ))}
        </svg>

        <div className="mt-1 flex flex-wrap gap-4 px-2 font-mono text-[0.7rem] text-muted-foreground">
          <LegendDot color="#c98b5f" label="active frame" />
          <LegendDot color="#7fb39d" label="best score so far" />
          <LegendDot color="#ecede9" label="sampled step" />
        </div>
      </CardContent>
    </Card>
  )
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="inline-block size-2 rounded-full" style={{ background: color }} />
      {label}
    </span>
  )
}
