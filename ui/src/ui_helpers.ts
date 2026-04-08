import type { circuit_task_spec, playback_frame } from './ui_types'

export function format_frequency(hz: number): string {
  const absolute_hz = Math.abs(hz)

  if (absolute_hz >= 1_000_000) {
    return `${(hz / 1_000_000).toFixed(2)} MHz`
  }

  if (absolute_hz >= 1_000) {
    return `${(hz / 1_000).toFixed(2)} kHz`
  }

  return `${hz.toFixed(2)} Hz`
}

export function format_resistance(ohms: number): string {
  const absolute_ohms = Math.abs(ohms)

  if (absolute_ohms >= 1_000_000) {
    return `${(ohms / 1_000_000).toFixed(2)} Mohm`
  }

  if (absolute_ohms >= 1_000) {
    return `${(ohms / 1_000).toFixed(2)} kohm`
  }

  return `${ohms.toFixed(0)} ohm`
}

export function format_capacitance(farads: number): string {
  const absolute_farads = Math.abs(farads)

  if (absolute_farads >= 1) {
    return `${farads.toFixed(3)} F`
  }

  if (absolute_farads >= 1e-3) {
    return `${(farads * 1e3).toFixed(2)} mF`
  }

  if (absolute_farads >= 1e-6) {
    return `${(farads * 1e6).toFixed(2)} uF`
  }

  if (absolute_farads >= 1e-9) {
    return `${(farads * 1e9).toFixed(2)} nF`
  }

  return `${(farads * 1e12).toFixed(2)} pF`
}

export function format_percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

export function format_score(value: number): string {
  return value.toFixed(3)
}

export function format_evaluations(value: number | null): string {
  return value === null ? 'stepwise' : String(value)
}

export function task_title(task: circuit_task_spec): string {
  const filter_label = task.circuit_type === 'low_pass' ? 'Low-pass' : 'High-pass'
  const task_flavor = task.task_id.includes('budget') ? 'budget' : 'low-cost'

  return `${filter_label} | ${format_frequency(task.target_hz)} | ${task_flavor}`
}

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum)
}

export function normalize_log_value(
  value: number,
  minimum_value: number,
  maximum_value: number,
): number {
  const log_minimum = Math.log10(minimum_value)
  const log_maximum = Math.log10(maximum_value)
  const log_value = Math.log10(value)

  return clamp((log_value - log_minimum) / (log_maximum - log_minimum), 0, 1)
}

export function get_signal_tone(task: circuit_task_spec, frame: playback_frame) {
  if (frame.within_tolerance) {
    return {
      tone_class_name: 'ok',
      tone_label: `Within the ${task.success_tolerance_pct.toFixed(1)}% target band`,
    }
  }

  if (Math.abs(frame.delta_hz) / task.target_hz <= 0.1) {
    return {
      tone_class_name: 'soft',
      tone_label: 'Near the lock zone',
    }
  }

  return {
    tone_class_name: 'warn',
    tone_label: 'Still converging',
  }
}
