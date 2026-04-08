import type { ui_catalog_response, ui_playback_payload } from './ui_types'

const api_base_url = (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(/\/$/, '')

async function request_json<response_type>(
  path: string,
  signal?: AbortSignal,
): Promise<response_type> {
  const response = await fetch(`${api_base_url}${path}`, {
    headers: {
      Accept: 'application/json',
    },
    signal,
  })

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`

    try {
      const error_payload = (await response.json()) as { detail?: string }
      if (error_payload.detail) {
        message = error_payload.detail
      }
    } catch {
      // Keep the HTTP-derived fallback message.
    }

    throw new Error(message)
  }

  return (await response.json()) as response_type
}

export function get_catalog(signal?: AbortSignal) {
  return request_json<ui_catalog_response>('/ui/catalog', signal)
}

export function get_preview(task_id: string, signal?: AbortSignal) {
  const params = new URLSearchParams({ task_id })
  return request_json<ui_playback_payload>(`/ui/preview?${params.toString()}`, signal)
}

export function get_episode(task_id: string, signal?: AbortSignal) {
  const params = new URLSearchParams({ task_id })
  return request_json<ui_playback_payload>(`/ui/episode?${params.toString()}`, signal)
}
