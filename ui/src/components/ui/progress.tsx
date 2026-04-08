import * as React from 'react'

import { cn } from '@/lib/utils'

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value: number
  tone?: 'ok' | 'warn' | 'bad'
}

const tone_class: Record<NonNullable<ProgressProps['tone']>, string> = {
  ok: 'bg-ok',
  warn: 'bg-warn',
  bad: 'bg-bad',
}

export function Progress({ className, value, tone = 'ok', ...props }: ProgressProps) {
  const width = Math.max(0, Math.min(100, value))
  return (
    <div
      className={cn(
        'relative h-1 w-full overflow-hidden rounded-full bg-inset',
        className,
      )}
      role="progressbar"
      aria-valuenow={width}
      aria-valuemin={0}
      aria-valuemax={100}
      {...props}
    >
      <div
        className={cn('h-full transition-[width] duration-300', tone_class[tone])}
        style={{ width: `${width}%` }}
      />
    </div>
  )
}
