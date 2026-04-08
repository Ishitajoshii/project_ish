import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'

import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center gap-1.5 rounded border px-2 py-0.5 font-mono text-[0.7rem] font-semibold uppercase tracking-[0.05em]',
  {
    variants: {
      variant: {
        default: 'border-line-strong bg-inset text-ink-dim',
        ok: 'border-ok/35 bg-inset text-ok',
        warn: 'border-warn/35 bg-inset text-warn',
        soft: 'border-line-strong bg-inset text-ink-dim',
        outline: 'border-line-strong bg-transparent text-muted-foreground',
      },
    },
    defaultVariants: { variant: 'default' },
  },
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />
}

export { Badge, badgeVariants }
