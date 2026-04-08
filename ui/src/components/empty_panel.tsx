import { Card, CardContent, CardKicker } from '@/components/ui/card'
import { cn } from '@/lib/utils'

interface empty_panel_props {
  title: string
  body: string
  accent?: 'score'
}

export function empty_panel({ title, body }: empty_panel_props) {
  return (
    <Card className={cn('border-dashed bg-transparent')}>
      <CardContent className="grid gap-1 py-6">
        <CardKicker>{title}</CardKicker>
        <h3 className="m-0 text-[0.92rem] font-semibold text-ink-dim">Waiting for runtime data.</h3>
        <p className="m-0 mt-1 max-w-[46ch] text-[0.8rem] leading-relaxed text-muted-foreground">
          {body}
        </p>
      </CardContent>
    </Card>
  )
}
