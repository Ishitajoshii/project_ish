import { empty_panel } from './empty_panel'
import type { baseline_comparison } from '../ui_types'
import { format_evaluations, format_frequency, format_percent, format_score } from '../ui_helpers'
import { Card, CardHeader, CardKicker, CardTitle } from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { cn } from '@/lib/utils'

interface baselines_card_props {
  comparisons: baseline_comparison[]
}

export function baselines_card({ comparisons }: baselines_card_props) {
  if (!comparisons.length) {
    return empty_panel({
      title: 'Baseline comparison',
      body: 'The comparison stack appears once the playback reaches the end of the episode — same task, same reward, agent vs baselines.',
    })
  }

  const best_score = Math.max(...comparisons.map((comparison) => comparison.score))

  return (
    <Card>
      <CardHeader>
        <div>
          <CardKicker>Baseline comparison</CardKicker>
          <CardTitle className="mt-1">Agent vs baselines on the same task</CardTitle>
        </div>
      </CardHeader>

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Method</TableHead>
            <TableHead className="text-right">Score</TableHead>
            <TableHead className="text-right">Achieved</TableHead>
            <TableHead className="text-right">Error</TableHead>
            <TableHead className="text-right">Steps</TableHead>
            <TableHead className="text-right">Evals</TableHead>
            <TableHead className="text-right">Δ</TableHead>
            <TableHead className="text-right">Status</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {comparisons.map((comparison) => {
            const is_agent = comparison.baseline_name === 'agent'
            const delta_to_best = comparison.score - best_score
            const is_best = delta_to_best === 0
            const delta_label = is_best
              ? 'best'
              : `${delta_to_best > 0 ? '+' : ''}${delta_to_best.toFixed(3)}`

            return (
              <TableRow
                key={comparison.baseline_name}
                className={cn(is_agent && 'bg-brand/5')}
              >
                <TableCell className="text-ink">
                  <span className="inline-flex items-center gap-2 font-sans font-medium">
                    <span
                      className={cn(
                        'size-2 rounded-full',
                        is_agent ? 'bg-brand' : 'bg-ok',
                      )}
                    />
                    {comparison.label}
                  </span>
                </TableCell>
                <TableCell className="text-right text-ink">
                  {format_score(comparison.score)}
                </TableCell>
                <TableCell className="text-right">
                  {format_frequency(comparison.achieved_hz)}
                </TableCell>
                <TableCell className="text-right">
                  {format_percent(comparison.normalized_error)}
                </TableCell>
                <TableCell className="text-right">
                  {comparison.steps_used === 0 ? 'n/a' : comparison.steps_used}
                </TableCell>
                <TableCell className="text-right">
                  {format_evaluations(comparison.evaluations)}
                </TableCell>
                <TableCell className={cn('text-right', is_best && 'text-brand')}>
                  {delta_label}
                </TableCell>
                <TableCell
                  className={cn(
                    'text-right uppercase tracking-[0.05em]',
                    comparison.success && 'text-ok',
                  )}
                >
                  {comparison.success ? 'pass' : 'track'}
                </TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </Card>
  )
}
