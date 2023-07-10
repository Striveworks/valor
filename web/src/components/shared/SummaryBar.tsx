import { TableList } from '@striveworks/minerva';
import { Stat } from '../../types/TableList';

export function SummaryBar({ stats }: { stats: Stat[] }) {
  return stats?.length ? (
    <TableList.SummaryBar
      stats={stats.map((stat) => (
        <TableList.Stat iconName={stat.icon} key={stat.name}>
          {stat.name}
        </TableList.Stat>
      ))}
    />
  ) : (
    <></>
  );
}
