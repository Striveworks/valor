import { Box, Page, TableList, Tag, Tooltip, Typography } from '@striveworks/minerva';
import { Loading } from '../../components/shared/Loading';
import { SummaryBar } from '../../components/shared/SummaryBar';
import { useGetDatasets } from '../../hooks/Datasets/useGetDatasets';
import { Datum } from '../../types/Datum';
import { Stat } from '../../types/TableList';
import { DisplayError } from '../DisplayError';

export function Datasets() {
  const { isLoading, isError, data, error } = useGetDatasets();

  if (isLoading) {
    return <Loading />;
  }

  if (isError) {
    return <DisplayError error={error} />;
  }

  const stats: Stat[] = [{ name: data.length, icon: 'collection' }];

  return (
    <Page.Main>
      <Page.Header>
        <Typography.PageTitle>Datasets</Typography.PageTitle>
      </Page.Header>
      <Page.Content>
        <TableList summaryBar={<SummaryBar stats={stats} />}>
          {data.length ? (
            data.map((dataset) => {
              return (
                <TableList.Row key={dataset.name}>
                  <Box>{dataset.name}</Box>
                  <Box>
                    <Tooltip placement='top' title='Data Type'>
                      <Tag
                        iconName={dataset.type === Datum.TABULAR ? 'table' : 'image'}
                        kind='meta'
                      >
                        {dataset.type}
                      </Tag>
                    </Tooltip>
                  </Box>
                  <Box>
                    <Tooltip placement='top' title='Finalized'>
                      <Tag iconName={dataset.finalized ? 'check' : 'close'} kind='meta'>
                        {dataset.finalized ? 'True' : 'False'}
                      </Tag>
                    </Tooltip>
                  </Box>
                  <Box>{dataset.description}</Box>
                </TableList.Row>
              );
            })
          ) : (
            <TableList.Row>No Items Found</TableList.Row>
          )}
        </TableList>
      </Page.Content>
    </Page.Main>
  );
}
