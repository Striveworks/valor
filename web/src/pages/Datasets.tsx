import { Page, TableList, Typography } from '@striveworks/minerva';
import { Link } from 'react-router-dom';
import { Loading } from '../components/shared/Loading';
import { SummaryBar } from '../components/shared/SummaryBar';
import { useGetDatasets } from '../hooks/useGetDatasets';
import { Stat } from '../types/TableList';

export function Datasets() {
  const { isLoading, isError, data } = useGetDatasets();

  if (isLoading) {
    return <Loading />;
  }

  if (isError) {
    return <>An error has occurered</>;
  }

  const stats: Stat[] = [{ name: data.length, icon: 'collection' }];

  return (
    <Page.Main>
      <Page.Header>
        <Typography.PageTitle>Datasets</Typography.PageTitle>
      </Page.Header>
      <Page.Content>
        <TableList summaryBar={<SummaryBar stats={stats} />}>
          <TableList.Row>
            <div>Name</div>
            <div>Description</div>
            <div>Type</div>
            <div>Finalized</div>
          </TableList.Row>
          {data.length ? (
            data.map((dataset) => {
              return (
                <TableList.Row key={dataset.name}>
                  <div>
                    <Link to={`models/${dataset.href}`} style={{ color: '#FFF' }}>
                      {dataset.name}
                    </Link>
                  </div>
                  <div>{dataset.description}</div>
                  <div>{dataset.type}</div>
                  <div>{dataset.finalized ? 'True' : 'False'}</div>
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
