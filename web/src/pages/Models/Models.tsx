import { Page, TableList, Typography } from '@striveworks/minerva';
import { Link } from 'react-router-dom';
import { Loading } from '../../components/shared/Loading';
import { SummaryBar } from '../../components/shared/SummaryBar';
import { useGetModels } from '../../hooks/Models/useGetModels';
import { Stat } from '../../types/TableList';
import { DisplayError } from '../DisplayError';

export function Models() {
  const { isLoading, isError, data, error } = useGetModels();

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
        <Typography.PageTitle>Models</Typography.PageTitle>
      </Page.Header>
      <Page.Content>
        <TableList summaryBar={<SummaryBar stats={stats} />}>
          <TableList.Row>
            <div>Name</div>
            <div>Description</div>
            <div>Type</div>
          </TableList.Row>
          {data.length ? (
            data.map((model) => {
              return (
                <TableList.Row key={model.name}>
                  <div>
                    <Link to={`${model.name}`} style={{ color: '#FFF' }}>
                      {model.name}
                    </Link>
                  </div>
                  <div>{model.description}</div>
                  <div>{model.type}</div>
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
