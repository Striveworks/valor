import {
  Box,
  Page,
  ResourceTitle,
  TableList,
  Tag,
  Tooltip,
  Typography
} from '@striveworks/minerva';
import { Link } from 'react-router-dom';
import { Loading } from '../../components/shared/Loading';
import { SummaryBar } from '../../components/shared/SummaryBar';
import { useGetModels } from '../../hooks/Models/useGetModels';
import { Datum } from '../../types/Datum';
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
          {/* <TableList.Row>
            <div>Name</div>
            <div>Description</div>
            <div>Type</div>
          </TableList.Row> */}
          {data.length ? (
            data.map((model) => {
              return (
                <TableList.Row key={model.name}>
                  <Box>
                    <ResourceTitle>
                      <Link to={`${model.name}`} style={{ color: '#FFF' }}>
                        {model.name}
                      </Link>
                    </ResourceTitle>
                  </Box>
                  <Box>
                    <Tooltip placement='top' title='Data Type'>
                      <Tag
                        iconName={model.type === Datum.TABULAR ? 'table' : 'image'}
                        kind='meta'
                      >
                        {model.type}
                      </Tag>
                    </Tooltip>
                  </Box>
                  <Box>{model.description}</Box>
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
