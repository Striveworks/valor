import { Page, TableList, Typography } from '@striveworks/minerva';
import { useParams } from 'react-router';
import { Link } from 'react-router-dom';
import { Loading } from '../../components/shared/Loading';
import { SummaryBar } from '../../components/shared/SummaryBar';
import { useGetModelDetails } from '../../hooks/Models/useGetModelDetails';
import { Stat } from '../../types/TableList';
import { DisplayError } from '../DisplayError';

type ModelDetailsParams = {
  modelName: string;
};

export function ModelDetails() {
  const { modelName } = useParams() as ModelDetailsParams;
  const { isLoading, isError, data, error } = useGetModelDetails(modelName);

  if (isLoading) {
    return <Loading />;
  }

  if (isError) {
    return <DisplayError error={error} />;
  }

  const stats: Stat[] = [{}];
  return (
    <Page.Main>
      <Page.Header>
        <Typography.PageTitle>{modelName}</Typography.PageTitle>
      </Page.Header>
      <Page.Content>
        <Typography.SectionTitle>Evaluation</Typography.SectionTitle>
        <TableList summaryBar={<SummaryBar stats={stats} />}>
          <TableList.Row>
            <div>Dataset</div>
            <div>Model Prediction Type</div>
            <div>Dataset Annotation Type</div>
            <div>Min. Area Of Objects</div>
            <div>Max Area Of Objects</div>
          </TableList.Row>
          {data.length ? (
            data.map((metric) => {
              return (
                <TableList.Row key={metric.id}>
                  <div>
                    <Link to={`${metric.id}`} style={{ color: '#FFF' }}>
                      {metric.dataset_name}
                    </Link>
                  </div>
                  <div>{metric.model_pred_task_type}</div>
                  <div>{metric.dataset_gt_task_type}</div>
                  <div>{metric.min_area}</div>
                  <div>{metric.max_area}</div>
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
