import {
  Box,
  Page,
  TableList,
  Tag,
  Tooltip,
  Typography
} from '@striveworks/minerva';
import { useParams } from 'react-router';
import { Link } from 'react-router-dom';
import { Loading } from '../../components/shared/Loading';
import { useGetModelDetails } from '../../hooks/Models/useGetModelDetails';
import { Task } from '../../types/Models';
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

  return (
    <Page.Main>
      <Page.Header>
        <Typography textStyle='titlePageLarge'>{modelName}</Typography>
      </Page.Header>
      <Page.Content>
        <Typography textStyle='titleSectionLarge'>Evaluation</Typography>
        <TableList>
          {data.length ? (
            data.map((metric) => {
              return (
                <TableList.Row key={metric.id}>
                  <Box>
                    <TableList.RowLabel>
                      <Link to={`${metric.id}`} style={{ color: '#FFF' }}>
                        {metric.dataset_name}
                      </Link>
                    </TableList.RowLabel>
                  </Box>
                  <Box gap={8}>
                    <Tooltip placement='top' title='Model Prediction Task Type'>
                      <Tag iconName='model' kind='meta'>
                        {metric.model_pred_task_type}
                      </Tag>
                    </Tooltip>
                    <Tooltip placement='top' title='Dataset Prediction Task Type'>
                      <Tag iconName='dataset' kind='meta'>
                        {metric.dataset_gt_task_type}
                      </Tag>
                    </Tooltip>
                    {/*   For these typesBBOX_OBJECT_DETECTION = 'Bounding Box Object Detection',
  POLY_OBJECT_DETECTION = 'Polygon Object Detection',
  INSTANCE_SEGMENTATION = 'Instance Segmentation', */}
                    {metric.model_pred_task_type === Task.BBOX_OBJECT_DETECTION && (
                      <>
                        <Tooltip placement='top' title='Minimum Area of Objects'>
                          <Tag iconName='chevronDown' kind='meta'>
                            {metric?.min_area}
                          </Tag>
                        </Tooltip>
                        <Tooltip placement='top' title='Maximum Area of Objects'>
                          <Tag iconName='chevronUp' kind='meta'>
                            {metric?.max_area}
                          </Tag>
                        </Tooltip>
                      </>
                    )}
                  </Box>
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
