import { Datum } from './Datum';

export enum Task {
  BBOX_OBJECT_DETECTION = 'Bounding Box Object Detection',
  POLY_OBJECT_DETECTION = 'Polygon Object Detection',
  INSTANCE_SEGMENTATION = 'Instance Segmentation',
  CLASSIFICATION = 'Classification',
  SEMANTIC_SEGMENTATION = 'Semantic Segmentation'
}

export type Model = {
  name: string;
  href: string;
  description: string;
  type: Datum;
};

export type ModelMetric = {
  model_name: string;
  dataset_name: string;
  model_pred_task_type: Task;
  dataset_gt_task_type: Task;
  min_area?: number;
  max_area?: number;
  group_by: string;
  label_key: string;
  id: number;
};
