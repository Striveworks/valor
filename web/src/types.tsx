export type EvaluationSetting = {
  model_name: string;
  dataset_name: string;
  model_pred_task_type: string;
  dataset_gt_task_type: string;
  min_area: number;
  max_area: number;
  id: number;
};

export type Metric = {
  type: string;
  parameters: { iou: number; ious: number[] };
  label?: { key: string; value: string };
  value: number;
};

export type MetricAtIOU = {
  labelKey?: string;
  labelValue?: string;
  value: number;
  iou: number;
  id: number;
};

export type EntityResponse = {
  name: string;
  href: string;
  description: string;
};
