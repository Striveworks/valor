import pandas as pd
from valor_core import schemas, enums
from typing import Union, List


# TODO these shouldn't be private
def _validate_parameters(
    parameters: schemas.EvaluationParameters, task_type: enums.TaskType
) -> schemas.EvaluationParameters:
    if parameters.metrics_to_return is None:
        parameters.metrics_to_return = {
            enums.TaskType.CLASSIFICATION: [
                enums.MetricType.Precision,
                enums.MetricType.Recall,
                enums.MetricType.F1,
                enums.MetricType.Accuracy,
                enums.MetricType.ROCAUC,
                enums.MetricType.PrecisionRecallCurve,
            ],
            enums.TaskType.OBJECT_DETECTION: [
                enums.MetricType.AP,
                enums.MetricType.AR,
                enums.MetricType.mAP,
                enums.MetricType.APAveragedOverIOUs,
                enums.MetricType.mAR,
                enums.MetricType.mAPAveragedOverIOUs,
                enums.MetricType.PrecisionRecallCurve,
            ],
        }[task_type]

    return parameters


def convert_groundtruth_or_prediction_to_dataframe(
    list_of_objects: Union[List[schemas.GroundTruth], List[schemas.Prediction]]
) -> pd.DataFrame:

    output = []

    dataset_name = "delete later"
    # TODO dataset and model number don't really do anything in this framework?

    for obj in list_of_objects:
        datum_uid = obj.datum.uid
        datum_id = hash(obj.datum.uid)
        datum_metadata = obj.datum.metadata

        for ann in obj.annotations:
            ann_id = hash(str(datum_uid) + str(ann))
            ann_metadata = ann.metadata
            ann_bbox = ann.bounding_box
            ann_raster = ann.raster
            ann_embeding = ann.embedding
            ann_polygon = ann.polygon
            ann_is_instance = ann.is_instance

            for label in ann.labels:
                id_ = hash(str(ann_id) + str(label))
                label_key = label.key
                label_value = label.value
                label_score = label.score
                label_id = hash(label_key + label_value + str(label_score))

                # only include scores for predictions
                if isinstance(obj, schemas.Prediction):
                    output.append(
                        {
                            "dataset_name": dataset_name,
                            "datum_uid": datum_uid,
                            "datum_id": datum_id,
                            "datum_metadata": datum_metadata,
                            "annotation_id": ann_id,
                            "annotation_metadata": ann_metadata,
                            "bounding_box": ann_bbox,
                            "raster": ann_raster,
                            "embedding": ann_embeding,
                            "polygon": ann_polygon,
                            "is_instance": ann_is_instance,
                            "label_key": label_key,
                            "label_value": label_value,
                            "score": label_score,
                            "label_id": label_id,
                            "id": id_,
                        }
                    )
                else:
                    output.append(
                        {
                            "dataset_name": dataset_name,
                            "datum_uid": datum_uid,
                            "datum_id": datum_id,
                            "datum_metadata": datum_metadata,
                            "annotation_id": ann_id,
                            "annotation_metadata": ann_metadata,
                            "bounding_box": ann_bbox,
                            "raster": ann_raster,
                            "embedding": ann_embeding,
                            "polygon": ann_polygon,
                            "is_instance": ann_is_instance,
                            "label_key": label_key,
                            "label_value": label_value,
                            "label_id": label_id,
                            "id": id_,
                        }
                    )

    return pd.DataFrame(output)


def validate_groundtruth_dataframe(
    obj: Union[pd.DataFrame, List[schemas.GroundTruth]],
    task_type: enums.TaskType,
) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        # TODO check for correct columns
        return obj
    elif (
        obj
        and isinstance(obj, list)
        and isinstance(obj[0], schemas.GroundTruth)
    ):
        return convert_groundtruth_or_prediction_to_dataframe(obj)
    else:
        raise ValueError(
            "Could not validate object as it's neither a dataframe nor a list of Valor objects."
        )


def validate_prediction_dataframe(
    obj: Union[pd.DataFrame, List[schemas.Prediction]],
    task_type: enums.TaskType,
) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        # TODO check for correct columns
        return obj
    elif (
        obj
        and isinstance(obj, list)
        and isinstance(obj[0], schemas.Prediction)
    ):
        return convert_groundtruth_or_prediction_to_dataframe(obj)
    else:
        raise ValueError(
            "Could not validate object as it's neither a dataframe nor a list of Valor objects."
        )
