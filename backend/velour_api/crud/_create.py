from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import backend, exceptions, schemas, enums
from velour_api.backend import state
from velour_api.backend import metrics as backend_metrics


@state.create
def create_dataset(db: Session, dataset: schemas.Dataset):
    """Creates a dataset

    Raises
    ------
    DatasetAlreadyExistsError
        if the dataset name already exists
    """
    backend.create_dataset(db, dataset)


@state.create
def create_model(db: Session, model: schemas.Model):
    """Creates a dataset

    Raises
    ------
    ModelAlreadyExistsError
        if the model uid already exists
    """
    backend.create_model(db, model)


@state.create
def create_groundtruths(
    db: Session,
    groundtruth: schemas.GroundTruth,
):
    backend.create_groundtruth(db, groundtruth=groundtruth)


@state.create
def create_predictions(
    db: Session,
    prediction: schemas.Prediction,
):
    backend.create_prediction(db, prediction=prediction)


### TEMPORARY ###
from velour_api.backend import models, core, query


def _create_metric_mappings(
    db: Session,
    metrics: list[
        schemas.APMetric
        | schemas.APMetricAveragedOverIOUs
        | schemas.mAPMetric
        | schemas.mAPMetricAveragedOverIOUs
    ],
    evaluation_settings_id: int,
) -> list[dict]:


    labels = set(
        [
            (metric.label.key, metric.label.value)
            for metric in metrics
            if hasattr(metric, "label")
        ]
    )
    label_map = {
        (label[0], label[1]): query.get_label(db, label=schemas.Label(key=label[0], value=label[1])).id
        for label in labels
    }

    ret = []
    for metric in metrics:
        if hasattr(metric, "label"):
            ret.append(
                metric.db_mapping(
                    label_id=label_map[(metric.label.key, metric.label.value)],
                    evaluation_settings_id=evaluation_settings_id,
                )
            )
        else:
            ret.append(
                metric.db_mapping(
                    evaluation_settings_id=evaluation_settings_id
                )
            )

    return ret


@state.create
def create_clf_metrics(
    db: Session,
    request_info: schemas.ClfMetricsRequest,
) -> int:
    confusion_matrices, metrics = backend_metrics.compute_clf_metrics(
        db=db,
        dataset_name=request_info.settings.dataset_name,
        model_name=request_info.settings.model_name,
        group_by=request_info.settings.group_by,
    )

    dataset = core.get_dataset(db, request_info.settings.dataset_name)
    model = core.get_model(db, request_info.settings.model_name)

    mapping={
        "dataset_id": dataset.id,
        "model_id": model.id,
        "task_type": enums.TaskType.CLASSIFICATION,
        "pd_type": enums.AnnotationType.NONE,
        "gt_type": enums.AnnotationType.NONE,
        "group_by": request_info.settings.group_by,
    }
    es = models.EvaluationSettings(**mapping)
    try:
        db.add(es)
        db.commit()
    except:
        db.rollback()
        raise RuntimeError

    confusion_matrices_mappings = _create_metric_mappings(
        db=db, metrics=confusion_matrices, evaluation_settings_id=es.id
    )
    for mapping in confusion_matrices_mappings:
        row = models.ConfusionMatrix(**mapping)
        try:
            db.add(row)
            db.commit()
        except:
            db.rollback()
            raise RuntimeError

    metric_mappings = _create_metric_mappings(
        db=db, metrics=metrics, evaluation_settings_id=es.id
    )
    for mapping in metric_mappings:
        # ignore value since the other columns are unique identifiers
        # and have empirically noticed value can slightly change due to floating
        # point errors
        row = models.Metric(**mapping)
        try:
            db.add(row)
            db.commit()
        except:
            db.rollback()
            raise RuntimeError
    
    # @TODO Return job id
    return -1
        

# @TODO: Make this return a job id
@state.create
def create_ap_metrics(
    db: Session,
    request_info: schemas.APRequest,
) -> int:

    # @TODO: This is hacky, fix schemas.APRequest
    dataset_name = request_info.settings.dataset_name
    model_name = request_info.settings.model_name
    gt_type = enums.AnnotationType.BOX #request_info.settings.pd_type
    pd_type = enums.AnnotationType.BOX #request_info.settings.gt_type
    target_type = enums.AnnotationType.BOX #request_info.settings.target_type
    label_key = request_info.settings.label_key
    min_area = request_info.settings.min_area
    max_area = request_info.settings.max_area

    metrics = backend_metrics.compute_ap_metrics(
        db=db,
        dataset_name=dataset_name,
        model_name=model_name,
        iou_thresholds=request_info.iou_thresholds,
        ious_to_keep=request_info.ious_to_keep,
        label_key=label_key,
        target_type=target_type,
        gt_type=gt_type,
        pd_type=pd_type,
        min_area=min_area,
        max_area=max_area,
    )

    dataset = core.get_dataset(db, dataset_name)
    model = core.get_model(db, model_name)

    mapping={
        "dataset_id": dataset.id,
        "model_id": model.id,
        "task_type": enums.TaskType.DETECTION,
        "pd_type": pd_type,
        "gt_type": gt_type,
        "label_key": label_key,
        "min_area": request_info.settings.min_area,
        "max_area": request_info.settings.max_area,
    }
    mp = models.EvaluationSettings(**mapping)
    try:
        db.add(mp)
        db.commit()
    except:
        db.rollback()
        raise RuntimeError

    metric_mappings = _create_metric_mappings(
        db=db, metrics=metrics, evaluation_settings_id=mp.id
    )

    for mapping in metric_mappings:
        # ignore value since the other columns are unique identifiers
        # and have empircally noticed value can slightly change due to floating
        # point errors
        
        mapping["value"] = None
        row = models.Metric(**mapping)
        try:
            db.add(row)
            db.commit()
        except:
            db.rollback()
            raise RuntimeError
    
    # @TODO Return job id
    return -1