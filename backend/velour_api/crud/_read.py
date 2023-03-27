import json

from geoalchemy2.functions import ST_AsGeoJSON
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from velour_api import exceptions, models, schemas


def get_datasets(db: Session) -> list[schemas.Dataset]:
    return [
        schemas.Dataset(name=d.name, draft=d.draft)
        for d in db.scalars(select(models.Dataset))
    ]


def get_dataset(db: Session, dataset_name: str) -> models.Dataset:
    ret = db.scalar(
        select(models.Dataset).where(models.Dataset.name == dataset_name)
    )
    if ret is None:
        raise exceptions.DatasetDoesNotExistError(dataset_name)

    return ret


def get_models(db: Session) -> list[schemas.Model]:
    return [
        schemas.Model(name=m.name) for m in db.scalars(select(models.Model))
    ]


def get_model(db: Session, model_name: str) -> models.Model:
    ret = db.scalar(
        select(models.Model).where(models.Model.name == model_name)
    )
    if ret is None:
        raise exceptions.ModelDoesNotExistError(model_name)

    return ret


def get_image(db: Session, uid: str, dataset_name: str) -> models.Image:
    ret = db.scalar(
        select(models.Image)
        .join(models.Dataset)
        .where(
            and_(
                models.Image.uid == uid,
                models.Image.dataset_id == models.Dataset.id,
                models.Dataset.name == dataset_name,
            )
        )
    )
    if ret is None:
        raise exceptions.ImageDoesNotExistError(uid, dataset_name)

    return ret


def _boundary_points_from_detection(
    db: Session,
    detection: models.PredictedDetection | models.GroundTruthDetection,
) -> list[tuple[float, float]]:
    geojson = db.scalar(ST_AsGeoJSON(detection.boundary))
    geojson = json.loads(geojson)
    coords = geojson["coordinates"]

    # make sure not a polygon
    assert len(coords) == 1

    return [tuple(coord) for coord in coords[0]]


def get_groundtruth_detections_in_image(
    db: Session, uid: str, dataset_name: str
) -> list[schemas.GroundTruthDetection]:
    db_img = get_image(db, uid, dataset_name)
    gt_dets = db_img.ground_truth_detections

    img = schemas.Image(
        uid=uid, height=db_img.height, width=db_img.width, frame=db_img.frame
    )

    return [
        schemas.GroundTruthDetection(
            boundary=_boundary_points_from_detection(db, gt_det),
            image=img,
            labels=[
                _db_label_to_schemas_label(labeled_gt_det.label)
                for labeled_gt_det in gt_det.labeled_ground_truth_detections
            ],
        )
        for gt_det in gt_dets
    ]


def get_labels_in_dataset(
    db: Session, dataset_name: str
) -> list[models.Label]:
    # TODO must be a better and more SQLy way of doing this
    dset = get_dataset(db, dataset_name)
    unique_ids = set()
    for image in dset.images:
        unique_ids.update(_get_unique_label_ids_in_image(image))

    return db.scalars(
        select(models.Label).where(models.Label.id.in_(unique_ids))
    ).all()


def get_all_labels(db: Session) -> list[schemas.Label]:
    return [
        schemas.Label(key=label.key, value=label.value)
        for label in db.scalars(select(models.Label))
    ]


def get_images_in_dataset(
    db: Session, dataset_name: str
) -> list[models.Image]:
    dset = get_dataset(db, dataset_name)
    return dset.images


def _get_unique_label_ids_in_image(image: models.Image) -> set[int]:
    ret = set()
    for det in image.ground_truth_detections:
        for labeled_det in det.labeled_ground_truth_detections:
            ret.add(labeled_det.label.id)

    for clf in image.ground_truth_classifications:
        ret.add(clf.label.id)

    for seg in image.ground_truth_segmentations:
        for labeled_seg in seg.labeled_ground_truth_segmentations:
            ret.add(labeled_seg.label.id)

    return ret


def _db_metric_params_to_pydantic_metric_params(
    metric_params: models.MetricParameters,
) -> schemas.MetricParameters:
    return schemas.MetricParameters(
        model_name=metric_params.model.name,
        dataset_name=metric_params.dataset.name,
        model_pred_task_type=metric_params.model_pred_task_type,
        dataset_gt_task_type=metric_params.dataset_gt_task_type,
    )


def _db_label_to_schemas_label(label: models.Label) -> schemas.Label:
    return schemas.Label(key=label.key, value=label.value)


def _db_metric_to_pydantic_metric(metric: models.APMetric) -> schemas.APMetric:
    # TODO: this will have to support more metrics
    return schemas.APMetric(
        iou=metric.iou,
        value=metric.value,
        label=_db_label_to_schemas_label(metric.label),
    )


def get_model_metrics(
    db: Session, model_name: str
) -> list[schemas.MetricResponse]:
    # TODO: may return multiple types of metrics
    # use get_model so exception get's raised if model does
    # not exist
    model = get_model(db, model_name)

    metric_params = db.scalars(
        select(models.MetricParameters)
        .join(models.Model)
        .where(models.Model.id == model.id)
    )

    return [
        schemas.MetricResponse(
            metric_name=m.__tablename__,
            parameters=_db_metric_params_to_pydantic_metric_params(mp),
            metric=_db_metric_to_pydantic_metric(m),
        )
        for mp in metric_params
        for m in mp.ap_metrics
    ]


def number_of_rows(db: Session, model_cls: type) -> int:
    return db.scalar(select(func.count(model_cls.id)))
