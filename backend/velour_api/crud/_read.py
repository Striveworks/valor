import io
import json
from base64 import b64encode
from collections.abc import Iterable

from geoalchemy2 import RasterElement
from geoalchemy2.functions import ST_AsGeoJSON, ST_AsPNG, ST_Envelope
from PIL import Image
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from velour_api import enums, exceptions, models, schemas


def _get_bounding_box_of_raster(
    db: Session, raster: RasterElement
) -> tuple[int, int, int, int]:
    env = json.loads(db.scalar(ST_AsGeoJSON(ST_Envelope(raster))))
    assert len(env["coordinates"]) == 1
    xs = [pt[0] for pt in env["coordinates"][0]]
    ys = [pt[1] for pt in env["coordinates"][0]]

    return min(xs), min(ys), max(xs), max(ys)


def _raster_to_png_b64(
    db: Session, raster: RasterElement, image: schemas.Image
) -> str:
    enveloping_box = _get_bounding_box_of_raster(db, raster)
    raster = Image.open(io.BytesIO(db.scalar(ST_AsPNG((raster))).tobytes()))

    assert raster.mode == "L"

    ret = Image.new(size=(image.width, image.height), mode=raster.mode)

    ret.paste(raster, box=enveloping_box)

    # mask is greyscale with values 0 and 1. to convert to binary
    # we first need to map 1 to 255
    ret = ret.point(lambda x: 255 if x == 1 else 0).convert("1")

    f = io.BytesIO()
    ret.save(f, format="PNG")
    f.seek(0)
    mask_bytes = f.read()
    return b64encode(mask_bytes).decode()


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


def get_groundtruth_segmentations_in_image(
    db: Session, uid: str, dataset_name: str, are_instance: bool
) -> list[schemas.GroundTruthSegmentation]:
    db_img = get_image(db, uid, dataset_name)
    gt_segs = db.scalars(
        select(models.GroundTruthSegmentation).where(
            and_(
                models.GroundTruthSegmentation.image_id == db_img.id,
                models.GroundTruthSegmentation.is_instance == are_instance,
            )
        )
    ).all()

    img = schemas.Image(
        uid=uid, height=db_img.height, width=db_img.width, frame=db_img.frame
    )

    return [
        schemas.GroundTruthSegmentation(
            shape=_raster_to_png_b64(db, gt_seg.shape, img),
            image=img,
            labels=[
                _db_label_to_schemas_label(labeled_gt_seg.label)
                for labeled_gt_seg in gt_seg.labeled_ground_truth_segmentations
            ],
            is_instance=gt_seg.is_instance,
        )
        for gt_seg in gt_segs
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


def get_metrics_from_metric_params(
    metric_params: list[models.MetricParameters],
) -> list[schemas.MetricResponse]:
    return [
        schemas.MetricResponse(
            metric_name=m.__tablename__,
            parameters=_db_metric_params_to_pydantic_metric_params(mp),
            metric=_db_metric_to_pydantic_metric(m),
        )
        for mp in metric_params
        for m in mp.ap_metrics
    ]


def get_metrics_from_metric_params_id(
    db: Session, metric_params_id: int
) -> list[schemas.MetricResponse]:
    metric_params = db.scalar(
        select(models.MetricParameters).where(
            models.MetricParameters.id == metric_params_id
        )
    )
    return get_metrics_from_metric_params([metric_params])


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

    return get_metrics_from_metric_params(metric_params)


def number_of_rows(db: Session, model_cls: type) -> int:
    return db.scalar(select(func.count(model_cls.id)))


def _get_detection_task_types(
    dets: Iterable[models.GroundTruthDetection | models.PredictedDetection],
) -> set[enums.Task]:
    # TODO: maybe more sql way of doing this
    ret = set()
    found_bbox, found_poly = False, False
    for det in dets:
        if det.is_bbox:
            found_bbox = True
            ret.add(enums.Task.BBOX_OBJECT_DETECTION)
        else:
            found_poly = True
            ret.add(enums.Task.POLY_OBJECT_DETECTION)
        if found_bbox and found_poly:
            break
    return ret


def _get_segmentation_task_types(
    segs: Iterable[
        models.GroundTruthSegmentation | models.PredictedSegmentation
    ],
) -> set[enums.Task]:
    # TODO: maybe more sql way of doing this
    ret = set()
    found_instance_seg, found_semantic_seg = False, False
    for seg in segs:
        if seg.is_instance:
            found_instance_seg = True
            ret.add(enums.Task.INSTANCE_SEGMENTATION)
        else:
            found_semantic_seg = True
            ret.add(enums.Task.SEMANTIC_SEGMENTATION)
        if found_instance_seg and found_semantic_seg:
            break

    return ret


def _get_model_pred_task_types(
    db: Session, model_name: str
) -> set[enums.Task]:
    model = get_model(db, model_name)
    ret = _get_detection_task_types(model.predicted_detections).union(
        _get_segmentation_task_types(model.predicted_segmentations)
    )

    if len(model.predicted_image_classifications):
        ret.add(enums.Task.IMAGE_CLASSIFICATION)

    return ret


def _get_dataset_task_types(db: Session, dataset_name: str) -> set[enums.Task]:
    dataset = get_dataset(db, dataset_name)

    def _detection_generator():
        for image in dataset.images:
            for detection in image.ground_truth_detections:
                yield detection

    def _segmentation_generator():
        for image in dataset.images:
            for segmentation in image.ground_truth_segmentations:
                yield segmentation

    ret = _get_detection_task_types(_detection_generator()).union(
        _segmentation_generator()
    )

    for image in dataset.images:
        if len(image.ground_truth_classifications) > 0:
            ret.add(enums.Task.IMAGE_CLASSIFICATION)
            break

    return ret
