import io
import json
from base64 import b64encode

from geoalchemy2 import RasterElement
from geoalchemy2.functions import ST_AsGeoJSON, ST_AsPNG, ST_Envelope
from PIL import Image
from sqlalchemy import (
    Float,
    Integer,
    Select,
    TextClause,
    and_,
    func,
    select,
    text,
)
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
        schemas.Dataset(
            **{k: getattr(d, k) for k in schemas.Dataset.__fields__}
        )
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
        schemas.Model(**{k: getattr(m, k) for k in schemas.Model.__fields__})
        for m in db.scalars(select(models.Model))
    ]


def get_model(db: Session, model_name: str) -> models.Model:
    ret = db.scalar(
        select(models.Model).where(models.Model.name == model_name)
    )
    if ret is None:
        raise exceptions.ModelDoesNotExistError(model_name)

    return ret


def get_image(db: Session, uid: str, dataset_name: str) -> models.Datum:
    ret = db.scalar(
        select(models.Datum)
        .join(models.Dataset)
        .where(
            and_(
                models.Datum.uid == uid,
                models.Datum.dataset_id == models.Dataset.id,
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

    def _single_db_gt_to_pydantic_gt(gt_det: models.GroundTruthDetection):
        labels = [
            _db_label_to_schemas_label(labeled_gt_det.label)
            for labeled_gt_det in gt_det.labeled_ground_truth_detections
        ]
        boundary = _boundary_points_from_detection(db, gt_det)

        if gt_det.is_bbox:
            xs = [b[0] for b in boundary]
            ys = [b[1] for b in boundary]
            return schemas.GroundTruthDetection(
                bbox=(min(xs), min(ys), max(xs), max(ys)),
                image=img,
                labels=labels,
            )
        else:
            return schemas.GroundTruthDetection(
                boundary=_boundary_points_from_detection(db, gt_det),
                image=img,
                labels=labels,
            )

    return [_single_db_gt_to_pydantic_gt(gt_det) for gt_det in gt_dets]


def get_groundtruth_segmentations_in_image(
    db: Session, uid: str, dataset_name: str, are_instance: bool
) -> list[schemas.GroundTruthSegmentation]:
    db_img = get_image(db, uid, dataset_name)
    gt_segs = db.scalars(
        select(models.GroundTruthSegmentation).where(
            and_(
                models.GroundTruthSegmentation.datum_id == db_img.id,
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


def get_detection_labels_in_dataset(
    db: Session, dataset_name: str
) -> list[models.Label]:
    return db.scalars(
        select(models.Label)
        .join(models.LabeledGroundTruthDetection)
        .join(models.GroundTruthDetection)
        .join(models.Datum)
        .join(models.Dataset)
        .where(
            and_(
                models.Dataset.name == dataset_name,
                models.Datum.id == models.GroundTruthDetection.datum_id,
            )
        )
        .distinct()
    ).all()


def get_segmentation_labels_in_dataset(
    db: Session, dataset_name: str
) -> list[models.Label]:
    return db.scalars(
        select(models.Label)
        .join(models.LabeledGroundTruthSegmentation)
        .join(models.GroundTruthSegmentation)
        .join(models.Datum)
        .join(models.Dataset)
        .where(
            and_(
                models.Dataset.name == dataset_name,
                models.Datum.id == models.GroundTruthSegmentation.datum_id,
            )
        )
        .distinct()
    ).all()


def get_classification_labels_in_dataset(
    db: Session, dataset_name: str, metadatum_id: int = None
) -> list[models.Label]:
    query = (
        select(models.Label)
        .join(models.GroundTruthClassification)
        .join(models.Datum)
        .join(models.Dataset)
    )

    if metadatum_id is not None:
        query = query.join(models.DatumMetadatumLink).where(
            and_(
                models.Dataset.name == dataset_name,
                models.Datum.id == models.GroundTruthClassification.datum_id,
                models.DatumMetadatumLink.metadatum_id == metadatum_id,
            )
        )
    else:
        query = query.where(
            and_(
                models.Dataset.name == dataset_name,
                models.Datum.id == models.GroundTruthClassification.datum_id,
            )
        )

    return db.scalars(query.distinct()).all()


def get_classification_prediction_labels(
    db: Session, model_name: str, dataset_name: str
):
    return db.scalars(
        select(models.Label)
        .join(models.PredictedClassification)
        .join(models.Model)
        .join(models.Datum)
        .join(models.Dataset)
        .where(
            and_(
                models.Dataset.name == dataset_name,
                models.Datum.id == models.PredictedClassification.datum_id,
                models.Model.name == model_name,
            )
        )
        .distinct()
    ).all()


def get_classification_label_values_in_dataset(
    db: Session, dataset_name: str, label_key: str
) -> list[str]:
    return db.scalars(
        select(models.Label.value)
        .join(models.GroundTruthClassification)
        .join(models.Datum)
        .join(models.Dataset)
        .where(
            and_(
                models.Dataset.name == dataset_name,
                models.Datum.id == models.GroundTruthClassification.datum_id,
                models.Label.key == label_key,
            )
        )
        .distinct()
    ).all()


def get_classification_prediction_label_values(
    db: Session, model_name: str, dataset_name: str, label_key: str
):
    return db.scalars(
        select(models.Label.value)
        .join(models.PredictedClassification)
        .join(models.Model)
        .join(models.Datum)
        .join(models.Dataset)
        .where(
            and_(
                models.Dataset.name == dataset_name,
                models.Datum.id == models.PredictedClassification.datum_id,
                models.Model.name == model_name,
                models.Label.key == label_key,
            )
        )
        .distinct()
    ).all()


def get_all_labels_in_dataset(
    db: Session, dataset_name: str
) -> set[models.Label]:
    return (
        set(get_classification_labels_in_dataset(db, dataset_name))
        .union(set(get_detection_labels_in_dataset(db, dataset_name)))
        .union(set(get_segmentation_labels_in_dataset(db, dataset_name)))
    )


def get_all_labels(db: Session) -> list[schemas.Label]:
    return [
        schemas.Label(key=label.key, value=label.value)
        for label in db.scalars(select(models.Label))
    ]


def get_datums_in_dataset(
    db: Session, dataset_name: str
) -> list[models.Datum]:
    dset = get_dataset(db, dataset_name)
    return dset.datums


def _get_unique_label_ids_in_image(image: models.Datum) -> set[int]:
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


def _db_evaluation_settings_to_pydantic_evaluation_settings(
    evaluation_settings: models.EvaluationSettings,
) -> schemas.EvaluationSettings:
    return schemas.EvaluationSettings(
        model_name=evaluation_settings.model.name,
        dataset_name=evaluation_settings.dataset.name,
        model_pred_task_type=evaluation_settings.model_pred_task_type,
        dataset_gt_task_type=evaluation_settings.dataset_gt_task_type,
        min_area=evaluation_settings.min_area,
        max_area=evaluation_settings.max_area,
        id=evaluation_settings.id,
    )


def _db_label_to_schemas_label(label: models.Label) -> schemas.Label:
    if label is None:
        return None
    return schemas.Label(key=label.key, value=label.value)


def _db_metric_to_pydantic_metric(metric: models.Metric) -> schemas.Metric:
    return schemas.Metric(
        type=metric.type,
        parameters=metric.parameters,
        value=metric.value,
        label=_db_label_to_schemas_label(metric.label),
    )


def get_metrics_from_evaluation_settings(
    evaluation_settings: list[models.EvaluationSettings],
) -> list[schemas.Metric]:
    return [
        _db_metric_to_pydantic_metric(m)
        for ms in evaluation_settings
        for m in ms.metrics
    ]


def get_metrics_from_evaluation_settings_id(
    db: Session, evaluation_settings_id: int
) -> list[schemas.Metric]:
    eval_settings = db.scalar(
        select(models.EvaluationSettings).where(
            models.EvaluationSettings.id == evaluation_settings_id
        )
    )
    return get_metrics_from_evaluation_settings([eval_settings])


def get_confusion_matrices_from_evaluation_settings_id(
    db: Session, evaluation_settings_id: int
) -> list[schemas.ConfusionMatrix]:
    eval_settings = db.scalar(
        select(models.EvaluationSettings).where(
            models.EvaluationSettings.id == evaluation_settings_id
        )
    )
    db_cms = eval_settings.confusion_matrices

    return [
        schemas.ConfusionMatrix(
            label_key=db_cm.label_key,
            entries=[
                schemas.ConfusionMatrixEntry(**entry) for entry in db_cm.value
            ],
        )
        for db_cm in db_cms
    ]


def get_evaluation_settings_from_id(
    db: Session, evaluation_settings_id: int
) -> schemas.EvaluationSettings:
    ms = db.scalar(
        select(models.EvaluationSettings).where(
            models.EvaluationSettings.id == evaluation_settings_id
        )
    )
    return _db_evaluation_settings_to_pydantic_evaluation_settings(ms)


def get_model_metrics(
    db: Session, model_name: str, evaluation_settings_id: int
) -> list[schemas.Metric]:
    # TODO: may return multiple types of metrics
    # use get_model so exception get's raised if model does
    # not exist
    model = get_model(db, model_name)

    evaluation_settings = db.scalars(
        select(models.EvaluationSettings)
        .join(models.Model)
        .where(
            and_(
                models.Model.id == model.id,
                models.EvaluationSettings.id == evaluation_settings_id,
            )
        )
    )

    return get_metrics_from_evaluation_settings(evaluation_settings)


def get_model_evaluation_settings(
    db: Session, model_name: str
) -> list[schemas.EvaluationSettings]:
    model_id = get_model(db, model_name).id
    all_eval_settings = db.scalars(
        select(models.EvaluationSettings).where(
            models.EvaluationSettings.model_id == model_id
        )
    ).all()
    return [
        _db_evaluation_settings_to_pydantic_evaluation_settings(eval_settings)
        for eval_settings in all_eval_settings
    ]


def number_of_rows(db: Session, model_cls: type) -> int:
    return db.scalar(select(func.count(model_cls.id)))


def _filter_instance_segmentations_by_area(
    stmt: str,
    seg_table: type,
    task_for_area_computation: schemas.Task,
    min_area: float | None,
    max_area: float | None,
) -> TextClause:
    if min_area is None and max_area is None:
        return text(stmt)

    if task_for_area_computation == schemas.Task.BBOX_OBJECT_DETECTION:
        area_stmt = f"(SELECT ST_Area(ST_Envelope(ST_Union(polygon))) FROM (SELECT ST_MakeValid((ST_DumpAsPolygons({seg_table.__tablename__}.shape)).geom) AS polygon) as subq)"
    elif task_for_area_computation == schemas.Task.POLY_OBJECT_DETECTION:
        # add convex hull?
        area_stmt = f"(SELECT ST_Area(ST_ConvexHull(ST_Union(polygon))) FROM (SELECT ST_MakeValid((ST_DumpAsPolygons({seg_table.__tablename__}.shape)).geom) AS polygon) as subq)"
    elif task_for_area_computation == schemas.Task.INSTANCE_SEGMENTATION:
        # segmentation
        area_stmt = f"ST_Count({seg_table.__tablename__}.shape)"
    else:
        raise ValueError(
            f"Got invalid value {task_for_area_computation} for `task_for_area_computation`."
        )

    if min_area is not None:
        stmt += f" AND {area_stmt} >= {min_area}"
    if max_area is not None:
        stmt += f" AND {area_stmt} <= {max_area}"

    return text(stmt)


def _instance_segmentations_in_dataset_statement(
    dataset_name: str,
    min_area: float = None,
    max_area: float = None,
    task_for_area_computation: schemas.Task = schemas.Task.INSTANCE_SEGMENTATION,
) -> TextClause:
    """Produces the text query to get all instance segmentations ids in a dataset,
    optionally filtered by area.

    Parameters
    ----------
    dataset_name
        name of the dataset
    min_area
        only select segmentations with area at least this value
    max_area
        only select segmentations with area at most this value
    task_for_area_computation
        one of Task.BBOX_OBJECT_DETECTION, Task.POLY_OBJECT_DETECTION, or
        Task.INSTANCE_SEGMENTATION. this determines how the area is calculated:
        if Task.BBOX_OBJECT_DETECTION then the area of the circumscribing polygon of the segmentation is used,
        if Task.POLY_OBJECT_DETECTION then the area of the convex hull of the segmentation is used
        if Task.INSTANCE_SEGMENTATION then the area of the segmentation itself is used.
    """

    stmt = f"""
        SELECT
            labeled_ground_truth_segmentation.id,
            labeled_ground_truth_segmentation.segmentation_id,
            labeled_ground_truth_segmentation.label_id
        FROM
            labeled_ground_truth_segmentation
            JOIN ground_truth_segmentation ON
                ground_truth_segmentation.id = labeled_ground_truth_segmentation.segmentation_id
            JOIN datum ON datum.id = ground_truth_segmentation.datum_id
            JOIN dataset ON dataset.id = datum.dataset_id
        WHERE
            ground_truth_segmentation.is_instance
            AND dataset.name = '{dataset_name}'
        """

    return _filter_instance_segmentations_by_area(
        stmt=stmt,
        seg_table=models.GroundTruthSegmentation,
        min_area=min_area,
        max_area=max_area,
        task_for_area_computation=task_for_area_computation,
    ).columns(id=Integer, segmentation_id=Integer, label_id=Integer)


def _filter_object_detections_by_area(
    stmt: str,
    det_table: type,
    task_for_area_computation: schemas.Task | None,
    min_area: float | None,
    max_area: float | None,
) -> TextClause:
    if min_area is None and max_area is None:
        return text(stmt)

    if task_for_area_computation == schemas.Task.BBOX_OBJECT_DETECTION:
        area_stmt = f"ST_Area(ST_Envelope({det_table.__tablename__}.boundary))"
    elif task_for_area_computation == schemas.Task.POLY_OBJECT_DETECTION:
        area_stmt = f"ST_Area({det_table.__tablename__}.boundary)"
    else:
        raise ValueError(
            f"Expected task_for_area_computation to be {schemas.Task.BBOX_OBJECT_DETECTION} or "
            f"{schemas.Task.POLY_OBJECT_DETECTION} but got {task_for_area_computation}."
        )

    if min_area is not None:
        stmt += f" AND {area_stmt} >= {min_area}"
    if max_area is not None:
        stmt += f" AND {area_stmt} <= {max_area}"

    return text(stmt)


def _object_detections_in_dataset_statement(
    dataset_name: str,
    task: schemas.Task,
    min_area: float = None,
    max_area: float = None,
    task_for_area_computation: schemas.Task = None,
) -> Select:
    """returns the select statement for all groundtruth object detections in a dataset.
    if min_area and/or max_area is None then it will filter accordingly by the area (pixels^2 and not proportion)
    """
    if task not in [
        enums.Task.POLY_OBJECT_DETECTION,
        enums.Task.BBOX_OBJECT_DETECTION,
    ]:
        raise ValueError(
            f"Expected task to be a detection task but got {task}"
        )
    stmt = f"""
        SELECT
            labeled_ground_truth_detection.id,
            labeled_ground_truth_detection.detection_id,
            labeled_ground_truth_detection.label_id
        FROM
            labeled_ground_truth_detection
            JOIN ground_truth_detection ON
                ground_truth_detection.id = labeled_ground_truth_detection.detection_id
            JOIN datum ON datum.id = ground_truth_detection.datum_id
            JOIN dataset ON dataset.id = datum.dataset_id
        WHERE dataset.name = '{dataset_name}' AND ground_truth_detection.is_bbox = '{(task == enums.Task.BBOX_OBJECT_DETECTION)}'
    """

    return _filter_object_detections_by_area(
        stmt=stmt,
        det_table=models.GroundTruthDetection,
        task_for_area_computation=task_for_area_computation,
        min_area=min_area,
        max_area=max_area,
    ).columns(id=Integer, detection_id=Integer, label_id=Integer)


def _classifications_in_dataset_statement(dataset_name: str) -> Select:
    return (
        select(models.GroundTruthClassification)
        .join(models.Datum)
        .join(models.Dataset)
        .where(models.Dataset.name == dataset_name)
    )


def _model_instance_segmentation_preds_statement(
    model_name: str,
    dataset_name: str,
    min_area: float = None,
    max_area: float = None,
    task_for_area_computation: schemas.Task = schemas.Task.INSTANCE_SEGMENTATION,
) -> Select:
    stmt = f"""
        SELECT
            labeled_predicted_segmentation.id,
            labeled_predicted_segmentation.segmentation_id,
            labeled_predicted_segmentation.label_id
        FROM
            labeled_predicted_segmentation
            JOIN predicted_segmentation ON
                predicted_segmentation.id = labeled_predicted_segmentation.segmentation_id
            JOIN datum ON datum.id = predicted_segmentation.datum_id
            JOIN model ON model.id = predicted_segmentation.model_id
            JOIN dataset ON dataset.id = datum.dataset_id
        WHERE
            model.name = '{model_name}'
            AND dataset.name = '{dataset_name}'
            AND predicted_segmentation.is_instance
    """
    return _filter_instance_segmentations_by_area(
        stmt=stmt,
        seg_table=models.PredictedSegmentation,
        task_for_area_computation=task_for_area_computation,
        min_area=min_area,
        max_area=max_area,
    ).columns(id=Integer, segmentation_id=Integer, label_id=Integer)


def _model_object_detection_preds_statement(
    model_name: str,
    dataset_name: str,
    task: enums.Task,
    min_area: float = None,
    max_area: float = None,
    task_for_area_computation: schemas.Task = None,
) -> Select:
    if task not in [
        enums.Task.POLY_OBJECT_DETECTION,
        enums.Task.BBOX_OBJECT_DETECTION,
    ]:
        raise ValueError(
            f"Expected task to be a detection task but got {task}"
        )

    stmt = f"""
    SELECT
        labeled_predicted_detection.id,
        labeled_predicted_detection.detection_id,
        labeled_predicted_detection.label_id,
        labeled_predicted_detection.score
    FROM
        labeled_predicted_detection
        JOIN predicted_detection ON
            predicted_detection.id = labeled_predicted_detection.detection_id
        JOIN datum ON datum.id = predicted_detection.datum_id
        JOIN model ON model.id = predicted_detection.model_id
        JOIN dataset ON dataset.id = datum.dataset_id
    WHERE
        model.name = '{model_name}'
        AND dataset.name = '{dataset_name}'
        AND predicted_detection.is_bbox = '{task == enums.Task.BBOX_OBJECT_DETECTION}'
    """

    return _filter_object_detections_by_area(
        stmt=stmt,
        det_table=models.PredictedDetection,
        task_for_area_computation=task_for_area_computation,
        min_area=min_area,
        max_area=max_area,
    ).columns(id=Integer, detection_id=Integer, label_id=Integer, score=Float)


def _model_classifications_preds_statement(
    model_name: str, dataset_name: str
) -> Select:
    return (
        select(models.PredictedClassification)
        .join(models.Datum)
        .join(models.Model)
        .join(models.Dataset)
        .where(
            and_(
                models.Model.name == model_name,
                models.Dataset.name == dataset_name,
            )
        )
    )


def get_dataset_task_types(db: Session, dataset_name: str) -> set[enums.Task]:
    ret = set()

    if db.query(
        _instance_segmentations_in_dataset_statement(
            dataset_name=dataset_name
        ).exists()
    ).scalar():
        ret.add(enums.Task.INSTANCE_SEGMENTATION)

    for task in [
        enums.Task.BBOX_OBJECT_DETECTION,
        enums.Task.POLY_OBJECT_DETECTION,
    ]:
        if db.query(
            _object_detections_in_dataset_statement(
                dataset_name, task
            ).exists()
        ).scalar():
            ret.add(task)

    return ret


def get_model_task_types(
    db: Session, model_name: str, dataset_name: str
) -> set[enums.Task]:
    ret = set()

    if db.query(
        _model_instance_segmentation_preds_statement(
            model_name=model_name, dataset_name=dataset_name
        ).exists()
    ).scalar():
        ret.add(enums.Task.INSTANCE_SEGMENTATION)

    for task in [
        enums.Task.BBOX_OBJECT_DETECTION,
        enums.Task.POLY_OBJECT_DETECTION,
    ]:
        if db.query(
            _model_object_detection_preds_statement(
                model_name=model_name, dataset_name=dataset_name, task=task
            ).exists()
        ).scalar():
            ret.add(task)

    return ret


def get_string_metadata_ids_and_vals(
    db: Session, dataset_name: str, metadata_name: str
) -> list[tuple[int, str]]:
    """Returns the ids of all metadata (for a given metadata name) in a dataset that
    have string values
    """
    return db.execute(
        text(
            f"""
        SELECT DISTINCT datum_metadatum_link.metadatum_id, metadatum.string_value
        FROM datum_metadatum_link
        JOIN metadatum ON datum_metadatum_link.metadatum_id=metadatum.id
        JOIN datum ON datum_metadatum_link.datum_id=datum.id
        JOIN dataset ON datum.dataset_id=dataset.id
        WHERE dataset.name='{dataset_name}'
            AND metadatum.name='{metadata_name}'
            AND metadatum.string_value IS NOT NULL
        """
        )
    ).all()
