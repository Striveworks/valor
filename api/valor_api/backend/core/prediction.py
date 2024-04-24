from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import enums, exceptions, schemas
from valor_api.backend import core, models


def _precompute_iou(
    db: Session,
    datum: models.Datum,
    prediction_annotations: list[models.Annotation],
):
    def _polygon_iou(gt_geom, pd_geom):
        gintersection = func.ST_Intersection(gt_geom, pd_geom)
        gunion = func.ST_Union(gt_geom, pd_geom)
        return func.ST_Area(gintersection) / func.ST_Area(gunion)

    def _raster_iou(gt_raster, pd_raster):
        return func.ST_Count(
            func.ST_MapAlgebra(
                gt_raster,
                pd_raster,
                "[rast1]*[rast2]",  # https://postgis.net/docs/RT_ST_MapAlgebra_expr.html
            )
        )

    ious = []

    # object detection
    obj_det_groundtruths = (
        db.query(models.Annotation)
        .where(
            and_(
                models.Annotation.datum_id == datum.id,
                models.Annotation.task_type
                == enums.TaskType.OBJECT_DETECTION.value,
            )
        )
        .all()
    )
    obj_det_pairings = [
        (gt, pd)
        for gt in obj_det_groundtruths
        for pd in prediction_annotations
        if pd.task_type == enums.TaskType.OBJECT_DETECTION.value
    ]
    for pairing in obj_det_pairings:
        gt, pd = pairing
        if gt.raster and pd.raster:
            iou_value = _raster_iou(gt.raster, pd.raster)
        elif gt.polygon and pd.polygon:
            iou_value = _polygon_iou(gt.polygon, pd.polygon)
        elif gt.box and pd.box:
            iou_value = _polygon_iou(gt.box, pd.box)
        else:
            continue
        ious.append(
            models.IOU(
                groundtruth_annotation_id=gt.id,
                prediction_annotation_id=pd.id,
                value=iou_value,
            )
        )

    # semantic segmentation
    sem_seg_groundtruths = (
        db.query(models.Annotation)
        .where(
            and_(
                models.Annotation.datum_id == datum.id,
                models.Annotation.task_type
                == enums.TaskType.SEMANTIC_SEGMENTATION.value,
            )
        )
        .all()
    )
    sem_seg_pairings = [
        (gt, pd)
        for gt in sem_seg_groundtruths
        for pd in prediction_annotations
        if pd.task_type == enums.TaskType.SEMANTIC_SEGMENTATION.value
    ]
    for pairing in sem_seg_pairings:
        gt, pd = pairing
        ious.append(
            models.IOU(
                groundtruth_annotation_id=gt.id,
                prediction_annotation_id=pd.id,
                value=_raster_iou(gt.raster, pd.raster),
            )
        )

    if ious:
        try:
            db.add_all(ious)
            db.commit()
        except IntegrityError as e:
            db.rollback()
            raise e


def create_prediction(
    db: Session,
    prediction: schemas.Prediction,
):
    """
    Creates a prediction.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    prediction : schemas.Prediction
        The prediction to create.
    """
    # check model status
    model_status = core.get_model_status(
        db=db,
        dataset_name=prediction.dataset_name,
        model_name=prediction.model_name,
    )
    if model_status != enums.TableStatus.CREATING:
        raise exceptions.ModelFinalizedError(
            dataset_name=prediction.dataset_name,
            model_name=prediction.model_name,
        )

    # retrieve existing table entries
    model = core.fetch_model(db, name=prediction.model_name)
    dataset = core.fetch_dataset(db, name=prediction.dataset_name)
    datum = core.fetch_datum(
        db, dataset_id=dataset.id, uid=prediction.datum.uid
    )

    # create labels
    all_labels = [
        label
        for annotation in prediction.annotations
        for label in annotation.labels
    ]
    label_list = core.create_labels(db=db, labels=all_labels)

    # create annotations
    annotation_list = core.create_annotations(
        db=db,
        annotations=prediction.annotations,
        datum=datum,
        model=model,
    )

    # create predictions
    label_idx = 0
    prediction_list = []
    for i, annotation in enumerate(prediction.annotations):
        indices = slice(label_idx, label_idx + len(annotation.labels))
        for j, label in enumerate(label_list[indices]):
            prediction_list.append(
                models.Prediction(
                    annotation_id=annotation_list[i].id,
                    label_id=label.id,
                    score=annotation.labels[j].score,
                )
            )
        label_idx += len(annotation.labels)

    try:
        db.add_all(prediction_list)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.PredictionAlreadyExistsError

    _precompute_iou(db=db, datum=datum, prediction_annotations=annotation_list)


def get_prediction(
    db: Session,
    model_name: str,
    dataset_name: str,
    datum_uid: str,
) -> schemas.Prediction:
    """
    Fetch a prediction.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    model_name : str
        The name of the model.
    dataset_name : str
        The name of the dataset.
    datum_uid: str
        The UID of the datum to fetch.

    Returns
    ----------
    schemas.Prediction
        The requested prediction.
    """
    model = core.fetch_model(db, name=model_name)
    dataset = core.fetch_dataset(db, name=dataset_name)
    datum = core.fetch_datum(db, dataset_id=dataset.id, uid=datum_uid)
    annotations = core.get_annotations(db, datum=datum, model=model)
    if len(annotations) == 0:
        raise exceptions.PredictionDoesNotExistError(
            model_name=model_name,
            dataset_name=dataset_name,
            datum_uid=datum_uid,
        )
    return schemas.Prediction(
        dataset_name=dataset.name,
        model_name=model_name,
        datum=schemas.Datum(
            uid=datum.uid,
            metadata=datum.meta,
        ),
        annotations=annotations,
    )


def delete_dataset_predictions(
    db: Session,
    dataset: models.Dataset,
):
    """
    Delete all predictions over a dataset.

    Parameters
    ----------
    db : Session
        The database session.
    dataset : models.Dataset
        The dataset row that is being deleted.

    Raises
    ------
    RuntimeError
        If dataset is not in deletion state.
    """

    if dataset.status != enums.TableStatus.DELETING:
        raise RuntimeError(
            f"Attempted to delete predictions from dataset `{dataset.name}` which has status `{dataset.status}`"
        )

    subquery = (
        select(models.Prediction.id.label("id"))
        .join(
            models.Annotation,
            models.Annotation.id == models.Prediction.annotation_id,
        )
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(models.Datum.dataset_id == dataset.id)
        .subquery()
    )
    delete_stmt = delete(models.Prediction).where(
        models.Prediction.id == subquery.c.id
    )

    try:
        db.execute(delete_stmt)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e


def delete_model_predictions(
    db: Session,
    model: models.Model,
):
    """
    Delete all predictions of a model.

    Parameters
    ----------
    db : Session
        The database session.
    model : models.Model
        The model row that is being deleted.

    Raises
    ------
    RuntimeError
        If dataset is not in deletion state.
    """

    if model.status != enums.ModelStatus.DELETING:
        raise RuntimeError(
            f"Attempted to delete annotations from dataset `{model.name}` which is not being deleted."
        )

    subquery = (
        select(models.Prediction.id.label("id"))
        .join(
            models.Annotation,
            models.Annotation.id == models.Prediction.annotation_id,
        )
        .where(models.Annotation.model_id == model.id)
        .subquery()
    )
    delete_stmt = delete(models.Prediction).where(
        models.Prediction.id == subquery.c.id
    )

    try:
        db.execute(delete_stmt)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
