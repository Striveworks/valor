from geoalchemy2.functions import (
    ST_Area,
    ST_Boundary,
    ST_ConvexHull,
    ST_Count,
    ST_Envelope,
    ST_Polygon,
)
from sqlalchemy import Select, and_, insert, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, models, schemas
from velour_api.metrics import compute_ap_metrics

from ._read import get_dataset, get_image, get_model


def _labels_in_query(
    db: Session, query_statement: Select
) -> list[models.Label]:
    return db.scalars(
        (select(models.Label).join(query_statement.subquery())).distinct()
    ).all()


def _wkt_polygon_from_detection(det: schemas.DetectionBase) -> str:
    """Returns the "Well-known text" format of a detection"""
    if det.is_bbox:
        pts = [
            (det.bbox[0], det.bbox[1]),
            (det.bbox[0], det.bbox[3]),
            (det.bbox[2], det.bbox[3]),
            (det.bbox[2], det.bbox[1]),
        ]
    else:
        pts = det.boundary
    return f"POLYGON ({_boundary_points_to_str(pts)})"


def _boundary_points_to_str(pts: list[tuple[float, float]]) -> str:
    # in PostGIS polygon has to begin and end at the same point
    if pts[0] != pts[-1]:
        pts = pts + [pts[0]]
    return (
        "("
        + ", ".join([" ".join([str(pt[0]), str(pt[1])]) for pt in pts])
        + ")"
    )


def _wkt_multipolygon_from_polygons_with_hole(
    polys: list[schemas.PolygonWithHole],
) -> str:
    def poly_str(poly: schemas.PolygonWithHole):
        if poly.hole is None:
            return f"({_boundary_points_to_str(poly.polygon)})"
        return f"({_boundary_points_to_str(poly.polygon)}, {_boundary_points_to_str(poly.hole)})"

    return f"MULTIPOLYGON ( {', '.join([poly_str(poly) for poly in polys])} )"


def _bulk_insert_and_return_ids(
    db: Session, model: type, mappings: list[dict]
) -> list[int]:
    """Bulk adds to the database

    model
        the class that represents the database table
    mappings
        dictionaries mapping column names to values
    """
    added_ids = db.scalars(insert(model).values(mappings).returning(model.id))
    db.commit()
    return added_ids.all()


def _create_detection_mappings(
    detections: list[schemas.DetectionBase], images: list[models.Image]
) -> list[dict[str, str]]:
    return [
        {
            "boundary": _wkt_polygon_from_detection(detection),
            "image_id": image.id,
            "is_bbox": detection.is_bbox,
        }
        for detection, image in zip(detections, images)
    ]


def _select_statement_from_poly(
    shape: list[schemas.PolygonWithHole],
) -> Select:
    """Statement that converts a polygon to a raster"""
    poly = _wkt_multipolygon_from_polygons_with_hole(shape)
    return select(
        text(f"ST_AsRaster(ST_GeomFromText('{poly}'), {1.0}, {1.0})")
    )


def _create_gt_segmentation_mappings(
    segmentations: list[schemas.GroundTruthSegmentation],
    images: list[models.Image],
) -> list[dict[str, str]]:
    assert len(segmentations) == len(images)

    def _create_single_mapping(
        seg: schemas.GroundTruthSegmentation, image: models.Image
    ):
        if seg.is_poly:
            shape = _select_statement_from_poly(seg.shape)
        else:
            shape = seg.mask_bytes

        return {
            "is_instance": seg.is_instance,
            "shape": shape,
            "image_id": image.id,
        }

    return [
        _create_single_mapping(segmentation, image)
        for segmentation, image in zip(segmentations, images)
    ]


def _create_pred_segmentation_mappings(
    segmentations: list[schemas.PredictedSegmentation],
    images: list[models.Image],
) -> list[dict[str, str]]:
    return [
        {
            "is_instance": segmentation.is_instance,
            "shape": segmentation.mask_bytes,
            "image_id": image.id,
        }
        for segmentation, image in zip(segmentations, images)
    ]


def _create_label_tuple_to_id_dict(
    db,
    labels: list[schemas.Label],
) -> dict[tuple, str]:
    """Goes through the labels and adds to the db if it doesn't exist. The return is a mapping from
    `tuple(label)` (since `label` is not hashable) to label id
    """
    label_tuple_to_id = {}
    for label in labels:
        label_tuple = tuple(label)
        if label_tuple not in label_tuple_to_id:
            label_tuple_to_id[label_tuple] = _get_or_create_row(
                db, models.Label, {"key": label.key, "value": label.value}
            ).id
    return label_tuple_to_id


def _add_images_to_dataset(
    db: Session, dataset_name, images: list[schemas.Image]
) -> list[models.Image]:
    """Adds images defined by URIs to a dataset (creating the Image rows if they don't exist),
    returning the list of image ids"""
    dset = get_dataset(db, dataset_name=dataset_name)
    if not dset.draft:
        raise exceptions.DatasetIsFinalizedError(dataset_name)
    dset_id = dset.id

    return [
        _get_or_create_row(
            db=db,
            model_class=models.Image,
            mapping={"dataset_id": dset_id, **img.dict()},
        )
        for img in images
    ]


def _create_gt_dets_or_segs(
    db: Session,
    dataset_name: str,
    dets_or_segs: list[
        schemas.GroundTruthDetection | schemas.GroundTruthSegmentation
    ],
    mapping_method: callable,
    labeled_mapping_method: callable,
    model_cls: type,
    labeled_model_cls: type,
):
    images = _add_images_to_dataset(
        db=db,
        dataset_name=dataset_name,
        images=[d_or_s.image for d_or_s in dets_or_segs],
    )
    mappings = mapping_method(dets_or_segs, images)

    ids = _bulk_insert_and_return_ids(db, model_cls, mappings)

    label_tuple_to_id = _create_label_tuple_to_id_dict(
        db, [label for d_or_s in dets_or_segs for label in d_or_s.labels]
    )

    labeled_gt_mappings = labeled_mapping_method(
        label_tuple_to_id, ids, dets_or_segs
    )

    return _bulk_insert_and_return_ids(
        db, labeled_model_cls, labeled_gt_mappings
    )


def _create_pred_dets_or_segs(
    db: Session,
    model_name: str,
    dataset_name: str,
    dets_or_segs: list[
        schemas.PredictedDetection | schemas.PredictedSegmentation
    ],
    mapping_method: callable,
    labeled_mapping_method: callable,
    model_cls: type,
    labeled_model_cls: type,
):
    model_id = get_model(db, model_name=model_name).id
    # get image ids from uids (these images should already exist)
    images = [
        get_image(db, uid=d_or_s.image.uid, dataset_name=dataset_name)
        for d_or_s in dets_or_segs
    ]
    mappings = mapping_method(dets_or_segs, images)
    for m in mappings:
        m["model_id"] = model_id

    ids = _bulk_insert_and_return_ids(db, model_cls, mappings)

    label_tuple_to_id = _create_label_tuple_to_id_dict(
        db,
        [
            scored_label.label
            for d_or_s in dets_or_segs
            for scored_label in d_or_s.scored_labels
        ],
    )

    labeled_pred_mappings = labeled_mapping_method(
        label_tuple_to_id, ids, dets_or_segs
    )

    return _bulk_insert_and_return_ids(
        db, labeled_model_cls, labeled_pred_mappings
    )


def _create_labeled_gt_detection_mappings(
    label_tuple_to_id,
    gt_det_ids: list[int],
    detections: list[schemas.GroundTruthDetection],
):
    return [
        {
            "detection_id": gt_det_id,
            "label_id": label_tuple_to_id[tuple(label)],
        }
        for gt_det_id, detection in zip(gt_det_ids, detections)
        for label in detection.labels
    ]


def _create_labeled_gt_segmentation_mappings(
    label_tuple_to_id,
    gt_seg_ids: list[int],
    segmentations: list[schemas.GroundTruthSegmentation],
):
    return [
        {
            "segmentation_id": gt_seg_id,
            "label_id": label_tuple_to_id[tuple(label)],
        }
        for gt_seg_id, segmentation in zip(gt_seg_ids, segmentations)
        for label in segmentation.labels
    ]


def _create_labeled_pred_detection_mappings(
    label_tuple_to_id,
    gt_det_ids: list[int],
    detections: list[schemas.PredictedDetection],
):
    return [
        {
            "detection_id": gt_id,
            "label_id": label_tuple_to_id[tuple(scored_label.label)],
            "score": scored_label.score,
        }
        for gt_id, detection in zip(gt_det_ids, detections)
        for scored_label in detection.scored_labels
    ]


def _create_labeled_pred_segmentation_mappings(
    label_tuple_to_id,
    gt_det_ids: list[int],
    segmentations: list[schemas.PredictedSegmentation],
):
    return [
        {
            "segmentation_id": gt_id,
            "label_id": label_tuple_to_id[tuple(scored_label.label)],
            "score": scored_label.score,
        }
        for gt_id, segmentation in zip(gt_det_ids, segmentations)
        for scored_label in segmentation.scored_labels
    ]


def create_groundtruth_detections(
    db: Session,
    data: schemas.GroundTruthDetectionsCreate,
) -> list[int]:

    return _create_gt_dets_or_segs(
        db=db,
        dataset_name=data.dataset_name,
        dets_or_segs=data.detections,
        mapping_method=_create_detection_mappings,
        labeled_mapping_method=_create_labeled_gt_detection_mappings,
        model_cls=models.GroundTruthDetection,
        labeled_model_cls=models.LabeledGroundTruthDetection,
    )


def create_predicted_detections(
    db: Session, data: schemas.PredictedDetectionsCreate
) -> list[int]:
    """
    Raises
    ------
    ModelDoesNotExistError
        if the model with name `data.model_name` does not exist
    """
    return _create_pred_dets_or_segs(
        db=db,
        model_name=data.model_name,
        dataset_name=data.dataset_name,
        dets_or_segs=data.detections,
        mapping_method=_create_detection_mappings,
        model_cls=models.PredictedDetection,
        labeled_mapping_method=_create_labeled_pred_detection_mappings,
        labeled_model_cls=models.LabeledPredictedDetection,
    )


def create_groundtruth_segmentations(
    db: Session,
    data: schemas.GroundTruthSegmentationsCreate,
) -> list[int]:
    return _create_gt_dets_or_segs(
        db=db,
        dataset_name=data.dataset_name,
        dets_or_segs=data.segmentations,
        mapping_method=_create_gt_segmentation_mappings,
        labeled_mapping_method=_create_labeled_gt_segmentation_mappings,
        model_cls=models.GroundTruthSegmentation,
        labeled_model_cls=models.LabeledGroundTruthSegmentation,
    )


def create_predicted_segmentations(
    db: Session, data: schemas.PredictedSegmentationsCreate
) -> list[int]:
    """
    Raises
    ------
    ModelDoesNotExistError
        if the model with name `data.model_name` does not exist
    """
    return _create_pred_dets_or_segs(
        db=db,
        model_name=data.model_name,
        dataset_name=data.dataset_name,
        dets_or_segs=data.segmentations,
        mapping_method=_create_pred_segmentation_mappings,
        model_cls=models.PredictedSegmentation,
        labeled_mapping_method=_create_labeled_pred_segmentation_mappings,
        labeled_model_cls=models.LabeledPredictedSegmentation,
    )


def create_ground_truth_image_classifications(
    db: Session, data: schemas.GroundTruthImageClassificationsCreate
):
    images = _add_images_to_dataset(
        db=db,
        dataset_name=data.dataset_name,
        images=[c.image for c in data.classifications],
    )
    label_tuple_to_id = _create_label_tuple_to_id_dict(
        db, [label for clf in data.classifications for label in clf.labels]
    )
    clf_mappings = [
        {"label_id": label_tuple_to_id[tuple(label)], "image_id": image.id}
        for clf, image in zip(data.classifications, images)
        for label in clf.labels
    ]

    return _bulk_insert_and_return_ids(
        db, models.GroundTruthImageClassification, clf_mappings
    )


def create_predicted_image_classifications(
    db: Session, data: schemas.PredictedImageClassificationsCreate
):
    model_id = get_model(db, model_name=data.model_name).id
    # get image ids from uids (these images should already exist)
    image_ids = [
        get_image(
            db,
            uid=classification.image.uid,
            dataset_name=data.dataset_name,
        ).id
        for classification in data.classifications
    ]

    label_tuple_to_id = _create_label_tuple_to_id_dict(
        db,
        [
            scored_label.label
            for clf in data.classifications
            for scored_label in clf.scored_labels
        ],
    )
    pred_mappings = [
        {
            "label_id": label_tuple_to_id[tuple(scored_label.label)],
            "score": scored_label.score,
            "image_id": image_id,
            "model_id": model_id,
        }
        for clf, image_id in zip(data.classifications, image_ids)
        for scored_label in clf.scored_labels
    ]

    return _bulk_insert_and_return_ids(
        db, models.PredictedImageClassification, pred_mappings
    )


def _get_or_create_row(db: Session, model_class: type, mapping: dict) -> any:
    """Tries to get the row defined by mapping. If that exists then
    its mapped object is returned. Otherwise a row is created by `mapping` and the newly created
    object is returned
    """
    # create the query from the mapping
    where_expressions = [
        (getattr(model_class, k) == v) for k, v in mapping.items()
    ]
    where_expression = where_expressions[0]
    for exp in where_expressions[1:]:
        where_expression = where_expression & exp

    db_element = db.scalar(select(model_class).where(where_expression))

    if not db_element:
        db_element = model_class(**mapping)
        db.add(db_element)
        db.flush()
        db.commit()

    return db_element


def create_dataset(
    db: Session, dataset: schemas.DatasetCreate, from_video=False
):
    """Creates a dataset

    Raises
    ------
    DatasetAlreadyExistsError
        if the dataset name already exists
    """
    try:
        db.add(
            models.Dataset(
                name=dataset.name, draft=True, from_video=from_video
            )
        )
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.DatasetAlreadyExistsError(dataset.name)


def create_model(db: Session, model: schemas.Model):
    """Creates a dataset

    Raises
    ------
    ModelAlreadyExistsError
        if the model uid already exists
    """
    try:
        db.add(models.Model(name=model.name))
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.ModelAlreadyExistsError(model.name)


def _label_key_value_to_id(
    db: Session, labels: set[tuple[str, str]]
) -> dict[tuple[str, str], int]:
    return {
        label: db.scalar(
            select(models.Label.id).where(
                and_(
                    models.Label.key == label[0],
                    models.Label.value == label[1],
                )
            )
        )
        for label in labels
    }


def _create_ap_metric_mappings(
    db: Session, metrics: list[schemas.APMetric], metric_parameters_id: int
) -> list[dict]:
    label_map = _label_key_value_to_id(
        db=db,
        labels=set(
            [(metric.label.key, metric.label.value) for metric in metrics]
        ),
    )
    return [
        {
            "value": metric.value,
            "label_id": label_map[(metric.label.key, metric.label.value)],
            "iou": metric.iou,
            "metric_parameters_id": metric_parameters_id,
        }
        for metric in metrics
    ]


def validate_requested_labels_and_get_new_defining_statements_and_missing_labels(
    db: Session,
    gts_statement: Select,
    preds_statement: Select,
    requested_labels: list[schemas.Label] = None,
) -> tuple[Select, Select, list[schemas.Label], list[schemas.Label]]:
    """Takes statements defining a collection of labeled groundtruths and labeled predictions,
    and a list of requsted labels and creates a new statement that further
    filters down to those groundtruths and preditions that have labels in the list of requested labels.
    It also checks that the requested labels are contained in the set of all possible labels
    and throws a ValueError if not.

    Parameters
    ----------
    db
    gts_statement
        the select statement that defines the colllection of labeled groundtruths
    preds_statement
        the select statement that defines the colllection of labeled predictions
    requested_labels
        list of labels requested. if this is None then all labels present in the collection
        defined by `gts_statement` are used.

    Returns
    -------
    a tuple with the following elements:
        - select statement defining the filtered groundtruths
        - select statement defining the filtered predictions
        - list of key/value label tuples of requested labels (or labels in the groundtruth collection
        in the case that `requested_labels` is None) that are not present in the predictions collection
        - list of key/value label tuples of labels that are present in the predictions collection but
        are not in `requested_labels` (or the labels in the groundtruth collection in the case that
        `requested_labels` is None)

    Raises
    ------
    ValueError
        if there are labels in `requested_labels` that are not in the groundtruth annotations defined
        by `gts_statement`.
    """

    available_labels = _labels_in_query(db, gts_statement)

    if requested_labels is None:
        requested_label_tuples = set(
            [(label.key, label.value) for label in available_labels]
        )
        pred_label_tuples = set(
            [
                (label.key, label.value)
                for label in _labels_in_query(db, preds_statement)
            ]
        )
        labels_to_use_ids = [label.id for label in available_labels]
    else:
        pred_labels = _labels_in_query(db, preds_statement)

        # convert available labels and requested labels to key/value tuples to allow easy comparison
        available_label_tuples = set(
            [(label.key, label.value) for label in available_labels]
        )
        requested_label_tuples = set(
            [(label.key, label.value) for label in requested_labels]
        )
        pred_label_tuples = set(
            [(label.key, label.value) for label in pred_labels]
        )

        # filter to those labels specified
        if not (requested_label_tuples <= available_label_tuples):
            raise ValueError(
                f"The following label key/value pairs are missing in the dataset: {requested_label_tuples - available_label_tuples}"
            )

        labels_to_use_ids = [
            label.id
            for label in available_labels
            if (label.key, label.value) in requested_label_tuples
        ]

        gts_statement = gts_statement.join(models.Label).where(
            models.Label.id.in_(labels_to_use_ids)
        )

    preds_statement = preds_statement.join(models.Label).where(
        models.Label.id.in_(labels_to_use_ids)
    )

    missing_pred_labels = requested_label_tuples - pred_label_tuples
    ignored_pred_labels = pred_label_tuples - requested_label_tuples

    # convert back to labels
    missing_pred_labels = [
        schemas.Label.from_key_value_tuple(la) for la in missing_pred_labels
    ]
    ignored_pred_labels = [
        schemas.Label.from_key_value_tuple(la) for la in ignored_pred_labels
    ]

    return (
        gts_statement,
        preds_statement,
        missing_pred_labels,
        ignored_pred_labels,
    )


def _filter_instance_segmentations_by_area(
    stmt: Select,
    seg_table: type,
    task: schemas.Task,
    min_area: float | None,
    max_area: float | None,
) -> Select:
    if min_area is None and max_area is None:
        return stmt

    if task == schemas.Task.BBOX_OBJECT_DETECTION:
        area_fn = lambda x: ST_Area(ST_Envelope(x))  # noqa: E731
    elif task == schemas.Task.POLY_OBJECT_DETECTION:
        area_fn = lambda x: ST_Area(  # noqa: E731
            ST_ConvexHull(ST_Boundary(ST_Polygon(x)))
        )
    else:
        area_fn = ST_Count

    if min_area is not None:
        stmt = stmt.where(area_fn(seg_table.shape) >= min_area)
    if max_area is not None:
        stmt = stmt.where(area_fn(seg_table.shape) <= max_area)

    return stmt


def _filter_object_detections_by_area(
    stmt: Select,
    det_table: type,
    task: schemas.Task,
    min_area: float | None,
    max_area: float | None,
) -> Select:
    if min_area is None and max_area is None:
        return stmt

    if task == schemas.Task.BBOX_OBJECT_DETECTION:
        area_fn = lambda x: ST_Area(ST_Envelope(x))  # noqa: E731
    elif task == schemas.Task.POLY_OBJECT_DETECTION:
        area_fn = ST_Area
    else:
        raise ValueError(
            f"Expected task to be {schemas.Task.BBOX_OBJECT_DETECTION} or "
            f"{schemas.Task.POLY_OBJECT_DETECTION} but got {task}."
        )

    if min_area is not None:
        stmt = stmt.where(area_fn(det_table.boundary) >= min_area)
    if max_area is not None:
        stmt = stmt.where(area_fn(det_table.boundary) <= max_area)

    return stmt


def _instance_segmentations_in_dataset_statement(
    dataset_name: str,
    min_area: float = None,
    max_area: float = None,
    task: schemas.Task = None,
) -> Select:
    return _filter_instance_segmentations_by_area(
        stmt=(
            select(models.LabeledGroundTruthSegmentation)
            .join(models.GroundTruthSegmentation)
            .join(models.Image)
            .join(models.Dataset)
            .where(
                and_(
                    models.GroundTruthSegmentation.is_instance,
                    models.Dataset.name == dataset_name,
                )
            )
        ),
        seg_table=models.GroundTruthSegmentation,
        task=task,
        min_area=min_area,
        max_area=max_area,
    )


def _object_detections_in_dataset_statement(
    dataset_name: str,
    min_area: float = None,
    max_area: float = None,
    task: schemas.Task = None,
) -> Select:
    """returns the select statement for all groundtruth object detections in a dataset.
    if min_area and/or max_area is None then it will filter accordingly by the area (pixels^2 and not proportion)
    """
    return _filter_object_detections_by_area(
        stmt=(
            select(models.LabeledGroundTruthDetection)
            .join(models.GroundTruthDetection)
            .join(models.Image)
            .join(models.Dataset)
            .where(models.Dataset.name == dataset_name)
        ),
        det_table=models.GroundTruthDetection,
        task=task,
        min_area=min_area,
        max_area=max_area,
    )


def _model_instance_segmentation_preds_statement(
    model_name: str,
    dataset_name: str,
    min_area: float = None,
    max_area: float = None,
    task: schemas.Task = None,
) -> Select:
    return _filter_instance_segmentations_by_area(
        stmt=(
            select(models.LabeledPredictedSegmentation)
            .join(models.PredictedSegmentation)
            .join(models.Image)
            .join(models.Model)
            .join(models.Dataset)
            .where(
                and_(
                    models.Model.name == model_name,
                    models.Dataset.name == dataset_name,
                    models.PredictedSegmentation.is_instance,
                )
            )
        ),
        seg_table=models.PredictedSegmentation,
        task=task,
        min_area=min_area,
        max_area=max_area,
    )


def _model_object_detection_preds_statement(
    model_name: str,
    dataset_name,
    min_area: float = None,
    max_area: float = None,
    task: schemas.Task = None,
) -> Select:
    return _filter_object_detections_by_area(
        stmt=(
            select(models.LabeledPredictedDetection)
            .join(models.PredictedDetection)
            .join(models.Image)
            .join(models.Model)
            .join(models.Dataset)
            .where(
                and_(
                    models.Model.name == model_name,
                    models.Dataset.name == dataset_name,
                )
            )
        ),
        det_table=models.PredictedDetection,
        task=task,
        min_area=min_area,
        max_area=max_area,
    )


def validate_create_ap_metrics(
    db: Session, request_info: schemas.APRequest
) -> tuple[Select, Select, list[schemas.Label], list[schemas.Label]]:
    """Validates request_info and produces select statements for grabbing groundtruth and
    prediction data

    Returns
    -------
    tuple[Select, Select, list[schemas.Label], list[schemas.Label]]
        first element is the select statement for groundtruths, second is the select statement
        for predictions, third is list of labels that were missing in the predictions, and fourth
        is list of labels in the predictions that were ignored (because they weren't in the groundtruth)
    """
    dataset_name = request_info.parameters.dataset_name
    model_name = request_info.parameters.model_name
    if get_dataset(db, dataset_name).draft:
        raise exceptions.DatasetIsDraftError(
            request_info.parameters.dataset_name
        )
    # check that inferences are finalized
    if not _check_finalized_inferences(
        db, model_name=model_name, dataset_name=dataset_name
    ):
        raise exceptions.InferencesAreNotFinalizedError(
            dataset_name=dataset_name, model_name=model_name
        )

    # do some validation
    allowable_tasks = [
        schemas.Task.BBOX_OBJECT_DETECTION,
        schemas.Task.POLY_OBJECT_DETECTION,
        schemas.Task.INSTANCE_SEGMENTATION,
    ]

    if request_info.parameters.model_pred_task_type not in allowable_tasks:
        raise ValueError(
            f"`pred_type` must be one of {allowable_tasks} but got {request_info.parameters.model_pred_task_type}."
        )
    if request_info.parameters.dataset_gt_task_type not in allowable_tasks:
        raise ValueError(
            f"`dataset_gt_task_type` must be one of {allowable_tasks} but got {request_info.parameters.dataset_gt_task_type}."
        )

    # when computing AP, the fidelity of a detection will drop to the minimum fidelity of the groundtruth and predicted
    # task type. e.g. if one is bounding box detection but the other is polygon object detection, then the polygons will be
    # converted to bounding boxes. we want the area filters to operate after this conversion.
    gt_and_pred_tasks = [
        request_info.parameters.dataset_gt_task_type,
        request_info.parameters.model_pred_task_type,
    ]
    if schemas.Task.BBOX_OBJECT_DETECTION in gt_and_pred_tasks:
        common_task = schemas.Task.BBOX_OBJECT_DETECTION
    elif schemas.Task.POLY_OBJECT_DETECTION in gt_and_pred_tasks:
        common_task = schemas.Task.POLY_OBJECT_DETECTION
    else:
        common_task = schemas.Task.INSTANCE_SEGMENTATION

    if request_info.parameters.dataset_gt_task_type in [
        schemas.Task.BBOX_OBJECT_DETECTION,
        schemas.Task.POLY_OBJECT_DETECTION,
    ]:
        gts_statement_method = _object_detections_in_dataset_statement
    else:
        gts_statement_method = _instance_segmentations_in_dataset_statement
    gts_statement = gts_statement_method(
        request_info.parameters.dataset_name,
        min_area=request_info.parameters.min_area,
        max_area=request_info.parameters.max_area,
        task=common_task,
    )

    if request_info.parameters.model_pred_task_type in [
        schemas.Task.BBOX_OBJECT_DETECTION,
        schemas.Task.POLY_OBJECT_DETECTION,
    ]:
        preds_statement_method = _model_object_detection_preds_statement

    else:
        preds_statement_method = _model_instance_segmentation_preds_statement
    preds_statement = preds_statement_method(
        model_name=request_info.parameters.model_name,
        dataset_name=request_info.parameters.dataset_name,
        min_area=request_info.parameters.min_area,
        max_area=request_info.parameters.max_area,
        task=common_task,
    )

    (
        gts_statement,
        preds_statement,
        missing_pred_labels,
        ignored_pred_labels,
    ) = validate_requested_labels_and_get_new_defining_statements_and_missing_labels(
        db=db,
        gts_statement=gts_statement,
        preds_statement=preds_statement,
        requested_labels=request_info.labels,
    )

    return (
        gts_statement,
        preds_statement,
        missing_pred_labels,
        ignored_pred_labels,
    )


def create_ap_metrics(
    db: Session,
    gts_statement: Select,
    preds_statement: Select,
    request_info: schemas.APRequest,
) -> int:
    # need to break down preds and gts by image
    gts = db.scalars(gts_statement).all()
    preds = db.scalars(preds_statement).all()

    image_id_to_gts = {}
    image_id_to_preds = {}
    all_image_ids = set()
    for gt in gts:
        if gt.image_id not in image_id_to_gts:
            image_id_to_gts[gt.image_id] = []
        image_id_to_gts[gt.image_id].append(gt)
        all_image_ids.add(gt.image_id)
    for pred in preds:
        if pred.image_id not in image_id_to_preds:
            image_id_to_preds[pred.image_id] = []
        image_id_to_preds[pred.image_id].append(pred)
        all_image_ids.add(pred.image_id)

    all_image_ids = list(all_image_ids)

    # all_gts and all_preds are list of lists of gts and preds per image
    all_gts = []
    all_preds = []
    for image_id in all_image_ids:
        all_gts.append(image_id_to_gts.get(image_id, []))
        all_preds.append(image_id_to_preds.get(image_id, []))

    ap_metrics = compute_ap_metrics(
        db=db,
        predictions=all_preds,
        groundtruths=all_gts,
        iou_thresholds=request_info.iou_thresholds,
    )

    dataset_id = get_dataset(db, request_info.parameters.dataset_name).id
    model_id = get_model(db, request_info.parameters.model_name).id

    mp = _get_or_create_row(
        db,
        models.MetricParameters,
        mapping={
            "dataset_id": dataset_id,
            "model_id": model_id,
            "model_pred_task_type": request_info.parameters.model_pred_task_type,
            "dataset_gt_task_type": request_info.parameters.dataset_gt_task_type,
        },
    )

    ap_metric_mappings = _create_ap_metric_mappings(
        db=db, metrics=ap_metrics, metric_parameters_id=mp.id
    )

    for mapping in ap_metric_mappings:
        _get_or_create_row(db, models.APMetric, mapping)
    db.commit()

    return mp.id


def _check_finalized_inferences(
    db: Session, model_name: str, dataset_name: str
) -> bool:
    """Checks if inferences of model given by `model_name` on dataset given by `dataset_name`
    are finalized
    """
    model_id = get_model(db, model_name).id
    dataset_id = get_dataset(db, dataset_name).id
    entries = db.scalars(
        select(models.FinalizedInferences).where(
            and_(
                models.FinalizedInferences.model_id == model_id,
                models.FinalizedInferences.dataset_id == dataset_id,
            )
        )
    ).all()
    # this should never happen because of uniqueness constraint
    if len(entries) > 1:
        raise RuntimeError(
            f"got multiple entries for finalized inferences with model id {model_id} "
            f"and dataset id {dataset_id}, which should never happen"
        )

    return len(entries) != 0


def finalize_inferences(
    db: Session, model_name: str, dataset_name: str
) -> None:
    dataset = get_dataset(db, dataset_name)
    if dataset.draft:
        raise exceptions.DatasetIsDraftError(dataset_name)

    model_id = get_model(db, model_name).id
    dataset_id = dataset.id

    db.add(
        models.FinalizedInferences(dataset_id=dataset_id, model_id=model_id)
    )
    db.commit()
