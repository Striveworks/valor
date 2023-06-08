import json

from geoalchemy2.functions import ST_GeomFromGeoJSON
from sqlalchemy import Select, TextualSelect, and_, insert, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import enums, exceptions, models, schemas
from velour_api.metrics import compute_ap_metrics, compute_clf_metrics

from ._read import (
    _classifications_in_dataset_statement,
    _instance_segmentations_in_dataset_statement,
    _model_classifications_preds_statement,
    _model_instance_segmentation_preds_statement,
    _model_object_detection_preds_statement,
    _object_detections_in_dataset_statement,
    get_dataset,
    get_dataset_task_types,
    get_image,
    get_model,
    get_model_task_types,
)


def _labels_in_query(db: Session, query_statement: TextualSelect) -> list[str]:
    label_ids = db.scalars(
        select(query_statement.alias().c.label_id).distinct()
    ).all()
    return db.scalars(
        select(models.Label).where(models.Label.id.in_(label_ids))
    ).all()


def _label_keys_in_query(db: Session, query_statement: Select) -> list[str]:
    return db.scalars(
        (select(models.Label.key).join(query_statement.subquery())).distinct()
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
    detections: list[schemas.DetectionBase], images: list[models.Datum]
) -> list[dict[str, str]]:
    return [
        {
            "boundary": _wkt_polygon_from_detection(detection),
            "datum_id": image.id,
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
    images: list[models.Datum],
) -> list[dict[str, str]]:
    assert len(segmentations) == len(images)

    def _create_single_mapping(
        seg: schemas.GroundTruthSegmentation, image: models.Datum
    ):
        if seg.is_poly:
            shape = _select_statement_from_poly(seg.shape)
        else:
            shape = seg.mask_bytes

        return {
            "is_instance": seg.is_instance,
            "shape": shape,
            "datum_id": image.id,
        }

    return [
        _create_single_mapping(segmentation, image)
        for segmentation, image in zip(segmentations, images)
    ]


def _create_pred_segmentation_mappings(
    segmentations: list[schemas.PredictedSegmentation],
    images: list[models.Datum],
) -> list[dict[str, str]]:
    return [
        {
            "is_instance": segmentation.is_instance,
            "shape": segmentation.mask_bytes,
            "datum_id": image.id,
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


def _metadatum_mapping(metadatum: schemas.DatumMetadatum) -> dict:
    ret = {"name": metadatum.name}
    val = metadatum.value
    if isinstance(val, str):
        ret["string_value"] = val
    elif isinstance(val, float):
        ret["numeric_value"] = val
    elif isinstance(val, dict):
        ret["geo"] = ST_GeomFromGeoJSON(json.dumps(val))
    else:
        raise ValueError(
            f"Got unexpected value {metadatum.value} for metadatum"
        )
    return ret


def _add_datums_to_dataset(
    db: Session, dataset_name, datums: list[schemas.Datum]
) -> list[models.Datum]:
    """Adds images defined by URIs to a dataset (creating the Image rows if they don't exist),
    returning the list of image ids"""
    dset = get_dataset(db, dataset_name=dataset_name)
    if not dset.draft:
        raise exceptions.DatasetIsFinalizedError(dataset_name)
    dset_id = dset.id

    db_datums = [
        _get_or_create_row(
            db=db,
            model_class=models.Datum,
            mapping={
                "dataset_id": dset_id,
                **datum.dict(exclude={"metadata"}),
            },
        )
        for datum in datums
    ]

    for datum, db_datum in zip(datums, db_datums):
        for metadatum in datum.metadata:
            metadatum_id = _get_or_create_row(
                db=db,
                model_class=models.Metadatum,
                mapping=_metadatum_mapping(metadatum=metadatum),
            ).id

            db.add(
                models.DatumMetadatumLink(
                    datum_id=db_datum.id, metadatum_id=metadatum_id
                )
            )

    db.commit()
    return db_datums


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
    images = _add_datums_to_dataset(
        db=db,
        dataset_name=dataset_name,
        datums=[d_or_s.image for d_or_s in dets_or_segs],
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


def create_ground_truth_classifications(
    db: Session, data: schemas.GroundTruthClassificationsCreate
):
    images = _add_datums_to_dataset(
        db=db,
        dataset_name=data.dataset_name,
        datums=[c.datum for c in data.classifications],
    )
    label_tuple_to_id = _create_label_tuple_to_id_dict(
        db, [label for clf in data.classifications for label in clf.labels]
    )
    clf_mappings = [
        {"label_id": label_tuple_to_id[tuple(label)], "datum_id": image.id}
        for clf, image in zip(data.classifications, images)
        for label in clf.labels
    ]

    return _bulk_insert_and_return_ids(
        db, models.GroundTruthClassification, clf_mappings
    )


def create_predicted_image_classifications(
    db: Session, data: schemas.PredictedClassificationsCreate
):
    model_id = get_model(db, model_name=data.model_name).id
    # get image ids from uids (these images should already exist)
    datum_ids = [
        get_image(
            db,
            uid=classification.datum.uid,
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
            "datum_id": datum_id,
            "model_id": model_id,
        }
        for clf, datum_id in zip(data.classifications, datum_ids)
        for scored_label in clf.scored_labels
    ]

    return _bulk_insert_and_return_ids(
        db, models.PredictedClassification, pred_mappings
    )


def _get_or_create_row(
    db: Session,
    model_class: type,
    mapping: dict,
    columns_to_ignore: list[str] = None,
) -> any:
    """Tries to get the row defined by mapping. If that exists then
    its mapped object is returned. Otherwise a row is created by `mapping` and the newly created
    object is returned. `columns_to_ignore` specifies any columns to ignore in forming the where
    expression. this can be used for numerical columns that might slightly differ but are essentially the same
    (and where the other columns serve as unique identifiers)
    """
    columns_to_ignore = columns_to_ignore or []
    # create the query from the mapping
    where_expressions = [
        (getattr(model_class, k) == v)
        for k, v in mapping.items()
        if k not in columns_to_ignore
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


def create_dataset(db: Session, dataset: schemas.DatasetCreate):
    """Creates a dataset

    Raises
    ------
    DatasetAlreadyExistsError
        if the dataset name already exists
    """
    try:
        db.add(models.Dataset(draft=True, **dataset.dict()))
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
        db.add(models.Model(**model.dict()))
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
    label_map = _label_key_value_to_id(
        db=db,
        labels=set(
            [
                (metric.label.key, metric.label.value)
                for metric in metrics
                if hasattr(metric, "label")
            ]
        ),
    )
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


def get_filtered_preds_statement_and_missing_labels(
    db: Session,
    gts_statement: TextualSelect,
    preds_statement: TextualSelect,
) -> tuple[Select, Select, list[schemas.Label], list[schemas.Label]]:
    """Takes statements defining a collection of labeled groundtruths and labeled predictions,
    and creates a new statement for predictions that only have labels in the list of groundtruth labels.


    Parameters
    ----------
    db
    gts_statement
        the select statement that defines the colllection of labeled groundtruths
    preds_statement
        the select statement that defines the colllection of labeled predictions

    Returns
    -------
    a tuple with the following elements:
        - select statement defining the predictions with only those with labels in the groundtruth set
        - list of key/value label tuples of requested labels (or labels in the groundtruth collection
        in the case that `requested_labels` is None) that are not present in the predictions collection
        - list of key/value label tuples of labels that are present in the predictions collection but
        are not in `requested_labels` (or the labels in the groundtruth collection in the case that
        `requested_labels` is None)
    """

    available_labels = _labels_in_query(db, gts_statement)

    label_tuples = set(
        [(label.key, label.value) for label in available_labels]
    )
    pred_label_tuples = set(
        [
            (label.key, label.value)
            for label in _labels_in_query(db, preds_statement)
        ]
    )
    labels_to_use_ids = [label.id for label in available_labels]

    missing_pred_labels = label_tuples - pred_label_tuples
    ignored_pred_labels = pred_label_tuples - label_tuples

    # convert back to labels
    missing_pred_labels = [
        schemas.Label.from_key_value_tuple(la) for la in missing_pred_labels
    ]
    ignored_pred_labels = [
        schemas.Label.from_key_value_tuple(la) for la in ignored_pred_labels
    ]

    preds_statement = preds_statement.alias()

    seg_or_det_id_col = None
    if "segmentation_id" in preds_statement.c.keys():
        seg_or_det_id_col = preds_statement.c.segmentation_id
    elif "detection_id" in preds_statement.c.keys():
        seg_or_det_id_col = preds_statement.c.detection_id
    else:
        raise RuntimeError(
            "Expected `preds_statement` to have a column 'segmentation_id' or 'detection_id'."
        )

    preds_statement = (
        select(
            preds_statement.c.id,
            seg_or_det_id_col,
            preds_statement.c.label_id,
        )
        .select_from(preds_statement)
        .where(preds_statement.c.label_id.in_(labels_to_use_ids))
    )

    return (
        preds_statement,
        missing_pred_labels,
        ignored_pred_labels,
    )


def _check_dataset_and_inferences_finalized(
    db: Session, evaluation_settings: schemas.EvaluationSettings
):
    dataset_name = evaluation_settings.dataset_name
    model_name = evaluation_settings.model_name
    if get_dataset(db, dataset_name).draft:
        raise exceptions.DatasetIsDraftError(dataset_name)
    # check that inferences are finalized
    if not _check_finalized_inferences(
        db, model_name=model_name, dataset_name=dataset_name
    ):
        raise exceptions.InferencesAreNotFinalizedError(
            dataset_name=dataset_name, model_name=model_name
        )


def _validate_and_update_evaluation_settings_task_type_for_detection(
    db: Session, evaluation_settings: schemas.EvaluationSettings
) -> None:
    """If the model or dataset task types are none, then get these from the
    datasets themselves. In either case verify that these task types are compatible
    for detection evaluation.
    """
    _check_dataset_and_inferences_finalized(db, evaluation_settings)

    dataset_name = evaluation_settings.dataset_name
    model_name = evaluation_settings.model_name

    # do some validation
    allowable_tasks = set(
        [
            schemas.Task.BBOX_OBJECT_DETECTION,
            schemas.Task.POLY_OBJECT_DETECTION,
            schemas.Task.INSTANCE_SEGMENTATION,
        ]
    )

    if evaluation_settings.dataset_gt_task_type is None:
        dset_task_types = get_dataset_task_types(db, dataset_name)
        inter = allowable_tasks.intersection(dset_task_types)
        if len(inter) > 1:
            raise RuntimeError(
                f"The dataset has the following tasks compatible for object detection evaluation: {dset_task_types}. Which one to use must be specified."
            )
        if len(inter) == 0:
            raise RuntimeError(
                "The dataset does not have any annotations to support object detection evaluation."
            )
        evaluation_settings.dataset_gt_task_type = inter.pop()
    elif evaluation_settings.dataset_gt_task_type not in allowable_tasks:
        raise ValueError(
            f"`dataset_gt_task_type` must be one of {allowable_tasks} but got {evaluation_settings.dataset_gt_task_type}."
        )

    if evaluation_settings.model_pred_task_type is None:
        model_task_types = get_model_task_types(
            db, model_name=model_name, dataset_name=dataset_name
        )
        inter = allowable_tasks.intersection(model_task_types)
        if len(inter) > 1:
            raise RuntimeError(
                f"The model has the following tasks compatible for object detection evaluation: {model_task_types}. Which one to use must be specified."
            )
        if len(inter) == 0:
            raise RuntimeError(
                "The model does not have any inferences to support object detection evaluation."
            )
        evaluation_settings.model_pred_task_type = inter.pop()
    elif evaluation_settings.model_pred_task_type not in allowable_tasks:
        raise ValueError(
            f"`pred_type` must be one of {allowable_tasks} but got {evaluation_settings.model_pred_task_type}."
        )


def validate_create_ap_metrics(
    db: Session, request_info: schemas.APRequest
) -> tuple[Select, list[schemas.Label], list[schemas.Label]]:
    """Validates request_info and produces select statements for grabbing groundtruth and
    prediction data

    Returns
    -------
    tuple[Select, Select, list[schemas.Label], list[schemas.Label]]
        first element is the select statement for groundtruths, second is the select statement
        for predictions, third is list of labels that were missing in the predictions, and fourth
        is list of labels in the predictions that were ignored (because they weren't in the groundtruth)
    """

    _validate_and_update_evaluation_settings_task_type_for_detection(
        db, evaluation_settings=request_info.settings
    )

    # when computing AP, the fidelity of a detection will drop to the minimum fidelity of the groundtruth and predicted
    # task type. e.g. if one is bounding box detection but the other is polygon object detection, then the polygons will be
    # converted to bounding boxes. we want the area filters to operate after this conversion.
    gt_and_pred_tasks = [
        request_info.settings.dataset_gt_task_type,
        request_info.settings.model_pred_task_type,
    ]
    if schemas.Task.BBOX_OBJECT_DETECTION in gt_and_pred_tasks:
        common_task = schemas.Task.BBOX_OBJECT_DETECTION
    elif schemas.Task.POLY_OBJECT_DETECTION in gt_and_pred_tasks:
        common_task = schemas.Task.POLY_OBJECT_DETECTION
    else:
        common_task = schemas.Task.INSTANCE_SEGMENTATION

    if request_info.settings.dataset_gt_task_type in [
        schemas.Task.BBOX_OBJECT_DETECTION,
        schemas.Task.POLY_OBJECT_DETECTION,
    ]:
        gts_statement = _object_detections_in_dataset_statement(
            dataset_name=request_info.settings.dataset_name,
            task=request_info.settings.dataset_gt_task_type,
            min_area=request_info.settings.min_area,
            max_area=request_info.settings.max_area,
            task_for_area_computation=common_task,
        )
    else:
        gts_statement = _instance_segmentations_in_dataset_statement(
            dataset_name=request_info.settings.dataset_name,
            min_area=request_info.settings.min_area,
            max_area=request_info.settings.max_area,
            task_for_area_computation=common_task,
        )

    if request_info.settings.model_pred_task_type in [
        schemas.Task.BBOX_OBJECT_DETECTION,
        schemas.Task.POLY_OBJECT_DETECTION,
    ]:
        preds_statement = _model_object_detection_preds_statement(
            model_name=request_info.settings.model_name,
            dataset_name=request_info.settings.dataset_name,
            task=request_info.settings.model_pred_task_type,
            min_area=request_info.settings.min_area,
            max_area=request_info.settings.max_area,
            task_for_area_computation=common_task,
        )
    else:
        preds_statement = _model_instance_segmentation_preds_statement(
            model_name=request_info.settings.model_name,
            dataset_name=request_info.settings.dataset_name,
            min_area=request_info.settings.min_area,
            max_area=request_info.settings.max_area,
            task_for_area_computation=common_task,
        )

    (
        preds_statement,
        missing_pred_labels,
        ignored_pred_labels,
    ) = get_filtered_preds_statement_and_missing_labels(
        db=db, gts_statement=gts_statement, preds_statement=preds_statement
    )

    return (
        gts_statement,
        preds_statement,
        missing_pred_labels,
        ignored_pred_labels,
    )


def validate_create_clf_metrics(
    db: Session, request_info: schemas.ClfMetricsRequest
) -> tuple[list[str], list[str]]:
    _check_dataset_and_inferences_finalized(db, request_info.settings)

    gts_statement = _classifications_in_dataset_statement(
        request_info.settings.dataset_name
    )
    gts_label_keys = _label_keys_in_query(db, gts_statement)

    preds_statement = _model_classifications_preds_statement(
        model_name=request_info.settings.model_name,
        dataset_name=request_info.settings.dataset_name,
    )
    preds_label_keys = _label_keys_in_query(db, preds_statement)

    missing_pred_keys = [
        k for k in gts_label_keys if k not in preds_label_keys
    ]
    ignored_pred_keys = [
        k for k in preds_label_keys if k not in gts_label_keys
    ]

    return missing_pred_keys, ignored_pred_keys


def create_ap_metrics(
    db: Session,
    request_info: schemas.APRequest,
) -> int:
    
    dataset_id = get_dataset(db, request_info.settings.dataset_name).id
    model_id = get_model(db, request_info.settings.model_name).id
    min_area     = request_info.settings.min_area
    max_area     = request_info.settings.max_area
    gt_type      = request_info.settings.dataset_gt_task_type
    pd_type      = request_info.settings.model_pred_task_type

    print(max_area)

    metrics = compute_ap_metrics(
        db=db,
        dataset_id=dataset_id,
        model_id=model_id,
        gt_type=gt_type,
        pd_type=pd_type,
        label_key='name',
        iou_thresholds=request_info.iou_thresholds,
        ious_to_keep=request_info.ious_to_keep,
        min_area=min_area,
        max_area=max_area,
    )

    dataset_id = get_dataset(db, request_info.settings.dataset_name).id
    model_id = get_model(db, request_info.settings.model_name).id

    mp = _get_or_create_row(
        db,
        models.EvaluationSettings,
        mapping={
            "dataset_id": dataset_id,
            "model_id": model_id,
            "model_pred_task_type": pd_type,
            "dataset_gt_task_type": gt_type,
            "min_area": request_info.settings.min_area,
            "max_area": request_info.settings.max_area,
        },
    )

    metric_mappings = _create_metric_mappings(
        db=db, metrics=metrics, evaluation_settings_id=mp.id
    )

    for mapping in metric_mappings:
        # ignore value since the other columns are unique identifiers
        # and have empircally noticed value can slightly change due to floating
        # point errors
        _get_or_create_row(
            db, models.Metric, mapping, columns_to_ignore=["value"]
        )
    db.commit()

    return mp.id

def create_clf_metrics(
    db: Session,
    request_info: schemas.ClfMetricsRequest,
) -> int:
    confusion_matrices, metrics = compute_clf_metrics(
        db=db,
        dataset_name=request_info.settings.dataset_name,
        model_name=request_info.settings.model_name,
    )

    dataset_id = get_dataset(db, request_info.settings.dataset_name).id
    model_id = get_model(db, request_info.settings.model_name).id

    es = _get_or_create_row(
        db,
        models.EvaluationSettings,
        mapping={
            "dataset_id": dataset_id,
            "model_id": model_id,
            "model_pred_task_type": enums.Task.CLASSIFICATION,
            "dataset_gt_task_type": enums.Task.CLASSIFICATION,
        },
    )

    confusion_matrices_mappings = _create_metric_mappings(
        db=db, metrics=confusion_matrices, evaluation_settings_id=es.id
    )
    for mapping in confusion_matrices_mappings:
        _get_or_create_row(db, models.ConfusionMatrix, mapping)

    metric_mappings = _create_metric_mappings(
        db=db, metrics=metrics, evaluation_settings_id=es.id
    )
    for mapping in metric_mappings:
        # ignore value since the other columns are unique identifiers
        # and have empircally noticed value can slightly change due to floating
        # point errors
        _get_or_create_row(
            db, models.Metric, mapping, columns_to_ignore=["value"]
        )

    db.commit()

    return es.id


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
