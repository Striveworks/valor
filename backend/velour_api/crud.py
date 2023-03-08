from sqlalchemy import Select, func, insert, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import exceptions, models, schemas


def _wkt_polygon_from_detection(det: schemas.DetectionBase) -> str:
    """Returns the "Well-known text" format of a detection"""
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


def bulk_insert_and_return_ids(
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
            label_tuple_to_id[label_tuple] = get_or_create_row(
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
        get_or_create_row(
            db=db,
            model_class=models.Image,
            mapping={
                "dataset_id": dset_id,
                **img.dict()
            },
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

    ids = bulk_insert_and_return_ids(db, model_cls, mappings)

    label_tuple_to_id = _create_label_tuple_to_id_dict(
        db, [label for d_or_s in dets_or_segs for label in d_or_s.labels]
    )

    labeled_gt_mappings = labeled_mapping_method(
        label_tuple_to_id, ids, dets_or_segs
    )

    return bulk_insert_and_return_ids(
        db, labeled_model_cls, labeled_gt_mappings
    )


def _create_pred_dets_or_segs(
    db: Session,
    model_name: str,
    dets_or_segs: list[
        schemas.PredictedDetection | schemas.PredictedSegmentation
    ],
    mapping_method: callable,
    labeled_mapping_method: callable,
    model_cls: type,
    labeled_model_cls: type,
):
    model_id = get_model(db, model_name=model_name).id
    # get image ids from uris (these images should already exist)
    images = [get_image(db, uri=d_or_s.image.uri) for d_or_s in dets_or_segs]
    mappings = mapping_method(dets_or_segs, images)
    for m in mappings:
        m["model_id"] = model_id

    ids = bulk_insert_and_return_ids(db, model_cls, mappings)

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

    return bulk_insert_and_return_ids(
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

    return bulk_insert_and_return_ids(
        db, models.GroundTruthImageClassification, clf_mappings
    )


def create_predicted_image_classifications(
    db: Session, data: schemas.PredictedImageClassificationsCreate
):
    model_id = get_model(db, model_name=data.model_name).id
    # get image ids from uris (these images should already exist)
    image_ids = [
        get_image(db, uri=classification.image.uri).id
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

    return bulk_insert_and_return_ids(
        db, models.PredictedImageClassification, pred_mappings
    )


def get_or_create_row(
    db: Session,
    model_class: type,
    mapping: dict,
) -> any:
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


def create_dataset(db: Session, dataset: schemas.DatasetCreate, from_video=False):
    """Creates a dataset

    Raises
    ------
    DatasetAlreadyExistsError
        if the dataset name already exists
    """
    try:
        db.add(models.Dataset(name=dataset.name, draft=True, from_video=from_video))
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.DatasetAlreadyExistsError(dataset.name)


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


def create_model(db: Session, model: schemas.Model):
    """Creates a dataset

    Raises
    ------
    ModelAlreadyExistsError
        if the model uri already exists
    """
    try:
        db.add(models.Model(name=model.name))
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.ModelAlreadyExistsError(model.name)


def finalize_dataset(db: Session, dataset_name: str) -> None:
    dset = get_dataset(db, dataset_name)
    dset.draft = False
    db.commit()


def get_model(db: Session, model_name: str) -> models.Model:
    ret = db.scalar(
        select(models.Model).where(models.Model.name == model_name)
    )
    if ret is None:
        raise exceptions.ModelDoesNotExistError(model_name)

    return ret


def get_image(db: Session, uri: str) -> models.Image:
    ret = db.scalar(select(models.Image).where(models.Image.uri == uri))
    if ret is None:
        raise exceptions.ImageDoesNotExistError(uri)

    return ret


def get_labels_in_dataset(
    db: Session, dataset_name: str
) -> list[models.Label]:
    # TODO must be a better and more SQLy way of doing this
    dset = get_dataset(db, dataset_name)
    unique_ids = set()
    for image in dset.images:
        unique_ids.update(get_unique_label_ids_in_image(image))

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
    # TODO must be a better and more SQLy way of doing this
    dset = get_dataset(db, dataset_name)
    return dset.images


def get_unique_label_ids_in_image(image: models.Image) -> set[int]:
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


def delete_dataset(db: Session, dataset_name: str):
    dset = get_dataset(db, dataset_name)

    db.delete(dset)
    db.commit()


def delete_model(db: Session, model_name: str):
    model = get_model(db, model_name)

    db.delete(model)
    db.commit()


def number_of_rows(db: Session, model_cls: type) -> int:
    return db.scalar(select(func.count(model_cls.id)))
