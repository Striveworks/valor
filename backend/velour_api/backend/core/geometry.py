from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import models


def add_bounding_box(bbox: schemas.BoundingBox):
    pass


def add_bounding_polygon(polygon: schemas.Polygon):
    pass


def add_segmentation_raster_from_multipolygon(polygons: schemas.MultiPolygon):
    pass


def add_segmentation_raster(raster: schemas.Raster):
    pass


def convert_polygon_to_bounding_box(
    dataset_id: int = None,
    model_id: int = None,
    datum_id: int = None,
    annotation_id: int = None,
):
    """Converts annotation column 'polygon' into column 'bbox'. Filter by input args."""
    pass


def convert_raster_to_bounding_box(
    dataset_id: int = None,
    model_id: int = None,
    datum_id: int = None,
    annotation_id: int = None,
):
    """Converts annotation column 'raster' into column 'bbox'."""
    pass


def convert_raster_to_bounding_polygon(
    dataset_id: int = None,
    model_id: int = None,
    datum_id: int = None,
    annotation_id: int = None,
):
    """Converts annotation column 'raster' into column 'polygon'."""
    pass


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
