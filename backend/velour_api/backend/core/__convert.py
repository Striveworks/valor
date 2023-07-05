#############
# DELETE ME #
#############

import json
from typing import Optional

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

# @MARK: SQL to SQL Conversion



def convert_raster_to_polygons(
    tablename: str,
    dataset_id: int = None,
    model_id: int = None,
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
):
    """Takes either 'ground_truth_segmentation' or 'predicted_segmentation' as tablename."""

    if dataset_id is not None and model_id is None:
        criteria_id = f"(select {get_dataset_id_from_datum_id('datum_id')} limit 1) =  {dataset_id}"
    elif dataset_id is None and model_id is not None:
        criteria_id = f"model_id =  {model_id}"
    elif dataset_id is not None and model_id is not None:
        raise ValueError

    subquery = f"""
    SELECT subquery.id as id, is_instance, boundary, datum_id
    FROM(
        SELECT id, ST_Union(geom) as boundary
        FROM (
            SELECT id, ST_MakeValid((ST_DumpAsPolygons(shape)).geom) as geom
            FROM {tablename}
            {f"WHERE ({criteria_id})" if criteria_id != '' else ''}
        ) AS conversion
        GROUP BY id
    )
    AS subquery
    CROSS JOIN {tablename}
    WHERE {tablename}.id = subquery.id
    """

    criteria_area = []
    if min_area is not None:
        criteria_area.append(f"ST_AREA(boundary) >= {min_area}")
    if max_area is not None:
        criteria_area.append(f"ST_AREA(boundary) <= {max_area}")
    if criteria_area:
        return f"""
        SELECT id, is_instance, boundary, datum_id
        FROM ({subquery}) AS subquery
        WHERE {' AND '.join(criteria_area)}
        """
    else:
        return subquery


def convert_raster_to_bbox(
    tablename: str,
    dataset_id: int = None,
    model_id: int = None,
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
):
    """Takes either 'ground_truth_segmentation' or 'predicted_segmentation' as tablename."""

    if dataset_id is not None and model_id is None:
        criteria_id = f"(select {get_dataset_id_from_datum_id('datum_id')} limit 1) =  {dataset_id}"
    elif dataset_id is None and model_id is not None:
        criteria_id = f"model_id =  {model_id}"
    elif dataset_id is not None and model_id is not None:
        raise ValueError

    subquery = f"""
    SELECT subquery.id AS id, is_instance, bbox, datum_id
    FROM(
        
    )
    AS subquery
    CROSS JOIN {tablename}
    WHERE {tablename}.id = subquery.id
    """

    criteria_area = []
    if min_area is not None:
        criteria_area.append(f"ST_AREA(bbox) >= {min_area}")
    if max_area is not None:
        criteria_area.append(f"ST_AREA(bbox) <= {max_area}")

    if criteria_area:
        return f"""
        SELECT id, is_instance, bbox, datum_id
        FROM ({subquery}) AS subquery
        WHERE {' AND '.join(criteria_area)}
        """
    else:
        return subquery


def create_geometry_polygon_mappings(
    polygons: list[schemas.Polygon],
):
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
