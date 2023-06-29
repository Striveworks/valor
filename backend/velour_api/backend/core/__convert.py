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


def convert_polygons_to_bbox(
    tablename: str,
    dataset_id: int = None,
    model_id: int = None,
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
):
    """Takes either 'ground_truth_detection' or 'predicted_detection' as tablename."""

    if dataset_id is not None and model_id is None:
        criteria_id = f"(select {get_dataset_id_from_datum_id('datum_id')} limit 1) =  {dataset_id}"
    elif dataset_id is None and model_id is not None:
        criteria_id = f"model_id =  {model_id}"
    elif dataset_id is not None and model_id is not None:
        raise ValueError

    subquery = f"""
    SELECT subquery.id as id, bbox, datum_id
    FROM(
        SELECT id, ST_Envelope(ST_Union(boundary)) as bbox
        FROM {tablename}
        {f"WHERE ({criteria_id})" if criteria_id else ''}
        GROUP BY id
    ) AS subquery
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
        SELECT id, bbox, datum_id
        FROM ({subquery}) AS subquery
        WHERE {' AND '.join(criteria_area)}
        """
    else:
        return subquery


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
        SELECT id, ST_Envelope(ST_Union(geom)) as bbox
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
