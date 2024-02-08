import operator

import pytest
from sqlalchemy import and_, func, not_

from velour_api.backend import models
from velour_api.backend.query.filtering import (
    _filter_by_metadatum,
    _get_boolean_op,
    _get_numeric_op,
    _get_spatial_op,
    _get_string_op,
    filter_by_annotation,
)
from velour_api.schemas import Filter, NumericFilter


def test__get_boolean_op():
    assert _get_boolean_op("==") == operator.eq
    assert _get_boolean_op("!=") == operator.ne

    with pytest.raises(ValueError):
        _get_boolean_op(">")
    with pytest.raises(ValueError):
        _get_boolean_op("<")
    with pytest.raises(ValueError):
        _get_boolean_op(">=")
    with pytest.raises(ValueError):
        _get_boolean_op("<=")

    with pytest.raises(ValueError):
        _get_boolean_op("intersect")
    with pytest.raises(ValueError):
        _get_boolean_op("inside")
    with pytest.raises(ValueError):
        _get_boolean_op("outside")


def test__get_string_op():
    assert _get_string_op("==") == operator.eq
    assert _get_string_op("!=") == operator.ne

    with pytest.raises(ValueError):
        _get_string_op(">")
    with pytest.raises(ValueError):
        _get_string_op("<")
    with pytest.raises(ValueError):
        _get_string_op(">=")
    with pytest.raises(ValueError):
        _get_string_op("<=")

    with pytest.raises(ValueError):
        _get_string_op("intersect")
    with pytest.raises(ValueError):
        _get_string_op("inside")
    with pytest.raises(ValueError):
        _get_string_op("outside")


def test__get_numeric_op():
    assert _get_numeric_op("==") == operator.eq
    assert _get_numeric_op("!=") == operator.ne
    assert _get_numeric_op(">") == operator.gt
    assert _get_numeric_op("<") == operator.lt
    assert _get_numeric_op(">=") == operator.ge
    assert _get_numeric_op("<=") == operator.le

    with pytest.raises(ValueError):
        _get_numeric_op("intersect")
    with pytest.raises(ValueError):
        _get_numeric_op("inside")
    with pytest.raises(ValueError):
        _get_numeric_op("outside")


def test__get_spatial_op():
    assert str(_get_spatial_op("intersect")(1, 2)) == str(
        func.ST_Intersects(1, 2)
    )
    assert str(_get_spatial_op("inside")(1, 2)) == str(func.ST_Covers(2, 1))
    assert str(_get_spatial_op("outside")(1, 2)) == str(
        not_(func.ST_Covers(2, 1))
    )

    with pytest.raises(ValueError):
        _get_spatial_op("==")
    with pytest.raises(ValueError):
        _get_spatial_op("!=")
    with pytest.raises(ValueError):
        _get_spatial_op(">")
    with pytest.raises(ValueError):
        _get_spatial_op("<")
    with pytest.raises(ValueError):
        _get_spatial_op(">=")
    with pytest.raises(ValueError):
        _get_spatial_op("<=")


def test__filter_by_metadatum():
    with pytest.raises(NotImplementedError):
        _filter_by_metadatum(
            key="key", value_filter="not_valid_type", table=models.Dataset
        )


def test_filter_by_annotation_box():
    filter_ = Filter(require_bounding_box=True)
    assert str(filter_by_annotation(filter_)[0]) == str(
        models.Annotation.box.isnot(None)
    )

    filter_ = Filter(require_bounding_box=False)
    assert str(filter_by_annotation(filter_)[0]) == str(
        models.Annotation.box.is_(None)
    )

    filter_ = Filter(
        bounding_box_area=[NumericFilter(value=100.0, operator=">=")]
    )
    assert str(filter_by_annotation(filter_)[0]) == str(
        func.ST_Area(models.Annotation.box) >= 100.0
    )

    filter_ = Filter(
        bounding_box_area=[
            NumericFilter(value=100.0, operator=">="),
            NumericFilter(value=200.0, operator="<"),
        ]
    )
    assert str(filter_by_annotation(filter_)[0]) == str(
        and_(
            func.ST_Area(models.Annotation.box) >= 100.0,
            func.ST_Area(models.Annotation.box) < 200.0,
        )
    )


def test_filter_by_annotation_polygon():
    filter_ = Filter(require_polygon=True)
    assert str(filter_by_annotation(filter_)[0]) == str(
        models.Annotation.polygon.isnot(None)
    )

    filter_ = Filter(require_polygon=False)
    assert str(filter_by_annotation(filter_)[0]) == str(
        models.Annotation.polygon.is_(None)
    )

    filter_ = Filter(polygon_area=[NumericFilter(value=100.0, operator=">=")])
    assert str(filter_by_annotation(filter_)[0]) == str(
        func.ST_Area(models.Annotation.polygon) >= 100.0
    )

    filter_ = Filter(
        polygon_area=[
            NumericFilter(value=100.0, operator=">="),
            NumericFilter(value=200.0, operator="<"),
        ]
    )
    assert str(filter_by_annotation(filter_)[0]) == str(
        and_(
            func.ST_Area(models.Annotation.polygon) >= 100.0,
            func.ST_Area(models.Annotation.polygon) < 200.0,
        )
    )


def test_filter_by_annotation_multipolygon():
    filter_ = Filter(require_multipolygon=True)
    assert str(filter_by_annotation(filter_)[0]) == str(
        models.Annotation.multipolygon.isnot(None)
    )

    filter_ = Filter(require_multipolygon=False)
    assert str(filter_by_annotation(filter_)[0]) == str(
        models.Annotation.multipolygon.is_(None)
    )

    filter_ = Filter(
        multipolygon_area=[NumericFilter(value=100.0, operator=">=")]
    )
    assert str(filter_by_annotation(filter_)[0]) == str(
        func.ST_Area(models.Annotation.multipolygon) >= 100.0
    )

    filter_ = Filter(
        multipolygon_area=[
            NumericFilter(value=100.0, operator=">="),
            NumericFilter(value=200.0, operator="<"),
        ]
    )
    assert str(filter_by_annotation(filter_)[0]) == str(
        and_(
            func.ST_Area(models.Annotation.multipolygon) >= 100.0,
            func.ST_Area(models.Annotation.multipolygon) < 200.0,
        )
    )


def test_filter_by_annotation_raster():
    filter_ = Filter(require_raster=True)
    assert str(filter_by_annotation(filter_)[0]) == str(
        models.Annotation.raster.isnot(None)
    )

    filter_ = Filter(require_raster=False)
    assert str(filter_by_annotation(filter_)[0]) == str(
        models.Annotation.raster.is_(None)
    )

    filter_ = Filter(raster_area=[NumericFilter(value=100.0, operator=">=")])
    assert str(filter_by_annotation(filter_)[0]) == str(
        func.ST_Count(models.Annotation.raster) >= 100.0
    )

    filter_ = Filter(
        raster_area=[
            NumericFilter(value=100.0, operator=">="),
            NumericFilter(value=200.0, operator="<"),
        ]
    )
    assert str(filter_by_annotation(filter_)[0]) == str(
        and_(
            func.ST_Count(models.Annotation.raster) >= 100.0,
            func.ST_Count(models.Annotation.raster) < 200.0,
        )
    )
