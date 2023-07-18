from dataclasses import asdict
import numpy as np

import pytest

from velour import schemas
from velour.schemas.metadata import _validate_href


""" schemas.metadata """

def test_metadata_geojson():
    # @TODO: Implement GeoJSON
    schemas.GeoJSON(type="this shouldnt work", coordinates=[])


def test_metadata__validate_href():
    _validate_href("http://test")
    _validate_href("https://test")

    with pytest.raises(ValueError) as e:
        _validate_href("test")
    assert "`href` must start with http:// or https://" in e
    with pytest.raises(ValueError) as e:
        _validate_href(1)
    assert "`href` must start with http:// or https://" in e


def test_metadata_metadatum():
    schemas.Metadatum(name="test", value="test")
    schemas.Metadatum(name="test", value=1)
    schemas.Metadatum(name="test", value=1.0)
    # @TODO: Fix when geojson is implemented
    schemas.Metadatum(name="test", value=schemas.GeoJSON(type="test", coordinates=[]))

    with pytest.raises(ValueError) as e:
        schemas.Metadatum(name=123, value=123)
    assert "should always be of type string" in e

    # Test supported value types

    with pytest.raises(NotImplementedError):
        schemas.Metadatum(name="test", value=(1,2))
    assert "has unsupported type <class 'tuple'>"

    with pytest.raises(NotImplementedError):
        schemas.Metadatum(name="test", value=[1,2])
    assert "has unsupported type <class 'list'>"
    
    # Test special type with name=href

    schemas.Metadatum(name="href", value="http://test")
    schemas.Metadatum(name="href", value="https://test")

    with pytest.raises(ValueError) as e:
        schemas.Metadatum(name="href", value="test")
    assert "`href` must start with http:// or https://" in e
    with pytest.raises(ValueError) as e:
        schemas.Metadatum(name="href", value=1)
    assert "`href` must start with http:// or https://" in e

    # Check int to float conversion
    m = schemas.Metadatum(name="test", value=1)
    assert isinstance(m.value, float)


def test_geometry_point():
    schemas.Point(x=1,y=1)
    schemas.Point(x=1.0, y=1.0)
    
    with pytest.raises(ValueError) as e:
        schemas.Point(x="1", y=1)
    assert "should be `float` type" in e

    with pytest.raises(ValueError) as e:
        schemas.Point(x=1, y="1")
    assert "should be `float` type" in e


def test_geometry_box():
    p1 = schemas.Point(x=0, y=0)
    p2 = schemas.Point(x=10, y=10)
    p3 = schemas.Point(x=10, y=-10)
    p4 = schemas.Point(x=-10, y=10)

    schemas.Box(min=p1, max=p2)

    with pytest.raises(ValueError) as e:
        schemas.Box(min=p2, max=p1)
    assert "xmin > xmax"

    with pytest.raises(ValueError) as e:
        schemas.Box(min=p1, max=p4)
    assert "xmin > xmax"

    with pytest.raises(ValueError) as e:
        schemas.Box(min=p1, max=p3)
    assert "ymin > ymax"


def test_geometry_basicpolygon():
    p1 = schemas.Point(x=0, y=0)
    p2 = schemas.Point(x=5, y=0)
    p3 = schemas.Point(x=10, y=5)
    p4 = schemas.Point(x=0, y=5)

    schemas.BasicPolygon(points=[p1,p2,p4])

    # Test __post_init__
    with pytest.raises(ValueError) as e:
        schemas.BasicPolygon(points=p1)
    assert "is not a list" in e
    with pytest.raises(ValueError) as e:
        schemas.BasicPolygon(points=[p1,p2,4])
    assert "is not a `Point`"
    with pytest.raises(ValueError) as e:
        schemas.BasicPolygon(points=[p1,p2])
    assert "needs at least 3 unique points" in e

    # Test fn xy_list 
    poly = schemas.BasicPolygon(points=[p1,p2,p3])
    assert poly.xy_list() == [p1,p2,p3]

    # Test properties
    assert poly.xmin == 0
    assert poly.xmax == 10
    assert poly.ymin == 0
    assert poly.ymax == 5

    # Test `from_box` classmethod
    cmin = schemas.Point(x=-1, y=-2)
    cmax = schemas.Point(x=10, y=11)
    poly = schemas.BasicPolygon.from_box(
        box=schemas.Box(
            min=cmin,
            max=cmax,
        )
    )
    assert poly.xy_list == [
        schemas.Point(x=-1, y=-2),
        schemas.Point(x=10, y=-2),
        schemas.Point(x=10, y=11),
        schemas.Point(x=-1, y=11),
    ]


def test_geometry_polygon():
    p1 = schemas.Point(x=0, y=0)
    p2 = schemas.Point(x=5, y=0)
    p3 = schemas.Point(x=0, y=5)
    poly = schemas.BasicPolygon(points=[p1,p2,p3])

    schemas.Polygon(boundary=poly)
    schemas.Polygon(boundary=poly, holes=[poly])
    with pytest.raises(AssertionError):
        schemas.Polygon(boundary=123)
    with pytest.raises(AssertionError):
        schemas.Polygon(boundary=poly, holes=123)
    with pytest.raises(AssertionError):
        schemas.Polygon(boundary=poly, holes=[123])


def test_geometry_boundingbox():
    p1 = schemas.Point(x=0, y=0)
    p2 = schemas.Point(x=5, y=0)
    p3 = schemas.Point(x=5, y=5)
    p4 = schemas.Point(x=0, y=5)
    poly = schemas.BasicPolygon(points=[p1,p2,p3,p4])

    # test __post_init__
    schemas.BoundingBox(polygon=poly)
    with pytest.raises(AssertionError):
        schemas.BoundingBox(polygon=p1)
    with pytest.raises(ValueError) as e:
        schemas.BoundingBox(polygon=schemas.BasicPolygon(points=[p1,p2,p3]))
    assert "should be made of a 4-point polygon" in e

    # test classmethod `from_extrema`
    bbox = schemas.BoundingBox.from_extrema(xmin=-1, xmax=10, ymin=-2, ymax=11)
    assert bbox.polygon.xy_list() == [
        schemas.Point(x=-1, y=-2),
        schemas.Point(x=10, y=-2),
        schemas.Point(x=10, y=11),
        schemas.Point(x=-1, y=11),
    ]


def test_geometry_multipolygon():
    p1 = schemas.Point(x=0, y=0)
    p2 = schemas.Point(x=5, y=0)
    p3 = schemas.Point(x=5, y=5)
    p4 = schemas.Point(x=0, y=5)
    component_poly = schemas.BasicPolygon(points=[p1,p2,p3,p4])
    poly = schemas.Polygon(boundary=component_poly)

    # test `__post_init__`
    schemas.MultiPolygon(polygons=[poly])
    with pytest.raises(AssertionError):
        schemas.MultiPolygon(polygons=component_poly)
    with pytest.raises(AssertionError):
        schemas.MultiPolygon(polygons=[component_poly])


def test_raster():
    # test `__post_init__`
    with pytest.raises(AssertionError):
        schemas.Raster(mask=123, height=10, width=10)
    with pytest.raises(AssertionError):
        schemas.Raster(mask="123", height="10", width=10)
    with pytest.raises(AssertionError):
        schemas.Raster(mask="123", height=10, width="10")

    # test classmethod `from_numpy`
    mask = np.ones((10,10))
    r = schemas.Raster.from_numpy(mask=mask)
    raise NotImplementedError


def test_core_dataset():
    # valid 
    schemas.Dataset(
        name="test",
        metadata=[schemas.Metadatum(name="test", value=123)],
    )
    schemas.Dataset(
        name="test",
        id=None,
        metadata=[],
    )
    schemas.Dataset(
        name="test",
        id=1,
        metadata=[],
    )
    schemas.Dataset(
        name="test",
    )

    # test `__post_init__`
    with pytest.raises(AssertionError):
        schemas.Dataset(name=123)
    with pytest.raises(AssertionError):
        schemas.Dataset(name="123", id="123")
    with pytest.raises(AssertionError):
        schemas.Dataset(name="123", metadata=1)
    with pytest.raises(AssertionError):
        schemas.Dataset(name="123", metadata=[1])


def test_core_model():
    # valid 
    schemas.Model(
        name="test",
        metadata=[schemas.Metadatum(name="test", value=123)],
    )
    schemas.Model(
        name="test",
        id=None,
        metadata=[],
    )
    schemas.Model(
        name="test",
        id=1,
        metadata=[],
    )
    schemas.Model(
        name="test",
    )

    # test `__post_init__`
    with pytest.raises(AssertionError):
        schemas.Model(name=123)
    with pytest.raises(AssertionError):
        schemas.Model(name="123", id="123")
    with pytest.raises(AssertionError):
        schemas.Model(name="123", metadata=1)
    with pytest.raises(AssertionError):
        schemas.Model(name="123", metadata=[1])

    
def test_core_info():
    # @TODO: Not fully implemented
    info = schemas.Info()


def test_core_datum():
    schemas.Datum(uid="123")
    schemas.Datum(uid="123", metadata=[])
    schemas.Datum(uid="123", metadata=[schemas.Metadatum(name="name", value=1)])
    
    # test `__post_init__`
    with pytest.raises(AssertionError):
        schemas.Datum(uid=123)
    with pytest.raises(AssertionError):
        schemas.Datum(uid="123", metadata=1)
    with pytest.raises(AssertionError):
        schemas.Datum(uid="123", metadata=[1])

    
def test_core_annotation__():
    pass