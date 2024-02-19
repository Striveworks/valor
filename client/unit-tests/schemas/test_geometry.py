import numpy as np
import pytest

from valor import schemas


def test_point():
    # valid
    p1 = schemas.Point(x=1, y=1)
    p2 = schemas.Point(x=1.0, y=1.0)
    p3 = schemas.Point(x=1.0, y=0.99)

    # test member fn `__hash__`
    assert p1.__hash__() == p2.__hash__()
    assert p1.__hash__() != p3.__hash__()

    # test member fn `resize`
    p11 = p1.resize(
        og_img_h=10,
        og_img_w=10,
        new_img_h=100,
        new_img_w=100,
    )
    assert p11.x == p1.x * 10
    assert p11.y == p1.y * 10


def test_basicpolygon():
    p1 = schemas.Point(x=0, y=0)
    p2 = schemas.Point(x=5, y=0)
    p3 = schemas.Point(x=10, y=5)
    p4 = schemas.Point(x=0, y=5)

    schemas.BasicPolygon(points=[p1, p2, p4])

    # Test __post_init__
    with pytest.raises(TypeError) as e:
        schemas.BasicPolygon(points=p1)  # type: ignore
    assert "is not a list" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.BasicPolygon(points=[p1, p2, 4])  # type: ignore
    assert "is not a `Point`"
    with pytest.raises(ValueError) as e:
        schemas.BasicPolygon(points=[p1, p2])
    assert "needs at least 3 unique points" in str(e)

    # Test member fn `xy_list`
    poly = schemas.BasicPolygon(points=[p1, p2, p3])
    assert poly.xy_list() == [p1, p2, p3]

    # Test properties
    assert poly.xmin == 0
    assert poly.xmax == 10
    assert poly.ymin == 0
    assert poly.ymax == 5


def test_polygon():
    p1 = schemas.Point(x=0, y=0)
    p2 = schemas.Point(x=5, y=0)
    p3 = schemas.Point(x=0, y=5)
    poly = schemas.BasicPolygon(points=[p1, p2, p3])

    # valid
    schemas.Polygon(boundary=poly)
    schemas.Polygon(boundary=poly, holes=[poly])

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        schemas.Polygon(boundary=123)  # type: ignore
    assert "boundary should be of type `valor.schemas.BasicPolygon`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Polygon(boundary=poly, holes=123)  # type: ignore
    assert "holes should be a list of `valor.schemas.BasicPolygon`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.Polygon(boundary=poly, holes=[123])  # type: ignore
    assert (
        "should contain elements of type `valor.schemas.BasicPolygon`"
        in str(e)
    )


def test_boundingbox():
    p1 = schemas.Point(x=0, y=0)
    p2 = schemas.Point(x=5, y=0)
    p3 = schemas.Point(x=5, y=5)
    p4 = schemas.Point(x=0, y=5)
    poly = schemas.BasicPolygon(points=[p1, p2, p3, p4])

    # test __post_init__
    schemas.BoundingBox(polygon=poly)
    with pytest.raises(TypeError) as e:
        schemas.BoundingBox(polygon=p1)  # type: ignore
    assert "should be of type `valor.schemas.BasicPolygon`" in str(e)
    with pytest.raises(ValueError) as e:
        schemas.BoundingBox(polygon=schemas.BasicPolygon(points=[p1, p2, p3]))
    assert "should be made of a 4-point polygon" in str(e)

    # test classmethod `from_extrema`
    bbox = schemas.BoundingBox.from_extrema(xmin=-1, xmax=10, ymin=-2, ymax=11)
    assert bbox.polygon.xy_list() == [
        schemas.Point(x=-1, y=-2),
        schemas.Point(x=10, y=-2),
        schemas.Point(x=10, y=11),
        schemas.Point(x=-1, y=11),
    ]


def test_multipolygon():
    p1 = schemas.Point(x=0, y=0)
    p2 = schemas.Point(x=5, y=0)
    p3 = schemas.Point(x=5, y=5)
    p4 = schemas.Point(x=0, y=5)
    component_poly = schemas.BasicPolygon(points=[p1, p2, p3, p4])
    poly = schemas.Polygon(boundary=component_poly)

    # valid
    schemas.MultiPolygon(polygons=[poly])

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        schemas.MultiPolygon(polygons=component_poly)  # type: ignore
    assert "polygons should be list of `valor.schemas.Polyon`" in str(e)
    with pytest.raises(TypeError) as e:
        schemas.MultiPolygon(polygons=[component_poly])  # type: ignore
    assert (
        "polygons list should contain elements of type `valor.schemas.Polygon`"
        in str(e)
    )


def test_raster(raster_raw_mask):
    mask1 = np.ones((10, 10)) == 1

    # valid
    schemas.Raster(mask="test")
    schemas.Raster.from_numpy(mask=mask1)

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        schemas.Raster(mask=123)  # type: ignore
    assert "mask should be of type `str`" in str(e)

    # test classmethod `from_numpy`
    mask2 = np.ones((10, 10, 10)) == 1
    mask3 = np.ones((10, 10))
    with pytest.raises(ValueError) as e:
        schemas.Raster.from_numpy(mask2)
    assert "raster currently only supports 2d arrays" in str(e)
    with pytest.raises(ValueError) as e:
        schemas.Raster.from_numpy(mask3)
    assert "Expecting a binary mask" in str(e)

    # test member fn `to_numpy`
    r = schemas.Raster.from_numpy(raster_raw_mask)
    assert (
        r.mask
        == "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUAQAAAACl8iCgAAAAF0lEQVR4nGP4f4CBiYGBIGZgsP9AjDoAuysDE0GVDN8AAAAASUVORK5CYII="
    )
    assert (r.to_numpy() == raster_raw_mask).all()
