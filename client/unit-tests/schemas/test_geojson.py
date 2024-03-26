import numpy as np
import pytest

from valor.schemas import BoundingBox, MultiPolygon, Point, Polygon, Raster


def test_point():
    # valid
    p1 = Point((1, 1))
    p2 = Point((1.0, 1.0))
    p3 = Point((1.0, 0.99))

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


def test_polygon():
    p1 = (0, 0)
    p2 = (5, 0)
    p3 = (0, 5)
    coords = [p1, p2, p3]

    # valid
    Polygon([coords])
    Polygon([coords, coords])  # defines a hole

    # test validation
    with pytest.raises(TypeError) as e:
        Polygon(123)  # type: ignore
    assert "boundary should be of type `valor.BasicPolygon`" in str(e)
    with pytest.raises(TypeError) as e:
        Polygon([poly, 123])  # type: ignore
    assert "holes should be a list of `valor.BasicPolygon`" in str(e)
    with pytest.raises(TypeError) as e:
        Polygon([poly, [123]])  # type: ignore
    assert "should contain elements of type `valor.BasicPolygon`" in str(e)


def test_boundingbox():
    p1 = (0, 0)
    p2 = (5, 0)
    p3 = (5, 5)
    p4 = (0, 5)
    coords = [[p1, p2, p3, p4, p1]]

    # test validation
    BoundingBox(coords)
    with pytest.raises(TypeError) as e:
        BoundingBox(polygon=p1)  # type: ignore
    with pytest.raises(ValueError) as e:
        BoundingBox([[p1, p2, p3, p4]])
    assert "should be made of a 4-point polygon" in str(e)

    # test classmethod `from_extrema`
    assert (
        BoundingBox.from_extrema(
            xmin=-1, xmax=10, ymin=-2, ymax=11
        ).get_value()
        == coords
    )


def test_multipolygon():
    p1 = (0, 0)
    p2 = (5, 0)
    p3 = (5, 5)
    p4 = (0, 5)
    coords = [p1, p2, p3, p4]

    # valid
    MultiPolygon([[coords]])

    # test validation
    with pytest.raises(TypeError) as e:
        MultiPolygon(coords)  # type: ignore
    with pytest.raises(TypeError) as e:
        MultiPolygon([coords])  # type: ignore
    with pytest.raises(TypeError) as e:
        MultiPolygon([[coords], 123])  # type: ignore
    with pytest.raises(TypeError) as e:
        MultiPolygon([[[coords]]])  # type: ignore


def test_raster(raster_raw_mask):
    mask1 = np.ones((10, 10)) == 1

    # valid
    Raster({"mask": "test", "geometry": None})
    Raster.from_numpy(mask=mask1)

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        Raster(mask=123)  # type: ignore
    assert "mask should be of type `str`" in str(e)

    # test classmethod `from_numpy`
    mask2 = np.ones((10, 10, 10)) == 1
    mask3 = np.ones((10, 10))
    with pytest.raises(ValueError) as e:
        Raster.from_numpy(mask2)
    assert "raster currently only supports 2d arrays" in str(e)
    with pytest.raises(ValueError) as e:
        Raster.from_numpy(mask3)
    assert "Expecting a binary mask" in str(e)

    # test member fn `to_numpy`
    r = Raster.from_numpy(raster_raw_mask)
    value = r.get_value()
    assert value
    assert (
        value["mask"]
        == "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUAQAAAACl8iCgAAAAF0lEQVR4nGP4f4CBiYGBIGZgsP9AjDoAuysDE0GVDN8AAAAASUVORK5CYII="
    )
    assert (r.array == raster_raw_mask).all()
