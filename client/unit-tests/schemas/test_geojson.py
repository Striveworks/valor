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
    p1 = (-1, 0)
    p2 = (-5, 2)
    p3 = (-2, 5)
    coords = [p1, p2, p3, p1]

    # valid
    poly = Polygon([coords])
    poly_w_hole = Polygon([coords, coords])  # defines a hole

    # test validation
    with pytest.raises(ValueError):
        assert Polygon([[p1, p2, p3]])
    with pytest.raises(TypeError):
        Polygon(123)  # type: ignore
    with pytest.raises(TypeError):
        Polygon([poly, 123])  # type: ignore
    with pytest.raises(TypeError):
        Polygon([poly, [123]])  # type: ignore

    # test property 'boundary'
    assert poly.boundary == coords
    assert poly_w_hole.boundary == coords

    # test property 'holes'
    assert poly.holes == []
    assert poly_w_hole.holes == [coords]

    # test property 'xmin'
    assert poly.xmin == -5

    # test property 'xmax'
    assert poly.xmax == -1

    # test property 'ymin'
    assert poly.ymin == 0

    # test property 'ymax'
    assert poly.ymax == 5


def test_boundingbox():
    p1 = (-1, -2)
    p2 = (10, -2)
    p3 = (10, 11)
    p4 = (-1, 11)
    coords = [[p1, p2, p3, p4, p1]]

    # test validation
    BoundingBox(coords)
    with pytest.raises(TypeError) as e:
        BoundingBox(polygon=p1)  # type: ignore
    with pytest.raises(ValueError) as e:
        BoundingBox([[p1, p2, p3, p4]])
    assert "at least 4 points with the first point being repeated" in str(e)

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
    coords = [p1, p2, p3, p4, p1]

    # valid
    MultiPolygon([[coords]])

    # test validation
    with pytest.raises(TypeError):
        MultiPolygon(coords)  # type: ignore
    with pytest.raises(TypeError):
        MultiPolygon([coords])  # type: ignore
    with pytest.raises(TypeError):
        MultiPolygon([[coords], 123])  # type: ignore
    with pytest.raises(TypeError):
        MultiPolygon([[[coords]]])  # type: ignore


def test_raster(raster_raw_mask):
    mask1 = np.ones((10, 10)) == 1
    poly1 = Polygon([[(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]])
    multipoly1 = MultiPolygon([[[(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]]])

    # valid
    Raster({"mask": mask1, "geometry": None})
    Raster({"mask": mask1, "geometry": poly1.get_value()})
    Raster({"mask": mask1, "geometry": multipoly1.get_value()})
    Raster.from_numpy(mask=mask1)
    Raster.from_geometry(geometry=poly1, height=10, width=10)
    Raster.from_geometry(geometry=multipoly1, height=10, width=10)

    # test validation
    with pytest.raises(TypeError):
        assert Raster({"mask": "test", "geometry": None})
    with pytest.raises(TypeError) as e:
        assert Raster(123)

    # test classmethod `from_numpy`
    mask2 = np.ones((10, 10, 10)) == 1
    mask3 = np.ones((10, 10))
    with pytest.raises(ValueError) as e:
        Raster.from_numpy(mask2)
    assert "raster only supports 2d arrays" in str(e)
    with pytest.raises(ValueError) as e:
        Raster.from_numpy(mask3)
    assert "Expecting a binary mask" in str(e)

    # test member fn `to_numpy`
    r = Raster.from_numpy(raster_raw_mask)
    value = r.encode_value()
    assert value
    assert (
        value["mask"]
        == "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUAQAAAACl8iCgAAAAF0lEQVR4nGP4f4CBiYGBIGZgsP9AjDoAuysDE0GVDN8AAAAASUVORK5CYII="
    )
    assert (r.array == raster_raw_mask).all()
