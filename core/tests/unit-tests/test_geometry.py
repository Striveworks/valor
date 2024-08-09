import math

import numpy as np
import pytest
from valor_core.schemas import (
    Box,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    Raster,
)


@pytest.fixture
def box_points() -> list[tuple[float, float]]:
    return [
        (-5, -5),
        (5, -5),
        (5, 5),
        (-5, 5),
        (-5, -5),
    ]


@pytest.fixture
def rotated_box_points() -> list[tuple[float, float]]:
    """Same area and sides as box_points, but rotated 45 degrees."""
    d = 5.0 * math.sqrt(2)
    return [
        (0, d),
        (d, 0),
        (0, -d),
        (-d, 0),
        (0, d),
    ]


@pytest.fixture
def skewed_box_points() -> list[tuple[float, float]]:
    """Skewed box_points."""
    return [
        (0, 0),
        (10, 0),
        (15, 10),
        (5, 10),
        (0, 0),
    ]


@pytest.fixture
def raster_raw_mask() -> np.ndarray:
    """
    Creates a 2d numpy of bools of shape:
    | T  F |
    | F  T |
    """
    ones = np.ones((10, 10))
    zeros = np.zeros((10, 10))
    top = np.concatenate((ones, zeros), axis=1)
    bottom = np.concatenate((zeros, ones), axis=1)
    return np.concatenate((top, bottom), axis=0) == 1


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

    # valid
    p1 = Point(value=(3.14, -3.14))
    assert Point(value=(3.14, -3.14))
    assert Point(value=(-3.14, 3.14))

    # test type validation
    with pytest.raises(TypeError):
        Point(value=("test", 0))  # type: ignore - purposefully throwing error
    with pytest.raises(TypeError):
        Point(value=(0, "test"))  # type: ignore - purposefully throwing error

    # test geojson conversion
    geojson = {"type": "Point", "coordinates": [3.14, -3.14]}
    assert p1.to_dict() == geojson
    assert Point.from_dict(geojson).value == [3.14, -3.14]

    # test wkt conversion
    wkt = "POINT (3.14 -3.14)"
    assert p1.to_wkt() == wkt


def test_polygon(box_points, skewed_box_points, rotated_box_points):
    p1 = (-1, 0)
    p2 = (-5, 2)
    p3 = (-2, 5)
    coords = [p1, p2, p3, p1]

    # valid
    poly = Polygon([coords])
    poly_w_hole = Polygon([coords, coords])  # defines a hole

    assert poly.to_wkt() == "POLYGON ((-1 0, -5 2, -2 5, -1 0))"
    assert (
        poly.to_array() == np.array([[-1, 0], [-5, 2], [-2, 5], [-1, 0]])
    ).all()

    # test validation
    with pytest.raises(ValueError):
        assert Polygon([[p1, p2, p3]])
    with pytest.raises(TypeError):
        Polygon(123)  # type: ignore - testing
    with pytest.raises(TypeError):
        Polygon([poly, 123])  # type: ignore - testing
    with pytest.raises(TypeError):
        Polygon([poly, [123]])  # type: ignore - testing

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

    # valid
    p1 = Polygon(value=[box_points])
    p2 = Polygon(value=[skewed_box_points, box_points])
    p3 = Polygon(value=[skewed_box_points, box_points, rotated_box_points])

    # test type validation
    with pytest.raises(TypeError):
        Polygon(value=[])
    with pytest.raises(TypeError):
        Polygon(value=box_points)  # type: ignore - purposefully throwing error
    with pytest.raises(TypeError):
        Polygon(
            value=["skewed_box_points"]  # type: ignore - purposefully throwing error
        )
    with pytest.raises(TypeError):
        Polygon(value=[box_points, []])
    with pytest.raises(TypeError):
        Polygon(
            value=[box_points, 123]  # type: ignore - purposefully throwing error
        )

    # test geojson conversion
    geojson = {
        "type": "Polygon",
        "coordinates": [
            [[point[0], point[1]] for point in box_points],
            [[point[0], point[1]] for point in skewed_box_points],
        ],
    }
    assert Polygon(value=[box_points, skewed_box_points]).to_dict() == geojson
    assert Polygon.from_dict(geojson).value == [
        [[-5, -5], [5, -5], [5, 5], [-5, 5], [-5, -5]],
        [[0, 0], [10, 0], [15, 10], [5, 10], [0, 0]],
    ]

    # test wkt conversion
    assert p1.to_wkt() == "POLYGON ((-5 -5, 5 -5, 5 5, -5 5, -5 -5))"
    assert (
        p2.to_wkt()
        == "POLYGON ((0 0, 10 0, 15 10, 5 10, 0 0),(-5 -5, 5 -5, 5 5, -5 5, -5 -5))"
    )
    assert (
        p3.to_wkt()
        == "POLYGON ((0 0, 10 0, 15 10, 5 10, 0 0),(-5 -5, 5 -5, 5 5, -5 5, -5 -5),(0 7.0710678118654755, 7.0710678118654755 0, 0 -7.0710678118654755, -7.0710678118654755 0, 0 7.0710678118654755))"
    )


def test_box(box_points, skewed_box_points, rotated_box_points):
    p1 = (-1, -2)
    p2 = (10, -2)
    p3 = (10, 11)
    p4 = (-1, 11)
    coords = [[p1, p2, p3, p4, p1]]

    obj = Box(coords)
    assert obj.to_wkt() == "POLYGON ((-1 -2, 10 -2, 10 11, -1 11, -1 -2))"
    assert (
        obj.to_array()
        == np.array([[-1, -2], [10, -2], [10, 11], [-1, 11], [-1, -2]])
    ).all()

    with pytest.raises(TypeError):
        Box(polygon=p1)  # type: ignore - testing
    with pytest.raises(ValueError):
        Box([[p1, p2, p3, p4]])

    # test classmethod `from_extrema`
    assert Box.from_extrema(xmin=-1, xmax=10, ymin=-2, ymax=11).value == coords

    assert Box(value=[box_points])
    assert Box(value=[rotated_box_points])
    assert Box(value=[skewed_box_points])

    # test type validation
    with pytest.raises(ValueError):
        Box(value=[])  # type: ignore - purposefully throwing error
    with pytest.raises(ValueError):
        Box(value=[box_points, box_points])  # box does not have holes
    with pytest.raises(TypeError):  # type checking
        Box(value=1234)  # type: ignore - purposefully throwing error
    with pytest.raises(TypeError):
        Box(value=box_points[0])  # type: ignore - purposefully throwing error
    with pytest.raises(ValueError):
        box_plus_one = [[*box_points[0:-1], (10, 10), box_points[0]]]
        Box(value=box_plus_one)
    with pytest.raises(ValueError):
        box_minus_one = [[*box_points[0:-2], box_points[0]]]
        Box(value=box_minus_one)

    box_points_xmin = min([point[0] for point in box_points])
    box_points_xmax = max([point[0] for point in box_points])
    box_points_ymin = min([point[1] for point in box_points])
    box_points_ymax = max([point[1] for point in box_points])
    assert Box.from_extrema(
        xmin=box_points_xmin,
        ymin=box_points_ymin,
        xmax=box_points_xmax,
        ymax=box_points_ymax,
    ).value == [box_points]

    # test geojson conversion
    geojson = {
        "type": "Polygon",
        "coordinates": [[[point[0], point[1]] for point in box_points]],
    }
    assert Box(value=[box_points]).to_dict() == geojson
    assert Box.from_dict(geojson).value == [
        [[-5, -5], [5, -5], [5, 5], [-5, 5], [-5, -5]]
    ]

    # test wkt conversion
    assert (
        Box(value=[box_points]).to_wkt()
        == "POLYGON ((-5 -5, 5 -5, 5 5, -5 5, -5 -5))"
    )
    assert (
        Box(value=[rotated_box_points]).to_wkt()
        == "POLYGON ((0 7.0710678118654755, 7.0710678118654755 0, 0 -7.0710678118654755, -7.0710678118654755 0, 0 7.0710678118654755))"
    )
    assert (
        Box(value=[skewed_box_points]).to_wkt()
        == "POLYGON ((0 0, 10 0, 15 10, 5 10, 0 0))"
    )


def test_multipolygon(box_points, skewed_box_points, rotated_box_points):
    p1 = (0, 0)
    p2 = (5, 0)
    p3 = (5, 5)
    p4 = (0, 5)
    coords = [p1, p2, p3, p4, p1]

    # valid
    obj = MultiPolygon([[coords]])
    assert obj.to_wkt() == "MULTIPOLYGON (((0 0,5 0,5 5,0 5,0 0)))"
    assert (
        obj.to_array() == np.array([[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]])
    ).all()

    # test validation
    with pytest.raises(TypeError):
        MultiPolygon(coords)  # type: ignore - testing
    with pytest.raises(TypeError):
        MultiPolygon([coords])  # type: ignore - testing
    with pytest.raises(TypeError):
        MultiPolygon([[coords], 123])  # type: ignore - testing
    with pytest.raises(TypeError):
        MultiPolygon([[[coords]]])  # type: ignore - testing

    assert MultiPolygon(value=[[rotated_box_points]])
    assert MultiPolygon(
        value=[[skewed_box_points, box_points], [rotated_box_points]]
    )

    with pytest.raises(ValueError):
        MultiPolygon(value=[])
    with pytest.raises(ValueError):
        MultiPolygon(value=[[]])
    with pytest.raises(TypeError):
        MultiPolygon(value=box_points)
    with pytest.raises(TypeError):
        MultiPolygon(value=[box_points])
    with pytest.raises(ValueError):
        MultiPolygon(value=[[box_points], []])

    # test geojson conversion
    geojson = {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [[point[0], point[1]] for point in skewed_box_points],
                [[point[0], point[1]] for point in box_points],
            ],
            [
                [[point[0], point[1]] for point in rotated_box_points],
            ],
        ],
    }
    assert (
        MultiPolygon(
            value=[[skewed_box_points, box_points], [rotated_box_points]]
        ).to_dict()
        == geojson
    )
    assert MultiPolygon.from_dict(geojson).value == [
        [
            [[0, 0], [10, 0], [15, 10], [5, 10], [0, 0]],
            [[-5, -5], [5, -5], [5, 5], [-5, 5], [-5, -5]],
        ],
        [
            [
                [0, 7.0710678118654755],
                [7.0710678118654755, 0],
                [0, -7.0710678118654755],
                [-7.0710678118654755, 0],
                [0, 7.0710678118654755],
            ]
        ],
    ]

    # test wkt conversion
    assert (
        MultiPolygon(
            value=[[skewed_box_points, box_points], [rotated_box_points]]
        ).to_wkt()
        == "MULTIPOLYGON (((0 0,10 0,15 10,5 10,0 0),(-5 -5,5 -5,5 5,-5 5,-5 -5)),((0 7.0710678118654755,7.0710678118654755 0,0 -7.0710678118654755,-7.0710678118654755 0,0 7.0710678118654755)))"
    )


def test_raster(
    raster_raw_mask, box_points, skewed_box_points, rotated_box_points
):
    mask1 = np.ones((10, 10)) == 1
    poly1 = Polygon([[(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]])
    multipoly1 = MultiPolygon([[[(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]]])

    # valid
    assert (
        Raster({"mask": mask1, "geometry": None}).to_array()
        == np.array(
            [
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
            ]
        )
    ).all()

    assert (
        Raster.from_numpy(mask=mask1).to_array()
        == np.array(
            [
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True],
            ]
        )
    ).all()

    assert (
        Raster.from_geometry(geometry=poly1).to_array()
        == np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
    ).all()

    assert (
        Raster.from_geometry(geometry=multipoly1).to_array()
        == np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
    ).all()

    # test validation
    with pytest.raises(TypeError):
        assert Raster({"mask": "test", "geometry": None})  # type: ignore - testing
    with pytest.raises(TypeError):
        assert Raster(123)  # type: ignore - testing
    with pytest.raises(TypeError):
        Raster({"mask": mask1, "geometry": poly1})
    with pytest.raises(TypeError):
        Raster({"mask": mask1, "geometry": multipoly1})

    # test classmethod `from_numpy`
    mask2 = np.ones((10, 10, 10)) == 1
    mask3 = np.ones((10, 10))
    with pytest.raises(ValueError):
        Raster.from_numpy(mask2)
    with pytest.raises(ValueError):
        Raster.from_numpy(mask3)

    # test member fn `to_numpy`
    r = Raster.from_numpy(raster_raw_mask)
    value = r.encode_value()
    assert value
    assert (
        value["mask"]
        == "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUAQAAAACl8iCgAAAAF0lEQVR4nGP4f4CBiYGBIGZgsP9AjDoAuysDE0GVDN8AAAAASUVORK5CYII="
    )
    assert (r.to_array() == raster_raw_mask).all()

    # test how from_numpy handles non-2D arrays
    with pytest.raises(ValueError):
        Raster.from_numpy(mask=np.array([False]))

    # test how from_numpy handles non-boolean arrays
    with pytest.raises(ValueError):
        Raster.from_numpy(mask=np.array([[1, 1]]))


def test_multipoint(box_points):
    # valid
    assert MultiPoint(value=[box_points[0]])
    assert MultiPoint(
        value=[
            box_points[0],
            box_points[1],
        ]
    )
    assert MultiPoint(
        value=box_points,
    )

    # test type validation
    with pytest.raises(TypeError):
        LineString(value=[])  # type: ignore - purposefully throwing error
    with pytest.raises(TypeError):
        LineString(value="points")  # type: ignore - purposefully throwing error
    with pytest.raises(TypeError):
        LineString(value=box_points[0])  # type: ignore - purposefully throwing error
    with pytest.raises(TypeError):
        LineString(value=[1, 2])  # type: ignore - purposefully throwing error

    # test geojson conversion
    geojson = {
        "type": "MultiPoint",
        "coordinates": [[point[0], point[1]] for point in box_points],
    }
    assert MultiPoint(value=box_points).to_dict() == geojson
    assert MultiPoint.from_dict(geojson).value == [
        [-5, -5],
        [5, -5],
        [5, 5],
        [-5, 5],
        [-5, -5],
    ]

    # test wkt conversion
    wkt = "MULTIPOINT ((-5 -5), (5 -5), (5 5), (-5 5), (-5 -5))"
    assert MultiPoint(value=box_points).to_wkt() == wkt


def test_linestring(box_points):
    # valid
    assert LineString(value=box_points[0:2])
    assert LineString(
        value=box_points,
    )

    # test that linestring requires at least two points
    with pytest.raises(TypeError):
        LineString(value=[])
    with pytest.raises(TypeError):
        LineString(value=[box_points[0]])

    # test type validation
    with pytest.raises(TypeError):
        LineString(value="points")  # type: ignore - purposefully throwing error
    with pytest.raises(TypeError):
        LineString(value=[1, 2])  # type: ignore - purposefully throwing error

    # test geojson conversion
    geojson = {
        "type": "LineString",
        "coordinates": [[point[0], point[1]] for point in box_points],
    }
    assert LineString(value=box_points).to_dict() == geojson
    assert LineString.from_dict(geojson).value == [
        [-5, -5],
        [5, -5],
        [5, 5],
        [-5, 5],
        [-5, -5],
    ]

    # test wkt conversion
    wkt = "LINESTRING (-5 -5, 5 -5, 5 5, -5 5, -5 -5)"
    assert LineString(value=box_points).to_wkt() == wkt


def test_multilinestring(
    box_points,
    skewed_box_points,
):
    assert MultiLineString(value=[box_points])
    assert MultiLineString(value=[box_points, box_points])

    # test type validation
    with pytest.raises(ValueError):
        MultiLineString(value=[])
    with pytest.raises(TypeError):
        MultiLineString(
            value=[
                box_points[0],
                box_points[1],
            ]
        )
    with pytest.raises(TypeError):
        MultiLineString(
            value=[
                box_points[0],
                box_points[1],
                (1, 3),  # type: ignore - purposefully throwing error
            ]
        )

    # test geojson conversion
    geojson = {
        "type": "MultiLineString",
        "coordinates": [
            [[point[0], point[1]] for point in box_points],
            [[point[0], point[1]] for point in skewed_box_points],
        ],
    }
    assert (
        MultiLineString(value=[box_points, skewed_box_points]).to_dict()
        == geojson
    )
    assert MultiLineString.from_dict(geojson).value == [
        [[-5, -5], [5, -5], [5, 5], [-5, 5], [-5, -5]],
        [[0, 0], [10, 0], [15, 10], [5, 10], [0, 0]],
    ]

    # test wkt conversion
    wkt = "MULTILINESTRING ((-5 -5, 5 -5, 5 5, -5 5, -5 -5),(0 0, 10 0, 15 10, 5 10, 0 0))"
    assert (
        MultiLineString(value=[box_points, skewed_box_points]).to_wkt() == wkt
    )
