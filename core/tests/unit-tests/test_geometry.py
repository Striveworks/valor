import numpy as np
import pytest
from valor_core import geometry
from valor_core.schemas import (
    Box,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
    Raster,
)


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
    assert poly.to_coordinates() == [
        [
            {"x": -1, "y": 0},
            {"x": -5, "y": 2},
            {"x": -2, "y": 5},
            {"x": -1, "y": 0},
        ]
    ]

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
        == "POLYGON ((0 0, 10 0, 15 10, 5 10, 0 0),(-5 -5, 5 -5, 5 5, -5 5, -5 -5),(0 -7.0710678118654755, 7.0710678118654755 0, 0 7.0710678118654755, -7.0710678118654755 0, 0 -7.0710678118654755))"
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
    assert obj.to_coordinates() == [
        [
            {"x": -1, "y": -2},
            {"x": 10, "y": -2},
            {"x": 10, "y": 11},
            {"x": -1, "y": 11},
            {"x": -1, "y": -2},
        ]
    ]

    with pytest.raises(TypeError):
        Box(polygon=p1)  # type: ignore - testing
    with pytest.raises(ValueError):
        Box([[p1, p2, p3, p4]])

    # test classmethod `from_extrema`
    assert Box.from_extrema(xmin=-1, xmax=10, ymin=-2, ymax=11).value == coords

    assert Box(value=[box_points])

    assert Box(value=[rotated_box_points])

    with pytest.raises(NotImplementedError):
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
        == "POLYGON ((0 -7.0710678118654755, 7.0710678118654755 0, 0 7.0710678118654755, -7.0710678118654755 0, 0 -7.0710678118654755))"
    )


def test_raster(
    raster_raw_mask, box_points, skewed_box_points, rotated_box_points
):
    mask1 = np.ones((10, 10)) == 1

    # valid
    assert (
        Raster(mask=mask1).to_array()
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
        Raster(mask=mask1).to_array()
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

    # test validation
    with pytest.raises(TypeError):
        assert Raster({"mask": "test", "geometry": None})  # type: ignore - testing
    with pytest.raises(TypeError):
        assert Raster(123)  # type: ignore - testing

    mask2 = np.ones((10, 10, 10)) == 1
    mask3 = np.ones((10, 10))
    with pytest.raises(ValueError):
        Raster(mask2)
    with pytest.raises(ValueError):
        Raster(mask3)

    # test member fn `to_numpy`
    r = Raster(raster_raw_mask)
    value = r.encode_value()
    assert value
    assert (
        value["mask"]
        == "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUAQAAAACl8iCgAAAAF0lEQVR4nGP4f4CBiYGBIGZgsP9AjDoAuysDE0GVDN8AAAAASUVORK5CYII="
    )
    assert (r.to_array() == raster_raw_mask).all()

    # test  non-2D arrays
    with pytest.raises(ValueError):
        Raster(mask=np.array([False]))

    # test non-boolean arrays
    with pytest.raises(ValueError):
        Raster(mask=np.array([[1, 1]]))


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


def test_convert_coordinates_to_raster():
    coordinates = [
        [
            {"x": 1, "y": 1},
            {"x": 3, "y": 1},
            {"x": 3, "y": 3},
            {"x": 1, "y": 3},
        ]
    ]
    height = 5
    width = 5
    expected_output = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    raster = Raster.from_coordinates(coordinates, height, width).to_array()
    assert np.array_equal(raster, expected_output)  # type: ignore - numpy typing error

    # test empty coordinates
    coordinates = []
    height = 5
    width = 5
    expected_output = np.zeros((5, 5), dtype=np.uint8)

    raster = Raster.from_coordinates(coordinates, height, width).to_array()
    assert np.array_equal(raster, expected_output)  # type: ignore - numpy typing error

    # test invalid contours
    coordinates = [[{"x": 1, "y": 1}]]  # Invalid contour (only 1 point)
    height = 5
    width = 5
    expected_output = np.zeros((5, 5), dtype=np.uint8)

    raster = Raster.from_coordinates(coordinates, height, width).to_array()
    assert np.array_equal(raster, expected_output)  # type: ignore - numpy typing error

    # test multiple contours
    coordinates = [
        [
            {"x": 1, "y": 1},
            {"x": 3, "y": 1},
            {"x": 3, "y": 3},
            {"x": 1, "y": 3},
        ],
        [
            {"x": 0, "y": 0},
            {"x": 1, "y": 0},
            {"x": 1, "y": 2},
            {"x": 0, "y": 2},
        ],
    ]
    height = 5
    width = 5
    expected_output = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    raster = Raster.from_coordinates(coordinates, height, width).to_array()
    assert np.array_equal(raster, expected_output)  # type: ignore - numpy typing error

    # test errors
    with pytest.raises(TypeError):
        Raster.from_coordinates(
            [
                [[1, 1], [1, 2], [3, 1], [4, 1]],
            ],  # type: ignore
            height,
            width,
        )

    with pytest.raises(TypeError):
        Raster.from_coordinates(
            [
                [
                    {"x": 1, "y": 1},
                    {"x": 3, "y": 1},
                    {"bad_key": 3, "y": 3},
                    {"x": 1, "y": 3},
                ],
            ],
            height,
            width,
        )


def test_convert_geometry_to_raster():
    # test box
    p1 = (1, 2)
    p2 = (3, 2)
    p3 = (3, 5)
    p4 = (1, 5)
    coords = [[p1, p2, p3, p4, p1]]
    box = Box(coords)
    expected_output = np.zeros((5, 5), dtype=bool)
    expected_output[2:5, 1:4] = True
    output = Raster.from_geometry(box, height=5, width=5).to_array()
    assert np.array_equal(
        output,
        expected_output,
    )

    p1 = (1, 2)
    p2 = (5, 2)
    p3 = (5, 7)
    p4 = (1, 7)
    coords = [[p1, p2, p3, p4, p1]]
    box = Box(coords)
    expected_output = np.zeros((8, 9), dtype=bool)
    expected_output[2:8, 1:6] = True
    output = Raster.from_geometry(box, width=9, height=8).to_array()
    assert output.shape == (8, 9)  # 8 rows, 9 cols
    assert np.array_equal(
        output,
        expected_output,
    )

    p1 = (1, 2)
    p2 = (10, 2)
    p3 = (10, 11)
    p4 = (1, 11)
    coords = [[p1, p2, p3, p4, p1]]
    box = Box(coords)
    expected_output = np.zeros((15, 15), dtype=bool)
    expected_output[2:12, 1:11] = True
    output = Raster.from_geometry(box, height=15, width=15).to_array()
    assert np.array_equal(
        output,
        expected_output,
    )

    # test incorrect box (can't use negative coordinates)
    p1 = (-1, -2)
    p2 = (10, -2)
    p3 = (10, 11)
    p4 = (-1, 11)
    coords = [[p1, p2, p3, p4, p1]]
    box = Box(coords)

    with pytest.raises(ValueError):
        Raster.from_geometry(box, height=15, width=15).to_array()

    # test case where the height and width is less than the implied height and width from the contours
    p1 = (1, 2)
    p2 = (10, 2)
    p3 = (10, 11)
    p4 = (1, 11)
    coords = [[p1, p2, p3, p4, p1]]
    box = Box(coords)
    expected_output = np.zeros((6, 7), dtype=bool)
    expected_output[2:6, 1:7] = True
    output = Raster.from_geometry(box, height=6, width=7).to_array()
    assert np.array_equal(
        output,
        expected_output,
    )

    # test polygons
    # triangle
    polygon = Polygon([[(2.0, 1.0), (6.0, 1.0), (4.0, 5.0), (2.0, 1.0)]])
    output = Raster.from_geometry(polygon, height=9, width=9).to_array()
    expected_output = np.array(
        [
            [False, False, False, False, False, False, False, False, False],
            [False, False, True, True, True, True, True, False, False],
            [False, False, False, True, True, True, False, False, False],
            [False, False, False, True, True, True, False, False, False],
            [False, False, False, False, True, False, False, False, False],
            [False, False, False, False, True, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False],
        ]
    )
    assert np.array_equal(output, expected_output)

    polygon = Polygon([[(0, 0), (2, 0), (1, 2), (0, 0)]])
    output = Raster.from_geometry(polygon, height=3, width=3).to_array()
    expected_output = np.array(
        [[True, True, True], [False, True, False], [False, True, False]]
    )
    assert np.array_equal(output, expected_output)

    # random five-pointed shape
    polygon = Polygon([[(5, 7), (2, 3), (8, 1), (9, 6), (4, 5), (5, 7)]])
    output = Raster.from_geometry(polygon, height=9, width=9).to_array()
    expected_output = np.array(
        [
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, True],
            [False, False, False, False, False, True, True, True, True],
            [False, False, True, True, True, True, True, True, True],
            [False, False, False, True, True, True, True, True, True],
            [False, False, False, False, True, True, True, True, True],
            [False, False, False, False, True, False, False, False, False],
            [False, False, False, False, False, True, False, False, False],
            [False, False, False, False, False, False, False, False, False],
        ]
    )
    assert np.array_equal(output, expected_output)

    # test multiple shapes
    polygon = Polygon([[(0, 0), (2, 0), (1, 2), (0, 0)]]).to_coordinates()
    box = Box([[(4, 4), (4, 5), (5, 5), (5, 4), (4, 4)]]).to_coordinates()
    output = Raster.from_coordinates(
        polygon + box, height=6, width=6
    ).to_array()
    expected_output = np.array(
        [
            [True, True, True, False, False, False],
            [False, True, False, False, False, False],
            [False, True, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, True, True],
            [False, False, False, False, True, True],
        ]
    )
    assert np.array_equal(output, expected_output)

    # test if we don't have the right number of points
    with pytest.raises(ValueError):
        polygon = Polygon([[(0, 0), (0, 2), (2, 1)]])


def test_calculate_bbox_iou():
    """Test ability to calculate IOU for axis-aligend and rotated bounding boxes."""

    # first, we test that we get the same IOU when we rotate polygon around the origin by the same number of degrees
    # these tests were created by taking the original bboxes and rotating them by using:
    # list(shapely.affinity.rotate(shapely.Polygon(bbox), angle=angle, origin="centroid").exterior.coords)
    tests = [
        {
            "original_bbox1": [(1, 1), (6, 1), (6, 6), (1, 6)],
            "original_bbox2": [(3, 3), (8, 3), (8, 8), (3, 8)],
            "angles": [0, 45, 90],
            "bbox1": [
                [(1.0, 1.0), (6.0, 1.0), (6.0, 6.0), (1.0, 6.0), (1.0, 1.0)],
                [
                    (1.1102230246251565e-16, 1.414213562373095),
                    (3.535533905932738, 4.949747468305833),
                    (8.881784197001252e-16, 8.485281374238571),
                    (-3.5355339059327373, 4.949747468305834),
                    (1.1102230246251565e-16, 1.414213562373095),
                ],
                [
                    (-1.0, 1.0),
                    (-1.0, 6.0),
                    (-6.0, 6.0),
                    (-6.0, 1.0),
                    (-1.0, 1.0),
                ],
            ],
            "bbox2": [
                [(3.0, 3.0), (8.0, 3.0), (8.0, 8.0), (3.0, 8.0), (3.0, 3.0)],
                [
                    (4.440892098500626e-16, 4.242640687119286),
                    (3.535533905932738, 7.7781745930520225),
                    (8.881784197001252e-16, 11.31370849898476),
                    (-3.535533905932737, 7.778174593052023),
                    (4.440892098500626e-16, 4.242640687119286),
                ],
                [
                    (-3.0, 3.0),
                    (-3.0, 8.0),
                    (-8.0, 8.0),
                    (-8.0, 3.0),
                    (-3.0, 3.0),
                ],
            ],
            # expected values come from shapely using the following function
            # def shapely_calc(bbox1, bbox2):
            #     poly1 = Pgon(bbox1)
            #     poly2 = Pgon(bbox2)
            #     intersection_area = poly1.intersection(poly2).area
            #     union_area = poly1.area + poly2.area - intersection_area
            #     return intersection_area / union_area if union_area != 0 else 0
            "expected": 0.2195,
        }
    ]

    for test in tests:
        for bbox1, bbox2 in zip(test["bbox1"], test["bbox2"]):

            expected = test["expected"]

            iou = geometry.calculate_bbox_iou(bbox1=bbox1, bbox2=bbox2)
            assert expected == round(iou, 4)

    # next we rotate shapes around their centroids to check that we get the same IOUs as shapely
    tests = [
        {
            "original_bbox1": [(1, 1), (6, 1), (6, 6), (1, 6)],
            "original_bbox2": [(3, 3), (8, 3), (8, 8), (3, 8)],
            "angles": [30, 60, 90, 112, 157, 249, 312],
            "bbox1": [
                [
                    (2.584936490538903, 0.08493649053890318),
                    (6.915063509461096, 2.5849364905389027),
                    (4.415063509461097, 6.915063509461096),
                    (0.08493649053890362, 4.415063509461096),
                    (2.584936490538903, 0.08493649053890318),
                ],
                [
                    (4.415063509461096, 0.08493649053890318),
                    (6.915063509461097, 4.415063509461096),
                    (2.5849364905389036, 6.915063509461096),
                    (0.08493649053890273, 2.5849364905389036),
                    (4.415063509461096, 0.08493649053890318),
                ],
                [(6.0, 1.0), (6.0, 6.0), (1.0, 6.0), (1.0, 1.0), (6.0, 1.0)],
                [
                    (6.754476119956748, 2.118556847122812),
                    (4.881443152877187, 6.754476119956749),
                    (0.2455238800432502, 4.881443152877188),
                    (2.118556847122811, 0.2455238800432511),
                    (6.754476119956748, 2.118556847122812),
                ],
                [
                    (6.778089954854287, 4.824434312407915),
                    (2.175565687592086, 6.778089954854286),
                    (0.221910045145715, 2.1755656875920844),
                    (4.8244343124079165, 0.22191004514571322),
                    (6.778089954854287, 4.824434312407915),
                ],
                [
                    (2.061968807620248, 6.729870940106256),
                    (0.27012905989374447, 2.0619688076202483),
                    (4.938031192379752, 0.27012905989374403),
                    (6.729870940106256, 4.938031192379752),
                    (2.061968807620248, 6.729870940106256),
                ],
                [
                    (-0.030688579590631093, 3.6850355477963417),
                    (3.3149644522036583, -0.030688579590631537),
                    (7.030688579590632, 3.3149644522036574),
                    (3.6850355477963417, 7.030688579590631),
                    (-0.030688579590631093, 3.6850355477963417),
                ],
            ],
            "bbox2": [
                [
                    (4.584936490538903, 2.084936490538903),
                    (8.915063509461095, 4.584936490538903),
                    (6.415063509461096, 8.915063509461095),
                    (2.0849364905389027, 6.4150635094610955),
                    (4.584936490538903, 2.084936490538903),
                ],
                [
                    (6.4150635094610955, 2.084936490538903),
                    (8.915063509461095, 6.4150635094610955),
                    (4.584936490538904, 8.915063509461097),
                    (2.0849364905389027, 4.5849364905389045),
                    (6.4150635094610955, 2.084936490538903),
                ],
                [(8.0, 3.0), (8.0, 8.0), (3.0, 8.0), (3.0, 3.0), (8.0, 3.0)],
                [
                    (8.754476119956747, 4.118556847122812),
                    (6.881443152877187, 8.754476119956749),
                    (2.245523880043251, 6.881443152877189),
                    (4.118556847122811, 2.2455238800432515),
                    (8.754476119956747, 4.118556847122812),
                ],
                [
                    (8.778089954854286, 6.824434312407915),
                    (4.175565687592085, 8.778089954854286),
                    (2.221910045145714, 4.175565687592085),
                    (6.824434312407915, 2.221910045145714),
                    (8.778089954854286, 6.824434312407915),
                ],
                [
                    (4.061968807620248, 8.729870940106256),
                    (2.270129059893745, 4.061968807620248),
                    (6.938031192379753, 2.270129059893746),
                    (8.729870940106256, 6.938031192379753),
                    (4.061968807620248, 8.729870940106256),
                ],
                [
                    (1.9693114204093698, 5.685035547796343),
                    (5.314964452203658, 1.9693114204093694),
                    (9.030688579590631, 5.314964452203658),
                    (5.685035547796342, 9.030688579590631),
                    (1.9693114204093698, 5.685035547796343),
                ],
            ],
            "expected": [
                0.2401,
                0.2401,
                0.2195,
                0.2295,
                0.2306,
                0.2285,
                0.2676,
            ],
        },
        {
            "original_bbox1": [(12, 15), (45, 15), (45, 48), (12, 48)],
            "original_bbox2": [(22, 25), (55, 25), (55, 58), (22, 58)],
            "angles": [
                7,
                24,
                40,
                65,
                84,
                107,
                120,
                143,
                167,
            ],
            "bbox1": [
                [
                    (14.13383266410312, 13.112144331733255),
                    (46.88785566826675, 17.13383266410312),
                    (42.866167335896876, 49.88785566826675),
                    (10.112144331733253, 45.866167335896876),
                    (14.13383266410312, 13.112144331733255),
                ],
                [
                    (20.13765455964779, 9.715345338146385),
                    (50.28465466185362, 23.13765455964779),
                    (36.862345440352215, 53.28465466185362),
                    (6.715345338146383, 39.862345440352215),
                    (20.13765455964779, 9.715345338146385),
                ],
                [
                    (26.466262248364764, 8.254271128708965),
                    (51.745728871291035, 29.46626224836476),
                    (30.53373775163524, 54.745728871291035),
                    (5.254271128708968, 33.53373775163524),
                    (26.466262248364764, 8.254271128708965),
                ],
                [
                    (36.480877167383184, 9.572720195173734),
                    (50.42727980482626, 39.480877167383184),
                    (20.519122832616816, 53.427279804826256),
                    (6.572720195173737, 23.519122832616812),
                    (36.480877167383184, 9.572720195173734),
                ],
                [
                    (43.18489162966023, 13.365669082507209),
                    (46.63433091749279, 46.18489162966023),
                    (13.815108370339779, 49.63433091749279),
                    (10.36566908250721, 16.81510837033977),
                    (43.18489162966023, 13.365669082507209),
                ],
                [
                    (49.10316160131525, 20.54510465453507),
                    (39.45489534546494, 52.10316160131524),
                    (7.896838398684764, 42.454895345464934),
                    (17.545104654535074, 10.89683839868476),
                    (49.10316160131525, 20.54510465453507),
                ],
                [
                    (51.03941916244324, 25.46058083755676),
                    (34.539419162443245, 54.03941916244324),
                    (5.960580837556762, 37.53941916244324),
                    (22.460580837556762, 8.960580837556762),
                    (51.03941916244324, 25.46058083755676),
                ],
                [
                    (51.60743379778914, 34.74753803377154),
                    (25.252461966228466, 54.60743379778913),
                    (5.3925662022108725, 28.252461966228463),
                    (31.747538033771548, 8.392566202210872),
                    (51.60743379778914, 34.74753803377154),
                ],
                [
                    (48.288798465630144, 43.865413672282614),
                    (16.13458632771738, 51.28879846563015),
                    (8.711201534369835, 19.134586327717386),
                    (40.86541367228261, 11.711201534369849),
                    (48.288798465630144, 43.865413672282614),
                ],
            ],
            "bbox2": [
                [
                    (24.133832664103117, 23.11214433173326),
                    (56.887855668266745, 27.133832664103124),
                    (52.866167335896876, 59.88785566826675),
                    (20.11214433173325, 55.86616733589688),
                    (24.133832664103117, 23.11214433173326),
                ],
                [
                    (30.137654559647785, 19.71534533814638),
                    (60.28465466185362, 33.137654559647785),
                    (46.862345440352215, 63.28465466185361),
                    (16.71534533814638, 49.862345440352215),
                    (30.137654559647785, 19.71534533814638),
                ],
                [
                    (36.46626224836476, 18.254271128708965),
                    (61.745728871291035, 39.46626224836476),
                    (40.53373775163524, 64.74572887129104),
                    (15.254271128708968, 43.53373775163524),
                    (36.46626224836476, 18.254271128708965),
                ],
                [
                    (46.480877167383184, 19.572720195173737),
                    (60.42727980482627, 49.480877167383184),
                    (30.51912283261682, 63.42727980482626),
                    (16.572720195173737, 33.519122832616816),
                    (46.480877167383184, 19.572720195173737),
                ],
                [
                    (53.18489162966023, 23.36566908250721),
                    (56.63433091749279, 56.18489162966023),
                    (23.81510837033978, 59.63433091749279),
                    (20.365669082507218, 26.81510837033977),
                    (53.18489162966023, 23.36566908250721),
                ],
                [
                    (59.10316160131525, 30.54510465453507),
                    (49.454895345464934, 62.10316160131524),
                    (17.896838398684764, 52.454895345464934),
                    (27.545104654535074, 20.89683839868476),
                    (59.10316160131525, 30.54510465453507),
                ],
                [
                    (61.03941916244324, 35.46058083755676),
                    (44.53941916244324, 64.03941916244324),
                    (15.960580837556762, 47.53941916244324),
                    (32.46058083755676, 18.960580837556765),
                    (61.03941916244324, 35.46058083755676),
                ],
                [
                    (61.60743379778913, 44.74753803377154),
                    (35.25246196622846, 64.60743379778913),
                    (15.392566202210872, 38.252461966228466),
                    (41.74753803377154, 18.392566202210872),
                    (61.60743379778913, 44.74753803377154),
                ],
                [
                    (58.28879846563015, 53.865413672282614),
                    (26.134586327717386, 61.28879846563015),
                    (18.71120153436985, 29.134586327717386),
                    (50.865413672282614, 21.71120153436984),
                    (58.28879846563015, 53.865413672282614),
                ],
            ],
            "expected": [
                0.3224,
                0.3403,
                0.3809,
                0.3421,
                0.3219,
                0.3303,
                0.3523,
                0.3711,
                0.3263,
            ],
        },
    ]

    for test in tests:
        for bbox1, bbox2, expected in zip(
            test["bbox1"], test["bbox2"], test["expected"]
        ):
            iou = geometry.calculate_bbox_iou(bbox1=bbox1, bbox2=bbox2)
            assert expected == round(iou, 4)


def test_is_axis_aligned(box_points, skewed_box_points, rotated_box_points):
    tests = [
        {
            "bbox": [(1, 1), (6, 1), (6, 6), (1, 6)],
            "expected": True,
        },
        # rotated box
        {
            "bbox": [
                (2.584936490538903, 0.08493649053890318),
                (6.915063509461096, 2.5849364905389027),
                (4.415063509461097, 6.915063509461096),
                (0.08493649053890362, 4.415063509461096),
                (2.584936490538903, 0.08493649053890318),
            ],
            "expected": False,
        },
    ]

    for test in tests:
        assert geometry.is_axis_aligned(bbox=test["bbox"]) == test["expected"]

    assert geometry.is_axis_aligned(bbox=box_points)
    assert not geometry.is_axis_aligned(bbox=skewed_box_points)
    assert not geometry.is_axis_aligned(bbox=rotated_box_points)


def test_is_skewed(box_points, skewed_box_points, rotated_box_points):
    tests = [
        {
            "bbox": [(1, 1), (6, 1), (6, 6), (1, 6)],
            "expected": False,
        },
        # rotated box
        {
            "bbox": [
                (2.584936490538903, 0.08493649053890318),
                (6.915063509461096, 2.5849364905389027),
                (4.415063509461097, 6.915063509461096),
                (0.08493649053890362, 4.415063509461096),
                (2.584936490538903, 0.08493649053890318),
            ],
            "expected": False,
        },
    ]

    for test in tests:
        assert geometry.is_skewed(bbox=test["bbox"]) == test["expected"]

    assert not geometry.is_skewed(bbox=box_points)
    assert geometry.is_skewed(bbox=skewed_box_points)
    assert not geometry.is_skewed(bbox=rotated_box_points)


def test_is_rotated(box_points, skewed_box_points, rotated_box_points):
    tests = [
        {
            "bbox": [(1, 1), (6, 1), (6, 6), (1, 6)],
            "expected": False,
        },
        # rotated box
        {
            "bbox": [
                (2.584936490538903, 0.08493649053890318),
                (6.915063509461096, 2.5849364905389027),
                (4.415063509461097, 6.915063509461096),
                (0.08493649053890362, 4.415063509461096),
                (2.584936490538903, 0.08493649053890318),
            ],
            "expected": True,
        },
    ]

    for test in tests:
        assert geometry.is_rotated(bbox=test["bbox"]) == test["expected"]

    assert not geometry.is_rotated(bbox=box_points)
    assert not geometry.is_rotated(bbox=skewed_box_points)
    assert geometry.is_rotated(bbox=rotated_box_points)
