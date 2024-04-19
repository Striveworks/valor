import os
from base64 import b64encode
from tempfile import TemporaryDirectory

import numpy as np
import PIL.Image
import pytest

from valor_api.schemas import (
    Box,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    Raster,
)


def _create_b64_mask(mode: str, ext: str = ".png", size=(20, 20)) -> str:
    with TemporaryDirectory() as tempdir:
        img = PIL.Image.new(mode=mode, size=size)
        img_path = os.path.join(tempdir, f"img.{ext}")
        img.save(img_path)

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        return b64encode(img_bytes).decode()


def test_point():
    # valid
    p1 = Point(value=(3.14, -3.14))
    assert Point(value=(3.14, -3.14))
    assert Point(value=(-3.14, 3.14))

    # test type validation
    with pytest.raises(ValueError):
        LineString(value=tuple())  # type: ignore - purposefully throwing error
    with pytest.raises(ValueError):
        Point(value=("test", 0))  # type: ignore - purposefully throwing error
    with pytest.raises(ValueError):
        Point(value=(0, "test"))  # type: ignore - purposefully throwing error

    # test geojson conversion
    geojson = {"type": "Point", "coordinates": [3.14, -3.14]}
    assert p1.to_dict() == geojson
    assert Point.from_dict(geojson).value == (3.14, -3.14)

    # test wkt conversion
    wkt = "POINT (3.14 -3.14)"
    assert p1.to_wkt() == wkt


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
    with pytest.raises(ValueError):
        LineString(value=[])  # type: ignore - purposefully throwing error
    with pytest.raises(ValueError):
        LineString(value="points")  # type: ignore - purposefully throwing error
    with pytest.raises(ValueError):
        LineString(value=box_points[0])  # type: ignore - purposefully throwing error
    with pytest.raises(ValueError):
        LineString(value=[1, 2])  # type: ignore - purposefully throwing error

    # test geojson conversion
    geojson = {
        "type": "MultiPoint",
        "coordinates": [[point[0], point[1]] for point in box_points],
    }
    assert MultiPoint(value=box_points).to_dict() == geojson
    assert MultiPoint.from_dict(geojson).value == box_points

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
    with pytest.raises(ValueError):
        LineString(value=[])
    with pytest.raises(ValueError):
        LineString(value=[box_points[0]])

    # test type validation
    with pytest.raises(ValueError):
        LineString(value="points")  # type: ignore - purposefully throwing error
    with pytest.raises(ValueError):
        LineString(value=[1, 2])  # type: ignore - purposefully throwing error

    # test geojson conversion
    geojson = {
        "type": "LineString",
        "coordinates": [[point[0], point[1]] for point in box_points],
    }
    assert LineString(value=box_points).to_dict() == geojson
    assert LineString.from_dict(geojson).value == box_points

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
    with pytest.raises(ValueError):
        MultiLineString(value=box_points[0])
    with pytest.raises(ValueError):
        MultiLineString(
            value=[
                box_points[0],
                box_points[1],
            ]
        )
    with pytest.raises(ValueError):
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
        box_points,
        skewed_box_points,
    ]

    # test wkt conversion
    wkt = "MULTILINESTRING ((-5 -5, 5 -5, 5 5, -5 5, -5 -5),(0 0, 10 0, 15 10, 5 10, 0 0))"
    assert (
        MultiLineString(value=[box_points, skewed_box_points]).to_wkt() == wkt
    )


def test_polygon(
    box_points: list[tuple[float, float]],
    rotated_box_points: list[tuple[float, float]],
    skewed_box_points: list[tuple[float, float]],
):
    # valid
    p1 = Polygon(value=[box_points])
    p2 = Polygon(value=[skewed_box_points, box_points])
    p3 = Polygon(value=[skewed_box_points, box_points, rotated_box_points])

    # test type validation
    with pytest.raises(ValueError):
        Polygon(value=[])
    with pytest.raises(ValueError):
        Polygon(value=box_points)  # type: ignore - purposefully throwing error
    with pytest.raises(ValueError):
        Polygon(
            value=["skewed_box_points"]  # type: ignore - purposefully throwing error
        )
    with pytest.raises(ValueError):
        Polygon(value=[box_points, []])
    with pytest.raises(ValueError):
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
        box_points,
        skewed_box_points,
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


def test_box(
    box_points: list[tuple[float, float]],
    rotated_box_points: list[tuple[float, float]],
    skewed_box_points: list[tuple[float, float]],
):
    assert Box(value=[box_points])
    assert Box(value=[rotated_box_points])
    assert Box(value=[skewed_box_points])

    # test type validation
    with pytest.raises(ValueError):
        Box(value=[])  # type: ignore - purposefully throwing error
    with pytest.raises(ValueError):
        Box(value=[box_points, box_points])  # box does not have holes
    with pytest.raises(ValueError):  # type checking
        Box(value=1234)  # type: ignore - purposefully throwing error
    with pytest.raises(ValueError):
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
    assert Box.from_dict(geojson).value == [box_points]

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


def test_multipolygon(
    box_points,
    rotated_box_points,
    skewed_box_points,
):
    assert MultiPolygon(value=[[rotated_box_points]])
    assert MultiPolygon(
        value=[[skewed_box_points, box_points], [rotated_box_points]]
    )

    with pytest.raises(ValueError):
        MultiPolygon(value=[])
    with pytest.raises(ValueError):
        MultiPolygon(value=[[]])
    with pytest.raises(ValueError):
        MultiPolygon(value=box_points)
    with pytest.raises(ValueError):
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
        [skewed_box_points, box_points],
        [rotated_box_points],
    ]

    # test wkt conversion
    assert (
        MultiPolygon(
            value=[[skewed_box_points, box_points], [rotated_box_points]]
        ).to_wkt()
        == "MULTIPOLYGON (((0 0,10 0,15 10,5 10,0 0),(-5 -5,5 -5,5 5,-5 5,-5 -5)),((0 7.0710678118654755,7.0710678118654755 0,0 -7.0710678118654755,-7.0710678118654755 0,0 7.0710678118654755)))"
    )


def test_raster(raster):
    # valid
    height = 20
    width = 20
    mask = _create_b64_mask(mode="1", size=(height, width))
    assert Raster(
        mask=mask,
    )

    # test property `mask`
    with pytest.raises(PIL.UnidentifiedImageError):
        # not any string can be passed
        Raster(mask="text")
    with pytest.raises(ValueError) as exc_info:
        base64_mask = _create_b64_mask(
            mode="RGB", ext="png", size=(width, height)
        )
        Raster(
            mask=base64_mask,  # only supports binary images
        )
    assert "Expected image mode to be binary but got mode" in str(exc_info)
    with pytest.raises(ValueError) as exc_info:
        base64_mask = _create_b64_mask(
            mode="1", ext="jpg", size=(width, height)
        )
        Raster(
            mask=base64_mask,  # Check we get an error if the format is not PNG
        )
    assert "Expected image format PNG but got" in str(exc_info)

    # test how from_numpy handles non-2D arrays
    with pytest.raises(ValueError):
        Raster.from_numpy(mask=np.array([False]))

    # test how from_numpy handles non-boolean arrays
    with pytest.raises(ValueError):
        Raster.from_numpy(mask=np.array([[1, 1]]))
