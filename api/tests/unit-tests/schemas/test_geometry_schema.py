import os
from base64 import b64encode
from tempfile import TemporaryDirectory

import numpy as np
import PIL.Image
import pytest
from pydantic import ValidationError

from valor_api import schemas


def _create_b64_mask(mode: str, ext: str = ".png", size=(20, 20)) -> str:
    with TemporaryDirectory() as tempdir:
        img = PIL.Image.new(mode=mode, size=size)
        img_path = os.path.join(tempdir, f"img.{ext}")
        img.save(img_path)

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        return b64encode(img_bytes).decode()


def test_geometry_Point():
    # valid
    p1 = schemas.geometry.Point(x=3.14, y=-3.14)
    p2 = schemas.geometry.Point(x=3.14, y=-3.14)
    p3 = schemas.geometry.Point(x=-3.14, y=3.14)

    # test properties
    with pytest.raises(ValidationError):
        schemas.geometry.Point(x="test", y=0)  # type: ignore - purposefully throwing error
    with pytest.raises(ValidationError):
        schemas.geometry.Point(x=0, y="test")  # type: ignore - purposefully throwing error

    # test member fn `__str__`
    assert str(p1) == "(3.14, -3.14)"

    # test member fn `__hash__`
    assert p1.__hash__() == p2.__hash__()
    assert p1.__hash__() != p3.__hash__()

    # test member fn `__eq__`
    assert p1 == p2
    assert not p1 == p3

    # check that we get an
    with pytest.raises(TypeError):
        assert p1 == "not_a_point"

    # test member fn `__neq__`
    assert not p1 != p2
    assert p1 != p3

    with pytest.raises(TypeError):
        assert p1 != "not_a_point"

    # test member fn `__neg__`
    assert p1 == -p3

    # test member fn `__add__`
    assert p1 + p3 == schemas.geometry.Point(x=0, y=0)

    with pytest.raises(TypeError):
        p1 + "not_a_point"

    # test member fn `__sub__`
    assert p1 - p2 == schemas.geometry.Point(x=0, y=0)

    with pytest.raises(TypeError):
        p1 - "not_a_point"

    # test member fn `__iadd__`
    p4 = p3
    p4 += p1
    assert p4 == schemas.geometry.Point(x=0, y=0)

    with pytest.raises(TypeError):
        p1 += "not_a_point"

    # test member fn `__isub__`
    p4 = p3
    p4 -= p3
    assert p4 == schemas.geometry.Point(x=0, y=0)

    with pytest.raises(TypeError):
        p1 -= "not_a_point"

    # test member fn `dot`
    assert p1.dot(p2) == (3.14**2) * 2

    with pytest.raises(TypeError):
        p1.dot("not_a_point")


def test_geometry_LineSegment(box_points):
    # valid
    l1 = schemas.geometry.LineSegment(
        points=(
            box_points[0],
            box_points[1],
        )
    )
    l2 = schemas.geometry.LineSegment(
        points=(
            box_points[-1],
            box_points[0],
        )
    )
    l3 = schemas.geometry.LineSegment(
        points=(
            box_points[2],
            box_points[3],
        )
    )

    # test property `points`
    with pytest.raises(ValidationError):
        schemas.geometry.LineSegment(points="points")  # type: ignore - purposefully throwing error
    with pytest.raises(ValidationError):
        schemas.geometry.LineSegment(points=box_points[0])  # type: ignore - purposefully throwing error
    with pytest.raises(ValidationError):
        schemas.geometry.LineSegment(points=(box_points[0],))  # type: ignore - purposefully throwing error
    with pytest.raises(ValidationError):
        schemas.geometry.LineSegment(
            points=(box_points[0], box_points[1], box_points[2])  # type: ignore - purposefully throwing error
        )
    with pytest.raises(ValidationError):
        schemas.geometry.LineSegment(points=(1, 2))  # type: ignore - purposefully throwing error

    # test member fn `delta_xy`
    assert l1.delta_xy() == box_points[0] - box_points[1]

    # test member fn `parallel`
    assert not l1.parallel(l2)
    assert l1.parallel(l3)
    assert l3.parallel(l1)

    with pytest.raises(TypeError):
        l1.parallel("not_a_line")

    # test member fn `perpendicular`
    assert l2.perpendicular(l3)
    assert l3.perpendicular(l2)
    assert not l1.perpendicular(l3)

    with pytest.raises(TypeError):
        l1.perpendicular("not_a_line")


def test_geometry_BasicPolygon(box_points):
    poly = schemas.geometry.BasicPolygon(
        points=box_points,
    )

    with pytest.raises(ValidationError):
        schemas.geometry.BasicPolygon(points=box_points[0])
    with pytest.raises(ValidationError):
        schemas.geometry.BasicPolygon(points=[])
    with pytest.raises(ValidationError):
        schemas.geometry.BasicPolygon(
            points=[
                box_points[0],
                box_points[1],
            ]
        )
    with pytest.raises(ValidationError):
        schemas.geometry.BasicPolygon(
            points=[
                box_points[0],
                box_points[1],
                (1, 3),  # type: ignore - purposefully throwing error
            ]
        )

    assert poly.left == -5
    assert poly.right == 5
    assert poly.top == 5
    assert poly.bottom == -5
    assert poly.width == 10
    assert poly.height == 10

    plist = box_points + [box_points[0]]
    assert poly.segments == [
        schemas.geometry.LineSegment(points=(plist[i], plist[i + 1]))
        for i in range(len(plist) - 1)
    ]

    assert (
        str(poly)
        == "((-5.0,-5.0),(5.0,-5.0),(5.0,5.0),(-5.0,5.0),(-5.0,-5.0))"
    )

    assert (
        poly.wkt()
        == "POLYGON ((-5.0 -5.0, 5.0 -5.0, 5.0 5.0, -5.0 5.0, -5.0 -5.0))"
    )


def test_geometry_Polygon(
    component_polygon_box,
    component_polygon_rotated_box,
    component_polygon_skewed_box,
):
    # valid
    p1 = schemas.Polygon(
        boundary=component_polygon_box,
    )
    schemas.Polygon(
        boundary=component_polygon_skewed_box,
        holes=[],
    )
    p2 = schemas.Polygon(
        boundary=component_polygon_skewed_box,
        holes=[component_polygon_box],
    )
    schemas.Polygon(
        boundary=component_polygon_skewed_box,
        holes=[component_polygon_box, component_polygon_rotated_box],
    )

    # test property `boundary`
    with pytest.raises(ValidationError):  # type checking
        schemas.Polygon(
            boundary="component_polygon_skewed_box",  # type: ignore - purposefully throwing error
        )
    with pytest.raises(ValidationError):
        schemas.Polygon(
            boundary=[component_polygon_box],  # type: ignore - purposefully throwing error
        )

    # test property `holes`
    with pytest.raises(ValidationError):  # type checking
        schemas.Polygon(
            boundary=component_polygon_box,
            holes=component_polygon_skewed_box,
        )
    with pytest.raises(ValidationError):
        schemas.Polygon(
            boundary=component_polygon_box,
            holes=[component_polygon_skewed_box, 123],  # type: ignore - purposefully throwing error
        )

    # test member fn `__str__`
    assert (
        str(p1)
        == "(((-5.0,-5.0),(5.0,-5.0),(5.0,5.0),(-5.0,5.0),(-5.0,-5.0)))"
    )
    assert (
        str(p2)
        == "(((0.0,0.0),(10.0,0.0),(15.0,10.0),(5.0,10.0),(0.0,0.0)),((-5.0,-5.0),(5.0,-5.0),(5.0,5.0),(-5.0,5.0),(-5.0,-5.0)))"
    )

    # test member fn `wkt`
    assert (
        p1.wkt()
        == "POLYGON ((-5.0 -5.0, 5.0 -5.0, 5.0 5.0, -5.0 5.0, -5.0 -5.0))"
    )
    assert (
        p2.wkt()
        == "POLYGON ((0.0 0.0, 10.0 0.0, 15.0 10.0, 5.0 10.0, 0.0 0.0), (-5.0 -5.0, 5.0 -5.0, 5.0 5.0, -5.0 5.0, -5.0 -5.0))"
    )


def test_geometry_MultiPolygon(
    component_polygon_box,
    component_polygon_rotated_box,
    component_polygon_skewed_box,
):
    p1 = schemas.Polygon(boundary=component_polygon_rotated_box)
    p2 = schemas.Polygon(
        boundary=component_polygon_skewed_box, holes=[component_polygon_box]
    )

    mp1 = schemas.MultiPolygon(polygons=[p1])
    mp2 = schemas.MultiPolygon(polygons=[p1, p2])

    with pytest.raises(ValidationError):  # type checking
        schemas.MultiPolygon(polygons=component_polygon_box)
    with pytest.raises(ValidationError):
        schemas.MultiPolygon(polygons=p1)  # type: ignore - purposefully throwing error
    with pytest.raises(ValidationError):
        schemas.MultiPolygon(polygons=[component_polygon_box])
    with pytest.raises(ValidationError):
        schemas.MultiPolygon(polygons=[p1, component_polygon_box])

    assert (
        mp1.wkt()
        == "MULTIPOLYGON (((0.0 7.0710678118654755, 7.0710678118654755 0.0, 0.0 -7.0710678118654755, -7.0710678118654755 0.0, 0.0 7.0710678118654755)))"
    )
    assert (
        mp2.wkt()
        == "MULTIPOLYGON (((0.0 7.0710678118654755, 7.0710678118654755 0.0, 0.0 -7.0710678118654755, -7.0710678118654755 0.0, 0.0 7.0710678118654755)), ((0.0 0.0, 10.0 0.0, 15.0 10.0, 5.0 10.0, 0.0 0.0), (-5.0 -5.0, 5.0 -5.0, 5.0 5.0, -5.0 5.0, -5.0 -5.0)))"
    )


def test_geometry_BoundingBox(
    component_polygon_box,
    component_polygon_rotated_box,
    component_polygon_skewed_box,
):
    bbox1 = schemas.BoundingBox(
        polygon=component_polygon_box,
    )
    bbox2 = schemas.BoundingBox(polygon=component_polygon_rotated_box)
    bbox3 = schemas.BoundingBox(polygon=component_polygon_skewed_box)

    with pytest.raises(ValidationError):  # type checking
        schemas.BoundingBox(polygon=1234)  # type: ignore - purposefully throwing error
    with pytest.raises(ValidationError):
        schemas.BoundingBox(polygon=component_polygon_box.points[0])
    with pytest.raises(ValidationError):
        schemas.BoundingBox(polygon=[component_polygon_box])  # type: ignore - purposefully throwing error
    with pytest.raises(ValidationError):
        box_plus_one = schemas.geometry.BasicPolygon(
            points=component_polygon_box.points
            + [schemas.geometry.Point(x=15, y=15)]
        )
        schemas.BoundingBox(polygon=box_plus_one)  # check for 4 unique points

    assert (
        schemas.BoundingBox.from_extrema(
            xmin=component_polygon_box.left,
            ymin=component_polygon_box.bottom,
            xmax=component_polygon_box.right,
            ymax=component_polygon_box.top,
        ).polygon
        == component_polygon_box
    )

    assert bbox1.left == -5 == bbox1.polygon.left
    assert bbox1.right == 5 == bbox1.polygon.right
    assert bbox1.top == 5 == bbox1.polygon.top
    assert bbox1.bottom == -5 == bbox1.polygon.bottom
    assert bbox1.width == 10 == bbox1.polygon.width
    assert bbox1.height == 10 == bbox1.polygon.height

    assert bbox1.is_rectangular()
    assert bbox2.is_rectangular()
    assert not bbox3.is_rectangular()

    assert not bbox1.is_rotated()
    assert bbox2.is_rotated()
    assert not bbox3.is_rotated()

    assert not bbox1.is_skewed()
    assert not bbox2.is_skewed()
    assert bbox3.is_skewed()

    assert (
        bbox1.wkt()
        == "POLYGON ((-5.0 -5.0, 5.0 -5.0, 5.0 5.0, -5.0 5.0, -5.0 -5.0))"
    )
    assert (
        bbox2.wkt()
        == "POLYGON ((0.0 7.0710678118654755, 7.0710678118654755 0.0, 0.0 -7.0710678118654755, -7.0710678118654755 0.0, 0.0 7.0710678118654755))"
    )
    assert (
        bbox3.wkt()
        == "POLYGON ((0.0 0.0, 10.0 0.0, 15.0 10.0, 5.0 10.0, 0.0 0.0))"
    )


def test_geometry_Raster(raster):
    # valid
    height = 20
    width = 20
    mask = _create_b64_mask(mode="1", size=(height, width))
    assert schemas.Raster(
        mask=mask,
    )

    # test property `mask`
    with pytest.raises(PIL.UnidentifiedImageError):
        # not any string can be passed
        schemas.Raster(mask="text")
    with pytest.raises(ValueError) as exc_info:
        base64_mask = _create_b64_mask(
            mode="RGB", ext="png", size=(width, height)
        )
        schemas.Raster(
            mask=base64_mask,  # only supports binary images
        )
    assert "Expected image mode to be binary but got mode" in str(exc_info)
    with pytest.raises(ValueError) as exc_info:
        base64_mask = _create_b64_mask(
            mode="1", ext="jpg", size=(width, height)
        )
        schemas.Raster(
            mask=base64_mask,  # Check we get an error if the format is not PNG
        )
    assert "Expected image format PNG but got" in str(exc_info)

    # test how from_numpy handles non-2D arrays
    with pytest.raises(ValueError):
        schemas.Raster.from_numpy(mask=np.array([False]))

    # test how from_numpy handles non-boolean arrays
    with pytest.raises(ValueError):
        schemas.Raster.from_numpy(mask=np.array([[1, 1]]))
