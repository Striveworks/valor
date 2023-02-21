from velour_api.crud import _wkt_polygon_from_detection
from velour_api.schemas import DetectionBase, Image, Label


def test__wkt_polygon_from_detection_adds_point():
    """Check that the first point gets added to the end in the WKT"""
    det = DetectionBase(
        boundary=[(0, 1), (1, 1), (2, 1)],
        labels=[Label(key="class", value="a")],
        image=Image(uri=""),
    )
    assert (
        _wkt_polygon_from_detection(det)
        == "POLYGON ((0.0 1.0, 1.0 1.0, 2.0 1.0, 0.0 1.0))"
    )


def test__wkt_polygon_from_detection_does_not_add_point():
    """Check that no points get added since boundary first and last points are teh same"""
    det = DetectionBase(
        boundary=[(0, 1), (1, 1), (2, 1), (0, 1)],
        labels=[Label(key="class", value="a")],
        image=Image(uri=""),
    )
    assert (
        _wkt_polygon_from_detection(det)
        == "POLYGON ((0.0 1.0, 1.0 1.0, 2.0 1.0, 0.0 1.0))"
    )
