import numpy as np
import PIL.Image
import pytest

from velour.data_types import (
    BoundingPolygon,
    GroundTruthDetection,
    GroundTruthInstanceSegmentation,
    Image,
    Label,
    Point,
    PolygonWithHole,
)
from velour.viz import (
    _polygons_to_binary_mask,
    combined_segmentation_mask,
    draw_detections_on_image,
)


@pytest.fixture
def bounding_poly() -> BoundingPolygon:
    return BoundingPolygon(
        [
            Point(100, 100),
            Point(200, 100),
            Point(200, 200),
            Point(100, 200),
        ]
    )


@pytest.fixture
def poly1(bounding_poly: BoundingPolygon) -> PolygonWithHole:
    return PolygonWithHole(
        polygon=bounding_poly,
        hole=BoundingPolygon(
            [
                Point(150, 120),
                Point(180, 120),
                Point(180, 140),
                Point(150, 140),
            ]
        ),
    )


def test__polygons_to_binary_mask(poly1):
    poly2 = PolygonWithHole(
        polygon=BoundingPolygon(
            [
                Point(10, 15),
                Point(20, 15),
                Point(20, 20),
                Point(10, 20),
            ]
        )
    )

    mask = _polygons_to_binary_mask([poly1, poly2], 500, 600)

    area_poly1 = (200 - 100 + 1) * (200 - 100 + 1) - (180 - 150 + 1) * (
        140 - 120 + 1
    )

    area_poly2 = (20 - 10 + 1) * (20 - 15 + 1)

    assert mask.sum() == area_poly1 + area_poly2


def test_combined_segmentation_mask(poly1: PolygonWithHole):
    with pytest.raises(ValueError) as exc_info:
        combined_segmentation_mask([], label_key="")
    assert "cannot be empty" in str(exc_info)

    image = Image("uid", 200, 200)

    seg1 = GroundTruthInstanceSegmentation(
        shape=[poly1],
        labels=[
            Label(key="k1", value="v1"),
            Label(key="k2", value="v2"),
            Label(key="k3", value="v3"),
        ],
        image=image,
    )
    seg2 = GroundTruthInstanceSegmentation(
        shape=np.array([[True, False], [False, True]]),
        labels=[Label(key="k1", value="v1"), Label(key="k2", value="v3")],
        image=image,
    )
    segs = [seg1, seg2]

    # check get an error since "k3" isn't a label key in seg2
    with pytest.raises(RuntimeError) as exc_info:
        combined_segmentation_mask(segs, label_key="k3")
    assert "doesn't have a label" in str(exc_info)

    # should have one distinct (non-black) color
    combined_mask, _ = combined_segmentation_mask(segs, label_key="k1")
    combined_mask = np.array(combined_mask)
    # check that we get two unique RGB values (black and one color for label value "v1")
    unique_rgb = np.unique(combined_mask.reshape(-1, 3), axis=0)
    assert unique_rgb.shape == (2, 3)

    # should have two distinct (non-black) color
    combined_mask, _ = combined_segmentation_mask(segs, label_key="k2")
    combined_mask = np.array(combined_mask)
    # check that we get two unique RGB values (black and one color for label value "v1")
    unique_rgb = np.unique(combined_mask.reshape(-1, 3), axis=0)
    assert unique_rgb.shape == (3, 3)

    with pytest.raises(RuntimeError) as exc_info:
        combined_segmentation_mask(
            [
                seg1,
                GroundTruthInstanceSegmentation(
                    shape=[],
                    labels=[],
                    image=Image("different uid", height=10, width=100),
                ),
            ],
            "",
        )
    assert "belong to the same image" in str(exc_info)


def test_draw_detections_on_image(bounding_poly: BoundingPolygon):
    detections = [
        GroundTruthDetection(
            boundary=bounding_poly,
            labels=[Label("k", "v")],
            image=Image("", 300, 300),
        )
    ]
    img = PIL.Image.new("RGB", (300, 300))

    img = draw_detections_on_image(detections, img)

    assert img.size == (300, 300)

    # check unique colors only have red component
    unique_rgb = np.unique(np.array(img).reshape(-1, 3), axis=0)
    assert unique_rgb[:, 1:].sum() == 0
