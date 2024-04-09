import numpy as np
import PIL.Image
import pytest

from valor import Annotation, GroundTruth, Label
from valor.enums import TaskType
from valor.metatypes import ImageMetadata
from valor.schemas import MultiPolygon, Polygon, Raster
from valor.viz import (
    _polygons_to_binary_mask,
    create_combined_segmentation_mask,
    draw_detections_on_image,
)


@pytest.fixture
def bounding_poly() -> Polygon:
    return Polygon(
        [
            [
                (100, 100),
                (200, 100),
                (200, 200),
                (100, 200),
                (100, 100),
            ]
        ]
    )


@pytest.fixture
def poly1(bounding_poly: Polygon) -> Polygon:
    return Polygon(
        [
            bounding_poly.get_value()[0],
            [
                (150, 120),
                (180, 120),
                (180, 140),
                (150, 140),
                (150, 120),
            ],
        ]
    )


def test__polygons_to_binary_mask(poly1):
    poly2 = Polygon(
        [
            [
                (10, 15),
                (20, 15),
                (20, 20),
                (10, 20),
                (10, 15),
            ]
        ]
    )

    mask = _polygons_to_binary_mask([poly1, poly2], 500, 600)

    area_poly1 = (200 - 100 + 1) * (200 - 100 + 1) - (180 - 150 + 1) * (
        140 - 120 + 1
    )

    area_poly2 = (20 - 10 + 1) * (20 - 15 + 1)

    assert mask.sum() == area_poly1 + area_poly2


def test_create_combined_segmentation_mask(poly1: Polygon):
    image = ImageMetadata.create(uid="uid", height=200, width=200).datum

    gt1 = GroundTruth(
        datum=image,
        annotations=[
            Annotation(
                task_type=TaskType.OBJECT_DETECTION,
                labels=[
                    Label(key="k1", value="v1"),
                    Label(key="k2", value="v2"),
                    Label(key="k3", value="v3"),
                ],
                raster=Raster.from_geometry(
                    MultiPolygon([poly1.get_value()]),
                    height=2,
                    width=2,
                ),
            ),
            Annotation(
                task_type=TaskType.SEMANTIC_SEGMENTATION,
                labels=[
                    Label(key="k1", value="v1"),
                    Label(key="k2", value="v3"),
                ],
                raster=Raster.from_numpy(
                    np.array([[True, False], [False, True]]),
                ),
            ),
        ],
    )

    gt_with_size_mismatch = GroundTruth(
        datum=image,
        annotations=[
            Annotation(
                task_type=TaskType.OBJECT_DETECTION,
                labels=[
                    Label(key="k1", value="v1"),
                    Label(key="k2", value="v2"),
                    Label(key="k3", value="v3"),
                ],
                raster=Raster.from_geometry(
                    MultiPolygon([poly1.get_value()]),
                    height=20,
                    width=20,
                ),
            ),
            Annotation(
                task_type=TaskType.SEMANTIC_SEGMENTATION,
                labels=[
                    Label(key="k1", value="v1"),
                    Label(key="k2", value="v3"),
                ],
                raster=Raster.from_numpy(
                    np.array([[True, False], [False, True]]),
                ),
            ),
        ],
    )

    # test that a size mistmatch between rasters is caught
    with pytest.raises(ValueError) as exc_info:
        create_combined_segmentation_mask(
            gt_with_size_mismatch,
            label_key="k2",
            task_type=TaskType.SEMANTIC_SEGMENTATION,
        )
    assert "(20, 20) != (2, 2)" in str(exc_info)

    # check get an error since "k3" isn't a label key in seg2
    with pytest.raises(RuntimeError) as exc_info:
        create_combined_segmentation_mask(
            gt1,
            label_key="k3",
            task_type=TaskType.SEMANTIC_SEGMENTATION,
        )
    assert "doesn't have a label" in str(exc_info)

    # should have one distinct (non-black) color
    combined_mask, legend = create_combined_segmentation_mask(
        gt1,
        label_key="k1",
        task_type=TaskType.SEMANTIC_SEGMENTATION,
    )
    combined_mask = np.array(combined_mask)
    # check that we get two unique RGB values (black and one color for label value "v1")
    assert combined_mask.shape == (2, 2, 3)
    assert len(legend) == 1  # background color 'black' is not counted

    # should have two distinct (non-black) color
    combined_mask, legend = create_combined_segmentation_mask(
        gt1, label_key="k2"
    )
    combined_mask = np.array(combined_mask)
    # check that we get three unique RGB values (black and two colors for label values "v2" and "v3")
    assert combined_mask.shape == (2, 2, 3)
    assert len(legend) == 2  # background color 'black' is not counted


def test_draw_detections_on_image(bounding_poly: Polygon):
    detections = [
        GroundTruth(
            datum=ImageMetadata.create("test", 300, 300).datum,
            annotations=[
                Annotation(
                    task_type=TaskType.OBJECT_DETECTION,
                    labels=[Label(key="k", value="v")],
                    polygon=Polygon(bounding_poly.get_value()),
                )
            ],
        ),
    ]
    img = PIL.Image.new("RGB", (300, 300))

    img = draw_detections_on_image(detections, img)

    assert img.size == (300, 300)

    # check unique colors only have red component
    unique_rgb = np.unique(np.array(img).reshape(-1, 3), axis=0)
    assert unique_rgb[:, 1:].sum() == 0
