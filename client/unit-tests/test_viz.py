import pytest

from valor.schemas import Polygon
from valor.viz import _polygons_to_binary_mask


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


# TODO move outside of Valor
# def test_create_combined_segmentation_mask(poly1: Polygon):
#     image = ImageMetadata.create(uid="uid", height=200, width=200).datum

#     gt1 = GroundTruth(
#         datum=image,
#         annotations=[
#             Annotation(
#                 labels=[
#                     Label(key="k1", value="v1"),
#                     Label(key="k2", value="v2"),
#                     Label(key="k3", value="v3"),
#                 ],
#                 raster=Raster.from_geometry(
#                     MultiPolygon([poly1.get_value()]),
#                     height=2,
#                     width=2,
#                 ),
#             ),
#             Annotation(
#                 labels=[
#                     Label(key="k1", value="v1"),
#                     Label(key="k2", value="v3"),
#                 ],
#                 raster=Raster.from_numpy(
#                     np.array([[True, False], [False, True]]),
#                 ),
#             ),
#         ],
#     )

#     gt_with_size_mismatch = GroundTruth(
#         datum=image,
#         annotations=[
#             Annotation(
#                 labels=[
#                     Label(key="k1", value="v1"),
#                     Label(key="k2", value="v2"),
#                     Label(key="k3", value="v3"),
#                 ],
#                 raster=Raster.from_geometry(
#                     MultiPolygon([poly1.get_value()]),
#                     height=20,
#                     width=20,
#                 ),
#             ),
#             Annotation(
#                 labels=[
#                     Label(key="k1", value="v1"),
#                     Label(key="k2", value="v3"),
#                 ],
#                 raster=Raster.from_numpy(
#                     np.array([[True, False], [False, True]]),
#                 ),
#             ),
#         ],
#     )

#     # test that a size mistmatch between rasters is caught
#     with pytest.raises(ValueError) as exc_info:
#         create_combined_segmentation_mask(
#             gt_with_size_mismatch,
#             label_key="k2",
#             task_type=TaskType.SEMANTIC_SEGMENTATION,
#         )
#     assert "(20, 20) != (2, 2)" in str(exc_info)

#     # check get an error since "k3" isn't a label key in seg2
#     with pytest.raises(RuntimeError) as exc_info:
#         create_combined_segmentation_mask(
#             gt1,
#             label_key="k3",
#             task_type=TaskType.SEMANTIC_SEGMENTATION,
#         )
#     assert "doesn't have a label" in str(exc_info)

#     # should have one distinct (non-black) color
#     combined_mask, legend = create_combined_segmentation_mask(
#         gt1,
#         label_key="k1",
#         task_type=TaskType.SEMANTIC_SEGMENTATION,
#     )
#     combined_mask = np.array(combined_mask)
#     # check that we get two unique RGB values (black and one color for label value "v1")
#     assert combined_mask.shape == (2, 2, 3)
#     assert len(legend) == 1  # background color 'black' is not counted

#     # should have two distinct (non-black) color
#     combined_mask, legend = create_combined_segmentation_mask(
#         gt1, label_key="k2"
#     )
#     combined_mask = np.array(combined_mask)
#     # check that we get three unique RGB values (black and two colors for label values "v2" and "v3")
#     assert combined_mask.shape == (2, 2, 3)
#     assert len(legend) == 2  # background color 'black' is not counted
