from typing import List, Tuple

import numpy as np
import pytest

from valor import Annotation, Dataset, Datum, Filter, GroundTruth, Label
from valor.enums import TaskType
from valor.schemas import BoundingBox, MultiPolygon, Polygon, Raster


@pytest.fixture
def heights_and_widths() -> List[Tuple[int, int]]:
    return [(10, 10), (12, 12), (17, 17), (20, 20)]


@pytest.fixture
def areas(heights_and_widths) -> List[int]:
    retvals = [100, 144, 289, 400]
    assert retvals == [h * w for h, w in heights_and_widths]
    return retvals


@pytest.fixture
def image_height_width(heights_and_widths) -> Tuple[int, int]:
    height = 100
    width = 100
    for h, w in heights_and_widths:
        assert height >= h
        assert width >= w
    return (height, width)


@pytest.fixture
def image_datum(image_height_width) -> Datum:
    h, w = image_height_width
    return Datum(
        uid="uid1",
        metadata={
            "height": h,
            "width": w,
        },
    )


def _create_raster(
    h: int, w: int, image_height_width, offset: int = 0
) -> Raster:
    raw_raster = np.zeros(image_height_width) == 1
    raw_raster[offset : w + offset, offset : h + offset] = True
    return Raster.from_numpy(raw_raster)


@pytest.fixture
def groundtruths_with_areas(
    heights_and_widths, image_height_width, image_datum
) -> List[GroundTruth]:
    groundtruths = []

    # create geometries
    for idx, hw in enumerate(heights_and_widths):
        h, w = hw
        bbox = BoundingBox.from_extrema(
            xmin=0,
            xmax=w,
            ymin=0,
            ymax=h,
        )
        polygon = Polygon(
            boundary=bbox.polygon,
        )
        multipolygon = MultiPolygon(polygons=[polygon])
        raster = _create_raster(h, w, image_height_width)

        groundtruths.extend(
            [
                GroundTruth(
                    datum=Datum(uid=f"box{idx}"),
                    annotations=[
                        Annotation(
                            task_type=TaskType.OBJECT_DETECTION,
                            labels=[Label(key="box", value=str(idx))],
                            bounding_box=bbox,
                        )
                    ],
                ),
                GroundTruth(
                    datum=Datum(uid=f"polygon{idx}"),
                    annotations=[
                        Annotation(
                            task_type=TaskType.OBJECT_DETECTION,
                            labels=[Label(key="polygon", value=str(idx))],
                            polygon=polygon,
                        )
                    ],
                ),
                GroundTruth(
                    datum=Datum(uid=f"multipolygon{idx}"),
                    annotations=[
                        Annotation(
                            task_type=TaskType.OBJECT_DETECTION,
                            labels=[Label(key="multipolygon", value=str(idx))],
                            raster=Raster.from_geometry(
                                multipolygon,
                                height=image_height_width[0],
                                width=image_height_width[1],
                            ),
                        )
                    ],
                ),
                GroundTruth(
                    datum=Datum(
                        uid=f"raster{idx}",
                        metadata={
                            "height": image_height_width[0],
                            "width": image_height_width[1],
                        },
                    ),
                    annotations=[
                        Annotation(
                            task_type=TaskType.OBJECT_DETECTION,
                            labels=[Label(key="raster", value=str(idx))],
                            raster=raster,
                        )
                    ],
                ),
            ]
        )

    return groundtruths


def test_filter_by_bounding_box(client, groundtruths_with_areas, areas):
    dataset = Dataset.create("myDataset")
    for gt in groundtruths_with_areas:
        dataset.add_groundtruth(gt)

    label_key = "box"
    all_labels = client.get_labels(
        Filter.create(
            [
                Annotation.bounding_box.exists(),
                Annotation.polygon.is_none(),
                Annotation.raster.is_none(),
            ]
        )
    )
    assert set(all_labels) == {
        Label(key=label_key, value=str(0)),
        Label(key=label_key, value=str(1)),
        Label(key=label_key, value=str(2)),
        Label(key=label_key, value=str(3)),
    }

    # threshold area
    for idx, area in enumerate(areas):
        thresholded_labels = client.get_labels(
            Filter.create(
                [
                    Annotation.bounding_box.exists(),
                    Annotation.polygon.is_none(),
                    Annotation.raster.is_none(),
                    Annotation.bounding_box.area > area,
                ]
            )
        )
        assert len(thresholded_labels) == len(areas) - idx - 1
        assert set(thresholded_labels) != {
            Label(key=label_key, value=str(0)),
            Label(key=label_key, value=str(1)),
            Label(key=label_key, value=str(2)),
            Label(key=label_key, value=str(3)),
        }
        assert set(thresholded_labels) == {
            Label(key=label_key, value=str(other_idx))
            for other_idx in range(len(areas))
            if other_idx > idx
        }


def test_filter_by_polygon(client, groundtruths_with_areas, areas):
    dataset = Dataset.create("myDataset")
    for gt in groundtruths_with_areas:
        dataset.add_groundtruth(gt)

    label_key = "polygon"
    all_labels = client.get_labels(
        Filter.create(
            [
                Annotation.bounding_box.is_none(),
                Annotation.polygon.exists(),
                Annotation.raster.is_none(),
            ]
        )
    )
    assert set(all_labels) == {
        Label(key=label_key, value=str(0)),
        Label(key=label_key, value=str(1)),
        Label(key=label_key, value=str(2)),
        Label(key=label_key, value=str(3)),
    }

    # threshold area
    for idx, area in enumerate(areas):
        thresholded_labels = client.get_labels(
            Filter.create(
                [
                    Annotation.bounding_box.is_none(),
                    Annotation.polygon.exists(),
                    Annotation.raster.is_none(),
                    Annotation.polygon.area > area,
                ]
            )
        )
        assert len(thresholded_labels) == len(areas) - idx - 1
        assert set(thresholded_labels) != {
            Label(key=label_key, value=str(0)),
            Label(key=label_key, value=str(1)),
            Label(key=label_key, value=str(2)),
            Label(key=label_key, value=str(3)),
        }
        assert set(thresholded_labels) == {
            Label(key=label_key, value=str(other_idx))
            for other_idx in range(len(areas))
            if other_idx > idx
        }


def test_filter_by_multipolygon(client, groundtruths_with_areas, areas):
    # NOTE - Valor currently transforms multipolygons into rasters.
    dataset = Dataset.create("myDataset")
    for gt in groundtruths_with_areas:
        dataset.add_groundtruth(gt)

    label_key = "multipolygon"
    all_labels = client.get_labels(
        Filter.create(
            [
                Label.key == label_key,
                Annotation.bounding_box.is_none(),
                Annotation.polygon.is_none(),
                Annotation.raster.exists(),
            ]
        )
    )
    assert set(all_labels) == {
        Label(key=label_key, value=str(0)),
        Label(key=label_key, value=str(1)),
        Label(key=label_key, value=str(2)),
        Label(key=label_key, value=str(3)),
    }

    # threshold area
    for idx, area in enumerate(areas):
        thresholded_labels = client.get_labels(
            Filter.create(
                [
                    Label.key == label_key,
                    Annotation.bounding_box.is_none(),
                    Annotation.polygon.is_none(),
                    Annotation.raster.exists(),
                    Annotation.raster.area > area,
                ]
            )
        )
        assert len(thresholded_labels) == len(areas) - idx - 1
        assert set(thresholded_labels) != {
            Label(key=label_key, value=str(0)),
            Label(key=label_key, value=str(1)),
            Label(key=label_key, value=str(2)),
            Label(key=label_key, value=str(3)),
        }
        assert set(thresholded_labels) == {
            Label(key=label_key, value=str(other_idx))
            for other_idx in range(len(areas))
            if other_idx > idx
        }


def test_filter_by_raster(client, groundtruths_with_areas, areas):
    dataset = Dataset.create("myDataset")
    for gt in groundtruths_with_areas:
        dataset.add_groundtruth(gt)

    label_key = "raster"
    all_labels = client.get_labels(
        Filter.create(
            [
                Label.key == label_key,
                Annotation.bounding_box.is_none(),
                Annotation.polygon.is_none(),
                Annotation.raster.exists(),
            ]
        )
    )
    assert set(all_labels) == {
        Label(key=label_key, value=str(0)),
        Label(key=label_key, value=str(1)),
        Label(key=label_key, value=str(2)),
        Label(key=label_key, value=str(3)),
    }

    # threshold area
    for idx, area in enumerate(areas):
        thresholded_labels = client.get_labels(
            Filter.create(
                [
                    Label.key == label_key,
                    Annotation.bounding_box.is_none(),
                    Annotation.polygon.is_none(),
                    Annotation.raster.exists(),
                    Annotation.raster.area > area,
                ]
            )
        )
        assert len(thresholded_labels) == len(areas) - idx - 1
        assert set(thresholded_labels) != {
            Label(key=label_key, value=str(0)),
            Label(key=label_key, value=str(1)),
            Label(key=label_key, value=str(2)),
            Label(key=label_key, value=str(3)),
        }
        assert set(thresholded_labels) == {
            Label(key=label_key, value=str(other_idx))
            for other_idx in range(len(areas))
            if other_idx > idx
        }
