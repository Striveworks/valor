import PIL.Image

from velour.enums import TaskType
from velour.integrations.coco import _merge_annotations, coco_rle_to_mask
from velour.schemas import Label


def test__merge_annotations():
    """Check that we get the correct annotation set after merging semantic segmentions"""

    initial_annotations = [
        dict(
            task_type=TaskType.SEGMENTATION,
            labels=set(
                [Label(key="k1", value="v1"), Label(key="k2", value="v2")]
            ),
            mask=[[True, False, False, False], [True, False, False, False]],
        ),
        dict(
            task_type=TaskType.SEGMENTATION,
            labels=set(
                [Label(key="k1", value="v1"), Label(key="k3", value="v3")]
            ),
            mask=[[False, False, True, False], [False, False, True, False]],
        ),
        dict(
            task_type=TaskType.SEGMENTATION,
            labels=set(
                [
                    Label(key="k1", value="v1"),
                    Label(key="k2", value="v2"),
                    Label(key="k4", value="v4"),
                ]
            ),
            mask=[[False, False, False, True], [False, False, False, True]],
        ),
        dict(
            task_type=TaskType.DETECTION,
            labels=set(
                [Label(key="k1", value="v1"), Label(key="k3", value="v3")]
            ),
            mask=[[False, True, False, False], [False, True, False, False]],
        ),
    ]

    expected = [
        dict(
            task_type=TaskType.SEGMENTATION,
            labels=set([Label(key="k3", value="v3")]),
            mask=[[False, False, True, False], [False, False, True, False]],
        ),
        dict(
            task_type=TaskType.SEGMENTATION,
            labels=set([Label(key="k4", value="v4")]),
            mask=[[False, False, False, True], [False, False, False, True]],
        ),
        dict(
            task_type=TaskType.DETECTION,
            labels=set(
                [
                    Label(key="k1", value="v1"),
                    Label(key="k3", value="v3"),
                ]
            ),
            mask=[[False, True, False, False], [False, True, False, False]],
        ),
        dict(
            task_type=TaskType.SEGMENTATION,
            labels=set([Label(key="k1", value="v1")]),
            mask=[[True, False, True, True], [True, False, True, True]],
        ),
        dict(
            task_type=TaskType.SEGMENTATION,
            labels=set([Label(key="k2", value="v2")]),
            mask=[[True, False, False, True], [True, False, False, True]],
        ),
    ]

    label_map = {
        Label(key="k1", value="v1"): [0, 1, 2],
        Label(key="k2", value="v2"): [0, 2],
        Label(key="k3", value="v3"): [1],
        Label(key="k4", value="v4"): [2],
    }

    merged_annotations = _merge_annotations(
        annotation_list=initial_annotations, label_map=label_map
    )

    for i, v in enumerate(merged_annotations):
        assert (
            merged_annotations[i]["labels"] == expected[i]["labels"]
        ), "Labels didn't merge as expected"
        assert set(map(tuple, merged_annotations[i]["mask"])) == set(
            map(tuple, expected[i]["mask"])
        ), "Masks didn't merge as expected"


def test_coco_rle_to_mask():
    h, w = 4, 6
    coco_rle_seg_dict = {"counts": [10, 4, 3, 7], "size": (h, w)}

    mask = coco_rle_to_mask(coco_rle_seg_dict=coco_rle_seg_dict)
    expected_mask = [
        [False, False, False, True, False, True],
        [False, False, False, True, True, True],
        [False, False, True, False, True, True],
        [False, False, True, False, True, True],
    ]

    assert mask.sum() == 4 + 7
    assert mask.tolist() == expected_mask

    img = PIL.Image.fromarray(mask)

    assert img.width == w
    assert img.height == h
