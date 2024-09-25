import numpy as np
from valor_lite.detection import compute_iou, BoundingBox, Bitmask
import pytest


def test_compute_bbox_iou():

    # xmin, xmax, ymin, ymax
    box1 = np.array([0.0, 10.0, 0.0, 10.0])
    box2 = np.array([5.0, 10.0, 0.0, 10.0])
    box3 = np.array([0.0, 5.0, 5.0, 10.0])
    box4 = np.array([5.0, 15.0, 0.0, 10.0])
    box5 = np.array([0.0, 15.0, 0.0, 10.0])

    pairs = np.array(
        [
            np.concatenate((box1, box1)),
            np.concatenate((box1, box2)),
            np.concatenate((box1, box3)),
            np.concatenate((box1, box4)),
            np.concatenate((box1, box5)),
        ]
    )

    ious = compute_iou(pairs, annotation_type=BoundingBox)
    assert len(ious) == 5
    assert ious[0] == 1.0
    assert ious[1] == 0.5
    assert ious[2] == 0.25
    assert round(ious[3], 5) == 0.33333
    assert round(ious[4], 5) == 0.66667


def test_compute_bitmask_iou():
    filled_8x8 = np.full((8, 8), True)
    filled_10x10 = (np.full((10, 10), True),)

    gt_bitmasks = [
        filled_10x10,
        filled_10x10,
        [
            [True, True, True, True, True, False, False, False],
            [True, True, True, True, True, False, False, False],
            [True, True, True, True, True, False, False, False],
            [True, True, True, True, True, False, False, False],
            [True, True, True, True, True, False, False, False],
            [True, True, True, True, True, False, False, False],
            [True, True, True, True, True, False, False, False],
            [True, True, True, True, True, False, False, False],
        ],
        filled_8x8,
        filled_8x8,
    ]

    pd_bitmasks = [
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
        ],
        [
            [
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
            [
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
            [
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
            [
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
            [
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
            [
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
            [
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
            [
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
            [
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
            [
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
        ],
        [
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, True, True, True],
        ],
        [
            [False, False, False, False, True, True, True, True],
            [False, False, False, False, True, True, True, True],
            [False, False, False, False, True, True, True, True],
            [False, False, False, False, True, True, True, True],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
        ],
        [
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, True, True, True],
            [True, True, True, False, False, True, True, True],
            [True, True, True, False, False, True, True, True],
            [True, True, True, False, False, True, True, True],
            [True, True, True, False, False, True, True, True],
        ],
    ]

    data = np.array(list(zip(gt_bitmasks, pd_bitmasks)), dtype=object)

    result = compute_iou(data=data, annotation_type=Bitmask)

    assert (result == [1, 0.5, 0, 0.25, 36 / 64]).all()

    gt_bitmasks = [
        filled_10x10,
        filled_10x10,
    ]

    pd_bitmasks = [
        filled_10x10,
        filled_8x8,
    ]

    data = np.array(list(zip(gt_bitmasks, pd_bitmasks)), dtype=object)

    with pytest.raises(ValueError) as e:
        compute_iou(data=data, annotation_type=Bitmask)
    assert "operands could not be broadcast together with shapes" in str(e)


# TODO add polygon test
