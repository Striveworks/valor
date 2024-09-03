import numpy as np
from valor_core.evaluator import _compute_ap, _compute_iou


def test__compute_iou():

    # datum id  0
    # gt        1
    # pd        2
    # gt xmin   3
    # gt xmax   4
    # gt ymin   5
    # gt ymax   6
    # pd xmin   7
    # pd xmax   8
    # pd ymin   9
    # pd ymax   10
    # gt label  11
    # pd label  12
    # pd score  13

    box1 = (0.0, 10.0, 0.0, 10.0)
    box2 = (5.0, 10.0, 0.0, 10.0)
    box3 = (0.0, 5.0, 5.0, 10.0)
    box4 = (5.0, 15.0, 0.0, 10.0)
    box5 = (0.0, 15.0, 0.0, 10.0)

    pairs = np.array(
        [
            [0.0, 0.0, 0.0, *box1, *box1, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, *box1, *box2, 0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0, *box1, *box3, 0.0, 0.0, 1.0],
            [0.0, 0.0, 3.0, *box1, *box4, 0.0, 0.0, 1.0],
            [0.0, 0.0, 4.0, *box1, *box5, 0.0, 0.0, 1.0],
        ]
    )
    iou_pairs = _compute_iou([pairs])

    assert iou_pairs[0][0][3] == 1.0
    assert iou_pairs[0][1][3] == 0.5
    assert iou_pairs[0][2][3] == 0.25
    assert round(iou_pairs[0][3][3], 5) == 0.33333
    assert round(iou_pairs[0][4][3], 5) == 0.66667


def test__compute_ap():

    box1 = (0.0, 10.0, 0.0, 10.0)
    box2 = (5.0, 10.0, 0.0, 10.0)
    box3 = (0.0, 5.0, 5.0, 10.0)
    box4 = (5.0, 15.0, 0.0, 10.0)
    box5 = (0.0, 15.0, 0.0, 10.0)

    sorted_pairs = np.array(
        [
            [0.0, 0.0, 2.0, *box1, *box3, 0.0, 0.0, 0.95],
            [0.0, 0.0, 3.0, *box1, *box4, 0.0, 0.0, 0.9],
            [0.0, 0.0, 4.0, *box1, *box5, 0.0, 0.0, 0.65],
            [0.0, 0.0, 0.0, *box1, *box1, 0.0, 0.0, 0.1],
            [0.0, 0.0, 1.0, *box1, *box2, 0.0, 0.0, 0.01],
        ]
    )
    sorted_iou_pairs = _compute_iou([sorted_pairs])

    gt_counts = np.array([1])
    iou_thresholds = np.array([0.1, 0.6])

    output = _compute_ap(
        sorted_iou_pairs,
        gt_counts=gt_counts,
        iou_thresholds=iou_thresholds,
    )

    print(output)
