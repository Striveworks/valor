import numpy as np
from valor_lite.detection import _compute_iou


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

    ious = _compute_iou(pairs)
    assert len(ious) == 5
    assert ious[0] == 1.0
    assert ious[1] == 0.5
    assert ious[2] == 0.25
    assert round(ious[3], 5) == 0.33333
    assert round(ious[4], 5) == 0.66667
