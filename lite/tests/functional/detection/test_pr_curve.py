import numpy as np
from valor_lite.detection import compute_metrics


def test__compute_average_precision():

    sorted_pairs = np.array(
        [
            # dt,  gt,  pd,  iou,  gl,  pl, score,
            [0.0, 0.0, 2.0, 0.25, 0.0, 0.0, 0.95],
            [0.0, 0.0, 3.0, 0.33333, 0.0, 0.0, 0.9],
            [0.0, 0.0, 4.0, 0.66667, 0.0, 0.0, 0.65],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1],
            [0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.01],
        ]
    )

    label_counts = np.array([[1, 5, 0]])
    iou_thresholds = np.array([0.1, 0.6])
    score_thresholds = np.array([0.0])

    (_, _, _, _, _, pr_curve) = compute_metrics(
        sorted_pairs,
        label_counts=label_counts,
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
    )

    assert pr_curve.shape == (2, 1, 101)
    assert np.isclose(pr_curve[0][0], 1.0).all()
    assert np.isclose(pr_curve[1][0], 1 / 3).all()
