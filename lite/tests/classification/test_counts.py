import numpy as np
from valor_lite.classification import compute_metrics


def test_compute_tp_count():

    data = np.array(
        [
            [0, 0, 0, 1.0],  # tp
            [0, 0, 1, 0.0],  # tn
            [1, 0, 2, 1.0],  # fp
            [2, 3, 3, 0.3],  # fn for score threshold > 0.3
        ],
        dtype=np.float64,
    )

    label_metadata = np.array(
        [
            [2, 1, 0],
            [0, 2, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=np.int32,
    )

    score_thresholds = np.array([0.5], dtype=np.float64)

    metrics = compute_metrics(
        data=data,
        label_metadata=label_metadata,
        score_thresholds=score_thresholds,
    )

    (
        counts,
        precision,
        recall,
        accuracy,
        f1_score,
        rocauc,
    ) = metrics

    print(rocauc)
