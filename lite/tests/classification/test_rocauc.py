import numpy as np
from valor_lite.classification.computation import _compute_rocauc


def test__compute_rocauc():
    """
    Test ROC auc computation. This agrees with scikit-learn: the code (whose data
    comes from classification_test_data)

    ```
    from sklearn.metrics import roc_auc_score

    # for the "animal" label key
    y_true = [0, 2, 0, 0, 1, 2]
    y_score = [
        [0.6, 0.2, 0.2],
        [0.0, 0.9, 0.1],
        [0.15, 0.8, 0.05],
        [0.15, 0.1, 0.75],
        [0.0, 1.0, 0.0],
        [0.2, 0.4, 0.4],
    ]

    print(roc_auc_score(y_true, y_score, multi_class="ovr"))

    # for the "color" label key
    y_true = [3, 3, 2, 1, 0, 2]
    y_score = [
        [0.05, 0.2, 0.1, 0.65],
        [0.2, 0.5, 0.0, 0.3],
        [0.3, 0.1, 0.4, 0.2],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.2, 0.8, 0.0],
        [0.03, 0.01, 0.9, 0.06],
    ]
    ```

    outputs:

    ```
    0.8009259259259259
    0.43125
    ```
    """

    animal_class = np.array(
        [
            [4, 2, 2, 1.0],
            [1, 1, 2, 0.9],
            [2, 0, 2, 0.8],
            [3, 0, 1, 0.75],
            [0, 0, 0, 0.6],
            [5, 1, 2, 0.4],
            [5, 1, 1, 0.4],
            [0, 0, 1, 0.2],
            [0, 0, 2, 0.2],
            [5, 1, 0, 0.2],
            [3, 0, 0, 0.15],
            [2, 0, 0, 0.15],
            [1, 1, 1, 0.1],
            [3, 0, 2, 0.1],
            [2, 0, 1, 0.05],
            [1, 1, 0, 0.0],
            [4, 2, 1, 0.0],
            [4, 2, 0, 0.0],
        ],
        dtype=np.float64,
    )

    color_gts = ["white", "white", "red", "blue", "black", "red"]
    color_preds = [
        {"white": 0.65, "red": 0.1, "blue": 0.2, "black": 0.05},
        {"blue": 0.5, "white": 0.3, "red": 0.0, "black": 0.2},
        {"red": 0.4, "white": 0.2, "blue": 0.1, "black": 0.3},
        {"white": 1.0, "red": 0.0, "blue": 0.0, "black": 0.0},
        {"red": 0.8, "white": 0.0, "blue": 0.2, "black": 0.0},
        {"red": 0.9, "white": 0.06, "blue": 0.01, "black": 0.03},
    ]

    pd_label_keys = np.zeros((animal_class.shape[0],), dtype=np.int32)
    mask_gt_exists = animal_class[:, 1] >= 0.0
    mask_matching_labels = np.isclose(animal_class[:, 1], animal_class[:, 2])

    rocauc = _compute_rocauc(
        pd_label_keys=pd_label_keys,
        mask_gt_exists=mask_gt_exists,
        mask_matching_labels=mask_matching_labels,
        n_label_keys=1,
    )

    print(rocauc)

    # assert len(results) == 2
    # assert results["animal"] == 0.8009259259259259
    # assert results["color"] == 0.43125
