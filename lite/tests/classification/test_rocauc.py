import numpy as np
from valor_lite.classification.computation import _compute_rocauc


def test__compute_rocauc():
    """
    Test ROC auc computation. This agrees with scikit-learn: the code (whose data
    comes from classification_test_data)

    animal_gts = ["bird", "dog", "bird", "bird", "cat", "dog"]
    animal_preds = [
        {"bird": 0.6, "dog": 0.2, "cat": 0.2},
        {"cat": 0.9, "dog": 0.1, "bird": 0.0},
        {"cat": 0.8, "dog": 0.05, "bird": 0.15},
        {"dog": 0.75, "cat": 0.1, "bird": 0.15},
        {"cat": 1.0, "dog": 0.0, "bird": 0.0},
        {"cat": 0.4, "dog": 0.4, "bird": 0.2},
    ]

    color_gts = ["white", "white", "red", "blue", "black", "red"]
    color_preds = [
        {"white": 0.65, "red": 0.1, "blue": 0.2, "black": 0.05},
        {"blue": 0.5, "white": 0.3, "red": 0.0, "black": 0.2},
        {"red": 0.4, "white": 0.2, "blue": 0.1, "black": 0.3},
        {"white": 1.0, "red": 0.0, "blue": 0.0, "black": 0.0},
        {"red": 0.8, "white": 0.0, "blue": 0.2, "black": 0.0},
        {"red": 0.9, "white": 0.06, "blue": 0.01, "black": 0.03},
    ]

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

    animals_and_colors = np.array(
        [
            # (animal, bird)
            [0, 0, 0, 0.6],
            [5, 1, 0, 0.2],
            [2, 0, 0, 0.15],
            [3, 0, 0, 0.15],
            [1, 1, 0, 0.0],
            [4, 2, 0, 0.0],
            # (animal, dog)
            [3, 0, 1, 0.75],
            [5, 1, 1, 0.4],
            [0, 0, 1, 0.2],
            [1, 1, 1, 0.1],
            [2, 0, 1, 0.05],
            [4, 2, 1, 0.0],
            # (animal, cat)
            [4, 2, 2, 1.0],
            [1, 1, 2, 0.9],
            [2, 0, 2, 0.8],
            [5, 1, 2, 0.4],
            [0, 0, 2, 0.2],
            [3, 0, 2, 0.1],
            # (color, white)
            [3, 5, 3, 1.0],
            [0, 3, 3, 0.65],
            [1, 3, 3, 0.3],
            [2, 4, 3, 0.2],
            [5, 4, 3, 0.06],
            [4, 6, 3, 0.0],
            # (color, red)
            [5, 4, 4, 0.9],
            [4, 6, 4, 0.8],
            [0, 3, 4, 0.1],
            [2, 4, 4, 0.4],
            [1, 3, 4, 0.0],
            [3, 5, 4, 0.0],
            # (color, blue)
            [1, 3, 5, 0.5],
            [0, 3, 5, 0.2],
            [4, 6, 5, 0.2],
            [2, 4, 5, 0.1],
            [5, 4, 5, 0.01],
            [3, 5, 5, 0.0],
            # (color, black)
            [2, 4, 6, 0.3],
            [1, 3, 6, 0.2],
            [0, 3, 6, 0.05],
            [5, 4, 6, 0.03],
            [3, 5, 6, 0.0],
            [4, 6, 6, 0.0],
        ],
        dtype=np.float64,
    )
    indices = np.argsort(-animals_and_colors[:, -1], axis=0)
    animals_and_colors = animals_and_colors[indices]

    # create args
    label_metadata = np.array(
        [
            [3, 6, 0],
            [2, 6, 0],
            [1, 6, 0],
            [2, 6, 1],
            [2, 6, 1],
            [1, 6, 1],
            [1, 6, 1],
        ],
        dtype=np.int32,
    )

    # compute ROCAUC and mROCAUC
    rocauc, mean_rocauc = _compute_rocauc(
        data=animals_and_colors,
        label_metadata=label_metadata,
        n_datums=6,
    )

    # test ROCAUC
    assert rocauc.shape == (label_metadata.shape[0],)
    assert rocauc[0] == 0.7777777777777778  # (animal, bird)
    assert rocauc[1] == 0.625  # (animal, dog)
    assert rocauc[2] == 1.0  # (animal, cat)
    assert rocauc[3] == 0.75  # (color, white)
    assert rocauc[4] == 0.875  # (color, red)
    assert rocauc[5] == 0.0  # (color, blue)
    assert rocauc[6] == 0.09999999999999998  # (color, black)

    # test mROCAUC
    assert mean_rocauc.shape == (np.unique(label_metadata[:, 2]).size,)
    assert (
        mean_rocauc == np.array([0.8009259259259259, 0.43125])  # animal, color
    ).all()
