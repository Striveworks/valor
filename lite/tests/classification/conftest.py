import pytest
from valor_lite.classification import Classification


@pytest.fixture
def basic_classifications() -> list[Classification]:
    return [
        Classification(
            uid="uid0",
            groundtruth="0",
            predictions=[
                "0",
                "1",
                "2",
                "3",
            ],
            scores=[1.0, 0.0, 0.0, 0.0],
        ),
        Classification(
            uid="uid1",
            groundtruth="0",
            predictions=[
                "0",
                "1",
                "2",
                "3",
            ],
            scores=[0.0, 0.0, 1.0, 0.0],
        ),
        Classification(
            uid="uid2",
            groundtruth="3",
            predictions=[
                "0",
                "1",
                "2",
                "3",
            ],
            scores=[0.0, 0.0, 0.0, 0.3],
        ),
    ]


@pytest.fixture
def classifications_from_api_unit_tests() -> list[Classification]:
    """c.f. with

    ```
    from sklearn.metrics import classification_report

    y_true = [0, 0, 0, 1, 2, 2]
    y_pred = [0, 1, 2, 1, 1, 1]

    print(classification_report(y_true, y_pred))
    ```
    """
    return [
        Classification(
            uid="uid0",
            groundtruth="0",
            predictions=[
                "0",
                "1",
                "2",
            ],
            scores=[1.0, 0.0, 0.0],
        ),
        Classification(
            uid="uid1",
            groundtruth="0",
            predictions=[
                "0",
                "1",
                "2",
            ],
            scores=[0.0, 1.0, 0.0],
        ),
        Classification(
            uid="uid2",
            groundtruth="0",
            predictions=[
                "0",
                "1",
                "2",
            ],
            scores=[0.0, 0.0, 1.0],
        ),
        Classification(
            uid="uid3",
            groundtruth="1",
            predictions=[
                "0",
                "1",
                "2",
            ],
            scores=[0.0, 1.0, 0.0],
        ),
        Classification(
            uid="uid4",
            groundtruth="2",
            predictions=[
                "0",
                "1",
                "2",
            ],
            scores=[0.0, 1.0, 0.0],
        ),
        Classification(
            uid="uid5",
            groundtruth="2",
            predictions=[
                "0",
                "1",
                "2",
            ],
            scores=[0.0, 1.0, 0.0],
        ),
    ]


@pytest.fixture
def classifications_animal_example() -> list[Classification]:
    animal_gts = ["bird", "dog", "bird", "bird", "cat", "dog"]
    animal_pds = [
        {"bird": 0.6, "dog": 0.2, "cat": 0.2},
        {"cat": 0.9, "dog": 0.1, "bird": 0.0},
        {"cat": 0.8, "dog": 0.05, "bird": 0.15},
        {"dog": 0.75, "cat": 0.1, "bird": 0.15},
        {"cat": 1.0, "dog": 0.0, "bird": 0.0},
        {"cat": 0.4, "dog": 0.4, "bird": 0.2},
        # Note: In the case of a tied score, the ordering of predictions is used.
    ]

    return [
        Classification(
            uid=f"uid{idx}",
            groundtruth=gt,
            predictions=list(pd.keys()),
            scores=list(pd.values()),
        )
        for idx, (gt, pd) in enumerate(zip(animal_gts, animal_pds))
    ]


@pytest.fixture
def classifications_color_example() -> list[Classification]:
    color_gts = ["white", "white", "red", "blue", "black", "red"]
    color_pds = [
        {"white": 0.65, "red": 0.1, "blue": 0.2, "black": 0.05},
        {"blue": 0.5, "white": 0.3, "red": 0.0, "black": 0.2},
        {"red": 0.4, "white": 0.2, "blue": 0.1, "black": 0.3},
        {"white": 1.0, "red": 0.0, "blue": 0.0, "black": 0.0},
        {"red": 0.8, "white": 0.0, "blue": 0.2, "black": 0.0},
        {"red": 0.9, "white": 0.06, "blue": 0.01, "black": 0.03},
    ]

    return [
        Classification(
            uid=f"uid{idx}",
            groundtruth=gt,
            predictions=list(pd.keys()),
            scores=list(pd.values()),
        )
        for idx, (gt, pd) in enumerate(zip(color_gts, color_pds))
    ]


@pytest.fixture
def classifications_image_example() -> list[Classification]:
    return [
        Classification(
            uid="uid5",
            groundtruth="v4",
            predictions=[
                "v1",
                "v8",
            ],
            scores=[0.47, 0.53],
        ),
        Classification(
            uid="uid6",
            groundtruth="v4",
            predictions=[
                "v4",
                "v5",
            ],
            scores=[0.71, 0.29],
        ),
    ]


@pytest.fixture
def classifications_tabular_example() -> list[Classification]:
    gt_clfs_tabular = [1, 1, 2, 0, 0, 0, 1, 1, 1, 1]
    pred_clfs_tabular = [
        [0.37, 0.35, 0.28],
        [0.24, 0.61, 0.15],
        [0.03, 0.88, 0.09],
        [0.97, 0.03, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.01, 0.96, 0.03],
        [0.28, 0.02, 0.7],
        [0.78, 0.21, 0.01],
        [0.45, 0.11, 0.44],
    ]
    return [
        Classification(
            uid=f"uid{i}",
            groundtruth=str(gt_label),
            predictions=[str(pd_label) for pd_label, _ in enumerate(pds)],
            scores=pds,
        )
        for i, (gt_label, pds) in enumerate(
            zip(gt_clfs_tabular, pred_clfs_tabular)
        )
    ]


@pytest.fixture
def classifications_no_predictions() -> list[Classification]:
    return [
        Classification(
            uid="uid1",
            groundtruth="v1",
            predictions=[],
            scores=[],
        )
    ]


@pytest.fixture
def classifications_multiclass() -> list[Classification]:
    return [
        Classification(
            uid="uid0",
            groundtruth="cat",
            predictions=[
                "cat",
                "dog",
                "bee",
            ],
            scores=[
                0.44598543489942505,
                0.3255517969601126,
                0.22846276814046224,
            ],
        ),
        Classification(
            uid="uid1",
            groundtruth="bee",
            predictions=[
                "cat",
                "dog",
                "bee",
            ],
            scores=[
                0.4076893257212283,
                0.14780458563955237,
                0.4445060886392194,
            ],
        ),
        Classification(
            uid="uid2",
            groundtruth="cat",
            predictions=[
                "cat",
                "dog",
                "bee",
            ],
            scores=[
                0.25060075263871917,
                0.3467428086425673,
                0.4026564387187136,
            ],
        ),
        Classification(
            uid="uid3",
            groundtruth="bee",
            predictions=[
                "cat",
                "dog",
                "bee",
            ],
            scores=[
                0.2003514145616792,
                0.2485912151889644,
                0.5510573702493565,
            ],
        ),
        Classification(
            uid="uid4",
            groundtruth="dog",
            predictions=[
                "cat",
                "dog",
                "bee",
            ],
            scores=[
                0.33443897813714385,
                0.5890646197236098,
                0.07649640213924616,
            ],
        ),
    ]


@pytest.fixture
def classifications_multiclass_true_negatives_check() -> (
    list[Classification]
):
    return [
        Classification(
            uid="uid1",
            groundtruth="ant",
            predictions=["ant", "bee", "cat"],
            scores=[0.15, 0.48, 0.37],
        ),
    ]


@pytest.fixture
def classifications_multiclass_zero_count() -> list[Classification]:
    return [
        Classification(
            uid="uid1",
            groundtruth="ant",
            predictions=["ant", "bee", "cat"],
            scores=[0.15, 0.48, 0.37],
        )
    ]
