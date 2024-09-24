import pytest
from valor_lite.classification import Classification


@pytest.fixture
def classifications_basic() -> list[Classification]:
    return [
        Classification(
            uid="uid0",
            groundtruths=[("class", "0")],
            predictions=[
                ("class", "0"),
                ("class", "1"),
                ("class", "2"),
                ("class", "3"),
            ],
            scores=[1.0, 0.0, 0.0, 0.0],
        ),
        Classification(
            uid="uid1",
            groundtruths=[("class", "0")],
            predictions=[
                ("class", "0"),
                ("class", "1"),
                ("class", "2"),
                ("class", "3"),
            ],
            scores=[0.0, 0.0, 1.0, 0.0],
        ),
        Classification(
            uid="uid2",
            groundtruths=[("class", "3")],
            predictions=[
                ("class", "0"),
                ("class", "1"),
                ("class", "2"),
                ("class", "3"),
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
            groundtruths=[("class", "0")],
            predictions=[
                ("class", "0"),
                ("class", "1"),
                ("class", "2"),
            ],
            scores=[1.0, 0.0, 0.0],
        ),
        Classification(
            uid="uid1",
            groundtruths=[("class", "0")],
            predictions=[
                ("class", "0"),
                ("class", "1"),
                ("class", "2"),
            ],
            scores=[0.0, 1.0, 0.0],
        ),
        Classification(
            uid="uid2",
            groundtruths=[("class", "0")],
            predictions=[
                ("class", "0"),
                ("class", "1"),
                ("class", "2"),
            ],
            scores=[0.0, 0.0, 1.0],
        ),
        Classification(
            uid="uid3",
            groundtruths=[("class", "1")],
            predictions=[
                ("class", "0"),
                ("class", "1"),
                ("class", "2"),
            ],
            scores=[0.0, 1.0, 0.0],
        ),
        Classification(
            uid="uid4",
            groundtruths=[("class", "2")],
            predictions=[
                ("class", "0"),
                ("class", "1"),
                ("class", "2"),
            ],
            scores=[0.0, 1.0, 0.0],
        ),
        Classification(
            uid="uid5",
            groundtruths=[("class", "2")],
            predictions=[
                ("class", "0"),
                ("class", "1"),
                ("class", "2"),
            ],
            scores=[0.0, 1.0, 0.0],
        ),
    ]


@pytest.fixture
def classifications_two_categeories() -> list[Classification]:
    animal_gts = ["bird", "dog", "bird", "bird", "cat", "dog"]
    animal_pds = [
        {"bird": 0.6, "dog": 0.2, "cat": 0.2},
        {"cat": 0.9, "dog": 0.1, "bird": 0.0},
        {"cat": 0.8, "dog": 0.05, "bird": 0.15},
        {"dog": 0.75, "cat": 0.1, "bird": 0.15},
        {"cat": 1.0, "dog": 0.0, "bird": 0.0},
        {"cat": 0.4, "dog": 0.4, "bird": 0.2},
    ]

    color_gts = ["white", "white", "red", "blue", "black", "red"]
    color_pds = [
        {"white": 0.65, "red": 0.1, "blue": 0.2, "black": 0.05},
        {"blue": 0.5, "white": 0.3, "red": 0.0, "black": 0.2},
        {"red": 0.4, "white": 0.2, "blue": 0.1, "black": 0.3},
        {"white": 1.0, "red": 0.0, "blue": 0.0, "black": 0.0},
        {"red": 0.8, "white": 0.0, "blue": 0.2, "black": 0.0},
        {"red": 0.9, "white": 0.06, "blue": 0.01, "black": 0.03},
    ]

    joint_gts = zip(animal_gts, color_gts)
    joint_pds = [
        {
            "animal": animal,
            "color": color,
        }
        for animal, color in zip(animal_pds, color_pds)
    ]

    return [
        Classification(
            uid=f"uid{idx}",
            groundtruths=[("animal", gt[0]), ("color", gt[1])],
            predictions=[
                (key, value)
                for key, values in pd.items()
                for value in values.keys()
            ],
            scores=[
                score for values in pd.values() for score in values.values()
            ],
        )
        for idx, (gt, pd) in enumerate(zip(joint_gts, joint_pds))
    ]


@pytest.fixture
def classifications_image_example() -> list[Classification]:
    return [
        Classification(
            uid="uid5",
            groundtruths=[
                ("k4", "v4"),
                ("k5", "v5"),
            ],
            predictions=[
                ("k4", "v1"),
                ("k4", "v8"),
                ("k5", "v1"),
            ],
            scores=[0.47, 0.53, 1.0],
        ),
        Classification(
            uid="uid6",
            groundtruths=[
                ("k4", "v4"),
            ],
            predictions=[("k4", "v4"), ("k4", "v5")],
            scores=[0.71, 0.29],
        ),
        Classification(
            uid="uid8",
            groundtruths=[
                ("k3", "v3"),
            ],
            predictions=[
                ("k3", "v1"),
            ],
            scores=[
                1.0,
            ],
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
            groundtruths=[("class", str(gt_label))],
            predictions=[
                ("class", str(pd_label)) for pd_label, _ in enumerate(pds)
            ],
            scores=pds,
        )
        for i, (gt_label, pds) in enumerate(
            zip(gt_clfs_tabular, pred_clfs_tabular)
        )
    ]
