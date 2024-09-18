import pytest
from valor_lite.classification import Classification


@pytest.fixture
def basic_classifications():
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
def classifications():
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
            uid=str(idx),
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
