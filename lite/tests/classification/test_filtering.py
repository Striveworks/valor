from dataclasses import replace
from random import choice, uniform

import numpy as np
import pytest
from valor_lite.classification import Classification, DataLoader, MetricType


@pytest.fixture
def one_classification(
    classifications_basic: list[Classification],
) -> list[Classification]:
    assert len(classifications_basic) == 3
    return [classifications_basic[0]]


@pytest.fixture
def three_classifications(
    classifications_basic: list[Classification],
) -> list[Classification]:
    assert len(classifications_basic) == 3
    return classifications_basic


@pytest.fixture
def six_classifications(
    classifications_basic: list[Classification],
) -> list[Classification]:
    assert len(classifications_basic) == 3
    clf1 = classifications_basic[0]
    clf2 = classifications_basic[1]
    clf3 = classifications_basic[2]

    clf4 = replace(classifications_basic[0])
    clf5 = replace(classifications_basic[1])
    clf6 = replace(classifications_basic[2])

    clf4.uid = "uid4"
    clf5.uid = "uid5"
    clf6.uid = "uid6"

    return [clf1, clf2, clf3, clf4, clf5, clf6]


def generate_random_classifications(
    n_classifications: int, n_categories: int, n_labels: int
) -> list[Classification]:

    labels = [
        [(str(key), str(value)) for value in range(n_labels)]
        for key in range(n_categories)
    ]

    return [
        Classification(
            uid=f"uid{i}",
            groundtruths=[choice(category) for category in labels],
            predictions=[label for category in labels for label in category],
            scores=[uniform(0, 1) for _ in range(n_labels * n_categories)],
        )
        for i in range(n_classifications)
    ]


def test_filtering_one_classification(
    one_classification: list[Classification],
):

    manager = DataLoader()
    manager.add_data(one_classification)
    evaluator = manager.finalize()

    assert evaluator._detailed_pairs.shape == (4, 5)
    assert (
        evaluator._detailed_pairs
        == np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0, 0.0],
            ]
        )
    ).all()

    assert evaluator._label_metadata_per_datum.shape == (2, 1, 4)
    assert (
        evaluator._label_metadata_per_datum
        == np.array(
            [
                [
                    [1, 0, 0, 0],
                ],
                [[1, 1, 1, 1]],
            ]
        )
    ).all()

    assert (
        evaluator._label_metadata
        == np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
    ).all()

    # test datum filtering

    filter_ = evaluator.create_filter(datum_uids=["uid0"])
    assert (filter_.indices == np.array([0, 1, 2, 3])).all()
    assert (
        filter_.label_metadata
        == np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
    ).all()

    # test label filtering

    filter_ = evaluator.create_filter(labels=[("class", "0")])
    assert (filter_.indices == np.array([0, 1, 2, 3])).all()

    filter_ = evaluator.create_filter(labels=[("class", "1")])
    assert (filter_.indices == np.array([])).all()

    # test label key filtering

    filter_ = evaluator.create_filter(label_keys=["class"])
    assert (filter_.indices == np.array([0, 1, 2, 3])).all()

    with pytest.raises(KeyError):
        filter_ = evaluator.create_filter(label_keys=["other"])

    # test combo
    filter_ = evaluator.create_filter(
        datum_uids=["uid0"],
        label_keys=["class"],
    )
    assert (filter_.indices == np.array([0, 1, 2, 3])).all()

    # test evaluation
    filter_ = evaluator.create_filter(datum_uids=["uid0"])
    metrics = evaluator.evaluate(
        score_thresholds=[0.5],
        hardmax=False,
        filter_=filter_,
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": [1],
                "fp": [0],
                "fn": [0],
                "tn": [0],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "class", "value": "0"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [0],
                "fn": [0],
                "tn": [1],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "class", "value": "1"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [0],
                "fn": [0],
                "tn": [1],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "class", "value": "2"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [0],
                "fn": [0],
                "tn": [1],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "class", "value": "3"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_filtering_three_classifications(
    three_classifications: list[Classification],
):

    manager = DataLoader()
    manager.add_data(three_classifications)
    evaluator = manager.finalize()

    assert evaluator._detailed_pairs.shape == (12, 5)
    assert (
        evaluator._detailed_pairs
        == np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 2.0, 1.0, 1.0],
                [2.0, 3.0, 3.0, 0.3, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 3.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [2.0, 3.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0],
                [2.0, 3.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0, 0.0],
                [1.0, 0.0, 3.0, 0.0, 0.0],
            ]
        )
    ).all()

    assert evaluator._label_metadata_per_datum.shape == (2, 3, 4)
    assert (
        evaluator._label_metadata_per_datum
        == np.array(
            [
                [
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                ],
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ],
            ]
        )
    ).all()

    assert (
        evaluator._label_metadata
        == np.array([[2, 3, 0], [0, 3, 0], [0, 3, 0], [1, 3, 0]])
    ).all()

    # test datum filtering

    filter_ = evaluator.create_filter(datum_uids=["uid0"])
    assert (filter_.indices == np.array([0, 5, 8, 10])).all()
    assert (
        filter_.label_metadata
        == np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
    ).all()

    filter_ = evaluator.create_filter(datum_uids=["uid2"])
    assert (filter_.indices == np.array([2, 4, 7, 9])).all()

    # test label filtering

    filter_ = evaluator.create_filter(labels=[("class", "0")])
    assert (filter_.indices == np.array([0, 1, 3, 5, 6, 8, 10, 11])).all()

    filter_ = evaluator.create_filter(labels=[("class", "1")])
    assert (filter_.indices == np.array([])).all()

    # test label key filtering

    filter_ = evaluator.create_filter(label_keys=["class"])
    assert (
        filter_.indices == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    ).all()

    with pytest.raises(KeyError):
        filter_ = evaluator.create_filter(label_keys=["other"])

    # test combo
    filter_ = evaluator.create_filter(
        datum_uids=["uid0"],
        label_keys=["class"],
    )
    assert (filter_.indices == np.array([0, 5, 8, 10])).all()

    # test evaluation
    filter_ = evaluator.create_filter(datum_uids=["uid0"])
    metrics = evaluator.evaluate(
        score_thresholds=[0.5],
        hardmax=False,
        filter_=filter_,
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": [1],
                "fp": [0],
                "fn": [0],
                "tn": [0],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "class", "value": "0"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [0],
                "fn": [0],
                "tn": [1],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "class", "value": "1"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [0],
                "fn": [0],
                "tn": [1],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "class", "value": "2"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [0],
                "fn": [0],
                "tn": [1],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "class", "value": "3"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_filtering_six_classifications(
    six_classifications: list[Classification],
):

    manager = DataLoader()
    manager.add_data(six_classifications)
    evaluator = manager.finalize()

    assert evaluator._detailed_pairs.shape == (24, 5)
    assert (
        evaluator._detailed_pairs
        == np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0],
                [3.0, 0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 2.0, 1.0, 1.0],
                [4.0, 0.0, 2.0, 1.0, 1.0],
                [2.0, 3.0, 3.0, 0.3, 1.0],
                [5.0, 3.0, 3.0, 0.3, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 3.0, 0.0, 0.0, 0.0],
                [5.0, 3.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [3.0, 0.0, 1.0, 0.0, 0.0],
                [4.0, 0.0, 1.0, 0.0, 0.0],
                [2.0, 3.0, 1.0, 0.0, 0.0],
                [5.0, 3.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0],
                [3.0, 0.0, 2.0, 0.0, 0.0],
                [2.0, 3.0, 2.0, 0.0, 0.0],
                [5.0, 3.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0, 0.0],
                [1.0, 0.0, 3.0, 0.0, 0.0],
                [3.0, 0.0, 3.0, 0.0, 0.0],
                [4.0, 0.0, 3.0, 0.0, 0.0],
            ]
        )
    ).all()

    assert evaluator._label_metadata_per_datum.shape == (2, 6, 4)
    assert (
        evaluator._label_metadata_per_datum
        == np.array(
            [
                [
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                ],
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ],
            ]
        )
    ).all()

    assert (
        evaluator._label_metadata
        == np.array([[4, 6, 0], [0, 6, 0], [0, 6, 0], [2, 6, 0]])
    ).all()

    # test datum filtering

    filter_ = evaluator.create_filter(datum_uids=["uid0"])
    assert (filter_.indices == np.array([0, 10, 16, 20])).all()
    assert (
        filter_.label_metadata
        == np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
    ).all()

    filter_ = evaluator.create_filter(datum_uids=["uid2"])
    assert (filter_.indices == np.array([4, 8, 14, 18])).all()

    # test label filtering

    filter_ = evaluator.create_filter(labels=[("class", "0")])
    assert (
        filter_.indices
        == np.array([0, 1, 2, 3, 6, 7, 10, 11, 12, 13, 16, 17, 20, 21, 22, 23])
    ).all()

    filter_ = evaluator.create_filter(labels=[("class", "1")])
    assert (filter_.indices == np.array([])).all()

    # test label key filtering

    filter_ = evaluator.create_filter(label_keys=["class"])
    assert (filter_.indices == np.arange(24)).all()

    with pytest.raises(KeyError):
        evaluator.create_filter(label_keys=["other"])

    # test combo
    filter_ = evaluator.create_filter(
        datum_uids=["uid0"],
        label_keys=["class"],
    )
    assert (filter_.indices == np.array([0, 10, 16, 20])).all()

    # test evaluation
    filter_ = evaluator.create_filter(datum_uids=["uid0"])

    metrics = evaluator.evaluate(
        score_thresholds=[0.5],
        hardmax=False,
        filter_=filter_,
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": [1],
                "fp": [0],
                "fn": [0],
                "tn": [0],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "class", "value": "0"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [0],
                "fn": [0],
                "tn": [1],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "class", "value": "1"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [0],
                "fn": [0],
                "tn": [1],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "class", "value": "2"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [0],
                "fn": [0],
                "tn": [1],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "class", "value": "3"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_filtering_random_classifications():
    loader = DataLoader()
    loader.add_data(generate_random_classifications(13, 2, 10))
    evaluator = loader.finalize()
    f = evaluator.create_filter(datum_uids=["uid0"])
    evaluator.evaluate(score_thresholds=[0.5], hardmax=False, filter_=f)
