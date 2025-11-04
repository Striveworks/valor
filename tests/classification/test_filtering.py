from dataclasses import replace
from pathlib import Path
from random import choice, uniform

import numpy as np
import pyarrow.compute as pc
import pytest

from valor_lite.classification import Classification, Loader, MetricType


@pytest.fixture
def one_classification(
    basic_classifications: list[Classification],
) -> list[Classification]:
    assert len(basic_classifications) == 3
    return [basic_classifications[0]]


@pytest.fixture
def three_classifications(
    basic_classifications: list[Classification],
) -> list[Classification]:
    assert len(basic_classifications) == 3
    return basic_classifications


@pytest.fixture
def six_classifications(
    basic_classifications: list[Classification],
) -> list[Classification]:
    assert len(basic_classifications) == 3
    clf1 = basic_classifications[0]
    clf2 = basic_classifications[1]
    clf3 = basic_classifications[2]

    clf4 = replace(basic_classifications[0])
    clf5 = replace(basic_classifications[1])
    clf6 = replace(basic_classifications[2])

    clf4.uid = "uid4"
    clf5.uid = "uid5"
    clf6.uid = "uid6"

    return [clf1, clf2, clf3, clf4, clf5, clf6]


def generate_random_classifications(
    n_classifications: int, n_categories: int, n_labels: int
) -> list[Classification]:

    labels = [str(value) for value in range(n_labels)]

    return [
        Classification(
            uid=f"uid{i}",
            groundtruth=choice(labels),
            predictions=labels,
            scores=[uniform(0, 1) for _ in range(n_labels)],
        )
        for i in range(n_classifications)
    ]


def test_filtering_one_classification(
    tmp_path: Path,
    one_classification: list[Classification],
):

    loader = Loader.persistent(tmp_path)
    loader.add_data(one_classification)
    evaluator = loader.finalize()

    # test evaluation
    filtered_evaluator = evaluator.filter(datums=["uid0"])
    metrics = filtered_evaluator.compute_precision_recall(
        score_thresholds=[0.5],
        hardmax=False,
    )
    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "0",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "1",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "2",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "3",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_filtering_three_classifications(
    tmp_path: Path,
    three_classifications: list[Classification],
):

    loader = Loader.persistent(tmp_path)
    loader.add_data(three_classifications)
    evaluator = loader.finalize()

    # test evaluation
    filtered_evaluator = evaluator.filter(datums=["uid0"])
    metrics = filtered_evaluator.compute_precision_recall(
        score_thresholds=[0.5],
        hardmax=False,
    )
    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "0",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "1",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "2",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "3",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_filtering_six_classifications(
    tmp_path: Path,
    six_classifications: list[Classification],
):

    loader = Loader.persistent(tmp_path)
    loader.add_data(six_classifications)
    evaluator = loader.finalize()

    # test evaluation
    filtered_evaluator = evaluator.filter(datums=["uid0"])
    metrics = filtered_evaluator.compute_precision_recall(
        score_thresholds=[0.5],
        hardmax=False,
    )
    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "0",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "1",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "2",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "3",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_filtering_random_classifications(tmp_path: Path):
    loader = Loader.persistent(tmp_path)
    loader.add_data(generate_random_classifications(13, 2, 10))
    evaluator = loader.finalize()
    filtered_evaluator = evaluator.filter(datums=["uid0"])
    filtered_evaluator.compute_precision_recall(
        score_thresholds=[0.5],
        hardmax=False,
    )


def test_filtering_six_classifications_by_indices(
    tmp_path: Path,
    six_classifications: list[Classification],
):

    loader = Loader.persistent(tmp_path)
    loader.add_data(six_classifications)
    evaluator = loader.finalize()

    # test evaluation
    filtered_evaluator = evaluator.filter(datums=np.array([0]))
    metrics = filtered_evaluator.compute_precision_recall(
        score_thresholds=[0.5],
        hardmax=False,
    )
    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "0",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "1",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "2",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "3",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_filtering_six_classifications_by_annotation(
    tmp_path: Path,
    six_classifications: list[Classification],
):

    loader = Loader.persistent(tmp_path)
    loader.add_data(six_classifications)
    evaluator = loader.finalize()

    # test groundtruth filter
    filtered_evaluator = evaluator.filter(
        datums=pc.field("datum_uid") == "uid0",
        groundtruths=pc.field("gt_label") != "0",
    )
    metrics = filtered_evaluator.compute_precision_recall(
        score_thresholds=[0.5],
        hardmax=False,
    )
    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "0",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "1",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "2",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "3",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test prediction filter
    filtered_evaluator = evaluator.filter(
        datums=pc.field("datum_uid") == "uid0",
        predictions=pc.field("pd_label") != "0",
    )
    metrics = filtered_evaluator.compute_precision_recall(
        score_thresholds=[0.5],
        hardmax=False,
    )
    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "0",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "1",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "2",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": False,
                "label": "3",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
