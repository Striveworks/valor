from dataclasses import replace
from pathlib import Path
from random import choice, uniform

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
    loader: Loader,
    tmp_path: Path,
    one_classification: list[Classification],
):

    loader.add_data(one_classification)
    evaluator = loader.finalize()

    # test evaluation
    filtered_evaluator = evaluator.filter(
        datums=pc.field("datum_uid") == "uid0", path=tmp_path / "filter"
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
    loader: Loader,
    three_classifications: list[Classification],
    tmp_path: Path,
):

    loader.add_data(three_classifications)
    evaluator = loader.finalize()

    # test evaluation
    filtered_evaluator = evaluator.filter(
        datums=pc.field("datum_uid") == "uid0", path=tmp_path / "filter"
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
    loader: Loader,
    six_classifications: list[Classification],
    tmp_path: Path,
):

    loader.add_data(six_classifications)
    evaluator = loader.finalize()

    # test evaluation
    filtered_evaluator = evaluator.filter(
        datums=pc.field("datum_uid") == "uid0", path=tmp_path / "filter"
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


def test_filtering_random_classifications(loader: Loader, tmp_path: Path):
    loader.add_data(generate_random_classifications(13, 2, 10))
    evaluator = loader.finalize()
    filtered_evaluator = evaluator.filter(
        datums=pc.field("datum_uid") == "uid0", path=tmp_path / "filter"
    )
    filtered_evaluator.compute_precision_recall(
        score_thresholds=[0.5],
        hardmax=False,
    )


def test_filtering_six_classifications_by_indices(
    loader: Loader,
    six_classifications: list[Classification],
    tmp_path: Path,
):

    loader.add_data(six_classifications)
    evaluator = loader.finalize()

    # test evaluation
    filtered_evaluator = evaluator.filter(
        datums=pc.field("datum_id") == 0, path=tmp_path / "filter"
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
    loader: Loader,
    six_classifications: list[Classification],
    tmp_path: Path,
):

    loader.add_data(six_classifications)
    evaluator = loader.finalize()

    # test groundtruth filter
    filtered_evaluator = evaluator.filter(
        datums=pc.field("datum_uid") == "uid0",
        groundtruths=pc.field("gt_label") != "0",
        path=tmp_path / "groundtruth_filter",
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
        path=tmp_path / "prediction_filter",
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


def test_filtering_six_classifications_inline(
    loader: Loader,
    six_classifications: list[Classification],
):

    loader.add_data(six_classifications)
    evaluator = loader.finalize()

    # test evaluation
    metrics = evaluator.compute_precision_recall(
        score_thresholds=[0.5],
        hardmax=False,
        datums=pc.field("datum_uid") == "uid0",
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


def test_filtering_remove_all(
    loader: Loader,
    six_classifications: list[Classification],
    tmp_path: Path,
):

    loader.add_data(six_classifications)
    evaluator = loader.finalize()

    datums = pc.field("datum_uid") == "does_not_exist"

    # test evaluation
    base_metrics = evaluator.compute_precision_recall(datums=datums)
    with pytest.raises(TypeError) as e:
        evaluator.compute_rocauc(datums=datums)  # type: ignore - testing
    assert "unexpected keyword" in str(e)
    confusion = evaluator.compute_confusion_matrix(datums=datums)
    examples = evaluator.compute_examples(datums=datums)

    for k, mlist in base_metrics.items():
        for m in mlist:
            if k == MetricType.Counts:
                assert isinstance(m.value, dict)
                for v in m.value.values():
                    assert isinstance(v, int)
                    assert v >= 0
            else:
                assert isinstance(m.value, float)
                assert m.value <= 1.0
                assert m.value >= 0.0
    for cm in confusion:
        assert isinstance(cm.value, dict)
        for row in cm.value["confusion_matrix"].values():
            for v in row.values():
                assert isinstance(v, int)
                assert v >= 0
        for v in cm.value["unmatched_ground_truths"].values():
            assert isinstance(v, int)
            assert v >= 0
    for example in examples:
        assert isinstance(example, dict)
        for v in example.values():
            if isinstance(v, list):
                assert len(v) == 0
