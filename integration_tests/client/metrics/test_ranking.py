""" These integration tests should be run with a back end at http://localhost:8000
that is no auth
"""

import pytest

from valor import (
    Annotation,
    Client,
    Dataset,
    Datum,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from valor.enums import EvaluationStatus, TaskType
from valor.exceptions import ClientException


@pytest.fixture
def ranking_gts():
    return [
        GroundTruth(
            datum=Datum(uid="uid1", metadata={}),
            annotations=[
                Annotation(
                    task_type=TaskType.RANKING,
                    labels=[
                        Label(key="k1", value="v1"),
                    ],
                    ranking=[
                        "relevant_doc1",
                        "relevant_doc2",
                        "relevant_doc3",
                        "relevant_doc4",
                    ],
                )
            ],
        ),
        GroundTruth(
            datum=Datum(uid="uid2", metadata={}),
            annotations=[
                Annotation(
                    task_type=TaskType.RANKING,
                    labels=[
                        Label(key="k2", value="v2"),
                    ],
                    ranking=[
                        "1",
                        "2",
                        "3",
                        "4",
                    ],
                )
            ],
        ),
    ]


@pytest.fixture
def ranking_pds():
    return [
        Prediction(
            datum=Datum(uid="uid1", metadata={}),
            annotations=[
                Annotation(
                    task_type=TaskType.RANKING,
                    labels=[
                        Label(key="k1", value="v1"),
                    ],
                    ranking=[
                        "foo",
                        "bar",
                        "relevant_doc2",
                    ],
                ),
                Annotation(
                    task_type=TaskType.RANKING,
                    labels=[
                        Label(key="k1", value="v1"),
                    ],
                    ranking=[
                        "foo",
                        "relevant_doc4",
                        "relevant_doc1",
                    ],
                ),
                # this prediction will be ignored since it doesn't have a groundtruth with a matching key/value
                Annotation(
                    task_type=TaskType.RANKING,
                    labels=[
                        Label(key="k1", value="v2"),
                    ],
                    ranking=["bbq", "iguana", "relevant_doc1"],
                ),
                Annotation(
                    task_type=TaskType.RANKING,
                    labels=[
                        Label(key="k3", value="k1"),
                    ],
                    ranking=[
                        "relevant_doc2",
                        "foo",
                        "relevant_doc1",
                    ],
                ),
            ],
        ),
        Prediction(
            datum=Datum(uid="uid2", metadata={}),
            annotations=[
                Annotation(
                    task_type=TaskType.RANKING,
                    labels=[
                        Label(key="k2", value="v2"),
                    ],
                    ranking=[
                        "4",
                        "1",
                    ],
                ),
                Annotation(
                    task_type=TaskType.RANKING,
                    labels=[
                        Label(key="k2", value="v2"),
                    ],
                    ranking=[
                        "foo",
                        "bar",
                    ],
                ),
            ],
        ),
    ]


def test_evaluate_ranking(
    client: Client,
    dataset_name: str,
    model_name: str,
    ranking_gts: list,
    ranking_pds: list,
):
    dataset = Dataset.create(dataset_name)
    model = Model.create(model_name)

    for gt in ranking_gts:
        dataset.add_groundtruth(gt)

    dataset.finalize()

    for pred in ranking_pds:
        model.add_prediction(dataset, pred)

    model.finalize_inferences(dataset)

    eval_job = model.evaluate_ranking(dataset)

    assert eval_job.id

    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metrics = eval_job.metrics

    assert metrics

    expected_metrics = [
        # parameters are left in for future revivew, but can't be used for assertions as the annotation_id varies across runs
        {
            "type": "MRRMetric",
            # "parameters": {"label_key": "k1"},
            "value": 0.4166666666666667,
        },  # (1/3 + 1/2)/2
        {
            "type": "MRRMetric",
            # "parameters": {"label_key": "k2"},
            "value": 0.5,
        },  # (1 + 0) / 2
        {
            "type": "PrecisionAtKMetric",
            # "parameters": {"k": 1, "annotation_id": 309},
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "RecallAtKMetric",
            # "parameters": {"k": 1, "annotation_id": 309},
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "PrecisionAtKMetric",
            # "parameters": {"k": 3, "annotation_id": 309},
            "value": 0.3333333333333333,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "RecallAtKMetric",
            # "parameters": {"k": 3, "annotation_id": 309},
            "value": 0.25,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "PrecisionAtKMetric",
            # "parameters": {"k": 5, "annotation_id": 309},
            "value": 0.2,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "RecallAtKMetric",
            # "parameters": {"k": 5, "annotation_id": 309},
            "value": 0.25,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "PrecisionAtKMetric",
            # "parameters": {"k": 1, "annotation_id": 310},
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "RecallAtKMetric",
            # "parameters": {"k": 1, "annotation_id": 310},
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "PrecisionAtKMetric",
            # "parameters": {"k": 3, "annotation_id": 310},
            "value": 0.6666666666666666,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "RecallAtKMetric",
            # "parameters": {"k": 3, "annotation_id": 310},
            "value": 0.5,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "PrecisionAtKMetric",
            # "parameters": {"k": 5, "annotation_id": 310},
            "value": 0.4,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "RecallAtKMetric",
            # "parameters": {"k": 5, "annotation_id": 310},
            "value": 0.5,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "PrecisionAtKMetric",
            # "parameters": {"k": 1, "annotation_id": 313},
            "value": 1.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "RecallAtKMetric",
            # "parameters": {"k": 1, "annotation_id": 313},
            "value": 0.25,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "PrecisionAtKMetric",
            # "parameters": {"k": 3, "annotation_id": 313},
            "value": 0.6666666666666666,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "RecallAtKMetric",
            # "parameters": {"k": 3, "annotation_id": 313},
            "value": 0.5,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "PrecisionAtKMetric",
            # "parameters": {"k": 5, "annotation_id": 313},
            "value": 0.4,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "RecallAtKMetric",
            # "parameters": {"k": 5, "annotation_id": 313},
            "value": 0.5,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "PrecisionAtKMetric",
            # "parameters": {"k": 1, "annotation_id": 314},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "RecallAtKMetric",
            # "parameters": {"k": 1, "annotation_id": 314},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "PrecisionAtKMetric",
            # "parameters": {"k": 3, "annotation_id": 314},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "RecallAtKMetric",
            # "parameters": {"k": 3, "annotation_id": 314},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "PrecisionAtKMetric",
            # "parameters": {"k": 5, "annotation_id": 314},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "RecallAtKMetric",
            # "parameters": {"k": 5, "annotation_id": 314},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "APAtKMetric",
            # "parameters": {"k_cutoffs": [1, 3, 5], "annotation_id": 309},
            "value": 0.17777777777777778,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "ARAtKMetric",
            # "parameters": {"k_cutoffs": [1, 3, 5], "annotation_id": 309},
            "value": 0.16666666666666666,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "APAtKMetric",
            # "parameters": {"k_cutoffs": [1, 3, 5], "annotation_id": 314},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "ARAtKMetric",
            # "parameters": {"k_cutoffs": [1, 3, 5], "annotation_id": 314},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "APAtKMetric",
            # "parameters": {"k_cutoffs": [1, 3, 5], "annotation_id": 310},
            "value": 0.35555555555555557,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "ARAtKMetric",
            # "parameters": {"k_cutoffs": [1, 3, 5], "annotation_id": 310},
            "value": 0.3333333333333333,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "APAtKMetric",
            # "parameters": {"k_cutoffs": [1, 3, 5], "annotation_id": 313},
            "value": 0.6888888888888889,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "ARAtKMetric",
            # "parameters": {"k_cutoffs": [1, 3, 5], "annotation_id": 313},
            "value": 0.4166666666666667,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "mAPAtKMetric",
            # "parameters": {"k_cutoffs": [1, 3, 5], "label_key": "k1"},
            "value": 0.26666666666666666,  # ((0 + 1/3 + 1/5)/3 + (0 + 2/3 + 2/5)/3)/2
        },
        {
            "type": "mARAtKMetric",
            # "parameters": {"k_cutoffs": [1, 3, 5], "label_key": "k1"},
            "value": 0.25,
        },
        {
            "type": "mAPAtKMetric",
            # "parameters": {"k_cutoffs": [1, 3, 5], "label_key": "k2"},
            "value": 0.34444444444444444,
        },
        {
            "type": "mARAtKMetric",
            # "parameters": {"k_cutoffs": [1, 3, 5], "label_key": "k2"},
            "value": 0.20833333333333334,  # ((1/4 + 2/4 + 2/4)/3 + (0 + 0 + 0)/3)/2
        },
    ]

    for metric in metrics:
        metric.pop("parameters")
        assert metric in expected_metrics

    for metric in expected_metrics:
        assert metric in metrics

    # test label map with a reduced number of metrics
    label_mapping = {
        Label(key="k2", value="v2"): Label(key="k1", value="v1"),
        Label(key="k1", value="v2"): Label(key="k1", value="v1"),
    }

    eval_job = model.evaluate_ranking(
        dataset,
        label_map=label_mapping,
        metrics_to_return=["MRRMetric", "PrecisionAtKMetric"],
        k_cutoffs=[3],
    )

    assert eval_job.id

    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metrics = eval_job.metrics

    assert metrics

    expected_metrics = [
        {
            "type": "MRRMetric",
            # "parameters": {"label_key": "k1"},
            "value": 0.43333333333333335,
        },  # (1/3 + 1/2 + 1/3 + 0 + 1) / 5
        {
            "type": "PrecisionAtKMetric",
            "value": 0.3333333333333333,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "PrecisionAtKMetric",
            "value": 0.6666666666666666,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "PrecisionAtKMetric",
            "value": 0.3333333333333333,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "PrecisionAtKMetric",
            "value": 0.6666666666666666,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "PrecisionAtKMetric",
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
    ]

    for metric in metrics:
        metric.pop("parameters")
        assert metric in expected_metrics

    for metric in expected_metrics:
        assert metric in metrics


def test_evaluate_ranking_error_cases(
    client: Client,
    dataset_name: str,
    model_name: str,
    ranking_gts: list,
    ranking_pds: list,
):
    dataset = Dataset.create(dataset_name)
    model = Model.create(model_name)

    for gt in ranking_gts:
        dataset.add_groundtruth(gt)

    dataset.finalize()

    for pred in ranking_pds:
        model.add_prediction(dataset, pred)

    model.finalize_inferences(dataset)

    # test bad k_cutoffs
    with pytest.raises(ClientException):
        model.evaluate_ranking(dataset, k_cutoffs="a")

    with pytest.raises(ClientException):
        model.evaluate_ranking(dataset, k_cutoffs=["a"])

    with pytest.raises(ClientException):
        model.evaluate_ranking(dataset, k_cutoffs=[0.1, 0.2])

    with pytest.raises(ClientException):
        model.evaluate_ranking(dataset, k_cutoffs=1)

    # test bad metric names
    with pytest.raises(ClientException):
        model.evaluate_ranking(dataset, metrics_to_return=["fake_name"])

    with pytest.raises(ClientException):
        model.evaluate_ranking(dataset, metrics_to_return=[1])

    with pytest.raises(ClientException):
        model.evaluate_ranking(dataset, metrics_to_return="fake_name")

    with pytest.raises(ClientException):
        model.evaluate_ranking(
            dataset, metrics_to_return=["MRRMetric", "fake_name"]
        )

    # test bad label maps
    with pytest.raises(TypeError):
        model.evaluate_ranking(dataset, label_map=["foo", "bar"])

    with pytest.raises(TypeError):
        model.evaluate_ranking(dataset, label_map={"foo": "bar"})

    with pytest.raises(TypeError):
        model.evaluate_ranking(
            dataset, label_map={Label(key="foo", value="bar"): "bar"}
        )
