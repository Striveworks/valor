import builtins
from collections import defaultdict

import pytest
from sqlalchemy.orm import Session

from valor_api import crud, enums, schemas
from valor_api.backend import models
from valor_api.backend.core import create_or_get_evaluations
from valor_api.backend.metrics.ranking import compute_ranking_metrics


def _parse_ranking_metrics(eval_metrics: list):
    """Parse the metrics attribute of an evaluation for easy comparison against a dictionary of expected values."""
    output = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for metric in eval_metrics:
        assert metric.parameters
        assert isinstance(metric.value, int | float)

        if metric.type == "MRRMetric":
            output["mrr"][metric.parameters["label_key"]] = metric.value  # type: ignore - defaultdict type error

        elif metric.type == "PrecisionAtKMetric":
            for aggregation in ("min", "max"):
                func = getattr(builtins, aggregation)
                existing_value = output[f"precision@k_{aggregation}"][
                    metric.label
                ][metric.parameters["k"]]

                output[f"precision@k_{aggregation}"][metric.label][
                    metric.parameters["k"]
                ] = func(metric.value, existing_value)

        elif metric.type == "APAtKMetric":
            for aggregation in ("min", "max"):
                func = getattr(builtins, aggregation)
                existing_value = output[f"ap@k_{aggregation}"].get(
                    metric.label, 0
                )

                output[f"ap@k_{aggregation}"][metric.label] = func(
                    metric.value, existing_value
                )

        elif metric.type == "mAPAtKMetric":
            output["map@k"][metric.parameters["label_key"]] = metric.value  # type: ignore - defaultdict type error

        else:
            raise ValueError("Encountered unknown metric type.")

    return output


@pytest.fixture
def ranking_test_data(
    db: Session,
    dataset_name: str,
    model_name: str,
    groundtruth_ranking: list[schemas.GroundTruth],
    prediction_ranking: list[schemas.Prediction],
):
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=dataset_name,
            metadata={"type": "image"},
        ),
    )
    for gt in groundtruth_ranking:
        crud.create_groundtruth(db=db, groundtruth=gt)
    crud.finalize(db=db, dataset_name=dataset_name)

    crud.create_model(
        db=db,
        model=schemas.Model(
            name=model_name,
            metadata={"type": "image"},
        ),
    )
    for pd in prediction_ranking:
        crud.create_prediction(db=db, prediction=pd)
    crud.finalize(db=db, dataset_name=dataset_name, model_name=model_name)

    assert len(db.query(models.Datum).all()) == 2
    assert len(db.query(models.Annotation).all()) == 9
    assert len(db.query(models.Label).all()) == 4
    assert len(db.query(models.GroundTruth).all()) == 2
    assert len(db.query(models.Prediction).all()) == 6


def test_ranking(
    db: Session,
    dataset_name: str,
    model_name: str,
    ranking_test_data,
):
    # default request
    job_request = schemas.EvaluationRequest(
        model_names=[model_name],
        datum_filter=schemas.Filter(dataset_names=[dataset_name]),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.RANKING,
        ),
        meta={},
    )

    # creates evaluation job
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status == enums.EvaluationStatus.PENDING

    # computation, normally run as background task
    _ = compute_ranking_metrics(
        db=db,
        evaluation_id=evaluations[0].id,
    )

    # get evaluations
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status in {
        enums.EvaluationStatus.RUNNING,
        enums.EvaluationStatus.DONE,
    }

    assert evaluations[0].metrics is not None

    metrics = _parse_ranking_metrics(evaluations[0].metrics)

    expected_metrics = {
        "mrr": {
            "k1": 0.5,  # (1 + .33 + .166 + .5) / 4
            "k2": 0,  # no predictions for this key
        },
        "precision@k_max": {
            # label : k_cutoff : max_value
            schemas.Label(key="k1", value="v1", score=None): {
                1: 1,  # only one annotation has a relevant doc in  k<=1
                3: 2,  # first and last docs have a two relevant docs in k<=3
                5: 3,  # the last annotation with this key has three relevant docs in k<=5
            }
        },
        "precision@k_min": {
            schemas.Label(key="k1", value="v1", score=None): {
                1: 0,
                3: 0,
                5: 0,
            }
        },
        "ap@k_max": {
            schemas.Label(
                key="k1", value="v1", score=None
            ): 1.6666666666666667  # the last annotation is (0 + 2 + 3) / 3
        },
        "ap@k_min": {schemas.Label(key="k1", value="v1", score=None): 0},
        # TODO should this include k2 if MRR does?
        "map@k": {
            "k1": 1.0833333333333333,  # (1 + 2 + 2) / 3 + (0 + 1 + 2) / 3 + 0 + 1.666667) / 4
        },
    }

    for metric_type, outputs in metrics.items():
        for k, v in outputs.items():
            assert expected_metrics[metric_type][k] == v

    for metric_type, outputs in expected_metrics.items():
        for k, v in outputs.items():
            assert metrics[metric_type][k] == v

    # check that the (k2, v2) annotation wasn't included in our metrics since it didn't have a corresponding groundtruth
    assert (
        len(
            set(
                [
                    metric.parameters["annotation_id"]  # type: ignore - we know metric.parameters exists for this metric type
                    for metric in evaluations[0].metrics
                    if metric.type == "PrecisionAtKMetric"
                ]
            )
        )
        == 4
    )


def test_ranking_with_label_map(
    db: Session,
    dataset_name: str,
    model_name: str,
    ranking_test_data,
):
    # test label map
    label_map = [[["k2", "v2"], ["k1", "v1"]]]

    # default request
    job_request = schemas.EvaluationRequest(
        model_names=[model_name],
        datum_filter=schemas.Filter(dataset_names=[dataset_name]),
        parameters=schemas.EvaluationParameters(
            task_type=enums.TaskType.RANKING, label_map=label_map
        ),
        meta={},
    )

    # creates evaluation job
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status == enums.EvaluationStatus.PENDING

    # computation, normally run as background task
    _ = compute_ranking_metrics(db=db, evaluation_id=evaluations[0].id)

    # get evaluations
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status in {
        enums.EvaluationStatus.RUNNING,
        enums.EvaluationStatus.DONE,
    }

    assert evaluations[0].metrics is not None

    metrics = _parse_ranking_metrics(evaluations[0].metrics)

    expected_metrics = {
        "mrr": {
            "k1": 0.4,  # (1 + .33 + .166 + .5 + 0) / 5
            "k2": 0,  # no predictions for this key
        },
        "precision@k_max": {
            # label : k_cutoff : max_value
            schemas.Label(key="k1", value="v1", score=None): {
                1: 1,  # only one annotation has a relevant doc in  k<=1
                3: 2,  # first and last docs have a two relevant docs in k<=3
                5: 3,  # the last annotation with this key has three relevant docs in k<=5
            }
        },
        "precision@k_min": {
            schemas.Label(key="k1", value="v1", score=None): {
                1: 0,
                3: 0,
                5: 0,
            }
        },
        "ap@k_max": {
            schemas.Label(key="k1", value="v1", score=None): 1.6666666666666667
        },
        "ap@k_min": {schemas.Label(key="k1", value="v1", score=None): 0},
        # TODO should this include k2 if MRR does?
        "map@k": {
            "k1": 0.8666666666666667,  # (1 + 2 + 2) / 3 + (0 + 1 + 2) / 3 + 0 + 1.666667 + 0) / 5
        },
    }

    for metric_type, outputs in metrics.items():
        for k, v in outputs.items():
            assert expected_metrics[metric_type][k] == v

    for metric_type, outputs in expected_metrics.items():
        for k, v in outputs.items():
            assert metrics[metric_type][k] == v

    # check that the label map added the extra annotation to our output
    assert (
        len(
            set(
                [
                    metric.parameters["annotation_id"]  # type: ignore - we know metric.parameters exists for this metric type
                    for metric in evaluations[0].metrics
                    if metric.type == "PrecisionAtKMetric"
                ]
            )
        )
        == 5
    )
