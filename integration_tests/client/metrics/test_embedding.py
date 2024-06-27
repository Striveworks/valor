import numpy as np
import pytest
from scipy.stats import cramervonmises_2samp, ks_2samp
from sqlalchemy.orm import Session

from valor import Client, Dataset, GroundTruth, Model, Prediction, schemas
from valor.enums import EvaluationStatus, MetricType
from valor_api.backend import core
from valor_api.backend.metrics.embedding import (
    _compute_distances,
    _compute_self_distances,
)


@pytest.fixture
def labels() -> list[schemas.Label]:
    return [
        schemas.Label(key="class", value="A"),
        schemas.Label(key="class", value="B"),
        schemas.Label(key="class", value="C"),
        schemas.Label(key="class", value="D"),
    ]


@pytest.fixture
def number_of_samples() -> int:
    return 10


@pytest.fixture
def offsets() -> list[float]:
    return [0, 0.1, 0.5, 2.0]


@pytest.fixture
def dataset(
    client: Client,
    dataset_name: str,
    labels: list[schemas.Label],
    number_of_samples: int,
) -> str:
    dataset = Dataset.create(dataset_name)
    dataset.add_groundtruths(
        groundtruths=[
            GroundTruth(
                datum=schemas.Datum(
                    uid=str(label_idx * number_of_samples + idx)
                ),
                annotations=[schemas.Annotation(labels=[label])],
            )
            for label_idx, label in enumerate(labels)
            for idx in range(number_of_samples)
        ],
    )
    dataset.finalize()
    return dataset_name


@pytest.fixture
def fixed_model(
    client: Client,
    dataset_name: str,
    labels: list[schemas.Label],
    number_of_samples: int,
    offsets: list[int],
) -> str:
    model_name = "fixed"

    embeddings = []
    for offset in offsets:
        embeddings.append([])
        for _ in range(number_of_samples):
            embeddings[-1].append([offset, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    dataset = Dataset.get(dataset_name)
    assert dataset
    model = Model.create(model_name)
    model.add_predictions(
        dataset=dataset,
        predictions=[
            Prediction(
                datum=schemas.Datum(
                    uid=str(label_idx * number_of_samples + idx)
                ),
                annotations=[
                    schemas.Annotation(embedding=embeddings[label_idx][idx])
                ],
            )
            for label_idx, label in enumerate(labels)
            for idx in range(number_of_samples)
        ],
    )
    return model_name


@pytest.fixture
def uniform_model(
    client: Client,
    dataset_name: str,
    labels: list[schemas.Label],
    number_of_samples: int,
    offsets: list[int],
) -> str:
    model_name = "uniform"

    embeddings = []
    for offset in offsets:
        embeddings.append([])
        for _ in range(number_of_samples):
            x = np.random.uniform(size=(10,))
            x[0] += offset
            embeddings[-1].append(x.tolist())

    dataset = Dataset.get(dataset_name)
    assert dataset
    model = Model.create(model_name)
    model.add_predictions(
        dataset=dataset,
        predictions=[
            Prediction(
                datum=schemas.Datum(
                    uid=str(label_idx * number_of_samples + idx)
                ),
                annotations=[
                    schemas.Annotation(embedding=embeddings[label_idx][idx])
                ],
            )
            for label_idx, label in enumerate(labels)
            for idx in range(number_of_samples)
        ],
    )
    return model_name


def test_embedding_metrics_on_fixed_points(
    db: Session,
    dataset: str,
    fixed_model: str,
    labels: list[schemas.Label],
    offsets: list[int],
    number_of_samples: int,
):
    limit = number_of_samples
    dataset_row = core.fetch_dataset(db=db, name=dataset)
    model_row = core.fetch_model(db=db, name=fixed_model)

    label_rows = [core.fetch_label(db=db, label=label) for label in labels]
    assert len(label_rows) == 4
    assert label_rows[0].value == "A"
    assert label_rows[1].value == "B"
    assert label_rows[2].value == "C"
    assert label_rows[3].value == "D"

    reference_dist = _compute_self_distances(
        db=db,
        model=model_row,
        dataset=dataset_row,
        label=label_rows[0],
        limit=limit,
    )
    queries_dist = [
        _compute_distances(
            db=db,
            model=model_row,
            query_dataset=dataset_row,
            reference_dataset=dataset_row,
            ref_label=label_rows[0],
            query_label=label_rows[idx],
            limit=limit,
        )
        for idx in range(1, len(label_rows))
    ]
    cvm_test_values = [
        cramervonmises_2samp(reference_dist, qd) for qd in queries_dist
    ]
    ks_test_values = [ks_2samp(reference_dist, qd) for qd in queries_dist]

    # client side

    client_dataset = Dataset.get(dataset)
    client_model = Model.get(fixed_model)

    assert client_dataset
    assert client_model

    job = client_model.evaluate_embeddings(client_dataset, limit=10)
    assert job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    assert len(job.metrics) == 2
    assert {job.metrics[0]["type"], job.metrics[1]["type"]} == {
        MetricType.CramerVonMises,
        MetricType.KolmgorovSmirnov,
    }
    for m in job.metrics:
        if m["type"] == MetricType.CramerVonMises:
            test_values = cvm_test_values
        elif m["type"] == MetricType.KolmgorovSmirnov:
            test_values = ks_test_values
        else:
            raise NotImplementedError

        assert m["value"]["statistics"]["A"]["B"] == test_values[0].statistic
        assert m["value"]["statistics"]["A"]["C"] == test_values[1].statistic
        assert m["value"]["statistics"]["A"]["D"] == test_values[2].statistic

        assert m["value"]["pvalues"]["A"]["B"] == test_values[0].pvalue
        assert m["value"]["pvalues"]["A"]["C"] == test_values[1].pvalue
        assert m["value"]["pvalues"]["A"]["D"] == test_values[2].pvalue
