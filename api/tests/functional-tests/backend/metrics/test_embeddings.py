import math

import numpy as np
import pytest
from scipy.stats import cramervonmises_2samp, ks_2samp
from sqlalchemy.orm import Session

from valor_api import crud, schemas
from valor_api.backend import core
from valor_api.backend.metrics.embedding import (
    _compute_distances,
    _compute_metrics,
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
    db: Session,
    dataset_name: str,
    labels: list[schemas.Label],
    number_of_samples: int,
) -> str:
    crud.create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))
    crud.create_groundtruths(
        db=db,
        groundtruths=[
            schemas.GroundTruth(
                dataset_name=dataset_name,
                datum=schemas.Datum(
                    uid=str(label_idx * number_of_samples + idx)
                ),
                annotations=[schemas.Annotation(labels=[label])],
            )
            for label_idx, label in enumerate(labels)
            for idx in range(number_of_samples)
        ],
    )
    crud.finalize(db=db, dataset_name=dataset_name)
    return dataset_name


@pytest.fixture
def fixed_model(
    db: Session,
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

    crud.create_model(db=db, model=schemas.Model(name=model_name))
    crud.create_predictions(
        db=db,
        predictions=[
            schemas.Prediction(
                dataset_name=dataset_name,
                model_name=model_name,
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
    db: Session,
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

    crud.create_model(db=db, model=schemas.Model(name=model_name))
    crud.create_predictions(
        db=db,
        predictions=[
            schemas.Prediction(
                dataset_name=dataset_name,
                model_name=model_name,
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


def test__compute_distances(
    db: Session,
    dataset: str,
    fixed_model: str,
    labels: list[schemas.Label],
    offsets: list[int],
    number_of_samples: int,
):
    limit = number_of_samples
    dataset = core.fetch_dataset(db=db, name=dataset)
    model = core.fetch_model(db=db, name=fixed_model)

    label_rows = [core.fetch_label(db=db, label=label) for label in labels]
    label_rows = [row for row in label_rows if row is not None]
    assert len(label_rows) == 4
    assert label_rows[0].value == "A"
    assert label_rows[1].value == "B"
    assert label_rows[2].value == "C"
    assert label_rows[3].value == "D"

    # show that two combinations are missing as both query
    # and reference are the same.
    distances = _compute_distances(
        db=db,
        model=model,
        query_dataset=dataset,
        reference_dataset=dataset,
        ref_label=label_rows[0],
        query_label=label_rows[0],
        limit=limit,
    )
    assert len(distances) == limit * (limit - 1)
    assert len(set(distances)) == 1
    assert math.isclose(distances[0], offsets[0], rel_tol=1e-6)

    # map query label to different reference label
    for idx in range(1, 4):
        distances = _compute_distances(
            db=db,
            model=model,
            query_dataset=dataset,
            reference_dataset=dataset,
            ref_label=label_rows[0],
            query_label=label_rows[idx],
            limit=limit,
        )
        assert len(distances) == limit * limit
        assert len(set(distances)) == 1
        assert math.isclose(distances[0], offsets[idx], rel_tol=1e-6)


def test__compute_self_distance(
    db: Session,
    dataset: str,
    fixed_model: str,
    labels: list[schemas.Label],
    offsets: list[int],
    number_of_samples: int,
):
    limit = number_of_samples
    dataset = core.fetch_dataset(db=db, name=dataset)
    model = core.fetch_model(db=db, name=fixed_model)

    label_rows = [core.fetch_label(db=db, label=label) for label in labels]
    label_rows = [row for row in label_rows if row is not None]
    assert len(label_rows) == 4
    assert label_rows[0].value == "A"
    assert label_rows[1].value == "B"
    assert label_rows[2].value == "C"
    assert label_rows[3].value == "D"

    for idx in range(4):
        distances = _compute_self_distances(
            db=db,
            dataset=dataset,
            model=model,
            label=label_rows[idx],
            limit=limit // 2,
        )
        assert len(distances) == (limit // 2) ** 2
        assert len(set(distances)) == 1
        assert distances[0] == 0


def test__compute_metrics_on_fixed(
    db: Session,
    dataset: str,
    fixed_model: str,
    labels: list[schemas.Label],
    offsets: list[int],
    number_of_samples: int,
):
    limit = number_of_samples
    dataset = core.fetch_dataset(db=db, name=dataset)
    model = core.fetch_model(db=db, name=fixed_model)

    label_rows = [core.fetch_label(db=db, label=label) for label in labels]
    label_rows = [row for row in label_rows if row is not None]
    assert len(label_rows) == 4
    assert label_rows[0].value == "A"
    assert label_rows[1].value == "B"
    assert label_rows[2].value == "C"
    assert label_rows[3].value == "D"

    reference_dist = _compute_self_distances(
        db=db,
        model=model,
        dataset=dataset,
        label=label_rows[0],
        limit=limit,
    )
    queries_dist = [
        _compute_distances(
            db=db,
            model=model,
            query_dataset=dataset,
            reference_dataset=dataset,
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

    cvm, ks = _compute_metrics(
        db=db,
        query_dataset=dataset,
        reference_dataset=None,
        model=model,
        labels=label_rows,
        limit=limit,
    )

    assert cvm.statistics["A"]["B"] == cvm_test_values[0].statistic
    assert cvm.statistics["A"]["C"] == cvm_test_values[1].statistic
    assert cvm.statistics["A"]["D"] == cvm_test_values[2].statistic

    assert cvm.pvalues["A"]["B"] == cvm_test_values[0].pvalue
    assert cvm.pvalues["A"]["C"] == cvm_test_values[1].pvalue
    assert cvm.pvalues["A"]["D"] == cvm_test_values[2].pvalue

    assert ks.statistics["A"]["B"] == ks_test_values[0].statistic  # type: ignore - typo in scipy
    assert ks.statistics["A"]["C"] == ks_test_values[1].statistic  # type: ignore - typo in scipy
    assert ks.statistics["A"]["D"] == ks_test_values[2].statistic  # type: ignore - typo in scipy

    assert ks.pvalues["A"]["B"] == ks_test_values[0].pvalue  # type: ignore - typo in scipy
    assert ks.pvalues["A"]["C"] == ks_test_values[1].pvalue  # type: ignore - typo in scipy
    assert ks.pvalues["A"]["D"] == ks_test_values[2].pvalue  # type: ignore - typo in scipy
