import numpy as np
import pytest
from sqlalchemy.orm import Session

from valor_api import crud, enums, schemas
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
    return 100


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
) -> str:
    model_name = "fixed"

    embeddings = []
    for offset in [0, 0.1, 0.5, 2.0]:
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
) -> str:
    model_name = "uniform"

    embeddings = []
    for offset in [0, 0.1, 0.5, 2.0]:
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
):

    dataset = core.fetch_dataset(db=db, name=dataset)
    model = core.fetch_model(db=db, name=fixed_model)
    labels = list(core.fetch_labels(db=db, filters=schemas.Filter()))

    print(labels[0].value)
    print(labels[1].value)

    distances = _compute_distances(
        db=db,
        model=model,
        query_dataset=dataset,
        reference_dataset=dataset,
        ref_label=labels[0],
        query_label=labels[1],
        limit=-1,
    )
    print(len(distances))
