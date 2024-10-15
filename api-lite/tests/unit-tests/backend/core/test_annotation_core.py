import pytest

from valor_api import enums, schemas
from valor_api.backend import models
from valor_api.backend.core.annotation import (
    create_annotations,
    delete_dataset_annotations,
    delete_model_annotations,
)


def test_malformed_input_create_annotations():

    with pytest.raises(ValueError):
        create_annotations(
            db=None,  # type: ignore - testing
            annotations=[[schemas.Annotation()], [schemas.Annotation()]],
            datum_ids=[1, 2],
            models_=[None],
        )

    with pytest.raises(ValueError):
        create_annotations(
            db=None,  # type: ignore - testing
            annotations=[[schemas.Annotation()]],
            datum_ids=[1, 2],
            models_=[None],
        )

    with pytest.raises(ValueError):
        create_annotations(
            db=None,  # type: ignore - testing
            annotations=[[schemas.Annotation()]],
            datum_ids=[1],
            models_=[None, None],
        )


def test_malformed_input_delete_dataset_annotations():

    for status in enums.TableStatus:
        if status == enums.TableStatus.DELETING:
            continue

        dataset = models.Dataset(
            name="dataset",
            status=status,
        )

        with pytest.raises(RuntimeError):
            delete_dataset_annotations(db=None, dataset=dataset)  # type: ignore - testing


def test_malformed_input_delete_model_annotations():

    for status in enums.ModelStatus:
        if status == enums.ModelStatus.DELETING:
            continue

        model = models.Model(
            name="model",
            status=status,
        )

        with pytest.raises(RuntimeError):
            delete_model_annotations(db=None, model=model)  # type: ignore - testing
