import pytest
from sqlalchemy.orm import Session

from valor_api import enums, schemas
from valor_api.backend.core import fetch_dataset
from valor_api.backend.core.geometry import (
    convert_geometry,
    get_annotation_type,
)
from valor_api.crud import create_dataset, create_groundtruth


@pytest.fixture
def create_clf_dataset(db: Session, dataset_name: str):
    create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))
    create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=schemas.Datum(uid="uid1", dataset_name=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )


def test_get_annotation_type(
    db: Session, dataset_name: str, create_clf_dataset
):
    # tests uncovered case where `AnnotationType.NONE` is returned.
    dataset = fetch_dataset(db, dataset_name)
    assert (
        get_annotation_type(db, enums.TaskType.CLASSIFICATION, dataset)
        == enums.AnnotationType.NONE
    )


def test_convert_geometry(
    db: Session, dataset_name: str, dataset_model_create
):
    dataset = fetch_dataset(db, dataset_name)

    with pytest.raises(ValueError) as e:
        convert_geometry(
            db=db,
            source_type=enums.AnnotationType.NONE,
            target_type=enums.AnnotationType.BOX,
            dataset=None,
            model=None,
        )
    assert "Source type" in str(e)

    with pytest.raises(ValueError) as e:
        convert_geometry(
            db=db,
            source_type=enums.AnnotationType.BOX,
            target_type=enums.AnnotationType.NONE,
            dataset=None,
            model=None,
        )
    assert "Target type" in str(e)

    with pytest.raises(ValueError) as e:
        convert_geometry(
            db=db,
            source_type=enums.AnnotationType.BOX,
            target_type=enums.AnnotationType.RASTER,
            dataset=None,
            model=None,
        )
    assert "not capable of being converted" in str(e)

    with pytest.raises(NotImplementedError) as e:
        convert_geometry(
            db=db,
            source_type=enums.AnnotationType.MULTIPOLYGON,
            target_type=enums.AnnotationType.BOX,
            dataset=dataset,
            model=None,
        )
    assert "currently unsupported" in str(e)

    with pytest.raises(NotImplementedError) as e:
        convert_geometry(
            db=db,
            source_type=enums.AnnotationType.MULTIPOLYGON,
            target_type=enums.AnnotationType.POLYGON,
            dataset=dataset,
            model=None,
        )
    assert "currently unsupported" in str(e)
