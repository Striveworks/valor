import pytest
from sqlalchemy.orm import Session

from velour_api import crud, schemas
from velour_api.enums import TaskType

dset_name = "dataset1"
model_name1 = "model1"
model_name2 = "model2"
datum_uid = "uid1"
label_key = "class"
gt_label = "dog"
pd_label = "cat"


@pytest.fixture
def dataset_sim(db: Session):
    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(name=dset_name),
    )
    crud.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=schemas.Datum(
                uid=datum_uid,
                dataset=dset_name,
            ),
            annotations=[
                schemas.Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[schemas.Label(key=label_key, value=gt_label)],
                )
            ],
        ),
    )
    crud.finalize(db=db, dataset_name=dset_name)


@pytest.fixture
def model_sim(
    db: Session,
    dataset_sim,
):
    crud.create_model(
        db=db,
        model=schemas.Model(name=model_name1),
    )
    crud.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model=model_name1,
            datum=schemas.Datum(
                uid=datum_uid,
                dataset=dset_name,
            ),
            annotations=[
                schemas.Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(
                            key=label_key, value=pd_label, score=0.9
                        ),
                        schemas.Label(
                            key=label_key, value=gt_label, score=0.1
                        ),
                    ],
                )
            ],
        ),
    )
    crud.finalize(db=db, dataset_name=dset_name, model_name=model_name1)

    crud.create_model(
        db=db,
        model=schemas.Model(name=model_name2),
    )
    crud.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model=model_name2,
            datum=schemas.Datum(
                uid=datum_uid,
                dataset=dset_name,
            ),
            annotations=[
                schemas.Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(
                            key=label_key, value=pd_label, score=0.2
                        ),
                        schemas.Label(
                            key=label_key, value=gt_label, score=0.8
                        ),
                    ],
                )
            ],
        ),
    )
    crud.finalize(db=db, dataset_name=dset_name, model_name=model_name2)


def test_ops_query(
    db: Session,
    model_sim,
):
    pass
