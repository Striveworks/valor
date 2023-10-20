import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from velour_api import crud, schemas
from velour_api.backend import graph, models
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
def model_sim(db: Session):
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
                            key=label_key, value=pd_label, score=0.1
                        ),
                        schemas.Label(
                            key=label_key, value=gt_label, score=0.9
                        ),
                    ],
                )
            ],
        ),
    )
    crud.finalize(db=db, dataset_name=dset_name, model_name=model_name2)


# def _test_extreme_generate_query(target: graph.Node):
#     filters = {
#         graph.dataset: [models.Dataset.name == "dataset1"],
#         graph.model: [models.Model.name == "model1"],
#         graph.groundtruth_label: [models.Label.value == "dog"],
#         graph.prediction_label: [models.Label.value == "cat"],
#     }
#     generated_query = graph.generate_query(target, filters)

#     match target:
#         case graph.model:
#             expected_query = (
#                 select(models.Model.id)
#                 .join(models.Annotation, models.Annotation.model_id == models.Model.id)
#                 .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
#                 .join(models.Prediction, models.Prediction.annotation_id == models.Annotation.id)
#                 .join(models.Label, models.Label.id == models.Prediction.label_id)
#                 .join(models.Dataset, models.Dataset.id == models.Datum.dataset_id)
#                 .where(
#                     models.Dataset.name == "dataset1",
#                     models.Label.value == "cat",
#                     models.Model.name == "model1",
#                     models.Datum.id.in_(
#                         select(models.Datum.id)
#                         .join(models.Annotation, models.Annotation.datum_id == models.Datum.id)
#                         .join(models.GroundTruth, models.GroundTruth.annotation_id == models.Annotation.id)
#                         .join(models.Label, models.Label.id == models.GroundTruth.label_id)
#                         .where(models.Label.value == "dog")
#                     )
#                 )
#             )
#         case graph.dataset:
#             expected_query = (
#                 select(models.Dataset.id)
#                 .join(models.Datum, models.Datum.dataset_id == models.Dataset.id)
#                 .join(models.Annotation, models.Annotation.datum_id == models.Datum.id)
#                 .join(models.Model, models.Model.id == models.Annotation.model_id)
#                 .join(models.Prediction, models.Prediction.annotation_id == models.Annotation.id)
#                 .join(models.Label, models.Label.id == models.Prediction.label_id)
#                 .where(
#                     models.Model.name == "model1",
#                     models.Label.value == "cat",
#                     models.Dataset.name == "dataset1",
#                     models.Datum.id.in_(
#                         select(models.Datum.id)
#                         .join(models.Annotation, models.Annotation.datum_id == models.Datum.id)
#                         .join(models.GroundTruth, models.GroundTruth.annotation_id == models.Annotation.id)
#                         .join(models.Label, models.Label.id == models.GroundTruth.label_id)
#                         .where(models.Label.value == "dog")
#                     )
#                 )
#             )

#     return generated_query, expected_query


# # WHERE dataset.name = :name_1 AND label.value = :value_1 AND model.name = :name_2 AND datum.id IN (
# #     SELECT anon_1.id
# #     FROM (
# #         SELECT datum.id AS id
# #         FROM datum
# #         JOIN annotation ON annotation.datum_id = datum.id
# #         JOIN groundtruth ON groundtruth.annotation_id = annotation.id
# #         JOIN label ON label.id = groundtruth.label_id
# #         WHERE label.value = :value_2
# #     )
# # AS anon_1)


def test_generate_query(
    db: Session,
    dataset_sim,
    model_sim,
):
    # Q: Get model ids for all models that operate over a dataset that meets the name equality
    target = graph.model
    filters = {graph.dataset: [models.Dataset.name == dset_name]}
    generated_query = graph.generate_query(target, filters)
    model_ids = db.scalars(generated_query).all()
    assert len(model_ids) == 2
    model_names = [
        db.scalar(select(models.Model).where(models.Model.id == id)).name
        for id in model_ids
    ]
    assert model_name1 in model_names
    assert model_name2 in model_names


def test_generate_query_extremities():

    pass

    # # extreme request (models)
    # generated_query, expected_query = _test_extreme_generate_query(graph.model)
    # assert str(generated_query.compile()) == str(expected_query.compile())

    # # extreme request (datasets)
    # generated_query, expected_query = _test_extreme_generate_query(graph.dataset)
    # assert str(generated_query) == str(expected_query)
