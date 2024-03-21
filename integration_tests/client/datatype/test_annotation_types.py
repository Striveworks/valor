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
from valor.enums import TaskType


def test_create_read_embedding_annotation(
    client: Client, dataset_name: str, model_name: str
):
    dataset = Dataset.create(name=dataset_name)
    dataset.add_groundtruth(
        GroundTruth(
            datum=Datum(uid="uid123"),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=[Label(key="k1", value="v1")],
                )
            ],
        )
    )

    model = Model.create(name=model_name)
    model.add_prediction(
        dataset=dataset,
        prediction=Prediction(
            datum=Datum(uid="uid123"),
            annotations=[
                Annotation(
                    task_type=TaskType.EMBEDDING,
                    embedding=[1, 2, 3, 4, 5],
                )
            ],
        ),
    )

    predictions = model.get_prediction(dataset=dataset, datum="uid123")
    assert predictions
    assert predictions.annotations[0].embedding == [1, 2, 3, 4, 5]
