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
    exceptions,
)


def test_dataset_exceptions(
    client: Client, dataset_name: str, model_name: str
):
    # test `DatasetDoesNotExistError`
    with pytest.raises(exceptions.DatasetDoesNotExistError):
        client.get_dataset("nonexistent")

    # test `DatasetAlreadyExistsError`
    dset = Dataset.create(dataset_name)

    with pytest.raises(exceptions.DatasetAlreadyExistsError):
        Dataset.create(dataset_name)

    # test `DatasetNotFinalizedError`
    model = Model.create(model_name)
    dset.add_groundtruth(GroundTruth(datum=Datum(uid="uid"), annotations=[]))
    model.add_prediction(
        dset, Prediction(datum=Datum(uid="uid"), annotations=[])
    )
    with pytest.raises(exceptions.EvaluationRequestError) as e:
        model.evaluate_classification(dset)
    assert "Failed request validation" in str(e)
    assert "DatasetNotFinalizedError" in str(e)
    exc_info = e._excinfo
    assert exc_info
    assert len(exc_info[1].errors) == 2

    assert exc_info[1].errors[0]["name"] == "DatasetNotFinalizedError"
    assert dataset_name in exc_info[1].errors[0]["detail"]
    assert model_name not in exc_info[1].errors[0]["detail"]

    assert exc_info[1].errors[1]["name"] == "ModelNotFinalizedError"
    assert dataset_name in exc_info[1].errors[1]["detail"]
    assert model_name in exc_info[1].errors[1]["detail"]

    dset.finalize()
    with pytest.raises(exceptions.DatasetFinalizedError):
        dset.add_groundtruth(
            GroundTruth(datum=Datum(uid="uid"), annotations=[])
        )


def test_datum_exceptions(client: Client, dataset_name: str):
    dset = Dataset.create(dataset_name)
    datum = Datum(uid="uid")
    dset.add_groundtruth(GroundTruth(datum=datum, annotations=[]))

    with pytest.raises(exceptions.DatumsAlreadyExistError):
        dset.add_groundtruth(GroundTruth(datum=datum, annotations=[]))

    with pytest.raises(exceptions.DatumDoesNotExistError):
        client.get_datum(dataset_name, "nonexistent")


def test_model_exceptions(client: Client, model_name: str, dataset_name: str):
    # test `ModelDoesNotExistError`
    with pytest.raises(exceptions.ModelDoesNotExistError):
        client.get_model("nonexistent")

    # test `ModelAlreadyExistsError`
    model = Model.create(model_name)
    with pytest.raises(exceptions.ModelAlreadyExistsError):
        Model.create(model_name)

    # test `ModelNotFinalizedError`
    dset = Dataset.create(dataset_name)
    dset.add_groundtruth(GroundTruth(datum=Datum(uid="uid"), annotations=[]))
    dset.finalize()
    with pytest.raises(exceptions.EvaluationRequestError) as e:
        model.evaluate_classification(dset)
    assert "Failed request validation" in str(e)
    assert "ModelNotFinalizedError" in str(e)
    exc_info = e._excinfo
    assert exc_info
    assert len(exc_info[1].errors) == 1
    assert exc_info[1].errors[0]["name"] == "ModelNotFinalizedError"
    assert dataset_name in exc_info[1].errors[0]["detail"]
    assert model_name in exc_info[1].errors[0]["detail"]

    # test `ModelFinalizedError`
    model.finalize_inferences(dset)
    with pytest.raises(exceptions.ModelFinalizedError):
        model.add_prediction(
            dset,
            Prediction(datum=Datum(uid="uid"), annotations=[]),
        )


def test_annotation_exceptions(
    client: Client, model_name: str, dataset_name: str
):
    model = Model.create(model_name)
    dset = Dataset.create(dataset_name)

    dset.add_groundtruth(GroundTruth(datum=Datum(uid="uid"), annotations=[]))
    model.add_prediction(
        dset,
        Prediction(
            datum=Datum(uid="uid"),
            annotations=[
                Annotation(
                    labels=[Label(key="key", value="value", score=1.0)],
                )
            ],
        ),
    )

    with pytest.raises(exceptions.PredictionAlreadyExistsError):
        model.add_prediction(
            dset,
            Prediction(datum=Datum(uid="uid"), annotations=[]),
        )


def test_prediction_exceptions(
    client: Client, model_name: str, dataset_name: str
):
    model = Model.create(model_name)
    dset = Dataset.create(dataset_name)
    dset.add_groundtruth(GroundTruth(datum=Datum(uid="uid"), annotations=[]))
    with pytest.raises(exceptions.PredictionDoesNotExistError):
        model.get_prediction(dset, "uid")
