import pytest

from valor import (
    Client,
    Dataset,
    Datum,
    GroundTruth,
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
    with pytest.raises(exceptions.DatasetNotFinalizedError):
        model.evaluate_classification(dset)

    dset.finalize()
    with pytest.raises(exceptions.DatasetFinalizedError):
        dset.add_groundtruth(
            GroundTruth(datum=Datum(uid="uid"), annotations=[])
        )


def test_datum_exceptions(client: Client, dataset_name: str):
    dset = Dataset.create(dataset_name)
    datum = Datum(uid="uid")
    dset.add_groundtruth(GroundTruth(datum=datum, annotations=[]))

    with pytest.raises(exceptions.DatumAlreadyExistsError):
        dset.add_groundtruth(GroundTruth(datum=datum, annotations=[]))

    with pytest.raises(exceptions.DatumDoesNotExistError):
        client.get_datum(dataset_name, "nonexistent")
