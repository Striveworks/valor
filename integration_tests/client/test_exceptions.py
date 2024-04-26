import pytest

from valor import Client, Dataset, Datum, GroundTruth, exceptions


def test_datum_exceptions(client: Client, dataset_name: str):
    dset = Dataset.create(dataset_name)
    datum = Datum(uid="uid")
    dset.add_groundtruth(GroundTruth(datum=datum, annotations=[]))

    with pytest.raises(exceptions.DatumAlreadyExistsError):
        dset.add_groundtruth(GroundTruth(datum=datum, annotations=[]))

    with pytest.raises(exceptions.DatumDoesNotExistError):
        client.get_datum(dataset_name, "nonexistent")
