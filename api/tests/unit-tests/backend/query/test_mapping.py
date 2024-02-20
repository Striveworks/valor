import pytest

from valor_api.backend import models
from valor_api.backend.query.mapping import _map_name_to_table


def test__map_name_to_table():
    assert _map_name_to_table(models.Dataset.__tablename__) == models.Dataset
    assert _map_name_to_table(models.Model.__tablename__) == models.Model
    assert _map_name_to_table(models.Datum.__tablename__) == models.Datum
    assert (
        _map_name_to_table(models.Annotation.__tablename__)
        == models.Annotation
    )
    assert (
        _map_name_to_table(models.GroundTruth.__tablename__)
        == models.GroundTruth
    )
    assert (
        _map_name_to_table(models.Prediction.__tablename__)
        == models.Prediction
    )
    assert _map_name_to_table(models.Label.__tablename__) == models.Label

    with pytest.raises(ValueError):
        _map_name_to_table("random_str")
