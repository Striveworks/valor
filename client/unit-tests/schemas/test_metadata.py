import datetime

import pytest

from valor.schemas.symbolic.structures import Dictionary


def test_validate_metadata():
    Dictionary({"test": "test"})
    Dictionary({"test": 1})
    Dictionary({"test": 1.0})

    with pytest.raises(TypeError):
        Dictionary({123: 123})  # type: ignore

    # Test supported value types
    with pytest.raises(TypeError):
        Dictionary({"test": (1, 2)})  # type: ignore
    with pytest.raises(TypeError):
        Dictionary({"test": [1, 2]})  # type: ignore

    # Check int to float conversion
    Dictionary({"test": 1})


def test_dump_metadata():

    # should not be changed as only uses str, int, float values
    metadata = {
        "a": int(123.4),
        "b": float(123.4),
        "c": str(123.4),
    }
    assert metadata == Dictionary(metadata).to_dict()

    # check datetime object conversion
    metadata["d"] = datetime.datetime.fromisoformat("2023-01-01T12:12:12")
    metadata["e"] = datetime.date.fromisoformat("2023-01-01")
    metadata["f"] = datetime.time.fromisoformat("12:12:12:100000")
    metadata["g"] = datetime.timedelta(days=1)
    assert Dictionary(metadata).to_dict() == {
        "a": int(123.4),
        "b": float(123.4),
        "c": str(123.4),
        "d": {"datetime": "2023-01-01T12:12:12"},
        "e": {"date": "2023-01-01"},
        "f": {"time": "12:12:12.100000"},
        "g": {"duration": "86400.0"},
    }


def test_load_metadata():
    # should not be changed as only uses str, int, float values
    metadata = {
        "a": int(123.4),
        "b": float(123.4),
        "c": str(123.4),
    }
    assert metadata == Dictionary(metadata)

    # check datetime object conversion
    metadata["d"] = {"datetime": "2023-01-01T12:12:12"}
    metadata["e"] = {"date": "2023-01-01"}
    metadata["f"] = {"time": "12:12:12.100000"}
    metadata["g"] = {"duration": "86400.0"}
    assert Dictionary(metadata) == {
        "a": int(123.4),
        "b": float(123.4),
        "c": str(123.4),
        "d": datetime.datetime.fromisoformat("2023-01-01T12:12:12"),
        "e": datetime.date.fromisoformat("2023-01-01"),
        "f": datetime.time.fromisoformat("12:12:12:100000"),
        "g": datetime.timedelta(days=1),
    }
