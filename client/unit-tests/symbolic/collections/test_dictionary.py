import datetime

import pytest

from valor.schemas import Dictionary, Float, Integer


def test_validate_metadata():
    Dictionary({"test": "test"})
    Dictionary({"test": 1})
    Dictionary({"test": 1.0})

    with pytest.raises(TypeError):
        Dictionary({123: 123})  # type: ignore

    # Check int to float conversion
    assert type(Dictionary({"test": 1})["test"]) is Integer
    assert type(Dictionary({"test": 1.0})["test"]) is Float


def test_init_dictionary_from_builtin_dict():
    metadata = dict()
    metadata["a"] = int(123.4)
    metadata["b"] = float(123.4)
    metadata["c"] = str(123.4)
    metadata["d"] = datetime.datetime.fromisoformat("2023-01-01T12:12:12")
    metadata["e"] = datetime.date.fromisoformat("2023-01-01")
    metadata["f"] = datetime.time.fromisoformat("12:12:12:100000")
    metadata["g"] = datetime.timedelta(days=1)
    assert Dictionary(metadata).get_value() == metadata


def test_dump_metadata_to_json():
    metadata = dict()
    metadata["a"] = int(123.4)
    metadata["b"] = float(123.4)
    metadata["c"] = str(123.4)
    metadata["d"] = datetime.datetime.fromisoformat("2023-01-01T12:12:12")
    metadata["e"] = datetime.date.fromisoformat("2023-01-01")
    metadata["f"] = datetime.time.fromisoformat("12:12:12:100000")
    metadata["g"] = datetime.timedelta(days=1)
    assert Dictionary(metadata).get_value() == metadata

    assert Dictionary(metadata).to_dict() == {
        "type": "dictionary",
        "value": {
            "a": 123,
            "b": 123.4,
            "c": "123.4",
            "d": {"type": "datetime", "value": "2023-01-01T12:12:12"},
            "e": {"type": "date", "value": "2023-01-01"},
            "f": {"type": "time", "value": "12:12:12.100000"},
            "g": {"type": "duration", "value": 86400.0},
        },
    }


def test_dictionary_encoding():
    metadata = dict()
    metadata["a"] = int(123.4)
    metadata["b"] = float(123.4)
    metadata["c"] = str(123.4)
    metadata["d"] = datetime.datetime.fromisoformat("2023-01-01T12:12:12")
    metadata["e"] = datetime.date.fromisoformat("2023-01-01")
    metadata["f"] = datetime.time.fromisoformat("12:12:12:100000")
    metadata["g"] = datetime.timedelta(days=1)

    metadata_json = {
        "a": 123,
        "b": 123.4,
        "c": "123.4",
        "d": {"type": "datetime", "value": "2023-01-01T12:12:12"},
        "e": {"type": "date", "value": "2023-01-01"},
        "f": {"type": "time", "value": "12:12:12.100000"},
        "g": {"type": "duration", "value": 86400.0},
    }

    assert Dictionary(metadata).encode_value() == metadata_json
    assert Dictionary.decode_value(metadata_json).get_value() == metadata
