import pytest
import datetime

from velour.exceptions import SchemaTypeError
from velour.schemas.metadata import (
    _validate_href,
    validate_metadata,
    dump_metadata,
    load_metadata,
)


def test__validate_href():
    _validate_href("http://test")
    _validate_href("https://test")

    with pytest.raises(ValueError) as e:
        _validate_href("test")
    assert "`href` must start with http:// or https://" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        _validate_href(1)
    assert "`href` should be of type" in str(e)


def test_validate_metadata():
    validate_metadata({"test": "test"})
    validate_metadata({"test": 1})
    validate_metadata({"test": 1.0})

    with pytest.raises(SchemaTypeError) as e:
        validate_metadata({123: 123})
    assert "`metadatum key` should be of type" in str(e)

    # Test supported value types
    with pytest.raises(SchemaTypeError):
        validate_metadata({"test": (1, 2)})
    assert "`metadatum value` should be of type"
    with pytest.raises(SchemaTypeError):
        validate_metadata({"test": [1, 2]})
    assert "`metadatum value` should be of type"

    # Test special type with name=href
    validate_metadata({"href": "http://test"})
    validate_metadata({"href": "https://test"})
    with pytest.raises(ValueError) as e:
        validate_metadata({"href": "test"})
    assert "`href` must start with http:// or https://" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        validate_metadata({"href": 1})
    assert "`href` should be of type" in str(e)

    # Check int to float conversion
    validate_metadata({"test": 1})


def test_dump_metadata():

    # should not be changed as only uses str, int, float values
    metadata = {
        "a": int(123.4),
        "b": float(123.4),
        "c": str(123.4),
    }
    assert metadata == dump_metadata(metadata)

    # check datetime object conversion
    metadata["d"] = datetime.datetime.fromisoformat("2023-01-01T12:12:12")
    metadata["e"] = datetime.date.fromisoformat("2023-01-01")
    metadata["f"] = datetime.time.fromisoformat("12:12:12:100000")
    metadata["g"] = datetime.timedelta(days=1)
    assert dump_metadata(metadata) == {
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
    assert metadata == load_metadata(metadata)

    # check datetime object conversion
    metadata["d"] = {"datetime": "2023-01-01T12:12:12"}
    metadata["e"] = {"date": "2023-01-01"}
    metadata["f"] = {"time": "12:12:12.100000"}
    metadata["g"] = {"duration": "86400.0"}
    assert load_metadata(metadata) == {
        "a": int(123.4),
        "b": float(123.4),
        "c": str(123.4),
        "d": datetime.datetime.fromisoformat("2023-01-01T12:12:12"),
        "e": datetime.date.fromisoformat("2023-01-01"),
        "f": datetime.time.fromisoformat("12:12:12:100000"),
        "g": datetime.timedelta(days=1),
    }