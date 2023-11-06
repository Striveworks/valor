import pytest

from velour import schemas
from velour.exceptions import SchemaTypeError
from velour.schemas.metadata import _validate_href, validate_metadata


def test__validate_href():
    _validate_href("http://test")
    _validate_href("https://test")

    with pytest.raises(ValueError) as e:
        _validate_href("test")
    assert "`href` must start with http:// or https://" in str(e)
    with pytest.raises(SchemaTypeError) as e:
        _validate_href(1)
    assert "`href` should be of type" in str(e)


def test_validate_metadata(metadata):
    validate_metadata({"test": "test"})
    validate_metadata({"test": 1})
    validate_metadata({"test": 1.0})
    # @TODO: Fix when geojson is implemented
    validate_metadata({"test": schemas.GeoJSON(type="test", coordinates=[])})

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


def test_metadata_geojson():
    # @TODO: Implement GeoJSON
    schemas.GeoJSON(type="this shouldnt work", coordinates=[])
