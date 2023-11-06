from typing import Dict, Union

from velour.exceptions import SchemaTypeError
from velour.schemas.geospatial import GeoJSON


def _validate_href(value: str):
    if not isinstance(value, str):
        raise SchemaTypeError("href", str, value)
    if not (value.startswith("http://") or value.startswith("https://")):
        raise ValueError("`href` must start with http:// or https://")


def validate_metadata(metadata):
    if not isinstance(metadata, dict):
        raise SchemaTypeError(
            "metadata", Dict[str, Union[float, int, str, GeoJSON]], metadata
        )
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise SchemaTypeError("metadatum key", str, key)
        if not (
            isinstance(value, int)
            or isinstance(value, float)
            or isinstance(value, str)
            or isinstance(value, GeoJSON)
        ):
            raise SchemaTypeError(
                "metadatum value", Union[float, int, str, GeoJSON], value
            )

        # Handle special key-values
        if key == "href":
            _validate_href(value)
