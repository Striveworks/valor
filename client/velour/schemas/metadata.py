from dataclasses import dataclass
from typing import Dict, Union

from velour.exceptions import SchemaTypeError


# TODO - Move this somewhere more appropriate
@dataclass
class GeoJSON:
    type: str
    coordinates: dict

    def validate(self):
        pass


@dataclass
class _BaseMetadatum:
    key: str
    value: Union[float, int, str, GeoJSON]


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
        if not isinstance(value, Union[float, int, str, GeoJSON]):
            raise SchemaTypeError(
                "metadatum value", Union[float, int, str, GeoJSON], value
            )

        # Handle special key-values
        if key == "href":
            _validate_href(value)
