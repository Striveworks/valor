from dataclasses import dataclass
from typing import Dict, Union


# TODO - Move this somewhere more appropriate
@dataclass
class GeoJSON:
    type: str
    coordinates: dict

    def validate(self):
        pass


def _validate_href(value: str):
    if not isinstance(v, str):
        raise TypeError("`href` is something other than 'str'")
    if not (value.startswith("http://") or value.startswith("https://")):
        raise ValueError("`href` must start with http:// or https://")


def serialize_metadata(metadata: dict) -> list:
    if not metadata:
        return []
    return [
        {
            "key": key,
            "value": metadata[key],
        }
        for key in metadata
    ]


def deserialize_metadata(metadata: list) -> dict:
    if not metadata:
        return {}
    return {
        element["key"]: element["value"]
        for element in metadata
        if (
            isinstance(element, dict)
            and "key" in element
            and "value" in element
        )
    }


def validate_metadata(metadata: Dict[str, Union[int, float, str, GeoJSON]]):
    if metadata is None:
        return {}
    if isinstance(metadata, list):
        metadata = deserialize_metadata(metadata)
    if not isinstance(metadata, dict):
        raise TypeError(
            f"Expected `metadata` to be of type `dict[str, int | float | str | GeoJSON]`, received {metadata}"
        )
    for key in metadata:
        if not isinstance(key, str):
            raise TypeError(
                f"Expected metadata key to be of type `str`, got {type(key)}"
            )
        if not isinstance(metadata[key], Union[int, float, str, GeoJSON]):
            raise TypeError(
                f"Expected metadata value to be of type int, float, str or GeoJSON, received {metadata[key]}"
            )

    # validate specific keys
    if "href" in metadata:
        _validate_href(metadata["href"])

    return metadata
