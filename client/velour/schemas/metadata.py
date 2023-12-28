from dataclasses import dataclass, asdict
from typing import Dict, Union
from copy import deepcopy

from velour.exceptions import SchemaTypeError


@dataclass
class DateTime:
    """
    An object describing a date and time.
    
    https://www.postgresql.org/docs/current/functions-formatting.html 

    Examples
    --------
    >>> DateTime(
    ...     value="2024",
    ...     pattern="YYYY"
    ... )
    """
    value: str
    pattern: str

    def __post_init__(self):
        if not isinstance(self.value, str):
            raise TypeError("DateTime value should be a string.")
        if not isinstance(self.pattern, str):
            raise TypeError("DateTime pattern should be a string.")


def _validate_href(value: str):
    if not isinstance(value, str):
        raise SchemaTypeError("href", str, value)
    if not (value.startswith("http://") or value.startswith("https://")):
        raise ValueError("`href` must start with http:// or https://")


def validate_metadata(metadata: dict):
    """Validates metadata dictionary."""
    if not isinstance(metadata, dict):
        raise SchemaTypeError(
            "metadata", Dict[str, Union[float, int, str]], metadata
        )
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise SchemaTypeError("metadatum key", str, key)
        if not (
            isinstance(value, int)
            or isinstance(value, float)
            or isinstance(value, str)
            or isinstance(value, DateTime)
        ):
            raise SchemaTypeError(
                "metadatum value", Union[float, int, str], value
            )

        # Handle special key-values
        if key == "href":
            _validate_href(value)

def dump_metadata(metadata: dict) -> dict:
    """Ensures that all nested attributes are numerics or str types."""
    metadata = deepcopy(metadata)
    for key, value in metadata.items():
        if isinstance(value, DateTime):
            metadata[key] = asdict(value)
    return metadata
