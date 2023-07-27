import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass
class GeoJSON:
    type: str
    coordinates: list = field(default_factory=list)


def _validate_href(v: str):
    if not isinstance(v, str):
        raise TypeError("passed something other than 'str'")
    if not (v.startswith("http://") or v.startswith("https://")):
        raise ValueError("`href` must start with http:// or https://")


@dataclass
class Metadatum:
    key: str
    value: Union[float, str, GeoJSON]

    def __post_init__(self):
        if isinstance(self.value, int):
            self.value = float(self.value)
        if not isinstance(self.key, str):
            raise TypeError("Name parameter should always be of type string.")
        if not isinstance(self.value, float | str | GeoJSON):
            raise NotImplementedError(f"Value {self.value} has unsupported type {type(self.value)}")
        if self.key == "href":
            _validate_href(self.value)
        

