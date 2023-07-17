import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass
class GeographicFeature:
    region: dict

    def __post_init__(self):
        if isinstance(self.region, dict):
            # check that the dict is JSON serializable
            try:
                json.dumps(self.region)
            except TypeError:
                raise ValueError(
                    f"if a dict, `region` must be valid GeoJSON but got {self.region}"
                )


def _validate_href(v: str):
    if not (v.startswith("http://") or v.startswith("https://")):
        raise ValueError("`href` must start with http:// or https://")


@dataclass
class Metadatum:
    name: str
    value: Union[float, str, GeographicFeature]

    def __post_init__(self):
        if self.name == "href":
            if isinstance(self.value, str):
                _validate_href(self.value)
