import json
from dataclasses import dataclass, field, asdict
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


@dataclass
class Metadatum:
    name: str
    value: Union[float, str, GeographicFeature]
