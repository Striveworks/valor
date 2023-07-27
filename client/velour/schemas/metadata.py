import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass
class GeoJSON:
    type: str
    coordinates: list = field(default_factory=list)



