from dataclasses import dataclass


@dataclass
class GeoJSON:
    type: str
    coordinates: dict

    def validate(self):
        pass
