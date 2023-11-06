from dataclasses import dataclass


# TODO Placeholder
@dataclass
class GeoJSON:
    type: str
    coordinates: dict

    def validate(self):
        pass
