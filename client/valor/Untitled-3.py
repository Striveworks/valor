import datetime
from typing import Optional, Any

from valor.symbolic import (
    Variable,
    Geometry,
    String,
    Metadata,
    Duration,
    Time,
    Where,
    jsonify,
)


#### Example implementation


class Box(Geometry):

    @classmethod
    def from_extrema(cls, xmin, xmax, ymin, ymax):
        geom = {
            "type": "Polygon",
            "coordinates": [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ],
        }
        return cls(value=geom)

    @property
    def points(self):
        if self.is_symbolic():
            raise ValueError()
        return self.value


class Annotation:

    box: Box = Box(name="bounding_box")

    def __init__(
        self,
        box: Optional[Box] = None,
    ):
        if box:
            self.box = box

    def to_dict(self):
        return {
            "box": self.box.to_dict() if self.box.is_value() else None,
        }


class Datum:
    uid = String(name="datum_uid")
    metadata = Metadata(name="datum_metadata")

    def __init__(
        self,
        uid: str,
        metadata: Optional[dict] = None,
    ):
        self.uid = String(value=uid)
        self.metadata = Metadata(value=metadata)

    @staticmethod
    def where(conditions):
        return Where(Variable(name="datum"), conditions)

a = Duration(name="some_duration")


cond = (
    (Datum.metadata["some_time"] < datetime.time(hour=6))
    | (Datum.metadata["some_time"] > datetime.time(hour=20))
)
print(cond)


cond = Annotation.box.area > 9

print(cond)


cond = Annotation.box.intersects(Box.from_extrema(0,1,0,1))
print(cond)

packet = jsonify(cond)

import json
print(json.dumps(packet, indent=2))