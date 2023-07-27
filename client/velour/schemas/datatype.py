from dataclasses import dataclass, field
from typing import Optional

from velour import schemas


@dataclass
class Image:
    uid: str
    height: int
    width: int
    frame: int = field(default=0)

    def __post_init__(self):
        if not isinstance(self.uid, str):
            raise TypeError("Image uid must be a string.")
        if not isinstance(self.height, int):
            raise TypeError("Image height must be a int.")
        if not isinstance(self.width, int):
            raise TypeError("Image height must be a int.")
        if self.frame:
            if not isinstance(self.frame, int):
                raise TypeError("Image frame must be a int or None.")
            
    @staticmethod
    def valid(datum: schemas.Datum) -> bool:
        metadata = {
            metadatum.key: metadatum.value
            for metadatum in datum.metadata
        }
        if "height" not in metadata:
            return False
        if "width" not in metadata:
            return False
        if "frame" not in metadata:
            return False
        return True
            
    @classmethod
    def from_datum(cls, datum: schemas.Datum):
        if not cls.valid(datum):
            raise TypeError("Datum does not conform to image type.")

        metadata = {
            metadatum.key: metadatum.value
            for metadatum in datum.metadata
        }
        return cls(
            uid=datum.uid,
            height=int(metadata["height"]),
            width=int(metadata["width"]),
            frame=int(metadata["frame"]),
        )
        
    def to_datum(self) -> schemas.Datum:
        return schemas.Datum(
            uid=self.uid,
            metadata=[
                schemas.Metadatum(key="height", value=self.height),
                schemas.Metadatum(key="width", value=self.width),
                schemas.Metadatum(key="frame", value=self.frame),
            ],
        )
    
    