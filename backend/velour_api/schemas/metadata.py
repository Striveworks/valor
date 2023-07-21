from pydantic import BaseModel, validator

from velour_api import enums
from velour_api.schemas.core import MetaDatum, Datum


class Image(BaseModel):
    uid: str
    height: int
    width: int
    frame: int

    @classmethod
    def from_datum(cls, datum: Datum):
        if not isinstance(datum, Datum):
            raise TypeError("Expecting `velour.schemas.Datum`")
        metadata = {metadatum.name: metadatum.value for metadatum in datum.metadata}
        try:
            assert "type" in metadata
            assert "height" in metadata
            assert "width" in metadata
            assert "frame" in metadata
            assert metadata["type"] == enums.DataType.IMAGE
        except:
            raise ValueError("Datum does not contain all the information that makes it an image.")
        
        return cls(
            uid=datum.uid,
            height=metadata["height"],
            width=metadata["width"],
            frame=metadata["frame"],
        )
    
    def to_datum(self) -> Datum:
        return Datum(
            uid=self.uid,
            metadata=[
                MetaDatum(name="type", value="image"),
                MetaDatum(name="height", value=self.height),
                MetaDatum(name="width", value=self.width),
                MetaDatum(name="frame", value=self.frame),
            ]
        )