from pydantic import BaseModel

from velour_api import enums
from velour_api.schemas.core import Datum, MetaDatum


class Image(BaseModel):
    uid: str
    height: int
    width: int
    frame: int = None

    @classmethod
    def from_datum(cls, datum: Datum):
        if not isinstance(datum, Datum):
            raise TypeError("Expecting `velour.schemas.Datum`")
        metadata = {
            metadatum.key: metadatum.value for metadatum in datum.metadata
        }
        try:
            assert "type" in metadata
            assert "height" in metadata
            assert "width" in metadata
            assert "frame" in metadata
            assert metadata["type"] == enums.DataType.IMAGE
        except AssertionError:
            raise ValueError(
                "Datum does not contain all the information that makes it an image."
            )

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
                MetaDatum(key="type", value="image"),
                MetaDatum(key="height", value=self.height),
                MetaDatum(key="width", value=self.width),
                MetaDatum(key="frame", value=self.frame),
            ],
        )
