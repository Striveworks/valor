from pydantic import BaseModel

from velour_api.schemas.core import Datum, MetaDatum


class Image(BaseModel):
    uid: str
    height: int
    width: int

    @classmethod
    def from_datum(cls, datum: Datum):
        if not isinstance(datum, Datum):
            raise TypeError("Expecting `velour.schemas.Datum`")

        metadata = {
            metadatum.key: metadatum.value for metadatum in datum.metadata
        }
        try:
            assert "height" in metadata
            assert "width" in metadata
        except AssertionError:
            raise ValueError(
                "Datum does not contain all the information that makes it an image."
            )

        return cls(
            uid=datum.uid,
            height=metadata["height"],
            width=metadata["width"],
        )

    def to_datum(self) -> Datum:
        return Datum(
            uid=self.uid,
            metadata=[
                MetaDatum(key="height", value=self.height),
                MetaDatum(key="width", value=self.width),
            ],
        )


class VideoFrame(BaseModel):
    image: Image
    frame: int | None = None

    @classmethod
    def from_datum(cls, datum: Datum):
        if not isinstance(datum, Datum):
            raise TypeError("Expecting `velour.schemas.Datum`")

        # Video frame number
        metadata = {
            metadatum.key: metadatum.value for metadatum in datum.metadata
        }
        try:
            assert "frame" in metadata
        except AssertionError:
            raise ValueError(
                "Datum does not contain all the information that makes it an image."
            )

        # Image
        image = Image.from_datum(datum)

        return cls(
            image=image,
            frame=int(metadata["frame"]),
        )

    def to_datum(self) -> Datum:
        return Datum(
            uid=self.image.uid,
            metadata=[
                MetaDatum(key="height", value=self.image.height),
                MetaDatum(key="width", value=self.image.width),
                MetaDatum(key="frame", value=self.frame),
            ],
        )
