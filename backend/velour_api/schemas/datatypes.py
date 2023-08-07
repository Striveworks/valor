from pydantic import BaseModel

from velour_api.schemas.core import Datum, MetaDatum


class Image(BaseModel):
    dataset: str
    uid: str
    height: int
    width: int
    metadata: list[MetaDatum] = []

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

        # cache and remove image metadata
        height = metadata["height"]
        width = metadata["width"]
        del metadata["height"]
        del metadata["width"]

        return cls(
            uid=datum.uid,
            dataset=datum.dataset,
            height=height,
            width=width,
            metadata=[
                MetaDatum(key=key, value=metadata[key]) for key in metadata
            ],
        )

    def to_datum(self) -> Datum:
        self.metadata.append(MetaDatum(key="height", value=self.height))
        self.metadata.append(MetaDatum(key="width", value=self.width))
        return Datum(
            uid=self.uid,
            dataset=self.dataset,
            metadata=self.metadata,
        )


class VideoFrame(BaseModel):
    image: Image
    frame: int | None = None

    @classmethod
    def from_datum(cls, datum: Datum):
        if not isinstance(datum, Datum):
            raise TypeError("Expecting `velour.schemas.Datum`")

        # Extract image
        image = Image.from_datum(datum)

        # Extract Video frame number
        metadata = {
            metadatum.key: metadatum.value for metadatum in image.metadata
        }
        if "frame" not in metadata:
            raise ValueError(
                "Datum does not contain all the information that makes it an image."
            )
        frame = int(metadata["frame"])
        del metadata["frame"]
        image.metadata = [
            MetaDatum(key=key, value=metadata[key]) for key in metadata
        ]

        return cls(
            image=image,
            frame=frame,
        )

    def to_datum(self) -> Datum:
        datum = self.image.to_datum()
        datum.metadata.append(MetaDatum(key="frame", value=self.frame))
        return datum
