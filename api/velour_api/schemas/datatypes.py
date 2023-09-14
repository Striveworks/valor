from pydantic import BaseModel

from velour_api.schemas.core import Datum
from velour_api.schemas.metadata import Metadatum


class ImageMetadata(BaseModel):
    dataset: str
    uid: str
    height: int
    width: int
    metadata: list[Metadatum] = []

    @classmethod
    def fromDatum(cls, datum: Datum):
        if not isinstance(datum, Datum):
            raise TypeError("Expecting `velour.schemas.Datum`")

        metadata = {
            metadatum.key: metadatum.value for metadatum in datum.metadata
        }
        if "height" not in metadata:
            raise ValueError("missing height")
        elif "width" not in metadata:
            raise ValueError("missing width")

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
                Metadatum(key=key, value=metadata[key]) for key in metadata
            ],
        )

    def toDatum(self) -> Datum:
        self.metadata.append(Metadatum(key="height", value=self.height))
        self.metadata.append(Metadatum(key="width", value=self.width))
        return Datum(
            uid=self.uid,
            dataset=self.dataset,
            metadata=self.metadata,
        )


class VideoFrameMetadata(BaseModel):
    image: ImageMetadata
    frame: int | None = None

    @classmethod
    def fromDatum(cls, datum: Datum):
        if not isinstance(datum, Datum):
            raise TypeError("Expecting `velour.schemas.Datum`")

        # Extract image
        image = ImageMetadata.from_datum(datum)

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
            Metadatum(key=key, value=metadata[key]) for key in metadata
        ]

        return cls(
            image=image,
            frame=frame,
        )

    def toDatum(self) -> Datum:
        datum = self.image.to_datum()
        datum.metadata.append(Metadatum(key="frame", value=self.frame))
        return datum
