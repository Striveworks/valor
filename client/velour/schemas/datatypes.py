from dataclasses import dataclass, field
from typing import List

import PIL.Image

from velour import schemas


@dataclass
class ImageMetadata:
    uid: str
    height: int
    width: int
    dataset: str = field(default="")
    metadata: List[schemas.MetaDatum] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.dataset, str):
            raise TypeError("ImageMetadata dataset name must be a string.")
        if not isinstance(self.uid, str):
            raise TypeError("ImageMetadata uid must be a string.")
        if not isinstance(self.height, int):
            raise TypeError("ImageMetadata height must be a int.")
        if not isinstance(self.width, int):
            raise TypeError("ImageMetadata height must be a int.")

    @staticmethod
    def valid(datum: schemas.Datum) -> bool:
        metadata = {
            metadatum.key: metadatum.value for metadatum in datum.metadata
        }
        if "height" not in metadata:
            return False
        if "width" not in metadata:
            return False
        return True

    @classmethod
    def from_datum(cls, datum: schemas.Datum):
        if not cls.valid(datum):
            raise TypeError("Datum does not conform to image type.")

        metadata = {
            metadatum.key: metadatum.value for metadatum in datum.metadata
        }
        return cls(
            dataset=datum.dataset,
            uid=datum.uid,
            height=int(metadata.pop("height")),
            width=int(metadata.pop("width")),
            metadata=metadata,
        )

    @classmethod
    def from_pil(cls, uid: str, image: PIL.Image.Image):
        width, height = image.size
        return cls(
            uid=uid,
            height=int(height),
            width=int(width),
        )

    def to_datum(self) -> schemas.Datum:
        return schemas.Datum(
            dataset=self.dataset,
            uid=self.uid,
            metadata=[
                schemas.MetaDatum(key="height", value=self.height),
                schemas.MetaDatum(key="width", value=self.width),
                *self.metadata,
            ],
        )


@dataclass
class VideoFrameMetadata:
    image: ImageMetadata
    frame: int

    def __post_init__(self):
        # validate image
        if isinstance(self.image, dict):
            self.image = ImageMetadata(**self.image)
        if not isinstance(self.image, ImageMetadata):
            raise TypeError("Video frame must contain valid image.")

        # validate frame
        if not isinstance(self.frame, int):
            raise TypeError("Video frame number must be a int.")

    @staticmethod
    def valid(datum: schemas.Datum) -> bool:
        metadata = {
            metadatum.key: metadatum.value for metadatum in datum.metadata
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
            raise TypeError("Datum does not conform to video frame type.")

        # Extract metadata
        metadata = {
            metadatum.key: metadatum.value for metadatum in datum.metadata
        }
        return cls(
            image=ImageMetadata(
                dataset=datum.dataset,
                uid=datum.uid,
                height=int(metadata.pop("height")),
                width=int(metadata.pop("width")),
                metadata=[
                    schemas.MetaDatum(key=key, value=metadata[key])
                    for key in metadata
                ],
            ),
            frame=int(metadata.pop("frame")),
        )

    def to_datum(self) -> schemas.Datum:
        return schemas.Datum(
            dataset=self.image.dataset,
            uid=self.image.uid,
            metadata=[
                schemas.MetaDatum(key="height", value=self.image.height),
                schemas.MetaDatum(key="width", value=self.image.width),
                schemas.MetaDatum(key="frame", value=self.frame)
                * self.metadata,
            ],
        )
