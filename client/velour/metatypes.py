from typing import Dict, List, Union

import PIL.Image

from velour.coretypes import Datum
from velour.exceptions import SchemaTypeError
from velour.schemas import validate_metadata


class ImageMetadata:
    def __init__(
        self,
        uid: str,
        height: int,
        width: int,
        dataset: str = "",
        metadata: Dict[str, Union[int, float, str]] = None,
        geo_metadata: Dict[str, List[List[List[float]]]] = None,
    ):
        self.uid = uid
        self.dataset = dataset
        self.height = height
        self.width = width
        self.metadata = validate_metadata(metadata if metadata else {})
        self.geo_metadata = geo_metadata if geo_metadata else {}

        if not isinstance(self.dataset, str):
            raise TypeError("ImageMetadata dataset name must be a string.")
        if not isinstance(self.uid, str):
            raise TypeError("ImageMetadata uid must be a string.")
        if not isinstance(self.height, int):
            raise TypeError("ImageMetadata height must be a int.")
        if not isinstance(self.width, int):
            raise TypeError("ImageMetadata height must be a int.")

    @staticmethod
    def valid(datum: Datum) -> bool:
        return {"height", "width"}.issubset(datum.metadata)

    @classmethod
    def from_datum(cls, datum: Datum):
        if not cls.valid(datum):
            raise ValueError(
                f"`datum` does not contain height and/or width in metadata `{datum.metadata}`"
            )
        metadata = datum.metadata.copy()
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

    def to_datum(self) -> Datum:
        metadata = self.metadata.copy() if self.metadata else {}
        geo_metadata = self.geo_metadata.copy() if self.geo_metadata else {}

        metadata["height"] = self.height
        metadata["width"] = self.width
        return Datum(
            dataset=self.dataset,
            uid=self.uid,
            metadata=metadata,
            geo_metadata=geo_metadata,
        )


class VideoFrameMetadata:
    def __init__(
        self,
        image: ImageMetadata,
        frame: int,
    ):
        self.image = image
        self.frame = frame

        if not isinstance(self.image, ImageMetadata):
            raise SchemaTypeError("image", ImageMetadata, self.image)
        if not isinstance(self.frame, int):
            raise SchemaTypeError("frame", int, self.frame)

    @staticmethod
    def valid(datum: Datum) -> bool:
        return {"height", "width", "frame"}.issubset(datum.metadata)

    @classmethod
    def from_datum(cls, datum: Datum):
        if not cls.valid(datum):
            raise ValueError(
                f"`datum` does not contain height, width and/or frame in metadata `{datum.metadata}`"
            )
        image = ImageMetadata.from_datum(datum)
        frame = image.metadata.pop("frame")
        return cls(
            image=image,
            frame=frame,
        )

    def to_datum(self) -> Datum:
        datum = self.image.to_datum()
        datum.metadata["frame"] = self.frame
        return datum
