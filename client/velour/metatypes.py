from typing import Dict, Union

import PIL.Image

from velour.coretypes import Datum
from velour.schemas import validate_metadata


class ImageMetadata:
    def __init__(
        self,
        uid: str,
        height: int,
        width: int,
        dataset: str = "",
        metadata: Dict[str, Union[int, float, str]] = None,
    ):
        self.uid = uid
        self.dataset = dataset
        self.metadata = validate_metadata(metadata if metadata else {})
        self.height = height
        self.width = width

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
        if "height" not in datum.metadata:
            return False
        if "width" not in datum.metadata:
            return False
        return True

    @classmethod
    def from_datum(cls, datum: Datum):
        if not cls.valid(datum):
            raise TypeError("Datum does not conform to image type.")
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
        if self.metadata:
            metadata = self.metadata.copy()
        else:
            metadata = {}
        metadata["height"] = self.height
        metadata["width"] = self.width
        return Datum(
            dataset=self.dataset,
            uid=self.uid,
            metadata=metadata,
        )


class VideoFrameMetadata:
    pass


# @dataclass
# class VideoFrameMetadata:
#     image: ImageMetadata
#     frame: int

#     def __post_init__(self):
#         # validate image
#         if isinstance(self.image, dict):
#             self.image = ImageMetadata(**self.image)
#         if not isinstance(self.image, ImageMetadata):
#             raise TypeError("Video frame must contain valid image.")

#         # validate frame
#         if not isinstance(self.frame, int):
#             raise TypeError("Video frame number must be a int.")

#     @staticmethod
#     def valid(datum: Datum) -> bool:
#         if "height" not in datum.metadata:
#             return False
#         if "width" not in datum.metadata:
#             return False
#         if "frame" not in datum.metadata:
#             return False
#         return True

#     @classmethod
#     def from_datum(cls, datum: Datum):
#         if not cls.valid(datum):
#             raise TypeError("Datum does not conform to video frame type.")
#         metadata = datum.metadata.copy()
#         return cls(
#             image=ImageMetadata(
#                 dataset=datum.dataset,
#                 uid=datum.uid,
#                 height=int(metadata.pop("height")),
#                 width=int(metadata.pop("width")),
#                 metadata=metadata,
#             ),
#             frame=int(metadata.pop("frame")),
#         )

#     def to_datum(self) -> Datum:
#         metadata = self.image.metadata.copy()
#         metadata["height"] = self.image.height
#         metadata["width"] = self.image.width
#         metadata["frame"] = self.frame
#         return Datum(
#             dataset=self.image.dataset,
#             uid=self.image.uid,
#             metadata=metadata,
#         )
