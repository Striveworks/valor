from typing import Dict, List, Optional, Union, cast

import PIL.Image

from velour import Datum
from velour.schemas import validate_metadata
from velour.types import MetadataType, DictMetadataType, GeoJSONType


class ImageMetadata:
    """
    A class describing the metadata for a particular image.

    Parameters
    ----------
    uid : str
        The UID of the image.
    height : int
        The height of the image.
    width : int
        The width of the image.
    metadata : dict
        A dictionary of metadata that describes the image.
    geospatial :  dict
        A GeoJSON-style dictionary describing the geospatial coordinates of the image.
    """

    def __init__(
        self,
        uid: str,
        height: int,
        width: int,
        metadata: Optional[MetadataType] = None,
        geospatial: Optional[GeoJSONType] = None,
    ):
        self.uid = uid
        self._dataset_name = None
        self.height = height
        self.width = width
        self.metadata: DictMetadataType = dict(metadata) if metadata else {}
        self.geospatial = geospatial if geospatial else {}

        if not isinstance(self.uid, str):
            raise TypeError("ImageMetadata uid must be a string.")
        if not isinstance(self.height, int):
            raise TypeError("ImageMetadata height must be a int.")
        if not isinstance(self.width, int):
            raise TypeError("ImageMetadata height must be a int.")
        validate_metadata(self.metadata)

    @staticmethod
    def valid(datum: Datum) -> bool:
        """
        Asserts wehether the `Datum's` height and width is a valid subset of the image's height and width.

        Parameters
        ----------
        datum : Datum
            The `Datum` to check validity for.
        """
        return {"height", "width"}.issubset(datum.metadata)

    @classmethod
    def from_datum(cls, datum: Datum):
        """
        Creates an `ImageMetadata` object from a `Datum`.

        Parameters
        ----------
        datum : Datum
            The `Datum` to extract metadata from.
        """
        if not cls.valid(datum):
            raise ValueError(
                f"`datum` does not contain height and/or width in metadata `{datum.metadata}`"
            )
        metadata = dict(datum.metadata)
        width = cast(int, metadata.pop("width"))
        height = cast(int, metadata.pop("height"))
        img = cls(
            uid=datum.uid,
            height=int(height),
            width=int(width),
            metadata=metadata,
        )
        return img

    @classmethod
    def from_pil(cls, uid: str, image: PIL.Image.Image):
        """
        Creates an `ImageMetadata` object from an image.

        Parameters
        ----------
        uid : str
            The UID of the image.
        image : PIL.Image.Image
            The image to create metadata for.
        """
        width, height = image.size
        return cls(
            uid=uid,
            height=int(height),
            width=int(width),
        )

    def to_datum(self) -> Datum:
        """
        Converts an `ImageMetadata` object into a `Datum`.
        """
        metadata = dict(self.metadata) if self.metadata else {}
        geospatial = self.geospatial.copy() if self.geospatial else {}

        metadata["height"] = self.height
        metadata["width"] = self.width
        datum = Datum(
            uid=self.uid,
            metadata=metadata,
            geospatial=geospatial,
        )
        return datum


class VideoFrameMetadata:
    """
    A class describing the metadata for the frame of a video.

    Parameters
    ----------
    image : ImageMetadata
        Metadata describing the frame of the video.
    frame: int
        The number of seconds into the video that the frame was taken.
    """

    def __init__(
        self,
        image: ImageMetadata,
        frame: int,
    ):
        self.image = image
        self.frame = frame

        if not isinstance(self.image, ImageMetadata):
            raise TypeError(
                "`image` should be of type `velour.metatypes.ImageMetadata`"
            )
        if not isinstance(self.frame, int):
            raise TypeError("`frame` should be of type `int`.")

    @staticmethod
    def valid(datum: Datum) -> bool:
        """
        Asserts wehether the `Datum's` height and width is a valid subset of the image's height and width.

        Parameters
        ----------
        datum : Datum
            The `Datum` to check validity for.
        """
        return {"height", "width", "frame"}.issubset(datum.metadata)

    @classmethod
    def from_datum(cls, datum: Datum):
        """
        Creates an `VideoFrameMetadata` object from a `Datum`.

        Parameters
        ----------
        datum : Datum
            The `Datum` to extract metadata from.
        """
        if not cls.valid(datum):
            raise ValueError(
                f"`datum` does not contain height, width and/or frame in metadata `{datum.metadata}`"
            )
        image = ImageMetadata.from_datum(datum)
        frame = cast(int, image.metadata.pop("frame"))
        return cls(
            image=image,
            frame=int(frame),
        )

    def to_datum(self) -> Datum:
        """
        Converts an `VideoFrameMetadata` object into a `Datum`.
        """
        datum = self.image.to_datum()
        datum.metadata["frame"] = self.frame
        return datum
