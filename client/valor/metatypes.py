from typing import Optional

from PIL.Image import Image

from valor import Datum
from valor.schemas import Integer


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
    """

    def __init__(self, datum: Datum):
        """
        Creates an `ImageMetadata` object from a `Datum`.

        Parameters
        ----------
        datum : Datum
            The `Datum` to extract metadata from.
        """
        if not isinstance(datum, Datum):
            raise TypeError
        elif datum.is_symbolic:
            raise ValueError

        height = int(datum.metadata.get_value()["height"].get_value())
        width = int(datum.metadata.get_value()["width"].get_value())
        datum.metadata["height"] = Integer(height)
        datum.metadata["width"] = Integer(width)
        self.datum = datum

    @classmethod
    def create(
        cls,
        uid: str,
        height: int,
        width: int,
        metadata: Optional[dict] = None,
    ):
        if not isinstance(height, int) or not isinstance(width, int):
            raise TypeError("Height and width must be integers.")
        metadata = metadata if metadata else dict()
        metadata["height"] = height
        metadata["width"] = width
        return cls(
            datum=Datum(
                uid=uid,
                metadata=metadata,
            )
        )

    @classmethod
    def from_pil(cls, image: Image, uid: str):
        """
        Creates an `ImageMetadata` object from an image.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to create metadata for.
        uid : str
            The UID of the image.
        """
        width, height = image.size
        return cls.create(uid=uid, height=height, width=width)

    @property
    def height(self) -> int:
        value = self.datum.metadata["height"].get_value()
        if not isinstance(value, int):
            raise TypeError
        return int(value)

    @property
    def width(self) -> int:
        value = self.datum.metadata["width"].get_value()
        if not isinstance(value, int):
            raise TypeError
        return int(value)


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

    def __init__(self, datum: Datum):
        """
        Creates a `VideoFrameMetadata` object from a `Datum`.

        Parameters
        ----------
        datum : Datum
            The `Datum` to extract metadata from.
        """
        if not isinstance(datum, Datum):
            raise TypeError
        elif datum.is_symbolic:
            raise ValueError

        height = int(datum.metadata.get_value()["height"].get_value())
        width = int(datum.metadata.get_value()["width"].get_value())
        frame = int(datum.metadata.get_value()["frame"].get_value())
        datum.metadata["height"] = Integer(height)
        datum.metadata["width"] = Integer(width)
        datum.metadata["frame"] = Integer(frame)
        self.datum = datum

    @classmethod
    def create(
        cls,
        uid: str,
        height: int,
        width: int,
        frame: int,
        metadata: Optional[dict] = None,
    ):
        if (
            not isinstance(height, int)
            or not isinstance(width, int)
            or not isinstance(frame, int)
        ):
            raise TypeError("Height, width and frame must be integers.")
        metadata = metadata if metadata else dict()
        metadata["height"] = height
        metadata["width"] = width
        metadata["frame"] = frame
        return cls(
            Datum(
                uid=uid,
                metadata=metadata,
            )
        )

    @property
    def height(self) -> int:
        value = self.datum.metadata["height"].get_value()
        if not isinstance(value, int):
            raise TypeError
        return int(value)

    @property
    def width(self) -> int:
        value = self.datum.metadata["width"].get_value()
        if not isinstance(value, int):
            raise TypeError
        return int(value)

    @property
    def frame(self) -> int:
        value = self.datum.metadata["frame"].get_value()
        if not isinstance(value, int):
            raise TypeError
        return int(value)
