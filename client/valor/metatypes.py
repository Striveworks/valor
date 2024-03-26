from typing import Optional

from PIL.Image import Image

from valor import Datum


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

        height = datum.metadata.get_value()["height"].get_value()
        width = datum.metadata.get_value()["width"].get_value()
        if not isinstance(height, int) or not isinstance(width, int):
            raise TypeError("Height and width metadata must be integers.")
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
            datum=Datum.definite(
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
    def height(self):
        return self.datum.metadata["height"]

    @property
    def width(self):
        return self.datum.metadata["width"]


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
        elif (
            not isinstance(datum.metadata["height"], int)
            or not isinstance(datum.metadata["width"], int)
            or not isinstance(datum.metadata["frame"], int)
        ):
            raise TypeError(
                "Height, width and frame number metadata must be integers."
            )
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
            Datum.definite(
                uid=uid,
                metadata=metadata,
            )
        )

    @property
    def height(self):
        return self.datum.metadata["height"]

    @property
    def width(self):
        return self.datum.metadata["width"]

    @property
    def frame(self):
        return self.datum.metadata["frame"]
