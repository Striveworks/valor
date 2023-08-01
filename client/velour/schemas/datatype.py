from dataclasses import dataclass, field

from velour import schemas


@dataclass
class Image:
    uid: str
    height: int
    width: int
    metadata: list[schemas.MetaDatum] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.uid, str):
            raise TypeError("Image uid must be a string.")
        if not isinstance(self.height, int):
            raise TypeError("Image height must be a int.")
        if not isinstance(self.width, int):
            raise TypeError("Image height must be a int.")

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
        height = int(metadata["height"])
        width = int(metadata["width"])
        del metadata["height"]
        del metadata["width"]
        return cls(
            uid=datum.uid,
            height=height,
            width=width,
            metadata=metadata,
        )

    def to_datum(self) -> schemas.Datum:
        return schemas.Datum(
            uid=self.uid,
            metadata=[
                schemas.MetaDatum(key="height", value=self.height),
                schemas.MetaDatum(key="width", value=self.width),
                *self.metadata,
            ],
        )


@dataclass
class VideoFrame:
    image: Image
    frame: int

    def __post_init__(self):
        # validate image
        if isinstance(self.image, dict):
            self.image = Image(**self.image)
        if not isinstance(self.image, Image):
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

        # Extract image metadata
        image = Image.from_datum(datum)

        # Extract frame number
        metadatum = filter(lambda x: x.key == "frame", datum.metadata)
        if len(metadatum) != 1:
            raise TypeError("Missing frame number in metadata.")
        frame = int(metadatum[0])

        return cls(
            image=image,
            frame=frame,
        )

    def to_datum(self) -> schemas.Datum:
        return schemas.Datum(
            uid=self.image.uid,
            metadata=[
                schemas.MetaDatum(key="height", value=self.image.height),
                schemas.MetaDatum(key="width", value=self.image.width),
                schemas.MetaDatum(key="frame", value=self.frame)
                * self.metadata,
            ],
        )
