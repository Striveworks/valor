from geoalchemy2 import Geometry
from sqlalchemy import Column, Float, Integer, String

from .database import Base

# need three representations for a lot of objects:
# - a (user-friendly) representation for the python client
# - a (compressed) representation for REST (these will be Pydantic models)
# - a representation for the database


class Detection(Base):
    """Represents a single detection in an image"""

    __tablename__ = "detection"

    id = Column(Integer, primary_key=True, index=True)
    boundary = Column(Geometry("POLYGON"))
    score = Column(Float)  # between 0 and 1, -1 if groundtruth
    class_label = Column(String)


class Image(Base):
    """Represents an image"""

    __tablename__ = "image"

    id = Column(Integer, primary_key=True, index=True)
    href = Column(String)


class Model(Base):
    """Represents a machine learning model"""

    __tablename__ = "model"

    id = Column(Integer, primary_key=True, index=True)


class ImageWithGroundTruth:
    # link image with detections
    pass


class ImagePrediction:
    # link image with detections and a model
    pass


class Dataset:
    __tablename__ = "dataset"

    id = Column(Integer, primary_key=True, index=True)


class DatasetImage:
    # collection of datasets and images
    pass
