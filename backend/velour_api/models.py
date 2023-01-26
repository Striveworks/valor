from geoalchemy2 import Geometry
from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

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
    image = Column(Integer, ForeignKey("image.id"))
    labeled_detections = relationship("LabeledDetection")
    # should add bounding box here too?
    # can get this from ST_Envelope
    # use https://docs.sqlalchemy.org/en/14/orm/mapping_columns.html#sqlalchemy.orm.column_property
    # score = Column(Float)  # between 0 and 1, -1 if groundtruth


class LabeledDetection(Base):
    # also used for instance segmentation
    __tablename__ = "labeled_detection"

    id = Column(Integer, primary_key=True, index=True)
    score = Column(Float)
    detection = Column(Integer, ForeignKey("detection.id"))
    label = Column(Integer, ForeignKey("label.id"))


class LabeledImage(Base):
    __tablename__ = "labeled_image"

    id = Column(Integer, primary_key=True, index=True)
    score = Column(Float)
    # need some uniquess for labels (a key can only appear once for a given image)
    image = Column(Integer, foreign_key="image.id")
    label = Column(Integer, ForeignKey("label.id"))


class Segmentation(Base):
    """Represents a single detection in an image"""

    __tablename__ = "segmentation"

    id = Column(Integer, primary_key=True, index=True)
    boundary = Column(Geometry("POLYGON"))
    image = Column(Integer, ForeignKey("image.id"))
    labeled_segmentations = relationship("LabeledSegmentation")


class LabeledSegmentation(Base):
    # also used for instance segmentation
    __tablename__ = "labeled_segmentation"

    id = Column(Integer, primary_key=True, index=True)
    score = Column(Float)
    segmentation = Column(Integer, ForeignKey("segmentation.id"))
    label = Column(Integer, ForeignKey("label.id"))


class Image(Base):
    """Represents an image"""

    # make every image belong to just one dataset but will be able to
    # have dataset "views"

    __tablename__ = "image"

    id = Column(Integer, primary_key=True, index=True)
    dataset = Column(Integer, foreign_key="dataset.id")
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
    name = Column(String)


class Label:
    __tablename__ = "label"

    id = Column(Integer, primary_key=True, index=True)
    # (key, value) should be unique
    key = Column(String)
    value = Column(String)
    labeled_detections = relationship("LabeledDetection")
    labeled_segmentations = relationship("LabeledSegmentation")
    labeled_images = relationship("LabeledImage")


"""
Queries:
- get all detections in a dataset with a given label key
- get all detections in a dataset with a given label key and values in a list
"""
