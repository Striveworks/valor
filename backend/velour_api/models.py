from geoalchemy2 import Geometry
from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, relationship

from velour_api.database import Base


class Label(Base):
    __tablename__ = "label"

    id = Column(Integer, primary_key=True, index=True)
    # (key, value) should be unique
    key = Column(String)
    value = Column(String)
    labeled_ground_truth_detections = relationship(
        "LabeledGroundTruthDetection", back_populates="label"
    )
    labeled_predicted_detections = relationship(
        "LabeledPredictedDetection", back_populates="label"
    )
    # labeled_segmentations = relationship("LabeledSegmentation")
    # labeled_images = relationship("LabeledImage")

    __table_args__ = (UniqueConstraint("key", "value"),)


class GroundTruthDetection(Base):
    """Represents a single groundtruth detection in an image. This purposefully does not have
    a label field to support the case where an object might have multiple labels (e.g. a
    car might have "make" and "model" labels)
    """

    __tablename__ = "ground_truth_detection"

    id = Column(Integer, primary_key=True, index=True)
    boundary = Column(Geometry("POLYGON"))
    image_id = Column(Integer, ForeignKey("image.id"))
    labeled_ground_truth_detections: Mapped[
        list["LabeledGroundTruthDetection"]
    ] = relationship(
        "LabeledGroundTruthDetection",
        back_populates="detection",
        cascade="all, delete",
    )
    # should add bounding box here too?
    # can get this from ST_Envelope
    # use https://docs.sqlalchemy.org/en/14/orm/mapping_columns.html#sqlalchemy.orm.column_property


class PredictedDetection(Base):
    """Represents a single groundtruth detection in an image. This purposefully does not have
    a label field to support the case where an object might have multiple labels (e.g. a
    car might have "make" and "model" labels)
    """

    __tablename__ = "predicted_detection"

    id = Column(Integer, primary_key=True, index=True)
    boundary = Column(Geometry("POLYGON"))
    image_id = Column(Integer, ForeignKey("image.id"))
    labeled_predicted_detections = relationship(
        "LabeledPredictedDetection",
        back_populates="detection",
        cascade="all, delete",
    )
    model_id = Column(
        Integer, ForeignKey("model.id")
    )  # the model that inferred this detection

    # should add bounding box here too?
    # can get this from ST_Envelope
    # use https://docs.sqlalchemy.org/en/14/orm/mapping_columns.html#sqlalchemy.orm.column_property


class LabeledGroundTruthDetection(Base):
    """Represents a grountruth detected object"""

    # also used for instance segmentation
    __tablename__ = "labeled_ground_truth_detection"

    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("ground_truth_detection.id"))
    detection = relationship(
        "GroundTruthDetection",
        back_populates="labeled_ground_truth_detections",
    )
    label_id = Column(Integer, ForeignKey("label.id"))
    label = relationship(
        "Label", back_populates="labeled_ground_truth_detections"
    )


class LabeledPredictedDetection(Base):
    """Represents a predicted detection from a model"""

    # also used for instance segmentation
    __tablename__ = "labeled_predicted_detection"

    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("predicted_detection.id"))
    detection = relationship(
        "PredictedDetection",
        back_populates="labeled_predicted_detections",
    )
    label_id = Column(Integer, ForeignKey("label.id"))
    label = relationship(
        "Label", back_populates="labeled_predicted_detections"
    )
    score = Column(Float)


class GroundTruthImageClassification(Base):
    """Groundtruth for an image classification"""

    __tablename__ = "ground_truth_image_classification"

    id = Column(Integer, primary_key=True, index=True)
    # need some uniquess for labels (a key can only appear once for a given image)
    image = Column(Integer, ForeignKey("image.id"))
    label = Column(Integer, ForeignKey("label.id"))


class PredictedImageClassification(Base):
    """Prediction for image classification from a model"""

    __tablename__ = "predicted_image_classification"

    id = Column(Integer, primary_key=True, index=True)
    score = Column(Float)
    # need some uniquess for labels (a key can only appear once for a given image)
    image = Column(Integer, ForeignKey("image.id"))
    label = Column(Integer, ForeignKey("label.id"))
    model = Column(Integer, ForeignKey("model.id"))


# class Segmentation(Base):
#     """Represents a entire image segmentation"""

#     __tablename__ = "ground_truth_segmentation"

#     id = Column(Integer, primary_key=True, index=True)
#     boundary = Column(Geometry("POLYGON"))
#     image = Column(Integer, ForeignKey("image.id"))
#     labeled_segmentations = relationship("LabeledSegmentation")


# class GroundTruthSegmentation(Base):
#     # also used for instance segmentation
#     __tablename__ = "ground_truth_segmentation"

#     id = Column(Integer, primary_key=True, index=True)
#     score = Column(Float)
#     segmentation = Column(Integer, ForeignKey("segmentation.id"))
#     label = Column(Integer, ForeignKey("label.id"))


# class PredictedSegmentation(Base):
#     # also used for instance segmentation
#     """Predicted semantic segmentation for a model"""
#     __tablename__ = "predicted_segmentation"

#     id = Column(Integer, primary_key=True, index=True)
#     score = Column(Float)
#     segmentation = Column(Integer, ForeignKey("segmentation.id"))
#     label = Column(Integer, ForeignKey("label.id"))
#     model = Column(Integer, ForeignKey("model.id"))


class Image(Base):
    """Represents an image"""

    # make every image belong to just one dataset but will be able to
    # have dataset "views"

    __tablename__ = "image"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("dataset.id"))
    uri = Column(String, unique=True)
    ground_truth_detections: Mapped[list[GroundTruthDetection]] = relationship(
        GroundTruthDetection, cascade="all, delete"
    )


class Model(Base):
    """Represents a machine learning model"""

    __tablename__ = "model"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)
    predicted_detections = relationship(
        PredictedDetection, cascade="all, delete"
    )


class Dataset(Base):
    """ """

    __tablename__ = "dataset"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)
    # whether or not the dataset is done being created
    draft = Column(Boolean, default=True)
    images = relationship("Image", cascade="all, delete")


"""
Queries:
- get all detections in a dataset with a given label key
- get all detections in a dataset with a given label key and values in a list
"""
