from geoalchemy2 import Geometry
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from velour_api.database import Base


class Label(Base):
    __tablename__ = "label"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    # (key, value) should be unique
    key: Mapped[str]
    value: Mapped[str]
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

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    boundary = mapped_column(Geometry("POLYGON"))
    image_id: Mapped[int] = mapped_column(ForeignKey("image.id"))
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

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    boundary = mapped_column(Geometry("POLYGON"))
    image_id: Mapped[int] = mapped_column(ForeignKey("image.id"))
    image: Mapped["Image"] = relationship(
        "Image", back_populates="predicted_detections"
    )
    labeled_predicted_detections = relationship(
        "LabeledPredictedDetection",
        back_populates="detection",
        cascade="all, delete",
    )
    model_id: Mapped[int] = mapped_column(
        ForeignKey("model.id")
    )  # the model that inferred this detection

    # should add bounding box here too?
    # can get this from ST_Envelope
    # use https://docs.sqlalchemy.org/en/14/orm/mapping_columns.html#sqlalchemy.orm.column_property


class LabeledGroundTruthDetection(Base):
    """Represents a grountruth detected object"""

    # also used for instance segmentation
    __tablename__ = "labeled_ground_truth_detection"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    detection_id: Mapped[int] = mapped_column(
        ForeignKey("ground_truth_detection.id")
    )
    detection: Mapped[GroundTruthDetection] = relationship(
        GroundTruthDetection,
        back_populates="labeled_ground_truth_detections",
    )
    label_id: Mapped[int] = mapped_column(ForeignKey("label.id"))
    label = relationship(
        "Label", back_populates="labeled_ground_truth_detections"
    )


class LabeledPredictedDetection(Base):
    """Represents a predicted detection from a model"""

    # also used for instance segmentation
    __tablename__ = "labeled_predicted_detection"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    detection_id: Mapped[int] = mapped_column(
        ForeignKey("predicted_detection.id")
    )
    detection: Mapped[PredictedDetection] = relationship(
        "PredictedDetection",
        back_populates="labeled_predicted_detections",
    )
    label_id: Mapped[int] = mapped_column(ForeignKey("label.id"))
    label = relationship(
        "Label", back_populates="labeled_predicted_detections"
    )
    score: Mapped[float]


# class GroundTruthImageClassification(Base):
#     """Groundtruth for an image classification"""

#     __tablename__ = "ground_truth_image_classification"

#     id: Mapped[int] = mapped_column(primary_key=True, index=True)
#     # need some uniquess for labels (a key can only appear once for a given image)
#     image: Mapped[int] = mapped_column(ForeignKey("image.id"))
#     label: Mapped[int] = mapped_column(ForeignKey("label.id"))


# class PredictedImageClassification(Base):
#     """Prediction for image classification from a model"""

#     __tablename__ = "predicted_image_classification"

#     id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
#     score = Column(Float)
#     # need some uniquess for labels (a key can only appear once for a given image)
#     image = Column(Integer, ForeignKey("image.id"))
#     label = Column(Integer, ForeignKey("label.id"))
#     model = Column(Integer, ForeignKey("model.id"))


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

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    uri: Mapped[str] = mapped_column(unique=True)
    ground_truth_detections: Mapped[list[GroundTruthDetection]] = relationship(
        GroundTruthDetection, cascade="all, delete"
    )
    predicted_detections: Mapped[list[PredictedDetection]] = relationship(
        PredictedDetection, cascade="all, delete"
    )


class Model(Base):
    """Represents a machine learning model"""

    __tablename__ = "model"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(index=True, unique=True)
    predicted_detections = relationship(
        PredictedDetection, cascade="all, delete"
    )


class Dataset(Base):
    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(index=True, unique=True)
    # whether or not the dataset is done being created
    draft: Mapped[bool] = mapped_column(default=True)
    images = relationship("Image", cascade="all, delete")
