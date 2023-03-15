from geoalchemy2 import Geometry, Raster
from geoalchemy2.functions import ST_SetBandNoDataValue, ST_SetGeoReference
from sqlalchemy import Enum, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from velour_api.database import Base
from velour_api.enums import Task


class Label(Base):
    __tablename__ = "label"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    # (key, value) should be unique
    key: Mapped[str]
    value: Mapped[str]
    labeled_ground_truth_detections: Mapped[
        list["LabeledGroundTruthDetection"]
    ] = relationship("LabeledGroundTruthDetection", back_populates="label")
    labeled_predicted_detections: Mapped[
        list["LabeledPredictedDetection"]
    ] = relationship("LabeledPredictedDetection", back_populates="label")
    ground_truth_image_classifications: Mapped[
        list["GroundTruthImageClassification"]
    ] = relationship("GroundTruthImageClassification", back_populates="label")
    predicted_image_classifications: Mapped[
        list["PredictedImageClassification"]
    ] = relationship("PredictedImageClassification", back_populates="label")
    labeled_ground_truth_segmentations: Mapped[
        list["LabeledGroundTruthSegmentation"]
    ] = relationship("LabeledGroundTruthSegmentation", back_populates="label")
    labeled_predicted_segmentations: Mapped[
        list["LabeledPredictedSegmentation"]
    ] = relationship("LabeledPredictedSegmentation", back_populates="label")

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

    # add image_id property for easier access
    @property
    def image_id(self):
        return self.detection.image_id


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

    # add image_id property for easier access
    @property
    def image_id(self):
        return self.detection.image_id


class GroundTruthImageClassification(Base):
    """Groundtruth for an image classification"""

    __tablename__ = "ground_truth_image_classification"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    # need some uniquess for labels (a key can only appear once for a given image)
    image_id: Mapped[int] = mapped_column(ForeignKey("image.id"))
    label_id: Mapped[int] = mapped_column(ForeignKey("label.id"))
    label = relationship(
        "Label", back_populates="ground_truth_image_classifications"
    )


class PredictedImageClassification(Base):
    """Prediction for image classification from a model"""

    __tablename__ = "predicted_image_classification"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    score: Mapped[float]
    # need some uniquess for labels (a key can only appear once for a given image)
    image_id: Mapped[int] = mapped_column(ForeignKey("image.id"))
    label_id: Mapped[int] = mapped_column(ForeignKey("label.id"))
    model_id: Mapped[int] = mapped_column(ForeignKey("model.id"))
    label = relationship(
        "Label", back_populates="predicted_image_classifications"
    )


class GDALRaster(Raster):
    # see https://github.com/geoalchemy/geoalchemy2/issues/290
    def bind_expression(self, bindvalue):
        # ST_SetBandNoDataValue tells PostGIS that values of 0 should be null
        # ST_SetGeoReference makes the convention consistent with image indices
        return ST_SetGeoReference(
            ST_SetBandNoDataValue(func.ST_FromGDALRaster(bindvalue), 0),
            "1 0 0 1 0 0",
            "GDAL",
        )


class GroundTruthSegmentation(Base):
    # also used for instance segmentation
    __tablename__ = "ground_truth_segmentation"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    is_instance: Mapped[bool] = mapped_column(nullable=False)
    shape = mapped_column(GDALRaster)
    image_id: Mapped[int] = mapped_column(ForeignKey("image.id"))
    image = relationship("Image", back_populates="ground_truth_segmentations")
    labeled_ground_truth_segmentations: Mapped[
        list["LabeledGroundTruthSegmentation"]
    ] = relationship(
        "LabeledGroundTruthSegmentation",
        back_populates="segmentation",
        cascade="all, delete",
    )


class PredictedSegmentation(Base):
    # also used for instance segmentation
    """Predicted semantic segmentation for a model"""
    __tablename__ = "predicted_segmentation"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    is_instance: Mapped[bool] = mapped_column(nullable=False)
    shape = mapped_column(GDALRaster)
    image_id: Mapped[int] = mapped_column(ForeignKey("image.id"))
    image: Mapped["Image"] = relationship(
        "Image", back_populates="predicted_segmentations"
    )
    labeled_predicted_segmentations = relationship(
        "LabeledPredictedSegmentation",
        back_populates="segmentation",
        cascade="all, delete",
    )
    model_id: Mapped[int] = mapped_column(
        ForeignKey("model.id")
    )  # the model that inferred this segmentation


class LabeledGroundTruthSegmentation(Base):
    """Represents a grountruth semantic segmentation"""

    # also used for instance segmentation
    __tablename__ = "labeled_ground_truth_segmentation"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    segmentation_id: Mapped[int] = mapped_column(
        ForeignKey("ground_truth_segmentation.id")
    )
    segmentation: Mapped[GroundTruthSegmentation] = relationship(
        GroundTruthSegmentation,
        back_populates="labeled_ground_truth_segmentations",
    )
    label_id: Mapped[int] = mapped_column(ForeignKey("label.id"))
    label = relationship(
        "Label", back_populates="labeled_ground_truth_segmentations"
    )

    # add image_id property for easier access
    @property
    def image_id(self):
        return self.segmentation.image_id


class LabeledPredictedSegmentation(Base):
    """Represents a predicted semantic segmentation"""

    # also used for instance segmentation
    __tablename__ = "labeled_predicted_segmentation"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    segmentation_id: Mapped[int] = mapped_column(
        ForeignKey("predicted_segmentation.id")
    )
    segmentation: Mapped[PredictedDetection] = relationship(
        PredictedSegmentation,
        back_populates="labeled_predicted_segmentations",
    )
    label_id: Mapped[int] = mapped_column(ForeignKey("label.id"))
    label = relationship(
        "Label", back_populates="labeled_predicted_segmentations"
    )
    score: Mapped[float]

    # add image_id property for easier access
    @property
    def image_id(self):
        return self.segmentation.image_id


class Image(Base):
    """Represents an image"""

    __tablename__ = "image"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    uid: Mapped[str] = mapped_column()
    height: Mapped[int] = mapped_column()
    width: Mapped[int] = mapped_column()
    ground_truth_detections: Mapped[list[GroundTruthDetection]] = relationship(
        GroundTruthDetection, cascade="all, delete"
    )
    predicted_detections: Mapped[list[PredictedDetection]] = relationship(
        PredictedDetection, cascade="all, delete"
    )
    ground_truth_classifications: Mapped[
        list[GroundTruthImageClassification]
    ] = relationship(GroundTruthImageClassification, cascade="all, delete")
    predicted_classifications: Mapped[
        list[PredictedImageClassification]
    ] = relationship(PredictedImageClassification, cascade="all, delete")
    ground_truth_segmentations: Mapped[
        list[GroundTruthSegmentation]
    ] = relationship(GroundTruthSegmentation, cascade="all, delete")
    predicted_segmentations: Mapped[
        list[PredictedSegmentation]
    ] = relationship(PredictedSegmentation, cascade="all, delete")

    __table_args__ = (UniqueConstraint("dataset_id", "uid"),)


class Model(Base):
    """Represents a machine learning model"""

    __tablename__ = "model"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(index=True, unique=True)
    predicted_detections = relationship(
        PredictedDetection, cascade="all, delete"
    )
    predicted_image_classifications = relationship(
        PredictedImageClassification, cascade="all, delete"
    )
    predicted_segmentations = relationship(
        PredictedSegmentation, cascade="all, delete"
    )


class Dataset(Base):
    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(index=True, unique=True)
    # whether or not the dataset is done being created
    draft: Mapped[bool] = mapped_column(default=True)
    images = relationship("Image", cascade="all, delete")


class MetricParameters(Base):
    __tablename__ = "metric_parameters"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    model_id: Mapped[int] = mapped_column(ForeignKey("model.id"))
    model_pred_type: Mapped[str] = mapped_column(Enum(Task))
    dataset_gt_type: Mapped[str] = mapped_column(Enum(Task))


class APMetric(Base):
    __tablename__ = "ap_metric"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    label_id: Mapped[int] = mapped_column(ForeignKey("label.id"))
    iou_threshold: Mapped[float] = mapped_column()
    value: Mapped[float] = mapped_column()
    metric_parameters_id: Mapped[int] = mapped_column(
        ForeignKey("metric_parameters.id")
    )
