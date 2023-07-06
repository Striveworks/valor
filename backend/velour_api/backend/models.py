from typing import Optional

from geoalchemy2 import Geography, Geometry, Raster
from geoalchemy2.functions import ST_SetBandNoDataValue, ST_SetGeoReference
from sqlalchemy import CheckConstraint, Enum, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from velour_api.backend.database import Base
from velour_api.enums import DatumTypes, TaskType


class Label(Base):
    __tablename__ = "label"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    key: Mapped[str]
    value: Mapped[str]

    # relationships
    labeled_ground_truth_detections: Mapped[
        list["GroundTruth"]
    ] = relationship(
        "GroundTruth", back_populates="label", cascade="all, delete-orphan"
    )
    labeled_predicted_detections: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="label"
    )

    __table_args__ = (UniqueConstraint("key", "value"),)


class GDALRaster(Raster):
    cache_ok = True

    # see https://github.com/geoalchemy/geoalchemy2/issues/290
    def bind_expression(self, bindvalue):
        # ST_SetBandNoDataValue tells PostGIS that values of 0 should be null
        # ST_SetGeoReference makes the convention consistent with image indices
        return ST_SetGeoReference(
            ST_SetBandNoDataValue(func.ST_FromGDALRaster(bindvalue), 0),
            "1 0 0 1 0 0",
            "GDAL",
        )


class GeometricAnnotation(Base):
    __tablename__ = "geometric_annotation"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    tasktype: Mapped[str] = mapped_column(nullable=True)
    box = mapped_column(Geometry("BOX"), nullable=True)
    polygon = mapped_column(Geometry("POLYGON"), nullable=True)
    raster = mapped_column(GDALRaster, nullable=True)

    # relationships
    metadatums: Mapped["MetaDatum"] = relationship(cascade="all, delete")


class GroundTruth(Base):
    __tablename__ = "groundtruth"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.id"), nullable=False
    )
    datum_id: Mapped[int] = mapped_column(
        ForeignKey("datum.id"), nullable=False
    )
    geometry_id: Mapped[int] = mapped_column(
        ForeignKey("geometry.id"), nullable=True
    )
    label_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"), nullable=False
    )

    # relationships
    datasets: Mapped[list["Dataset"]] = relationship(
        "Dataset", cascade="all, delete"
    )
    datums: Mapped[list["Datum"]] = relationship(
        "Datum", cascade="all, delete"
    )
    geometries: Mapped[list["Geometry"]] = relationship(
        "GeometricAnnotation", cascade="all, delete"
    )


class Prediction(Base):
    __tablename__ = "prediction"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.id"), nullable=False
    )
    model_id: Mapped[int] = mapped_column(
        ForeignKey("model.id"), nullable=False
    )
    datum_id: Mapped[int] = mapped_column(
        ForeignKey("datum.id"), nullable=False
    )
    geometry_id: Mapped[int] = mapped_column(
        ForeignKey("geometric_annotation.id"), nullable=True
    )
    label_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"), nullable=False
    )
    score: Mapped[float] = mapped_column(nullable=False)

    # relationships
    geometries: Mapped[list["Geometry"]] = relationship(
        "GeometricAnnotation", cascade="all, delete"
    )
    labels: Mapped[list["Label"]] = relationship(
        "Label", cascade="all, delete"
    )


class Datum(Base):
    """Represents an image"""

    __tablename__ = "datum"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    uid: Mapped[str] = mapped_column(nullable=False)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.id"), index=True
    )

    # relationship
    ground_truths: Mapped[list[GroundTruth]] = relationship(
        GroundTruth, cascade="all, delete"
    )
    predictions: Mapped[list[Prediction]] = relationship(
        Prediction, cascade="all, delete"
    )
    metadatums: Mapped[list["MetaDatum"]] = relationship(
        "MetaDatum", cascade="all, delete"
    )

    __table_args__ = (UniqueConstraint("dataset_id", "uid"),)


class MetaDatum(Base):
    __tablename__ = "metadatum"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(nullable=False)

    # targets
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.id"), nullable=True
    )
    model_id: Mapped[int] = mapped_column(
        ForeignKey("model.id"), nullable=True
    )
    datum_id: Mapped[int] = mapped_column(
        ForeignKey("datum.id"), nullable=True
    )
    geometry_id: Mapped[int] = mapped_column(
        ForeignKey("geometric_annotation.id"), nullable=True
    )

    # metadata
    string_value: Mapped[str] = mapped_column(nullable=True)
    numeric_value: Mapped[float] = mapped_column(nullable=True)
    geo = mapped_column(Geography(), nullable=True)
    image_id: Mapped[int] = mapped_column(
        ForeignKey("image.id"), nullable=True
    )

    # relationships
    images: Mapped["ImageMetadata"] = relationship(cascade="all, delete")

    __table_args__ = (
        CheckConstraint("num_nonnulls(string_value, numeric_value, geo) = 1"),
    )


class ImageMetadata(Base):
    __tablename__ = "metadatum_image"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    height: Mapped[int] = mapped_column(nullable=True)
    width: Mapped[int] = mapped_column(nullable=True)
    frame: Mapped[int] = mapped_column(nullable=True)

    # relationships
    metadatums = relationship(MetaDatum, cascade="all, delete")


class Model(Base):
    """Represents a machine learning model"""

    __tablename__ = "model"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(index=True, unique=True)
    href: Mapped[str] = mapped_column(index=True, nullable=True)
    description: Mapped[str] = mapped_column(index=True, nullable=True)
    type: Mapped[str] = mapped_column(
        Enum(DatumTypes), default=DatumTypes.IMAGE
    )
    predictions = relationship(Prediction, cascade="all, delete")
    evaluation_settings = relationship(
        "EvaluationSettings", cascade="all, delete"
    )


class Dataset(Base):
    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(index=True, unique=True)
    href: Mapped[str] = mapped_column(index=True, nullable=True)
    description: Mapped[str] = mapped_column(index=True, nullable=True)
    type: Mapped[str] = mapped_column(
        Enum(DatumTypes), default=DatumTypes.IMAGE
    )

    # relationships
    datums = relationship("Datum", cascade="all, delete")
    evaluation_settings = relationship(
        "EvaluationSettings", cascade="all, delete", back_populates="dataset"
    )


class EvaluationSettings(Base):
    __tablename__ = "evaluation_settings"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    model_id: Mapped[int] = mapped_column(ForeignKey("model.id"))
    model_pred_task_type: Mapped[str] = mapped_column(Enum(TaskType))
    dataset_gt_task_type: Mapped[str] = mapped_column(Enum(TaskType))
    min_area: Mapped[float] = mapped_column(nullable=True)
    max_area: Mapped[float] = mapped_column(nullable=True)
    group_by: Mapped[str] = mapped_column(nullable=True)
    label_key: Mapped[str] = mapped_column(nullable=True)

    # relationships
    dataset = relationship(Dataset, viewonly=True)
    model = relationship(Model, back_populates="evaluation_settings")
    metrics: Mapped[list["Metric"]] = relationship(
        "Metric", cascade="all, delete"
    )
    confusion_matrices: Mapped[list["ConfusionMatrix"]] = relationship(
        "ConfusionMatrix", cascade="all, delete"
    )


class Metric(Base):
    __tablename__ = "metric"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    label_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"), nullable=True
    )
    type: Mapped[str] = mapped_column()
    value: Mapped[float] = mapped_column(nullable=True)
    parameters = mapped_column(JSONB)  # {"label": ..., "iou": ..., }
    evaluation_settings_id: Mapped[int] = mapped_column(
        ForeignKey("evaluation_settings.id")
    )
    group_id: Mapped[int] = mapped_column(
        ForeignKey("metadatum.id"), nullable=True
    )

    # relationships
    label = relationship(Label)
    settings: Mapped[EvaluationSettings] = relationship(
        "EvaluationSettings", back_populates="metrics"
    )
    group = relationship(MetaDatum)


class ConfusionMatrix(Base):
    __tablename__ = "confusion_matrix"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    label_key: Mapped[str] = mapped_column()
    value = mapped_column(JSONB)

    # relationships
    settings: Mapped[EvaluationSettings] = relationship(
        "EvaluationSettings", back_populates="confusion_matrices"
    )
    evaluation_settings_id: Mapped[int] = mapped_column(
        ForeignKey("evaluation_settings.id")
    )
