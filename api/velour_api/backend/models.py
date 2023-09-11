from geoalchemy2 import Geography, Geometry, Raster
from geoalchemy2.functions import ST_SetBandNoDataValue, ST_SetGeoReference
from sqlalchemy import CheckConstraint, Enum, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from velour_api.backend.database import Base
from velour_api.enums import AnnotationType, TaskType


class Label(Base):
    __tablename__ = "label"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    key: Mapped[str]
    value: Mapped[str]

    # relationships
    groundtruths: Mapped[list["GroundTruth"]] = relationship(
        back_populates="label"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="label"
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


class GroundTruth(Base):
    __tablename__ = "groundtruth"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    annotation_id: Mapped[int] = mapped_column(
        ForeignKey("annotation.id"), nullable=True
    )
    label_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"), nullable=False
    )

    # relationships
    annotation: Mapped["Annotation"] = relationship(
        back_populates="groundtruths"
    )
    label: Mapped["Label"] = relationship(back_populates="groundtruths")


class Prediction(Base):
    __tablename__ = "prediction"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    annotation_id: Mapped[int] = mapped_column(
        ForeignKey("annotation.id"), nullable=True
    )
    label_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"), nullable=False
    )
    score: Mapped[float] = mapped_column(nullable=True)

    # relationships
    annotation: Mapped["Annotation"] = relationship(
        back_populates="predictions"
    )
    label: Mapped["Label"] = relationship(back_populates="predictions")


class MetaDatum(Base):
    __tablename__ = "metadatum"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    key: Mapped[str] = mapped_column(nullable=False)

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
    annotation_id: Mapped[int] = mapped_column(
        ForeignKey("annotation.id"), nullable=True
    )

    # metadata
    string_value: Mapped[str] = mapped_column(nullable=True)
    numeric_value: Mapped[float] = mapped_column(nullable=True)
    geo = mapped_column(Geography(), nullable=True)

    # relationships
    dataset: Mapped["Dataset"] = relationship(back_populates="metadatums")
    model: Mapped["Model"] = relationship(back_populates="metadatums")
    datum: Mapped["Datum"] = relationship(back_populates="metadatums")
    annotation: Mapped["Annotation"] = relationship(
        back_populates="metadatums"
    )

    __table_args__ = (
        CheckConstraint("num_nonnulls(string_value, numeric_value, geo) = 1"),
    )


class Annotation(Base):
    __tablename__ = "annotation"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    datum_id: Mapped[int] = mapped_column(
        ForeignKey("datum.id"), nullable=False
    )
    model_id: Mapped[int] = mapped_column(
        ForeignKey("model.id"), nullable=True
    )
    task_type: Mapped[str] = mapped_column(nullable=False)

    # Annotation
    box = mapped_column(Geometry("POLYGON"), nullable=True)
    polygon = mapped_column(Geometry("POLYGON"), nullable=True)
    multipolygon = mapped_column(Geometry("MULTIPOLYGON"), nullable=True)
    raster = mapped_column(GDALRaster, nullable=True)

    # relationships
    datum: Mapped["Datum"] = relationship(back_populates="annotations")
    model: Mapped["Model"] = relationship(back_populates="annotations")
    groundtruths: Mapped[list["GroundTruth"]] = relationship(
        cascade="all, delete-orphan"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        cascade="all, delete-orphan"
    )
    metadatums: Mapped[list["MetaDatum"]] = relationship(cascade="all, delete")


class Datum(Base):
    __tablename__ = "datum"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    uid: Mapped[str] = mapped_column(nullable=False)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.id"), index=True
    )

    # relationship
    dataset: Mapped["Dataset"] = relationship(back_populates="datums")
    annotations: Mapped[list[Annotation]] = relationship(
        cascade="all, delete-orphan"
    )
    metadatums: Mapped[list["MetaDatum"]] = relationship(cascade="all, delete")

    __table_args__ = (UniqueConstraint("dataset_id", "uid"),)


class Model(Base):
    """Represents a machine learning model"""

    __tablename__ = "model"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(index=True, unique=True)

    # relationships
    annotations: Mapped[list[Annotation]] = relationship(
        cascade="all, delete-orphan"
    )
    metadatums: Mapped[list["MetaDatum"]] = relationship(cascade="all, delete")
    evaluation_settings = relationship(
        "EvaluationSettings", cascade="all, delete"
    )


class Dataset(Base):
    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(index=True, unique=True)

    # relationships
    datums: Mapped[list[Datum]] = relationship(cascade="all, delete")
    metadatums: Mapped[list[MetaDatum]] = relationship(cascade="all, delete")
    evaluation_settings = relationship(
        "EvaluationSettings", cascade="all, delete", back_populates="dataset"
    )


class EvaluationSettings(Base):
    __tablename__ = "evaluation_settings"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    model_id: Mapped[int] = mapped_column(ForeignKey("model.id"))
    task_type: Mapped[str] = mapped_column(Enum(TaskType))
    pd_type: Mapped[str] = mapped_column(Enum(AnnotationType))
    gt_type: Mapped[str] = mapped_column(Enum(AnnotationType))
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


annotation_type_to_geometry = {
    AnnotationType.BOX: Annotation.box,
    AnnotationType.POLYGON: Annotation.polygon,
    AnnotationType.MULTIPOLYGON: Annotation.multipolygon,
    AnnotationType.RASTER: Annotation.raster,
}
