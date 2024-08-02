import datetime

from geoalchemy2 import Geometry, Raster
from geoalchemy2.functions import ST_SetBandNoDataValue, ST_SetGeoReference
from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from valor_api.backend.database import Base


class Label(Base):
    __tablename__ = "label"
    __table_args__ = (UniqueConstraint("key", "value"),)

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    key: Mapped[str]
    value: Mapped[str]
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())

    # relationships
    groundtruths: Mapped[list["GroundTruth"]] = relationship(
        back_populates="label"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="label"
    )


class Embedding(Base):
    __tablename__ = "embedding"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    value = mapped_column(Vector())
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())

    # relationships
    annotations: Mapped[list["Annotation"]] = relationship(
        back_populates="embedding"
    )


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
    __table_args__ = (
        UniqueConstraint(
            "annotation_id",
            "label_id",
        ),
    )

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    annotation_id: Mapped[int] = mapped_column(
        ForeignKey("annotation.id"), nullable=True
    )
    label_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"),
        nullable=False,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())

    # relationships
    annotation: Mapped["Annotation"] = relationship(
        back_populates="groundtruths"
    )
    label: Mapped["Label"] = relationship(back_populates="groundtruths")


class Prediction(Base):
    __tablename__ = "prediction"
    __table_args__ = (
        UniqueConstraint(
            "annotation_id",
            "label_id",
        ),
    )

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    annotation_id: Mapped[int] = mapped_column(
        ForeignKey("annotation.id"), nullable=True
    )
    label_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"),
        nullable=False,
    )
    score: Mapped[float] = mapped_column(nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())

    # relationships
    annotation: Mapped["Annotation"] = relationship(
        back_populates="predictions"
    )
    label: Mapped["Label"] = relationship(back_populates="predictions")


class Annotation(Base):
    __tablename__ = "annotation"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    datum_id: Mapped[int] = mapped_column(
        ForeignKey("datum.id"), nullable=False, index=True
    )
    model_id: Mapped[int] = mapped_column(
        ForeignKey("model.id"), nullable=True, index=True
    )
    text: Mapped[str] = mapped_column(nullable=True)
    context = mapped_column(JSONB)

    meta = mapped_column(JSONB)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())

    # columns - linked objects
    box = mapped_column(Geometry("POLYGON"), nullable=True)
    polygon = mapped_column(Geometry("POLYGON"), nullable=True)
    raster = mapped_column(GDALRaster, nullable=True)
    embedding_id = mapped_column(ForeignKey("embedding.id"), nullable=True)
    is_instance: Mapped[bool] = mapped_column(nullable=False)
    implied_task_types = mapped_column(JSONB)

    # relationships
    datum: Mapped["Datum"] = relationship(back_populates="annotations")
    model: Mapped["Model"] = relationship(back_populates="annotations")
    embedding: Mapped[Embedding] = relationship(back_populates="annotations")
    groundtruths: Mapped[list["GroundTruth"]] = relationship(
        cascade="all, delete-orphan"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        cascade="all, delete-orphan"
    )
    groundtruth_ious: Mapped[list["IoU"]] = relationship(
        foreign_keys="IoU.groundtruth_annotation_id",
        cascade="all, delete-orphan",
    )
    prediction_ious: Mapped[list["IoU"]] = relationship(
        foreign_keys="IoU.prediction_annotation_id",
        cascade="all, delete-orphan",
    )


class Datum(Base):
    __tablename__ = "datum"
    __table_args__ = (UniqueConstraint("dataset_id", "uid"),)

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.id"), nullable=False
    )
    uid: Mapped[str] = mapped_column(nullable=False)
    text: Mapped[str] = mapped_column(nullable=True)
    meta = mapped_column(JSONB)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())

    # relationship
    dataset: Mapped["Dataset"] = relationship(back_populates="datums")
    annotations: Mapped[list[Annotation]] = relationship(
        cascade="all, delete-orphan"
    )


class Model(Base):
    """Represents a machine learning model"""

    __tablename__ = "model"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(index=True, unique=True)
    meta = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())

    # relationships
    annotations: Mapped[list[Annotation]] = relationship(
        cascade="all, delete-orphan"
    )


class Dataset(Base):
    __tablename__ = "dataset"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(index=True, unique=True)
    meta = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())

    # relationships
    datums: Mapped[list[Datum]] = relationship(cascade="all, delete")


class Evaluation(Base):
    __tablename__ = "evaluation"
    __table_args__ = (
        UniqueConstraint(
            "dataset_names",
            "model_name",
            "filters",
            "parameters",
        ),
    )

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_names = mapped_column(JSONB, nullable=False)
    model_name: Mapped[str] = mapped_column(nullable=False)
    filters = mapped_column(JSONB, nullable=False)
    parameters = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())
    meta = mapped_column(JSONB)

    # relationships
    metrics: Mapped[list["Metric"]] = relationship(
        "Metric", cascade="all, delete"
    )
    confusion_matrices: Mapped[list["ConfusionMatrix"]] = relationship(
        "ConfusionMatrix", cascade="all, delete"
    )


class Metric(Base):
    __tablename__ = "metric"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    evaluation_id: Mapped[int] = mapped_column(ForeignKey("evaluation.id"))
    label_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"), nullable=True
    )
    type: Mapped[str] = mapped_column()
    value = mapped_column(JSONB, nullable=True)
    parameters = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())

    # relationships
    label = relationship(Label)
    settings: Mapped[Evaluation] = relationship(back_populates="metrics")


class ConfusionMatrix(Base):
    __tablename__ = "confusion_matrix"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    evaluation_id: Mapped[int] = mapped_column(ForeignKey("evaluation.id"))
    label_key: Mapped[str] = mapped_column()
    value = mapped_column(JSONB)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())

    # relationships
    settings: Mapped[Evaluation] = relationship(
        back_populates="confusion_matrices"
    )


class IoU(Base):
    __tablename__ = "iou"
    __table_args__ = (
        UniqueConstraint(
            "groundtruth_annotation_id",
            "prediction_annotation_id",
            "type",
        ),
    )

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    groundtruth_annotation_id: Mapped[int] = mapped_column(
        ForeignKey("annotation.id"), nullable=False
    )
    prediction_annotation_id: Mapped[int] = mapped_column(
        ForeignKey("annotation.id"), nullable=False
    )
    iou: Mapped[float] = mapped_column(nullable=False)
    type: Mapped[str] = mapped_column(nullable=False)

    # relationships
    groundtruth = relationship(
        "Annotation",
        foreign_keys=[groundtruth_annotation_id],
        back_populates="groundtruth_ious",
    )
    prediction = relationship(
        "Annotation",
        foreign_keys=[prediction_annotation_id],
        back_populates="prediction_ious",
    )
