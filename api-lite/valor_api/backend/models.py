import datetime

from geoalchemy2 import Geometry, Raster
from geoalchemy2.functions import ST_SetBandNoDataValue, ST_SetGeoReference
from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from valor_api.backend.database import Base


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


class Label(Base):
    __tablename__ = "label_annotation"
    __table_args__ = (UniqueConstraint("key", "value"),)

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    key: Mapped[str]
    value: Mapped[str]
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class Embedding(Base):
    __tablename__ = "embedding_annotation"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    value = mapped_column(Vector(), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class Box(Base):
    __tablename__ = "box_annotation"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    value = mapped_column(Geometry("BOX"), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class Polygon(Base):
    __tablename__ = "polygon_annotation"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    value = mapped_column(Geometry("POLYGON"), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class Bitmask(Base):
    __tablename__ = "bitmask_annotation"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    value = mapped_column(GDALRaster, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class Text(Base):
    __tablename__ = "text_annotation"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    value: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class ContextList(Base):
    __tablename__ = "context_list_annotation"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    value = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class ClassificationGroundTruth(Base):
    __tablename__ = "classification_groundtruth"
    __table_args__ = (UniqueConstraint("dataset_id", "label_id"),)

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    label_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"),
        nullable=False,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class ClassificationPrediction(Base):
    __tablename__ = "classification_prediction"
    __table_args__ = (UniqueConstraint("dataset_id", "label_id"),)

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    label_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"),
        nullable=False,
    )
    score: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class Classification(Base):
    __tablename__ = "classification"
    __table_args__ = (UniqueConstraint("dataset_id", "uid"),)

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    uid: Mapped[str] = mapped_column()
    meta = mapped_column(JSONB)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class DetectionGroundTruth(Base):
    __tablename__ = "object_detection_groundtruth"
    __table_args__ = (UniqueConstraint("dataset_id", "label_id"),)

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    label_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"),
        nullable=False,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class DetectionPrediction(Base):
    __tablename__ = "object_detection_prediction"
    __table_args__ = (UniqueConstraint("dataset_id", "label_id"),)

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    label_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"),
        nullable=False,
    )
    score: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class Detection(Base):
    __tablename__ = "object_detection"
    __table_args__ = (UniqueConstraint("dataset_id", "uid"),)

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    uid: Mapped[str] = mapped_column()
    meta = mapped_column(JSONB)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class Segmentation(Base):
    __tablename__ = "semantic_segmentation"
    __table_args__ = (UniqueConstraint("dataset_id", "uid"),)

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    uid: Mapped[str] = mapped_column()
    meta = mapped_column(JSONB)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class TextGeneration(Base):
    __tablename__ = "text_generation"
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

    meta = mapped_column(JSONB)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())

    # columns - linked objects
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
