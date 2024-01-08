from geoalchemy2 import Geography, Geometry, Raster
from geoalchemy2.functions import ST_SetBandNoDataValue, ST_SetGeoReference
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import datetime

from velour_api.backend.database import Base


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
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(default=func.now(), onupdate=func.now())

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
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(default=func.now(), onupdate=func.now())


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
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(default=func.now(), onupdate=func.now())


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
    meta = mapped_column(JSONB)
    geo = mapped_column(Geography(), nullable=True)

    # Geometric
    box = mapped_column(Geometry("POLYGON"), nullable=True)
    polygon = mapped_column(Geometry("POLYGON"), nullable=True)
    multipolygon = mapped_column(Geometry("MULTIPOLYGON"), nullable=True)
    raster = mapped_column(GDALRaster, nullable=True)

    # Generic
    json = mapped_column(JSONB)

    # relationships
    datum: Mapped["Datum"] = relationship(back_populates="annotations")
    model: Mapped["Model"] = relationship(back_populates="annotations")
    groundtruths: Mapped[list["GroundTruth"]] = relationship(
        cascade="all, delete-orphan"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        cascade="all, delete-orphan"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(default=func.now(), onupdate=func.now())


class Datum(Base):
    __tablename__ = "datum"
    __table_args__ = (UniqueConstraint("dataset_id", "uid"),)

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.id"),
        nullable=False,
    )
    uid: Mapped[str] = mapped_column(nullable=False)
    meta = mapped_column(JSONB)
    geo = mapped_column(Geography(), nullable=True)

    # relationship
    dataset: Mapped["Dataset"] = relationship(back_populates="datums")
    annotations: Mapped[list[Annotation]] = relationship(
        cascade="all, delete-orphan"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(default=func.now(), onupdate=func.now())


class Model(Base):
    """Represents a machine learning model"""

    __tablename__ = "model"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(index=True, unique=True)
    meta = mapped_column(JSONB)
    geo = mapped_column(Geography(), nullable=True)

    # relationships
    annotations: Mapped[list[Annotation]] = relationship(
        cascade="all, delete-orphan"
    )
    evaluation: Mapped[list["Evaluation"]] = relationship(
        cascade="all, delete"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(default=func.now(), onupdate=func.now())


class Dataset(Base):
    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(index=True, unique=True)
    meta = mapped_column(JSONB)
    geo = mapped_column(Geography(), nullable=True)

    # relationships
    datums: Mapped[list[Datum]] = relationship(cascade="all, delete")
    evaluation: Mapped[list["Evaluation"]] = relationship(
        cascade="all, delete"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(default=func.now(), onupdate=func.now())

class Evaluation(Base):
    __tablename__ = "evaluation"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    model_id: Mapped[int] = mapped_column(ForeignKey("model.id"))
    task_type: Mapped[str] = mapped_column(nullable=False)
    settings = mapped_column(JSONB, nullable=True)
    geo = mapped_column(Geography(), nullable=True)

    # relationships
    dataset = relationship(Dataset, back_populates="evaluation")
    model = relationship(Model, back_populates="evaluation")
    metrics: Mapped[list["Metric"]] = relationship(
        "Metric", cascade="all, delete"
    )
    confusion_matrices: Mapped[list["ConfusionMatrix"]] = relationship(
        "ConfusionMatrix", cascade="all, delete"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(default=func.now(), onupdate=func.now())


class Metric(Base):
    __tablename__ = "metric"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    evaluation_id: Mapped[int] = mapped_column(ForeignKey("evaluation.id"))
    label_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"), nullable=True
    )
    type: Mapped[str] = mapped_column()
    value: Mapped[float] = mapped_column(nullable=True)
    parameters = mapped_column(JSONB)

    # relationships
    label = relationship(Label)
    settings: Mapped[Evaluation] = relationship(
        "Evaluation", back_populates="metrics"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(default=func.now(), onupdate=func.now())


class ConfusionMatrix(Base):
    __tablename__ = "confusion_matrix"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    evaluation_id: Mapped[int] = mapped_column(ForeignKey("evaluation.id"))
    label_key: Mapped[str] = mapped_column()
    value = mapped_column(JSONB)

    # relationships
    settings: Mapped[Evaluation] = relationship(
        back_populates="confusion_matrices"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(default=func.now(), onupdate=func.now())
