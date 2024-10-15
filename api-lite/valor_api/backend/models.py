import datetime

from geoalchemy2 import Geometry
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from valor_api.backend.database import Base


class Dataset(Base):
    __tablename__ = "dataset"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(index=True, unique=True)
    meta = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class Model(Base):
    __tablename__ = "model"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(index=True, unique=True)
    meta = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class Datum(Base):
    __tablename__ = "datum"
    __table_args__ = (UniqueConstraint("dataset_id", "uid"),)

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.id"), nullable=False
    )
    uid: Mapped[str] = mapped_column(nullable=False)
    geo: Mapped[int] = mapped_column(
        ForeignKey("geospatial.id"), nullable=True
    )
    meta = mapped_column(JSONB)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class Geospatial(Base):
    __tablename__ = "geospatial"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    point = mapped_column(Geometry("POINT"), nullable=False)
    region = mapped_column(Geometry("POLYGON"), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class Label(Base):
    __tablename__ = "label_annotation"
    __table_args__ = (UniqueConstraint("key", "value"),)

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    key: Mapped[str]
    value: Mapped[str]
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class Classification(Base):
    __tablename__ = "classification"
    __table_args__ = (
        UniqueConstraint("datum_id", "groundtruth_id", "prediction_id"),
    )

    # columns
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
    groundtruth_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"), nullable=False
    )
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"), nullable=False
    )
    score: Mapped[float] = mapped_column(nullable=False)
    hardmax: Mapped[bool] = mapped_column(nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class DetectionExamples(Base):
    __tablename__ = "object_detection_examples"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    value = mapped_column(Geometry("BOX"), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class Detection(Base):
    __tablename__ = "object_detection"
    __table_args__ = (
        UniqueConstraint("datum_id", "groundtruth_id", "prediction_id"),
    )

    # columns
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
    groundtruth_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"), nullable=False
    )
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"), nullable=False
    )
    score: Mapped[float] = mapped_column(nullable=False)
    iou: Mapped[float] = mapped_column(nullable=False)
    example_id: Mapped[int] = mapped_column(
        ForeignKey("object_detection_examples.id"), nullable=False
    )
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


class Segmentation(Base):
    __tablename__ = "semantic_segmentation"
    __table_args__ = (
        UniqueConstraint("datum_id", "groundtruth_id", "prediction_id"),
    )

    # columns
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
    groundtruth_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"), nullable=True
    )
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("label.id"), nullable=True
    )
    count: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


# class TextGeneration(Base):
#     __tablename__ = "text_generation"
#     __table_args__ = (UniqueConstraint("dataset_id", "uid"),)

#     # columns
#     id: Mapped[int] = mapped_column(primary_key=True, index=True)
#     dataset_id: Mapped[int] = mapped_column(
#         ForeignKey("dataset.id"), nullable=False
#     )
#     uid: Mapped[str] = mapped_column(nullable=False)
#     text: Mapped[str] = mapped_column(nullable=True)
#     meta = mapped_column(JSONB)
#     created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())


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


class Metric(Base):
    __tablename__ = "metric"

    # columns
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    evaluation_id: Mapped[int] = mapped_column(ForeignKey("evaluation.id"))
    type: Mapped[str] = mapped_column(nullable=False)
    value = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(default=func.now())

    # relationships
    settings: Mapped[Evaluation] = relationship(back_populates="metrics")
