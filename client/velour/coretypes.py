from dataclasses import asdict
from typing import Dict, List, Union

from velour import schemas
from velour.enums import TaskType
from velour.schemas.core import (
    _BaseAnnotation,
    _BaseDatum,
    _BaseGroundTruth,
    _BasePrediction,
)
from velour.schemas.metadata import serialize_metadata, validate_metadata


class Datum:
    def __init__(
        self,
        uid: str,
        metadata: Dict[str, Union[int, float, str]] = None,
        dataset: str = "",
    ):
        self.uid = uid
        self.metadata = metadata
        self.dataset = dataset
        self._validate()

    def _validate(self):
        if not isinstance(self.dataset, str):
            raise TypeError("`dataset` should be of type `str`")
        if not isinstance(self.uid, str):
            raise TypeError("`uid` should be of type `str`")
        self.metadata = validate_metadata(self.metadata)

    def dict(self):
        return asdict(
            _BaseDatum(
                **{
                    "uid": self.uid,
                    "dataset": self.dataset,
                    "metadata": serialize_metadata(self.metadata),
                }
            )
        )


class Annotation:
    def __init__(
        self,
        task_type: TaskType,
        labels: List[schemas.Label],
        metadata: Dict[str, Union[int, float, str]] = None,
        bounding_box: schemas.BoundingBox = None,
        polygon: schemas.Polygon = None,
        multipolygon: schemas.MultiPolygon = None,
        raster: schemas.Raster = None,
        jsonb: Dict = None,
    ):
        self.task_type = task_type
        self.labels = labels
        self.metadata = metadata
        self.bounding_box = bounding_box
        self.polygon = polygon
        self.multipolygon = multipolygon
        self.raster = raster
        self.jsonb = jsonb
        self._validate()

    def _validate(self):
        # task_type
        if not isinstance(self.task_type, TaskType):
            self.task_type = TaskType(self.task_type)

        # labels
        if not isinstance(self.labels, list):
            raise TypeError("labels should be of type `list`")
        for idx, label in enumerate(self.labels):
            if isinstance(self.labels[idx], dict):
                self.labels[idx] = schemas.Label(**label)
            if not isinstance(self.labels[idx], schemas.Label):
                raise TypeError(
                    "elements of labels should be of type `velour.schemas.Label`"
                )

        # annotation data
        if self.bounding_box:
            if isinstance(self.bounding_box, dict):
                self.bounding_box = schemas.BoundingBox(**self.bounding_box)
            if not isinstance(self.bounding_box, schemas.BoundingBox):
                raise TypeError(
                    "bounding_box should be of type `velour.schemas.BoundingBox` or None"
                )
        if self.polygon:
            if isinstance(self.polygon, dict):
                self.polygon = schemas.Polygon(**self.polygon)
            if not isinstance(self.polygon, schemas.Polygon):
                raise TypeError(
                    "polygon should be of type `velour.schemas.Polygon` or None"
                )
        if self.multipolygon:
            if isinstance(self.multipolygon, dict):
                self.multipolygon = schemas.MultiPolygon(**self.multipolygon)
            if not isinstance(self.multipolygon, schemas.MultiPolygon):
                raise TypeError(
                    "multipolygon should be of type `velour.schemas.MultiPolygon` or None"
                )
        if self.raster:
            if isinstance(self.raster, dict):
                self.raster = schemas.Raster(**self.raster)
            if not isinstance(self.raster, schemas.Raster):
                raise TypeError(
                    "raster should be of type `velour.schemas.Raster` or None"
                )

        # metadata
        self.metadata = validate_metadata(self.metadata)

    def dict(self) -> dict:
        return asdict(
            _BaseAnnotation(
                **{
                    "task_type": self.task_type.value,
                    "labels": self.labels,
                    "metadata": serialize_metadata(self.metadata),
                    "bounding_box": self.bounding_box,
                    "polygon": self.polygon,
                    "multipolygon": self.multipolygon,
                    "raster": self.raster,
                    "jsonb": self.jsonb,
                }
            )
        )


class GroundTruth:
    def __init__(
        self,
        datum: Datum,
        annotations: List[Annotation],
    ):
        self.datum = datum
        self.annotations = annotations
        self._validate()

    def _validate(self):
        # validate datum
        if isinstance(self.datum, dict):
            self.datum = Datum(**self.datum)
        if not isinstance(self.datum, Datum):
            raise TypeError(
                f"datum should be of type `velour.Datum`, not {type(self.datum)}."
            )
        # validate annotations
        if not isinstance(self.annotations, list):
            raise TypeError("`annotations` should be of type `list`")
        for i in range(len(self.annotations)):
            if isinstance(self.annotations[i], dict):
                self.annotations[i] = Annotation(**self.annotations[i])
            if not isinstance(self.annotations[i], Annotation):
                raise TypeError(
                    "elements of `annotations` should be of type `velour.schemas.Annotation`"
                )

    def dict(self) -> dict:
        return asdict(
            _BaseGroundTruth(
                **{
                    "datum": self.datum.dict(),
                    "annotations": [
                        annotation.dict() for annotation in self.annotations
                    ],
                }
            )
        )


class Prediction:
    def __init__(
        self,
        datum: Datum,
        annotations: List[Annotation] = None,
        model: str = "",
    ):
        self.datum = datum
        self.annotations = annotations if annotations is not None else []
        self.model = model
        self._validate()

    def _validate(self):
        # validate datum
        if isinstance(self.datum, dict):
            self.datum = Datum(**self.datum)
        if not isinstance(self.datum, Datum):
            raise TypeError("datum should be of type `velour.Datum`")
        # validate model
        if not isinstance(self.model, str):
            raise TypeError("model should be of type `str`")

        # validate annotations
        if not isinstance(self.annotations, list):
            raise TypeError("annotations should be of type `list`")
        for idx, annotation in enumerate(self.annotations):
            if isinstance(annotation, dict):
                self.annotations[idx] = Annotation(**annotation)
            if not isinstance(self.annotations[idx], Annotation):
                raise TypeError(
                    "elements of annotations should be of type `velour.schemas.Annotation`."
                )

        # TaskType-specific validations
        for annotation in self.annotations:
            if annotation.task_type in [
                TaskType.CLASSIFICATION,
                TaskType.DETECTION,
            ]:
                for label in annotation.labels:
                    if label.score is None:
                        raise ValueError(
                            f"For task type {annotation.task_type} prediction labels must have scores, but got None."
                        )
            elif annotation.task_type == TaskType.CLASSIFICATION:
                label_keys_to_sum = {}
                for scored_label in annotation.labels:
                    label_key = scored_label.key
                    if label_key not in label_keys_to_sum:
                        label_keys_to_sum[label_key] = 0.0
                    label_keys_to_sum[label_key] += scored_label.score

                for k, total_score in label_keys_to_sum.items():
                    if abs(total_score - 1) > 1e-5:
                        raise ValueError(
                            "For each label key, prediction scores must sum to 1, but"
                            f" for label key {k} got scores summing to {total_score}."
                        )

    def dict(self) -> dict:
        return asdict(
            _BasePrediction(
                **{
                    "datum": self.datum.dict(),
                    "model": self.model,
                    "annotations": [
                        annotation.dict() for annotation in self.annotations
                    ],
                }
            )
        )
