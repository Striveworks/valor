import math
from dataclasses import asdict
from typing import Dict, List, Tuple, Union

from velour.enums import AnnotationType, TaskType
from velour.exceptions import SchemaTypeError
from velour.filters import DeclarativeMapper, Geometry
from velour.schemas.geometry import BoundingBox, MultiPolygon, Polygon, Raster
from velour.schemas.metadata import validate_metadata


class Label:
    key = DeclarativeMapper("label.key", str)
    label = DeclarativeMapper("label.label", str)

    def __init__(self, key: str, value: str, score: Union[float, None] = None):
        self.key = key
        self.value = value
        self.score = score
        self._validate()

    def _validate(self):
        if not isinstance(self.key, str):
            raise TypeError("key should be of type `str`")
        if not isinstance(self.value, str):
            raise TypeError("value should be of type `str`")
        if isinstance(self.score, int):
            self.score = float(self.score)
        if not isinstance(self.score, (float, type(None))):
            raise TypeError("score should be of type `float`")

    def tuple(self) -> Tuple[str, str, Union[float, None]]:
        return (self.key, self.value, self.score)

    def __eq__(self, other):
        if (
            not hasattr(other, "key")
            or not hasattr(other, "key")
            or not hasattr(other, "score")
        ):
            return False

        # if the scores aren't the same type return False
        if (other.score is None) != (self.score is None):
            return False

        scores_equal = (other.score is None and self.score is None) or (
            math.isclose(self.score, other.score)
        )

        return (
            scores_equal
            and self.key == other.key
            and self.value == other.value
        )

    def __hash__(self) -> int:
        return hash(f"key:{self.key},value:{self.value},score:{self.score}")

    def dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "score": self.score,
        }


class Datum:
    id = DeclarativeMapper("datum.id", int)
    uid = DeclarativeMapper("datum.uid", str)
    metadata = DeclarativeMapper("datum.metadata", Union[int, float, str])

    def __init__(
        self,
        uid: str,
        metadata: Dict[str, Union[int, float, str]] = None,
        geo_metadata: Dict[
            str, List[List[List[float]]] | List[float] | str
        ] = None,
        dataset: str = "",
    ):
        self.uid = uid
        self.metadata = metadata if metadata else {}
        self.geo_metadata = geo_metadata if geo_metadata else {}
        self.dataset = dataset
        self._validate()

    def _validate(self):
        if not isinstance(self.dataset, str):
            raise SchemaTypeError("dataset", str, self.dataset)
        if not isinstance(self.uid, str):
            raise SchemaTypeError("uid", str, self.uid)
        validate_metadata(self.metadata)

    def dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "uid": self.uid,
            "metadata": self.metadata,
            "geo_metadata": self.geo_metadata,
        }

    def __eq__(self, other):
        if not isinstance(other, Datum):
            raise TypeError(f"Expected type `{type(Datum)}`, got `{other}`")
        return self.dict() == other.dict()


class Annotation:
    task_type = DeclarativeMapper("annotation.task_type", TaskType)
    annotation_type = DeclarativeMapper(
        "annotation.annotation_type", AnnotationType
    )
    box = Geometry("box")
    polygon = Geometry("polygon")
    multipolygon = Geometry("multipolygon")
    raster = Geometry("raster")
    json = DeclarativeMapper("annotation.json", object)
    metadata = DeclarativeMapper("annotation.metadata", Union[int, float, str])

    def __init__(
        self,
        task_type: TaskType,
        labels: List[Label],
        metadata: Dict[str, Union[int, float, str]] = None,
        bounding_box: BoundingBox = None,
        polygon: Polygon = None,
        multipolygon: MultiPolygon = None,
        raster: Raster = None,
        jsonb: Dict = None,
    ):
        self.task_type = task_type
        self.labels = labels
        self.metadata = metadata if metadata else {}
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
            raise SchemaTypeError("labels", List[Label], self.labels)
        for idx, label in enumerate(self.labels):
            if isinstance(self.labels[idx], dict):
                self.labels[idx] = Label(**label)
            if not isinstance(self.labels[idx], Label):
                raise SchemaTypeError("label", Label, self.labels[idx])

        # annotation data
        if self.bounding_box:
            if isinstance(self.bounding_box, dict):
                self.bounding_box = BoundingBox(**self.bounding_box)
            if not isinstance(self.bounding_box, BoundingBox):
                raise SchemaTypeError(
                    "bounding_box", BoundingBox, self.bounding_box
                )
        if self.polygon:
            if isinstance(self.polygon, dict):
                self.polygon = Polygon(**self.polygon)
            if not isinstance(self.polygon, Polygon):
                raise SchemaTypeError("polygon", Polygon, self.polygon)
        if self.multipolygon:
            if isinstance(self.multipolygon, dict):
                self.multipolygon = MultiPolygon(**self.multipolygon)
            if not isinstance(self.multipolygon, MultiPolygon):
                raise SchemaTypeError(
                    "multipolygon", MultiPolygon, self.multipolygon
                )
        if self.raster:
            if isinstance(self.raster, dict):
                self.raster = Raster(**self.raster)
            if not isinstance(self.raster, Raster):
                raise SchemaTypeError("raster", Raster, self.raster)

        # metadata
        validate_metadata(self.metadata)

    def dict(self) -> dict:
        return {
            "task_type": self.task_type.value,
            "labels": [label.dict() for label in self.labels],
            "metadata": self.metadata,
            "bounding_box": asdict(self.bounding_box)
            if self.bounding_box
            else None,
            "polygon": asdict(self.polygon) if self.polygon else None,
            "multipolygon": asdict(self.multipolygon)
            if self.multipolygon
            else None,
            "raster": asdict(self.raster) if self.raster else None,
            "jsonb": self.jsonb,
        }

    def __eq__(self, other):
        if not isinstance(other, Annotation):
            raise TypeError(
                f"Expected type `{type(Annotation)}`, got `{other}`"
            )
        return self.dict() == other.dict()


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
            raise SchemaTypeError("datum", Datum, self.datum)

        # validate annotations
        if not isinstance(self.annotations, list):
            raise SchemaTypeError(
                "annotations", List[Annotation], self.annotations
            )
        for idx, annotation in enumerate(self.annotations):
            if isinstance(self.annotations[idx], dict):
                self.annotations[idx] = Annotation(**annotation)
            if not isinstance(self.annotations[idx], Annotation):
                raise SchemaTypeError(
                    "annotation", Annotation, self.annotations[idx]
                )

    def dict(self) -> dict:
        return {
            "datum": self.datum.dict(),
            "annotations": [
                annotation.dict() for annotation in self.annotations
            ],
        }

    def __eq__(self, other):
        if not isinstance(other, GroundTruth):
            raise TypeError(
                f"Expected type `{type(GroundTruth)}`, got `{other}`"
            )
        return self.dict() == other.dict()


class Prediction:
    score = DeclarativeMapper("prediction.score", Union[int, float])

    def __init__(
        self,
        datum: Datum,
        annotations: List[Annotation] = None,
        model: str = "",
    ):
        self.datum = datum
        self.annotations = annotations
        self.model = model
        self._validate()

    def _validate(self):
        # validate datum
        if isinstance(self.datum, dict):
            self.datum = Datum(**self.datum)
        if not isinstance(self.datum, Datum):
            raise SchemaTypeError("datum", Datum, self.datum)

        # validate annotations
        if not isinstance(self.annotations, list):
            raise SchemaTypeError(
                "annotations", List[Annotation], self.annotations
            )
        for idx, annotation in enumerate(self.annotations):
            if isinstance(self.annotations[idx], dict):
                self.annotations[idx] = Annotation(**annotation)
            if not isinstance(self.annotations[idx], Annotation):
                raise SchemaTypeError(
                    "annotation", Annotation, self.annotations[idx]
                )

        # validate model
        if not isinstance(self.model, str):
            raise SchemaTypeError("model", str, self.model)

        # TaskType-specific validations
        for annotation in self.annotations:
            if annotation.task_type in [
                TaskType.CLASSIFICATION,
                TaskType.DETECTION,
            ]:
                for label in annotation.labels:
                    if label.score is None:
                        raise ValueError(
                            f"For task type `{annotation.task_type}` prediction labels must have scores, but got `None`"
                        )
            if annotation.task_type == TaskType.CLASSIFICATION:
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
        return {
            "datum": self.datum.dict(),
            "model": self.model,
            "annotations": [
                annotation.dict() for annotation in self.annotations
            ],
        }

    def __eq__(self, other):
        if not isinstance(other, Prediction):
            raise TypeError(
                f"Expected type `{type(Prediction)}`, got `{other}`"
            )
        return self.dict() == other.dict()
