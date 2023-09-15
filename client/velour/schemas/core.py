import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

from velour.enums import AnnotationType, DataType, EvaluationType, TaskType
from velour.schemas.geometry import BoundingBox, MultiPolygon, Polygon, Raster
from velour.schemas.metadata import GeoJSON


def _validate_href(v: str):
    if not isinstance(v, str):
        raise TypeError("passed something other than 'str'")
    if not (v.startswith("http://") or v.startswith("https://")):
        raise ValueError("`href` must start with http:// or https://")


@dataclass
class Metadatum:
    key: str
    value: Union[float, str, GeoJSON]

    def __post_init__(self):
        if isinstance(self.value, int):
            self.value = float(self.value)
        if not isinstance(self.key, str):
            raise TypeError("Name parameter should always be of type string.")
        if (
            not isinstance(self.value, float)
            and not isinstance(self.value, str)
            and not isinstance(self.value, GeoJSON)
        ):
            raise NotImplementedError(
                f"Value {self.value} has unsupported type {type(self.value)}"
            )
        if self.key == "href":
            _validate_href(self.value)


@dataclass
class Dataset:
    name: str
    id: int = None
    metadata: List[Metadatum] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError("name should be of type `str`")
        if not isinstance(self.id, int) and self.id is not None:
            raise TypeError("id should be of type `int`")
        if not isinstance(self.metadata, list):
            raise TypeError("metadata should be of type `list`")
        for i in range(len(self.metadata)):
            if isinstance(self.metadata[i], dict):
                self.metadata[i] = Metadatum(**self.metadata[i])
            if not isinstance(self.metadata[i], Metadatum):
                raise TypeError(
                    "elements of metadata should be of type `velour.schemas.Metadatum`"
                )


@dataclass
class Model:
    name: str
    id: int = None
    metadata: List[Metadatum] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError("name should be of type `str`")
        if not isinstance(self.id, int) and self.id is not None:
            raise TypeError("id should be of type `int`")
        if not isinstance(self.metadata, list):
            raise TypeError("metadata should be of type `list`")
        for i in range(len(self.metadata)):
            if isinstance(self.metadata[i], dict):
                self.metadata[i] = Metadatum(**self.metadata[i])
            if not isinstance(self.metadata[i], Metadatum):
                raise TypeError(
                    "elements should be of type `velour.schemas.Metadatum`"
                )


@dataclass
class Info:
    type: DataType = None
    annotation_types: List[AnnotationType] = field(default_factory=list)
    associated_datasets: List[str] = field(default_factory=list)


@dataclass
class Datum:
    uid: str
    metadata: List[Metadatum] = field(default_factory=list)
    dataset: str = field(default="")

    def __post_init__(self):
        if not isinstance(self.dataset, str):
            raise TypeError("dataset should be of type `str`")
        if not isinstance(self.uid, str):
            raise TypeError("uid should be of type `str`")
        if not isinstance(self.metadata, list):
            raise TypeError("metadata should be of type `list`")
        for i in range(len(self.metadata)):
            if isinstance(self.metadata[i], dict):
                self.metadata[i] = Metadatum(**self.metadata[i])
            if not isinstance(self.metadata[i], Metadatum):
                raise TypeError(
                    "element of metadata should be of type `velour.schemas.Metadatum`"
                )


@dataclass
class Label:
    key: str
    value: str
    score: Union[float, None] = None

    def __post_init__(self):
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


@dataclass
class Annotation:
    task_type: TaskType
    labels: List[Label] = field(default_factory=list)
    metadata: List[Metadatum] = field(default_factory=list)

    # geometric types
    bounding_box: BoundingBox = None
    polygon: Polygon = None
    multipolygon: MultiPolygon = None
    raster: Raster = None

    # @ TODO implement json annotation type
    jsonb: Dict | None = None

    def __post_init__(self):
        # task_type
        if not isinstance(self.task_type, TaskType):
            self.task_type = TaskType(self.task_type)

        # labels
        if not isinstance(self.labels, list):
            raise TypeError("labels should be of type `list`")
        for i in range(len(self.labels)):
            if isinstance(self.labels[i], dict):
                self.labels[i] = Label(**self.labels[i])
            if not isinstance(self.labels[i], Label):
                raise TypeError(
                    "elements of labels should be of type `velour.schemas.Label`"
                )

        # annotation data
        if self.bounding_box:
            if isinstance(self.bounding_box, dict):
                self.bounding_box = BoundingBox(**self.bounding_box)
            if not isinstance(self.bounding_box, BoundingBox):
                raise TypeError(
                    "bounding_box should be of type `velour.schemas.BoundingBox` or None"
                )
        if self.polygon:
            if isinstance(self.polygon, dict):
                self.polygon = Polygon(**self.polygon)
            if not isinstance(self.polygon, Polygon):
                raise TypeError(
                    "polygon should be of type `velour.schemas.Polygon` or None"
                )
        if self.multipolygon:
            if isinstance(self.multipolygon, dict):
                self.multipolygon = MultiPolygon(**self.multipolygon)
            if not isinstance(self.multipolygon, MultiPolygon):
                raise TypeError(
                    "multipolygon should be of type `velour.schemas.MultiPolygon` or None"
                )
        if self.raster:
            if isinstance(self.raster, dict):
                self.raster = Raster(**self.raster)
            if not isinstance(self.raster, Raster):
                raise TypeError(
                    "raster should be of type `velour.schemas.Raster` or None"
                )

        # metadata
        if not isinstance(self.metadata, list):
            raise TypeError("metadata should be of type `list`")
        for i in range(len(self.metadata)):
            if isinstance(self.metadata[i], dict):
                self.metadata[i] = Metadatum(**self.metadata[i])
            if not isinstance(self.metadata[i], Metadatum):
                raise TypeError(
                    "elements of metadata should be of type `velour.schemas.Metadatum`"
                )


@dataclass
class GroundTruth:
    datum: Datum
    annotations: List[Annotation] = field(default_factory=list)

    def __post_init__(self):
        # validate datum
        if isinstance(self.datum, dict):
            self.datum = Datum(**self.datum)
        if not isinstance(self.datum, Datum):
            raise TypeError("datum should be of type `velour.schemas.Datum`.")

        # validate annotations
        if not isinstance(self.annotations, list):
            raise TypeError("annotations should be of type `list`")
        for i in range(len(self.annotations)):
            if isinstance(self.annotations[i], dict):
                self.annotations[i] = Annotation(**self.annotations[i])
            if not isinstance(self.annotations[i], Annotation):
                raise TypeError(
                    "elements of annotations should be of type `velour.schemas.Annotation`"
                )


@dataclass
class Prediction:
    datum: Datum
    annotations: List[Annotation] = field(default_factory=list)
    model: str = field(default="")

    def __post_init__(self):
        # validate datum
        if isinstance(self.datum, dict):
            self.datum = Datum(**self.datum)
        if not isinstance(self.datum, Datum):
            raise TypeError("datum should be of type `velour.schemas.Datum`")

        # validate annotations
        if not isinstance(self.annotations, list):
            raise TypeError("annotations should be of type `list`")
        for i in range(len(self.annotations)):
            if isinstance(self.annotations[i], dict):
                self.annotations[i] = Annotation(**self.annotations[i])
            if not isinstance(self.annotations[i], Annotation):
                raise TypeError(
                    "elements of annotations should be of type `velour.schemas.Annotation`."
                )

        # validate model
        if not isinstance(self.model, str):
            raise TypeError("model should be of type `str`")

        for annotation in self.annotations:
            if annotation.task_type in [
                TaskType.CLASSIFICATION,
                TaskType.DETECTION,
                TaskType.INSTANCE_SEGMENTATION,
            ]:
                for label in annotation.labels:
                    if label.score is None:
                        raise ValueError(
                            f"For task type {annotation.task_type} prediction labels must have scores, but got None."
                        )

        for annotation in self.annotations:
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


@dataclass
class EvaluationSettings:
    """General parameters defining any filters of the data such
    as model, dataset, groundtruth and prediction type, model, dataset,
    size constraints, coincidence/intersection constraints, etc.
    """

    model: str
    dataset: str
    evaluation_type: EvaluationType
    task_type: TaskType
    target_type: AnnotationType = AnnotationType.NONE
    parameters: List[Metadatum] = field(default_factory=list)
    id: int | None = None

    def __post_init__(self):
        if not isinstance(self.model, str):
            raise TypeError("model should be of type `str`")
        if not isinstance(self.dataset, str):
            raise TypeError("dataset should be of type `str`")
        if not isinstance(self.evaluation_type, EvaluationType):
            raise TypeError(
                "evaluation_type should be of type `enums.EvaluationType`"
            )
        if not isinstance(self.task_type, TaskType):
            raise TypeError("task_type should be of type `enums.TaskType`")
        if not isinstance(self.target_type, AnnotationType):
            raise TypeError(
                "target_type should be of type `enums.AnnotationType`"
            )
        if not isinstance(self.parameters, list):
            raise TypeError(
                "parameters should be of type list[schemas.Metadatum]"
            )
        for idx in range(len(self.parameters)):
            if isinstance(self.parameters[idx], dict):
                self.parameters[idx] = Metadatum(**self.parameters[idx])
            if not isinstance(self.parameters[idx], Metadatum):
                raise TypeError(
                    "parameter should be of type `schemas.Metadatum`"
                )
        if self.id is not None and not isinstance(self.id, int):
            raise TypeError("id should be of type `int`")
