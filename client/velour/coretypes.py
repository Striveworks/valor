from dataclasses import asdict
from typing import Dict, Set, Tuple, Union

from velour import schemas
from velour.enums import AnnotationType, TaskType
from velour.filters import DeclarativeMapper, Geometry
from velour.schemas import GeoJSON


def _validate_labels(labels: Dict[str, Set[Tuple[str, Union[float, None]]]]):
    if not isinstance(labels, dict):
        raise TypeError(f"Expected `labels` to be of type `Dict[str, Tuple[str, Union[float, None]]]`, got {type(metadata)}")
    for key in labels:
        if not isinstance(key, str):
            raise TypeError(f"Expected label key to be of type `str`, got {type(key)}")
        if not isinstance(labels[key], set):
            raise TypeError(f"Expected metadata value to be of type int, float, str or GeoJSON, got {type(metadata[key])}")


def _validate_metadata(metadata: Dict[str, Union[int, float, str, GeoJSON]]):
    if not isinstance(metadata, dict):
        raise TypeError(f"Expected `metadata` to be of type `dict[str, int | float | str | GeoJSON]`, got {type(metadata)}")
    for key in metadata:
        if not isinstance(key, str):
            raise TypeError(f"Expected metadata key to be of type `str`, got {type(key)}")
        if not isinstance(metadata[key], Union[int, float, str, GeoJSON]):
            raise TypeError(f"Expected metadata value to be of type int, float, str or GeoJSON, got {type(metadata[key])}")

class Label:
    key = DeclarativeMapper("label.key", str)
    label = DeclarativeMapper("label.label", str)

    def __init__(
        self,
        key: str,
        value: str,
        score: Union[float, None] = None,
    ):
        self.key = key
        self.value = value
        self.score = score

        if not isinstance(self.key, str):
            raise TypeError("key should be of type `str`")
        if not isinstance(self.value, str):
            raise TypeError("value should be of type `str`")
        if isinstance(self.score, int):
            self.score = float(self.score)
        if not isinstance(self.score, (float, type(None))):
            raise TypeError("score should be of type `float`")

    # def serialize(self) ->

    def tuple(self) -> Tuple[str, str, Union[float, None]]:
        return (self.key, self.value, self.score)

    def from_tuple(self, label: Tuple[str, str, Union[float, None]]):
        if len(label) != 3:
            raise ValueError
        elif (
            isinstance(label[0], str)
            and isinstance(label[1], str)
            and isinstance(label[1], Union[float, None])
        ):
            raise ValueError
        self.key = label[0]
        self.value = label[1]
        self.score = label[2]

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


class Datum:
    id = DeclarativeMapper("datum.id", int)
    uid = DeclarativeMapper("datum.uid", str)
    metadata = DeclarativeMapper("datum.metadata", Union[int, float, str])

    def __init__(
        self,
        uid: str,
        metadata: Dict[str, Union[int, float, str]] = None,
        dataset: str = "",
    ):
        if not isinstance(dataset, str):
            raise TypeError("`dataset` should be of type `str`")
        if not isinstance(uid, str):
            raise TypeError("`uid` should be of type `str`")

        metadata = metadata if metadata is not None else {}
        _validate_metadata(metadata)

        self.uid = uid
        self.metadata = metadata
        self.dataset = dataset

    def dict(self):
        return {
            "uid": self.uid,
            "dataset": self.dataset,
            "metadata": self.metadata,
        }


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
        labels: List[schemas.Label] = None,
        metadata: List[schemas.Metadatum] = None,
        bounding_box: schemas.BoundingBox = None,
        polygon: schemas.Polygon = None,
        multipolygon: schemas.MultiPolygon = None,
        raster: schemas.Raster = None,
        jsonb: Dict = None,
    ):
        self.task_type = task_type
        self.labels = labels if labels is not None else []
        self.metadata = metadata if metadata is not None else []
        self.bounding_box = bounding_box
        self.polygon = polygon
        self.multipolygon = multipolygon
        self.raster = raster
        self.jsonb = jsonb

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
        if not isinstance(self.metadata, list):
            raise TypeError("metadata should be of type `list`")
        for i in range(len(self.metadata)):
            if isinstance(self.metadata[i], dict):
                self.metadata[i] = schemas.Metadatum(**self.metadata[i])
            if not isinstance(self.metadata[i], schemas.Metadatum):
                raise TypeError(
                    "elements of metadata should be of type `velour.schemas.Metadatum`"
                )


class GroundTruth:
    def __init__(
        self,
        datum: Datum,
        annotations: List[Annotation] = None,
    ):
        self.datum = datum
        self.annotations = annotations if annotations is not None else []

        # validate datum
        if isinstance(self.datum, dict):
            self.datum = Datum(**self.datum)
        if not isinstance(self.datum, Datum):
            raise TypeError(
                f"datum should be of type `velour.Datum`, not {type(self.datum)}."
            )

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

    @classmethod
    def from_dict(cls, groundtruth):


    def to_dict(self):
        return asdict(

        )

class Prediction:
    score = DeclarativeMapper("prediction.score", Union[int, float])

    def __init__(
        self,
        datum: Datum,
        annotations: List[Annotation] = None,
        model: str = "",
    ):
        self.score = None # mask the static class variable

        self.datum = datum
        self.annotations = annotations if annotations is not None else []
        self.model = model

        # validate datum
        if isinstance(self.datum, dict):
            self.datum = Datum(**self.datum)
        if not isinstance(self.datum, Datum):
            raise TypeError("datum should be of type `velour.Datum`")

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

    def to_dict(self):
        pass

    def from_dict(self):
        pass
