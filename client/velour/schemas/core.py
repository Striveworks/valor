from dataclasses import dataclass, field
from typing import List, Tuple, Union

from velour import enums
from velour.schemas.geometry import BoundingBox, MultiPolygon, Polygon, Raster
from velour.schemas.metadata import GeoJSON


def _validate_href(v: str):
    if not isinstance(v, str):
        raise TypeError("passed something other than 'str'")
    if not (v.startswith("http://") or v.startswith("https://")):
        raise ValueError("`href` must start with http:// or https://")


@dataclass
class MetaDatum:
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
    metadata: List[MetaDatum] = field(default_factory=list)

    def __post_init__(self):
        assert isinstance(self.name, str)
        assert isinstance(self.id, Union[int, None])
        assert isinstance(self.metadata, list)
        for i in range(len(self.metadata)):
            if isinstance(self.metadata[i], dict):
                self.metadata[i] = MetaDatum(**self.metadata[i])
                assert isinstance(self.metadata[i], MetaDatum)
            else:
                assert isinstance(self.metadata[i], MetaDatum)


@dataclass
class Model:
    name: str
    id: int = None
    metadata: List[MetaDatum] = field(default_factory=list)

    def __post_init__(self):
        assert isinstance(self.name, str)
        assert isinstance(self.id, Union[int, None])
        assert isinstance(self.metadata, list)
        for i in range(len(self.metadata)):
            if isinstance(self.metadata[i], dict):
                self.metadata[i] = MetaDatum(**self.metadata[i])
                assert isinstance(self.metadata[i], MetaDatum)
            else:
                assert isinstance(self.metadata[i], MetaDatum)


@dataclass
class Info:
    type: enums.DataType = None
    annotation_types: List[enums.AnnotationType] = field(default_factory=list)
    associated_datasets: List[str] = field(default_factory=list)


@dataclass
class Datum:
    uid: str
    metadata: List[MetaDatum] = field(default_factory=list)

    def __post_init__(self):
        assert isinstance(self.uid, str)
        assert isinstance(self.metadata, list)
        for i in range(len(self.metadata)):
            if isinstance(self.metadata[i], dict):
                self.metadata[i] = MetaDatum(**self.metadata[i])
                assert isinstance(self.metadata[i], MetaDatum)
            else:
                assert isinstance(self.metadata[i], MetaDatum)


@dataclass
class Label:
    key: str
    value: str

    def __post_init__(self):
        if not isinstance(self.key, str):
            raise TypeError("Label key is not a `str`.")
        if not isinstance(self.value, str):
            raise TypeError("Label value is not a `str`.")

    def tuple(self) -> Tuple[str, str]:
        return (self.key, self.value)

    def __eq__(self, other):
        if hasattr(other, "key") and hasattr(other, "value"):
            return self.key == other.key and self.value == other.value
        return False

    def __hash__(self) -> int:
        return hash(f"key:{self.key},value:{self.value}")


@dataclass
class ScoredLabel:
    label: Label
    score: float

    def __post_init__(self):
        # unpack
        if isinstance(self.label, dict):
            self.label = Label(**self.label)

        # validate
        if not isinstance(self.label, Label):
            raise TypeError("label should be of type `velour.schemas.Label`.")
        if not isinstance(self.score, Union[float, int]):
            raise TypeError("score should be of type `float`")

    @property
    def key(self):
        return self.label.key

    @property
    def value(self):
        return self.label.value

    def __eq__(self, other):
        if hasattr(other, "label") and hasattr(other, "score"):
            return self.score == other.score and self.label == other.label
        return False

    def __neq__(self, other):
        if hasattr(other, "label") and hasattr(other, "score"):
            return not self == other
        return True

    def __hash__(self) -> int:
        return hash(f"key:{self.key},value:{self.value},score:{self.score}")


@dataclass
class Annotation:
    task_type: enums.TaskType
    labels: List[Label] = field(default_factory=list)
    metadata: List[MetaDatum] = field(default_factory=list)

    # geometric types
    bounding_box: BoundingBox = None
    polygon: Polygon = None
    multipolygon: MultiPolygon = None
    raster: Raster = None

    def __post_init__(self):
        # task_type
        if not isinstance(self.task_type, enums.TaskType):
            self.task_type = enums.TaskType(self.task_type)

        # labels
        try:
            if not isinstance(self.labels, list):
                raise TypeError(
                    "labels should be a list of `velour.schemas.Label`."
                )
            for i in range(len(self.labels)):
                if isinstance(self.labels[i], dict):
                    self.labels[i] = Label(**self.labels[i])
                assert isinstance(self.labels[i], Label)
        except AssertionError:
            raise TypeError(
                "elements of labels should be of type `velour.schemas.Label`"
            )

        # annotation data
        try:
            if self.bounding_box:
                if isinstance(self.bounding_box, dict):
                    self.bounding_box = BoundingBox(**self.bounding_box)
                assert isinstance(self.bounding_box, BoundingBox)
            if self.polygon:
                if isinstance(self.polygon, dict):
                    self.polygon = Polygon(**self.polygon)
                assert isinstance(self.polygon, Polygon)
            if self.multipolygon:
                if isinstance(self.multipolygon, dict):
                    self.multipolygon = MultiPolygon(**self.multipolygon)
                assert isinstance(self.multipolygon, MultiPolygon)
            if self.raster:
                if isinstance(self.raster, dict):
                    self.raster = Raster(**self.raster)
                assert isinstance(self.raster, Raster)
        except AssertionError:
            raise TypeError("Annotation value does not match intended type.")

        # metadata
        try:
            assert isinstance(self.metadata, list)
            for i in range(len(self.metadata)):
                if isinstance(self.metadata[i], dict):
                    self.metadata[i] = MetaDatum(**self.metadata[i])
                assert isinstance(self.metadata[i], MetaDatum)
        except AssertionError:
            raise TypeError("Metadata contains incorrect type.")


@dataclass
class ScoredAnnotation:
    task_type: enums.TaskType
    scored_labels: List[ScoredLabel] = field(default_factory=list)
    metadata: List[MetaDatum] = field(default_factory=list)

    # geometric types
    bounding_box: BoundingBox = None
    polygon: Polygon = None
    multipolygon: MultiPolygon = None
    raster: Raster = None

    def __post_init__(self):
        # task_type
        if not isinstance(self.task_type, enums.TaskType):
            self.task_type = enums.TaskType(self.task_type)

        # scored_labels
        try:
            if not isinstance(self.scored_labels, list):
                raise TypeError(
                    "scored_labels should be a list of `velour.schemas.ScoredLabel`."
                )
            for i in range(len(self.scored_labels)):
                if isinstance(self.scored_labels[i], dict):
                    self.scored_labels[i] = ScoredLabel(
                        **self.scored_labels[i]
                    )
                assert isinstance(self.scored_labels[i], ScoredLabel)
        except AssertionError:
            raise TypeError(
                "elements of scored_labels should be of type `velour.schemas.ScoredLabel`"
            )

        # annotation data
        try:
            if self.bounding_box:
                if isinstance(self.bounding_box, dict):
                    self.bounding_box = BoundingBox(**self.bounding_box)
                assert isinstance(self.bounding_box, BoundingBox)
            if self.polygon:
                if isinstance(self.polygon, dict):
                    self.polygon = Polygon(**self.polygon)
                assert isinstance(self.polygon, Polygon)
            if self.multipolygon:
                if isinstance(self.multipolygon, dict):
                    self.multipolygon = MultiPolygon(**self.multipolygon)
                assert isinstance(self.multipolygon, MultiPolygon)
            if self.raster:
                if isinstance(self.raster, dict):
                    self.raster = Raster(**self.raster)
                assert isinstance(self.raster, Raster)
        except AssertionError:
            raise TypeError("Value does not match intended type.")

        # metadata
        try:
            assert isinstance(self.metadata, list)
            for i in range(len(self.metadata)):
                if isinstance(self.metadata[i], dict):
                    self.metadata[i] = MetaDatum(**self.metadata[i])
                assert isinstance(self.metadata[i], MetaDatum)
        except AssertionError:
            raise TypeError("Metadata contains incorrect type.")

        # check that for each label key all the predictions sum to ~1
        # the backend should also do this validation but its good to do
        # it here on creation, before sending to backend
        if self.task_type == enums.TaskType.CLASSIFICATION:
            label_keys_to_sum = {}
            for scored_label in self.scored_labels:
                label_key = scored_label.label.key
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
class GroundTruth:
    datum: Datum
    annotations: List[Annotation] = field(default_factory=list)
    dataset: str = field(default="")

    def __post_init__(self):
        # validate datum
        if isinstance(self.datum, dict):
            self.datum = Datum(**self.datum)
        if not isinstance(self.datum, Datum):
            raise TypeError("datum should be type `velour.schemas.Datum`.")

        # validate annotations
        if not isinstance(self.annotations, list):
            raise TypeError(
                "annotations should be a list of `velour.schemas.Annotation`."
            )
        for i in range(len(self.annotations)):
            if isinstance(self.annotations[i], dict):
                self.annotations[i] = Annotation(**self.annotations[i])
            if not isinstance(self.annotations[i], Annotation):
                raise TypeError(
                    "annotations list should contain only `velour.schemas.Annotation`."
                )

        # validate dataset
        if not isinstance(self.dataset, str):
            raise TypeError("dataset should be type `str`.")


@dataclass
class Prediction:
    datum: Datum
    annotations: List[ScoredAnnotation] = field(default_factory=list)
    model: str = field(default="")

    def __post_init__(self):
        if isinstance(self.datum, dict):
            self.datum = Datum(**self.datum)
        if not isinstance(self.annotations, list):
            raise TypeError(
                "annotations should be a list of `velour.schemas.ScoredAnnotation`."
            )
        for i in range(len(self.annotations)):
            if isinstance(self.annotations[i], dict):
                self.annotations[i] = ScoredAnnotation(**self.annotations[i])

        if not isinstance(self.datum, Datum):
            raise TypeError("datum should be type `velour.schemas.Datum`.")
        for annotation in self.annotations:
            if not isinstance(annotation, ScoredAnnotation):
                raise TypeError(
                    "annotations list should contain only `velour.schemas.ScoredAnnotation`."
                )
        if not isinstance(self.model, str):
            raise TypeError("model should be type `str`.")
