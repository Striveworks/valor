import json
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from velour.client import Client, ClientException
from velour.enums import AnnotationType, EvaluationStatus, TaskType
from velour.schemas.evaluation import EvaluationParameters, EvaluationRequest
from velour.schemas.filters import Filter, FilterExpressionsType
from velour.schemas.geometry import BoundingBox, MultiPolygon, Polygon, Raster
from velour.schemas.metadata import (
    dump_metadata,
    load_metadata,
    validate_metadata,
)
from velour.schemas.properties import (
    DictionaryProperty,
    GeometryProperty,
    GeospatialProperty,
    LabelProperty,
    NumericProperty,
    StringProperty,
)
from velour.types import DictMetadataType, GeoJSONType, is_floating


class Label:
    """
    An object for labeling datasets, models, and annotations.

    Parameters
    ----------
    key : str
        The class key of the label.
    value : str
        The class value of the label.
    score : float, optional
        The score associated with the label (if applicable).

    Attributes
    ----------
    filter_by : filter_factory
        Declarative mappers used to create filters.
    """

    value = NumericProperty("value")
    key = StringProperty("label_keys")
    score = NumericProperty("prediction_scores")

    def __init__(
        self,
        key: str,
        value: str,
        score: Union[float, np.floating, None] = None,
    ):
        if not isinstance(key, str):
            raise TypeError("Attribute `key` should have type `str`.")
        if not isinstance(value, str):
            raise TypeError("Attribute `value` should have type `str`.")
        if score is not None:
            if not is_floating(score):
                raise TypeError(
                    "Attribute `score` should be a floating-point number or `None`."
                )

        self.key = key
        self.value = value
        self.score = score

    def __str__(self):
        return str(self.tuple())

    def to_dict(self):
        return {
            "key": self.key,
            "value": self.value,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, resp):
        return cls(**resp)

    def __eq__(self, other):
        """
        Defines how `Labels` are compared to one another

        Parameters
        ----------
        other : Label
            The object to compare with the `Label`.

        Returns
        ----------
        boolean
            A boolean describing whether the two objects are equal.
        """
        # type mismatch
        if type(other) is not type(self):
            return False

        # k,v mismatch
        if self.key != other.key or self.value != other.value:
            return False

        # score is None
        if self.score is None or other.score is None:
            return (other.score is None) == (self.score is None)

        # scores not equal
        if is_floating(self.score) and is_floating(other.score):
            return np.isclose(self.score, other.score)

        return False

    def __hash__(self) -> int:
        """
        Defines how a `Label` is hashed.

        Returns
        ----------
        int
            The hashed 'Label`.
        """
        return hash(f"key:{self.key},value:{self.value},score:{self.score}")

    def tuple(self) -> Tuple[str, str, Union[float, np.floating, None]]:
        """
        Defines how the `Label` is turned into a tuple.

        Returns
        ----------
        tuple
            A tuple of the `Label's` arguments.
        """
        return (self.key, self.value, self.score)


class Datum:
    """
    A class used to store datum about `GroundTruths` and `Predictions`.

    Parameters
    ----------
    uid : str
        The UID of the `Datum`.
    metadata : dict
        A dictionary of metadata that describes the `Datum`.
    geospatial :  dict
        A GeoJSON-style dictionary describing the geospatial coordinates of the `Datum`.
    """

    uid = StringProperty("datum_uids")
    metadata = DictionaryProperty("datum_metadata")
    geospatial = GeospatialProperty("datum_geospatial")

    def __init__(
        self,
        uid: str,
        metadata: Optional[DictMetadataType] = None,
        geospatial: Optional[GeoJSONType] = None,
    ):
        self.uid = uid
        self.metadata = metadata if metadata else {}
        self.geospatial = geospatial

        if not isinstance(self.uid, str):
            raise TypeError("Attribute `uid` should have type `str`.")
        validate_metadata(self.metadata)
        self.metadata = load_metadata(self.metadata)

    def __str__(self):
        return str(self.to_dict(dataset_name=None).pop("dataset_name"))

    def to_dict(self, dataset_name: Optional[str] = None) -> dict:
        """
        Defines how a `Datum` object is transformed into a dictionary.

        Returns
        ----------
        dict
            A dictionary of the `Datum's` attributes.
        """
        return {
            "dataset_name": dataset_name,
            "uid": self.uid,
            "metadata": dump_metadata(self.metadata),
            "geospatial": self.geospatial if self.geospatial else None,
        }

    @classmethod
    def from_dict(cls, resp: dict) -> "Datum":
        resp.pop("dataset_name", None)
        return cls(**resp)

    def __eq__(self, other):
        """
        Defines how `Datums` are compared to one another

        Parameters
        ----------
        other : Datum
            The object to compare with the `Datum`.

        Returns
        ----------
        boolean
            A boolean describing whether the two objects are equal.
        """
        if not isinstance(other, Datum):
            raise TypeError(f"Expected type `{type(Datum)}`, got `{other}`")
        return self.to_dict() == other.to_dict()


class Annotation:
    """
    A class used to annotate `GroundTruths` and `Predictions`.

    Parameters
    ----------
    task_type: TaskType
        The task type associated with the `Annotation`.
    labels: List[Label]
        A list of labels to use for the `Annotation`.
    metadata: Dict[str, Union[int, float, str, bool, datetime.datetime, datetime.date, datetime.time]]
        A dictionary of metadata that describes the `Annotation`.
    bounding_box: BoundingBox
        A bounding box to assign to the `Annotation`.
    polygon: Polygon
        A polygon to assign to the `Annotation`.
    multipolygon: MultiPolygon
        A multipolygon to assign to the `Annotation`.
    raster: Raster
        A raster to assign to the `Annotation`.

    Attributes
    ----------
    geometric_area : float
        The area of the annotation.

    Examples
    --------

    Classification
    >>> Annotation(
    ...     task_type=TaskType.CLASSIFICATION,
    ...     labels=[
    ...         Label(key="class", value="dog"),
    ...         Label(key="category", value="animal"),
    ...     ]
    ... )

    Object-Detection BoundingBox
    >>> annotation = Annotation(
    ...     task_type=TaskType.DETECTION,
    ...     labels=[Label(key="k1", value="v1")],
    ...     bounding_box=box2,
    ... )

    Object-Detection Polygon
    >>> annotation = Annotation(
    ...     task_type=TaskType.DETECTION,
    ...     labels=[Label(key="k1", value="v1")],
    ...     polygon=polygon1,
    ... )

    Object-Detection Mulitpolygon
    >>> annotation = Annotation(
    ...     task_type=TaskType.DETECTION,
    ...     labels=[Label(key="k1", value="v1")],
    ...     multipolygon=multipolygon,
    ... )

    Object-Detection Raster
    >>> annotation = Annotation(
    ...     task_type=TaskType.DETECTION,
    ...     labels=[Label(key="k1", value="v1")],
    ...     raster=raster1,
    ... )

    Semantic-Segmentation Raster
    >>> annotation = Annotation(
    ...     task_type=TaskType.SEGMENTATION,
    ...     labels=[Label(key="k1", value="v1")],
    ...     raster=raster1,
    ... )

    Defining all supported annotation-types for a given task_type is allowed!
    >>> Annotation(
    ...     task_type=TaskType.DETECTION,
    ...     labels=[Label(key="k1", value="v1")],
    ...     bounding_box=box1,
    ...     polygon=polygon1,
    ...     multipolygon=multipolygon,
    ...     raster=raster1,
    ... )
    """

    task_type = StringProperty("task_types")
    labels = LabelProperty("labels")
    metadata = DictionaryProperty("annotation_metadata")
    bounding_box = GeometryProperty("annotation_bounding_box")
    polygon = GeometryProperty("annotation_polygon")
    multipolygon = GeometryProperty("annotation_multipolygon")
    raster = GeometryProperty("annotation_raster")

    def __init__(
        self,
        task_type: TaskType,
        labels: List[Label],
        metadata: Optional[DictMetadataType] = None,
        bounding_box: Optional[BoundingBox] = None,
        polygon: Optional[Polygon] = None,
        multipolygon: Optional[MultiPolygon] = None,
        raster: Optional[Raster] = None,
    ):
        self.task_type = TaskType(task_type)
        self.labels = labels
        self.metadata = metadata if metadata else {}
        self.bounding_box = bounding_box
        self.polygon = polygon
        self.multipolygon = multipolygon
        self.raster = raster
        self._validate()

    def _validate(self):
        """
        Validates the parameters used to create a `Annotation` object.
        """
        # labels
        if not isinstance(self.labels, list):
            raise TypeError(
                "Attribute `labels` should have type `List[velour.Label]`."
            )
        for idx, label in enumerate(self.labels):
            if not isinstance(label, Label):
                raise TypeError(
                    f"Attribute `labels[{idx}]` should have type `velour.Label`."
                )

        # bounding box
        if self.bounding_box:
            if not isinstance(self.bounding_box, BoundingBox):
                raise TypeError(
                    "Attribute `bounding_box` should have type `velour.schemas.BoundingBox`."
                )

        # polygon
        if self.polygon:
            if not isinstance(self.polygon, Polygon):
                raise TypeError(
                    "Attribute `polygon` should have type `velour.schemas.Polygon`."
                )

        # multipolygon
        if self.multipolygon:
            if not isinstance(self.multipolygon, MultiPolygon):
                raise TypeError(
                    "Attribute `multipolygon` should have type `velour.schemas.MultiPolygon`."
                )

        # raster
        if self.raster:
            if not isinstance(self.raster, Raster):
                raise TypeError(
                    "Attribute `raster` should have type `velour.schemas.Raster`."
                )

        # metadata
        if not isinstance(self.metadata, dict):
            raise TypeError("Attribute `metadata` should have type `dict`.")
        validate_metadata(self.metadata)
        self.metadata = load_metadata(self.metadata)

    @classmethod
    def from_dict(cls, resp: dict):
        """
        Deserializes a `velour.Annotation` from a dictionary.

        Parameters
        ----------
        resp : dict
            The serialized annotation.

        Returns
        -------
        velour.Annotation
            The deserialized Annotation object.
        """

        task_type = TaskType(resp["task_type"])
        labels = [Label.from_dict(label) for label in resp["labels"]]
        metadata = load_metadata(resp["metadata"])

        bounding_box = None
        polygon = None
        multipolygon = None
        raster = None
        if "bounding_box" in resp:
            bounding_box = (
                BoundingBox(**resp["bounding_box"])
                if resp["bounding_box"]
                else None
            )
        if "polygon" in resp:
            polygon = Polygon(**resp["polygon"]) if resp["polygon"] else None
        if "multipolygon" in resp:
            multipolygon = (
                MultiPolygon(**resp["multipolygon"])
                if resp["multipolygon"]
                else None
            )
        if "raster" in resp:
            raster = Raster(**resp["raster"]) if resp["raster"] else None

        return cls(
            task_type=task_type,
            labels=labels,
            metadata=metadata,
            bounding_box=bounding_box,
            polygon=polygon,
            multipolygon=multipolygon,
            raster=raster,
        )

    def to_dict(self) -> dict:
        """
        Defines how a `velour.Annotation` object is serialized into a dictionary.

        Returns
        ----------
        dict
            A dictionary of the `Annotation's` attributes.
        """
        return {
            "task_type": self.task_type.value,
            "labels": [label.to_dict() for label in self.labels],
            "metadata": dump_metadata(self.metadata),
            "bounding_box": asdict(self.bounding_box)
            if self.bounding_box
            else None,
            "polygon": asdict(self.polygon) if self.polygon else None,
            "multipolygon": asdict(self.multipolygon)
            if self.multipolygon
            else None,
            "raster": asdict(self.raster) if self.raster else None,
        }

    def __str__(self):
        return str(self.to_dict())

    def __eq__(self, other):
        """
        Defines how `Annotations` are compared to one another

        Parameters
        ----------
        other : Annotation
            The object to compare with the `Annotation`.

        Returns
        ----------
        boolean
            A boolean describing whether the two objects are equal.
        """
        if not isinstance(other, Annotation):
            raise TypeError(
                f"Expected type `{type(Annotation)}`, got `{other}`"
            )
        return self.to_dict() == other.to_dict()


class GroundTruth:
    """
    An object describing a groundtruth (e.g., a human-drawn bounding box on an image).

    Parameters
    ----------
    datum : Datum
        The `Datum` associated with the `GroundTruth`.
    annotations : List[Annotation]
        The list of `Annotations` associated with the `GroundTruth`.
    """

    def __init__(self, datum: Datum, annotations: List[Annotation]):
        self.datum = datum
        self.annotations = annotations
        self._validate()

    def _validate(self):
        """
        Validate the inputs of the `GroundTruth`.
        """
        # validate datum
        if not isinstance(self.datum, Datum):
            raise TypeError(
                "Attribute `datum` should have type `velour.Datum`."
            )

        # validate annotations
        if not isinstance(self.annotations, list):
            raise TypeError(
                "Attribute `datum` should have type `List[velour.Annotation]`."
            )
        for idx, annotation in enumerate(self.annotations):
            if not isinstance(annotation, Annotation):
                raise TypeError(
                    f"Attribute `annotations[{idx}]` should have type `velour.Annotation`."
                )

    def to_dict(
        self,
        dataset_name: Optional[str] = None,
    ) -> dict:
        """
        Defines how a `GroundTruth` is transformed into a dictionary.

        Returns
        ----------
        dict
            A dictionary of the `GroundTruth's` attributes.
        """
        return {
            "datum": self.datum.to_dict(dataset_name),
            "annotations": [
                annotation.to_dict() for annotation in self.annotations
            ],
        }

    @classmethod
    def from_dict(cls, resp: dict):
        expected_keys = {"datum", "annotations"}
        if set(resp.keys()) != expected_keys:
            raise ValueError(
                f"Expected keys `{expected_keys}`, received `{set(resp.keys())}`."
            )
        if not isinstance(resp["annotations"], list):
            raise TypeError("Expected `annotations` member to be a `list`.")
        return cls(
            datum=Datum.from_dict(resp["datum"]),
            annotations=[
                Annotation.from_dict(annotation)
                for annotation in resp["annotations"]
            ],
        )

    def __str__(self):
        return str(self.to_dict(None))

    def __eq__(self, other):
        """
        Defines how `GroundTruths` are compared to one another

        Parameters
        ----------
        other : GroundTruth
            The object to compare with the `GroundTruth`.

        Returns
        ----------
        boolean
            A boolean describing whether the two objects are equal.
        """
        if not isinstance(other, GroundTruth):
            raise TypeError(
                f"Expected type `{type(GroundTruth)}`, got `{other}`"
            )
        return self.to_dict() == other.to_dict()


class Prediction:
    """
    An object describing a prediction (e.g., a machine-drawn bounding box on an image).

    Parameters
    ----------
    datum : Datum
        The `Datum` associated with the `Prediction`.
    annotations : List[Annotation]
        The list of `Annotations` associated with the `Prediction`.

    Attributes
    ----------
    score : Union[float, int]
        The score assigned to the `Prediction`.
    """

    def __init__(
        self,
        datum: Datum,
        annotations: List[Annotation],
    ):
        self.datum = datum
        self.annotations = annotations
        self._validate()

    def _validate(self):
        """
        Validate the inputs of the `Prediction`.
        """
        # validate datum
        if not isinstance(self.datum, Datum):
            raise TypeError(
                "Attribute `datum` should have type `velour.Datum`."
            )

        # validate annotations
        if not isinstance(self.annotations, list):
            raise TypeError(
                "Attribute `datum` should have type `List[velour.Annotation]`."
            )
        for idx, annotation in enumerate(self.annotations):
            if not isinstance(annotation, Annotation):
                raise TypeError(
                    f"Attribute `annotations[{idx}]` should have type `velour.Annotation`."
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

    def to_dict(
        self,
        dataset_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> dict:
        """
        Defines how a `Prediction` is transformed into a dictionary.

        Returns
        ----------
        dict
            A dictionary of the `Prediction's` attributes.
        """
        return {
            "datum": self.datum.to_dict(dataset_name=dataset_name),
            "model_name": model_name,
            "annotations": [
                annotation.to_dict() for annotation in self.annotations
            ],
        }

    @classmethod
    def from_dict(cls, resp: dict):
        expected_keys = {"datum", "annotations", "model_name"}
        if set(resp.keys()) != expected_keys:
            raise ValueError(
                f"Expected keys `{expected_keys}`, received `{set(resp.keys())}`."
            )
        if not isinstance(resp["annotations"], list):
            raise TypeError("Expected `annotations` member to be a `list`.")
        return cls(
            datum=Datum.from_dict(resp["datum"]),
            annotations=[
                Annotation.from_dict(annotation)
                for annotation in resp["annotations"]
            ],
        )

    def __str__(self):
        return str(self.to_dict(None, None))

    def __eq__(self, other):
        """
        Defines how `Predictions` are compared to one another

        Parameters
        ----------
        other : Prediction
            The object to compare with the `Prediction`.

        Returns
        ----------
        boolean
            A boolean describing whether the two objects are equal.
        """
        if not isinstance(other, Prediction):
            raise TypeError(
                f"Expected type `{type(Prediction)}`, got `{other}`"
            )
        return self.to_dict(None, None) == other.to_dict(None, None)


class Evaluation:
    """
    Wraps `velour.client.Job` to provide evaluation-specifc members.
    """

    def __init__(self, client: Client, *_, **kwargs):
        """
        Defines important attributes of the API's `EvaluationResult`.

        Attributes
        ----------
        id : int
            The id of the evaluation.
        model_name : str
            The name of the evaluated model.
        datum_filter : schemas.Filter
            The filter used to select the datums for evaluation.
        status : EvaluationStatus
            The status of the evaluation.
        metrics : List[dict]
            A list of metric dictionaries returned by the job.
        confusion_matrices : List[dict]
            A list of confusion matrix dictionaries returned by the job.
        """
        self.client = client
        self.update(**kwargs)

    def dict(self):
        return {
            "id": self.id,
            "model_name": self.model_name,
            "datum_filter": asdict(self.datum_filter),
            "parameters": asdict(self.parameters),
            "status": self.status.value,
            "metrics": self.metrics,
            "confusion_matrices": self.confusion_matrices,
            **self.kwargs,
        }

    def update(
        self,
        *_,
        id: int,
        model_name: str,
        datum_filter: Filter,
        parameters: EvaluationParameters,
        status: EvaluationStatus,
        metrics: List[Dict],
        confusion_matrices: List[Dict],
        **kwargs,
    ):
        self.id = id
        self.model_name = model_name
        self.datum_filter = (
            Filter(**datum_filter)
            if isinstance(datum_filter, dict)
            else datum_filter
        )
        self.parameters = (
            EvaluationParameters(**parameters)
            if isinstance(parameters, dict)
            else parameters
        )
        self.status = EvaluationStatus(status)
        self.metrics = metrics
        self.confusion_matrices = confusion_matrices
        self.kwargs = kwargs
        self.ignored_pred_labels: Optional[List[Label]] = None
        self.missing_pred_labels: Optional[List[Label]] = None
        self.ignored_pred_keys: Optional[List[str]] = None
        self.missing_pred_keys: Optional[List[str]] = None

        for k, v in kwargs.items():
            setattr(self, k, v)

    def poll(self) -> EvaluationStatus:
        """
        Poll the backend.

        Updates the evaluation with the latest state from the backend.

        Returns
        -------
        enums.EvaluationStatus
            The status of the evaluation.

        Raises
        ----------
        ClientException
            If an Evaluation with the given `evaluation_id` is not found.
        """
        response = self.client.get_evaluations(evaluation_ids=[self.id])
        if not response:
            raise ClientException("Not Found")
        self.update(**response[0])
        return self.status

    def wait_for_completion(
        self,
        *,
        timeout: Optional[int] = None,
        interval: float = 1.0,
    ) -> EvaluationStatus:
        """
        Blocking function that waits for evaluation to finish.

        Parameters
        ----------
        timeout : int, optional
            Length of timeout in seconds.
        interval : float, default=1.0
            Polling interval in seconds.
        """
        t_start = time.time()
        while self.poll() not in [
            EvaluationStatus.DONE,
            EvaluationStatus.FAILED,
        ]:
            time.sleep(interval)
            if timeout and time.time() - t_start > timeout:
                raise TimeoutError
        return self.status

    def to_dataframe(
        self,
        stratify_by: Optional[Tuple[str, str]] = None,
    ):
        """
        Get all metrics associated with a Model and return them in a `pd.DataFrame`.

        Returns
        ----------
        pd.DataFrame
            Evaluation metrics being displayed in a `pd.DataFrame`.

        Raises
        ------
        ModuleNotFoundError
            This function requires the use of `pandas.DataFrame`.

        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Must have pandas installed to use `get_metric_dataframes`."
            )

        if not stratify_by:
            column_type = "evaluation"
            column_name = self.id
        else:
            column_type = stratify_by[0]
            column_name = stratify_by[1]

        metrics = [
            {**metric, column_type: column_name} for metric in self.metrics
        ]
        df = pd.DataFrame(metrics)
        for k in ["label", "parameters"]:
            df[k] = df[k].fillna("n/a")
        df["parameters"] = df["parameters"].apply(json.dumps)
        df["label"] = df["label"].apply(
            lambda x: f"{x['key']}: {x['value']}" if x != "n/a" else x
        )
        df = df.pivot(
            index=["type", "parameters", "label"], columns=[column_type]
        )
        return df


@dataclass
class DatasetSummary:
    """Dataclass for storing dataset summary information"""

    name: str
    num_datums: int
    num_annotations: int
    num_bounding_boxes: int
    num_polygons: int
    num_groundtruth_multipolygons: int
    num_rasters: int
    task_types: List[TaskType]
    labels: List[Label]
    datum_metadata: List[DictMetadataType]
    annotation_metadata: List[DictMetadataType]

    def __post_init__(self):
        for i, tt in enumerate(self.task_types):
            if isinstance(tt, str):
                self.task_types[i] = TaskType(tt)
        for i, label in enumerate(self.labels):
            if isinstance(label, dict):
                self.labels[i] = Label(**label)


class Dataset:
    """
    A class describing a given dataset.

    Attribute
    ----------
    client : Client
        The `Client` object associated with the session.
    id : int
        The ID of the dataset.
    name : str
        The name of the dataset.
    metadata : dict
        A dictionary of metadata that describes the dataset.
    geospatial :  dict
        A GeoJSON-style dictionary describing the geospatial coordinates of the dataset.
    """

    name = StringProperty("dataset_names")
    metadata = DictionaryProperty("dataset_metadata")
    geospatial = GeospatialProperty("dataset_geospatial")

    def __init__(
        self,
        client: Client,
        name: str,
        metadata: Optional[DictMetadataType] = None,
        geospatial: Optional[GeoJSONType] = None,
        delete_if_exists: bool = False,
    ):
        """
        Create or get a `Dataset` object.

        Parameters
        ----------
        client : Client
            The `Client` object associated with the session.
        name : str
            The name of the dataset.
        metadata : dict
            A dictionary of metadata that describes the dataset.
        geospatial : dict
            A GeoJSON-style dictionary describing the geospatial coordinates of the dataset.
        delete_if_exists : bool, default=False
            Deletes any existing dataset with the same name.
        """
        self.name = name
        self.metadata = metadata if metadata else {}
        self.geospatial = geospatial

        # validation
        if not isinstance(self.name, str):
            raise TypeError("`name` should be of type `str`")
        validate_metadata(self.metadata)
        self.metadata = load_metadata(self.metadata)

        if delete_if_exists and client.get_dataset(name) is not None:
            client.delete_dataset(name, timeout=30)

        if delete_if_exists or client.get_dataset(name) is None:
            client.create_dataset(self.to_dict())

        for k, v in (client.get_dataset(name) or {}).items():
            setattr(self, k, v)
        self.client = client

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self, id: Optional[int] = None) -> dict:
        """
        Defines how a `Dataset` object is transformed into a dictionary.

        Returns
        ----------
        dict
            A dictionary of the `Dataset's` attributes.
        """
        return {
            "id": id,
            "name": self.name,
            "metadata": dump_metadata(self.metadata),
            "geospatial": self.geospatial,
        }

    def add_groundtruth(
        self,
        groundtruth: GroundTruth,
    ):
        """
        Add a groundtruth to a given dataset.

        Parameters
        ----------
        groundtruth : GroundTruth
            The `GroundTruth` object to add to the `Dataset`.
        """
        if not isinstance(groundtruth, GroundTruth):
            raise TypeError(f"Invalid type `{type(groundtruth)}`")

        if len(groundtruth.annotations) == 0:
            warnings.warn(
                f"GroundTruth for datum with uid `{groundtruth.datum.uid}` contains no annotations."
            )

        self.client._requests_post_rel_host(
            "groundtruths",
            json=groundtruth.to_dict(self.name),
        )

    def get_groundtruth(self, datum: Union[Datum, str]) -> GroundTruth:
        """
        Fetches a given groundtruth from the backend.

        Parameters
        ----------
        datum : Datum
            The Datum of the 'GroundTruth' to fetch.


        Returns
        ----------
        GroundTruth
            The requested `GroundTruth`.
        """
        uid = datum.uid if isinstance(datum, Datum) else datum
        resp = self.client._requests_get_rel_host(
            f"groundtruths/dataset/{self.name}/datum/{uid}"
        ).json()
        return GroundTruth.from_dict(resp)

    def get_labels(
        self,
    ) -> List[Label]:
        """
        Get all labels associated with a given dataset.

        Returns
        ----------
        List[Label]
            A list of `Labels` associated with the dataset.
        """
        labels = self.client._requests_get_rel_host(
            f"labels/dataset/{self.name}"
        ).json()

        return [
            Label(key=label["key"], value=label["value"]) for label in labels
        ]

    def get_datums(self) -> List[Datum]:
        """
        Get all datums associated with a given dataset.

        Returns
        ----------
        List[Datum]
            A list of `Datums` associated with the dataset.
        """
        datums = self.client.get_datums(
            filters=Filter(dataset_names=[self.name])
        )
        return [Datum.from_dict(datum) for datum in datums]

    def get_evaluations(
        self,
    ) -> List[Evaluation]:
        """
        Get all evaluations associated with a given dataset.

        Returns
        ----------
        List[Evaluation]
            A list of `Evaluations` associated with the dataset.
        """
        return [
            Evaluation(self.client, **resp)
            for resp in self.client.get_evaluations(datasets=self.name)
        ]

    def get_summary(self) -> DatasetSummary:
        """
        Get the summary of a given dataset.

        Returns
        -------
        DatasetSummary
            The summary of the dataset. This class has the following fields:

            name: name of the dataset

            num_datums: total number of datums in the dataset

            num_annotations: total number of labeled annotations in the dataset. if an
            object (such as a bounding box) has multiple labels then each label is counted separately

            num_bounding_boxes: total number of bounding boxes in the dataset

            num_polygons: total number of polygons in the dataset

            num_groundtruth_multipolygons: total number of multipolygons in the dataset

            num_rasters: total number of rasters in the dataset

            task_types: list of the unique task types in the dataset

            labels: list of the unique labels in the dataset

            datum_metadata: list of the unique metadata dictionaries in the dataset that are associated
            to datums

            groundtruth_annotation_metadata: list of the unique metadata dictionaries in the dataset that are
            associated to annotations
        """
        resp = self.client.get_dataset_summary(self.name)
        return DatasetSummary(**resp)

    def finalize(
        self,
    ):
        """
        Finalize the `Dataset` object such that new `GroundTruths` cannot be added to it.
        """
        return self.client._requests_put_rel_host(
            f"datasets/{self.name}/finalize"
        )

    def delete(
        self,
    ):
        """
        Delete the `Dataset` object from the backend.
        """
        self.client._requests_delete_rel_host(f"datasets/{self.name}").json()


class Model:
    """
    A class describing a model that was trained on a particular dataset.

    Attribute
    ----------
    client : Client
        The `Client` object associated with the session.
    id : int
        The ID of the model.
    name : str
        The name of the model.
    metadata : dict
        A dictionary of metadata that describes the model.
    geospatial :  dict
        A GeoJSON-style dictionary describing the geospatial coordinates of the model.
    """

    name = StringProperty("model_names")
    metadata = DictionaryProperty("model_metadata")
    geospatial = GeospatialProperty("model_geospatial")

    def __init__(
        self,
        client: Client,
        name: str,
        metadata: Optional[DictMetadataType] = None,
        geospatial: Optional[GeoJSONType] = None,
        delete_if_exists: bool = False,
    ):
        """
        Create or get a `Model` object.

        Parameters
        ----------
        client : Client
            The `Client` object associated with the session.
        name : str
            The name of the model.
        metadata : dict
            A dictionary of metadata that describes the model.
        geospatial : dict
            A GeoJSON-style dictionary describing the geospatial coordinates of the model.
        delete_if_exists : bool, default=False
            Deletes any existing model with the same name.
        """
        self.name = name
        self.metadata = metadata if metadata else {}
        self.geospatial = geospatial

        # validation
        if not isinstance(self.name, str):
            raise TypeError("`name` should be of type `str`")
        validate_metadata(self.metadata)
        self.metadata = load_metadata(self.metadata)

        if delete_if_exists and client.get_model(name) is not None:
            client.delete_model(name, timeout=30)

        if delete_if_exists or client.get_model(name) is None:
            client.create_model(self.to_dict())

        for k, v in (client.get_model(name) or {}).items():
            setattr(self, k, v)
        self.client = client

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self, id: Optional[int] = None) -> dict:
        """
        Defines how a `Model` object is transformed into a dictionary.

        Returns
        ----------
        dict
            A dictionary of the `Model's` attributes.
        """
        return {
            "id": id,
            "name": self.name,
            "metadata": dump_metadata(self.metadata),
            "geospatial": self.geospatial,
        }

    def add_prediction(
        self, dataset: Union[Dataset, str], prediction: Prediction
    ):
        """
        Add a prediction to a given model.

        Parameters
        ----------
        prediction : Prediction
            The `Prediction` object to add to the `Model`.
        """
        if not isinstance(prediction, Prediction):
            raise TypeError(
                f"Expected `velour.Prediction`, got `{type(prediction)}`"
            )

        if len(prediction.annotations) == 0:
            warnings.warn(
                f"Prediction for datum with uid `{prediction.datum.uid}` contains no annotations."
            )

        dataset_name = (
            dataset.name if isinstance(dataset, Dataset) else dataset
        )
        return self.client._requests_post_rel_host(
            "predictions",
            json=prediction.to_dict(
                dataset_name=dataset_name, model_name=self.name
            ),
        )

    def get_prediction(self, dataset: Dataset, datum: Datum) -> Prediction:
        """
        Fetch a particular prediction.

        Parameters
        ----------
        datum : Union[Datum, str]
            The `Datum` or datum UID of the prediction to return.

        Returns
        ----------
        Prediction
            The requested `Prediction`.
        """
        resp = self.client._requests_get_rel_host(
            f"predictions/model/{self.name}/dataset/{dataset.name}/datum/{datum.uid}",
        ).json()
        return Prediction.from_dict(resp)

    def finalize_inferences(self, dataset: "Dataset") -> None:
        """
        Finalize the `Model` object such that new `Predictions` cannot be added to it.
        """
        return self.client._requests_put_rel_host(
            f"models/{self.name}/datasets/{dataset.name}/finalize"
        ).json()

    def _format_filters(
        self,
        datasets: Optional[Union[Dataset, List[Dataset]]],
        filters: Optional[Union[Dict, FilterExpressionsType]],
    ) -> Filter:
        """Formats evaluation request's `datum_filter` input."""

        # get list of dataset names
        dataset_names_from_obj = []
        if isinstance(datasets, list):
            dataset_names_from_obj = [dataset.name for dataset in datasets]
        elif isinstance(datasets, Dataset):
            dataset_names_from_obj = [datasets.name]

        # format filtering object
        if isinstance(filters, Sequence) or filters is None:
            filters = filters if filters else []
            filter_obj = Filter.create(filters)

            # reset model name
            filter_obj.model_names = None
            filter_obj.model_geospatial = None
            filter_obj.model_metadata = None

            # set dataset names
            if not filter_obj.dataset_names:
                filter_obj.dataset_names = []
            filter_obj.dataset_names.extend(dataset_names_from_obj)
            return filter_obj

        elif isinstance(filters, dict):
            # reset model name
            filters["model_names"] = None
            filters["model_geospatial"] = None
            filters["model_metadata"] = None

            # set dataset names
            if (
                "dataset_names" not in filters
                or filters["dataset_names"] is None
            ):
                filters["dataset_names"] = []
            filters["dataset_names"].extend(dataset_names_from_obj)

        return Filter(**filters)

    def _create_label_map(
        self,
        label_map: Optional[Dict[Label, Label]] = None,
    ) -> Union[List[List[List[str]]], None]:
        """Convert a dictionary of label maps to a serializable list format."""
        if not label_map:
            return None

        if not isinstance(label_map, dict) or not all(
            [
                isinstance(key, Label) and isinstance(value, Label)
                for key, value in label_map.items()
            ]
        ):
            raise TypeError(
                "label_map should be a dictionary with valid Labels for both the key and value."
            )

        return_value = [
            [[key.key, key.value], [value.key, value.value]]
            for key, value in label_map.items()
        ]

        return return_value

    def evaluate_classification(
        self,
        datasets: Optional[Union[Dataset, List[Dataset]]] = None,
        filters: Optional[Union[Dict, FilterExpressionsType]] = None,
        label_map: Optional[Dict[Label, Label]] = None,
    ) -> Evaluation:
        """
        Start a classification evaluation job.

        Parameters
        ----------
        datasets : Union[Dataset, List[Dataset]], optional
            The dataset or list of datasets to evaluate against.
        filters : Union[Dict, FilterExpressionsType = Sequence[Union[BinaryExpression, Sequence[BinaryExpression]]]], optional
            Optional set of filters to constrain evaluation by.
        label_map : Dict[Label, Label]
            Optional mapping of individual Labels to a grouper Label. Useful when you need to evaluate performance using Labels that differ across datasets and models.

        Returns
        -------
        Evaluation
            A job object that can be used to track the status of the job and get the metrics of it upon completion.
        """
        if not datasets and not filters:
            raise ValueError(
                "Evaluation requires the definition of either datasets, dataset filters or both."
            )

        datum_filter = self._format_filters(datasets, filters)

        evaluation = EvaluationRequest(
            model_names=self.name,
            datum_filter=datum_filter,
            parameters=EvaluationParameters(
                task_type=TaskType.CLASSIFICATION,
                label_map=self._create_label_map(label_map=label_map),
            ),
        )
        resp = self.client.evaluate(evaluation)
        if len(resp) != 1:
            raise RuntimeError
        resp = resp[0]

        # resp should have keys "missing_pred_keys", "ignored_pred_keys", with values
        # list of label dicts. convert label dicts to Label objects

        evaluation_job = Evaluation(
            client=self.client,
            **resp,
        )

        return evaluation_job

    def evaluate_detection(
        self,
        datasets: Optional[Union[Dataset, List[Dataset]]] = None,
        filters: Optional[Union[Dict, FilterExpressionsType]] = None,
        convert_annotations_to_type: Optional[AnnotationType] = None,
        iou_thresholds_to_compute: Optional[List[float]] = None,
        iou_thresholds_to_return: Optional[List[float]] = None,
        label_map: Optional[Dict[Label, Label]] = None,
    ) -> Evaluation:
        """
        Start a object-detection evaluation job.

        Parameters
        ----------
        datasets : Union[Dataset, List[Dataset]], optional
            The dataset or list of datasets to evaluate against.
        filters : Union[Dict, FilterExpressionsType = Sequence[Union[BinaryExpression, Sequence[BinaryExpression]]]], optional
            Optional set of filters to constrain evaluation by.
        convert_annotations_to_type : enums.AnnotationType, optional
            Forces the object detection evaluation to compute over this type.
        iou_thresholds_to_compute : List[float], optional
            Thresholds to compute mAP against.
        iou_thresholds_to_return : List[float], optional
            Thresholds to return AP for. Must be subset of `iou_thresholds_to_compute`.
        label_map : Dict[Label, Label]
            Optional mapping of individual Labels to a grouper Label. Useful when you need to evaluate performance using Labels that differ across datasets and models.

        Returns
        -------
        Evaluation
            A job object that can be used to track the status of the job and get the metrics of it upon completion.
        """
        if iou_thresholds_to_compute is None:
            iou_thresholds_to_compute = [
                round(0.5 + 0.05 * i, 2) for i in range(10)
            ]
        if iou_thresholds_to_return is None:
            iou_thresholds_to_return = [0.5, 0.75]

        parameters = EvaluationParameters(
            task_type=TaskType.DETECTION,
            convert_annotations_to_type=convert_annotations_to_type,
            iou_thresholds_to_compute=iou_thresholds_to_compute,
            iou_thresholds_to_return=iou_thresholds_to_return,
            label_map=self._create_label_map(label_map=label_map),
        )

        datum_filter = self._format_filters(datasets, filters)

        evaluation = EvaluationRequest(
            model_names=self.name,
            datum_filter=datum_filter,
            parameters=parameters,
        )

        resp = self.client.evaluate(evaluation)
        if len(resp) != 1:
            raise RuntimeError
        resp = resp[0]

        # resp should have keys "missing_pred_labels", "ignored_pred_labels", with values
        # list of label dicts. convert label dicts to Label objects

        evaluation_job = Evaluation(
            client=self.client,
            **resp,
        )

        return evaluation_job

    def evaluate_segmentation(
        self,
        datasets: Optional[Union[Dataset, List[Dataset]]] = None,
        filters: Optional[Union[Dict, FilterExpressionsType]] = None,
        label_map: Optional[Dict[Label, Label]] = None,
    ) -> Evaluation:
        """
        Start a semantic-segmentation evaluation job.

        Parameters
        ----------
        datasets : Union[Dataset, List[Dataset]], optional
            The dataset or list of datasets to evaluate against.
        filters : Union[Dict, FilterExpressionsType = Sequence[Union[BinaryExpression, Sequence[BinaryExpression]]]], optional
            Optional set of filters to constrain evaluation by.
        label_map : Dict[Label, Label]
            Optional mapping of individual Labels to a grouper Label. Useful when you need to evaluate performance using Labels that differ across datasets and models.


        Returns
        -------
        Evaluation
            a job object that can be used to track the status of the job and get the metrics of it upon completion
        """

        datum_filter = self._format_filters(datasets, filters)

        # create evaluation job
        evaluation = EvaluationRequest(
            model_names=self.name,
            datum_filter=datum_filter,
            parameters=EvaluationParameters(
                task_type=TaskType.SEGMENTATION,
                label_map=self._create_label_map(label_map=label_map),
            ),
        )
        resp = self.client.evaluate(evaluation)
        if len(resp) != 1:
            raise RuntimeError
        resp = resp[0]

        # resp should have keys "missing_pred_labels", "ignored_pred_labels", with values
        # list of label dicts. convert label dicts to Label objects

        # create client-side evaluation handler
        evaluation_job = Evaluation(
            client=self.client,
            **resp,
        )

        return evaluation_job

    def delete(
        self,
    ):
        """
        Delete the `Model` object from the backend.
        """
        self.client._requests_delete_rel_host(f"models/{self.name}").json()

    def get_labels(
        self,
    ) -> List[Label]:
        """
        Get all labels associated with a given model.

        Returns
        ----------
        List[Label]
            A list of `Labels` associated with the model.
        """
        labels = self.client._requests_get_rel_host(
            f"labels/model/{self.name}"
        ).json()

        return [
            Label(key=label["key"], value=label["value"]) for label in labels
        ]

    def get_evaluations(
        self,
    ) -> List[Evaluation]:
        """
        Get all evaluations associated with a given model.

        Returns
        ----------
        List[Evaluation]
            A list of `Evaluations` associated with the model.
        """
        return [
            Evaluation(self.client, **resp)
            for resp in self.client.get_evaluations(models=self.name)
        ]
