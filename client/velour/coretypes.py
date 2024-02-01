import json
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from velour.client import ClientConnection, connect, get_connection
from velour.enums import (
    AnnotationType,
    EvaluationStatus,
    TableStatus,
    TaskType,
)
from velour.exceptions import ClientException
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
from velour.types import GeoJSONType, MetadataType, is_floating


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

    value = StringProperty("value")
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
        metadata: Optional[MetadataType] = None,
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
        if self.bounding_box is not None:
            if not isinstance(self.bounding_box, BoundingBox):
                raise TypeError(
                    "Attribute `bounding_box` should have type `velour.schemas.BoundingBox`."
                )

        # polygon
        if self.polygon is not None:
            if not isinstance(self.polygon, Polygon):
                raise TypeError(
                    "Attribute `polygon` should have type `velour.schemas.Polygon`."
                )

        # multipolygon
        if self.multipolygon is not None:
            if not isinstance(self.multipolygon, MultiPolygon):
                raise TypeError(
                    "Attribute `multipolygon` should have type `velour.schemas.MultiPolygon`."
                )

        # raster
        if self.raster is not None:
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
        metadata: Optional[MetadataType] = None,
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

    def __init__(
        self, connection: Optional[ClientConnection] = None, **kwargs
    ):
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
        if not connection:
            connection = get_connection()
        self.conn = connection
        self.update(**kwargs)

    def to_dict(self):
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
        response = self.conn.get_evaluations(evaluation_ids=[self.id])
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
    datum_metadata: List[MetadataType]
    annotation_metadata: List[MetadataType]

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

    Parameters
    ----------
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
        name: str,
        metadata: Optional[MetadataType] = None,
        geospatial: Optional[GeoJSONType] = None,
        connection: Optional[ClientConnection] = None,
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
        self.conn = connection
        self.name = name
        self.metadata = metadata if metadata else {}
        self.geospatial = geospatial

        # validation
        if not isinstance(self.name, str):
            raise TypeError("`name` should be of type `str`")
        validate_metadata(self.metadata)
        self.metadata = load_metadata(self.metadata)

    @classmethod
    def create(
        cls,
        name: str,
        metadata: Optional[MetadataType] = None,
        geospatial: Optional[GeoJSONType] = None,
    ):
        """
        Creates a dataset that persists in the backend.

        Parameters
        ----------
        name : str
            The name of the dataset.
        metadata : dict
            A dictionary of metadata that describes the dataset.
        geospatial :  dict
            A GeoJSON-style dictionary describing the geospatial coordinates of the dataset.

        Returns
        -------
        velour.Dataset
            The created dataset.
        """
        dataset = cls(
            name=name,
            metadata=metadata,
            geospatial=geospatial,
        )
        Client(dataset.conn).create_dataset(dataset)
        return dataset

    @classmethod
    def get(
        cls,
        name: str,
    ):
        """
        Retrieves a dataset from the backend database.

        Parameters
        ----------
        dataset : str
            The name of the dataset.

        Returns
        -------
        Union[velour.Dataset, None]
            The dataset or 'None' if it doesn't exist.
        """
        return Client().get_dataset(name)

    @classmethod
    def from_dict(
        cls, resp: dict, connection: Optional[ClientConnection] = None
    ):
        """
        Construct a dataset from a dictionary.

        Parameters
        ----------
        resp : dict
            The dictionary containing a dataset definition.
        connection : ClientConnection, optional
            Option to share a ClientConnection rather than request a new one.

        Returns
        -------
        velour.Dataset
        """
        resp.pop("id")
        return cls(**resp, connection=connection)

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

    def __str__(self):
        return str(self.to_dict())

    def add_groundtruth(
        self,
        groundtruth: GroundTruth,
    ) -> None:
        """
        Add a groundtruth to the dataset.

        Parameters
        ----------
        groundtruth : GroundTruth
            The groundtruth to create.
        """
        if not isinstance(groundtruth, GroundTruth):
            raise TypeError(f"Invalid type `{type(groundtruth)}`")

        if len(groundtruth.annotations) == 0:
            warnings.warn(
                f"GroundTruth for datum with uid `{groundtruth.datum.uid}` contains no annotations."
            )

        Client(self.conn).create_groundtruth(self, groundtruth)

    def get_groundtruth(
        self,
        datum: Union[Datum, str],
    ) -> Union[GroundTruth, None]:
        """
        Get a particular groundtruth.

        Parameters
        ----------
        datum: Union[Datum, str]
            The desired datum.

        Returns
        ----------
        Union[GroundTruth, None]
            The matching groundtruth or 'None' if it doesn't exist.
        """
        return Client(self.conn).get_groundtruth(dataset=self, datum=datum)

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
        return Client(self.conn).get_labels_from_dataset(self)

    def get_datums(self) -> List[Datum]:
        """
        Get all datums associated with a given dataset.

        Returns
        ----------
        List[Datum]
            A list of `Datums` associated with the dataset.
        """
        return Client(self.conn).get_datums(
            filter_=Filter(dataset_names=[self.name])
        )

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
        return Client(self.conn).get_evaluations(datasets=[self])

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
        return Client(self.conn).get_dataset_summary(self.name)

    def finalize(
        self,
    ):
        """
        Finalizes the dataset such that new groundtruths cannot be added to it.
        """
        return Client(self.conn).finalize_dataset(self)

    def delete(
        self,
        timeout: int = 0,
    ):
        """
        Delete the dataset from the backend.

        Parameters
        ----------
        timeout : int, default=0
            Sets a timeout in seconds.
        """
        Client(self.conn).delete_dataset(self.name, timeout)


class Model:
    """
    A class describing a model that was trained on a particular dataset.

    Parameters
    ----------
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
        name: str,
        metadata: Optional[MetadataType] = None,
        geospatial: Optional[GeoJSONType] = None,
        connection: Optional[ClientConnection] = None,
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
        self.conn = connection
        self.name = name
        self.metadata = metadata if metadata else {}
        self.geospatial = geospatial

        # validation
        if not isinstance(self.name, str):
            raise TypeError("`name` should be of type `str`")
        validate_metadata(self.metadata)
        self.metadata = load_metadata(self.metadata)

    @classmethod
    def create(
        cls,
        name: str,
        metadata: Optional[MetadataType] = None,
        geospatial: Optional[GeoJSONType] = None,
    ):
        """
        Creates a model that persists in the backend.

        Parameters
        ----------
        name : str
            The name of the model.
        metadata : dict
            A dictionary of metadata that describes the model.
        geospatial :  dict
            A GeoJSON-style dictionary describing the geospatial coordinates of the model.

        Returns
        -------
        velour.Model
            The created model.
        """
        model = cls(
            name=name,
            metadata=metadata,
            geospatial=geospatial,
        )
        Client(model.conn).create_model(model)
        return model

    @classmethod
    def get(
        cls,
        name: str,
    ):
        """
        Retrieves a model from the backend database.

        Parameters
        ----------
        name : str
            The name of the model.

        Returns
        -------
        Union[velour.Model, None]
            The model or 'None' if it doesn't exist.
        """
        return Client().get_model(name)

    @classmethod
    def from_dict(
        cls, resp: dict, connection: Optional[ClientConnection] = None
    ):
        """
        Construct a model from a dictionary.

        Parameters
        ----------
        resp : dict
            The dictionary containing a model definition.
        connection : ClientConnection, optional
            Option to share a ClientConnection rather than request a new one.

        Returns
        velour.Model
        """
        resp.pop("id")
        return cls(**resp, connection=connection)

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

    def __str__(self):
        return str(self.to_dict())

    def add_prediction(
        self,
        dataset: Union[Dataset, str],
        prediction: Prediction,
    ) -> None:
        """
        Add a prediction to the model.

        Parameters
        ----------
        dataset : velour.Dataset
            The dataset that is being operated over.
        prediction : velour.Prediction
            The prediction to create.
        """

        if not isinstance(prediction, Prediction):
            raise TypeError(f"Invalid type `{type(prediction)}`")

        if len(prediction.annotations) == 0:
            warnings.warn(
                f"Prediction for datum with uid `{prediction.datum.uid}` contains no annotations."
            )

        Client(self.conn).create_prediction(
            dataset=dataset, model=self, prediction=prediction
        )

    def get_prediction(
        self, dataset: Union[Dataset, str], datum: Union[Datum, str]
    ) -> Union[Prediction, None]:
        """
        Get a particular prediction.

        Parameters
        ----------
        dataset: Union[Dataset, str]
            The dataset the datum belongs to.
        datum: Union[Datum, str]
            The desired datum.

        Returns
        ----------
        Union[Prediction, None]
            The matching prediction or 'None' if it doesn't exist.
        """
        return Client(self.conn).get_prediction(
            dataset=dataset, model=self, datum=datum
        )

    def finalize_inferences(self, dataset: Union[Dataset, str]) -> None:
        """
        Finalizes the model over a dataset such that new prediction cannot be added to it.
        """
        if isinstance(dataset, Dataset):
            dataset = dataset.name
        return Client(self.conn).finalize_inferences(
            dataset=dataset, model=self
        )

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
        self, label_map: Optional[Dict[Label, Label]]
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

        # format request
        datum_filter = self._format_filters(datasets, filters)
        request = EvaluationRequest(
            model_names=self.name,
            datum_filter=datum_filter,
            parameters=EvaluationParameters(
                task_type=TaskType.CLASSIFICATION,
                label_map=self._create_label_map(label_map=label_map),
            ),
        )

        # create evaluation
        evaluation = Client(self.conn).evaluate(request)
        if len(evaluation) != 1:
            raise RuntimeError
        return evaluation[0]

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

        # format request
        parameters = EvaluationParameters(
            task_type=TaskType.DETECTION,
            convert_annotations_to_type=convert_annotations_to_type,
            iou_thresholds_to_compute=iou_thresholds_to_compute,
            iou_thresholds_to_return=iou_thresholds_to_return,
            label_map=self._create_label_map(label_map=label_map),
        )
        datum_filter = self._format_filters(datasets, filters)
        request = EvaluationRequest(
            model_names=self.name,
            datum_filter=datum_filter,
            parameters=parameters,
        )

        # create evaluation
        evaluation = Client(self.conn).evaluate(request)
        if len(evaluation) != 1:
            raise RuntimeError
        return evaluation[0]

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
        # format request
        datum_filter = self._format_filters(datasets, filters)
        request = EvaluationRequest(
            model_names=self.name,
            datum_filter=datum_filter,
            parameters=EvaluationParameters(
                task_type=TaskType.SEGMENTATION,
                label_map=self._create_label_map(label_map=label_map),
            ),
        )

        # create evaluation
        evaluation = Client(self.conn).evaluate(request)
        if len(evaluation) != 1:
            raise RuntimeError
        return evaluation[0]

    def delete(self, timeout: int = 0):
        """
        Delete the `Model` object from the backend.

        Parameters
        ----------
        timeout : int, default=0
            Sets a timeout in seconds.
        """
        Client(self.conn).delete_model(self.name, timeout)

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
        return Client(self.conn).get_labels_from_model(self)

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
        return Client(self.conn).get_evaluations(models=[self])


class Client:
    """
    Velour client object for interacting with the api.

    Parameters
    ----------
    connection : ClientConnection, optional
        Option to use an existing connection object.
    """

    def __init__(self, connection: Optional[ClientConnection] = None):
        if not connection:
            connection = get_connection()
        self.conn = connection

    @classmethod
    def connect(
        cls,
        host: str,
        access_token: Optional[str] = None,
        reconnect: bool = False,
    ):
        """
        Establishes a connection to the Velour API.

        Parameters
        ----------
        host : str
            The host to connect to. Should start with "http://" or "https://".
        access_token : str
            The access token for the host (if the host requires authentication).
        """
        connect(host=host, access_token=access_token, reconnect=reconnect)
        return cls(get_connection())

    def get_labels(
        self,
        filter_: Union[Filter, dict, None] = None,
    ) -> List[Label]:
        """
        Gets all labels with option to filter.

        Parameters
        ----------
        filter_ : velour.Filter, optional
            Optional filter to constrain by.

        Returns
        ------
        List[velour.Label]
            A list of labels.
        """
        if isinstance(filter_, Filter):
            filter_ = asdict(filter_)
        return [
            Label.from_dict(label) for label in self.conn.get_labels(filter_)
        ]

    def get_labels_from_dataset(
        self, dataset: Union[Dataset, str]
    ) -> List[Label]:
        """
        Get all labels associated with a dataset's groundtruths.

        Parameters
        ----------
        dataset : velour.Dataset
            The dataset to search by.

        Returns
        ------
        List[velour.Label]
            A list of labels.
        """
        if isinstance(dataset, Dataset):
            dataset = dataset.name
        return [
            Label.from_dict(label)
            for label in self.conn.get_labels_from_dataset(dataset)
        ]

    def get_labels_from_model(self, model: Union[Model, str]) -> List[Label]:
        """
        Get all labels associated with a model's groundtruths.

        Parameters
        ----------
        model : velour.Model
            The model to search by.

        Returns
        ------
        List[velour.Label]
            A list of labels.
        """
        if isinstance(model, Model):
            model = model.name
        return [
            Label.from_dict(label)
            for label in self.conn.get_labels_from_model(model)
        ]

    def create_dataset(
        self,
        dataset: Union[Dataset, dict],
    ) -> None:
        """
        Creates a dataset.

        Parameters
        ----------
        dataset : velour.Dataset
            The dataset to create.
        """
        if isinstance(dataset, Dataset):
            dataset = dataset.to_dict()
        self.conn.create_dataset(dataset)

    def create_groundtruth(
        self,
        dataset: Union[Dataset, str],
        groundtruth: Union[GroundTruth, dict],
    ):
        """
        Create a groundtruth.

        Parameters
        ----------
        dataset : velour.Dataset
            The dataset to create the groundtruth for.
        groundtruth : velour.GroundTruth
            The groundtruth to create.
        """
        if isinstance(groundtruth, GroundTruth):
            if isinstance(dataset, Dataset):
                dataset = dataset.name
            groundtruth = groundtruth.to_dict(dataset_name=dataset)
        return self.conn.create_groundtruths(groundtruth)

    def get_groundtruth(
        self,
        dataset: Union[Dataset, str],
        datum: Union[Datum, str],
    ) -> Union[GroundTruth, None]:
        """
        Get a particular groundtruth.

        Parameters
        ----------
        dataset: Union[Dataset, str]
            The dataset the datum belongs to.
        datum: Union[Datum, str]
            The desired datum.

        Returns
        ----------
        Union[GroundTruth, None]
            The matching groundtruth or 'None' if it doesn't exist.
        """
        if isinstance(dataset, Dataset):
            dataset = dataset.name
        if isinstance(datum, Datum):
            datum = datum.uid

        try:
            resp = self.conn.get_groundtruth(
                dataset_name=dataset, datum_uid=datum
            )
            return GroundTruth.from_dict(resp)
        except ClientException as e:
            if e.status_code == 404:
                return None
            raise e

    def finalize_dataset(self, dataset: Union[Dataset, str]) -> None:
        """
        Finalizes a dataset such that new groundtruths cannot be added to it.

        Parameters
        ----------
        dataset : str
            The dataset to be finalized.
        """
        if isinstance(dataset, Dataset):
            dataset = dataset.name
        return self.conn.finalize_dataset(name=dataset)

    def get_dataset(
        self,
        name: str,
    ) -> Union[Dataset, None]:
        """
        Gets a dataset by name.

        Parameters
        ----------
        name : str
            The name of the dataset to fetch.

        Returns
        -------
        Union[Dataset, None]
            A Dataset with matching name or 'None' if one doesn't exist.
        """
        try:
            return Dataset.from_dict(
                self.conn.get_dataset(name), connection=self.conn
            )
        except ClientException as e:
            if e.status_code == 404:
                return None
            raise e

    def get_datasets(
        self,
        filter_: Union[Filter, dict, None] = None,
    ) -> List[Dataset]:
        """
        Get all datasets with option to filter results.

        Parameters
        ----------
        filter_ : velour.Filter, optional
            Optional filter to constrain by.

        Returns
        ------
        List[velour.Dataset]
            A list of datasets.
        """
        if isinstance(filter_, Filter):
            filter_ = asdict(filter_)
        return [
            Dataset.from_dict(dataset, connection=self.conn)
            for dataset in self.conn.get_datasets(filter_)
        ]

    def get_datums(
        self,
        filter_: Union[Filter, dict, None] = None,
    ) -> List[Datum]:
        """
        Get all datums with option to filter results.

        Parameters
        ----------
        filter_ : velour.Filter, optional
            Optional filter to constrain by.

        Returns
        -------
        List[velour.Datum]
            A list datums.
        """
        if isinstance(filter_, Filter):
            filter_ = asdict(filter_)
        return [
            Datum.from_dict(datum) for datum in self.conn.get_datums(filter_)
        ]

    def get_datum(
        self,
        dataset: Union[Dataset, str],
        uid: str,
    ) -> Union[Datum, None]:
        """
        Get datum.

        `GET` endpoint.

        Parameters
        ----------
        dataset : velour.Dataset
            The dataset the datum belongs to.
        uid : str
            The uid of the datum.

        Returns
        -------
        velour.Datum
            The requested datum or 'None' if it doesn't exist.
        """
        if isinstance(dataset, Dataset):
            dataset = dataset.name
        try:
            resp = self.conn.get_datum(dataset_name=dataset, uid=uid)
            return Datum.from_dict(resp)
        except ClientException as e:
            if e.status_code == 404:
                return None
            raise e

    def get_dataset_status(
        self,
        name: str,
    ) -> Union[TableStatus, None]:
        """
        Get the state of a given dataset.

        Parameters
        ----------
        name : str
            The name of the dataset we want to fetch the state of.

        Returns
        ------
        TableStatus | None
            The state of the dataset or 'None' if dataset does not exist.
        """
        try:
            return self.conn.get_dataset_status(name)
        except ClientException as e:
            if e.status_code == 404:
                return None
            raise e

    def get_dataset_summary(self, name: str) -> DatasetSummary:
        """
        Gets the summary of a dataset.

        Parameters
        ----------
        name : str
            The name of the dataset to create a summary for.

        Returns
        -------
        DatasetSummary
            A dataclass containing the dataset summary.
        """
        return DatasetSummary(**self.conn.get_dataset_summary(name))

    def delete_dataset(self, name: str, timeout: int = 0) -> None:
        """
        Deletes a dataset.

        Parameters
        ----------
        name : str
            The name of the dataset to be deleted.
        timeout : int
            The number of seconds to wait in order to confirm that the dataset was deleted.
        """
        self.conn.delete_dataset(name)
        if timeout:
            for _ in range(timeout):
                if self.get_dataset(name) is None:
                    break
                else:
                    time.sleep(1)
            else:
                raise TimeoutError(
                    "Dataset wasn't deleted within timeout interval"
                )

    def create_model(
        self,
        model: Union[Model, dict],
    ):
        """
        Creates a model.

        Parameters
        ----------
        model : velour.Model
            The model to create.
        """
        if isinstance(model, Model):
            model = model.to_dict()
        self.conn.create_model(model)

    def create_prediction(
        self,
        dataset: Union[Dataset, str],
        model: Union[Model, str],
        prediction: Union[Prediction, dict],
    ) -> None:
        """
        Create a prediction.

        Parameters
        ----------
        dataset : velour.Dataset
            The dataset that is being operated over.
        model : velour.Model
            The model making the prediction.
        prediction : velour.Prediction
            The prediction to create.
        """
        if isinstance(prediction, Prediction):
            if isinstance(dataset, Dataset):
                dataset = dataset.name
            if isinstance(model, Model):
                model = model.name
            prediction = prediction.to_dict(
                dataset_name=dataset,
                model_name=model,
            )
        return self.conn.create_predictions(prediction)

    def get_prediction(
        self,
        dataset: Union[Dataset, str],
        model: Union[Model, str],
        datum: Union[Datum, str],
    ) -> Union[Prediction, None]:
        """
        Get a particular prediction.

        Parameters
        ----------
        dataset: Union[Dataset, str]
            The dataset the datum belongs to.
        model: Union[Model, str]
            The model that made the prediction.
        datum: Union[Datum, str]
            The desired datum.

        Returns
        ----------
        Union[Prediction, None]
            The matching prediction or 'None' if it doesn't exist.
        """
        if isinstance(dataset, Dataset):
            dataset = dataset.name
        if isinstance(model, Model):
            model = model.name
        if isinstance(datum, Datum):
            datum = datum.uid

        try:
            resp = self.conn.get_prediction(
                dataset_name=dataset, model_name=model, datum_uid=datum
            )
            return Prediction.from_dict(resp)
        except ClientException as e:
            if e.status_code == 404:
                return None
            raise e

    def finalize_inferences(
        self, dataset: Union[Dataset, str], model: Union[Model, str]
    ) -> None:
        """
        Finalizes a model-dataset pairing such that new prediction cannot be added to it.
        """
        if isinstance(dataset, Dataset):
            dataset = dataset.name
        if isinstance(model, Model):
            model = model.name
        return self.conn.finalize_inferences(
            dataset_name=dataset, model_name=model
        )

    def get_model(
        self,
        name: str,
    ) -> Union[Model, None]:
        """
        Gets a model by name.

        Parameters
        ----------
        name : str
            The name of the model to fetch.

        Returns
        -------
        Union[velour.Model, None]
            A Model with matching name or 'None' if one doesn't exist.
        """
        try:
            return Model.from_dict(
                self.conn.get_model(name), connection=self.conn
            )
        except ClientException as e:
            if e.status_code == 404:
                return None
            raise e

    def get_models(
        self,
        filter_: Union[Filter, dict, None] = None,
    ) -> List[Model]:
        """
        Get all models with option to filter results.

        Parameters
        ----------
        filter_ : velour.Filter, optional
            Optional filter to constrain by.

        Returns
        ------
        List[velour.Model]
            A list of models.
        """
        if isinstance(filter_, Filter):
            filter_ = asdict(filter_)
        return [
            Model.from_dict(model, connection=self.conn)
            for model in self.conn.get_models(filter_)
        ]

    def get_model_status(
        self,
        dataset_name: str,
        model_name: str,
    ) -> Optional[TableStatus]:
        """
        Get the state of a given model over a dataset.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset that the model is operating over.
        model_name : str
            The name of the model we want to fetch the state of.

        Returns
        ------
        Union[TableStatus, None]
            The state of the model or 'None' if the model doesn't exist.
        """
        try:
            return self.conn.get_model_status(dataset_name, model_name)
        except ClientException as e:
            if e.status_code == 404:
                return None
            raise e

    def get_model_eval_requests(
        self, model: Union[Model, str]
    ) -> List[Evaluation]:
        """
        Get all evaluations that have been created for a model.

        This does not return evaluation results.

        `GET` endpoint.

        Parameters
        ----------
        model : str
            The model to search by.

        Returns
        -------
        List[Evaluation]
            A list of evaluations.
        """
        if isinstance(model, Model):
            model = model.name
        return [
            Evaluation(**evaluation, connection=self.conn)
            for evaluation in self.conn.get_model_eval_requests(model)
        ]

    def delete_model(self, name: str, timeout: int = 0) -> None:
        """
        Deletes a model.

        Parameters
        ----------
        name : str
            The name of the model to be deleted.
        timeout : int
            The number of seconds to wait in order to confirm that the model was deleted.
        """
        self.conn.delete_model(name)
        if timeout:
            for _ in range(timeout):
                if self.get_model(name) is None:
                    break
                else:
                    time.sleep(1)
            else:
                raise TimeoutError(
                    "Model wasn't deleted within timeout interval"
                )

    def get_evaluations(
        self,
        *,
        evaluation_ids: Optional[List[int]] = None,
        models: Union[List[Model], List[str], None] = None,
        datasets: Union[List[Dataset], List[str], None] = None,
    ) -> List[Evaluation]:
        """
        Returns all evaluations associated with user-supplied dataset and/or model names.

        Parameters
        ----------
        evaluation_ids : List[int], optional.
            A list of job ids to return metrics for.
        models : Union[List[velour.Model], List[str]], optional
            A list of model names that we want to return metrics for.
        datasets : Union[List[velour.Dataset], List[str]], optional
            A list of dataset names that we want to return metrics for.

        Returns
        -------
        List[velour.Evaluation]
            A list of evaluations.
        """
        if isinstance(datasets, list):
            datasets = [
                element.name if isinstance(element, Dataset) else element
                for element in datasets
            ]
        if isinstance(models, list):
            models = [
                element.name if isinstance(element, Model) else element
                for element in models
            ]
        return [
            Evaluation(connection=self.conn, **evaluation)
            for evaluation in self.conn.get_evaluations(
                evaluation_ids=evaluation_ids,
                models=models,
                datasets=datasets,
            )
        ]

    def evaluate(self, request: EvaluationRequest) -> List[Evaluation]:
        """
        Creates as many evaluations as necessary to fulfill the request.

        Parameters
        ----------
        request : schemas.EvaluationRequest
            The requested evaluation parameters.

        Returns
        -------
        List[Evaluation]
            A list of evaluations that meet the parameters.
        """
        return [
            Evaluation(**evaluation)
            for evaluation in self.conn.evaluate(request)
        ]
