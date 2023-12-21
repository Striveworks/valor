import json
import math
import warnings
from dataclasses import asdict
from typing import Dict, List, Tuple, Union

from velour.client import Client, ClientException, Job, wait_for_predicate
from velour.enums import AnnotationType, TaskType, JobStatus
from velour.exceptions import SchemaTypeError
from velour.schemas.evaluation import (
    DetectionParameters,
    EvaluationJob,
    EvaluationResult,
    EvaluationSettings,
)
from velour.schemas.filters import BinaryExpression, DeclarativeMapper, Filter
from velour.schemas.geometry import BoundingBox, MultiPolygon, Polygon, Raster
from velour.schemas.metadata import validate_metadata


class Label:
    """
    An object for labeling datasets, models, and annotations.

    Parameters
    ----------
    key : str
        A key for the `Label`.
    value : str
        A value for the `Label`.
    score : float
        The score associated with the `Label` (where applicable).

    Attributes
    ----------
    id : int
        A unique ID for the `Label`.
    """

    def __init__(self, key: str, value: str, score: Union[float, None] = None):
        self.key = key
        self.value = value
        self.score = score
        self._validate()

    def _validate(self):
        """
        Validate the inputs of the `Label`.
        """
        if not isinstance(self.key, str):
            raise TypeError("key should be of type `str`")
        if not isinstance(self.value, str):
            raise TypeError("value should be of type `str`")
        if isinstance(self.score, int):
            self.score = float(self.score)
        if not isinstance(self.score, (float, type(None))):
            raise TypeError("score should be of type `float`")

    def tuple(self) -> Tuple[str, str, Union[float, None]]:
        """
        Defines how the `Label` is turned into a tuple.

        Returns
        ----------
        tuple
            A tuple of the `Label's` arguments.
        """
        return (self.key, self.value, self.score)

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
        """
        Defines how a `Label` is hashed.

        Returns
        ----------
        int
            The hashed 'Label`.
        """
        return hash(f"key:{self.key},value:{self.value},score:{self.score}")

    def dict(self) -> dict:
        """
        Defines how a `Label` is transformed into a dictionary.

        Returns
        ----------
        dict
            A dictionary of the `Label's` attributes.
        """
        return {
            "key": self.key,
            "value": self.value,
            "score": self.score,
        }

    @classmethod
    def all(cls, client: Client) -> List["Label"]:
        """
        Returns a list of all labels in the backend.
        """
        return [
            cls(key=label["key"], value=label["value"], score=label["score"])
            for label in client.get_labels()
        ]


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
    dataset : str
        The name of the dataset to associate the `Datum` with.
    """

    def __init__(
        self,
        uid: str,
        metadata: Dict[str, Union[int, float, str]] = None,
        geospatial: Dict[
            str,
            Union[
                List[List[List[List[Union[float, int]]]]],
                List[List[List[Union[float, int]]]],
                List[Union[float, int]],
                str,
            ],
        ] = None,
        dataset: str = "",
    ):
        self.uid = uid
        self.metadata = metadata if metadata else {}
        self.geospatial = geospatial if geospatial else {}
        self.dataset = dataset
        self._validate()

    def _validate(self):
        """
        Validates the parameters used to create a `Datum` object.
        """
        if not isinstance(self.dataset, str):
            raise SchemaTypeError("dataset", str, self.dataset)
        if not isinstance(self.uid, str):
            raise SchemaTypeError("uid", str, self.uid)
        validate_metadata(self.metadata)

    def dict(self) -> dict:
        """
        Defines how a `Datum` object is transformed into a dictionary.

        Returns
        ----------
        dict
            A dictionary of the `Datum's` attributes.
        """
        return {
            "dataset": self.dataset,
            "uid": self.uid,
            "metadata": self.metadata,
            "geospatial": self.geospatial,
        }

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
        return self.dict() == other.dict()


class Annotation:
    """
    A class used to annotate `GroundTruths` and `Predictions`.

    Parameters
    ----------
    task_type: TaskType
        The task type associated with the `Annotation`.
    labels: List[Label]
        A list of labels to use for the `Annotation`.
    metadata: Dict[str, Union[int, float, str]]
        A dictionary of metadata that describes the `Annotation`.
    bounding_box: BoundingBox
        A bounding box to assign to the `Annotation`.
    polygon: Polygon
        A polygon to assign to the `Annotation`.
    multipolygon: MultiPolygon
        A multipolygon to assign to the `Annotation`.
    raster: Raster
        A raster to assign to the `Annotation`.
    jsonb: Dict
        A jsonb to assign to the `Annotation`.

    Attributes
    ----------
    geometric_area : float
        The area of the annotation.
    """

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
        """
        Validates the parameters used to create a `Annotation` object.
        """

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
        """
        Defines how a `Annotation` object is transformed into a dictionary.

        Returns
        ----------
        dict
            A dictionary of the `Annotation's` attributes.
        """
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
        return self.dict() == other.dict()


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
        Validate the inputs of the `GroundTruth`.
        """
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
        """
        Defines how a `GroundTruth` is transformed into a dictionary.

        Returns
        ----------
        dict
            A dictionary of the `GroundTruth's` attributes.
        """
        return {
            "datum": self.datum.dict(),
            "annotations": [
                annotation.dict() for annotation in self.annotations
            ],
        }

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
        return self.dict() == other.dict()


class Prediction:
    """
    An object describing a prediction (e.g., a machine-drawn bounding box on an image).

    Parameters
    ----------
    datum : Datum
        The `Datum` associated with the `Prediction`.
    annotations : List[Annotation]
        The list of `Annotations` associated with the `Prediction`.
    model : str
        The name of the model that produced the `Prediction`.

    Attributes
    ----------
    score : Union[float, int]
        The score assigned to the `Prediction`.
    """

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
        """
        Validate the inputs of the `Prediction`.
        """
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
        """
        Defines how a `Prediction` is transformed into a dictionary.

        Returns
        ----------
        dict
            A dictionary of the `Prediction's` attributes.
        """
        return {
            "datum": self.datum.dict(),
            "model": self.model,
            "annotations": [
                annotation.dict() for annotation in self.annotations
            ],
        }

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
        return self.dict() == other.dict()


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

    def __init__(self):
        self.client: Client = None
        self.id: int = None
        self.name: str = None
        self.metadata: dict = None
        self.geospatial: dict = None

    @classmethod
    def create(
        cls,
        client: Client,
        name: str,
        metadata: Dict[str, Union[int, float, str]] = None,
        geospatial: Dict[
            str,
            Union[
                List[List[List[List[Union[float, int]]]]],
                List[List[List[Union[float, int]]]],
                List[Union[float, int]],
                str,
            ],
        ] = None,
        id: Union[int, None] = None,
    ):
        """
        Create a new `Dataset` object.

        Parameters
        ----------
        client : Client
            The `Client` object associated with the session.
        name : str
            The name of the dataset.
        metadata : dict
            A dictionary of metadata that describes the dataset.
        geospatial :  dict
            A GeoJSON-style dictionary describing the geospatial coordinates of the dataset.


        Returns
        ----------
        Dataset
            The newly-created `Dataset`.
        """
        dataset = cls()
        dataset.client = client
        dataset.name = name
        dataset.metadata = metadata
        dataset.geospatial = geospatial
        dataset.id = id
        dataset._validate()
        client._requests_post_rel_host("datasets", json=dataset.dict())
        return cls.get(client, name)

    @classmethod
    def get(cls, client: Client, name: str):
        """
        Fetches a given dataset from the backend.

        Parameters
        ----------
        client : Client
            The `Client` object associated with the session.
        name : str
            The name of the dataset.


        Returns
        ----------
        Dataset
            The requested `Dataset`.
        """
        resp = client._requests_get_rel_host(f"datasets/{name}").json()
        dataset = cls()
        dataset.client = client
        dataset.name = resp["name"]
        dataset.metadata = resp["metadata"]
        dataset.geospatial = resp["geospatial"]
        dataset.id = resp["id"]
        dataset._validate()
        return dataset

    def _validate(self):
        """
        Validates the arguments used to create a `Dataset` object.
        """
        # validation
        if not isinstance(self.name, str):
            raise TypeError("`name` should be of type `str`")
        if not isinstance(self.id, int) and self.id is not None:
            raise TypeError("`id` should be of type `int`")
        if not self.metadata:
            self.metadata = {}
        if not self.geospatial:
            self.geospatial = {}
        validate_metadata(self.metadata)

    def dict(self) -> dict:
        """
        Defines how a `Dataset` object is transformed into a dictionary.

        Returns
        ----------
        dict
            A dictionary of the `Dataset's` attributes.
        """
        return {
            "id": self.id,
            "name": self.name,
            "metadata": self.metadata,
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
                f"GroundTruth for datum with uid `{groundtruth.datum.uid}` contains no annotations. Skipping..."
            )
            return

        groundtruth.datum.dataset = self.name
        self.client._requests_post_rel_host(
            "groundtruths",
            json=groundtruth.dict(),
        )

    def get_groundtruth(self, uid: str) -> GroundTruth:
        """
        Fetches a given groundtruth from the backend.

        Parameters
        ----------
        uid : str
            The UID of the 'GroundTruth' to fetch.


        Returns
        ----------
        GroundTruth
            The requested `GroundTruth`.
        """
        resp = self.client._requests_get_rel_host(
            f"groundtruths/dataset/{self.name}/datum/{uid}"
        ).json()
        return GroundTruth(**resp)

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

    def get_datums(
        self,
    ) -> List[Datum]:
        """
        Get all datums associated with a given dataset.

        Returns
        ----------
        List[Datum]
            A list of `Datums` associated with the dataset.
        """
        datums = self.client._requests_get_rel_host(
            f"data/dataset/{self.name}"
        ).json()
        return [Datum(**datum) for datum in datums]

    def get_evaluations(
        self,
    ) -> List[EvaluationResult]:
        """
        Get all evaluations associated with a given dataset.

        Returns
        ----------
        List[Evaluation]
            A list of `Evaluations` associated with the dataset.
        """
        return self.client.get_bulk_evaluations(datasets=self.name)

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
        job = Job(self.client, dataset_name=self.name)
        self.client._requests_delete_rel_host(f"datasets/{self.name}").json()
        return job


class Evaluation:
    """
    Wraps `velour.client.Job` to provide evaluation-specifc members.
    """

    def __init__(self, client: Client, evaluation_id: int, **kwargs):
        self.client = client
        self.evaluation_id = evaluation_id

        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def job_id(self):
        return self.evaluation_id

    def get_result(self) -> EvaluationResult:
        """
        Fetch the first `EvaluationResult` for our `job_id`.

        Returns
        ----------
        schemas.EvaluationResult
            The result of the evaluation job

        Raises
        ----------
        ClientException
            If an Evaluation with the given `job_id` is not found.
        """
        response = self.client.get_bulk_evaluations(
            job_ids=[self.evaluation_id]
        )
        if not response:
            raise ClientException("Not Found")
        return response[0]

    def wait_for_completion(
        self,
        *,
        timeout: int = None,
        interval: float = 1.0,
    ) -> EvaluationResult:
        return wait_for_predicate(
            lambda: self.get_result(),
            lambda result: result.status in [JobStatus.DONE, JobStatus.FAILED],
            timeout,
            interval,
        )


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

    def __init__(self):
        self.client: Client = None
        self.id: int = None
        self.name: str = ""
        self.metadata: dict = None
        self.geospatial: dict = None

    @classmethod
    def create(
        cls,
        client: Client,
        name: str,
        metadata: Dict[str, Union[int, float, str]] = None,
        geospatial: Dict[
            str,
            Union[
                List[List[List[List[Union[float, int]]]]],
                List[List[List[Union[float, int]]]],
                List[Union[float, int]],
                str,
            ],
        ] = None,
        id: Union[int, None] = None,
    ):
        """
        Create a new Model

        Attributes
        ----------
        client : Client
            The `Client` object associated with the session.
        name : str
            The name of the model.
        metadata : dict
            A dictionary of metadata that describes the model.
        geospatial :  dict
            A GeoJSON-style dictionary describing the geospatial coordinates of the model.
        id : int
            The ID of the model.


        Returns
        ----------
        Model
            The newly-created `Model` object.
        """
        model = cls()
        model.client = client
        model.name = name
        model.metadata = metadata
        model.geospatial = geospatial
        model.id = id
        model._validate()
        client._requests_post_rel_host("models", json=model.dict())
        return cls.get(client, name)

    @classmethod
    def get(cls, client: Client, name: str):
        """
        Fetches a given model from the backend.

        Parameters
        ----------
        client : Client
            The `Client` object associated with the session.
        name : str
            The name of the model.


        Returns
        ----------
        Model
            The requested `Model`.
        """
        resp = client._requests_get_rel_host(f"models/{name}").json()
        model = cls()
        model.client = client
        model.name = resp["name"]
        model.metadata = resp["metadata"]
        model.geospatial = resp["geospatial"]
        model.id = resp["id"]
        model._validate()
        return model

    def _validate(self):
        """
        Validates the arguments used to create a `Model` object.
        """
        if not isinstance(self.name, str):
            raise TypeError("`name` should be of type `str`")
        if not isinstance(self.id, int) and self.id is not None:
            raise TypeError("`id` should be of type `int`")
        if not self.metadata:
            self.metadata = {}
        if not self.geospatial:
            self.geospatial = {}
        validate_metadata(self.metadata)

    def dict(self) -> dict:
        """
        Defines how a `Model` object is transformed into a dictionary.

        Returns
        ----------
        dict
            A dictionary of the `Model's` attributes.
        """
        return {
            "id": self.id,
            "name": self.name,
            "metadata": self.metadata,
            "geospatial": self.geospatial,
        }

    def add_prediction(self, prediction: Prediction):
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
                f"Prediction for datum with uid `{prediction.datum.uid}` contains no annotations. Skipping..."
            )
            return

        prediction.model = self.name
        return self.client._requests_post_rel_host(
            "predictions",
            json=prediction.dict(),
        )

    def finalize_inferences(self, dataset: "Dataset") -> None:
        """
        Finalize the `Model` object such that new `Predictions` cannot be added to it.
        """
        return self.client._requests_put_rel_host(
            f"models/{self.name}/datasets/{dataset.name}/finalize"
        ).json()

    def evaluate_classification(
        self,
        dataset: Dataset,
        filters: Union[Dict, List[BinaryExpression]] = None,
    ) -> Evaluation:
        """
        Start a classification evaluation job.

        Parameters
        ----------
        dataset : Dataset
            The dataset to evaluate against.
        filters : Union[Dict, List[BinaryExpression]]
            Optional set of filters to constrain evaluation by.
        timeout : int
            The number of seconds to wait for the job to finish. Used to ensure deterministic behavior when testing.

        Returns
        -------
        Evaluation
            A job object that can be used to track the status of the job and get the metrics of it upon completion.
        """

        # If list[BinaryExpression], convert to filter object
        if not isinstance(filters, dict) and filters is not None:
            filters = Filter.create(filters)

        evaluation = EvaluationJob(
            model=self.name,
            dataset=dataset.name,
            task_type=TaskType.CLASSIFICATION.value,
            settings=EvaluationSettings(
                filters=filters,
            ),
        )

        resp = self.client._requests_post_rel_host(
            "evaluations", json=asdict(evaluation)
        ).json()

        evaluation_id = resp.pop("job_id")
        evaluation_job = Evaluation(
            client=self.client,
            dataset_name=dataset.name,
            model_name=self.name,
            evaluation_id=evaluation_id,
            **resp,
        )

        return evaluation_job

    def evaluate_detection(
        self,
        dataset: "Dataset",
        iou_thresholds_to_compute: List[float] = None,
        iou_thresholds_to_keep: List[float] = None,
        filters: Union[Dict, List[BinaryExpression]] = None,
    ) -> Evaluation:
        """
        Start a object-detection evaluation job.

        Parameters
        ----------
        dataset : Dataset
            The dataset to evaluate against.
        iou_thresholds_to_compute : List[float]
            Thresholds to compute mAP against.
        iou_thresholds_to_keep : List[float]
            Thresholds to return AP for. Must be subset of `iou_thresholds_to_compute`.
        filters : Union[Dict, List[BinaryExpression]]
            Optional set of filters to constrain evaluation by.
        timeout : int
            The number of seconds to wait for the job to finish. Used to ensure deterministic behavior when testing.

        Returns
        -------
        Evaluation
            A job object that can be used to track the status of the job and get the metrics of it upon completion.
        """

        # Default iou thresholds
        if iou_thresholds_to_compute is None:
            iou_thresholds_to_compute = [
                round(0.5 + 0.05 * i, 2) for i in range(10)
            ]
        if iou_thresholds_to_keep is None:
            iou_thresholds_to_keep = [0.5, 0.75]

        parameters = DetectionParameters(
            iou_thresholds_to_compute=iou_thresholds_to_compute,
            iou_thresholds_to_keep=iou_thresholds_to_keep,
        )

        if not isinstance(filters, dict) and filters is not None:
            filters = Filter.create(filters)

        evaluation = EvaluationJob(
            model=self.name,
            dataset=dataset.name,
            task_type=TaskType.DETECTION.value,
            settings=EvaluationSettings(
                parameters=parameters,
                filters=filters,
            ),
        )

        resp = self.client._requests_post_rel_host(
            "evaluations", json=asdict(evaluation)
        ).json()

        # resp should have keys "missing_pred_labels", "ignored_pred_labels", with values
        # list of label dicts. convert label dicts to Label objects

        for k in ["missing_pred_labels", "ignored_pred_labels"]:
            resp[k] = [Label(**la) for la in resp[k]]

        evaluation_id = resp.pop("job_id")
        evaluation_job = Evaluation(
            client=self.client,
            evaluation_id=evaluation_id,
            **resp,
        )

        return evaluation_job

    def evaluate_segmentation(
        self,
        dataset: Dataset,
        filters: Union[Dict, List[BinaryExpression]] = None,
    ) -> Evaluation:
        """
        Start a semantic-segmentation evaluation job.

        Parameters
        ----------
        dataset : Dataset
            The dataset to evaluate against.
        filters : Union[Dict, List[BinaryExpression]]
            Optional set of filters to constrain evaluation by.
        timeout : int
            The number of seconds to wait for the job to finish. Used to ensure deterministic behavior when testing.

        Returns
        -------
        Evaluation
            a job object that can be used to track the status of the job and get the metrics of it upon completion
        """

        # if list[BinaryExpression], convert to filter object
        if not isinstance(filters, dict) and filters is not None:
            filters = Filter.create(filters)

        # create evaluation job
        evaluation = EvaluationJob(
            model=self.name,
            dataset=dataset.name,
            task_type=TaskType.SEGMENTATION.value,
            settings=EvaluationSettings(
                filters=filters,
            ),
        )
        resp = self.client._requests_post_rel_host(
            "evaluations",
            json=asdict(evaluation),
        ).json()

        # create client-side evaluation handler
        evaluation_id = resp.pop("job_id")
        evaluation_job = Evaluation(
            client=self.client,
            evaluation_id=evaluation_id,
            **resp,
        )

        return evaluation_job

    def delete(
        self,
    ):
        """
        Delete the `Model` object from the backend.
        """
        job = Job(self.client, model_name=self.name)
        self.client._requests_delete_rel_host(f"models/{self.name}").json()
        return job

    def get_prediction(self, datum: Datum) -> Prediction:
        """
        Fetch a particular prediction.

        Parameters
        ----------
        datum : Datum
            The `Datum` of the prediction to return.

        Returns
        ----------
        Prediction
            The requested `Prediction`.
        """
        resp = self.client._requests_get_rel_host(
            f"predictions/model/{self.name}/dataset/{datum.dataset}/datum/{datum.uid}",
        ).json()
        return Prediction(**resp)

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
    ) -> List[EvaluationResult]:
        """
        Get all evaluations associated with a given model.

        Returns
        ----------
        List[Evaluation]
            A list of `Evaluations` associated with the model.
        """
        return self.client.get_bulk_evaluations(models=self.name)

    def get_metric_dataframes(
        self,
    ) -> dict:
        """
        Get all metrics associated with a Model and return them in a `pd.DataFrame`.

        Returns
        ----------
        dict
            A dictionary of the `Model's` metrics and settings, with the metrics being displayed in a `pd.DataFrame`.
        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Must have pandas installed to use `get_metric_dataframes`."
            )

        ret = []
        for evaluation in self.get_evaluations():
            metrics = [
                {**metric, "dataset": evaluation.dataset}
                for metric in evaluation.metrics
            ]
            df = pd.DataFrame(metrics)
            for k in ["label", "parameters"]:
                df[k] = df[k].fillna("n/a")
            df["parameters"] = df["parameters"].apply(json.dumps)
            df["label"] = df["label"].apply(
                lambda x: f"{x['key']}: {x['value']}" if x != "n/a" else x
            )
            df = df.pivot(
                index=["type", "parameters", "label"], columns=["dataset"]
            )
            ret.append({"settings": evaluation.settings, "df": df})

        return ret


# Assign all declarative mappers.

# Label
Label.id = DeclarativeMapper("label_ids", int)
Label.key = DeclarativeMapper("label_keys", str)
Label.label = DeclarativeMapper("labels", Label)

# Datum
Datum.uid = DeclarativeMapper("datum_uids", str)
Datum.metadata = DeclarativeMapper("datum_metadata", Union[int, float, str])
Datum.geospatial = DeclarativeMapper(
    "datum_geospatial",
    Union[
        List[List[List[List[Union[float, int]]]]],
        List[List[List[Union[float, int]]]],
        List[Union[float, int]],
        str,
    ],
)

# Prediction
Prediction.score = DeclarativeMapper("prediction_scores", Union[int, float])

# Dataset
Dataset.name = DeclarativeMapper("dataset_names", str)
Dataset.metadata = DeclarativeMapper("dataset_metadata", Union[int, float, str])
Dataset.geospatial = DeclarativeMapper(
    "dataset_geospatial",
    Union[
        List[List[List[List[Union[float, int]]]]],
        List[List[List[Union[float, int]]]],
        List[Union[float, int]],
        str,
    ],
)

# Model
Annotation.task = DeclarativeMapper("task_types", TaskType)
Annotation.type = DeclarativeMapper("annotation_types", AnnotationType)
Annotation.geometric_area = DeclarativeMapper("annotation_geometric_area", float)
Annotation.metadata = DeclarativeMapper("annotation_metadata", Union[int, float, str])
Annotation.geospatial = DeclarativeMapper(
    "annotation_geospatial",
    Union[
        List[List[List[List[Union[float, int]]]]],
        List[List[List[Union[float, int]]]],
        List[Union[float, int]],
        str,
    ],
)

Model.name = DeclarativeMapper("models_names", str)
Model.metadata = DeclarativeMapper("models_metadata", Union[int, float, str])
Model.geospatial = DeclarativeMapper(
    "model_geospatial",
    Union[
        List[List[List[List[Union[float, int]]]]],
        List[List[List[Union[float, int]]]],
        List[Union[float, int]],
        str,
    ],
)
