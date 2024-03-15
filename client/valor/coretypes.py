from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

from valor.client import ClientConnection, connect, get_connection
from valor.enums import AnnotationType, EvaluationStatus, TableStatus, TaskType
from valor.exceptions import ClientException
from valor.schemas.evaluation import EvaluationParameters, EvaluationRequest
from valor.schemas.filters import Filter
from valor.symbolic import (
    Symbol,
    Equatable,
    Nullable,
    Value,
    String,
    Score,
    BoundingBox,
    BoundingPolygon,
    Raster,
    Embedding,
    Metadata,
    StaticCollection,
    jsonify,
)


FilterType = Union[list, dict]  # TODO - Remove this


def _format_filter(filter_by: Optional[FilterType]) -> Filter:
    """
    Formats the various filter or constraint representations into a 'schemas.Filter' object.

    Parameters
    ----------
    filter_by : FilterType, optional
        The reference filter.

    Returns
    -------
    valor.schemas.Filter
        A properly formatted 'schemas.Filter' object.
    """
    if isinstance(filter_by, Filter):
        return filter_by
    elif isinstance(filter_by, list) or filter_by is None:
        filter_by = filter_by if filter_by else []
        return Filter.create(filter_by)
    elif isinstance(filter_by, dict):
        return Filter(**filter_by)
    else:
        raise TypeError


class Label(StaticCollection):
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

    key = String.symbolic(name="label", attribute="key")
    value = String.symbolic(name="label", attribute="value")
    score = Score.symbolic(name="label", attribute="score")

    @classmethod
    def create(
        cls,
        key: str,
        value: str,
        score: Union[float, np.floating, None] = None,
    ):
        ret = cls(value={})
        ret.key = String.definite(key)
        ret.value = String.definite(value)
        ret.score = Score.definite(score)
        return ret
    
    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) is dict

    # def __eq__(self, other) -> bool:
    #     """
    #     Defines how `Labels` are compared to one another.

    #     Parameters
    #     ----------
    #     other : Label
    #         The object to compare with the `Label`.

    #     Returns
    #     ----------
    #     boolean
    #         A boolean describing whether the two objects are equal.
    #     """
    #     # type mismatch
    #     if type(other) is not type(self):
    #         return False

    #     # k,v mismatch
    #     if (
    #         self.key._value != other.key._value 
    #         or self.value._value != other.value._value
    #     ):
    #         return False

    #     # score is None
    #     if self.score._value is None or other.score._value is None:
    #         return (other.score._value is None) == (self.score._value is None)

    #     # scores not equal
    #     if self.score._value is not None and other.score._value is not None:
    #         return bool(np.isclose(self.score._value, other.score._value))

    #     return False

    def __hash__(self) -> int:
        """
        Defines how a `Label` is hashed.

        Returns
        ----------
        int
            The hashed 'Label`.
        """
        return hash(self.tuple())

    def tuple(self):
        """
        Defines how the `Label` is turned into a tuple.

        Returns
        ----------
        tuple
            A tuple of the `Label's` arguments.
        """
        return (self.key._value, self.value._value, self.score._value)


class LabelList(Equatable, Nullable):

    @staticmethod
    def supports(value: Any) -> bool:
        if type(value) is not list:
            return False
        for element in value:
            if (
                type(element) is not Label
                or not element.is_value()
            ):
                return False
        return True
    
    @staticmethod
    def encode(value: List[Label]):
        return [
            label.encode(label)
            for label in value
        ]
    
    def decode(self):
        pass


class Annotation(StaticCollection):
    """
    A class used to annotate `GroundTruths` and `Predictions`.

    Parameters
    ----------
    task_type: TaskType
        The task type associated with the `Annotation`.
    labels: List[Label], optional
        A list of labels to use for the `Annotation`.
    metadata: Dict[str, Union[int, float, str, bool, datetime.datetime, datetime.date, datetime.time]]
        A dictionary of metadata that describes the `Annotation`.
    bounding_box: BoundingBox, optional
        A bounding box to assign to the `Annotation`.
    polygon: Polygon, optional
        A polygon to assign to the `Annotation`.
    raster: Raster, optional
        A raster to assign to the `Annotation`.
    embedding: List[float], optional
        An embedding, described by a list of values with type float and a maximum length of 16,000.

    Attributes
    ----------
    geometric_area : float
        The area of the annotation.

    Examples
    --------

    Classification
    >>> Annotation.create(
    ...     task_type=TaskType.CLASSIFICATION,
    ...     labels=[
    ...         Label.create(key="class", value="dog"),
    ...         Label.create(key="category", value="animal"),
    ...     ]
    ... )

    Object-Detection BoundingBox
    >>> annotation = Annotation.create(
    ...     task_type=TaskType.OBJECT_DETECTION,
    ...     labels=[Label.create(key="k1", value="v1")],
    ...     bounding_box=box2,
    ... )

    Object-Detection Polygon
    >>> annotation = Annotation.create(
    ...     task_type=TaskType.OBJECT_DETECTION,
    ...     labels=[Label.create(key="k1", value="v1")],
    ...     polygon=polygon1,
    ... )

    Object-Detection Raster
    >>> annotation = Annotation.create(
    ...     task_type=TaskType.OBJECT_DETECTION,
    ...     labels=[Label.create(key="k1", value="v1")],
    ...     raster=raster1,
    ... )

    Semantic-Segmentation Raster
    >>> annotation = Annotation.create(
    ...     task_type=TaskType.SEMANTIC_SEGMENTATION,
    ...     labels=[Label.create(key="k1", value="v1")],
    ...     raster=raster1,
    ... )

    Defining all supported annotation types for a given `task_type` is allowed!
    >>> Annotation.create(
    ...     task_type=TaskType.OBJECT_DETECTION,
    ...     labels=[Label.create(key="k1", value="v1")],
    ...     bounding_box=box1,
    ...     polygon=polygon1,
    ...     raster=raster1,
    ... )
    """

    task_type = String.symbolic(name="annotation", attribute="task_type")
    labels = LabelList.symbolic(name="annotation", attribute="labels")
    metadata = Metadata.symbolic(name="annotation", attribute="metadata")
    bounding_box = BoundingBox.symbolic(name="annotation", attribute="bounding_box")
    polygon = BoundingPolygon.symbolic(name="annotation", attribute="bounding_polygon")
    raster = Raster.symbolic(name="annotation", attribute="raster")
    embedding = Embedding.symbolic(name="annotation", attribute="embedding")

    @classmethod
    def create(
        cls,
        task_type: TaskType,
        labels: Optional[List[Label]] = None,
        metadata: Optional[dict] = None,
        bounding_box: Optional[BoundingBox] = None,
        polygon: Optional[BoundingPolygon] = None,
        raster: Optional[Raster] = None,
        embedding: Optional[List[Embedding]] = None,
    ):
        ret = cls(value={})
        ret.task_type = String.definite(task_type.value)
        ret.labels = LabelList.definite(labels if labels else [])
        ret.metadata = Metadata.definite(metadata if metadata else {})
        ret.bounding_box = BoundingBox.definite(bounding_box)
        ret.polygon = BoundingPolygon.definite(polygon)
        ret.raster = Raster.definite(raster)
        ret.embedding = Embedding.definite(embedding)
        return ret


class AnnotationList(Value):

    @staticmethod
    def supports(value: Any) -> bool:
        if type(value) is not list:
            return False
        for element in value:
            if (
                type(element) is not Label
                or not element.is_value()
            ):
                return False
        return True


class Datum(StaticCollection):
    """
    A class used to store datum about `GroundTruths` and `Predictions`.

    Parameters
    ----------
    uid : str
        The UID of the `Datum`.
    metadata : dict
        A dictionary of metadata that describes the `Datum`.
    """

    uid = String.symbolic(name="uid")
    metadata = Metadata.symbolic(name="datum", attribute="metadata")

    @classmethod
    def create(
        cls,
        uid: Optional[str],
        metadata: Optional[dict] = None,
    ):
        ret = cls.symbolic()
        ret.uid = String.definite(uid)
        ret.metadata = Metadata.definite(metadata if metadata else {})
        return ret

    def get_uid(self):
        if not self.uid._value:
            raise ValueError
        elif not self.uid.is_value():
            raise ValueError("Datum uid is symbolic and has no value.")
        else:
            return self.uid._value
        

if __name__ == "__main__":
    l = Label.symbolic()
    label = Label.create(key="k1", value="v2")

    print(str(l))
    print(str(label))

    Label()

    cond = (
        Annotation.labels.is_none() 
        | (Annotation.bounding_box.area > 200) 
        | (Label.symbolic() == Label.create(key="k1", value="v1"))
        | (Datum.uid == "1234")
        & (Annotation.labels == [label])
    )
    print(cond)

    # TODO - list and dict types should be replaced by their proper titles of LabelList or Label

    print(json.dumps(jsonify(cond), indent=2))


class GroundTruth(StaticCollection):
    """
    An object describing a ground truth (e.g., a human-drawn bounding box on an image).

    Parameters
    ----------
    datum : Datum
        The `Datum` associated with the `GroundTruth`.
    annotations : List[Annotation]
        The list of `Annotations` associated with the `GroundTruth`.
    """
    
    datum = Datum.symbolic(name="groundtruth", attribute="datum")
    annotations = AnnotationList.symbolic(name="groundtruth", attribute="annotations")

    def __init__(self, datum: Datum, annotations: List[Annotation], **kwargs):
        self.datum = Datum.definite(datum)
        self.annotations = AnnotationList.definite(annotations)
        self.kwargs = kwargs
        super().__init__(value=Symbol)


class Prediction(StaticCollection):
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

    datum = Datum.symbolic(name="prediction", attribute="datum")
    annotations = AnnotationList.symbolic(name="prediction", attribute="annotations")

    def __init__(self, datum: Datum, annotations: List[Annotation], **kwargs):
        self.datum = Datum.definite(datum)
        self.annotations = Annotation.definite(annotations)
        self.kwargs = kwargs
        self._validate()
        super().__init__(value=Symbol())

    def _validate(self):
        """
        Validate the inputs of the `Prediction` based on task type.
        """
        if type(self.annotations._value) is not list:
            raise TypeError  # NOTE - This exists only for typing.
        
        for annotation in self.annotations._value:
            task_type = annotation.task_type._value
            if task_type in [
                TaskType.CLASSIFICATION,
                TaskType.OBJECT_DETECTION,
            ]:
                labels = annotation.labels._value
                for label in labels:
                    label_score = label.score._value
                    if label_score is None:
                        raise ValueError(
                            f"For task type `{task_type}` prediction labels must have scores, but got `None`"
                        )
            elif task_type == TaskType.CLASSIFICATION:
                label_keys_to_sum = {}
                labels = annotation.labels._value
                for scored_label in labels:
                    label_key = scored_label.key._value
                    label_score = scored_label.score._value
                    if label_key not in label_keys_to_sum:
                        label_keys_to_sum[label_key] = 0.0
                    label_keys_to_sum[label_key] += label_score

                for k, total_score in label_keys_to_sum.items():
                    if abs(total_score - 1) > 1e-5:
                        raise ValueError(
                            "For each label key, prediction scores must sum to 1, but"
                            f" for label key {k} got scores summing to {total_score}."
                        )


class Evaluation:
    """
    Wraps `valor.client.Job` to provide evaluation-specifc members.
    """

    def __init__(
        self, connection: Optional[ClientConnection] = None, **kwargs
    ):
        """
        Defines important attributes of the API's `EvaluationResult`.

        Attributes
        ----------
        id : int
            The ID of the evaluation.
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

    def __str__(self) -> str:
        """Dumps the object into a JSON formatted string."""
        return json.dumps(self.to_dict(), indent=4)

    def to_dict(self) -> dict:
        """
        Defines how a `valor.Evaluation` object is serialized into a dictionary.

        Returns
        ----------
        dict
            A dictionary describing an evaluation.
        """
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
        Poll the back end.

        Updates the evaluation with the latest state from the back end.

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
    num_rasters: int
    task_types: List[TaskType]
    labels: List[Label]
    datum_metadata: List[dict]
    annotation_metadata: List[dict]

    def __post_init__(self):
        for i, tt in enumerate(self.task_types):
            if isinstance(tt, str):
                self.task_types[i] = TaskType(tt)
        for i, label in enumerate(self.labels):
            if isinstance(label, dict):
                self.labels[i] = Label.create(**label)


class Dataset(StaticCollection):
    """
    A class describing a given dataset.

    Parameters
    ----------
    name : str
        The name of the dataset.
    metadata : dict
        A dictionary of metadata that describes the dataset.
    """

    name = String.symbolic(name="dataset", attribute="name")
    metadata = Metadata.symbolic(name="dataset", attribute="metadata")

    def __init__(
        self,
        name: str,
        metadata: Optional[dict] = None,
        connection: Optional[ClientConnection] = None,
        **kwargs,
    ):
        """
        Create or get a `Dataset` object.

        Parameters
        ----------
        name : str
            The name of the dataset.
        metadata : dict
            An optional dictionary of metadata that describes the dataset.
        connection : ClientConnnetion
            An optional Valor client object for interacting with the API.
        """
        self.conn = connection
        self.name = String.definite(name)
        self.metadata = Metadata.definite(metadata if metadata else {})
        self.kwargs = kwargs

    @classmethod
    def create(
        cls,
        name: str,
        metadata: Optional[dict] = None,
    ) -> Dataset:
        """
        Creates a dataset that persists in the back end.

        Parameters
        ----------
        name : str
            The name of the dataset.
        metadata : dict
            A dictionary of metadata that describes the dataset.

        Returns
        -------
        valor.Dataset
            The created dataset.
        """
        dataset = cls(
            name=name,
            metadata=metadata,
        )
        Client(dataset.conn).create_dataset(dataset)
        return dataset

    @classmethod
    def get(
        cls,
        name: str,
    ) -> Union[Dataset, None]:
        """
        Retrieves a dataset from the back end database.

        Parameters
        ----------
        name : str
            The name of the dataset.

        Returns
        -------
        Union[valor.Dataset, None]
            The dataset or 'None' if it doesn't exist.
        """
        return Client().get_dataset(name)

    def add_groundtruth(
        self,
        groundtruth: GroundTruth,
    ) -> None:
        """
        Add a ground truth to the dataset.

        Parameters
        ----------
        groundtruth : GroundTruth
            The ground truth to create.
        """
        Client(self.conn).create_groundtruths(
            dataset=self,
            groundtruths=[groundtruth],
        )

    def add_groundtruths(
        self,
        groundtruths: List[GroundTruth],
    ) -> None:
        """
        Add multiple ground truths to the dataset.

        Parameters
        ----------
        groundtruths : List[GroundTruth]
            The ground truths to create.
        """
        Client(self.conn).create_groundtruths(
            dataset=self,
            groundtruths=groundtruths,
        )

    def get_groundtruth(
        self,
        datum: Union[Datum, str],
    ) -> Union[GroundTruth, None]:
        """
        Get a particular ground truth.

        Parameters
        ----------
        datum: Union[Datum, str]
            The desired datum.

        Returns
        ----------
        Union[GroundTruth, None]
            The matching ground truth or 'None' if it doesn't exist.
        """
        return Client(self.conn).get_groundtruth(dataset=self, datum=datum)

    def get_name(self) -> str:
        if not self.name._value:
            raise ValueError
        elif not self.name.is_value():
            raise ValueError("Dataset name is symbolic and has no value.")
        else:
            return self.name._value

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

    def get_datums(
        self, filter_by: Optional[FilterType] = None
    ) -> List[Datum]:
        """
        Get all datums associated with a given dataset.

        Parameters
        ----------
        filter_by
            Optional constraints to filter by.

        Returns
        ----------
        List[Datum]
            A list of `Datums` associated with the dataset.
        """
        filter_ = _format_filter(filter_by)
        if isinstance(filter_, Filter):
            filter_ = asdict(filter_)

        if filter_.get("dataset_names"):
            raise ValueError(
                "Cannot filter by dataset_names when calling `Dataset.get_datums`."
            )
        filter_["dataset_names"] = [self.name]
        return Client(self.conn).get_datums(filter_by=filter_)

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

            num_annotations: total number of labeled annotations in the dataset; if an
            object (such as a bounding box) has multiple labels, then each label is counted separately

            num_bounding_boxes: total number of bounding boxes in the dataset

            num_polygons: total number of polygons in the dataset

            num_rasters: total number of rasters in the dataset

            task_types: list of the unique task types in the dataset

            labels: list of the unique labels in the dataset

            datum_metadata: list of the unique metadata dictionaries in the dataset that are associated
            to datums

            groundtruth_annotation_metadata: list of the unique metadata dictionaries in the dataset that are
            associated to annotations
        """
        return Client(self.conn).get_dataset_summary(self.get_name())

    def finalize(
        self,
    ):
        """
        Finalizes the dataset such that new ground truths cannot be added to it.
        """
        return Client(self.conn).finalize_dataset(self)

    def delete(
        self,
        timeout: int = 0,
    ):
        """
        Delete the dataset from the back end.

        Parameters
        ----------
        timeout : int, default=0
            Sets a timeout in seconds.
        """
        Client(self.conn).delete_dataset(self.get_name(), timeout)


class Model(StaticCollection):
    """
    A class describing a model that was trained on a particular dataset.

    Parameters
    ----------
    name : str
        The name of the model.
    metadata : dict
        A dictionary of metadata that describes the model.
    """

    name = String.symbolic(name="model", attribute="name")
    metadata = Metadata.symbolic(name="model", attribute="metadata")

    def __init__(
        self,
        name: str,
        metadata: Optional[dict] = None,
        connection: Optional[ClientConnection] = None,
        **kwargs,
    ):
        """
        Create or get a `Model` object.

        Parameters
        ----------
        name : str
            The name of the model.
        metadata : dict
            An optional dictionary of metadata that describes the dataset.
        connection : ClientConnnetion
            An optional Valor client object for interacting with the API.
        """
        self.conn = connection
        self.name = String.definite(name)
        self.metadata = Metadata.definite(metadata if metadata else {})
        self.kwargs = kwargs

    @classmethod
    def create(
        cls,
        name: str,
        metadata: Optional[dict] = None,
    ) -> Model:
        """
        Creates a model that persists in the back end.

        Parameters
        ----------
        name : str
            The name of the model.
        metadata : dict
            A dictionary of metadata that describes the model.

        Returns
        -------
        valor.Model
            The created model.
        """
        model = cls(
            name=name,
            metadata=metadata,
        )
        Client(model.conn).create_model(model)
        return model

    @classmethod
    def get(
        cls,
        name: str,
    ) -> Union[Model, None]:
        """
        Retrieves a model from the back end database.

        Parameters
        ----------
        name : str
            The name of the model.

        Returns
        -------
        Union[valor.Model, None]
            The model or 'None' if it doesn't exist.
        """
        return Client().get_model(name)

    def add_prediction(
        self,
        dataset: Dataset,
        prediction: Prediction,
    ) -> None:
        """
        Add a prediction to the model.

        Parameters
        ----------
        dataset : valor.Dataset
            The dataset that is being operated over.
        prediction : valor.Prediction
            The prediction to create.
        """
        Client(self.conn).create_predictions(
            dataset=dataset,
            model=self,
            predictions=[prediction],
        )

    def add_predictions(
        self,
        dataset: Dataset,
        predictions: List[Prediction],
    ) -> None:
        """
        Add multiple predictions to the model.

        Parameters
        ----------
        dataset : valor.Dataset
            The dataset that is being operated over.
        predictions : List[valor.Prediction]
            The predictions to create.
        """
        Client(self.conn).create_predictions(
            dataset=dataset,
            model=self,
            predictions=predictions,
        )

    def get_name(self) -> str:
        if not self.name._value:
            raise ValueError
        elif not self.name.is_value():
            raise ValueError("Model name is symbolic and has no value.")
        else:
            return self.name._value

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
        Finalizes the model over a dataset such that new predictions cannot be added to it.
        """
        return Client(self.conn).finalize_inferences(
            dataset=dataset, model=self
        )

    def _format_constraints(
        self,
        datasets: Optional[Union[Dataset, List[Dataset]]] = None,
        filter_by: Optional[FilterType] = None,
    ) -> Filter:
        """Formats the 'datum_filter' for any evaluation requests."""

        # get list of dataset names
        dataset_names_from_obj = []
        if isinstance(datasets, list):
            dataset_names_from_obj = [
                dataset.get_name() 
                for dataset in datasets
            ]
        elif isinstance(datasets, Dataset):
            dataset_names_from_obj = [datasets.get_name()]

        # create a 'schemas.Filter' object from the constraints.
        filter_ = _format_filter(filter_by)

        # reset model name
        filter_.model_names = None
        filter_.model_metadata = None

        # set dataset names
        if not filter_.dataset_names:
            filter_.dataset_names = []
        filter_.dataset_names.extend(dataset_names_from_obj)
        return filter_

    def _create_label_map(
        self,
        label_map: Optional[Dict[Label, Label]],
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

        return_value = []
        for key, value in label_map.items():
            if not all(
                [
                    (
                        isinstance(v.key._value, str)
                        and isinstance(v.value._value, str)
                    )
                    for v in [key, value]
                ]
            ):
                raise TypeError
            return_value.append(
                [
                    [key.key._value, key.value._value], 
                    [value.key._value, value.value._value]
                ]
            )
        return return_value

    def evaluate_classification(
        self,
        datasets: Optional[Union[Dataset, List[Dataset]]] = None,
        filter_by: Optional[FilterType] = None,
        label_map: Optional[Dict[Label, Label]] = None,
    ) -> Evaluation:
        """
        Start a classification evaluation job.

        Parameters
        ----------
        datasets : Union[Dataset, List[Dataset]], optional
            The dataset or list of datasets to evaluate against.
        filter_by : FilterType, optional
            Optional set of constraints to filter evaluation by.
        label_map : Dict[Label, Label], optional
            Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.

        Returns
        -------
        Evaluation
            A job object that can be used to track the status of the job and get the metrics of it upon completion.
        """
        if not datasets and not filter_by:
            raise ValueError(
                "Evaluation requires the definition of either datasets, dataset filters or both."
            )

        # format request
        datum_filter = self._format_constraints(datasets, filter_by)
        request = EvaluationRequest(
            model_names=[self.get_name()],
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
        filter_by: Optional[FilterType] = None,
        convert_annotations_to_type: Optional[AnnotationType] = None,
        iou_thresholds_to_compute: Optional[List[float]] = None,
        iou_thresholds_to_return: Optional[List[float]] = None,
        label_map: Optional[Dict[Label, Label]] = None,
        recall_score_threshold: float = 0,
    ) -> Evaluation:
        """
        Start an object-detection evaluation job.

        Parameters
        ----------
        datasets : Union[Dataset, List[Dataset]], optional
            The dataset or list of datasets to evaluate against.
        filter_by : FilterType, optional
            Optional set of constraints to filter evaluation by.
        convert_annotations_to_type : enums.AnnotationType, optional
            Forces the object detection evaluation to compute over this type.
        iou_thresholds_to_compute : List[float], optional
            Thresholds to compute mAP against.
        iou_thresholds_to_return : List[float], optional
            Thresholds to return AP for. Must be subset of `iou_thresholds_to_compute`.
        label_map : Dict[Label, Label], optional
            Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.
        recall_score_threshold: float, default=0
            The confidence score threshold for use when determining whether to count a prediction as a true positive or not while calculating Average Recall.
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
            task_type=TaskType.OBJECT_DETECTION,
            convert_annotations_to_type=convert_annotations_to_type,
            iou_thresholds_to_compute=iou_thresholds_to_compute,
            iou_thresholds_to_return=iou_thresholds_to_return,
            label_map=self._create_label_map(label_map=label_map),
            recall_score_threshold=recall_score_threshold,
        )
        datum_filter = self._format_constraints(datasets, filter_by)
        request = EvaluationRequest(
            model_names=[self.get_name()],
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
        filter_by: Optional[FilterType] = None,
        label_map: Optional[Dict[Label, Label]] = None,
    ) -> Evaluation:
        """
        Start a semantic-segmentation evaluation job.

        Parameters
        ----------
        datasets : Union[Dataset, List[Dataset]], optional
            The dataset or list of datasets to evaluate against.
        filter_by : FilterType, optional
            Optional set of constraints to filter evaluation by.
        label_map : Dict[Label, Label], optional
            Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.

        Returns
        -------
        Evaluation
            A job object that can be used to track the status of the job and get the metrics of it upon completion
        """
        # format request
        datum_filter = self._format_constraints(datasets, filter_by)
        request = EvaluationRequest(
            model_names=[self.get_name()],
            datum_filter=datum_filter,
            parameters=EvaluationParameters(
                task_type=TaskType.SEMANTIC_SEGMENTATION,
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
        Delete the `Model` object from the back end.

        Parameters
        ----------
        timeout : int, default=0
            Sets a timeout in seconds.
        """
        Client(self.conn).delete_model(self.get_name(), timeout)

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
    Valor client object for interacting with the api.

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
    ) -> Client:
        """
        Establishes a connection to the Valor API.

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
        filter_by: Optional[FilterType] = None,
    ) -> List[Label]:
        """
        Gets all labels using an optional filter.

        Parameters
        ----------
        filter_by : FilterType, optional
            Optional constraints to filter by.

        Returns
        ------
        List[valor.Label]
            A list of labels.
        """
        filter_ = _format_filter(filter_by)
        filter_ = asdict(filter_)
        return [
            Label.create(**label) for label in self.conn.get_labels(filter_)
        ]

    def get_labels_from_dataset(
        self, dataset: Union[Dataset, str]
    ) -> List[Label]:
        """
        Get all labels associated with a dataset's ground truths.

        Parameters
        ----------
        dataset : valor.Dataset
            The dataset to search by.

        Returns
        ------
        List[valor.Label]
            A list of labels.
        """
        dataset_name = dataset.get_name() if isinstance(dataset, Dataset) else dataset
        return [
            Label.create(**label)
            for label in self.conn.get_labels_from_dataset(dataset_name)
        ]

    def get_labels_from_model(self, model: Union[Model, str]) -> List[Label]:
        """
        Get all labels associated with a model's ground truths.

        Parameters
        ----------
        model : valor.Model
            The model to search by.

        Returns
        ------
        List[valor.Label]
            A list of labels.
        """
        model_name = model.get_name() if isinstance(model, Model) else model
        return [
            Label.create(**label)
            for label in self.conn.get_labels_from_model(model_name)
        ]

    def create_dataset(
        self,
        dataset: Union[Dataset, dict],
    ) -> None:
        """
        Creates a dataset.

        Parameters
        ----------
        dataset : valor.Dataset
            The dataset to create.
        """
        if isinstance(dataset, Dataset):
            dataset = dataset.to_dict()
        self.conn.create_dataset(dataset)

    def create_groundtruths(
        self,
        dataset: Dataset,
        groundtruths: List[GroundTruth],
    ):
        """
        Creates ground truths.

        Parameters
        ----------

        dataset : valor.Dataset
            The dataset to create the ground truth for.
        groundtruths : List[valor.GroundTruth]
            The ground truths to create.
        """
        groundtruths_json = []
        for groundtruth in groundtruths:
            if not isinstance(groundtruth, GroundTruth):
                raise TypeError(
                    f"Expected ground truth to be of type 'valor.GroundTruth' not '{type(groundtruth)}'."
                )
            if not isinstance(groundtruth.annotations._value, list):
                raise TypeError
            if len(groundtruth.annotations._value) == 0:
                warnings.warn(
                    f"GroundTruth for datum with uid `{groundtruth.datum.uid}` contains no annotations."
                )
            groundtruth_dict = groundtruth.to_dict()
            groundtruth_dict["dataset_name"] = dataset.get_name()
            groundtruths_json.append(groundtruth_dict)
        self.conn.create_groundtruths(groundtruths_json)

    def get_groundtruth(
        self,
        dataset: Union[Dataset, str],
        datum: Union[Datum, str],
    ) -> Union[GroundTruth, None]:
        """
        Get a particular ground truth.

        Parameters
        ----------
        dataset: Union[Dataset, str]
            The dataset the datum belongs to.
        datum: Union[Datum, str]
            The desired datum.

        Returns
        ----------
        Union[GroundTruth, None]
            The matching ground truth or 'None' if it doesn't exist.
        """
        dataset_name = dataset.get_name() if isinstance(dataset, Dataset) else dataset
        datum_uid = datum.get_uid() if isinstance(datum, Datum) else datum
        try:
            resp = self.conn.get_groundtruth(
                dataset_name=dataset_name, datum_uid=datum_uid
            )
            return GroundTruth(**resp)
        except ClientException as e:
            if e.status_code == 404:
                return None
            raise e

    def finalize_dataset(self, dataset: Union[Dataset, str]) -> None:
        """
        Finalizes a dataset such that new ground truths cannot be added to it.

        Parameters
        ----------
        dataset : str
            The dataset to be finalized.
        """
        dataset_name = dataset.get_name() if isinstance(dataset, Dataset) else dataset
        return self.conn.finalize_dataset(name=dataset_name)

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
            A Dataset with a matching name, or 'None' if one doesn't exist.
        """
        try:
            return Dataset(
                connection=self.conn,
                **self.conn.get_dataset(name),
            )
        except ClientException as e:
            if e.status_code == 404:
                return None
            raise e

    def get_datasets(
        self,
        filter_by: Optional[FilterType] = None,
    ) -> List[Dataset]:
        """
        Get all datasets, with an option to filter results according to some user-defined parameters.

        Parameters
        ----------
        filter_by : FilterType, optional
            Optional constraints to filter by.

        Returns
        ------
        List[valor.Dataset]
            A list of datasets.
        """
        filter_ = _format_filter(filter_by)
        if isinstance(filter_, Filter):
            filter_ = asdict(filter_)
        return [
            Dataset(connection=self.conn, **dataset)
            for dataset in self.conn.get_datasets(filter_)
        ]

    def get_datums(
        self,
        filter_by: Optional[FilterType] = None,
    ) -> List[Datum]:
        """
        Get all datums using an optional filter.

        Parameters
        ----------
        filter_by : FilterType, optional
            Optional constraints to filter by.

        Returns
        -------
        List[valor.Datum]
            A list datums.
        """
        filter_ = _format_filter(filter_by)
        if isinstance(filter_, Filter):
            filter_ = asdict(filter_)
        return [
            Datum.create(**datum) for datum in self.conn.get_datums(filter_)
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
        dataset : valor.Dataset
            The dataset the datum belongs to.
        uid : str
            The UID of the datum.
        Returns
        -------
        valor.Datum
            The requested datum or 'None' if it doesn't exist.
        """
        dataset_name = dataset.get_name() if isinstance(dataset, Dataset) else dataset
        try:
            resp = self.conn.get_datum(dataset_name=dataset_name, uid=uid)
            return Datum.create(**resp)
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
            The state of the dataset, or 'None' if the dataset does not exist.
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
        model : valor.Model
            The model to create.
        """
        if isinstance(model, Model):
            model = model.to_dict()
        self.conn.create_model(model)

    def create_predictions(
        self,
        dataset: Dataset,
        model: Model,
        predictions: List[Prediction],
    ) -> None:
        """
        Creates predictions.

        Parameters
        ----------
        dataset : valor.Dataset
            The dataset that is being operated over.
        model : valor.Model
            The model making the prediction.
        predictions : List[valor.Prediction]
            The predictions to create.
        """
        predictions_json = []
        for prediction in predictions:
            if not isinstance(prediction, Prediction):
                raise TypeError(
                    f"Expected prediction to be of type 'valor.Prediction' not '{type(prediction)}'."
                )
            if not isinstance(prediction.annotations._value, list):
                raise TypeError
            if len(prediction.annotations._value) == 0:
                warnings.warn(
                    f"Prediction for datum with uid `{prediction.datum.uid}` contains no annotations."
                )
            prediction_dict = prediction.to_dict()
            prediction_dict["datum"]["dataset_name"] = dataset.get_name()
            prediction_dict["model_name"] = model.get_name()
            predictions_json.append(prediction_dict)
        self.conn.create_predictions(predictions_json)

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
        dataset_name = dataset.get_name() if isinstance(dataset, Dataset) else dataset
        model_name = model.get_name() if isinstance(model, Model) else model
        datum_uid = datum.get_uid() if isinstance(datum, Datum) else datum
        try:
            resp = self.conn.get_prediction(
                dataset_name=dataset_name,
                model_name=model_name,
                datum_uid=datum_uid,
            )
            return Prediction(**resp)
        except ClientException as e:
            if e.status_code == 404:
                return None
            raise e

    def finalize_inferences(
        self, dataset: Union[Dataset, str], model: Union[Model, str]
    ) -> None:
        """
        Finalizes a model-dataset pairing such that new predictions cannot be added to it.
        """
        dataset_name = dataset.get_name() if isinstance(dataset, Dataset) else dataset
        model_name = model.get_name() if isinstance(model, Model) else model
        return self.conn.finalize_inferences(
            dataset_name=dataset_name,
            model_name=model_name,
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
        Union[valor.Model, None]
            A Model with matching name or 'None' if one doesn't exist.
        """
        try:
            return Model(
                connection=self.conn,
                **self.conn.get_model(name)
            )
        except ClientException as e:
            if e.status_code == 404:
                return None
            raise e

    def get_models(
        self,
        filter_by: Optional[FilterType] = None,
    ) -> List[Model]:
        """
        Get all models using an optional filter.

        Parameters
        ----------
        filter_by : FilterType, optional
            Optional constraints to filter by.

        Returns
        ------
        List[valor.Model]
            A list of models.
        """
        filter_ = _format_filter(filter_by)
        if isinstance(filter_, Filter):
            filter_ = asdict(filter_)
        return [
            Model(connection=self.conn, **model)
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
        model_name = model.get_name() if isinstance(model, Model) else model
        return [
            Evaluation(**evaluation, connection=self.conn)
            for evaluation in self.conn.get_model_eval_requests(model_name)
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
            A list of job IDs to return metrics for.
        models : Union[List[valor.Model], List[str]], optional
            A list of model names that we want to return metrics for.
        datasets : Union[List[valor.Dataset], List[str]], optional
            A list of dataset names that we want to return metrics for.

        Returns
        -------
        List[valor.Evaluation]
            A list of evaluations.
        """
        if isinstance(datasets, list):
            datasets = [
                element.get_name() if isinstance(element, Dataset) else element
                for element in datasets
            ]
        if isinstance(models, list):
            models = [
                element.get_name() if isinstance(element, Model) else element
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
