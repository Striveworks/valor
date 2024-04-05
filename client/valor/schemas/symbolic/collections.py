from typing import Any, Dict, List, Optional

from valor.enums import TaskType
from valor.schemas.symbolic.types import (
    Box,
    Dictionary,
    Embedding,
    Equatable,
    Float,
)
from valor.schemas.symbolic.types import List as SymbolicList
from valor.schemas.symbolic.types import (
    Polygon,
    Raster,
    String,
    Symbol,
    TaskTypeEnum,
    Variable,
    _get_type_by_name,
)


def _get_schema_type_by_name(name: str):
    types_ = {
        "label": Label,
        "annotation": Annotation,
        "datum": Datum,
    }
    return _get_type_by_name(name=name, additional_types=types_)


class StaticCollection(Equatable):
    """
    A static collection is a Variable that defines its contents by static attributes.
    """

    def __init__(self):
        super().__init__(value=None, symbol=None)

    @classmethod
    def definite(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @classmethod
    def nullable(cls, *args, **kwargs):
        raise NotImplementedError(
            "Static collections do not define 'nullable' by default."
        )

    @classmethod
    def symbolic(
        cls,
        name: Optional[str] = None,
        key: Optional[str] = None,
        attribute: Optional[str] = None,
        owner: Optional[str] = None,
    ):
        """
        Initialize object as a symbol.

        Parameters
        ----------
        name: str, optional
            The name of the symbol. Defaults to the name of the parent class.
        key: str, optional
            An optional dictionary key.
        attribute: str, optional
            An optional attribute name.
        owner: str, optional
            An optional name describing the class that owns this symbol.
        """
        symbol = Symbol(
            name=name if name else cls.__name__.lower(),
            key=key,
            attribute=attribute,
            owner=owner,
        )
        obj = cls.__new__(cls)
        obj._value = symbol
        return obj

    @staticmethod
    def format(**kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def __validate__(cls, value: Any):
        """Validate typing."""
        if value is not None:
            raise TypeError(
                "A StaticCollection does not store an internal value."
            )

    @classmethod
    def decode_value(cls, value: dict):
        """Decode object from JSON compatible dictionary."""
        return cls(**value)

    def encode_value(self):
        """Encode object to JSON compatible dictionary."""
        return {
            k: v.encode_value() for k, v in self._get_dynamic_values().items()
        }

    @classmethod
    def _get_static_types(cls) -> Dict[str, type]:
        """Returns any static members that inherit from 'Variable'."""
        fields = getattr(cls, "__annotations__", dict())
        retval = dict()
        for k, v in fields.items():
            if isinstance(v, type) and issubclass(v, Variable):
                retval[k] = v
            elif isinstance(v, str):
                retval[k] = _get_schema_type_by_name(v)
            else:
                raise NotImplementedError(
                    f"Unknown typing. Attribute '{k}' with type '{v}'."
                )
        return retval

    def _get_dynamic_values(self):
        """Returns the values of attributes that inherit from 'Variable'."""
        return {
            name: self.__getattribute__(name)
            for name in self._get_static_types().keys()
        }

    def __repr__(self):
        if self.is_symbolic:
            return super().__repr__()
        return self.encode_value().__repr__()

    def __str__(self):
        if self.is_symbolic:
            return super().__str__()
        return str(self.encode_value())

    @property
    def is_symbolic(self) -> bool:
        return super().is_symbolic


class Label(StaticCollection):
    """
    An object for labeling datasets, models, and annotations.

    Attributes
    ----------
    key : String
        The class label key.
    value : String
        The class label value.
    score : Score
        The label score.

    Examples
    --------
    >>> Label(key="k1", value="v1")
    >>> Label(key="k1", value="v1", score=None)
    >>> Label(key="k1", value="v1", score=0.9)
    """

    key: String = String.symbolic(owner="label", name="key")
    value: String = String.symbolic(owner="label", name="value")
    score: Float = Float.symbolic(owner="label", name="score")

    def __init__(
        self,
        key: str,
        value: str,
        score: Optional[float] = None,
    ):
        self.key = String.definite(key)
        self.value = String.definite(value)
        self.score = Float.nullable(score)
        super().__init__()

    def tuple(self):
        """
        Defines how the `Label` is turned into a tuple.

        Returns
        ----------
        tuple
            A tuple of the `Label's` arguments.
        """
        return (
            self.key.get_value(),
            self.value.get_value(),
            self.score.get_value(),
        )


class Annotation(StaticCollection):
    """
    A class used to annotate `GroundTruths` and `Predictions`.

    Attributes
    ----------
    task_type: TaskTypeEnum
        The task type associated with the `Annotation`.
    metadata: Dictionary
        A dictionary of metadata that describes the `Annotation`.
    labels: List[Label], optional
        A list of labels to use for the `Annotation`.
    bounding_box: Box
        A bounding box to assign to the `Annotation`.
    polygon: BoundingPolygon
        A polygon to assign to the `Annotation`.
    raster: Raster
        A raster to assign to the `Annotation`.
    embedding: List[float]
        An embedding, described by a list of values with type float and a maximum length of 16,000.

    Examples
    --------

    Classification
    >>> Annotation.create(
    ...     task_type=TaskType.CLASSIFICATION,
    ...     labels=[
    ...         Label(key="class", value="dog"),
    ...         Label(key="category", value="animal"),
    ...     ]
    ... )

    Object-Detection Box
    >>> annotation = Annotation.create(
    ...     task_type=TaskType.OBJECT_DETECTION,
    ...     labels=[Label(key="k1", value="v1")],
    ...    bounding_box=box2,
    ... )

    Object-Detection Polygon
    >>> annotation = Annotation.create(
    ...     task_type=TaskType.OBJECT_DETECTION,
    ...     labels=[Label(key="k1", value="v1")],
    ...     polygon=BoundingPolygon(...),
    ... )

    Object-Detection Raster
    >>> annotation = Annotation.create(
    ...     task_type=TaskType.OBJECT_DETECTION,
    ...     labels=[Label(key="k1", value="v1")],
    ...     raster=Raster(...),
    ... )

    Semantic-Segmentation Raster
    >>> annotation = Annotation.create(
    ...     task_type=TaskType.SEMANTIC_SEGMENTATION,
    ...     labels=[Label(key="k1", value="v1")],
    ...     raster=Raster(...),
    ... )

    Defining all supported annotation types for a given `task_type` is allowed!
    >>> Annotation.create(
    ...     task_type=TaskType.OBJECT_DETECTION,
    ...     labels=[Label(key="k1", value="v1")],
    ...     bounding_box=Box(...),
    ...     polygon=BoundingPolygon(...),
    ...     raster=Raster(...),
    ... )
    """

    task_type: TaskTypeEnum = TaskTypeEnum.symbolic(
        owner="annotation", name="task_type"
    )
    metadata: Dictionary = Dictionary.symbolic(
        owner="annotation", name="metadata"
    )
    labels: SymbolicList[Label] = SymbolicList[Label].symbolic(
        owner="annotation", name="labels"
    )
    bounding_box: Box = Box.symbolic(owner="annotation", name="box")
    polygon: Polygon = Polygon.symbolic(owner="annotation", name="polygon")
    raster: Raster = Raster.symbolic(owner="annotation", name="raster")
    embedding: Embedding = Embedding.symbolic(
        owner="annotation", name="embedding"
    )

    def __init__(
        self,
        task_type: TaskType,
        metadata: Optional[dict] = None,
        labels: Optional[List[Label]] = None,
        bounding_box: Optional[Box] = None,
        polygon: Optional[Polygon] = None,
        raster: Optional[Raster] = None,
        embedding: Optional[Embedding] = None,
    ):
        """
        Constructs an annotation.

        Parameters
        ----------
        task_type: TaskTypeEnum
            The task type associated with the `Annotation`.
        metadata: Dict[str, Union[int, float, str, bool, datetime.datetime, datetime.date, datetime.time]]
            A dictionary of metadata that describes the `Annotation`.
        labels: List[Label]
            A list of labels to use for the `Annotation`.
        bounding_box: Box, optional
            A bounding box annotation.
        polygon: Polygon, optional
            A polygon annotation.
        raster: Raster, optional
            A raster annotation.
        embedding: List[float], optional
            An embedding, described by a list of values with type float and a maximum length of 16,000.
        """
        self.task_type = TaskTypeEnum.definite(task_type)
        self.metadata = Dictionary.definite(metadata if metadata else dict())
        self.labels = SymbolicList[Label].definite(
            labels if labels else list()
        )
        self.bounding_box = Box.nullable(bounding_box)
        self.polygon = Polygon.nullable(polygon)
        self.raster = Raster.nullable(raster)
        self.embedding = Embedding.nullable(embedding)
        super().__init__()


class Datum(StaticCollection):
    """
    A class used to store information about a datum for either a 'GroundTruth' or a 'Prediction'.

    Attributes
    ----------
    uid : String
        The UID of the datum.
    metadata : Dictionary
        A dictionary of metadata that describes the datum.

    Examples
    --------
    >>> Datum(uid="uid1")
    >>> Datum(uid="uid1", metadata={})
    >>> Datum(uid="uid1", metadata={"foo": "bar", "pi": 3.14})
    """

    uid: String = String.symbolic(owner="datum", name="uid")
    metadata: Dictionary = Dictionary.symbolic(owner="datum", name="metadata")

    def __init__(
        self,
        uid: str,
        metadata: Optional[dict] = None,
    ):
        """
        Constructs a datum.

        Parameters
        ----------
        uid : str
            The UID of the datum.
        metadata : dict, optional
            A dictionary of metadata that describes the datum.
        """
        self.uid = String.definite(uid)
        self.metadata = Dictionary.definite(metadata if metadata else dict())
        super().__init__()

    def get_uid(self) -> str:
        """Safely get UID."""
        return self.uid.get_value()
