import typing
from typing import Any, Optional

from valor.enums import TaskType
from valor.schemas.symbolic.annotations import (
    BoundingBox,
    BoundingPolygon,
    Embedding,
    Raster,
    Score,
    TaskTypeEnum,
)
from valor.schemas.symbolic.atomics import Equatable, String, Symbol, Variable
from valor.schemas.symbolic.structures import (
    Dictionary,
    List,
    _get_atomic_type_by_name,
)


def _get_schema_type_by_name(name: str):
    name = name.lower()
    if name == "score":
        return Score
    elif name == "boundingbox":
        return BoundingBox
    elif name == "boundingpolygon":
        return BoundingPolygon
    elif name == "raster":
        return Raster
    elif name == "embedding":
        return Embedding
    elif name == "dictionary":
        return Dictionary
    elif name == "label":
        return Label
    elif "list[label]" in name:
        return List[Label]
    elif name == "annotation":
        return Annotation
    elif "list[annotation]" in name:
        return List[Annotation]
    elif name == "datum":
        return Datum
    else:
        return _get_atomic_type_by_name(name)


class StaticCollection(Equatable):
    """
    A static collection is a Variable that defines its contents by static attributes.
    """

    def __init__(self, **kwargs):
        symbol = kwargs.pop("symbol", None)
        class_name = type(self).__name__

        static_types = self._get_static_types()
        static_types_keys = set(static_types.keys())

        kwarg_keys = set(kwargs.keys())
        if not kwarg_keys.issubset(static_types_keys):
            raise TypeError(
                f"{class_name}() does not take the following keyword arguments '{kwarg_keys - static_types_keys}'"
            )

        if isinstance(symbol, Symbol):
            if kwargs:
                raise ValueError(
                    f"{class_name}() is being initialized as a symbol. The following keyword arguments are ignored: {kwarg_keys}"
                )
            for attr, obj in static_types.items():
                self.__setattr__(
                    attr, obj.symbolic(owner=class_name, name=attr)
                )
            super().__init__(value=None, symbol=symbol)
        else:
            for attr, obj in static_types.items():
                value = kwargs[attr] if attr in kwargs else None
                value = (
                    value
                    if isinstance(value, Variable)
                    else obj.definite(value)
                )
                self.__setattr__(attr, value)
            super().__init__(value=None, symbol=None)

    def __post_init__(self):
        pass

    @classmethod
    def definite(cls, **kwargs):
        kwargs["symbol"] = None
        return cls(**kwargs)

    @classmethod
    def __validate__(cls, value: Any):
        if value is not None:
            raise TypeError(
                "A StaticCollection does not store an internal value."
            )

    @classmethod
    def decode_value(cls, value: dict):
        return cls(**value)

    def encode_value(self):
        return {
            k: v.encode_value() for k, v in self._get_dynamic_values().items()
        }

    @classmethod
    def _get_static_types(cls):
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

    key: String = String.symbolic(owner="label", name="key")
    value: String = String.symbolic(owner="label", name="value")
    score: Score = Score.symbolic(owner="label", name="score")

    @classmethod
    def create(
        cls,
        key: str,
        value: str,
        score: Optional[float] = None,
    ):
        return cls.definite(
            key=key,
            value=value,
            score=score,
        )

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
    ...         Label(key="class", value="dog"),
    ...         Label(key="category", value="animal"),
    ...     ]
    ... )

    Object-Detection BoundingBox
    >>> annotation = Annotation.create(
    ...     task_type=TaskType.OBJECT_DETECTION,
    ...     labels=[Label(key="k1", value="v1")],
    ...     bounding_box=box2,
    ... )

    Object-Detection Polygon
    >>> annotation = Annotation.create(
    ...     task_type=TaskType.OBJECT_DETECTION,
    ...     labels=[Label(key="k1", value="v1")],
    ...     polygon=polygon1,
    ... )

    Object-Detection Raster
    >>> annotation = Annotation.create(
    ...     task_type=TaskType.OBJECT_DETECTION,
    ...     labels=[Label(key="k1", value="v1")],
    ...     raster=raster1,
    ... )

    Semantic-Segmentation Raster
    >>> annotation = Annotation.create(
    ...     task_type=TaskType.SEMANTIC_SEGMENTATION,
    ...     labels=[Label(key="k1", value="v1")],
    ...     raster=raster1,
    ... )

    Defining all supported annotation types for a given `task_type` is allowed!
    >>> Annotation.create(
    ...     task_type=TaskType.OBJECT_DETECTION,
    ...     labels=[Label(key="k1", value="v1")],
    ...     bounding_box=box1,
    ...     polygon=polygon1,
    ...     raster=raster1,
    ... )
    """

    task_type: TaskTypeEnum = TaskTypeEnum.symbolic(
        owner="annotation", name="task_type"
    )
    labels: List[Label] = List[Label].symbolic(
        owner="annotation", name="labels"
    )
    metadata: Dictionary = Dictionary.symbolic(
        owner="annotation", name="metadata"
    )
    bounding_box: BoundingBox = BoundingBox.symbolic(
        owner="annotation", name="box"
    )
    polygon: BoundingPolygon = BoundingPolygon.symbolic(
        owner="annotation", name="polygon"
    )
    raster: Raster = Raster.symbolic(owner="annotation", name="raster")
    embedding: Embedding = Embedding.symbolic(
        owner="annotation", name="embedding"
    )

    @classmethod
    def create(
        cls,
        task_type: TaskType,
        labels: Optional[typing.List[Label]] = None,
        metadata: Optional[dict] = None,
        bounding_box: Optional[BoundingBox] = None,
        polygon: Optional[BoundingPolygon] = None,
        raster: Optional[Raster] = None,
        embedding: Optional[Embedding] = None,
    ):
        return cls.definite(
            task_type=task_type,
            labels=labels,
            metadata=metadata,
            bounding_box=bounding_box,
            polygon=polygon,
            raster=raster,
            embedding=embedding,
        )


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

    uid: String = String.symbolic(owner="datum", name="uid")
    metadata: Dictionary = Dictionary.symbolic(owner="datum", name="metadata")

    @classmethod
    def create(
        cls,
        uid: str,
        metadata: Optional[dict] = None,
    ):
        return cls.definite(
            uid=uid,
            metadata=metadata,
        )

    def get_uid(self) -> str:
        return self.uid.get_value()
