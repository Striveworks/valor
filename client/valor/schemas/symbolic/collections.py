import typing
from typing import Any, Optional

from valor.enums import TaskType
from valor.schemas.symbolic.types import (
    Box,
    Dictionary,
    Embedding,
    Equatable,
    Float,
    List,
    Nullable,
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
                if not isinstance(value, obj):
                    if issubclass(obj, StaticCollection):
                        if not isinstance(value, dict):
                            raise TypeError(
                                f"{class_name}.{attr} expected a value with type '{obj.__name__}' received value with type '{type(value).__name__}'"
                            )
                        value = obj.definite(**value)
                    else:
                        value = obj.definite(value)
                self.__setattr__(attr, value)
            self.__post_init__()
            super().__init__(value=None, symbol=None)

    def __post_init__(self):
        pass

    @classmethod
    def definite(cls, *args, **kwargs):
        """Initialize variable with a value."""
        if args:
            raise TypeError(f"{cls.__name__}() takes no positional arguments.")
        kwargs["symbol"] = None
        return cls(**kwargs)

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
    def _get_static_types(cls):
        """Returns any static members that inherit from 'Variable'."""
        fields = getattr(cls, "__annotations__", dict())
        retval = dict()
        for k, v in fields.items():
            if isinstance(v, type) and issubclass(v, Variable):
                retval[k] = v
            # elif (
            #     '__origin__' in v.__dict__
            #     and '__args__' in v.__dict__
            # ):
            #     origin = v.__dict__.get('__origin__')
            #     args = v.__dict__.get('__args__')
            #     if not origin or origin not in {List, Nullable}:
            #         raise NotImplementedError(origin)
            #     elif not args or len(args) != 1:
            #         raise NotImplementedError(args)
            #     retval[k] = origin[args[0]]
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
    score: Nullable[Float] = Nullable[Float].symbolic(
        owner="label", name="score"
    )

    @classmethod
    def create(
        cls,
        key: str,
        value: str,
        score: Optional[float] = None,
        **_,
    ):
        """
        Constructs a label.

        Parameters
        ----------
        key : str
            The class key of the label.
        value : str
            The class value of the label.
        score : float, optional
            The score associated with the label (if applicable).
        """
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

    Attributes
    ----------
    task_type: TaskTypeEnum
        The task type associated with the `Annotation`.
    labels: List[Label], optional
        A list of labels to use for the `Annotation`.
    metadata: Dictionary
        A dictionary of metadata that describes the `Annotation`.
    bounding_box: BoundingBox
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
    ...     bounding_box=BoundingBox(...),
    ...     polygon=BoundingPolygon(...),
    ...     raster=Raster(...),
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
    bounding_box: Nullable[Box] = Nullable[Box].symbolic(
        owner="annotation", name="box"
    )
    polygon: Nullable[Polygon] = Nullable[Polygon].symbolic(
        owner="annotation", name="polygon"
    )
    raster: Nullable[Raster] = Nullable[Raster].symbolic(
        owner="annotation", name="raster"
    )
    embedding: Nullable[Embedding] = Nullable[Embedding].symbolic(
        owner="annotation", name="embedding"
    )

    @classmethod
    def create(
        cls,
        task_type: TaskType,
        labels: Optional[typing.List[Label]] = None,
        metadata: Optional[dict] = None,
        bounding_box: Optional[Box] = None,
        polygon: Optional[Polygon] = None,
        raster: Optional[Raster] = None,
        embedding: Optional[Embedding] = None,
        **_,
    ):
        """
        Constructs an annotation.

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
        polygon: BoundingPolygon, optional
            A polygon to assign to the `Annotation`.
        raster: Raster, optional
            A raster to assign to the `Annotation`.
        embedding: List[float], optional
            An embedding, described by a list of values with type float and a maximum length of 16,000.
        """
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

    @classmethod
    def create(
        cls,
        uid: str,
        metadata: Optional[dict] = None,
        **_,
    ):
        """
        Constructs a datum.

        Parameters
        ----------
        uid : str
            The UID of the datum.
        metadata : dict
            A dictionary of metadata that describes the datum.
        """
        return cls.definite(
            uid=uid,
            metadata=metadata,
        )

    def get_uid(self) -> str:
        """Safely get UID."""
        return self.uid.get_value()
