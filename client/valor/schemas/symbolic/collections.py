import typing
from typing import Any, Optional

from valor.enums import TaskType
from valor.schemas.symbolic.types import (
    Bool,
    BoundingBox,
    BoundingPolygon,
    Date,
    DateTime,
    Duration,
    Embedding,
    Equatable,
    Float,
    Integer,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Nullable,
    Point,
    Polygon,
    Raster,
    Score,
    String,
    Symbol,
    TaskTypeEnum,
    Time,
    Variable,
)


def _get_atomic_type_by_value(other: Any):
    """
    Retrieves variable type using built-in type.

    Order of checking is very important as certain types are subsets of others.
    """
    if Bool.supports(other):
        return Bool
    elif String.supports(other):
        return String
    elif Integer.supports(other):
        return Integer
    elif Float.supports(other):
        return Float
    elif DateTime.supports(other):
        return DateTime
    elif Date.supports(other):
        return Date
    elif Time.supports(other):
        return Time
    elif Duration.supports(other):
        return Duration
    elif MultiPolygon.supports(other):
        return MultiPolygon
    elif Polygon.supports(other):
        return Polygon
    elif MultiLineString.supports(other):
        return MultiLineString
    elif LineString.supports(other):
        return LineString
    elif MultiPoint.supports(other):
        return MultiPoint
    elif Point.supports(other):
        return Point
    else:
        raise NotImplementedError(str(type(other).__name__))


def _get_atomic_type_by_name(name: str):
    """Retrieves variable type by name."""
    types_ = {
        "bool": Bool,
        "string": String,
        "integer": Integer,
        "float": Float,
        "datetime": DateTime,
        "date": Date,
        "time": Time,
        "duration": Duration,
        "multipolygon": MultiPolygon,
        "polygon": Polygon,
        "multilinestring": MultiLineString,
        "linestring": LineString,
        "multipoint": MultiPoint,
        "point": Point,
    }
    type_ = types_.get(name.lower(), None)
    if type_ is None:
        raise NotImplementedError(name)
    return type_


T = typing.TypeVar("T", bound=Variable)


class List(typing.Generic[T], Equatable):
    """
    List is both a method of typing and a class-factory.

    The '__class_getitem__' classmethod produces strongly-typed ValueLists.

    Examples
    --------
    >>> x = List[String](["foo", "bar"])
    """

    _registered_classes = dict()

    @classmethod
    def __class_getitem__(cls, item_class: typing.Type[T]):

        if item_class in cls._registered_classes:
            return cls._registered_classes[item_class]

        class ValueList(Equatable):
            """
            Strongly-typed variable list.

            Parameters
            ----------
            value : List[T], optional
                A list of items with type T.
            symbol : Symbol, optional
                A symbolic representation.
            """

            def __init__(
                self,
                value: Optional[Any] = None,
                symbol: Optional[Symbol] = None,
            ):
                if value is not None:
                    if not isinstance(value, list):
                        raise TypeError(
                            f"Expected a value with type 'List[{item_class.__name__}]' but received type '{type(value).__name__}'"
                        )
                    vlist = []
                    for item in value:
                        if isinstance(item, item_class):
                            vlist.append(item)
                        elif isinstance(item, dict) and set(item.keys()) != {
                            "type",
                            "value",
                        }:
                            vlist.append(item_class.definite(**item))
                        else:
                            vlist.append(item_class.definite(item))
                    value = vlist
                super().__init__(value=value, symbol=symbol)

            @classmethod
            def definite(cls, value: Any):
                """Initialize variable with a value."""
                if value is None:
                    value = list()
                return cls(value=value)

            @classmethod
            def symbolic(
                cls,
                name: Optional[str] = None,
                key: Optional[str] = None,
                attribute: Optional[str] = None,
                owner: Optional[str] = None,
            ):
                """Initialize variable as a symbol."""
                if name is None:
                    name = f"list[{item_class.__name__.lower()}]"
                return super().symbolic(name, key, attribute, owner)

            @classmethod
            def __validate__(cls, value: list):
                """Validate typing."""
                if not isinstance(value, list):
                    raise TypeError(
                        f"Expected type '{list}' received type '{type(value)}'"
                    )
                for element in value:
                    if not item_class.supports(element) and not issubclass(
                        type(element), Variable
                    ):
                        raise TypeError(
                            f"Expected list elements with type '{item_class}' received type '{type(element)}'"
                        )

            @classmethod
            def decode_value(cls, value: Any):
                """Decode object from JSON compatible dictionary."""
                if not value:
                    return cls(value=[])
                return cls(
                    value=[
                        item_class.decode_value(element) for element in value
                    ]
                )

            def encode_value(self):
                """Encode object to JSON compatible dictionary."""
                return [element.encode_value() for element in self.get_value()]

            def to_dict(self) -> dict:
                """Encode variable to a JSON-compatible dictionary."""
                if isinstance(self._value, Symbol):
                    return self._value.to_dict()
                else:
                    return {
                        "type": f"list[{item_class.__name__.lower()}]",
                        "value": self.encode_value(),
                    }

            def __getitem__(self, __key: int) -> T:
                return self.get_value()[__key]

            def __setitem__(self, __key: int, __value: Any):
                vlist = self.get_value()
                vlist[__key] = item_class.preprocess(__value)

            def __iter__(self) -> typing.Iterator[T]:
                return iter([element for element in self.get_value()])

            def __len__(self):
                return len(self.get_value())

            @staticmethod
            def get_element_type():
                return item_class

        cls._registered_classes[item_class] = ValueList
        return ValueList

    def __getitem__(self, __key: int) -> T:
        raise NotImplementedError

    def __setitem__(self, __key: int, __value: Any):
        raise NotImplementedError

    def __iter__(self) -> typing.Iterator[T]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class DictionaryValue(Nullable):
    """Helper class for routing dictionary expressions."""

    def __init__(self, symbol: Symbol):
        if not isinstance(symbol, Symbol):
            raise ValueError(
                "DictionaryValue should only be initialized as a symbol."
            )
        if symbol._attribute:
            raise ValueError(
                "DictionaryValue symbol should not contain attribute."
            )
        if not symbol._key:
            raise ValueError("DictionaryValue symbol should contain key.")
        super().__init__(value=None, symbol=symbol)

    @classmethod
    def definite(cls, value: Any):
        """Assigning a value is not supported."""
        raise NotImplementedError(
            "DictionaryValue should only be initialized as a symbol."
        )

    @classmethod
    def symbolic(
        cls,
        name: Optional[str] = None,
        key: Optional[str] = None,
        attribute: Optional[str] = None,
        owner: Optional[str] = None,
    ):
        """Initialize variable as a symbol."""
        return cls(
            Symbol(
                name=name if name else cls.__name__.lower(),
                key=key,
                attribute=attribute,
                owner=owner,
            )
        )

    def __eq__(self, other: Any):
        return self._generate(fn="__eq__", other=other)

    def __ne__(self, other: Any):
        return self._generate(fn="__ne__", other=other)

    def __gt__(self, other: Any):
        return self._generate(fn="__gt__", other=other)

    def __ge__(self, other: Any):
        return self._generate(fn="__ge__", other=other)

    def __lt__(self, other: Any):
        return self._generate(fn="__lt__", other=other)

    def __le__(self, other: Any):
        return self._generate(fn="__le__", other=other)

    def intersects(self, other: Any):
        return self._generate(fn="intersects", other=other)

    def inside(self, other: Any):
        return self._generate(fn="inside", other=other)

    def outside(self, other: Any):
        return self._generate(fn="outside", other=other)

    def is_none(self):
        return super().is_none()

    def is_not_none(self):
        return super().is_not_none()

    @property
    def area(self):
        """Returns area attribute."""
        symbol = self.get_symbol()
        return Float.symbolic(
            owner=symbol._owner,
            name=symbol._name,
            key=symbol._key,
            attribute="area",
        )

    def _generate(self, other: Any, fn: str):
        """Generate expression."""
        if isinstance(other, Variable):
            obj = type(other)
        else:
            obj = _get_atomic_type_by_value(other)
        symbol = self.get_symbol()
        sym = obj.symbolic(
            owner=symbol._owner,
            name=symbol._name,
            attribute=symbol._attribute,
            key=symbol._key,
        )
        return sym.__getattribute__(fn)(other)


class Dictionary(Equatable):
    """
    Symbolic implementation of the built-in type 'dict'.

    Parameters
    ----------
    value : Dict[str, Any], optional
        A dictionary of items.
    symbol : Symbol, optional
        A symbolic representation.

    Examples
    --------
    >>> v = Dictionary({"k1": "v1", "k2": 3.14})
    >>> s = Dictionary.symbolic(name="some_var")

    # Create an equality expression.
    >>> s["k1"] == v["k1"]
    Eq(Symbol(name='some_var', key='k1'), 'v1')
    """

    def __init__(
        self,
        value: Optional[typing.Dict[str, Any]] = None,
        symbol: Optional[Symbol] = None,
    ):
        if isinstance(value, dict):
            _value = dict()
            for k, v in value.items():
                if v is None:
                    raise ValueError(
                        "Dictionary does not accept 'None' as a value."
                    )
                elif isinstance(v, Variable):
                    if v.is_symbolic:
                        raise ValueError(
                            "Dictionary does not accpet symbols as values."
                        )
                    _value[k] = v
                else:
                    _value[k] = _get_atomic_type_by_value(v).definite(v)
            value = _value
        super().__init__(value, symbol)

    @classmethod
    def definite(
        cls,
        value: Optional[typing.Dict[str, Any]] = None,
    ):
        """Initialize variable with a value."""
        value = value if value else dict()
        return super().definite(value)

    @classmethod
    def __validate__(cls, value: Any):
        """Validate typing."""
        if not isinstance(value, dict):
            raise TypeError(
                f"Expected type '{dict}' received type '{type(value)}'"
            )
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError("Dictionary keys must be of type 'str'")

    @classmethod
    def decode_value(cls, value: dict) -> Any:
        """Decode object from JSON compatible dictionary."""
        return cls(
            {
                k: _get_atomic_type_by_name(v["type"]).decode_value(v["value"])
                for k, v in value.items()
            }
        )

    def encode_value(self) -> dict:
        """Encode object to JSON compatible dictionary."""
        return {k: v.to_dict() for k, v in self.items()}

    def __getitem__(self, key: str):
        if self.is_symbolic:
            symbol = self.get_symbol()
            return DictionaryValue.symbolic(
                owner=symbol._owner,
                name=symbol._name,
                attribute=None,
                key=key,
            )
        else:
            value = self.get_value()
            if not value:
                raise KeyError(key)
            return value[key]

    def __setitem__(self, key: str, value: Any):
        if not isinstance(value, Variable):
            obj = _get_atomic_type_by_value(value)
            value = obj.definite(value)
        self.get_value()[key] = value

    def __len__(self) -> int:
        return len(self.get_value())

    def pop(self, key: str):
        value = self.get_value()
        if not value:
            raise KeyError(key)
        return value.pop(key)

    def items(self):
        if isinstance(self._value, Symbol):
            raise NotImplementedError("Variable is symbolic")
        return self._value.items() if self._value else dict.items({})


def _get_schema_type_by_name(name: str):
    types_ = {
        "score": Score,
        "boundingbox": BoundingBox,
        "boundingpolygon": BoundingPolygon,
        "raster": Raster,
        "embedding": Embedding,
        "dictionary": Dictionary,
        "label": Label,
        "annotation": Annotation,
        "datum": Datum,
    }
    name = name.lower()
    if name in types_:
        return types_[name]
    elif "list[label]" in name:
        return List[Label]
    elif "list[annotation]" in name:
        return List[Annotation]
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
    key
    value
    score

    Examples
    --------
    >>> Label(key="k1", value="v1")
    >>> Label(key="k1", value="v1", score=None)
    >>> Label(key="k1", value="v1", score=0.9)
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
    task_type
    labels
    metadata
    bounding_box
    polygon
    raster
    embedding

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
        polygon: Polygon, optional
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
    uid
    metadata

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
            The UID of the `Datum`.
        metadata : dict
            A dictionary of metadata that describes the `Datum`.
        """
        return cls.definite(
            uid=uid,
            metadata=metadata,
        )

    def get_uid(self) -> str:
        """Safely get UID."""
        return self.uid.get_value()
