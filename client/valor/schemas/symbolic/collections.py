from typing import Any, Dict, List, Optional, Union

import numpy as np

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
    TaskTypeEnum,
    Variable,
    get_type_by_name,
)


def _get_schema_type_by_name(name: str):
    types_ = {
        "label": Label,
        "annotation": Annotation,
        "datum": Datum,
    }
    return get_type_by_name(name=name, additional_types=types_)


def _convert_simple_variables_to_standard_types(var: Any):
    """Converts a variable to a standard type. This operates recursively.
    in the case that the variable represents a dictionary
    """
    if isinstance(var, StaticCollection):
        return var
    if isinstance(var, Variable):
        val = var.get_value()
        if isinstance(val, (str, int, float, bool, type(None))):
            var = val
    if isinstance(var, (dict, Dictionary)):
        return {
            k: _convert_simple_variables_to_standard_types(v)
            for k, v in var.items()
        }
    return var


class StaticCollection(Equatable):
    """
    A static collection is a Variable that defines its contents by static attributes.
    """

    def __init__(self, **kwargs):
        if set(kwargs.keys()) != set(self._get_static_types().keys()):
            kwarg_keys = set(kwargs.keys())
            static_keys = set(self._get_static_types().keys())
            raise ValueError(
                f"Expected the following keyword arguments '{static_keys}'. Received '{kwarg_keys}'."
            )
        # import pdb

        # pdb.set_trace()
        for k, v in kwargs.items():
            setattr(self, k, v)
        super().__init__(value=None)

    @classmethod
    def nullable(cls, *args, **kwargs):
        """
        Initializes variable with an optional value.
        """
        raise NotImplementedError(
            "Static collections do not define 'nullable'."
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
        Initializes the object and its attributes as symbols.

        Parameters
        ----------
        name : str, optional
            The name of the symbol.
        key : str, optional
            The key of the value if its a dictionary element.
        attribute : str, optional
            The name of a an attribute this symbol represents.
        owner : str, optional
            The name of an object that this symbol belongs to.
        """
        obj = super().symbolic(name, key, attribute, owner)
        for __name, __type in obj._get_static_types().items():
            if not issubclass(__type, Variable):
                raise TypeError
            setattr(
                obj,
                __name,
                __type.symbolic(owner=cls.__name__.lower(), name=__name),
            )
        return obj

    @staticmethod
    def formatting() -> Dict[str, Any]:
        """Attribute format mapping."""
        return dict()

    def format(self, __name: str, __value: Any) -> Any:
        """Either formats or passes throught a name-value pair."""
        if __name in self._get_static_types():
            __type = self._get_static_types()[__name]
            if not isinstance(__value, __type):
                __fmt = (
                    self.formatting()[__name]
                    if __name in self.formatting()
                    else __type
                )
                if issubclass(__type, StaticCollection):
                    return __fmt(**__value)
                else:
                    return __fmt(__value)
        return __value

    def __setattr__(self, __name: str, __value: Any) -> None:
        super().__setattr__(__name, self.format(__name, __value))

    def __getattribute__(self, __name: str) -> Any:
        ret = super().__getattribute__(__name)
        if isinstance(ret, Variable) and ret.is_value:
            return _convert_simple_variables_to_standard_types(ret)
            # ret = ret.get_value()
            # if isinstance(ret, dict):
            #     import pdb

            #     pdb.set_trace()
        return ret

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
        kwargs = dict()
        types = cls._get_static_types()
        for k, v in value.items():
            type_ = types.get(k)
            if type_ and issubclass(type_, Variable):
                kwargs[k] = type_.decode_value(v)
            else:
                kwargs[k] = v
        return cls(**kwargs)

    def encode_value(self):
        """Encode object to JSON compatible dictionary."""
        # TODO: make this less hacky
        return {
            k: (
                v.encode_value()
                if v is not None and hasattr(v, "encode_value")
                else v
            )
            for k, v in self._get_dynamic_values().items()
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
        *,
        key: str,
        value: str,
        score: Union[float, np.floating, None] = None,
    ):
        """
        Initializes an instance of a label.

        Attributes
        ----------
        key : str
            The class label key.
        value : str
            The class label value.
        score : float, optional
            The label score.
        """
        super().__init__(key=key, value=value, score=score)

    @staticmethod
    def formatting() -> Dict[str, Any]:
        """Attribute format mapping."""
        return {
            "score": Float.nullable,
        }

    def tuple(self):
        """
        Defines how the `Label` is turned into a tuple.

        Returns
        ----------
        tuple
            A tuple of the `Label's` arguments.
        """
        return (self.key, self.value, self.score)


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
    bounding_box: Box = Box.symbolic(owner="annotation", name="bounding_box")
    polygon: Polygon = Polygon.symbolic(owner="annotation", name="polygon")
    raster: Raster = Raster.symbolic(owner="annotation", name="raster")
    embedding: Embedding = Embedding.symbolic(
        owner="annotation", name="embedding"
    )

    def __init__(
        self,
        *,
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
        super().__init__(
            task_type=task_type,
            metadata=metadata if metadata else dict(),
            labels=labels if labels else list(),
            bounding_box=bounding_box,
            polygon=polygon,
            raster=raster,
            embedding=embedding,
        )

    @staticmethod
    def formatting() -> Dict[str, Any]:
        """Attribute format mapping."""
        return {
            "bounding_box": Box.nullable,
            "polygon": Polygon.nullable,
            "raster": Raster.nullable,
            "embedding": Embedding.nullable,
        }


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
        *,
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
        super().__init__(uid=uid, metadata=metadata if metadata else dict())
