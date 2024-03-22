import io
import typing
import warnings
from base64 import b64decode, b64encode
from typing import Any, List, Optional, Union

import numpy as np
import PIL.Image

from valor.enums import TaskType
from valor.symbolic.atomics import (
    Dictionary,
    Equatable,
    Float,
    Listable,
    MultiPolygon,
    Nullable,
    Polygon,
    Spatial,
    String,
    Symbol,
    Variable,
    _get_atomic_type_by_name,
)


class Score(Float, Nullable):
    @classmethod
    def __validate__(cls, value: Any):
        if value is not None:
            Float.__validate__(value)
            if value < 0.0:
                raise ValueError("score must be non-negative")
            elif value > 1.0:
                raise ValueError("score must not exceed 1.0")


class TaskTypeEnum(String):
    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, TaskType):
            raise TypeError(
                f"Expected type '{TaskType}' received type '{type(value)}'"
            )

    @classmethod
    def decode_value(cls, value: str):
        return cls(TaskType(value))

    def encode_value(self) -> Any:
        return self.get_value().value


class BoundingBox(Polygon, Nullable):
    """
    Represents a bounding box defined by a 4-point polygon. Note that this does not need to be axis-aligned.

    Parameters
    ----------
    polygon : BasicPolygon or dict
        The 4-point polygon defining the bounding box. Can be a `BasicPolygon` object
        or a dictionary with the necessary information to create a `BasicPolygon`.

    Raises
    ------
    TypeError
        If `polygon` is not a `BasicPolygon` or cannot be converted to one.
    ValueError
        If the number of points in `polygon` is not equal to 4, making it invalid as a bounding box.

    Examples
    --------
    Create a BoundingBox from Points.
    Note that ordering is important to prevent self-intersection!
    >>> box1 = schemas.BoundingBox(
    ...     polygon=schemas.BasicPolygon(
    ...         points=[
    ...             schemas.Point(0,0),
    ...             schemas.Point(0,1),
    ...             schemas.Point(1,1),
    ...             schemas.Point(1,0),
    ...         ]
    ...     ),
    ... )

    Create a BoundingBox using extrema.
    >>> box2 = BoundingBox.from_extrema(
    ...     xmin=0, xmax=1,
    ...     ymin=0, ymax=1,
    ... )
    """

    @classmethod
    def __validate__(cls, value: Any):
        if value is not None:
            Polygon.__validate__(value)
            if len(value) != 1:
                raise ValueError("Bounding Box should not contain holes.")
            elif len(value[0]) != 5:
                raise ValueError(
                    "Bounding Box should consist of four unique points."
                )

    @classmethod
    def from_extrema(
        cls,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ):
        """
        Create a BoundingBox from extrema values.

        Parameters
        ----------
        xmin : float
            Minimum x-coordinate of the bounding box.
        xmax : float
            Maximum x-coordinate of the bounding box.
        ymin : float
            Minimum y-coordinate of the bounding box.
        ymax : float
            Maximum y-coordinate of the bounding box.

        Returns
        -------
        BoundingBox
            A BoundingBox created from the provided extrema values.
        """
        points = [
            [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin),
            ]
        ]
        return cls(value=points)


class BoundingPolygon(Polygon, Nullable):
    """
    Represents a polygon with a boundary and optional holes.

    Parameters
    ----------
    boundary : BasicPolygon or dict
        The outer boundary of the polygon. Can be a `BasicPolygon` object or a
        dictionary with the necessary information to create a `BasicPolygon`.
    holes : List[BasicPolygon], optional
        List of holes inside the polygon. Defaults to an empty list.

    Raises
    ------
    TypeError
        If `boundary` is not a `BasicPolygon` or cannot be converted to one.
        If `holes` is not a list, or an element in `holes` is not a `BasicPolygon`.

    Examples
    --------
    Create a polygon.
    >>> polygon1 = Polygon(
    ...     value = [[
    ...         (0, 1),
    ...         (1, 1),
    ...         (1, 0),
    ...         (0, 1),
    ...     ]],
    ... )

    Create a polygon with holes.
    >>> polygon2 = Polygon(
    ...     value = [[
    ...         (0, 1),
    ...         (1, 1),
    ...         (1, 0),
    ...         (0, 1),
    ...     ],[
    ...         (0, 0.5),
    ...         (0.5, 0.5),
    ...         (0.5, 0),
    ...         (0, 0.5),
    ...     ]],
    ... )
    """

    @classmethod
    def __validate__(cls, value: Any):
        if value is not None:
            Polygon.__validate__(value)


class Raster(Spatial, Nullable):
    """
    Represents a binary mask.

    Parameters
    ----------
    mask : str
        Base64-encoded string representing the raster mask.
    geometry : Union[Polygon, MultiPolygon], optional
        Option to input the raster as a geometry. Overrides the mask.

    Raises
    ------
    TypeError
        If `encoding` is not a string.

    Examples
    --------
    Generate a random mask.
    >>> import numpy.random
    >>> height = 640
    >>> width = 480
    >>> array = numpy.random.rand(height, width)

    Convert to binary mask.
    >>> mask = (array > 0.5)

    Create Raster.
    >>> Raster.from_numpy(mask)
    """

    @classmethod
    def __validate__(cls, value: Any):
        if value is not None:
            if not isinstance(value, dict):
                raise TypeError(
                    "Raster should contain a dictionary describing a mask and optionally a geometry."
                )
            elif set(value.keys()) != {"mask", "geometry"}:
                raise ValueError(
                    "Raster should be described by a dictionary with keys 'mask' and 'geometry'"
                )
            elif not isinstance(value["mask"], np.ndarray):
                raise TypeError(
                    f"Expected mask to have type '{np.ndarray}' receieved type '{value['mask']}'"
                )
            elif len(value["mask"].shape) != 2:
                raise ValueError("raster only supports 2d arrays")
            elif value["mask"].dtype != bool:
                raise ValueError(
                    f"Expecting a binary mask (i.e. of dtype bool) but got dtype {value['mask'].dtype}"
                )
            elif (
                value["geometry"] is not None
                and not Polygon.supports(value["geometry"])
                and not MultiPolygon.supports(value["geometry"])
            ):
                raise TypeError(
                    "Expected geometry to conform to either Polygon or MultiPolygon or be 'None'"
                )

    def encode_value(self) -> Any:
        value = self.get_value()
        if value is not None:
            f = io.BytesIO()
            PIL.Image.fromarray(value["mask"]).save(f, format="PNG")
            f.seek(0)
            mask_bytes = f.read()
            f.close()
            value = {
                "mask": b64encode(mask_bytes).decode(),
                "geometry": value["geometry"],
            }
        return value

    @classmethod
    def decode_value(cls, value: Any):
        if value is not None:
            mask_bytes = b64decode(value["mask"])
            with io.BytesIO(mask_bytes) as f:
                img = PIL.Image.open(f)
                value = {
                    "mask": np.array(img),
                    "geometry": value["geometry"],
                }
        return cls(value=value)

    @classmethod
    def from_numpy(cls, mask: np.ndarray):
        """
        Create a Raster object from a NumPy array.

        Parameters
        ----------
        mask : np.ndarray
            The 2D binary array representing the mask.

        Returns
        -------
        Raster

        Raises
        ------
        ValueError
            If the input array is not 2D or not of dtype bool.
        """
        return cls(value={"mask": mask, "geometry": None})

    @classmethod
    def from_geometry(
        cls,
        geometry: Union[Polygon, MultiPolygon],
        height: int,
        width: int,
    ):
        """
        Create a Raster object from a geometric mask.

        Parameters
        ----------
        geometry : Union[Polygon, MultiPolygon]
            Defines the bitmask as a geometry. Overrides any existing mask.
        height : int
            The intended height of the binary mask.
        width : int
            The intended width of the binary mask.

        Returns
        -------
        Raster
        """
        bitmask = np.full((int(height), int(width)), False)
        return cls(value={"mask": bitmask, "geometry": geometry.get_value()})

    @property
    def area(self):
        """
        Symbolic representation of area.
        """
        if not isinstance(self._value, Symbol):
            raise ValueError
        return Float.symbolic(name=self._value._name, attribute="area")

    @property
    def array(self) -> Optional[np.ndarray]:
        """
        The bitmask as a numpy array.

        Returns
        -------
        Optional[np.ndarray]
            A 2D binary array representing the mask if it exists.
        """
        if value := self.get_value():
            if value["geometry"] is not None:
                warnings.warn(
                    "array does not hold information as this is a geometry-based raster",
                    RuntimeWarning,
                )
            return value["mask"]
        warnings.warn("raster has no value", RuntimeWarning)
        return None


class Embedding(Spatial, Nullable):
    @classmethod
    def __validate__(cls, value: Any):
        if value is not None:
            if not isinstance(value, list):
                raise TypeError(
                    f"Expected type '{Optional[List[float]]}' received type '{type(value)}'"
                )
            elif len(value) < 1:
                raise ValueError(
                    "embedding should have at least one dimension"
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
    elif name == "list[label]":
        return Label.list()
    elif name == "annotation":
        return Annotation
    elif name == "list[annotation]":
        return Annotation.list()
    elif name == "datum":
        return Datum
    else:
        return _get_atomic_type_by_name(name)


class StaticCollection(Equatable, Listable):
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
                "StaticCollection's do not store an internal value."
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
            elif "__origin__" in v.__dict__:
                if v.__dict__["__origin__"] is list:
                    retval[k] = typing.get_args(v)[0].list()
                else:
                    raise NotImplementedError(
                        "Only 'typing.List' is supported."
                    )
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

    key: String
    value: String
    score: Score

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

    task_type: TaskTypeEnum
    labels: List[Label]
    metadata: Dictionary
    bounding_box: BoundingBox
    polygon: BoundingPolygon
    raster: Raster
    embedding: Embedding

    @classmethod
    def create(
        cls,
        task_type: TaskType,
        labels: Optional[List[Label]] = None,
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

    uid: String
    metadata: Dictionary

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
