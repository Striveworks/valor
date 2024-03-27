import io
import typing
import warnings
from base64 import b64decode, b64encode
from typing import Any, Optional, Union

import numpy as np
import PIL.Image

from valor.enums import TaskType
from valor.schemas.symbolic.atomics import (
    Float,
    MultiPolygon,
    Nullable,
    Polygon,
    Spatial,
    String,
    Symbol,
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
                f"Expected value with type '{TaskType.__name__}' received type '{type(value).__name__}'"
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
        return Float.symbolic(
            owner=self._value._owner,
            name=self._value._name,
            key=self._value._key,
            attribute="area",
        )

    @property
    def array(self) -> Optional[np.ndarray]:
        """
        The bitmask as a numpy array.

        Returns
        -------
        Optional[np.ndarray]
            A 2D binary array representing the mask if it exists.
        """
        value = self.get_value()
        if value is not None:
            if value["geometry"] is not None:
                warnings.warn(
                    "array does not hold information as this is a geometry-based raster",
                    RuntimeWarning,
                )
            return value["mask"]
        warnings.warn("raster has no value", RuntimeWarning)
        return None

    @property
    def geometry(self) -> Optional[Union[Polygon, MultiPolygon]]:
        """
        The optional geometry that describes the bitmask.

        Returns
        -------
        Polygon | MultiPolygon | None
            The geometry if it exists.
        """
        value = self.get_value()
        if value is not None:
            return value["geometry"]
        warnings.warn("raster has no value", RuntimeWarning)
        return None

    @property
    def height(self) -> Optional[int]:
        array = self.array
        if array is not None:
            return array.shape[0]
        return None

    @property
    def width(self) -> Optional[int]:
        array = self.array
        if array is not None:
            return array.shape[1]
        return None


class Embedding(Spatial, Nullable):
    @classmethod
    def __validate__(cls, value: Any):
        if value is not None:
            if not isinstance(value, list):
                raise TypeError(
                    f"Expected type '{Optional[typing.List[float]]}' received type '{type(value)}'"
                )
            elif len(value) < 1:
                raise ValueError(
                    "embedding should have at least one dimension"
                )
