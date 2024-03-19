import io
from base64 import b64decode, b64encode
from typing import Any, Union

import numpy as np
import PIL.Image

from valor.symbolic.atomics import Float
from valor.symbolic.geojson import MultiPolygon, Polygon
from valor.symbolic.modifiers import Nullable, Spatial, Symbol


class Score(Float, Nullable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return Float.supports(value) or value is None


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

    @classmethod
    def supports(cls, value: Any) -> bool:
        return Polygon.supports(value) or value is None


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
    def supports(cls, value: Any) -> bool:
        return Polygon.supports(value) or value is None


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
        if len(mask.shape) != 2:
            raise ValueError("raster currently only supports 2d arrays")
        if mask.dtype != bool:
            raise ValueError(
                f"Expecting a binary mask (i.e. of dtype bool) but got dtype {mask.dtype}"
            )
        f = io.BytesIO()
        PIL.Image.fromarray(mask).save(f, format="PNG")
        f.seek(0)
        mask_bytes = f.read()
        f.close()
        value = {
            "mask": b64encode(mask_bytes).decode(),
            "geometry": None,
        }
        return cls(value=value)

    @classmethod
    def from_geometry(
        cls,
        mask: Union[Polygon, MultiPolygon],
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
        r = cls.from_numpy(np.full((int(height), int(width)), False))
        if r._value is None:
            raise ValueError
        r._value["geometry"] = mask
        return r

    @classmethod
    def supports(cls, value: Any) -> bool:
        return (value is None) or (
            isinstance(value, dict)
            and set(value.keys()) == {"mask", "geometry"}
            and isinstance(value["mask"], str)
            and (
                isinstance(value["geometry"], (Polygon, MultiPolygon))
                or value["geometry"] is None
            )
        )

    @property
    def area(self):
        """
        Symbolic representation of area.
        """
        if not isinstance(self._value, Symbol):
            raise ValueError
        return Float.symbolic(name=self._value._name, attribute="area")

    @property
    def array(self) -> np.ndarray:
        """
        Convert the base64-encoded mask to a NumPy array.

        Returns
        -------
        np.ndarray
            A 2D binary array representing the mask.
        """
        if self.is_symbolic or not self._value:
            raise ValueError
        if self._value["geometry"] is not None:
            raise ValueError(
                "NumPy conversion is not supported for geometry-based rasters."
            )
        mask_bytes = b64decode(self._value["mask"])
        with io.BytesIO(mask_bytes) as f:
            img = PIL.Image.open(f)
            return np.array(img)


class Embedding(Spatial, Nullable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return (isinstance(value, list) and len(value) > 0) or value is None
