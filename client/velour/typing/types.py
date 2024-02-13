from typing import List, TypeVar, Union

from velour.schemas.filters import BinaryExpression, Filter

T = TypeVar("T")

FilterType = Union[
    Filter, List[Union[BinaryExpression, List[BinaryExpression]]], dict
]
