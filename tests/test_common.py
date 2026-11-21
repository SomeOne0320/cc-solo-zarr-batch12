from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, get_args

import numpy as np
import pytest

from zarr.core.common import (
    ANY_ACCESS_MODE,
    AccessModeLiteral,
    normalize_axis,
    parse_name,
    parse_shapelike,
    product,
)
from zarr.core.config import parse_indexing_order

if TYPE_CHECKING:
    from typing import Any, Literal


@pytest.mark.parametrize("data", [(0, 0, 0, 0), (1, 3, 4, 5, 6), (2, 4)])
def test_product(data: tuple[int, ...]) -> None:
    assert product(data) == np.prod(data)


def test_access_modes() -> None:
    """
    Test that the access modes type and variable for run-time checking are equivalent.
    """
    assert set(ANY_ACCESS_MODE) == set(get_args(AccessModeLiteral))


# todo: test
def test_concurrent_map() -> None: ...


# todo: test
def test_to_thread() -> None: ...


# todo: test
def test_enum_names() -> None: ...


# todo: test
def test_parse_enum() -> None: ...


@pytest.mark.parametrize("data", [("foo", "bar"), (10, 11)])
def test_parse_name_invalid(data: tuple[Any, Any]) -> None:
    observed, expected = data
    if isinstance(observed, str):
        with pytest.raises(ValueError, match=f"Expected '{expected}'. Got {observed} instead."):
            parse_name(observed, expected)
    else:
        with pytest.raises(
            TypeError, match=f"Expected a string, got an instance of {type(observed)}."
        ):
            parse_name(observed, expected)


@pytest.mark.parametrize("data", [("foo", "foo"), ("10", "10")])
def test_parse_name_valid(data: tuple[Any, Any]) -> None:
    observed, expected = data
    assert parse_name(observed, expected) == observed


@pytest.mark.parametrize("data", [0, 1, "hello", "f"])
def test_parse_indexing_order_invalid(data: Any) -> None:
    with pytest.raises(ValueError, match="Expected one of"):
        parse_indexing_order(data)


@pytest.mark.parametrize("data", ["C", "F"])
def parse_indexing_order_valid(data: Literal["C", "F"]) -> None:
    assert parse_indexing_order(data) == data


@pytest.mark.parametrize("data", [lambda v: v, slice(None)])
def test_parse_shapelike_invalid_single_type(data: Any) -> None:
    """
    Test that we get the expected error message when passing in a value that is not an integer
    or an iterable of integers.
    """
    with pytest.raises(TypeError, match="Expected an integer or an iterable of integers."):
        parse_shapelike(data)


def test_parse_shapelike_invalid_single_value() -> None:
    """
    Test that we get the expected error message when passing in a negative integer.
    """
    with pytest.raises(ValueError, match="Expected a non-negative integer."):
        parse_shapelike(-1)


@pytest.mark.parametrize("data", ["shape", ("0", 1, 2, 3), {"0": "0"}, ((1, 2), (2, 2)), (4.0, 2)])
def test_parse_shapelike_invalid_iterable_types(data: Any) -> None:
    """
    Test that we get the expected error message when passing in an iterable containing
    non-integer elements
    """
    with pytest.raises(TypeError, match="Expected an iterable of integers"):
        parse_shapelike(data)


@pytest.mark.parametrize("data", [(1, 2, 3, -1), (-10,)])
def test_parse_shapelike_invalid_iterable_values(data: Any) -> None:
    """
    Test that we get the expected error message when passing in an iterable containing negative
    integers
    """
    with pytest.raises(ValueError, match="Expected all values to be non-negative."):
        parse_shapelike(data)


@pytest.mark.parametrize(
    "data", [range(10), [0, 1, 2, np.uint64(3)], (3, 4, 5), (), 1, np.uint8(1)]
)
def test_parse_shapelike_valid(data: Iterable[int] | int) -> None:
    if isinstance(data, Iterable):
        expected = tuple(data)
    else:
        expected = (data,)
    assert parse_shapelike(data) == expected


# todo: more dtypes
@pytest.mark.parametrize("data", [("uint8", np.uint8), ("float64", np.float64)])
def parse_dtype(data: tuple[str, np.dtype[Any]]) -> None:
    unparsed, parsed = data
    assert parse_dtype(unparsed) == parsed


# todo: figure out what it means to test this
def test_parse_fill_value() -> None: ...


@pytest.mark.parametrize(
    ("axis", "ndim", "expected"),
    [
        (0, 3, 0),
        (2, 3, 2),
        (-1, 3, 2),
        (-3, 3, 0),
        (np.int64(-2), 3, 1),
    ],
)
def test_normalize_axis_valid(axis: int, ndim: int, expected: int) -> None:
    assert normalize_axis(axis, ndim) == expected


@pytest.mark.parametrize(
    ("axis", "ndim"),
    [(3, 3), (-4, 3), (10, 2), (-1, 0), (0, 0)],
)
def test_normalize_axis_out_of_bounds(axis: int, ndim: int) -> None:
    with pytest.raises(np.exceptions.AxisError):
        normalize_axis(axis, ndim)


@pytest.mark.parametrize("axis", [1.0, "0", None, slice(None)])
def test_normalize_axis_bad_type(axis: Any) -> None:
    with pytest.raises(TypeError, match="axis must be an integer"):
        normalize_axis(axis, 2)


def test_normalize_axis_rejects_bool() -> None:
    with pytest.raises(TypeError, match="axis must be an integer"):
        normalize_axis(True, 2)
