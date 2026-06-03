# structree — portable pytree/struct infrastructure
#
# Portions of this file are derived from Flax (https://github.com/google/flax),
# Copyright (c) 2024 The Flax Authors, licensed under the Apache License, Version 2.0.
#
# Modifications and additions:
# Copyright (c) 2025-2026 Pine Tree Labs, LLC, licensed under the MIT License.
#
# See LICENSE and LICENSES/Apache-2.0.txt at the repository root.

"""
Utilities for defining custom classes that can be used with tree transformations.

This module provides tools for creating structured data types that work
seamlessly with structree's tree functions. These tools are built on Python's
dataclasses with extensions for tree-specific behavior.

The module re-exports several names from the dataclasses module:

InitVar : Type annotation for init-only variables in dataclasses
    Used to mark fields that should be passed to __post_init__ but not stored.

fields : Function to retrieve fields of a dataclass
    Returns a list of Field objects representing the fields of the dataclass.

replace : Function to create a new dataclass instance with updated fields
    For tree nodes created with @struct, use the .replace() method instead.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from dataclasses import InitVar, fields, replace
from typing import Any, TypeVar

from typing_extensions import dataclass_transform

from ._registry import register_dataclass

__all__ = [
    "field",
    "struct",
    "InitVar",
    "is_struct",
    "fields",
    "replace",
]


T = TypeVar("T")


def field(
    static: bool = False,
    *,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dataclasses.Field:
    """
    Create a field specification with struct-related metadata.

    This function extends :py:func:`dataclasses.field()` with additional metadata
    to control how fields are treated in tree operations. Fields can be marked as
    static (metadata) or dynamic (data). Except for the ``static`` argument, all
    other arguments are passed directly to :py:func:`dataclasses.field()`.

    Parameters
    ----------
    static : bool, default=False
        If True, the field is treated as static metadata rather than dynamic
        data. Static fields are preserved during tree transformations but not
        included in the flattened representation.
    metadata : dict, optional
        Additional metadata to include in the field specification. This will be
        merged with the ``static`` setting.
    **kwargs : dict
        Additional keyword arguments passed to :py:func:`dataclasses.field()`.

    Returns
    -------
    field_object : dataclasses.Field
        A field specification with the appropriate metadata.

    Examples
    --------
    >>> import structree as st
    >>> import numpy as np
    >>>
    >>> @st.struct
    >>> class Vehicle:
    ...     position: np.ndarray
    ...     velocity: np.ndarray
    ...     mass: float = st.field(static=True, default=1000.0)

    See Also
    --------
    struct : Decorator for creating tree-compatible dataclasses
    register_dataclass : Register a dataclass as compatible with tree operations
    """
    f: dataclasses.Field = dataclasses.field(
        metadata=(metadata or {}) | {"static": static},
        **kwargs,
    )
    return f


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
def struct(cls: T | None = None, **kwargs) -> T | Callable:
    """
    Decorator to convert a class into a tree-compatible frozen dataclass.

    This decorator creates a structured data class that can be seamlessly used
    with structree's tree functions. The class will be registered with the tree
    system, allowing its instances to be flattened, mapped over, and transformed
    while preserving its structure.

    Parameters
    ----------
    cls : type, optional
        The class to convert into a tree-compatible dataclass.
    **kwargs : dict
        Additional keyword arguments passed to ``dataclasses.dataclass()``.
        By default, ``frozen=True`` is set unless explicitly overridden.

    Returns
    -------
    decorated_class : type
        The decorated class, now a frozen dataclass registered as
        tree-compatible.

    Notes
    -----
    The "frozen" attribute makes the class immutable. The ``replace()`` method
    allows you to create modified copies of the object with new values for
    specific fields.

    Fields are automatically classified as either "data" (dynamic values that
    change during operations) or "static" (configuration parameters). By
    default, all fields are treated as data unless marked with
    ``field(static=True)``.

    The decorated class:

    - Is frozen (immutable) by default
    - Has a ``replace()`` method for creating modified copies
    - Will be properly handled by ``flatten()``, ``map()``, etc.
    - Can be nested within other tree nodes (structs, dicts, tuples, etc.)

    Examples
    --------
    >>> import structree as st
    >>> import numpy as np
    >>>
    >>> @st.struct
    >>> class Vehicle:
    ...     position: np.ndarray
    ...     velocity: np.ndarray
    ...     mass: float = st.field(static=True, default=1000.0)
    >>>
    >>> car = Vehicle(np.zeros(2), np.array([10.0, 0.0]))
    >>> car2 = car.replace(position=np.array([5.0, 0.0]))
    >>> scaled = st.map(lambda x: x * 2, car)

    See Also
    --------
    field : Define fields with tree-specific metadata
    """
    # Support passing arguments to the decorator (e.g. @struct(kw_only=True))
    if cls is None:
        return functools.partial(struct, **kwargs)

    # check if already recognized as a tree node
    if "_is_struct" in cls.__dict__:
        return cls

    if "frozen" not in kwargs.keys():
        kwargs["frozen"] = True
    data_cls = dataclasses.dataclass(**kwargs)(cls)  # type: ignore
    meta_fields = []
    data_fields = []
    for field_info in dataclasses.fields(data_cls):
        is_static = field_info.metadata.get("static", False)
        if not is_static:
            data_fields.append(field_info.name)
        else:
            meta_fields.append(field_info.name)

    def replace(self, **updates) -> T:
        """Returns a new object replacing the specified fields with new values."""
        new: T = dataclasses.replace(self, **updates)
        return new

    data_cls.replace = replace

    register_dataclass(data_cls, data_fields, meta_fields)

    # add a marker flag to distinguish from regular dataclasses
    data_cls._is_struct = True  # type: ignore[attr-defined]

    return data_cls  # type: ignore


def is_struct(obj: Any) -> bool:
    """
    Check if an object is a registered struct class.

    This function determines whether an object was created using the
    :py:func:`struct` decorator, which indicates it has special handling for tree
    operations.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    is_node : bool
        ``True`` if the object is a struct created with the decorator, ``False``
        otherwise.

    See Also
    --------
    struct : Decorator for creating tree-compatible dataclasses
    """
    return hasattr(obj, "_is_struct")
