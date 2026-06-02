# structree — portable pytree/struct infrastructure
#
# Portions of this file are derived from JAX (https://github.com/jax-ml/jax),
# Copyright (c) 2021 The JAX Authors, licensed under the Apache License, Version 2.0.
#
# Modifications and additions:
# Copyright (c) 2025-2026 Pine Tree Labs, LLC, licensed under the MIT License.
#
# See LICENSE and LICENSES/Apache-2.0.txt at the repository root.

"""Creation and management of types recognized as tree nodes."""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, TypeVar

Typ = TypeVar("Typ", bound=type[Any])


class _RegistryEntry(NamedTuple):
    to_iter: Callable[[Any], tuple[tuple[Any, ...], Any]]
    from_iter: Callable[[Any, tuple[Any, ...]], Any]


_registry: dict[type, _RegistryEntry] = {}


def unzip2(pairs):
    lst1, lst2 = [], []
    for x1, x2 in pairs:
        lst1.append(x1)
        lst2.append(x2)
    return lst1, lst2


def register_struct(ty: Any, to_iter: Callable, from_iter: Callable) -> None:
    """
    Register a custom type as a tree-compatible node.

    This function allows custom types to be recognized and processed by the
    tree functions. You provide functions that convert between your type and
    its components.

    Parameters
    ----------
    ty : type
        The type to register as a tree node.
    to_iter : callable
        A function that accepts an instance of type ``ty`` and returns a tuple
        of ``(children, aux_data)``, where:

        - ``children`` is an iterable of the tree node's children

        - ``aux_data`` is any auxiliary metadata needed to reconstruct the node\
        but not part of the tree structure itself

    from_iter : callable
        A function that accepts ``aux_data`` and an iterable of children and
        returns a reconstructed instance of type ``ty``.

    Returns
    -------
    None

    Notes
    -----
    The ``to_iter`` function should extract the relevant parts of your data
    structure, and the ``from_iter`` function should be able to reconstruct it
    exactly.

    Usually, instead of using this function directly, you'll want to use the
    ``@struct`` decorator for classes, which automatically handles registration
    for dataclass-like structures. This function is used internally to register
    the decorated classes. It is also available as an alternative interface for
    low-level control of flattening/unflattening behavior and static data for
    custom classes.

    The ``(children, aux_data)`` convention is intentionally identical to JAX's
    pytree node registration, so flattened structree data lines up element for
    element with the JAX/Flax pytree contract.

    See Also
    --------
    struct : Decorator for creating tree-compatible dataclasses
    register_dataclass : Register a dataclass as a struct
    """
    _registry[ty] = _RegistryEntry(to_iter, from_iter)


register_struct(None, lambda x: (None, None), lambda _, xs: None)
register_struct(tuple, lambda t: (t, None), lambda _, xs: tuple(xs))
register_struct(list, lambda lst: (lst, None), lambda _, xs: list(xs))


# dict
def _dict_to_iter(d: dict):
    keys, vals = unzip2(sorted(d.items()))
    return map(tuple, (vals, keys))


def _dict_from_iter(keys, vals):
    return dict(zip(keys, vals, strict=False))


register_struct(dict, _dict_to_iter, _dict_from_iter)


# OrderedDict
def _od_from_iter(keys, vals):
    return OrderedDict(zip(keys, vals, strict=False))


register_struct(OrderedDict, _dict_to_iter, _od_from_iter)


def register_dataclass(
    nodetype: Typ,
    data_fields: Sequence[str] | None = None,
    meta_fields: Sequence[str] | None = None,
    drop_fields: Sequence[str] = (),
) -> Typ:
    """
    Register a dataclass as tree-compatible with customized field handling.

    This function registers a dataclass type as a tree node, with control over
    which fields are treated as leaf data versus metadata. Fields marked as
    metadata are excluded from transformations and treated as static
    configuration.

    Parameters
    ----------
    nodetype : type
        The dataclass type to register as tree-compatible.
    data_fields : sequence of str, optional
        Names of fields that should be treated as data (leaf values). If None
        and the type is a dataclass, fields are inferred based on metadata.
    meta_fields : sequence of str, optional
        Names of fields that should be treated as metadata. If None and the
        type is a dataclass, fields are inferred based on metadata.
    drop_fields : sequence of str, optional
        Names of fields to exclude from both data and metadata categories.

    Returns
    -------
    nodetype : type
        The input type, now registered as tree-compatible.

    Notes
    -----
    When to use:

    - For fine-grained control over how dataclass fields are handled in tree ops
    - When you need some fields treated as static configuration rather than data
    - For advanced customization of tree behavior for complex data models

    Usually, instead of using this function directly, you'll want to use the
    ``@struct`` decorator which handles registration automatically and allows
    field classification via the ``field(static=True)`` parameter. This function
    is mainly used internally to register the decorated classes.

    Data fields are included when flattening a tree and are considered leaf
    values that can be transformed. Meta fields are static configuration not
    included in transformations but preserved during reconstruction.

    Raises
    ------
    TypeError
        If ``data_fields`` and ``meta_fields`` aren't both specified when either
        is specified, or if they are both None and the type is not a dataclass.
    ValueError
        If the specified fields don't match the actual dataclass fields with
        ``init=True``.

    See Also
    --------
    struct : Decorator for creating tree-compatible classes
    field : Function to create fields with metadata for tree behavior
    register_struct : Register any custom type as a tree-compatible node
    """
    if data_fields is None or meta_fields is None:
        if (data_fields is None) != (meta_fields is None):
            raise TypeError(
                "register_dataclass: data_fields and meta_fields must both be "
                f"specified when either is specified. Got {data_fields=} and "
                f"{meta_fields=}."
            )
        if not dataclasses.is_dataclass(nodetype):
            raise TypeError(
                "register_dataclass: data_fields and meta_fields are required when"
                f" nodetype is not a dataclass. Got {nodetype=}."
            )
        data_fields = [
            f.name
            for f in dataclasses.fields(nodetype)
            if not f.metadata.get("static", False)
        ]
        meta_fields = [
            f.name
            for f in dataclasses.fields(nodetype)
            if f.metadata.get("static", False)
        ]

    # Store inputs as immutable tuples in this scope, because we close over them
    # for later evaluation. This prevents potentially confusing behavior if the
    # caller were to pass in lists that are later mutated.
    meta_fields = tuple(meta_fields)
    data_fields = tuple(data_fields)

    if dataclasses.is_dataclass(nodetype):
        init_fields = {f.name for f in dataclasses.fields(nodetype) if f.init}
        init_fields.difference_update(*drop_fields)
        if {*meta_fields, *data_fields} != init_fields:
            msg = (
                "data_fields and meta_fields must include all dataclass fields with"
                " ``init=True`` and only them."
            )
            if missing := init_fields - {*meta_fields, *data_fields}:
                msg += (
                    f" Missing fields: {missing}. Add them to drop_fields to suppress"
                    " this error."
                )
            if unexpected := {*meta_fields, *data_fields} - init_fields:
                msg += f" Unexpected fields: {unexpected}."
            raise ValueError(msg)

    def unflatten_func(meta, data):
        meta_args = tuple(zip(meta_fields, meta, strict=False))
        data_args = tuple(zip(data_fields, data, strict=False))
        kwargs = dict(meta_args + data_args)
        return nodetype(**kwargs)

    def flatten_func(x):
        meta = tuple(getattr(x, name) for name in meta_fields)
        data = tuple(getattr(x, name) for name in data_fields)
        return data, meta

    register_struct(nodetype, flatten_func, unflatten_func)
    return nodetype
