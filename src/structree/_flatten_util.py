# structree — portable pytree/struct infrastructure
#
# Portions of this file are derived from JAX (https://github.com/jax-ml/jax),
# Copyright (c) 2021 The JAX Authors, licensed under the Apache License, Version 2.0.
#
# Modifications and additions:
# Copyright (c) 2025-2026 Pine Tree Labs, LLC, licensed under the MIT License.
#
# See LICENSE and LICENSES/Apache-2.0.txt at the repository root.

"""Flatten/unflatten a tree to and from a single 1-D array.

This is the ``ravel`` half of the package — the flat <-> structured mapping a
downstream optimizer needs (e.g. a ``Control.to_array`` / ``from_array`` pair).

Array-backend independence
--------------------------
``ravel`` is deliberately **coercion-free**: it never calls ``np.asarray`` on a
leaf that is already an array (which would force a numpy copy and destroy a
symbolic leaf). Instead it relies only on:

* the NumPy dispatch protocol (NEP 18) for ``np.ravel`` / ``np.concatenate`` /
  ``np.atleast_1d`` / ``np.split`` — so a leaf type that implements
  ``__array_function__`` (e.g. an Archimedes ``SymbolicArray``) is preserved;
* the duck-typed ``.dtype`` / ``.shape`` attributes for dtype promotion; and
* the duck-typed ``.astype`` / ``.reshape`` methods for casting and reshaping.

Numpy arrays, Python scalars, and any array type satisfying that small surface
(notably casadi-backed symbolic arrays) all ravel through the same code with no
backend registration. This mirrors the dtype-promotion and casting rules of
Archimedes' ``_result_type`` / ``array``, reproduced here without a casadi
dependency. Non-numeric leaves (e.g. ``str``) raise ``TypeError`` via
``np.result_type``, matching the upstream behavior.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np

from ._registry import unzip2
from ._tree_util import tree_flatten, tree_unflatten

if TYPE_CHECKING:
    from ._typing import ArrayLike, Tree


_SIGNATURE_CACHE: dict[int, inspect.Signature] = {}


def _make_hashable(x):
    """Recursively convert unhashable types (e.g. numpy arrays) to hashable form."""
    if isinstance(x, np.ndarray):
        return (x.shape, x.dtype, x.tobytes())
    if isinstance(x, (tuple, list)):  # covers NamedTuples (e.g. TreeDef) too
        return tuple(_make_hashable(i) for i in x)
    return x


# Original: jax._src.util.HashablePartial
class HashablePartial:
    def __init__(self, f, *args, **kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

        # Create a new call signature that doesn't include any of the provided args
        f_id = id(f)
        if f_id not in _SIGNATURE_CACHE:
            _SIGNATURE_CACHE[f_id] = inspect.signature(f)
        signature = _SIGNATURE_CACHE[f_id]

        parameters = []
        for i, (name, param) in enumerate(signature.parameters.items()):
            if i < len(args) or name in kwargs:
                continue
            parameters.append(param)

        self.__signature__ = inspect.Signature(
            parameters=parameters,
            return_annotation=signature.return_annotation,
        )
        self._name = f.__name__
        if self._name.startswith("_"):
            self._name = self._name[1:]

    @property
    def __name__(self):
        return self._name

    def __eq__(self, other):
        return (
            type(other) is HashablePartial
            and self.f.__code__ == other.f.__code__
            and _make_hashable(self.args) == _make_hashable(other.args)
            and _make_hashable(self.kwargs) == _make_hashable(other.kwargs)
        )

    def __hash__(self):
        return hash(
            (
                self.f.__code__,
                _make_hashable(self.args),
                _make_hashable(
                    tuple(sorted(self.kwargs.items(), key=lambda kv: kv[0]))
                ),
            ),
        )

    def __call__(self, *args, **kwargs):
        return self.f(*self.args, *args, **self.kwargs, **kwargs)


def _dtype_stand_in(x):
    """A numpy stand-in carrying ``x``'s dtype, for promotion without coercion.

    Mirrors Archimedes' ``_empty_like``: an array-like (numpy or symbolic) is
    represented by a numpy array of the same dtype/shape so ``np.result_type``
    sees its dtype rather than trying to coerce its values; plain scalars and
    other objects are passed through unchanged (so ``np.result_type`` applies
    value-based rules, and raises ``TypeError`` on non-numeric leaves).
    """
    if isinstance(x, np.ndarray):
        return x
    dtype = getattr(x, "dtype", None)
    shape = getattr(x, "shape", None)
    if dtype is not None and shape is not None:
        return np.empty(shape, dtype)
    return x


def _coerce(e, dtype):
    """Cast a leaf to ``dtype`` while preserving its array type.

    Numpy arrays and any array exposing ``.astype`` (e.g. ``SymbolicArray``) are
    cast in place via ``.astype``; everything else (Python scalars, lists) goes
    through ``np.asarray``. Mirrors Archimedes' ``array(e, dtype=...)``.
    """
    if isinstance(e, np.ndarray) or hasattr(e, "astype"):
        return e.astype(dtype)
    return np.asarray(e, dtype=dtype)


def ravel_tree(tree: Tree) -> tuple[ArrayLike, HashablePartial]:
    """
    Flatten a tree to a single 1D array.

    This function flattens a tree into a single 1D array by concatenating all
    leaf values (which must be arrays or scalars), and provides a function to
    reconstruct the original structure.

    Parameters
    ----------
    tree : Any
        A tree of arrays and scalars to flatten. A tree is a nested structure of
        containers (lists, tuples, dicts) and leaves (arrays or scalars).

    Returns
    -------
    flat_array : ndarray
        A 1D array containing all flattened leaf values concatenated together.
        The dtype is determined by promoting the dtypes of all leaf values. If
        the input tree is empty, a 1D empty array of dtype ``np.float32`` is
        returned. The *array type* follows the leaves: a tree of symbolic leaves
        produces a symbolic flat array.
    unravel : callable
        A function that takes a 1D array of the same length as ``flat_array`` and
        returns a tree with the same structure as the input ``tree``, with the
        values from the 1D array reshaped to match the original leaf shapes.

    Examples
    --------
    >>> import structree as st
    >>> import numpy as np
    >>>
    >>> state = {"pos": np.array([0.0, 1.0, 2.0]), "vel": np.array([3.0, 4.0, 5.0])}
    >>> flat_state, unravel = st.ravel(state)
    >>> new_state = unravel(flat_state * 2)

    See Also
    --------
    tree_flatten : Flatten a tree into a list of leaves and a treedef
    """
    leaves, treedef = tree_flatten(tree)
    flat, unravel_list = _ravel_list(leaves)
    return flat, HashablePartial(unravel_tree, treedef, unravel_list)


def unravel_tree(treedef, unravel_list, flat):
    return tree_unflatten(treedef, unravel_list(flat))


def _ravel_list(lst):
    if not lst:
        return np.array([], np.float32), lambda _: []

    # Promote dtypes from the leaves' own dtypes (no value coercion). This also
    # raises TypeError on a non-numeric leaf, matching the upstream behavior.
    to_dtype = np.result_type(*(_dtype_stand_in(x) for x in lst))
    from_dtypes = tuple(np.result_type(_dtype_stand_in(x)) for x in lst)
    sizes, shapes = unzip2((np.size(x), np.shape(x)) for x in lst)
    indices = tuple(np.cumsum(sizes).astype(int))
    shapes = tuple(shapes)

    # Faster version for trivial case with only one element
    if len(lst) == 1:
        raveled = np.atleast_1d(np.ravel(lst[0]))

    else:
        # Cast each leaf to the promoted dtype (type-preserving) and ravel.
        # np.ravel / np.concatenate dispatch via __array_function__, so symbolic
        # leaves stay symbolic.
        raveled = np.atleast_1d(
            np.concatenate([np.ravel(_coerce(e, to_dtype)) for e in lst])
        )

    unrav = HashablePartial(_unravel_list, indices, shapes, from_dtypes, to_dtype)

    return raveled, unrav


def _unravel_list(indices, shapes, from_dtypes, to_dtype, arr):
    # Fast version for trivial case with only one element
    if len(shapes) == 1:
        return [arr.reshape(shapes[0]).astype(from_dtypes[0])]

    chunks = np.split(arr, indices[:-1])
    return [
        chunk.reshape(shape).astype(dtype)
        for chunk, shape, dtype in zip(chunks, shapes, from_dtypes, strict=False)
    ]
