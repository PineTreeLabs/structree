# structree — portable pytree/struct infrastructure
#
# Portions of this file are derived from JAX (https://github.com/jax-ml/jax),
# Copyright (c) 2021 The JAX Authors, licensed under the Apache License, Version 2.0.
#
# Modifications and additions:
# Copyright (c) 2025-2026 Pine Tree Labs, LLC, licensed under the MIT License.

from __future__ import annotations

import itertools as it
from collections.abc import Callable, Hashable, Iterable, Iterator
from functools import partial, reduce
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    TypeVar,
    cast,
)

import numpy as np

from ._registry import _registry, unzip2

if TYPE_CHECKING:
    from ._typing import ArrayLike, Tree

    T = TypeVar("T", bound=Tree)
    V = TypeVar("V")


class TreeDef(NamedTuple):
    node_data: None | tuple[type, Hashable]
    children: tuple[TreeDef, ...]
    num_leaves: int

    def unflatten(self, xs: list[Any]) -> Any:
        return tree_unflatten(self, xs)

    @property
    def tree_str(self) -> str:
        stars = ["*"] * self.num_leaves
        return cast(str, self.unflatten(stars))

    def __repr__(self) -> str:
        return (f"TreeDef({self.tree_str})").replace("'*'", "*")

    def __eq__(self, other: object) -> bool:
        other = cast(TreeDef, other)
        return (
            self.node_data == other.node_data
            and self.children == other.children
            and self.num_leaves == other.num_leaves
        )


LEAF = TreeDef(None, (), 1)
NONE_DEF = TreeDef(None, (), 0)


#
# Flatten/unflatten functions
#
def tree_flatten(
    x: Tree, is_leaf: Callable[[Any], bool] | None = None
) -> tuple[list[ArrayLike], TreeDef]:
    """
    Flatten a tree into a list of leaves and a treedef.

    This function recursively traverses the tree and extracts all leaf values
    while recording the structure. This is useful when you need to apply
    operations to all leaf values or convert structured data to a flat
    representation.

    Parameters
    ----------
    x : Tree
        A tree to be flattened. Here, a tree is a nested structure of containers
        (lists, tuples, dicts, etc) and leaves (arrays, scalars, objects not
        registered as trees).
    is_leaf : callable, optional
        A function that takes a tree as input and returns a boolean indicating
        whether it should be considered a leaf. If not provided, the default
        leaf types are used.

    Returns
    -------
    leaves : list
        A list of all leaf values from the tree.
    treedef : TreeDef
        A structure definition that can be used to reconstruct the original
        tree using ``unflatten``.

    Examples
    --------
    >>> import structree as st
    >>> import numpy as np
    >>>
    >>> data = {"a": np.array([1.0, 2.0]), "b": {"c": np.array([3.0])}}
    >>> leaves, treedef = st.flatten(data)
    >>> reconstructed = st.unflatten(treedef, leaves)

    See Also
    --------
    tree_unflatten : Reconstruct a tree from leaves and a treedef
    tree_leaves : Extract just the leaf values from a tree
    tree_structure : Extract just the structure from a tree
    """
    children_iter, treedef = _tree_flatten(x, is_leaf)
    return list(children_iter), treedef


def _tree_flatten(
    x: Tree, is_leaf: Callable[[Any], bool] | None
) -> tuple[Iterable, TreeDef]:
    if x is None:
        return [], NONE_DEF

    _tree_flatten_leaf = partial(_tree_flatten, is_leaf=is_leaf)

    node_type = type(x)
    # If the node is a namedtuple, use the tuple flatten/unflatten functions
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        node_type = tuple
    if node_type not in _registry or (is_leaf is not None and is_leaf(x)):
        return [x], LEAF

    children, node_metadata = _registry[node_type].to_iter(x)
    children_flat, child_trees = unzip2(map(_tree_flatten_leaf, children))
    flattened = list(it.chain.from_iterable(children_flat))

    node_data = (type(x), node_metadata)
    treedef = TreeDef(
        node_data=node_data,
        children=tuple(child_trees),
        num_leaves=len(flattened),
    )
    return flattened, treedef


def tree_unflatten(treedef: TreeDef, xs: list[ArrayLike]) -> Tree:
    """
    Reconstruct a tree from a list of leaves and a treedef.

    This function is the inverse of ``tree_flatten``. It takes a list of leaf
    values and a tree definition, and reconstructs the original tree structure.

    Parameters
    ----------
    treedef : TreeDef
        A tree definition, typically produced by ``tree_flatten`` or
        ``tree_structure``.
    xs : list[ArrayLike]
        A list of leaf values to be placed in the reconstructed tree. The length
        must match the number of leaves in ``treedef``.

    Returns
    -------
    tree : Tree
        The reconstructed tree with the same structure as defined by treedef and
        with leaf values from ``xs``.

    Raises
    ------
    ValueError
        If the number of leaves in ``xs`` doesn't match the expected number in
        ``treedef``.

    See Also
    --------
    tree_flatten : Flatten a tree into a list of leaves and a treedef
    tree_structure : Extract just the structure from a tree
    """
    return _tree_unflatten(treedef, iter(xs))


def _tree_unflatten(treedef: TreeDef, xs: Iterator) -> Tree:
    if treedef is NONE_DEF:
        return None  # Special case for None
    if treedef.node_data is None:
        return next(xs)
    else:
        children = tuple(_tree_unflatten(t, xs) for t in treedef.children)
        node_type, node_metadata = treedef.node_data

        # Special logic for NamedTuple classes
        if issubclass(node_type, tuple) and hasattr(node_type, "_fields"):
            return node_type(*children)

        return _registry[node_type].from_iter(node_metadata, children)


#
# Other utility functions
#


def tree_structure(tree: Tree, is_leaf: Callable[[Any], bool] | None = None) -> TreeDef:
    """
    Extract the structure of a tree without the leaf values.

    Returns a ``TreeDef`` that describes the structure of the tree, which can be
    used with ``tree_unflatten`` to reconstruct a tree with new leaf values.

    Parameters
    ----------
    tree : Tree
        A tree whose structure is to be determined.
    is_leaf : callable, optional
        A function that takes a tree node as input and returns a boolean
        indicating whether it should be considered a leaf.

    Returns
    -------
    treedef : TreeDef
        A tree definition that describes the structure of the input tree.

    See Also
    --------
    tree_flatten : Flatten a tree into a list of leaves and a treedef
    tree_unflatten : Reconstruct a tree from leaves and a treedef
    """
    flat, treedef = tree_flatten(tree, is_leaf)
    return treedef


def tree_leaves(
    tree: Tree, is_leaf: Callable[[Any], bool] | None = None
) -> list[ArrayLike]:
    """
    Extract all leaf values from a tree.

    Parameters
    ----------
    tree : Tree
        A tree from which to extract leaves.
    is_leaf : callable, optional
        A function that takes a tree node as input and returns a boolean
        indicating whether it should be considered a leaf.

    Returns
    -------
    leaves : list
        A list of all leaf values from the tree.

    See Also
    --------
    tree_flatten : Flatten a tree into a list of leaves and a treedef
    tree_map : Apply a function to each leaf in a tree
    """
    flat, treedef = tree_flatten(tree, is_leaf)
    return flat


def tree_all(tree: Tree, is_leaf: Callable[[Any], bool] | None = None) -> bool:
    """
    Check if all leaves in the tree evaluate to True.

    Parameters
    ----------
    tree : Tree
        A tree to check.
    is_leaf : callable, optional
        A function that takes a tree node as input and returns a boolean
        indicating whether it should be considered a leaf.

    Returns
    -------
    result : bool
        True if all leaves in the tree evaluate to True, False otherwise.

    See Also
    --------
    tree_map : Apply a function to each leaf in a tree
    tree_leaves : Extract just the leaf values from a tree
    """
    flat, treedef = tree_flatten(tree, is_leaf)
    return np.all(map(np.all, flat))  # type: ignore


def tree_map(
    f: Callable,
    tree: T,
    *rest: tuple[T, ...],
    is_leaf: Callable[[Any], bool] | None = None,
) -> T:
    """
    Apply a function to each leaf in a tree.

    Traverses the tree and applies the function ``f`` to each leaf, returning a
    new tree with the same structure but transformed leaf values. If additional
    trees are provided, the function is applied to corresponding leaves from all
    trees.

    Parameters
    ----------
    f : callable
        A function to apply to each leaf. When multiple trees are provided, this
        function should accept as many arguments as there are trees.
    tree : Any
        The main tree whose structure will be followed.
    *rest : Any
        Additional trees with exactly the same structure as the first tree.
    is_leaf : callable, optional
        A function that takes a tree node as input and returns a boolean
        indicating whether it should be considered a leaf.

    Returns
    -------
    mapped_tree : Any
        A new tree with the same structure as ``tree`` but with leaf values
        transformed by function ``f``.

    Raises
    ------
    ValueError
        If additional trees do not have exactly the same structure as the main
        tree.

    Examples
    --------
    >>> import structree as st
    >>> import numpy as np
    >>>
    >>> state = {"pos": np.array([1.0, 2.0]), "vel": np.array([3.0, 4.0])}
    >>> doubled = st.map(lambda x: x * 2, state)

    See Also
    --------
    tree_flatten : Flatten a tree into a list of leaves and a treedef
    tree_leaves : Extract just the leaf values from a tree
    """
    leaves, treedef = tree_flatten(tree, is_leaf)
    flat = [leaves]
    for r in rest:
        r_flat, r_treedef = tree_flatten(r, is_leaf)
        if treedef != r_treedef:
            raise ValueError(
                "Trees must have the same structure but got treedefs: "
                f"{treedef} and {r_treedef}"
            )
        flat.append(r_flat)

    flat = [f(*args) for args in zip(*flat, strict=False)]
    tree_out: T = tree_unflatten(treedef, flat)  # type: ignore
    return tree_out


def tree_reduce(
    function: Callable[[V, ArrayLike], V],
    tree: Tree,
    initializer: V,
    is_leaf: Callable[[Any], bool] | None = None,
) -> V:
    """
    Reduce a tree to a single value using a function and initializer.

    Traverses the tree, applying the reduction function to each leaf and an
    accumulator, similar to Python's built-in ``functools.reduce`` but operating
    on all leaves of a tree.

    Parameters
    ----------
    function : callable
        A function of two arguments: ``(accumulated_result, leaf_value)`` that
        returns a new accumulated result.
    tree : Tree
        A tree to reduce.
    initializer : Any
        The initial value for the accumulator.
    is_leaf : callable, optional
        A function that takes a tree node as input and returns a boolean
        indicating whether it should be considered a leaf.

    Returns
    -------
    result : Any
        The final accumulated value after applying the function to all leaves.

    See Also
    --------
    tree_map : Apply a function to each leaf in a tree
    tree_leaves : Extract just the leaf values from a tree
    """
    flat, _treedef = tree_flatten(tree, is_leaf)
    return reduce(function, flat, initializer)


def is_leaf(x: Any) -> bool:
    """Check if a value is a leaf in a tree.

    Returns True if the value is not a container (i.e. is an array, a scalar, or
    None).

    Parameters
    ----------
    x : Any
        The value to check.

    Returns
    -------
    bool
        True if the value is a leaf, False otherwise.
    """
    treedef = tree_structure(x)
    return treedef is LEAF or treedef is NONE_DEF


def tree_equal(a, b):
    """Check if two trees are equal by comparing their leaves with np.allclose.

    This function checks if two trees have the same structure and if all their
    corresponding leaves are approximately equal using ``np.allclose``. It returns
    True if and only if the trees are structurally the same and all leaves are close.

    Parameters
    ----------
    a : Tree
        The first tree to compare.
    b : Tree
        The second tree to compare.

    Returns
    -------
    bool
        True if the trees are equal (same structure and all leaves are close).
    """
    same_structure = tree_structure(a) == tree_structure(b)
    if not same_structure:
        return False
    return tree_all(tree_map(lambda a, b: np.allclose(a, b), a, b))
