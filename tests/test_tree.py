"""Unit tests for tree flattening and struct behavior."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import structree as tree
from structree import field, struct


@struct
class State:
    pos: np.ndarray
    vel: np.ndarray
    mass: float = field(static=True, default=1.0)


@struct
class Reservoir:
    p0: float
    t0: float

    def __post_init__(self) -> None:
        # frozen dataclass => must set via object.__setattr__
        object.__setattr__(self, "rho0", self.p0 / self.t0)


# ---------------------------------------------------------------------------
# flatten / unflatten
# ---------------------------------------------------------------------------


def test_flatten_excludes_static_fields():
    s = State(np.array([0.0, 1.0]), np.array([2.0, 3.0]), mass=5.0)
    leaves, _ = tree.flatten(s)
    # only pos and vel are leaves; mass is aux/static
    assert len(leaves) == 2
    assert np.allclose(leaves[0], [0.0, 1.0])
    assert np.allclose(leaves[1], [2.0, 3.0])


def test_unflatten_roundtrip_preserves_static():
    s = State(np.array([0.0, 1.0]), np.array([2.0, 3.0]), mass=5.0)
    leaves, treedef = tree.flatten(s)
    s2 = tree.unflatten(treedef, [2 * x for x in leaves])
    assert np.allclose(s2.pos, [0.0, 2.0])
    assert np.allclose(s2.vel, [4.0, 6.0])
    assert s2.mass == 5.0  # static preserved through the roundtrip


def test_nested_container_roundtrip():
    tree_data = {"a": np.array([1.0, 2.0]), "b": {"c": np.array([3.0])}}
    leaves, treedef = tree.flatten(tree_data)
    assert len(leaves) == 2
    out = tree.unflatten(treedef, leaves)
    assert out.keys() == tree_data.keys()
    assert np.allclose(out["b"]["c"], [3.0])


def test_jax_convention():
    # register_dataclass stores (data, meta) == (children, aux_data), the JAX
    # pytree node convention. Verify the flatten_func returns that shape.
    from structree._registry import _registry

    entry = _registry[State]
    s = State(np.zeros(2), np.ones(2), mass=3.0)
    children, aux = entry.to_iter(s)
    assert len(children) == 2  # pos, vel
    assert aux == (3.0,)  # mass, as static aux data


# ---------------------------------------------------------------------------
# struct behavior
# ---------------------------------------------------------------------------


def test_struct_is_frozen():
    s = State(np.zeros(2), np.zeros(2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.pos = np.ones(2)  # frozen


def test_replace_creates_modified_copy():
    s = State(np.zeros(2), np.zeros(2), mass=1.0)
    s2 = s.replace(mass=9.0)
    assert s2.mass == 9.0
    assert s.mass == 1.0  # original untouched
    assert s2 is not s


def test_is_struct():
    assert tree.is_struct(State(np.zeros(1), np.zeros(1)))
    assert not tree.is_struct({"a": 1})
    assert not tree.is_struct(np.zeros(3))


def test_map_skips_static():
    s = State(np.array([1.0]), np.array([2.0]), mass=4.0)
    out = tree.map(lambda x: x * 10, s)
    assert np.allclose(out.pos, [10.0])
    assert np.allclose(out.vel, [20.0])
    assert out.mass == 4.0  # not a leaf, untouched


def test_unflatten_reruns_post_init():
    r = Reservoir(p0=10.0, t0=2.0)
    assert r.rho0 == 5.0
    leaves, treedef = tree.flatten(r)
    # change p0 leaf; rho0 must be recomputed on reconstruction
    r2 = tree.unflatten(treedef, [20.0, 2.0])
    assert r2.p0 == 20.0
    assert r2.rho0 == 10.0  # 20 / 2, recomputed via __post_init__


# ---------------------------------------------------------------------------
# ravel (flat <-> structured)
# ---------------------------------------------------------------------------


def test_ravel_roundtrip():
    s = State(np.array([0.0, 1.0, 2.0]), np.array([3.0, 4.0]), mass=7.0)
    flat, unravel = tree.ravel(s)
    assert flat.shape == (5,)
    assert np.allclose(flat, [0.0, 1.0, 2.0, 3.0, 4.0])
    s2 = unravel(flat * 2.0)
    assert np.allclose(s2.pos, [0.0, 2.0, 4.0])
    assert np.allclose(s2.vel, [6.0, 8.0])
    assert s2.mass == 7.0


def test_ravel_dict_roundtrip():
    tree_data = {"pos": np.array([0.0, 1.0, 2.0]), "vel": np.array([3.0, 4.0, 5.0])}
    flat, unravel = tree.ravel(tree_data)
    assert np.allclose(flat, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    out = unravel(flat)
    assert np.allclose(out["pos"], [0.0, 1.0, 2.0])
    assert np.allclose(out["vel"], [3.0, 4.0, 5.0])


def test_ravel_empty_tree():
    flat, unravel = tree.ravel({})
    assert flat.shape == (0,)
    assert unravel(flat) == {}
