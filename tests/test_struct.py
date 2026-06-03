"""Tests for struct decorator"""
# This code combines and modifies code from multiple sources:

# 1. Originally derived from pybaum
#    Original Copyright (c) 2022 Janoś Gabler, Tobias Raabe
#    Licensed under MIT
#    https://github.com/OpenSourceEconomics/pybaum

# 2. Incorporates code from JAX
#    Copyright (c) 2021 The JAX Authors
#    Licensed under Apache License 2.0
#    https://github.com/jax-ml/jax

# Modifications and additions to the original code:
# Copyright (c) 2025-2026 Pine Tree Labs, LLC
# Licensed under MIT License

# As a combined work, use of this code requires compliance with the MIT License.
# The original license terms are included below for attribution:

# === MIT License ===
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# === Apache License 2.0 ===
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import dataclasses
from collections import OrderedDict, namedtuple
from functools import partial
from typing import NamedTuple

import numpy as np
import pytest

import structree as tree
from structree import field, struct
from structree._tree_util import NONE_DEF


def assert_flat_equal(f1, f2):
    assert len(f1) == len(f2)
    for a, b in zip(f1, f2):
        if isinstance(a, np.ndarray):
            assert np.allclose(a, b)
        else:
            assert a == b


def _example_tree():
    return (
        [0, np.array([1, 2]), {"a": (3.0, 4), "b": 5}],
        6,
    )


@pytest.fixture
def example_tree():
    return _example_tree()


@pytest.fixture
def example_flat():
    return [0, np.array([1, 2]), 3.0, 4, 5, 6]


@pytest.fixture
def example_treedef():
    return tree.structure(_example_tree())


def test_flatten(example_tree, example_flat):
    flat, treedef = tree.flatten(example_tree)
    assert_flat_equal(flat, example_flat)
    unflat = treedef.unflatten(flat)
    assert unflat == example_tree


def test_flatten_with_is_leaf(example_tree):
    flat, _ = tree.flatten(
        example_tree,
        is_leaf=lambda tree: isinstance(tree, dict),
    )
    expected_flat = [0, np.array([1, 2]), {"a": (3.0, 4), "b": 5}, 6]
    assert_flat_equal(flat, expected_flat)


def test_unflatten(example_flat, example_treedef, example_tree):
    unflat = tree.unflatten(example_treedef, example_flat)
    assert tree.equal(unflat, example_tree)


def test_unflatten_with_is_leaf(example_tree):
    treedef = tree.structure(example_tree, is_leaf=lambda tree: isinstance(tree, dict))
    flat = tree.leaves(example_tree, is_leaf=lambda tree: isinstance(tree, dict))
    unflat = treedef.unflatten(flat)
    assert tree.equal(unflat, example_tree)


def test_map():
    t = [{"a": 1, "b": 2, "c": {"d": 3, "e": 4}}]
    calculated = tree.map(lambda x: x * 2, t)
    expected = [{"a": 2, "b": 4, "c": {"d": 6, "e": 8}}]
    assert calculated == expected


def test_multimap():
    t = [{"a": 1, "b": 2, "c": {"d": 3, "e": 4}}]
    mapped = tree.map(lambda x: x**2, t)
    multimapped = tree.map(lambda x, y: x * y, t, t)
    assert mapped == multimapped

    with pytest.raises(ValueError, match=r"Trees must have the same structure.*"):
        tree.map(lambda x, y: x * y, t, (0, 2, 3))


def test_reduce():
    t = [{"a": 1, "b": 2, "c": {"d": 3, "e": 4}}]
    _sum = tree.reduce(lambda x, y: x + y, t, 0)
    assert _sum == sum(range(1, 5))


def test_flatten_none():
    flat, treedef = tree.flatten(None)
    assert flat == []
    assert treedef is NONE_DEF


def test_flatten_namedtuple():
    bla = namedtuple("bla", ["a", "b"])(1, 2)
    flat, _ = tree.flatten(bla)
    assert flat == [1, 2]


def test_namedtuple_class():
    class Point(NamedTuple):
        x: float
        y: float

    p = Point(1.0, 2.0)
    flat, treedef = tree.flatten(p)
    assert flat == [1.0, 2.0]
    unflat = treedef.unflatten(flat)
    assert unflat == p


def test_ordered_dict():
    d = OrderedDict({"a": 1, "b": 2})
    flat, treedef = tree.flatten(d)
    assert flat == [1, 2]
    unflat = treedef.unflatten(flat)
    assert unflat == d


def test_custom_node():
    class Point:
        def __init__(self, x, y, name):
            self.x = x
            self.y = y
            self.name = name

    def point_flatten(point: Point):
        aux_data = (point.name,)
        children = (point.x, point.y)
        return children, aux_data

    def point_unflatten(aux_data, children) -> Point:
        x, y = children
        (name,) = aux_data
        return Point(x, y, name)

    tree.register_struct(Point, point_flatten, point_unflatten)

    p = Point(1.0, 2.0, "p")
    flat, treedef = tree.flatten(p)
    assert flat == [1.0, 2.0]
    unflat = treedef.unflatten(flat)
    assert unflat.x == p.x
    assert unflat.y == p.y
    assert unflat.name == p.name


def test_register_dataclass_field():
    @tree.register_dataclass
    @dataclasses.dataclass
    class Point:
        x: float
        y: float
        name: str = dataclasses.field(metadata=dict(static=True))

    p = Point(1.0, 2.0, "p")
    flat, treedef = tree.flatten(p)
    assert flat == [1.0, 2.0]
    unflat = treedef.unflatten(flat)
    assert unflat.x == p.x
    assert unflat.y == p.y
    assert unflat.name == p.name


def test_register_dataclass_metadata():
    @partial(
        tree.register_dataclass,
        meta_fields=("name",),
        data_fields=("x", "y"),
    )
    @dataclasses.dataclass
    class Point:
        x: float
        y: float
        name: str

    p = Point(1.0, 2.0, "p")
    flat, treedef = tree.flatten(p)
    assert flat == [1.0, 2.0]
    unflat = treedef.unflatten(flat)
    assert unflat.x == p.x
    assert unflat.y == p.y
    assert unflat.name == p.name

    with pytest.raises(TypeError, match=r".*data_fields and meta_fields must both.*"):

        @partial(tree.register_dataclass, meta_fields=("name",))
        @dataclasses.dataclass
        class Point2:
            x: float
            y: float
            name: str

    with pytest.raises(TypeError, match=r".*fields are required.*"):

        @tree.register_dataclass
        class Point3:
            x: float
            y: float

    with pytest.raises(ValueError, match=r".*``init=True`` and only them."):

        @partial(
            tree.register_dataclass,
            meta_fields=("name",),
            data_fields=("x", "y"),
        )
        @dataclasses.dataclass
        class Point4:
            x: float
            y: float = dataclasses.field(init=False)

    with pytest.raises(ValueError, match=r".*Missing fields.*"):

        @partial(
            tree.register_dataclass,
            meta_fields=("name",),
            data_fields=("x"),
        )
        @dataclasses.dataclass
        class Point5:
            x: float
            y: float


@pytest.mark.skipif(
    __import__("sys").version_info < (3, 10),
    reason="kw_only requires Python 3.10+",
)
def test_register_struct_decorator():
    @struct(kw_only=True)
    class Point:
        x: float
        y: float
        name: str = field(static=True)

    assert tree.is_struct(Point)

    p = Point(x=1.0, y=2.0, name="p")

    with pytest.raises(dataclasses.FrozenInstanceError):
        p.x = 3.0

    flat, treedef = tree.flatten(p)
    assert flat == [1.0, 2.0]
    unflat = treedef.unflatten(flat)
    assert unflat.x == p.x
    assert unflat.y == p.y
    assert unflat.name == p.name

    p2 = p.replace(y=0.0)
    assert p2 is not p
    assert p2.y == 0.0
    assert p.y == 2.0

    # Re-applying @struct is a no-op (marker already set).
    assert struct(Point) is Point
