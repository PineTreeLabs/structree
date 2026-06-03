"""Test ravel/unravel operations, including on Archimedes symbolic arrays"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

# Symbolic leaves come from Archimedes (dev dependency, requires Python 3.11+).
archimedes = pytest.importorskip("archimedes", reason="archimedes not available")
from archimedes._core import SymbolicArray, compile, sym  # noqa: E402

import structree as tree
from structree import field, struct


class TestRavel:
    """structree.ravel on Archimedes SymbolicArray leaves."""

    def _test_ravel_unravel(self, struct, compile_fn=False):
        flat, unravel = tree.ravel(struct)
        assert isinstance(flat, SymbolicArray)

        if compile_fn:
            unravel = compile(unravel)

        unraveled = unravel(flat)

        for x, y in zip(struct, unraveled, strict=False):
            assert x.shape == y.shape
            assert x.dtype == y.dtype

        return unraveled

    def test_ravel_none(self):
        flat, unravel = tree.ravel(None)
        assert flat.size == 0
        assert unravel(flat) is None

    def test_ravel_identity(self):
        x = sym("x", shape=(3,), dtype=np.float64)
        flat, unravel = tree.ravel(x)
        assert isinstance(flat, SymbolicArray)
        unraveled = unravel(flat)
        assert isinstance(unraveled, SymbolicArray)

    def test_ravel_scalar(self):
        x = 0.0
        flat, unravel = tree.ravel(x)
        assert flat == 0
        unraveled = unravel(flat)
        assert unraveled == 0

    def test_ravel_tuple(self):
        struct = []
        for i, (shape, dtype) in enumerate(
            [
                ((2, 3), np.float64),
                ((3,), np.int32),
                ((2, 5), np.int32),
                ((), np.bool_),
            ]
        ):
            struct.append(sym(f"x{i}", shape=shape, dtype=dtype))
        struct = tuple(struct)
        self._test_ravel_unravel(struct)

    def test_ravel_list(self):
        struct = []
        for i, (shape, dtype) in enumerate(
            [
                ((2, 3), np.float64),
                ((3,), np.int32),
                ((2, 5), np.int32),
                ((), np.bool_),
            ]
        ):
            struct.append(sym(f"x{i}", shape=shape, dtype=dtype))
        self._test_ravel_unravel(struct)

    def test_ravel_namedtuple(self):
        class Point(NamedTuple):
            x: SymbolicArray
            y: SymbolicArray
            z: SymbolicArray

        struct = Point(
            sym("x", shape=(3,), dtype=np.float64),
            sym("y", shape=(3,), dtype=np.float64),
            sym("z", shape=(3,), dtype=np.float64),
        )

        unraveled = self._test_ravel_unravel(struct)
        assert isinstance(unraveled, Point)

        # Also exercise the compiled unravel path (namedtuple output is in both
        # registries, so archimedes.compile can flatten it).
        unraveled = self._test_ravel_unravel(struct, compile_fn=True)
        assert isinstance(unraveled, Point)

    def test_ravel_dict(self):
        struct = {
            "a": sym("a", shape=(2, 3), dtype=np.float64),
            "b": sym("b", shape=(3,), dtype=np.int32),
            "c": sym("c", shape=(2, 5), dtype=np.int32),
            "d": sym("d", shape=(), dtype=np.bool_),
        }

        flat, unravel = tree.ravel(struct)
        assert isinstance(flat, SymbolicArray)

        unraveled = unravel(flat)
        for x, y in zip(struct.values(), unraveled.values(), strict=False):
            assert x.shape == y.shape
            assert x.dtype == y.dtype

    def test_ravel_error_handling(self):
        struct = "abc"
        with pytest.raises(TypeError):
            tree.ravel(struct)

    def test_ravel_nested(self):
        struct = (4.0, 3.0), np.array([2, 1])
        flat, unravel = tree.ravel(struct)
        assert np.allclose(flat, [4, 3, 2, 1])
        unflat = unravel(flat)
        assert tree.equal(struct, unflat)

    def test_ravel_hash_dict(self):
        # dict is recognized by both registries, so archimedes.compile can
        # flatten it; verifies that dict keys are hashed into the cache key.
        @compile
        def f(x, p):
            a = p.get("a", 0)
            b = p.get("b", 0)
            return a * x[0] + b * x[1]

        x = np.array([1, 2])
        assert f(x, {"a": 1}) == x[0]
        assert f(x, {"b": 1}) == x[1]

    def test_ravel_hashable_with_array_static_field(self):
        # Regression: ravel returns a HashablePartial wrapping the TreeDef. When
        # a struct has numpy arrays as static fields, they end up in
        # TreeDef.node_data; HashablePartial.__hash__ must handle them without
        # raising "unhashable type: 'numpy.ndarray'". (Pure structree path.)
        @struct
        class LookupTable:
            breakpoints: np.ndarray = field(static=True)
            values: np.ndarray

        t = LookupTable(
            breakpoints=np.array([0.0, 1.0, 2.0]),
            values=np.array([10.0, 20.0, 30.0]),
        )
        _, unravel = tree.ravel(t)
        assert isinstance(hash(unravel), int)

        t2 = LookupTable(
            breakpoints=np.array([0.0, 1.0, 2.0]),
            values=np.array([100.0, 200.0, 300.0]),
        )
        _, unravel2 = tree.ravel(t2)
        assert hash(unravel) == hash(unravel2)
        assert unravel == unravel2

        t3 = LookupTable(
            breakpoints=np.array([0.0, 5.0, 10.0]),
            values=np.array([10.0, 20.0, 30.0]),
        )
        _, unravel3 = tree.ravel(t3)
        assert hash(unravel) != hash(unravel3)
        assert unravel != unravel3

    @pytest.mark.skip(
        reason="archimedes.compile flattens its inputs with Archimedes' own "
        "pytree registry, which does not recognize a structree @struct as a "
        "node until Archimedes migrates its tree/ onto structree (shared "
        "registry). Passing a structree struct directly as a compiled-function "
        "argument is therefore out of scope here; the symbolic-ravel path is "
        "covered by the tests above."
    )
    def test_ravel_compile_with_array_static_field(self):
        @struct
        class LookupTable:
            breakpoints: np.ndarray = field(static=True)
            values: np.ndarray

        @compile
        def f(t):
            return np.sum(t.values)

        t = LookupTable(
            breakpoints=np.array([0.0, 1.0, 2.0]),
            values=np.array([10.0, 20.0, 30.0]),
        )
        assert float(f(t)) == 60.0
