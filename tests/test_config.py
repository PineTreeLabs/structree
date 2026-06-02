"""Tests for structree._config."""

from __future__ import annotations

import numpy as np
import pytest

from structree import StructConfig, UnionConfig, struct


@struct
class _VariantBase:
    pass


@struct(kw_only=True)
class _VariantA(_VariantBase):
    param: float

    def __call__(self):
        return self.param


@struct(kw_only=True)
class _VariantB(_VariantBase):
    param: float

    def __call__(self):
        return -self.param


class _VariantConfigBase(StructConfig):
    param: float


class _VariantAConfig(_VariantConfigBase, type="positive"):
    def build(self):
        return _VariantA(param=self.param)


class _VariantBConfig(_VariantConfigBase, type="negative"):
    def build(self):
        return _VariantB(param=self.param)


def test_variant_config():
    UnionConfig[_VariantAConfig, _VariantBConfig]

    with pytest.raises(TypeError, match=r".*must be a subclass of StructConfig.*"):
        UnionConfig[_VariantA, _VariantB]

    UnionConfig[_VariantAConfig]


def test_struct_config():
    param = 42
    config = _VariantAConfig(param=param)
    model = config.build()
    y = model()
    assert np.allclose(y, param)
