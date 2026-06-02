# structree — portable pytree/struct infrastructure
#
# Copyright (c) 2025-2026 Pine Tree Labs, LLC, licensed under the MIT License.

"""Lightweight type aliases.

Inlined so the package carries no dependency on any array backend (the
upstream Archimedes module pulled these from a casadi-backed ``typing``
module). Leaves are intentionally untyped (``Any``): structree describes
tree *topology* and is agnostic to what the leaves are — numpy arrays,
scalars, or a downstream library's symbolic arrays.
"""

from __future__ import annotations

from typing import Any, TypeAlias

# A leaf value. Deliberately ``Any`` — structree does not constrain leaves.
ArrayLike: TypeAlias = Any

# A pytree: a nested structure of registered containers and leaves.
Tree: TypeAlias = Any

__all__ = ["ArrayLike", "Tree"]
