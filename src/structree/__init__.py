# structree — portable pytree/struct infrastructure
#
# Copyright (c) 2025-2026 Pine Tree Labs, LLC, licensed under the MIT License.
# Portions derived from JAX and Flax (Apache-2.0); see LICENSES/Apache-2.0.txt.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""structree: a small, dependency-light pytree/struct toolkit.

Flatten/unflatten nested structures and ``@struct`` dataclasses to and from flat
arrays, with a JAX-compatible ``(children, aux_data)`` node contract. Extracted
from the Archimedes ``tree`` package so it can be shared across projects without
pulling in a heavyweight (casadi/pydantic) dependency stack.
"""

from ._config import (
    StructConfig,
    UnionConfig,
)
from ._flatten_util import ravel_tree as ravel
from ._registry import (
    register_dataclass,
    register_struct,
)
from ._struct import (
    InitVar,
    field,
    fields,
    is_struct,
    replace,
    struct,
)
from ._tree_util import is_leaf
from ._tree_util import (
    tree_all as all,
)
from ._tree_util import (
    tree_equal as equal,
)
from ._tree_util import (
    tree_flatten as flatten,
)
from ._tree_util import (
    tree_leaves as leaves,
)
from ._tree_util import (
    tree_map as map,
)
from ._tree_util import (
    tree_reduce as reduce,
)
from ._tree_util import (
    tree_structure as structure,
)
from ._tree_util import (
    tree_unflatten as unflatten,
)

__all__ = [
    "register_struct",
    "register_dataclass",
    "is_leaf",
    "flatten",
    "unflatten",
    "structure",
    "leaves",
    "map",
    "all",
    "equal",
    "reduce",
    "ravel",
    "struct",
    "field",
    "InitVar",
    "is_struct",
    "fields",
    "replace",
    "StructConfig",
    "UnionConfig",
]
