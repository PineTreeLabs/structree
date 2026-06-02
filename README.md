# structree

A lightweight **pytree / struct** toolkit: flatten and unflatten nested Python structures (and `@struct` dataclasses) to and from flat arrays, inspired by the JAX [pytree](https://docs.jax.dev/en/latest/pytrees.html) concept.


```python
import numpy as np

import structree as tree
from structree import struct, field

@struct
class State:
    pos: np.ndarray
    vel: np.ndarray
    mass: float = field(static=True, default=1.0)   # static => aux data, not a leaf

s = State(np.zeros(3), np.ones(3))

leaves, treedef = tree.flatten(s)        # leaves = [pos, vel]; mass is aux, preserved
s2 = tree.unflatten(treedef, leaves)     # reconstructs (re-runs __post_init__)
doubled = tree.map(lambda x: 2 * x, s)   # static fields untouched

flat, unravel = tree.ravel(s)            # flat 1-D vector  <->  structured state
s3 = unravel(flat * 0.5)
```

## Array-backend independence (works with symbolic arrays)

`structree` was extracted from the `archimedes.tree` package so the same pytree infrastructure can be shared across projects.
It depends only on `numpy`, `typing-extensions`, and `pydantic` — **not** on casadi or any symbolic engine.
It can be used with any objects that support:

- the NumPy dispatch protocol (NEP 18) for `np.ravel`/`np.concatenate`/`np.split`,
- the `.dtype`/`.shape` attributes for dtype promotion, and
- the `.astype`/`.reshape` methods for casting/reshaping

## What's included

The full `archimedes.tree` surface: tree topology (`flatten`/`unflatten`/`map`/
`reduce`/`leaves`/`structure`), `@struct`/`field`/`replace`, `ravel`, the node
registry (`register_dataclass`/`register_struct`), and the pydantic-backed
config layer (`StructConfig`/`UnionConfig`) for validated, discriminated
config-then-`build()` schemas.

## Provenance & license

MIT (Pine Tree Labs modifications) plus Apache-2.0 attribution for the
JAX/Flax-derived portions - see `LICENSE`, `NOTICE`, and `LICENSES/Apache-2.0.txt`.
The `(children, aux_data)` convention matches JAX's pytree registration, so
flattened data lines up element-for-element with JAX/Flax pytrees.
