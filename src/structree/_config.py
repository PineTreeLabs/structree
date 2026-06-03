# structree — portable pytree/struct infrastructure
#
# Copyright (c) 2025-2026 Pine Tree Labs, LLC, licensed under the MIT License.

"""Pydantic-backed configuration objects with automatic type discrimination.

``StructConfig`` / ``UnionConfig`` pair a validated, serializable config schema
with a ``build()`` step that constructs the corresponding (``@struct``) runtime
object. They are the "offline" half of a config-then-build pattern: validation,
preprocessing, and data loading happen once at construction, not at runtime.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "StructConfig",
    "UnionConfig",
]


T = TypeVar("T")


class StructConfig(BaseModel):
    """
    Base class for creating configuration objects with automatic type discrimination.

    This class extends Pydantic's ``BaseModel`` to automatically add a ``type``
    field based on the class name, enabling type-safe configuration systems with
    automatic serialization and validation. Subclasses specify their type using
    the ``type`` parameter in the class definition.

    Parameters
    ----------
    type : str
        The type identifier for this configuration class, specified in the class
        definition using ``StructConfig, type="typename"``.

    Notes
    -----
    The ``type`` field is automatically added to the class and set to the value
    specified in the class definition. This enables automatic discrimination when
    working with unions of different configuration types.

    Subclasses are expected to implement a ``build()`` method that constructs the
    corresponding object based on the configuration parameters. This may include
    any "offline" validation, preprocessing, or data loading that should occur
    once at initialization time rather than at runtime.

    Examples
    --------
    >>> import numpy as np
    >>> import structree as st
    >>>
    >>> @st.struct
    >>> class ConstantGravity:
    ...     g0: float
    ...
    ...     def __call__(self, position: np.ndarray) -> np.ndarray:
    ...         return np.array([0, 0, self.g0])
    >>>
    >>> class ConstantGravityConfig(st.StructConfig, type="constant"):
    ...     g0: float = 9.81
    ...
    ...     def build(self) -> ConstantGravity:
    ...         return ConstantGravity(self.g0)
    >>>
    >>> ConstantGravityConfig(g0=9.81).build()
    ConstantGravity(g0=9.81)
    >>>
    >>> # Create a discriminated union of configuration types
    >>> GravityConfig = st.UnionConfig[ConstantGravityConfig, PointGravityConfig]

    See Also
    --------
    UnionConfig : Create discriminated unions of StructConfig subclasses
    struct : Decorator for creating modular dataclass components
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init_subclass__(cls, type: Optional[str] = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if type is not None:
            cls.__annotations__ = {"type": Literal[type], **cls.__annotations__}
            cls.type = type  # type: ignore

    # When printing the config, show the class name and fields only but
    # not the type field
    def __repr__(self):
        rep = super().__repr__()
        if hasattr(self, "type"):
            return rep.replace(f"type='{self.type}', ", "")
        return rep

    def build(self):
        raise NotImplementedError("Subclasses must implement the build() method.")


class UnionConfig:
    """
    Discriminated union of StructConfig subclasses.

    Usage:
        AnyConfig = UnionConfig[ConfigTypeA, ConfigTypeB]

    Equivalent to:
        AnyConfig = Annotated[
            Union[ConfigTypeA, ConfigTypeB],
            Field(discriminator="type"),
        ]

    See Also
    --------
    StructConfig : Base class for configuration management
    struct : Decorator for creating modular components
    """

    def __class_getitem__(cls, item) -> Type:
        # Handle single type (UnionConfig[OneType])
        if not isinstance(item, tuple):
            item = (item,)

        # Validate that all types inherit from StructConfig
        for config_type in item:
            if not (
                isinstance(config_type, type) and issubclass(config_type, StructConfig)
            ):
                raise TypeError(
                    f"{config_type} must be a subclass of StructConfig. "
                    f"UnionConfig is only for StructConfig discriminated unions."
                )

        # Create the discriminated union
        return Annotated[Union[item], Field(discriminator="type")]  # type: ignore
