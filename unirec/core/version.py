from abc import ABC
from packaging.version import Version
from typing import ClassVar


__all__ = ["Version", "Versioned"]


class Versioned(ABC):
    VERSION: ClassVar[Version]

    @classmethod
    def version(cls) -> Version:
        return cls.VERSION

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if getattr(cls, "__abstractmethods__", None):
            return
        v = getattr(cls, "VERSION", None)
        if not isinstance(v, Version):
            raise TypeError(
                f"{cls.__name__}.VERSION must be 'packaging.version.Version'"
            )
