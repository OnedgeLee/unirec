from dataclasses import dataclass
from typing import Any, ClassVar, Mapping
from ...core.interfaces import Context, Profile, Session
from ...core.version import Version


class ItemContext(Context):
    VERSION: ClassVar[Version] = Version("0.0.0")


@dataclass
class ItemProfileContext(Profile[ItemContext]):
    VERSION: ClassVar[Version] = Version("0.0.0")

    _id: int
    _meta: Mapping[str, Any]

    @property
    def id(self) -> int:
        return self._id

    @property
    def meta(self) -> Mapping[str, Any]:
        return self._meta


@dataclass
class ItemSessionContext(Session[ItemContext]):
    VERSION: ClassVar[Version] = Version("0.0.0")

    _meta: Mapping[str, Any]

    @property
    def meta(self) -> Mapping[str, Any]:
        return self._meta
