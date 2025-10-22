from dataclasses import dataclass
from typing import Any, ClassVar, Mapping
from ...core.interfaces import Context, Profile, Session
from ...core.version import Version


class UserContext(Context):
    VERSION: ClassVar[Version] = Version("0.0.0")


@dataclass
class UserProfileContext(Profile[UserContext]):
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
class UserSessionContext(Session[UserContext]):
    VERSION: ClassVar[Version] = Version("0.0.0")

    _meta: Mapping[str, Any]

    @property
    def meta(self) -> Mapping[str, Any]:
        return self._meta
