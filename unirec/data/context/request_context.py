from dataclasses import dataclass
from typing import Any, ClassVar, Mapping
from ...core.interfaces import Request
from ...core.version import Version


@dataclass
class RequestContext(Request):
    VERSION: ClassVar[Version] = Version("0.0.0")

    _meta: Mapping[str, Any]

    @property
    def meta(self) -> Mapping[str, Any]:
        return self._meta
