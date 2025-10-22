from typing import ClassVar
from ...core.interfaces import Encodable
from ...core.version import Version
from ...data.context import UserContext


class UserEncodable(Encodable[UserContext]):
    VERSION: ClassVar[Version] = Version("0.0.0")
