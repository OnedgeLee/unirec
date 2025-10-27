from typing import ClassVar
from ...core.interfaces import Encodable
from ...core.version import Version
from ...data.context import ItemContext


class ItemEncodable(Encodable[ItemContext]):
    VERSION: ClassVar[Version] = Version("0.0.0")
