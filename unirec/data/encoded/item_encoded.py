from torch import Tensor
from typing import ClassVar, override
from ...core.interfaces import Encodable, Encoded, Request
from ...core.version import Version
from ...data.context import ItemContext


class ItemEncoded(Encoded[ItemContext]):
    VERSION: ClassVar[Version] = Version("0.0.0")

    @override
    def __init__(
        self,
        vector: Tensor,
        origin: Encodable[ItemContext],
        request: Request | None = None,
    ):
        super().__init__(vector, origin, request)
