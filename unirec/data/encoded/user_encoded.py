from torch import Tensor
from typing import ClassVar, override
from ...core.interfaces import Encodable, Encoded, Request
from ...core.version import Version
from ...data.context import UserContext


class UserEncoded(Encoded[UserContext]):
    VERSION: ClassVar[Version] = Version("0.0.0")

    @override
    def __init__(
        self,
        vector: Tensor,
        origin: Encodable[UserContext],
        request: Request | None = None,
    ):
        super().__init__(vector, origin, request)
