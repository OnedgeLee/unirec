from torch import Tensor
from typing import Any, ClassVar
from ...core.interfaces import Encodable, Encoder
from ...core.version import Version
from ...data.context import RequestContext, UserContext
from ...data.encodable import UserEncodable
from ...data.encoded import UserEncoded


class UserEncoder(Encoder[UserContext]):
    VERSION: ClassVar[Version] = Version("0.0.0")

    def encode(
        self,
        encodable: Encodable[UserContext],
        *,
        request: RequestContext | None = None,
        **kwargs: Any,
    ) -> UserEncoded:
        return UserEncoded(Tensor())
