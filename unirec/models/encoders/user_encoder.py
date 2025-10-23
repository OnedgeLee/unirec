import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from typing import Any, ClassVar, cast, override
from ...core.interfaces import Encodable, Encoded, Encoder, Request
from ...core.version import Version
from ...data.context import RequestContext, UserContext
from ...data.encodable import UserEncodable
from ...data.encoded import UserEncoded


class UserEncoder(Encoder[UserContext]):
    VERSION: ClassVar[Version] = Version("0.0.0")

    @override
    def encode(
        self,
        encodable: Encodable[UserContext],
        *,
        request: Request | None = None,
        **kwargs: Any,
    ) -> Encoded[UserContext]:
        return self.encode_user(
            cast(UserEncodable, encodable),
            cast(RequestContext, request),
            cast(NDArray[np.float32], kwargs.get("item_memmap")),
            cast(int, kwargs.get("emb_dim")),
        )

    def encode_user(
        self,
        encodable: UserEncodable,
        request: RequestContext,
        item_memmap: NDArray[np.float32],
        emb_dim: int,
    ) -> UserEncoded:
        return UserEncoded(Tensor(), encodable, request)
