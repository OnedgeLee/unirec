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

    def setup(self, emb_dim: int, item_memmap: NDArray[np.float32]):
        self.emb_dim = emb_dim
        self.item_memmap = item_memmap

    @override
    def encode(
        self,
        encodable: Encodable[UserContext],
        *,
        request: Request | None = None,
        **kwargs: Any,
    ) -> Encoded[UserContext]:
        return self.encode_user(
            cast(UserEncodable, encodable), cast(RequestContext, request)
        )

    def encode_user(
        self, encodable: UserEncodable, request: RequestContext
    ) -> UserEncoded:
        return UserEncoded(Tensor(), encodable, request)
