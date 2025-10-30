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

    def __init__(self, d: int, **params: Any):
        super().__init__(d=d, **params)
        self.d = d

    @override
    def setup(self, resources: dict[str, Any]):
        super().setup(resources)
        self.item_memmap: NDArray[np.float32] | None = self.optional_resource(
            "item_memmap", np.ndarray
        )

    @override
    def encode(
        self,
        encodable: Encodable[UserContext],
        *,
        request: Request | None = None,
        **kwargs: Any,
    ) -> Encoded[UserContext]:
        return self.encode_user(
            cast(UserEncodable, encodable), cast(RequestContext | None, request)
        )

    def encode_user(
        self, encodable: UserEncodable, request: RequestContext | None
    ) -> UserEncoded:
        return UserEncoded(Tensor(), encodable, request)
