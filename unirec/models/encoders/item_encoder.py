from torch import Tensor
from typing import Any, ClassVar
from ...core.interfaces import Encodable, Encoder
from ...core.version import Version
from ...data.context import ItemContext, RequestContext
from ...data.encoded import ItemEncoded


class ItemEncoder(Encoder[ItemContext]):
    VERSION: ClassVar[Version] = Version("0.0.0")

    def encode(
        self,
        encodable: Encodable[ItemContext],
        *,
        request: RequestContext | None = None,
        **kwargs: Any,
    ) -> ItemEncoded:
        return ItemEncoded(Tensor())
