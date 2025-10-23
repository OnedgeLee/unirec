from torch import Tensor
from typing import Any, ClassVar, cast, override
from ...core.interfaces import Encodable, Encoded, Encoder, Request
from ...core.version import Version
from ...data.context import ItemContext, RequestContext
from ...data.encodable import ItemEncodable
from ...data.encoded import ItemEncoded


class ItemEncoder(Encoder[ItemContext]):
    VERSION: ClassVar[Version] = Version("0.0.0")

    @override
    def encode(
        self,
        encodable: Encodable[ItemContext],
        *,
        request: Request | None = None,
        **kwargs: Any,
    ) -> Encoded[ItemContext]:
        return self.encode_item(
            cast(ItemEncodable, encodable), cast(RequestContext, request)
        )

    def encode_item(
        self, encodable: ItemEncodable, request: RequestContext
    ) -> ItemEncoded:
        return ItemEncoded(Tensor(), encodable, request)
