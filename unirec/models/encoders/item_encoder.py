from typing import Any
from ..core.interface import Encoder
from ..data.encodable import ItemEncodable
from ..data.encoded import ItemEncoded


class ItemEncoder(Encoder[ItemEncodable, ItemEncoded]):
    def encode(self, encodable: ItemEncodable, **kwargs: Any) -> ItemEncoded: ...
