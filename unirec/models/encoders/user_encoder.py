from typing import Any
from ..core.interface import Encoder
from ..data.encodable import UserEncodable
from ..data.encoded import UserEncoded


class UserEncoder(Encoder[UserEncodable, UserEncoded]):
    def encode(self, encodable: UserEncodable, **kwargs: Any) -> UserEncoded: ...
