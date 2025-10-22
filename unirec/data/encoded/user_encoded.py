from torch import Tensor
from typing import ClassVar
from ...core.interfaces import Encoded
from ...core.version import Version
from ...data.context import UserContext


class UserEncoded(Encoded[UserContext]):
    VERSION: ClassVar[Version] = Version("0.0.0")

    def __init__(self, vector: Tensor):
        self.__vector: Tensor = vector

    @property
    def vector(self) -> Tensor:
        return self.__vector
