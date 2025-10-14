import numpy as np
from torch import Tensor
from ..core.interfaces import Encoded


class ItemEncoded(Encoded):
    def __init__(self, feature: Tensor):
        self.__feature: Tensor = feature

    @property
    def feature(self) -> Tensor:
        return self.__feature
