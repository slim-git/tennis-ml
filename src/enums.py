
from enum import Enum
from typing import List, Literal

class Feature(Enum):
    _name: str
    _type: Literal['category', 'number']

    SERIES = ('Series', 'category')
    SURFACE = ('Surface', 'category')
    COURT = ('Court', 'category')
    DIFF_PLAY_HAND = ('diffPlayHand', 'number')
    DIFF_BACK_HAND = ('diffBackHand', 'number')
    DIFF_RANKING = ('diffRanking', 'number')
    MEAN_RANKING = ('meanRanking', 'number')
    DIFF_HEIGHT = ('diffHeight', 'number')
    MEAN_HEIGHT = ('meanHeight', 'number')
    DIFF_WEIGHT = ('diffWeight', 'number')
    MEAN_WEIGHT = ('meanWeight', 'number')
    DIFF_PRO_YEAR = ('diffProYear', 'number')
    DIFF_AGE = ('diffAge', 'number')
    

    def __new__(cls, name: str, type: Literal['category', 'number']):
        obj = object.__new__(cls)
        obj._value_ = name
        obj._name = name
        obj._type = type

        return obj

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type
    
    @classmethod
    def get_features_by_type(cls, type: Literal['category', 'number']) -> List['Feature']:
        return [feature for feature in cls if feature.type == type]
    
    @classmethod
    def get_all_features(cls) -> List['Feature']:
        return [feature for feature in cls]

class PlayHand(Enum):
    LEFT = 'Left'
    RIGHT = 'Right'

    @classmethod
    def get_play_hands(cls) -> List[str]:
        return [hand.value for hand in cls]
    
    def __sub__(self, other: 'PlayHand') -> int:
        if self == other:
            return 0
        elif self == PlayHand.RIGHT:
            return 1
        else:
            return -1