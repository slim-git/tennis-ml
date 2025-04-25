
from enum import Enum
from typing import List, Literal

class Feature(Enum):
    _name: str
    _type: Literal['category', 'number']

    SERIES = ('Series', 'category')
    SURFACE = ('Surface', 'category')
    COURT = ('Court', 'category')
    # ROUND = ('Round', 'category')
    DIFF_RANKING = ('diffRanking', 'number')
    DIFF_POINTS = ('diffPoints', 'number')
    DIFF_HEIGHT = ('diffHeight', 'number')
    DIFF_WEIGHT = ('diffWeight', 'number')
    DIFF_AGE = ('diffAge', 'number')
    DIFF_PRO_AGE = ('diffProAge', 'number')
    P1_PLAY_HAND = ('p1PlayHand', 'category')
    P1_BACK_HAND = ('p1BackHand', 'category')
    P2_PLAY_HAND = ('p2PlayHand', 'category')
    P2_BACK_HAND = ('p2BackHand', 'category')
    

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