from typing import Literal, Optional, List
from pydantic import BaseModel, Field
from src.enums import PlayHand


class ModelInput(BaseModel):
    p1_rank: int = Field(gt=0, default=1, description="The rank of the 1st player")
    p2_rank: int = Field(gt=0, default=100, description="The rank of the 2nd player")
    court: Literal['Outdoor', 'Indoor'] = Field(default='Outdoor', description="The type of court")
    surface: Literal['Grass', 'Carpet', 'Clay', 'Hard'] = Field(default='Clay', description="The type of surface")
    series: Literal['Grand Slam', 'Masters 1000', 'Masters', 'Masters Cup', 'ATP500', 'ATP250', 'International Gold', 'International'] = Field(default='Grand Slam', description="The series of the tournament")
    p1_height: Optional[int] = Field(gt=0, default=180, description="The height of the 1st player in centimeters")
    p2_height: Optional[int] = Field(gt=0, default=180, description="The height of the 2nd player in centimeters")
    p1_weight: Optional[int] = Field(gt=0, default=80, description="The weight of the 1st player in kilograms")
    p2_weight: Optional[int] = Field(gt=0, default=80, description="The weight of the 2nd player in kilograms")
    p1_year_of_birth: Optional[int] = Field(gt=1950, default=1980, description="The year of birth of the 1st player")
    p2_year_of_birth: Optional[int] = Field(gt=1950, default=1980, description="The year of birth of the 2nd player")
    p1_play_hand: PlayHand = Field(default=PlayHand.RIGHT, description="The play hand of the 1st player")
    p2_play_hand: PlayHand = Field(default=PlayHand.RIGHT, description="The play hand of the 2nd player")
    p1_back_hand: int = Field(default=1, ge=1, le=2, description="The back hand of the 1st player. 1 for one-handed, 2 for two-handed")
    p2_back_hand: int = Field(default=1, ge=1, le=2, description="The back hand of the 2nd player. 1 for one-handed, 2 for two-handed")
    p1_pro_year: Optional[int] = Field(gt=1970, default=2000, description="The year the 1st player turned pro")
    p2_pro_year: Optional[int] = Field(gt=1970, default=2000, description="The year the 2nd player turned pro")
    model: Optional[str] = Field(default='LogisticRegression', description="The name of the model to use for prediction")
    alias: Optional[str] = Field(default='latest', description="The alias of the model to use for prediction")

class ModelOutput(BaseModel):
    result: int = Field(description="The prediction result. 1 if player 1 is expected to win, 0 otherwise.", json_schema_extra={"example": "1"})
    prob: List[float] = Field(description="Probability of [defeat, victory] of player 1.", json_schema_extra={"example": "[0.15, 0.85]"})
