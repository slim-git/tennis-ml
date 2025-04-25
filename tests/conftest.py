import pytest
import pandas as pd

# ----------------------------------------------------------------
# Fixtures for Match Data
# ----------------------------------------------------------------
@pytest.fixture
def simple_match():
    return pd.DataFrame({
        'tournament_series': ['Masters',],
        'tournament_surface': ['Hard',],
        'tournament_court': ['Outdoor',],
        'winner_rank': [1],
        'winner_points': [6605],
        'w_play_hand': ['R',],
        'w_back_hand': [1],
        'w_height_cm': [185],
        'w_weight_kg': [85],
        'w_year_of_birth': [1981],
        'w_pro_year': [1998],
        'loser_rank': [64],
        'loser_points': [665],
        'l_play_hand': ['R',],
        'l_back_hand': [2],
        'l_height_cm': [183],
        'l_weight_kg': [79],
        'l_year_of_birth': [1982],
        'l_pro_year': [2001],
    })

@pytest.fixture
def simple_match_pairwise_data(simple_match: pd.DataFrame):
    return pd.DataFrame({
        'Series': ['Masters', 'Masters'],
        'Surface': ['Hard', 'Hard'],
        'Court': ['Outdoor', 'Outdoor'],
        'diffPoints': [-5940, 5940],
        'diffRanking': [-63, 63],
        'diffHeight': [2, -2],
        'diffWeight': [6, -6],
        'diffProAge': [3, -3],
        'diffAge': [1, -1],
        'p1PlayHand': ['R', 'R'],
        'p1BackHand': [1, 2],
        'p2PlayHand': ['R', 'R'],
        'p2BackHand': [2, 1],
        'target': [1, 0]
    })

@pytest.fixture
def simple_match_empty():
    return pd.DataFrame({
        'Series': [],
        'Surface': [],
        'Court': [],
        'diffPoints': [],
        'diffRanking': [],
        'diffHeight': [],
        'diffWeight': [],
        'diffProAge': [],
        'diffAge': [],
        'p1PlayHand': [],
        'p1BackHand': [],
        'p2PlayHand': [],
        'p2BackHand': [],
        'target': []
    })
