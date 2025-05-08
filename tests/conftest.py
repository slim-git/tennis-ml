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
    series = simple_match['tournament_series'].values[0]
    surface = simple_match['tournament_surface'].values[0]
    court = simple_match['tournament_court'].values[0]
    winner_rank = simple_match['winner_rank'].values[0]
    winner_points = simple_match['winner_points'].values[0]
    loser_rank = simple_match['loser_rank'].values[0]
    loser_points = simple_match['loser_points'].values[0]
    winner_height = simple_match['w_height_cm'].values[0]
    winner_weight = simple_match['w_weight_kg'].values[0]
    loser_height = simple_match['l_height_cm'].values[0]
    loser_weight = simple_match['l_weight_kg'].values[0]
    winner_year_of_birth = simple_match['w_year_of_birth'].values[0]
    winner_pro_year = simple_match['w_pro_year'].values[0]
    loser_year_of_birth = simple_match['l_year_of_birth'].values[0]
    loser_pro_year = simple_match['l_pro_year'].values[0]
    winner_play_hand = simple_match['w_play_hand'].values[0]
    winner_back_hand = simple_match['w_back_hand'].values[0]
    loser_play_hand = simple_match['l_play_hand'].values[0]
    loser_back_hand = simple_match['l_back_hand'].values[0]
    diff_points = winner_points - loser_points
    diff_ranking = winner_rank - loser_rank
    diff_height = winner_height - loser_height
    diff_weight = winner_weight - loser_weight
    diff_pro_age = winner_pro_year - loser_pro_year
    diff_age = winner_year_of_birth - loser_year_of_birth

    return pd.DataFrame({
        'Series': [series, series],
        'Surface': [surface, surface],
        'Court': [court, court],
        'diffPoints': [diff_points, -diff_points],
        'diffRanking': [diff_ranking, -diff_ranking],
        'diffHeight': [diff_height, -diff_height],
        'diffWeight': [diff_weight, -diff_weight],
        'diffProAge': [diff_pro_age, -diff_pro_age],
        'diffAge': [diff_age, -diff_age],
        'p1PlayHand': [winner_play_hand, loser_play_hand],
        'p1BackHand': [winner_back_hand, loser_back_hand],
        'p2PlayHand': [loser_play_hand, winner_play_hand],
        'p2BackHand': [loser_back_hand, winner_back_hand],
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
